"""
GenAI classifier core application.

This module contains the GenAIClassifierApp class that orchestrates the classification
pipeline for processing translated datasets using an LLM Router service.
"""

import json
import logging
import queue
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional

import pandas as pd
import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

from llm_router_lib.client import LLMRouterClient
from llm_router_utils.core.hf_dataset_handler import HfDatasetHandler
from llm_router_utils.core.jsonl_to_xlsx import convert_jsonl_to_xlsx
from rdl_ml_utils.handlers.prompt_handler import PromptHandler

# --------------------------------------------------------------------------- #
# Logging configuration (JSON‑compatible format)
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}',
)
log = logging.getLogger(__name__)

_ADDITIONAL_PROMPT_JSON = """
**Format wyjścia (JSON)**
```
json
{
  "exists": true|false,                 // Czy klasa występuje w tekście
  "confidence": 0.0‑1.0,                // Szacowana pewność decyzji
  "reason": "Krótka przyczyna decyzji"  // Dlaczego tak/nie
}
```

**Zachowanie modelu**
‑ Generuj **jedno** poprawne JSON‑owe wyjście na każde zapytanie.
‑ Unikaj dodatkowego tekstu przed lub po JSON‑ie.
"""

_ADDITIONAL_PROMPT_JSON = None


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class AggregatedRecord:
    """One line that will be stored in the JSON‑Lines output file."""

    text: str
    field: str
    features: List[Dict[str, Any]]

    def to_json(self) -> str:
        """Serialize to a JSON string (ASCII‑safe)."""
        return json.dumps(
            {"text": self.text, "field": self.field, "features": self.features},
            ensure_ascii=False,
        )


# --------------------------------------------------------------------------- #
# Main application class
# --------------------------------------------------------------------------- #
class GenAIClassifierApp:
    """
    High-level orchestrator for GenAI classification pipeline.

    This class handles the classification of translated datasets using an LLM Router service.
    It can be used both from CLI and as a library component.
    """

    def __init__(
        self,
        dataset_dir: Path,
        prompts_dir: Path,
        llm_router_url: str,
        model_name: str,
        temperature: float = 0.0,
        prompts_list: Optional[List[str]] = None,
        batch_save_size: int = 5,
        dry_run: bool = False,
        output_dir: Optional[Path] = None,
        verbose: bool = False,
        num_workers: int = 2,
        n_sample: Optional[int] = 50,
        export_xlsx: bool = True,
        text_column_name: str = "Tekst",
    ):
        self.dataset_dir = dataset_dir
        self.prompts_dir = prompts_dir
        self.llm_router_url = llm_router_url
        self.model_name = model_name
        self.temperature = temperature
        self.prompts_list = prompts_list or []
        self.batch_save_size = batch_save_size
        self.dry_run = dry_run
        self.output_dir = output_dir
        self.verbose = verbose
        self.num_workers = num_workers
        self.n_sample = n_sample
        self.export_xlsx = export_xlsx
        self.text_column_name = text_column_name

        # Shared structures for thread-safe buffering
        self._buffers: dict[Path, list[AggregatedRecord]] = {}
        self._file_locks: dict[Path, threading.Lock] = {}
        self._buffers_lock = threading.Lock()

        if self.verbose:
            log.setLevel(logging.DEBUG)

    # --------------------------------------------------------------------------- #
    # Helper functions
    # --------------------------------------------------------------------------- #
    def _load_datasets(
        self, dataset_list: List[Tuple[str, List[str]]]
    ) -> List[Dict[str, Any]]:
        """
        Load all datasets defined in ``dataset_list`` and optionally from XLSX files.
        """
        loaded: List[Dict[str, Any]] = []

        # Load datasets from dataset_list (HuggingFace datasets)
        for ds_name, fields in dataset_list:
            log.info("Loading dataset %s", ds_name)
            try:
                ds = HfDatasetHandler.load_saved_dataset(
                    ds_name, self.dataset_dir, config="train"
                )
                loaded.append({"name": ds_name, "fields": fields, "dataset": ds})
            except Exception as exc:  # pragma: no cover – runtime safeguard
                log.exception("Failed to load dataset %s: %s", ds_name, exc)
        return loaded

    def _load_local_datasets(
        self, df_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        loaded: List[Dict[str, Any]] = []
        # Load datasets from XLSX or JSONL files if they exist in dataset_dir
        xlsx_files = list(self.dataset_dir.glob("*.xlsx"))
        jsonl_files = list(self.dataset_dir.glob("*.jsonl"))

        all_files = xlsx_files + jsonl_files

        if all_files:
            log.info("Found %d local file(s) in dataset directory", len(all_files))

            for data_file in all_files:
                try:
                    if data_file.suffix == ".xlsx":
                        # Read the first sheet from XLSX
                        df = pd.read_excel(data_file, sheet_name=0)
                        # Convert DataFrame to a simple list of dicts for consistency
                        dataset_records = df.to_dict("records")
                        columns = list(df.columns)
                    elif data_file.suffix == ".jsonl":
                        dataset_records = []
                        with data_file.open("r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    dataset_records.append(json.loads(line))
                        # Use keys of the first record as columns if available
                        columns = (
                            list(dataset_records[0].keys())
                            if dataset_records
                            else []
                        )
                    else:
                        continue

                    # Use all columns as fields (similar to HF dataset)
                    fields = df_fields or columns
                    ds_name = (
                        data_file.stem
                    )  # Use filename without extension as dataset name

                    log.info(
                        "Loading dataset %s (from %s) with fields: %s",
                        ds_name,
                        data_file.suffix,
                        fields,
                    )

                    loaded.append(
                        {
                            "name": ds_name,
                            "fields": fields,
                            "dataset": dataset_records,
                        }
                    )

                except Exception as exc:
                    log.exception("Failed to load local file %s: %s", data_file, exc)
        return loaded

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _classify_text(
        self,
        llm_client: LLMRouterClient,
        prompt_handler: PromptHandler,
        text: str,
        feature_name: str,
        retry_when_invalid_json: int = 5,
    ) -> Dict[str, Any]:
        """Call the LLM for a single (text, feature) pair and return parsed JSON."""
        prompt_str = prompt_handler.get_prompt(feature_name)

        if _ADDITIONAL_PROMPT_JSON and len(_ADDITIONAL_PROMPT_JSON.strip()):
            prompt_str += f"\n{_ADDITIONAL_PROMPT_JSON}"

        payload = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "system_prompt": prompt_str,
            "user_last_statement": text,
        }

        parsed = None
        raw_json = None
        while retry_when_invalid_json > 0:
            response = llm_client.extended_conversation_with_model(payload=payload)
            raw_json = response.get("response", "{}")
            try:
                raw_json = raw_json.replace("json\n", "")
                parsed = json.loads(raw_json)
                break
            except json.JSONDecodeError:
                log.warning(
                    "Invalid JSON from LLM for text %r... (feature %s) – retrying %d",
                    text[:20],
                    feature_name,
                    retry_when_invalid_json,
                )
                retry_when_invalid_json -= 1

                log.warning("=" * 100)
                log.warning(response)
                log.warning("=" * 100)

        if parsed and len(parsed) and self.verbose:
            parsed["_raw_response"] = raw_json

        if not parsed:
            parsed = {}

        return parsed

    def _load_existing_texts(self, path: Path) -> Set[Tuple[str, str]]:
        """Return a set of (field, text) tuples already present in *path*."""
        seen: Set[Tuple[str, str]] = set()
        if not path.is_file():
            return seen

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    txt = obj.get("text")
                    fld = obj.get("field")
                    if isinstance(txt, str) and isinstance(fld, str):
                        seen.add((fld, txt))
                except json.JSONDecodeError:
                    log.debug("Skipping malformed line in %s", path)
        log.info(
            "Loaded %d previously processed records from %s", len(seen), path.name
        )
        return seen

    def _flush_buffer(self, path: Path) -> None:
        """Write the accumulated records for *path* to disk (thread-safe)."""
        # Grab a reference to the buffer and its file-lock – then release the dict lock.
        with self._buffers_lock:
            buffer = self._buffers.get(path, [])
            lock = self._file_locks.setdefault(path, threading.Lock())

        if not buffer or self.dry_run:
            buffer.clear()
            return

        with lock, path.open("a", encoding="utf-8") as f:
            for rec in buffer:
                f.write(rec.to_json() + "\n")

        log.debug("Flushed %d records to %s", len(buffer), path.name)
        buffer.clear()

    # --------------------------------------------------------------------------- #
    # Worker implementation
    # --------------------------------------------------------------------------- #
    def _worker(self, task_queue: queue.Queue, prompts_dir: Path) -> None:
        """Thread target – consumes tasks and classifies texts."""
        prompt_handler = PromptHandler(str(prompts_dir))
        llm_client = LLMRouterClient(self.llm_router_url)

        self.prompts_list = list(prompt_handler.list_prompts().keys())

        while True:
            try:
                output_path, field, text = task_queue.get(
                    timeout=1
                )  # (Path, str, str)
            except queue.Empty:
                break  # no more work

            # ---- classification -------------------------------------------------
            feature_responses: List[Dict[str, Any]] = []
            for feature_name in self.prompts_list:
                llm_response = self._classify_text(
                    llm_client,
                    prompt_handler,
                    text,
                    feature_name,
                    retry_when_invalid_json=5,
                )
                feature_responses.append(
                    {"name": feature_name, "response": llm_response}
                )

            aggregated = AggregatedRecord(
                text=text, field=field, features=feature_responses
            )

            # ---- store result in shared buffer -----------------------------------
            need_flush = False
            with self._buffers_lock:
                buf = self._buffers.setdefault(output_path, [])
                buf.append(aggregated)
                if len(buf) >= self.batch_save_size:
                    need_flush = True

            if need_flush:
                self._flush_buffer(output_path)

            task_queue.task_done()

    # --------------------------------------------------------------------------- #
    # Dataset preparation – enqueues tasks
    # --------------------------------------------------------------------------- #
    def _process_dataset(
        self, ds_item: Dict[str, Any], task_queue: queue.Queue
    ) -> None:
        """
        Scan a dataset and enqueue all (field, text) pairs that have not been
        processed yet.
        """
        ds_name = ds_item["name"]
        fields = ds_item["fields"]
        dataset = ds_item["dataset"]

        log.info("Preparing tasks for dataset %s (fields: %s)", ds_name, fields)

        out_dir = self.output_dir or self.dataset_dir
        output_path = out_dir / f"{ds_name.replace('/', '__')}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        already_done = self._load_existing_texts(output_path)

        # initialise shared buffers & file lock for this output file
        with self._buffers_lock:
            self._buffers.setdefault(output_path, [])
            self._file_locks.setdefault(output_path, threading.Lock())

        rng = random.Random()

        # Handle both HuggingFace datasets and list of dicts (from XLSX)
        if hasattr(dataset, "column_names"):
            # HuggingFace dataset
            for field in tqdm.tqdm(fields, desc=f"{ds_name} fields", leave=False):
                if field not in dataset.column_names:
                    log.warning(
                        "Field %s not present in dataset %s – skipping",
                        field,
                        ds_name,
                    )
                    continue

                values = list(dataset[field])
                rng.shuffle(values)

                # -------------------------------------------------------------
                # Apply n_sample logic – take at most self.n_sample items (if >0)
                # -------------------------------------------------------------
                if self.n_sample is None or self.n_sample <= 0:
                    sampled_values = values
                else:
                    sampled_values = values[: self.n_sample]

                for value in sampled_values:
                    key = (field, value)
                    if key in already_done:
                        continue
                    already_done.add(key)
                    task_queue.put((output_path, field, value))
        else:
            # List of dicts (from XLSX)
            # Extract all values for each field
            for field in tqdm.tqdm(fields, desc=f"{ds_name} fields", leave=False):
                if field not in fields:
                    log.warning(
                        "Field %s not present in dataset %s – skipping",
                        field,
                        ds_name,
                    )
                    continue

                # Extract values from all records
                values = []
                for record in dataset:
                    if field in record:
                        values.append(record[field])

                rng.shuffle(values)

                # -------------------------------------------------------------
                # Apply n_sample logic – take at most self.n_sample items (if >0)
                # -------------------------------------------------------------
                if self.n_sample is None or self.n_sample <= 0:
                    sampled_values = values
                else:
                    sampled_values = values[: self.n_sample]

                for value in sampled_values:
                    key = (
                        field,
                        str(value),
                    )  # Ensure value is string for consistency
                    if key in already_done:
                        continue
                    already_done.add(key)
                    task_queue.put((output_path, field, str(value)))

    def _log_startup_info(self) -> None:
        """Log version info and optional verbose configuration details."""
        client = LLMRouterClient(self.llm_router_url)
        version_info = client.version()
        log.info(
            "Using LLMRouter version %s",
            version_info.get("version", "unknown"),
        )
        if self.verbose:
            log.debug("Full configuration: %s", self.__dict__)

    def _convert_output_files_to_xlsx(self) -> None:
        """Convert all generated JSONL files to XLSX format."""
        if not self.export_xlsx or self.dry_run:
            return

        out_dir = self.output_dir or self.dataset_dir
        if not out_dir.is_dir():
            log.warning("Output directory does not exist: %s", out_dir)
            return

        jsonl_files = list(out_dir.glob("*.jsonl"))
        if not jsonl_files:
            log.info("No JSONL files found to convert to XLSX")
            return

        log.info("Converting %d JSONL file(s) to XLSX format...", len(jsonl_files))

        for jsonl_file in jsonl_files:
            try:
                xlsx_file = jsonl_file.with_suffix(".xlsx")
                log.info("Converting %s to %s", jsonl_file.name, xlsx_file.name)
                convert_jsonl_to_xlsx(jsonl_file, xlsx_file)
            except Exception as exc:
                log.error("Failed to convert %s to XLSX: %s", jsonl_file.name, exc)

    # --------------------------------------------------------------------------- #
    # Main workflow
    # --------------------------------------------------------------------------- #
    def run(self) -> None:
        """Execute the classification pipeline."""
        # Validate paths early
        if not self.dataset_dir.is_dir():
            raise ValueError(f"Dataset directory does not exist: {self.dataset_dir}")
        if not self.prompts_dir.is_dir():
            raise ValueError(f"Prompts directory does not exist: {self.prompts_dir}")
        if not self.output_dir:
            raise ValueError(f"Output directory is not given: {self.output_dir}")
        if self.output_dir and not self.output_dir.is_dir():
            raise ValueError(f"Output directory does not exist: {self.output_dir}")

        # Helpers used only by the main thread
        PromptHandler(str(self.prompts_dir))  # just to validate path early

        self._log_startup_info()

        # Load all datasets
        # TODO: if dataset_list is not empty - load HF datasets
        # all_datasets = self._load_datasets(dataset_list=)
        all_datasets = self._load_local_datasets(df_fields=[self.text_column_name])

        # ---- task queue -------------------------------------------------
        task_q: queue.Queue = queue.Queue()

        # Enqueue work for every dataset
        for ds_item in all_datasets:
            self._process_dataset(ds_item, task_q)

        # ---- start worker threads ----------------------------------------
        workers: List[threading.Thread] = []
        for _ in range(self.num_workers):
            t = threading.Thread(
                target=self._worker,
                args=(task_q, self.prompts_dir),
                daemon=True,
            )
            t.start()
            workers.append(t)

        # Wait until every queued task is marked as done
        task_q.join()

        # Ensure all workers have terminated
        for w in workers:
            w.join()

        # ---- final flush of any remaining records -----------------------
        # Grab the list of paths while holding the lock, then flush *outside* the lock
        with self._buffers_lock:
            paths_to_flush = list(self._buffers.keys())

        for path in paths_to_flush:
            self._flush_buffer(path)

        # ---- convert JSONL files to XLSX format ---------------------------
        self._convert_output_files_to_xlsx()

        log.info("Processing finished.")
