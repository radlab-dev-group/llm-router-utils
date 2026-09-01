"""
GenAI data augmentation core application.

This module contains the GenAIDataAugmentationApp class that orchestrates the data augmentation
pipeline for processing datasets using an LLM Router service.
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
from tenacity import retry, stop_after_attempt, wait_fixed

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


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class AugmentedRecord:
    """One line that will be stored in the JSON‑Lines output file."""

    original_text: str
    labels: List[str]
    augmented_text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization, parsing augmented_text if possible."""
        data = {
            "original_text": self.original_text,
            "labels": self.labels,
            "metadata": self.metadata,
        }

        # Try to parse augmented_text if it looks like JSON
        try:
            # Simple cleaning for common markdown artifacts
            clean_text = self.augmented_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            parsed_augmented = json.loads(clean_text)
            if isinstance(parsed_augmented, dict):
                data.update(parsed_augmented)
            elif isinstance(parsed_augmented, list):
                # If LLM returns a list of objects (e.g. one for each class),
                # include it as a structured field instead of a raw string.
                data["augmented_results"] = parsed_augmented
            else:
                data["augmented_text"] = self.augmented_text
        except (json.JSONDecodeError, TypeError):
            data["augmented_text"] = self.augmented_text
        return data

    def to_json(self) -> str:
        """Serialize to a JSON string (ASCII‑safe)."""
        data = self.to_dict()
        return json.dumps(data, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Main application class
# --------------------------------------------------------------------------- #
class GenAIDataAugmentationApp:
    """
    High-level orchestrator for GenAI data augmentation pipeline.

    - Loads datasets from XLSX or JSONL.
    - Samples original data for specified labels.
    - Uses LLM to generate augmented versions of the samples.
    - Saves results in JSONL and XLSX formats.
    """

    def __init__(
        self,
        dataset_path: Path,
        prompt_path: Path,
        labels: List[str],
        llm_router_url: str,
        model_name: str,
        llm_router_token: Optional[str] = None,
        llm_router_timeout: int = 10,
        temperature: float = 0.7,
        n_samples: int = 5,
        n_examples: int = 3,
        samples_as_examples: int = 5,
        batch_save_size: int = 5,
        dry_run: bool = False,
        output_dir: Optional[Path] = None,
        verbose: bool = False,
        num_workers: int = 2,
        export_xlsx: bool = True,
        text_column_name: str = "Tekst",
        label_column_name: str = "labels",
    ):
        self.dataset_path = dataset_path
        self.prompt_path = prompt_path
        self.labels = [L.strip() for L in labels]
        self.llm_router_url = llm_router_url
        self.llm_router_token = llm_router_token
        self.llm_router_timeout = llm_router_timeout
        self.model_name = model_name
        self.temperature = temperature
        self.n_samples = n_samples
        self.n_examples = n_examples
        self.samples_as_examples = samples_as_examples
        self.batch_save_size = batch_save_size
        self.dry_run = dry_run
        self.output_dir = output_dir
        self.verbose = verbose
        self.num_workers = num_workers
        self.export_xlsx = export_xlsx
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name

        self._buffers: dict[Path, list[AugmentedRecord]] = {}
        self._file_locks: dict[Path, threading.Lock] = {}
        self._buffers_lock = threading.Lock()

        if self.verbose:
            log.setLevel(logging.DEBUG)

    def _load_dataset(self) -> pd.DataFrame:
        """Load dataset from XLSX or JSONL file."""
        log.info("Loading dataset from %s", self.dataset_path)
        if self.dataset_path.suffix == ".xlsx":
            df = pd.read_excel(self.dataset_path)
        elif self.dataset_path.suffix == ".jsonl":
            df = pd.read_json(self.dataset_path, lines=True)
        else:
            raise ValueError(
                f"Unsupported dataset format: {self.dataset_path.suffix}"
            )

        if self.text_column_name not in df.columns:
            raise ValueError(
                f"Column '{self.text_column_name}' not found in dataset."
            )
        if self.label_column_name not in df.columns:
            log.warning(
                "Column '%s' not found. Data augmentation will proceed without label filtering if possible.",
                self.label_column_name,
            )

        return df

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(30),
    )
    def _augment_text(
        self,
        llm_client: LLMRouterClient,
        prompt: str,
        text: str,
        labels: List[str],
        all_samples_info: str = "",
    ) -> str:
        """Call the LLM to augment a single text."""
        # Replace placeholders in the prompt
        final_prompt = prompt.replace(
            "{CLASS_LIST_PLACEHOLDER}", ", ".join(self.labels)
        )
        final_prompt = final_prompt.replace(
            "{SAMPLES_PER_CLASS_PLACEHOLDER}", str(self.n_examples)
        )
        final_prompt = final_prompt.replace(
            "{CLASS_EXAMPLES_PLACEHOLDER}", all_samples_info
        )

        user_input = text

        response = llm_client.extended_conversation_with_model(
            user_last_statement=user_input,
            system_prompt=final_prompt,
            model=self.model_name,
            temperature=self.temperature,
        )
        return (response.response or "").strip()

    def _flush_buffer(self, path: Path) -> None:
        """Write records from buffer to disk and clear buffer."""
        train_path = path.with_name(f"{path.stem}-train.jsonl")

        with self._file_locks[path]:
            with self._buffers_lock:
                records = self._buffers.get(path, [])
                if not records:
                    return
                self._buffers[path] = []

            log.debug("Flushing %d records to %s", len(records), path)
            with (
                path.open("a", encoding="utf-8") as f,
                train_path.open("a", encoding="utf-8") as f_train,
            ):
                for rec in records:
                    f.write(rec.to_json() + "\n")

                    # Extract augmented examples for the -train file
                    rec_dict = rec.to_dict()
                    examples = rec_dict.get("augmented_examples", [])
                    if isinstance(examples, list):
                        for ex in examples:
                            if isinstance(ex, str):
                                train_rec = {"text": ex, "labels": rec.labels}
                                f_train.write(
                                    json.dumps(train_rec, ensure_ascii=False) + "\n"
                                )
                            elif isinstance(ex, dict) and "text" in ex:
                                # Fallback if LLM returns objects instead of strings
                                # but usually we expect strings in augmented_examples
                                train_rec = {
                                    "text": ex["text"],
                                    "labels": ex.get("labels", rec.labels),
                                }
                                f_train.write(
                                    json.dumps(train_rec, ensure_ascii=False) + "\n"
                                )

    def _worker(
        self, task_queue: queue.Queue, prompt: str, all_samples_info: str
    ) -> None:
        """Worker thread for processing augmentation tasks."""
        llm_client = LLMRouterClient(
            self.llm_router_url,
            token=self.llm_router_token,
            timeout=self.llm_router_timeout,
        )

        try:
            while True:
                try:
                    task = task_queue.get(timeout=1)
                except queue.Empty:
                    break

                output_path, labels, text = task

                try:
                    augmented_text = self._augment_text(
                        llm_client, prompt, text, labels, all_samples_info
                    )

                    record = AugmentedRecord(
                        original_text=text,
                        labels=labels,
                        augmented_text=augmented_text,
                        metadata={
                            "model": self.model_name,
                            "temperature": self.temperature,
                        },
                    )

                    need_flush = False
                    with self._buffers_lock:
                        self._buffers[output_path].append(record)
                        if len(self._buffers[output_path]) >= self.batch_save_size:
                            need_flush = True

                    if need_flush:
                        self._flush_buffer(output_path)

                except Exception as exc:
                    log.exception(
                        "Failed to augment text for labels %s: %s", labels, exc
                    )
                finally:
                    task_queue.task_done()
        finally:
            llm_client.close()

    def _convert_output_files_to_xlsx(self) -> None:
        """Convert all generated JSONL files to XLSX format."""
        if not self.export_xlsx or self.dry_run:
            return

        out_dir = self.output_dir or self.dataset_path.parent
        jsonl_files = list(out_dir.glob("*_augmented.jsonl"))
        jsonl_files += list(out_dir.glob("*_augmented-train.jsonl"))

        for jsonl_path in jsonl_files:
            xlsx_path = jsonl_path.with_suffix(".xlsx")
            log.info("Converting %s to %s", jsonl_path.name, xlsx_path.name)
            try:
                convert_jsonl_to_xlsx(jsonl_path, xlsx_path)
            except Exception as exc:
                log.exception("Failed to convert %s to XLSX: %s", jsonl_path, exc)

    def run(self) -> None:
        """Run the augmentation pipeline."""
        df = self._load_dataset()

        with open(self.prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()

        out_dir = self.output_dir or self.dataset_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{self.dataset_path.stem}_augmented.jsonl"

        with self._buffers_lock:
            self._buffers[output_path] = []
            self._file_locks[output_path] = threading.Lock()

        task_queue = queue.Queue()
        example_samples = []

        for label in self.labels:
            # Filter by labels (check if label is in the list of labels)
            if self.label_column_name in df.columns:
                # Handle both list of labels and single label (string) for backward compatibility
                import ast

                def matches_label(val):
                    # Handle string representation of a list (e.g. from XLSX or JSONL)
                    if (
                        isinstance(val, str)
                        and val.strip().startswith("[")
                        and val.strip().endswith("]")
                    ):
                        try:
                            val = ast.literal_eval(val)
                        except (ValueError, SyntaxError):
                            pass

                    if isinstance(val, list):
                        return label in [str(v).strip() for v in val]
                    return str(val).strip() == label

                subset = df[df[self.label_column_name].apply(matches_label)]
            else:
                subset = df  # If no labels, use whole dataset? Requirement says labels are provided.

            if subset.empty:
                log.warning("No samples found for label: %s", label)
                continue

            # Sample n_samples for processing
            if self.n_samples <= 0:
                n = len(subset)
            else:
                n = min(len(subset), self.n_samples)
            sampled = subset.sample(n=n)

            # Sample samples_as_examples for the context (json/prompt)
            if self.samples_as_examples <= 0:
                n_ex = len(subset)
            else:
                n_ex = min(len(subset), self.samples_as_examples)
            sampled_for_context = subset.sample(n=n_ex)

            for _, row in sampled_for_context.iterrows():
                text_ex = str(row[self.text_column_name])
                row_labels_ex = row.get(self.label_column_name, [label])

                # Handle string representation of a list
                if (
                    isinstance(row_labels_ex, str)
                    and row_labels_ex.strip().startswith("[")
                    and row_labels_ex.strip().endswith("]")
                ):
                    try:
                        row_labels_ex = ast.literal_eval(row_labels_ex)
                    except (ValueError, SyntaxError):
                        pass

                if not isinstance(row_labels_ex, list):
                    row_labels_ex = [str(row_labels_ex).strip()]
                else:
                    row_labels_ex = [str(L).strip() for L in row_labels_ex]

                row_labels_ex = [L for L in row_labels_ex if L in self.labels]
                example_samples.append({"text": text_ex, "labels": row_labels_ex})

            log.info("Enqueuing %d samples for label: %s", n, label)
            for _, row in sampled.iterrows():
                text = str(row[self.text_column_name])
                # We pass the full list of labels for this record to the worker
                row_labels = row.get(self.label_column_name, [label])

                # Handle string representation of a list
                if (
                    isinstance(row_labels, str)
                    and row_labels.strip().startswith("[")
                    and row_labels.strip().endswith("]")
                ):
                    try:
                        row_labels = ast.literal_eval(row_labels)
                    except (ValueError, SyntaxError):
                        pass

                if not isinstance(row_labels, list):
                    row_labels = [str(row_labels).strip()]
                else:
                    row_labels = [str(L).strip() for L in row_labels]

                # Keep only labels that are in the filter list (self.labels)
                row_labels = [L for L in row_labels if L in self.labels]

                task_queue.put((output_path, row_labels, text))

        if task_queue.empty():
            log.warning("No tasks to process.")
            return

        # Prepare all_samples_info for CLASS_EXAMPLES_PLACEHOLDER
        all_samples_info = ""
        for i, sample in enumerate(example_samples, 1):
            all_samples_info += (
                f"Przykład {i}:\nTekst: {sample['text']}\n"
                f"Klasy: {', '.join(sample['labels'])}\n\n"
            )

        # Start workers
        threads = []
        for _ in range(self.num_workers):
            t = threading.Thread(
                target=self._worker,
                args=(task_queue, prompt_content, all_samples_info),
            )
            t.start()
            threads.append(t)

        # Wait for completion
        task_queue.join()
        for t in threads:
            t.join()

        # Final flush
        self._flush_buffer(output_path)

        # Convert to XLSX
        self._convert_output_files_to_xlsx()
        log.info("Augmentation finished. Output saved to %s", output_path)
