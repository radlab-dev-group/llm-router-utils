import json
import argparse

from tqdm import tqdm
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_router_lib.client import LLMRouterClient
from rdl_ml_utils.utils.dataset_processor import DatasetProcessor


class TextTranslationService:
    """Wrap LLMRouterClient for translation calls."""

    def __init__(self, router_host: str, model: str):
        self.client = LLMRouterClient(api=router_host, timeout=30, retries=2)
        self.model = model

    def translate(self, texts: List[str]) -> Any:
        """Send ``texts`` to the router and return the raw response."""
        return self.client.translate(model=self.model, texts=texts)


class TranslateApp:
    """High‑level orchestrator usable from CLI or as a library."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.accept_fields = args.accept_field

        self.processor = DatasetProcessor(
            dataset_paths=args.dataset_path,
            dataset_type=args.dataset_type,
            accept_fields=self.accept_fields,
        )
        self.service = TextTranslationService(args.llm_router_host, args.model)

        # optional configuration values
        self.num_workers = getattr(args, "num_workers", 1)
        self.batch_size = getattr(args, "batch_size", 8)

        self.translations: List[Any] = []

    # ----------------------------------------------------------------------
    # Helper methods – flatten records, translate, then rebuild JSON output
    # ----------------------------------------------------------------------
    def _flatten_records(
        self, records: List[dict]
    ) -> tuple[List[str], List[tuple[int, str]]]:
        """
        Convert a list of record dictionaries into:
        * ``flat_texts`` – a simple list of strings that need translation.
        * ``positions``  – a list of ``(record_index, field_name)`` tuples that map
          each translated string back to its original location.
        Only fields listed in ``self.accept_fields`` are considered.
        """
        flat_texts: List[str] = []
        positions: List[tuple[int, str]] = []
        for idx, rec in enumerate(records):
            for field in self.accept_fields:
                if field in rec:
                    flat_texts.append(str(rec[field]))
                    positions.append((idx, field))
        return flat_texts, positions

    def _reconstruct_records(
        self,
        records: List[dict],
        positions: List[tuple[int, str]],
        translations: List[str],
    ) -> List[str]:
        """
        Insert the translated strings back into their original dictionaries and
        return a list of JSON‑encoded records that contain **only** the accepted fields.
        """
        for (rec_idx, field_name), translated in zip(positions, translations):
            records[rec_idx][field_name] = translated

        json_records = [
            json.dumps({k: v for k, v in rec.items() if k in self.accept_fields})
            for rec in records
        ]
        return json_records

    def _batch_texts(self, texts: List[str]) -> List[List[str]]:
        """Split the full list of texts into ``batch_size`` chunks."""
        return [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

    # ----------------------------------------------------------------------
    # Main workflow
    # ----------------------------------------------------------------------
    def run(self) -> None:
        """Execute the translation pipeline and store JSON results in ``self.translations``."""
        self.translations.clear()

        # 1️⃣ Load raw records (list of dicts)
        records = self.processor.load_records()
        if not records:
            return

        # 2️⃣ Flatten to a simple list of texts + remember where each belongs
        flat_texts, positions = self._flatten_records(records)
        if not flat_texts:
            return

        # 3️⃣ Batch the texts
        batches = self._batch_texts(flat_texts)

        # ------------------------------------------------------------------
        # Translation step – single‑threaded vs. multi‑threaded with tqdm progress
        # ------------------------------------------------------------------
        if self.num_workers <= 1:
            # ----- single‑threaded -------------------------------------------------
            flat_results: List[str] = []
            for batch in tqdm(
                batches, desc="Translating (single thread)", unit="batch"
            ):
                response = self.service.translate(batch)
                if isinstance(response, list):
                    flat_results.extend(response)
                else:
                    flat_results.append(response)
        else:
            # ----- multi‑threaded ---------------------------------------------------
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_idx = {
                    executor.submit(self.service.translate, batch): idx
                    for idx, batch in enumerate(batches)
                }

                pbar = tqdm(
                    total=len(batches),
                    desc="Translating (multi thread)",
                    unit="batch",
                )
                ordered_results: List[Any] = [None] * len(batches)

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    ordered_results[idx] = future.result()
                    pbar.update(1)

                pbar.close()

            # Flatten ordered results (they may be a list or a single string)
            flat_results: List[str] = []
            for res in ordered_results:
                if isinstance(res, list):
                    flat_results.extend(res)
                else:
                    flat_results.append(res)

        # 4️⃣ Re‑assemble original records with translations and store JSON strings
        self.translations = self._reconstruct_records(
            records, positions, flat_results
        )
