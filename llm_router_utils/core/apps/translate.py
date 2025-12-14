import argparse

from typing import List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_router_lib.client import LLMRouterClient
from rdl_ml_utils.utils.dataset_processor import DatasetProcessor


class TextTranslationService:
    """Wrap LLMRouterClient for translation calls."""

    def __init__(self, router_host: str, model: str):
        self.client = LLMRouterClient(api=router_host)
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

        # new configuration values
        self.num_workers = getattr(args, "num_workers", 1)
        self.batch_size = getattr(args, "batch_size", 8)

        self.translations = []

    def _batch_texts(self, texts) -> List:
        """Split the full list of texts into ``batch_size`` chunks."""
        return [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

    def run(self) -> None:
        """Execute the full translation workflow and print the result."""
        self.translations.clear()

        texts = self.processor.load_records()
        if not texts:
            return

        batches = self._batch_texts(texts)
        if self.num_workers <= 1:
            for batch in batches:
                response = self.service.translate(batch)
                self.translations.append(response)

        else:
            # Preserve order: each future carries its batch index
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_idx = {
                    executor.submit(self.service.translate, batch): idx
                    for idx, batch in enumerate(batches)
                }

                # Prepare a placeholder list so we can restore original order later
                ordered_results: List[Any] = [None] * len(batches)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    ordered_results[idx] = future.result()

                self.translations.extend(ordered_results)
