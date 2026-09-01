"""
Command-line interface for the GenAI data augmentation application.

Example usage:
    genai-data-augmentation \
        --dataset-path ./data/dataset_for_augmentation.jsonl \
        --prompt-file ./prompts/augmentation_prompt.txt \
        --labels "label1,label2" \
        --n-samples 5 \
        --num-workers 2
"""

import argparse
from typing import List
from pathlib import Path

from llm_router_utils.core.apps.genai_data_augmentation import (
    GenAIDataAugmentationApp,
)


def prepare_parser(description: str = "") -> argparse.ArgumentParser:
    """Build the ``argparse`` parser for the command-line interface."""
    parser = argparse.ArgumentParser(
        description=description or "Augment datasets using LLMRouter."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the dataset file (XLSX or JSONL).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="Path to the prompt file.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Comma-separated list of labels to augment.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of random samples per class as examples to augment (use 0 for all).",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=3,
        help="Number of augmented examples the LLM should generate for each input text.",
    )
    parser.add_argument(
        "--samples-as-examples",
        type=int,
        default=5,
        help="Number of random samples per class from the dataset to include in the prompt context.",
    )
    parser.add_argument(
        "--llm-router-url",
        default="http://192.168.100.65:8080",
        help="Base URL of the LLMRouter service.",
    )
    parser.add_argument(
        "--llm-router-token",
        required=False,
        default=None,
        help="Authentication token for the LLMRouter service.",
    )
    parser.add_argument(
        "--llm-router-timeout",
        type=int,
        default=10,
        help="Per-request timeout in seconds for LLMRouter calls (default: 10).",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-oss:120b",
        help="Model identifier passed to the router.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--batch-save-size",
        type=int,
        default=5,
        help="How many records are written to disk at once.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process data but do not write output files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override directory where result files are stored.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG level logging.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel worker threads.",
    )
    parser.add_argument(
        "--text-column-name",
        default="Tekst",
        help="Name of the column containing the text (default: 'Tekst').",
    )
    parser.add_argument(
        "--label-column-name",
        default="label",
        help="Name of the column containing the label (default: 'label').",
    )
    parser.add_argument(
        "--export-xlsx",
        action="store_true",
        help="Convert output JSONL files to XLSX format (default: True).",
    )
    parser.add_argument(
        "--no-export-xlsx",
        dest="export_xlsx",
        action="store_false",
        help="Disable XLSX export.",
    )
    parser.set_defaults(export_xlsx=True)
    return parser


def main(argv: List[str] | None = None) -> None:
    """Parse arguments, build the app and run it."""
    args = prepare_parser().parse_args(argv)

    labels = args.labels.split(",")

    app = GenAIDataAugmentationApp(
        dataset_path=args.dataset_path,
        prompt_path=args.prompt_file,
        labels=labels,
        llm_router_url=args.llm_router_url,
        llm_router_token=args.llm_router_token,
        llm_router_timeout=args.llm_router_timeout,
        model_name=args.model_name,
        temperature=args.temperature,
        n_samples=args.n_samples,
        n_examples=args.n_examples,
        samples_as_examples=args.samples_as_examples,
        batch_save_size=args.batch_save_size,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        verbose=args.verbose,
        num_workers=args.num_workers,
        export_xlsx=args.export_xlsx,
        text_column_name=args.text_column_name,
        label_column_name=args.label_column_name,
    )
    app.run()


if __name__ == "__main__":
    main()
