"""
Command-line interface for the GenAI classifier application.

This module provides the CLI entry point for classifying translated datasets
using an LLM Router service.
"""

import argparse

from typing import List
from pathlib import Path

from llm_router_utils.core.apps.genai_classifier import GenAIClassifierApp


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def prepare_parser(description: str = "") -> argparse.ArgumentParser:
    """Build the ``argparse`` parser for the command-line interface."""
    parser = argparse.ArgumentParser(
        description=description or "Classify translated datasets using LLMRouter."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing downloaded HF datasets.",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        required=True,
        help="Directory with prompt files.",
    )
    parser.add_argument(
        "--llm-router-url",
        default="http://192.168.100.65:8080",
        help="Base URL of the LLMRouter service.",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-oss:120b",
        help="Model identifier passed to the router.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--batch-save-size",
        type=int,
        default=5,
        help="How many aggregated records are written to disk at once.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process data but do not write output files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override directory where result .jsonl files are stored.",
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
        help="Number of parallel worker threads (and LLM clients).",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=50,
        help=(
            "Number of random samples per field (default: all). "
            "If omitted, zero or negative, all examples are processed."
        ),
    )
    parser.add_argument(
        "--text-column-name",
        default="Tekst",
        help="Name of the column containing the text to classify (default: 'Tekst').",
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
        help="Disable XLSX export (only save JSONL files).",
    )
    parser.set_defaults(export_xlsx=True)
    return parser


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(argv: List[str] | None = None) -> None:
    """Parse arguments, build the app and run it."""
    args = prepare_parser().parse_args(argv)

    # Convert n_sample to None if it's zero or negative (meaning "process all")
    n_sample = args.n_sample if args.n_sample and args.n_sample > 0 else None

    app = GenAIClassifierApp(
        dataset_dir=args.dataset_dir,
        prompts_dir=args.prompts_dir,
        llm_router_url=args.llm_router_url,
        model_name=args.model_name,
        temperature=args.temperature,
        batch_save_size=args.batch_save_size,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        verbose=args.verbose,
        num_workers=args.num_workers,
        n_sample=n_sample,
        export_xlsx=args.export_xlsx,
        text_column_name=args.text_column_name,
    )
    app.run()


if __name__ == "__main__":
    main()
