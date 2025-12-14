"""
Command‑line utility for translating texts using an LLM Router service.

The script accepts one or more dataset files (JSON or JSON Lines) that contain
records with a ``text`` field. It loads the records, extracts the texts,
sends them to the router for translation, and prints the JSON response.

Example usage
-------------
$ python translate_texts.py \\
    --llm-router-host http://localhost:8000 \\
    --model speakleash/Bielik-11B-v2.3-Instruct \\
    --dataset-path data1.jsonl \\
    --dataset-path data2.json \\
    --accept-field text \\
    --accept-field title
"""

import argparse
from typing import List

from llm_router_utils.core.apps.translate import TranslateApp


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def prepare_parser(description: str = "") -> argparse.ArgumentParser:
    """Build the ``argparse`` parser for the command‑line interface."""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--llm-router-host",
        required=True,
        help="Base URL of the LLM router service (e.g., http://localhost:port)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to use for translation "
        "(e.g., speakleash/Bielik-11B-v2.3-Instruct)",
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        required=True,
        help="Path to a dataset file. This option can be provided multiple times "
        "to process several files.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["json", "jsonl"],
        help="Explicit type of dataset files (json or jsonl). "
        "If omitted, the type is inferred from each file's extension.",
    )
    parser.add_argument(
        "--accept-field",
        action="append",
        default=[],
        help="Name of a field to retain from each record. "
        "Can be supplied multiple times; if omitted all fields are kept.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads for parallel translation "
        "(default: 1 – runs sequentially).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="How many texts to send in a single request to the router "
        "(default: 8).",
    )
    return parser


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(argv: List[str] | None = None) -> None:
    """Parse arguments, build the app and run it."""
    args = prepare_parser().parse_args(argv)
    app = TranslateApp(args)
    app.run()


if __name__ == "__main__":
    main()
