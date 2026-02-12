#!/usr/bin/env python
# genai_classifier/cli/genai_classifier_cli.py
"""Console‑script entry point for the GenAI‑classifier.

The CLI mirrors the arguments that were previously defined in
``Config.from_cli()``.  All heavy lifting lives in the core ``classifier`` module,
so this file stays tiny and testable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from llm_router_utils.core.apps.genai_classifier import GenAIClassifierApp


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the ``argparse`` parser that matches the original ``Config`` CLI.
    The help strings are kept short – run ``--help`` for the full list.
    """
    parser = argparse.ArgumentParser(
        description="Classify HuggingFace datasets using an LLMRouter service."
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("./downloaded_translated_to_pl"),
        help="Directory containing downloaded HF datasets.",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=Path("./resources/prompts"),
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
        help="Process data but do **not** write output files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override directory where result .jsonl files are stored.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG‑level logging.",
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
            "Number of random samples per field (default: 50). "
            "If omitted, zero or negative, all examples are processed."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """
    Parse CLI arguments, instantiate the core app and run it.

    ``argv`` is injected only for unit‑testing; when ``None`` the arguments
    from ``sys.argv`` are used automatically by ``argparse``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # The core class does the heavy lifting.
    app = GenAIClassifierApp(args)
    app.run()


if __name__ == "__main__":
    # ``sys.argv[1:]`` is passed implicitly by ``argparse`` when ``argv=None``.
    main()
