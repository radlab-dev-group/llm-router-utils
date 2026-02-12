# genai_classifier/core/apps/classifier.py
"""Core implementation of the GenAI‑classifier pipeline.

The class is deliberately lightweight: it only receives an ``argparse.Namespace``,
builds a :class:`~genai_classifier.genai_classifier_types_of_services.Config`
object, and then runs the original processing logic (now factored out into
``run_with_config``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
#
# from llm_router_utils.core.genai_classifier_types_of_services import (
#     Config,
#     run_with_config,
# )


class GenAIClassifierApp:
    """
    High‑level orchestrator usable from the CLI **or** from other Python code.

    Parameters
    ----------
    args : argparse.Namespace
        The namespace produced by the CLI parser.  It must contain **all**
        attributes that ``Config`` expects (the same names as the original
        ``--`` options).

    Example – programmatic use
    -------------------------
    >>> from argparse import Namespace
    >>> from pathlib import Path
    >>> from genai_classifier.core.apps.classifier import GenAIClassifierApp
    >>> args = Namespace(
    ...     dataset_dir=Path("./dataset_path"),
    ...     prompts_dir=Path("./resources/prompts/"),
    ...     llm_router_url="http://192.168.100.65:8080",
    ...     model_name="gpt-oss:120b",
    ...     temperature=0.0,
    ...     batch_save_size=5,
    ...     dry_run=False,
    ...     output_dir=Path("./results"),
    ...     verbose=True,
    ...     num_workers=4,
    ...     n_sample=50,
    ... )
    >>> GenAIClassifierApp(args).run()
    """

    def __init__(self, args: Any):
        # ``args`` may be a real ``argparse.Namespace`` or any object with the
        # same attribute names (useful for tests).
        self._config = Config(
            dataset_dir=Path(args.dataset_dir),
            prompts_dir=Path(args.prompts_dir),
            llm_router_url=args.llm_router_url,
            model_name=args.model_name,
            temperature=args.temperature,
            batch_save_size=args.batch_save_size,
            dry_run=args.dry_run,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            verbose=args.verbose,
            num_workers=args.num_workers,
            n_sample=args.n_sample,
        )
        # Validation is performed in ``Config._validate``.
        self._config._validate()

    def run(self) -> None:
        """
        Execute the full classification workflow.

        The heavy lifting lives in ``run_with_config`` (which contains the
        original ``main`` logic without the CLI boilerplate).  Keeping the
        implementation here tiny makes the class easy to unit‑test.
        """
        # If the user asked for verbose output we bump the logger level.
        if self._config.verbose:
            logging.getLogger(__name__).setLevel(logging.DEBUG)

        run_with_config(self._config)