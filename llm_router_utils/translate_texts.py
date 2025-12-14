"""
Command‑line utility for translating texts using an LLM Router service.

The script accepts one or more dataset files (JSON or JSON Lines) that contain
records with a ``text`` field.  It loads the records, extracts the texts,
sends them to the router for translation, and prints the JSON response.

Example usage
-------------
$ python translate_texts.py \\
    --llm-router-host http://localhost:8000 \\
    --model speakleash/Bielik-11B-v2.3-Instruct \\
    --dataset-path data1.jsonl \\
    --dataset-path data2.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from llm_router_lib import LLMRouterClient
from rdl_ml_utils.utils.loaders import JSONLLoader, JSONLoader


def prepare_parser(description: str = "") -> argparse.ArgumentParser:
    """
    Build the ``argparse`` parser for the command‑line interface.

    Parameters
    ----------
    description: str, optional
        Human‑readable description shown in the help output.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
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

    return parser


def infer_dataset_type(path: Path) -> str:
    """
    Infer the dataset type from a file's extension.

    Parameters
    ----------
    path: pathlib.Path
        Path to the dataset file.

    Returns
    -------
    str
        Either ``"json"`` or ``"jsonl"``.

    Raises
    ------
    ValueError
        If the extension is not recognised.
    """
    ext = path.suffix.lower()
    if ext == ".json":
        return "json"
    if ext == ".jsonl":
        return "jsonl"
    raise ValueError(
        f"Cannot infer dataset type from extension '{ext}' for file {path}"
    )


def load_records(
    dataset_paths: List[str], dataset_type: str | None
) -> List[Dict[str, Any]]:
    """
    Load all records from the supplied dataset files.

    Parameters
    ----------
    dataset_paths: list[str]
        List of file paths provided via ``--dataset-path``.
    dataset_type: str | None
        Explicit dataset type supplied via ``--dataset-type``. If ``None``,
        each file's type is inferred from its extension.

    Returns
    -------
    list[dict]
        Flattened list of dictionaries read from the files.
    """
    records: List[Dict[str, Any]] = []

    for path_str in dataset_paths:
        path = Path(path_str)

        # Determine which loader to use
        dtype = dataset_type or infer_dataset_type(path)

        if dtype == "json":
            loader = JSONLoader(path)
        else:  # "jsonl"
            loader = JSONLLoader(path)

        # Extend the global list with the generator's output
        records.extend(list(loader.load()))

    return records


def extract_texts(records: List[Dict[str, Any]]) -> List[str]:
    """
    Pull the ``text`` field from each record.

    Parameters
    ----------
    records: list[dict]
        Records loaded from the dataset files.

    Returns
    -------
    list[str]
        List of texts to be sent for translation. Records without a ``text``
        key are silently ignored.
    """
    return [record["text"] for record in records if "text" in record]


def main(argv: List[str] | None = None) -> None:
    """
    Entry point for the script.

    Parses command‑line arguments, loads dataset files, extracts texts,
    calls the LLM Router ``translate`` endpoint, and prints the formatted
    JSON response.
    """
    args = prepare_parser().parse_args(argv)

    # ----------------------------------------------------------------------
    # Load dataset records
    # ----------------------------------------------------------------------
    records = load_records(args.dataset_path, args.dataset_type)

    # ----------------------------------------------------------------------
    # Prepare texts for translation
    # ----------------------------------------------------------------------
    texts = extract_texts(records)

    # If no texts were found we fall back to a single example string.
    if not texts:
        texts = ["Hello, how are you?"]

    # ----------------------------------------------------------------------
    # Call the LLM Router service
    # ----------------------------------------------------------------------
    client = LLMRouterClient(api=args.llm_router_host)

    response = client.translate(
        model=args.model,
        texts=texts,
    )

    # Pretty‑print the JSON response
    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
