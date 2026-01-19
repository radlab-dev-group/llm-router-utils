"""
Utility module for handling Hugging Face datasets.

This module provides a small collection of static helper methods that make it
easier to work with datasets from the Hugging Face Hub:

* Normalise a dataset identifier supplied either as a short ``org/name`` string
  or as a full URL.
* Convert a normalised identifier into a file‑system‑safe directory name.
* Load a previously downloaded dataset from a local directory.
* Download a dataset (optionally a specific configuration or revision) and
  persist it to disk.

All public functions are implemented as ``@staticmethod``s on the
``HfDatasetHandler`` class, allowing them to be used without instantiating the
class.
"""

import re

from pathlib import Path
from typing import Optional

from datasets import load_dataset, load_from_disk


class HfDatasetHandler:
    """
    Collection of static utilities for working with Hugging Face datasets.

    The class does not maintain any state; each method operates purely on the
    arguments provided and returns the result directly.
    """

    @staticmethod
    def normalize_dataset_id(dataset_id: str) -> str:
        """
        Normalise a Hugging Face dataset identifier.

        The function accepts either a short ``org/name`` identifier or a full
        URL pointing to the dataset on the Hub and returns the canonical
        ``org/name`` form.

        Parameters
        ----------
        dataset_id: str
            The identifier to normalise. Examples:
            * ``"usham/mental-health-companion-new"``
            * ``"https://huggingface.co/datasets/jquiros/suicide"``

        Returns
        -------
        str
            The normalised ``org/name`` identifier.

        Raises
        ------
        ValueError
            If the supplied string cannot be interpreted as a valid dataset
            identifier.
        """
        s = dataset_id.strip().strip(",").strip()
        s = re.sub(r"^https?://", "", s)
        s = re.sub(r"^www\.", "", s)

        # Match full Hub URLs like ``huggingface.co/datasets/org/name``.
        m = re.match(r"^huggingface\.co/datasets/([^/?#]+/[^/?#]+)", s)
        if m:
            return m.group(1)

        # Accept plain ``org/name`` strings.
        if re.match(r"^[^/\s]+/[^/\s]+$", s):
            return s

        raise ValueError(
            f"I do not recognise the dataset identifier: {dataset_id!r}"
        )

    @staticmethod
    def safe_dirname(dataset_id: str) -> str:
        """
        Convert a normalised dataset identifier into a directory‑safe name.

        The Hugging Face downloader stores datasets under a directory whose name
        replaces the forward slash with a double underscore (``org/name`` →
        ``org__name``). This helper reproduces that mapping.

        Parameters
        ----------
        dataset_id: str
            The normalised ``org/name`` identifier.

        Returns
        -------
        str
            A file‑system‑safe directory name.
        """
        return dataset_id.replace("/", "__")

    @staticmethod
    def load_saved_dataset(
        dataset_id: str,
        data_dir: Path,
        config: Optional[str] = None,
    ):
        """
        Load a dataset that has been previously downloaded to disk.

        The function looks for the dataset under ``data_dir/<org__name>`` or,
        if a configuration name is supplied, under ``data_dir/<org__name>/<config>``.

        Parameters
        ----------
        dataset_id: str
            The dataset identifier (any form accepted by :meth:`normalize_dataset_id`).
        data_dir: pathlib.Path
            Base directory where datasets are stored.
        config: str, optional
            Specific configuration name to load (for ``DatasetDict`` objects).

        Returns
        -------
        datasets.Dataset or datasets.DatasetDict
            The loaded dataset object.

        Raises
        ------
        FileNotFoundError
            If the expected directory does not exist.
        """
        base = data_dir / HfDatasetHandler.safe_dirname(
            dataset_id=HfDatasetHandler.normalize_dataset_id(dataset_id=dataset_id)
        )
        path = base if config is None else (base / config)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found on disk: {dataset_id} (config={config})\n"
                f"Expected path: {path}"
            )

        ds = load_from_disk(str(path))
        return ds

    @staticmethod
    def download_and_save_dataset(
        dataset_id: str,
        data_dir: Path,
        *,
        config: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> Path:
        """
        Download a Hugging Face dataset and persist it to a local directory.

        The dataset is fetched via ``datasets.load_dataset`` and then saved to
        ``data_dir/<org__name>`` (or ``data_dir/<org__name>/<config>`` when a
        configuration is specified).

        Parameters
        ----------
        dataset_id: str
            The identifier of the dataset to download (any form accepted by
            :meth:`normalize_dataset_id`).
        data_dir: pathlib.Path
            Directory where the dataset should be stored.
        config: str, optional
            Name of the configuration to download (for ``DatasetDict`` objects).
        revision: str, optional
            Specific revision (e.g., a git commit hash or tag) to download.

        Returns
        -------
        pathlib.Path
            Path to the directory where the dataset was saved.
        """
        out_base = data_dir / HfDatasetHandler.safe_dirname(
            dataset_id=HfDatasetHandler.normalize_dataset_id(dataset_id=dataset_id)
        )
        out_dir = out_base if config is None else (out_base / config)

        out_dir.parent.mkdir(parents=True, exist_ok=True)

        print(
            "\n=== Downloading: "
            + dataset_id
            + (f" (config={config})" if config else "")
            + " ==="
        )
        ds = load_dataset(dataset_id, name=config, revision=revision)

        # ``ds`` can be either a ``DatasetDict`` (common) or a single ``Dataset``.
        print(f"Saving to: {out_dir}")
        ds.save_to_disk(str(out_dir))
        return out_dir
