import re
from pathlib import Path
from typing import Optional

from datasets import load_dataset, load_from_disk


class HfDatasetHandler:
    @staticmethod
    def normalize_dataset_id(dataset_id: str) -> str:
        """
        Accepts:
          - "usham/mental-health-companion-new"
          - "https://huggingface.co/datasets/jquiros/suicide"
        Returns:
          - "usham/mental-health-companion-new"
          - "jquiros/suicide"
        """
        s = dataset_id.strip().strip(",").strip()
        s = re.sub(r"^https?://", "", s)
        s = re.sub(r"^www\.", "", s)

        m = re.match(r"^huggingface\.co/datasets/([^/?#]+/[^/?#]+)", s)
        if m:
            return m.group(1)

        if re.match(r"^[^/\s]+/[^/\s]+$", s):
            return s

        raise ValueError(f"Nie rozpoznaję identyfikatora datasetu: {dataset_id!r}")

    @staticmethod
    def safe_dirname(dataset_id: str) -> str:
        """
        Must match the downloader's mapping:
          "org/name" -> "org__name"
        """
        return dataset_id.replace("/", "__")

    @staticmethod
    def load_saved_dataset(
        dataset_id: str,
        data_dir: Path,
        config: Optional[str] = None,
    ):
        """
        Loads dataset from: data_dir/<org__name> or data_dir/<org__name>/<config>
        Returns Dataset or DatasetDict.
        """
        base = data_dir / HfDatasetHandler.safe_dirname(
            dataset_id=HfDatasetHandler.normalize_dataset_id(dataset_id=dataset_id)
        )
        path = base if config is None else (base / config)

        if not path.exists():
            raise FileNotFoundError(
                f"Nie znaleziono datasetu na dysku: {dataset_id} (config={config})\n"
                f"Oczekiwana ścieżka: {path}"
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
        Downloads dataset via `datasets.load_dataset` and saves to disk.
        Output dir: data_dir/<org__name> or data_dir/<org__name>/<config>
        """

        out_base = data_dir / HfDatasetHandler.safe_dirname(
            dataset_id=HfDatasetHandler.normalize_dataset_id(dataset_id=dataset_id)
        )
        out_dir = out_base if config is None else (out_base / config)

        out_dir.parent.mkdir(parents=True, exist_ok=True)

        print(
            f"\n=== Pobieram: {dataset_id}"
            + (f" (config={config})" if config else "")
            + " ==="
        )
        ds = load_dataset(dataset_id, name=config, revision=revision)

        # `ds` is either DatasetDict (common) or Dataset
        print(f"Zapisuję do: {out_dir}")
        ds.save_to_disk(str(out_dir))
        return out_dir
