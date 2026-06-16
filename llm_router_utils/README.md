# llm‑router‑utils

Utilities for working with an **LLM Router** service – a lightweight toolkit that helps you translate text datasets,
handle Hugging Face datasets, and integrate the router into Python applications.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start (CLI)](#quick-start-cli)
    - [Translation](#translation)
    - [GenAI Classification](#genai-classification)
    - [GenAI Data Augmentation](#genai-data-augmentation)
- [Programmatic Usage](#programmatic-usage)
- [Dataset Helpers (Hugging Face)](#dataset-helpers-huggingface)
- [License](#license)

---

## Overview

`llm-router-utils` bundles a small but handy set of helpers around the **LLM Router** HTTP API:

* **CLI** (`translate_texts.py`) – translate one or more JSON/JSON‑Lines datasets in a single command.
* **GenAI Classifier** (`genai_classifier.py`) – classify text datasets using prompts and LLM Router.
* **GenAI Data Augmentation** (`genai_data_augmentation.py`) – augment text datasets by generating new examples based on
  existing samples.
* **Core library** – reusable classes (`TranslateApp`, `GenAIClassifierApp`, `GenAIDataAugmentationApp`,
  `TextTranslationService`) for embedding logic in your own scripts.
* **Hugging Face dataset utilities** – normalize identifiers, safely map them to local directories, download, and load
  datasets.

The package is deliberately lightweight and has no external runtime dependencies beyond what is already listed in the
project’s `requirements.txt`/environment.

---

## Features

| Feature                    | Description                                                                                      |
|----------------------------|--------------------------------------------------------------------------------------------------|
| **Batch translation**      | Sends texts to the router in configurable batch sizes (default 8).                               |
| **Parallel execution**     | Optional multi‑threaded mode with a configurable worker pool.                                    |
| **Field selection**        | Keep only the fields you care about from each record (`--accept‑field`).                         |
| **Dataset type detection** | Handles both `.json` and `.jsonl` files, auto‑detecting the format.                              |
| **Hugging Face helpers**   | Normalise dataset IDs, convert them to safe directory names, download and load datasets locally. |
| **Progress feedback**      | `tqdm` progress bars for both single‑ and multi‑threaded translation runs.                       |

---

## Installation

The library is pure Python and can be installed directly from the repository:

```shell script
# Clone the repo
git clone https://github.com/radlab-dev-group/llm-router-utils.git
cd llm-router-utils

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package in editable mode
pip install -e .
```

---

## Quick Start (CLI)

### Translation

Translate one or more JSON/JSON‑Lines files using the bundled CLI tool:

```shell script
translate-texts \
    --llm-router-host http://localhost:8000 \
    --model speakleash/Bielik-11B-v2.3-Instruct \
    --dataset-path data1.jsonl \
    --dataset-path data2.json \
    --accept-field text \
    --accept-field title \
    --batch-size 16 \
    --num-workers 4
```

**Explanation of the most useful flags**

| Flag                | Purpose                                                        |
|---------------------|----------------------------------------------------------------|
| `--llm-router-host` | Base URL of the LLM Router service.                            |
| `--model`           | Model name to be used for translation.                         |
| `--dataset-path`    | Path to a dataset file; can be repeated.                       |
| `--accept-field`    | Fields that should be retained (and translated) in the output. |
| `--batch-size`      | Number of texts per API request (default 8).                   |
| `--num-workers`     | Number of parallel threads (default 1 → sequential).           |

---

### GenAI Classification

Classify datasets using LLM Router and specific prompts for each feature.

```shell script
genai-classifier \
    --dataset-dir ./data \
    --prompts-dir ./prompts \
    --llm-router-url http://localhost:8080 \
    --model-name gpt-oss:120b \
    --num-workers 2 \
    --n-sample 50
```

**Key flags for Classifier**

| Flag                 | Purpose                                                                |
|----------------------|------------------------------------------------------------------------|
| `--dataset-dir`      | Directory containing `.xlsx` or `.jsonl` datasets.                     |
| `--prompts-dir`      | Directory containing prompt files (one per feature).                   |
| `--llm-router-url`   | Base URL of the LLMRouter service.                                     |
| `--model-name`       | Model identifier passed to the router.                                 |
| `--n-sample`         | Number of random samples per field (default: 50, use 0 for all).       |
| `--num-workers`      | Number of parallel worker threads.                                     |
| `--text-column-name` | Name of the column containing the text to classify (default: `Tekst`). |

**Outputs:**

- Generates a main `.jsonl` file with all LLM responses.
- Generates a `*_for_augmentation.jsonl` file with simplified labels for use with the augmentator.
- Automatically exports to `.xlsx` (if not disabled by `--no-export-xlsx`).

---

### GenAI Data Augmentation

Augment datasets by generating new examples based on existing samples.

```shell script
genai-data-augmentation \
    --dataset-path ./data/dataset_for_augmentation.jsonl \
    --prompt-file ./prompts/augmentation_prompt.txt \
    --labels "label1,label2" \
    --n-samples 5 \
    --samples-as-examples 5 \
    --n-examples 3 \
    --num-workers 2
```

**Key flags for Augmentation**

| Flag                  | Purpose                                                               |
|-----------------------|-----------------------------------------------------------------------|
| `--dataset-path`      | Path to the dataset file (XLSX or JSONL).                             |
| `--prompt-file`       | Path to the prompt file defining how to augment.                      |
| `--labels`            | Comma-separated list of labels for which data should be augmented.    |
| `--n-samples`         | Number of random samples per class used as examples for augmentation. |
| `--samples-as-examples` | Number of random samples per class from the dataset to include in the prompt context. |
| `--n-examples`        | Number of augmented examples the LLM should generate for each input text. |
| `--text-column-name`  | Name of the column containing the text (default: `Tekst`).            |
| `--label-column-name` | Name of the column containing the labels (default: `labels`).         |

**Outputs:**

- Generates `*_augmented.jsonl` and `*_augmented.xlsx` with original texts and their augmented versions.

The command prints a stream of JSON objects or progress information depending on the tool.

---

## Programmatic Usage

You can also drive the translation pipeline from Python code:

```python
import argparse
from llm_router_utils.core.apps.translate import TranslateApp

# Build an argparse.Namespace manually (or reuse a parser)
args = argparse.Namespace(
    llm_router_host="http://localhost:8000",
    model="speakleash/Bielik-11B-v2.3-Instruct",
    dataset_path=["data1.jsonl", "data2.json"],
    dataset_type=None,  # auto‑detect
    accept_field=["text", "title"],
    num_workers=2,
    batch_size=8,
)

app = TranslateApp(args)
app.run()

# `app.translations` now holds a list of JSON strings with translated fields
for line in app.translations:
    print(line)
```

The `TranslateApp` class encapsulates the whole workflow:

1. Load records via `DatasetProcessor`.
2. Flatten selected fields.
3. Translate, Classify or Augment in batches (single‑ or multi‑threaded).
4. Re‑assemble the original structure and expose the result.

---

## Dataset Helpers (Hugging Face)

The `HfDatasetHandler` class offers static utilities for working with Hugging Face datasets:

```python
from pathlib import Path
from llm_router_utils.core.hf_dataset_handler import HfDatasetHandler

# Normalise an identifier (accepts short or full URL)
norm_id = HfDatasetHandler.normalize_dataset_id(
    "https://huggingface.co/datasets/jquiros/suicide"
)  # → "jquiros/suicide"

# Convert to a safe directory name
safe_name = HfDatasetHandler.safe_dirname(norm_id)  # → "jquiros__suicide"

# Download and save locally
data_dir = Path("./datasets")
local_path = HfDatasetHandler.download_and_save_dataset(
    dataset_id=norm_id,
    data_dir=data_dir,
    config=None,
    revision=None,
)

# Load the saved copy later
dataset = HfDatasetHandler.load_saved_dataset(
    dataset_id=norm_id,
    data_dir=data_dir,
    config=None,
)
```

These helpers make it easy to cache datasets on disk, avoid naming collisions, and load them without repeatedly hitting
the Hub.

---

## License

`llm-router-utils` is released under the **Apache License 2.0**. See the `LICENSE` file in the repository for the full
text.