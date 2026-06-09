# llm-router-utils

**v0.0.3** — Utility package for [LLM Router](https://github.com/radlab-dev-group/llm-router) – ready-made tools and examples for text translation, dataset classification, and HuggingFace dataset management.

## What is it?

`llm-router-utils` is a collection of **CLI tools** and **reusable helpers** built on top of **LLM Router** – a flexible language-model router that load-balances across multiple backends (vLLM, llama.cpp, Ollama).

| Tool | Description |
|------|-------------|
| `translate-texts` | Batch-translate texts in JSON / JSONL datasets via the LLM Router API. Supports multi-threading, configurable batch sizes, and field selection. |
| `genai-classifier` | Classify texts from HuggingFace datasets or XLSX files against custom prompts, with optional automatic Excel (XLSX) export. |
| `HfDatasetHandler` | Static helpers for normalising dataset IDs, downloading & saving datasets to disk, and loading them back. |
| `dataset_list` | Pre-configured mental-health dataset field mappings for translation and classification pipelines. |
| `jsonl_to_xlsx` | Convert classifier JSONL output into beautifully formatted Excel workbooks. |

## Features

| Feature | Description |
|---------|-------------|
| **Batch translation** | Sends texts to the router in configurable batch sizes (default 8). |
| **Parallel execution** | Optional multi-threaded mode with a configurable worker pool. |
| **Field selection** | Keep only the fields you care about (`--accept-field`). |
| **Dataset type detection** | Auto-detects `.json` and `.jsonl` files. |
| **GenAI classification** | Classify texts against prompt-defined classes with retry logic and JSON parsing. |
| **XLSX export** | Automatic or manual conversion of JSONL to formatted Excel with zebra striping and conditional colors. |
| **HuggingFace helpers** | Normalise identifiers, download, save to disk, and load datasets locally. |
| **Progress feedback** | `tqdm` progress bars for both single- and multi-threaded runs. |

## Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/radlab-dev-group/llm-router-utils.git
cd llm-router-utils

# 2️⃣ Install the package (editable mode is handy during development)
pip install -e .

# 3️⃣ Install optional dependencies
pip install ".[llm-router]"   # llm-router + llm-router-services from git

# The extra pulls in:
#    - llm-router @ git+https://github.com/radlab-dev-group/llm-router
#    - llm-router-services @ git+https://github.com/radlab-dev-group/llm-router-services

# Additional runtime dependencies (for classifier features):
pip install rdl-ml-utils datasets pandas openpyxl
```

## Quick Start

### Batch Translation (`translate-texts`)

```bash
translate-texts \
    --llm-router-host http://localhost:8080 \
    --model speakleash/Bielik-11B-v2.3-Instruct \
    --dataset-path data.jsonl \
    --accept-field questionText \
    --accept-field title \
    --batch-size 16 \
    --num-workers 4
```

**Key flags:**

| Flag | Purpose |
|------|---------|
| `--llm-router-host` | Base URL of the LLM Router service. |
| `--model` | Model name for translation (e.g., `speakleash/Bielik-11B-v2.3-Instruct`). |
| `--dataset-path` | Path to a dataset file; repeatable. |
| `--accept-field` | Fields to retain and translate; repeatable. |
| `--batch-size` | Texts per API request (default 8). |
| `--num-workers` | Parallel threads (default 1 → sequential). |
| `--dataset-type` | Force `json` or `jsonl`; otherwise auto-detected from extension. |

### GenAI Classifier (`genai-classifier`)

```bash
genai-classifier \
    --dataset-dir ./datasets \
    --prompts-dir ./prompts \
    --llm-router-url http://localhost:8080 \
    --model-name "speakleash/Bielik-11B-v2.3-Instruct" \
    --temperature 0.0 \
    --num-workers 4 \
    --n-sample 100 \
    --output-dir ./output \
    --verbose
```

**Key flags:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset-dir` | `Path` | **(required)** | Directory with HF datasets or XLSX files. |
| `--prompts-dir` | `Path` | **(required)** | Directory with `.prompt` files (one per class). |
| `--llm-router-url` | `str` | `http://192.168.100.65:8080` | LLM Router URL. |
| `--model-name` | `str` | `gpt-oss:120b` | Model identifier for the router. |
| `--temperature` | `float` | `0.0` | Generation temperature (0 = deterministic). |
| `--num-workers` | `int` | `2` | Parallel worker threads. |
| `--batch-save-size` | `int` | `5` | Records flushed to disk per batch. |
| `--n-sample` | `int` | `50` | Random samples per field (0 or omit = all). |
| `--dry-run` | `flag` | `false` | Process without writing output. |
| `--output-dir` | `Path` | *(dataset-dir)* | Override output directory. |
| `--verbose` | `flag` | `false` | DEBUG-level logging. |
| `--export-xlsx` | `flag` | `true` | Export results to Excel. |
| `--no-export-xlsx` | `flag` | `false` | Disable XLSX export. |

## Programmatic Usage

### Translation

```python
from llm_router_utils.core.apps.translate import TranslateApp
import argparse

args = argparse.Namespace(
    llm_router_host="http://localhost:8080",
    model="speakleash/Bielik-11B-v2.3-Instruct",
    dataset_path=["data.jsonl", "data.json"],
    dataset_type=None,        # auto-detect
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

### Classification

```python
from pathlib import Path
from llm_router_utils.core.apps.genai_classifier import GenAIClassifierApp

app = GenAIClassifierApp(
    dataset_dir=Path("./datasets"),
    prompts_dir=Path("./prompts"),
    llm_router_url="http://localhost:8080",
    model_name="speakleash/Bielik-11B-v2.3-Instruct",
    temperature=0.0,
    num_workers=4,
    n_sample=100,
    export_xlsx=True,
)
app.run()
```

### HuggingFace Dataset Helpers

```python
from pathlib import Path
from llm_router_utils.core.hf_dataset_handler import HfDatasetHandler

# Normalise an identifier (accepts short org/name or full URL)
norm_id = HfDatasetHandler.normalize_dataset_id(
    "https://huggingface.co/datasets/jquiros/suicide"
)  # → "jquiros/suicide"

# Safe directory name
safe_name = HfDatasetHandler.safe_dirname(norm_id)  # → "jquiros__suicide"

# Download and save locally
data_dir = Path("./datasets")
HfDatasetHandler.download_and_save_dataset(
    dataset_id=norm_id,
    data_dir=data_dir,
)

# Load the saved copy later
dataset = HfDatasetHandler.load_saved_dataset(norm_id, data_dir, config="train")
```

### JSONL → XLSX

```python
from pathlib import Path
from llm_router_utils.core.jsonl_to_xlsx import convert_jsonl_to_xlsx

convert_jsonl_to_xlsx(
    jsonl_path=Path("output/dataset.jsonl"),
    xlsx_path=Path("output/dataset.xlsx"),
)
```

## Dataset List

The package ships with `dataset_list.py` containing pre-configured mental-health datasets and their simplified field mappings:

| Dataset | Simplified fields |
|---------|-------------------|
| `nbertagnolli/counsel-chat` | `questionText` |
| `ShenLab/MentalChat16K` | `input` |
| `amaye15/suicide-descriptions` | `text` |
| `marmikpandya/mental-health` | `input` |
| `usham/mental-health-companion-new` | `input` |

See `llm_router_utils/dataset_list.py` for the full list.

## Repository Structure

```
llm-router-utils/
├── llm_router_utils/                  # Python package
│   ├── __init__.py
│   ├── README.md                      # Package-level documentation
│   ├── dataset_list.py                # Pre-configured dataset field mappings
│   ├── cli/
│   │   ├── translate_texts.py         # translate-texts CLI entry point
│   │   └── genai_classifier.py        # genai-classifier CLI entry point
│   ├── core/
│   │   ├── apps/
│   │   │   ├── translate.py           # TranslateApp + TextTranslationService
│   │   │   └── genai_classifier.py    # GenAIClassifierApp
│   │   ├── hf_dataset_handler.py      # HfDatasetHandler (HF dataset utilities)
│   │   └── jsonl_to_xlsx.py           # JSONL → XLSX conversion
│   └── README.md
├── resources/llm-router-speakleash/   # Speakleash model configs & launch scripts
│   ├── configs/speakleash-models.json
│   ├── run-bielik-11b-v2_3-vllm_*.sh  # vLLM startup scripts (one per GPU)
│   ├── run-sojka-guardrail.sh          # Bielik-Guard guardrail service
│   ├── run-rest-api-gunicorn.sh        # LLM Router REST API (Gunicorn)
│   └── README[-en].md
├── cache-translations/                # Cached translation outputs
├── CLASSIFIER_DATASET.md              # Dataset generation guide (English)
├── CLASSIFIER_DATASET_PL.md           # Guide do generowania datasetu (Polish)
├── pyproject.toml                     # Build config
├── setup.py                           # Package setup with entry points
├── requirements.txt                   # Core runtime deps (tqdm, datasets)
├── .version                           # Current version (0.0.3)
└── README.md
```

## Documentation

- **[CLASSIFIER_DATASET.md](CLASSIFIER_DATASET.md)** — Full end-to-end guide for dataset generation with LLM Router and GenAI Classifier (English).
- **[CLASSIFIER_DATASET_PL.md](CLASSIFIER_DATASET_PL.md)** — Przewodnik generowania datasetu (Polish).
- **[resources/llm-router-speakleash/README.md](resources/llm-router-speakleash/README.md)** — Speakleash model configuration for LLM Router.

## License

`llm-router-utils` is released under the **Apache License 2.0**. See the `LICENSE` file for the full text.
