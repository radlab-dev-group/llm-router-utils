# Guide: Dataset Generation with LLM Router and GenAI Classifier

## Table of Contents

1. [Introduction](#1-introduction)
2. [Solution Architecture](#2-solution-architecture)
3. [Local Environment Setup](#3-local-environment-setup)
   - [vLLM](#31-vllm)
   - [llama.cpp](#32-llamacpp)
   - [Ollama](#33-ollama)
4. [LLM Router Configuration](#4-llm-router-configuration)
5. [genai-classifier Utility](#5-genai-classifier-utility)
6. [Complete End-to-End Example](#6-complete-end-to-end-example)
7. [Output Format](#7-output-format)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Introduction

This document describes the full process of generating datasets for NLP model training using:

- **LLM Router** — an API gateway that load-balances across multiple LLM backends (vLLM, llama.cpp, Ollama)
- **genai-classifier** — a utility from the `llm-router-utils` package that classifies texts using an LLM according to provided prompts
- **translate-texts** — a utility for batch text translation

**Typical use cases**:
- Data augmentation (classification, attribute extraction, categorization)
- Preparing training data for fine-tuning
- Evaluating content for presence of specific classes (e.g., hate speech detection, topic categorization)
- Batch translation and transliteration

---

## 2. Solution Architecture

```
+--------------------------------------------------------------+
|                      Data Flow                               |
|                                                              |
|  HF Dataset / XLSX  →  genai-classifier  →  JSONL output   |
|       (source)              (LLM Router)        (result)    |
|                             │                                |
|                      LLMRouterClient                         |
|                             │                                |
|              ┌──────────────┼──────────────┐                 |
|              │              │              │                 |
|        ┌─────▼──────┐  ┌───▼────┐  ┌──────▼──────┐         |
|        │ vLLM       │  │llama.cpp│  │  Ollama     │         |
|        │ provider   │  │provider │  │  provider   │         |
|        └────────────┘  └────────┘  └─────────────┘         |
+--------------------------------------------------------------+
```

### Components

| Component | Role |
|---|---|
| **`llm_router_lib.client.LLMRouterClient`** | HTTP client — sends requests to the router, handles retries and errors |
| **`llm_router_api.rest_api`** | REST gateway — load-balances, proxies to backends (vLLM, Ollama, llama.cpp) |
| **`GenAIClassifierApp`** | Orchestrator — loads data, sends to LLM, aggregates results into JSONL |
| **`PromptHandler`** (from `rdl_ml_utils`) | Loads prompts from `.prompt` files in `prompts_dir` |
| **`HfDatasetHandler`** | Downloads and loads datasets from HuggingFace Hub |
| **`convert_jsonl_to_xlsx`** | Converts JSONL to Excel with formatting |

### Classification Pipeline

```
1. Load data (HF Dataset / XLSX)
   ↓
2. Extract fields to classify
   ↓
3. For each (text, field):
   a. Load prompt from prompts_dir
   b. Send to LLM Router → LLM → JSON response
   c. Parse response (exists, confidence, reason)
   ↓
4. Aggregate results into JSONL
   ↓
5. Convert to XLSX (optional)
```

---

## 3. Local Environment Setup

### 3.1 vLLM

vLLM is a low-latency, high-throughput inference engine for large language models.

#### Prerequisites

- Ubuntu 20.04+
- CUDA 11.8+
- GPU with at least 12 GB VRAM (40 GB+ recommended for models >7B)
- Python 3.10+

#### Step 1: Install vLLM and Dependencies

```bash
# Create working directory
mkdir -p ~/llm-classifier && cd ~/llm-classifier

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install vLLM (for CUDA 11.8)
pip install vllm

# Verify installation
python -c "import vllm; print(vllm.__version__)"
```

#### Step 2: Download the Model

```bash
# Example: Bielik 11B (no license acceptance required)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='speakleash/Bielik-11B-v2.3-Instruct',
    local_dir='./models/Bielik-11B-v2.3-Instruct'
)
"

# Example: Bielik 7B (requires HF token)
python3 -c "
import os
os.environ['HF_TOKEN'] = 'hf_xxxxxxxxxxxxxxxxxxxxxx'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='speakleash/Bielik-7B-Instruct',
    local_dir='./models/Bielik-7B-Instruct',
    token=os.environ['HF_TOKEN']
)
"
```

#### Step 3: Start the vLLM Server

```bash
# launch-vllm.sh — vLLM startup script
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=speakleash/Bielik-11B-v2.3-Instruct

vllm serve \
    "${MODEL_PATH}" \
    --port 7000 \
    --host 0.0.0.0 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --max-model-len=32768 \
    --max_num_seqs=8 \
    --gpu-memory-utilization=0.90
```

```bash
chmod +x launch-vllm.sh
# Run in tmux (server runs in background)
tmux new -s vllm './launch-vllm.sh'
```

> **Tip:** The vLLM server listens on `http://0.0.0.0:7000/v1/chat/completions`.
> Test: `curl http://localhost:7000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"speakleash/Bielik-11B-v2.3-Instruct","messages":[{"role":"user","content":"Test"}],"max_tokens":10}'`

---

### 3.2 llama.cpp

llama.cpp is a C++ engine with Python bindings, optimized for CPU and GPU.

#### Prerequisites

- Ubuntu 20.04+
- gcc/g++ (minimum version 7)
- Python 3.10+
- Optional: CUDA for GPU acceleration

#### Step 1: Install

```bash
mkdir -p ~/llm-classifier && cd ~/llm-classifier

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake ..
make -j$(nproc)  # compile with all cores

# Build the server (server.cpp)
make server -j$(nproc)

# Verify
./server --version
```

#### Step 2: Download and Convert the Model

```bash
cd ~/llm-classifier
# Download GGUF model (already quantized)
huggingface-cli download speakleash/Bielik-11B-v2.3-Instruct \
    --include "*.gguf" \
    --local-dir ./models/

# Or download and convert from safetensors → GGUF
python3 ../llama.cpp/convert.py ../models/Bielik-11B-v2.3-Instruct/ \
    --outtype f16 \
    --outfile ./models/Bielik-11B-v2.3-Instruct-q8_0.gguf
```

#### Step 3: Convert to KV-cache Format (GGUF Q4_K_M)

```bash
# Recommended quantization for good quality/speed balance
python3 ../llama.cpp/quantize ../models/Bielik-11B-v2.3-Instruct-q8_0.gguf \
    ../models/Bielik-11B-v2.3-Instruct-q4_k_m.gguf \
    Q4_K_M
```

#### Step 4: Start the llama.cpp Server

```bash
# launch-llama.sh — llama.cpp server startup
#!/bin/bash
cd ~/llm-classifier/llama.cpp/build

./server \
    -m ../models/Bielik-11B-v2.3-Instruct-q4_k_m.gguf \
    --port 7001 \
    --host 0.0.0.0 \
    -ngl 35  # number of layers to offload to GPU (0 = CPU only)
    --ctx-size 32768 \
    --n-predict 1024
```

```bash
chmod +x launch-llama.sh
tmux new -s llama './launch-llama.sh'
```

> **Tip:** The llama.cpp server exposes an OpenAI-compatible API on `http://localhost:7001/v1/chat/completions`.

---

### 3.3 Ollama

Ollama is a simple model runner, ideal for testing and prototyping.

#### Prerequisites

- Ubuntu 20.04+
- Python 3.10+

#### Step 1: Install Ollama

```bash
# Install Ollama (official installation script)
curl -fsSL https://ollama.com/install.sh | sh

# Start the service
systemctl start ollama
systemctl enable ollama

# Verify
ollama --version
```

#### Step 2: Pull the Model

```bash
# Pull the Bielik model
ollama pull speakleash/bielik-11b-v2.3-instruct

# Or use a smaller model (faster)
ollama pull speakleash/bielik-7b-instruct
```

#### Step 3: Verify

```bash
# List downloaded models
ollama list

# Interactive test
ollama run speakleash/bielik-11b-v2.3-instruct "Hello, how are you?"
```

> **Tip:** Ollama automatically exposes its API on `http://localhost:11434`.
> Endpoint: `http://localhost:11434/v1/chat/completions` in OpenAI-compatible format.

---

## 4. LLM Router Configuration

LLM Router acts as a gateway that load-balances across models running on vLLM, llama.cpp, and Ollama.

### 4.1 Install the Router

```bash
cd ~/llm-classifier

python3 -m venv .venv
source .venv/bin/activate

# Install the router and libraries
git clone https://github.com/radlab-dev-group/llm-router.git
cd llm-router
pip install .[api]

# Required dependencies
pip install redis

# Verify
python3 -m llm_router_api.rest_api --help
```

### 4.2 `models-config.json` Configuration File

Example configuration with multiple providers:

```json
{
  "speakleash_models": {
    "speakleash/Bielik-11B-v2.3-Instruct": {
      "providers": [
        {
          "id": "bielik-vllm-0",
          "api_host": "http://192.168.100.70:7000",
          "api_token": "",
          "api_type": "vllm",
          "input_size": 32768,
          "weight": 1.0,
          "keep_alive": "5m"
        },
        {
          "id": "bielik-llama-0",
          "api_host": "http://192.168.100.70:7001",
          "api_token": "",
          "api_type": "ollama",
          "input_size": 32768,
          "weight": 1.0,
          "keep_alive": "5m"
        },
        {
          "id": "bielik-ollama-0",
          "api_host": "http://192.168.100.70:11434",
          "api_token": "",
          "api_type": "ollama",
          "input_size": 32768,
          "weight": 1.0,
          "keep_alive": "5m"
        }
      ]
    }
  }
}
```

### 4.3 Start the Router

```bash
# launch-router.sh
#!/bin/bash
export LLM_ROUTER_MODELS_CONFIG="/home/$USER/llm-classifier/llm-router/resources/configs/models-config.json"
export LLM_ROUTER_PROMPTS_DIR="/home/$USER/llm-classifier/llm-router/resources/prompts"
export LLM_ROUTER_SERVER_TYPE="gunicorn"
export LLM_ROUTER_SERVER_PORT="8080"
export LLM_ROUTER_SERVER_WORKERS_COUNT="4"
export LLM_ROUTER_SERVER_THREADS_COUNT="16"
export LLM_ROUTER_DEFAULT_EP_LANGUAGE="pl"
export LLM_ROUTER_BALANCE_STRATEGY="balanced"
export LLM_ROUTER_REDIS_HOST="192.168.100.67"
export LLM_ROUTER_REDIS_PORT="6379"
export LLM_ROUTER_FORCE_MASKING="0"

cd ~/llm-classifier/llm-router
./run-rest-api.sh
```

> **Verify:** `curl http://localhost:8080/ping` should return `"pong"`.
> Check available models: `curl http://localhost:8080/models`

---

## 5. genai-classifier Utility

### 5.1 Install

```bash
cd ~/llm-classifier

# Install from git
pip install git+https://github.com/radlab-dev-group/llm-router-utils.git

# Or install locally
git clone https://github.com/radlab-dev-group/llm-router-utils.git
cd llm-router-utils
pip install .[llm-router]
```

### 5.2 CLI Arguments

```bash
genai-classifier --help
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset-dir` | `Path` | **(required)** | Directory with downloaded HF datasets or XLSX |
| `--prompts-dir` | `Path` | **(required)** | Directory with `.prompt` files |
| `--llm-router-url` | `str` | `http://192.168.100.65:8080` | LLM Router URL |
| `--model-name` | `str` | `gpt-oss:120b` | Model identifier to send to the router |
| `--temperature` | `float` | `0.0` | Generation temperature (0 = deterministic, 1 = creative) |
| `--num-workers` | `int` | `2` | Number of parallel worker threads (and LLM clients) |
| `--batch-save-size` | `int` | `5` | Number of records written to disk at once |
| `--n-sample` | `int` | `50` | Number of random samples per field (0 = all) |
| `--dry-run` | `flag` | `false` | Process data without writing output files |
| `--output-dir` | `Path` | *(dataset-dir)* | Output directory |
| `--verbose` | `flag` | `false` | DEBUG-level logging |
| `--export-xlsx` | `flag` | `true` | Export results to Excel (default: enabled) |
| `--no-export-xlsx` | `flag` | `false` | Disable XLSX export (save only JSONL) |

### 5.3 Prepare Prompts

Prompts are text files with the `.prompt` extension, where the filename is the **class name**.

Example directory structure:

```
prompts/
├── hate_speech.prompt
├── self_promotion.prompt
├── question.prompt
└── response.prompt
```

Example `hate_speech.prompt`:

```
You are a content analyst. Check whether the following text contains hate speech.

Output format (JSON):
{
  "exists": true|false,
  "confidence": 0.0-1.0,
  "reason": "Brief reason for the decision"
}

Behavior:
- Return EXACTLY ONE valid JSON output
- Do not add any text before or after the JSON
```

### 5.4 Prepare Data

#### From HuggingFace

```python
from llm_router_utils.core.hf_dataset_handler import HfDatasetHandler
from pathlib import Path

# Download dataset from HF
data_dir = Path("./datasets")
HfDatasetHandler.download_and_save_dataset(
    dataset_id="nbertagnolli/counsel-chat",
    data_dir=data_dir,
)
```

#### From XLSX

```python
import pandas as pd

# XLSX files are automatically loaded by GenAIClassifierApp
# Save your file in dataset_dir — it will be detected automatically
```

---

## 6. Complete End-to-End Example

### Step 1: Set up the Environment (vLLM)

```bash
# 1. Environment
mkdir -p ~/llm-classifier && cd ~/llm-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 2. vLLM + router libraries
pip install vllm
pip install git+https://github.com/radlab-dev-group/llm-router.git
pip install git+https://github.com/radlab-dev-group/llm-router-utils.git
pip install redis

# 3. Download model
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('speakleash/Bielik-11B-v2.3-Instruct', local_dir='./models/Bielik')
"
```

### Step 2: Start the Components

```bash
# Terminal 1 — vLLM
export CUDA_VISIBLE_DEVICES=0
vllm serve speakleash/Bielik-11B-v2.3-Instruct \
    --port 7000 \
    --host 0.0.0.0 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --max-model-len=32768 \
    --gpu-memory-utilization=0.90

# Terminal 2 — LLM Router (in a separate venv)
cd ~/llm-classifier/llm-router
export LLM_ROUTER_MODELS_CONFIG="$PWD/resources/configs/models-config.json"
export LLM_ROUTER_PROMPTS_DIR="$PWD/resources/prompts"
export LLM_ROUTER_SERVER_PORT="8080"
export LLM_ROUTER_BALANCE_STRATEGY="balanced"
./run-rest-api.sh

# Terminal 3 — Redis (if using first_available strategy)
redis-server
```

### Step 3: Prepare Prompts and Data

```bash
# 1. Create prompts directory
mkdir -p ~/llm-classifier/prompts

# Example prompt — medical topic detection
cat > ~/llm-classifier/prompts/medical_topic.prompt << 'EOF'
You are a medical text analyst. Determine whether the text belongs to the "medical" category.

Output format (JSON):
{
  "exists": true|false,
  "confidence": 0.0-1.0,
  "reason": "Brief reason for the decision"
}
EOF

# Example prompt — hate speech detection
cat > ~/llm-classifier/prompts/hate_speech.prompt << 'EOF'
Check whether the text contains hate speech.

Output format (JSON):
{
  "exists": true|false,
  "confidence": 0.0-1.0,
  "reason": "Brief reason for the decision"
}
EOF

# 2. Prepare data — download from HF
mkdir -p ~/llm-classifier/datasets
python3 -c "
from llm_router_utils.core.hf_dataset_handler import HfDatasetHandler
from pathlib import Path
HfDatasetHandler.download_and_save_dataset(
    dataset_id='usham/mental-health-companion-new',
    data_dir=Path('./datasets'),
)
"
```

### Step 4: Run genai-classifier

```bash
# Classify — process 100 samples from a specific field
genai-classifier \
    --dataset-dir ~/llm-classifier/datasets \
    --prompts-dir ~/llm-classifier/prompts \
    --llm-router-url http://localhost:8080 \
    --model-name "speakleash/Bielik-11B-v2.3-Instruct" \
    --temperature 0.0 \
    --num-workers 4 \
    --n-sample 100 \
    --output-dir ~/llm-classifier/output \
    --verbose

# Classify ALL data (no n-sample limit)
genai-classifier \
    --dataset-dir ~/llm-classifier/datasets \
    --prompts-dir ~/llm-classifier/prompts \
    --llm-router-url http://localhost:8080 \
    --model-name "speakleash/Bielik-11B-v2.3-Instruct" \
    --temperature 0.0 \
    --num-workers 4 \
    --n-sample 0 \
    --output-dir ~/llm-classifier/output \
    --dry-run  # test without writing files
```

### Step 5: Check Results

```bash
# Preview JSONL (first 5 lines)
head -5 ~/llm-classifier/output/usham__mental-health-companion-new.jsonl

# Full conversion to Excel (default — happens automatically)
ls ~/llm-classifier/output/usham__mental-health-companion-new.xlsx
```

---

## 7. Output Format

### 7.1 JSONL (JSON Lines)

Each line is a single record with classifications for a given field:

```json
{
  "text": "Can I take aspirin during pregnancy?",
  "field": "input",
  "features": [
    {
      "name": "medical_topic",
      "response": {
        "exists": true,
        "confidence": 0.95,
        "reason": "Text contains medical term 'aspirin' and context 'pregnancy'."
      }
    },
    {
      "name": "hate_speech",
      "response": {
        "exists": false,
        "confidence": 0.99,
        "reason": "Text is a medical question, contains no offensive content."
      }
    }
  ]
}
```

### 7.2 Output Directory Structure

```
output/
├── dataset_name__field1__field2.jsonl
├── dataset_name__field1__field2.xlsx       ← Excel with formatting
└── ...
```

### 7.3 Excel (XLSX)

Each worksheet sheet corresponds to one field (`field`). Columns:

| Column | Description |
|---|---|
| `text` | Source text (column width ≈ 55 characters) |
| `<prompt_name>` | Class name (always present) |
| `<prompt_name>-class` | Class (if available) |
| `<prompt_name>-confidence` | Confidence (0.0–1.0) |
| `<prompt_name>-exists` | Whether class is present (1/True, 0/False) |
| `<prompt_name>-reason` | Reason for the model's decision |

**Formatting**:
- Headers: bold, centered, light-blue background
- Even data rows: light-gray background (zebra striping)
- Rows with `exists=1`: light-green background
- Rows with `exists=0`: light-blue background
- Text: word-wrapped

---

## 8. Troubleshooting

### Problem: `RuntimeError: Redis is mandatory`

**Solution:**

```bash
# Install Redis
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server

# Or change the load-balancing strategy to "balanced" (does not require Redis)
export LLM_ROUTER_BALANCE_STRATEGY="balanced"
```

### Problem: `ValueError: Unknown model 'xyz'`

**Solution:** Ensure `--model-name` matches a model available at the providers in your configuration. Check `models-config.json`.

### Problem: Ollama returns `error: model not found`

**Solution:**

```bash
# Check available models
ollama list

# Check if model is downloaded
ollama pull speakleash/bielik-11b-v2.3-instruct

# Check if the service is running
curl http://localhost:11434/api/tags
```

### Problem: vLLM low VRAM

**Solution:** Reduce `--gpu-memory-utilization` or increase quantization:

```bash
vllm serve ... \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --gpu-memory-utilization=0.7
```

### Problem: `genai-classifier` timeouts

**Solution:** Reduce `--num-workers` or increase the router timeout:

```bash
export LLM_ROUTER_TIMEOUT=120
export LLM_ROUTER_EXTERNAL_TIMEOUT=300
```

### Problem: Invalid JSON from LLM

**Solution:** genai-classifier automatically retries (max 5+3=8 attempts). If the problem persists:

- Ensure the prompt contains "Return EXACTLY ONE valid JSON output"
- Set `--temperature 0.0` for deterministic responses
- Check that the `.prompt` file contains the `Output format (JSON)` section

### Problem: `ImportError: No module named 'rdl_ml_utils'`

**Solution:** `PromptHandler` comes from the `rdl_ml_utils` package:

```bash
pip install rdl-ml-utils
```

### Problem: CUDA Errors

**Solution:**

```bash
# Check if CUDA is visible
nvidia-smi

# Check CUDA version in Python
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# If vLLM does not detect GPU — install with the correct CUDA tag
pip uninstall vllm -y
pip install vllm --no-cache-dir
```

---

## Useful Commands

### Batch Translation (translate-texts)

```bash
translate-texts \
    --llm-router-host http://localhost:8080 \
    --model "speakleash/Bielik-11B-v2.3-Instruct" \
    --dataset-path ./datasets.json \
    --accept-field questionText \
    --num-workers 4 \
    --batch-size 8
```

### Manual JSONL → XLSX Conversion

```python
from pathlib import Path
from llm_router_utils.core.jsonl_to_xlsx import convert_jsonl_to_xlsx

convert_jsonl_to_xlsx(
    jsonl_path=Path("output/dataset.jsonl"),
    xlsx_path=Path("output/dataset.xlsx"),
)
```

---

## Summary

Full pipeline:

```
1. Install venv + vLLM/llama.cpp/Ollama → local model
2. Install LLM Router → gateway with load balancing
3. Install llm-router-utils → CLI tools
4. Prepare prompts (*.prompt) in a directory
5. Download data (HF or XLSX)
6. Run genai-classifier
7. Verify results (JSONL / XLSX)
```
