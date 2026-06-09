# Przewodnik: Generowanie datasetu z wykorzystaniem LLM Router i GenAI Classifier

## 📋 Spis treści

1. [Wstęp](#1-wstęp)
2. [Architektura rozwiązania](#2-architektura-rozwiązania)
3. [Instalacja środowiska lokalnego](#3-instalacja-środowiska-lokalnego)
    - [vLLM](#31-vllm)
    - [llama.cpp](#32-llamacpp)
    - [Ollama](#33-ollama)
4. [Konfiguracja LLM Router](#4-konfiguracja-llm-router)
5. [Utility genai-classifier](#5-utility-genai-classifier)
6. [Kompletny przykład end-to-end](#6-kompletny-przykład-end-to-end)
7. [Format wyjściowy](#7-format-wyjściowy)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Wstęp

Ten dokument opisuje pełny proces generowania datasetu do uczenia modeli NLP z wykorzystaniem:

- **LLM Router** — brama API, która równoważy obciążenie między wieloma dostawcami LLM (vLLM, llama.cpp, Ollama)
- **genai-classifier** — utility z pakietu `llm-router-utils`, które klasyfikuje teksty za pomocą LLM według zadanych
  promptów
- **Translate-texts** — utility do batchowego tłumaczenia tekstów

**Typowe zastosowania**:

- Augmentacja danych (klasyfikacja, ekstrakcja atrybutów, kategoryzacja)
- Przygotowanie danych treningowych do fine-tuningu
- Ewaluacja treści pod kątem obecności konkretnych klas (np. detekcja mowy nienawistnej, kategorii tematycznych)
- Batch translation i transliteracja

---

## 2. Architektura rozwiązania

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Data Flow                                      │
│                                                                     │
│  HF Dataset / XLSX  →  genai-classifier  →  JSONL output          │
│       (źródło)              (LLM Router)        (wynik)            │
│                             │                                         │
│                      LLMRouterClient                                │
│                             │                                         │
│              ┌──────────────┼──────────────┐                        │
│              │              │              │                         │
│        ┌─────▼─────┐  ┌────▼────┐  ┌─────▼─────┐                  │
│        │ vLLM      │  │llama.cpp│  │  Ollama   │                  │
│        │ provider  │  │provider │  │  provider  │                  │
│        └───────────┘  └─────────┘  └───────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Komponenty

| Komponent                                   | Rola                                                                            |
|---------------------------------------------|---------------------------------------------------------------------------------|
| **`llm_router_lib.client.LLMRouterClient`** | Klient HTTP — wysyła żądania do routera, obsługuje retry i błędy                |
| **`llm_router_api.rest_api`**               | Brama REST — balansıje obciążenie, proxy do backendów (vLLM, Ollama, llama.cpp) |
| **`GenAIClassifierApp`**                    | Orkiestrator — ładuje dane, wysyła do LLM, agreguje wyniki w JSONL              |
| **`PromptHandler`** (z `rdl_ml_utils`)      | Ładuje prompty z plików `.prompt` w katalogu `prompts_dir`                      |
| **`HfDatasetHandler`**                      | Pobiera i ładuje dane z HuggingFace Hub                                         |
| **`convert_jsonl_to_xlsx`**                 | Konwertuje JSONL na Excel z formatowaniem                                       |

### Pipeline klasyfikacji

```
1. Ładowanie danych (HF Dataset / XLSX)
   ↓
2. Ekstrakcja pól do sklasyfikowania
   ↓
3. Dla każdego (tekst, pole):
   a. Pobranie promptu z prompts_dir
   b. Wysyłka do LLM Router → LLM → odpowiedź JSON
   c. Parsowanie odpowiedzi (exists, confidence, reason)
   ↓
4. Agregacja wyników do JSONL
   ↓
5. Konwersja do XLSX (opcjonalnie)
```

---

## 3. Instalacja środowiska lokalnego

### 3.1 vLLM

vLLM to engine do generacji low-latency i high-throughput dla dużych modeli językowych.

#### Wymagania

- Ubuntu 20.04+
- CUDA 11.8+
- GPU z min. 12 GB VRAM (40 GB+ rekomendowane dla modeli >7B)
- Python 3.10+

#### Krok 1: Instalacja venv i vLLM

```bash
# Utworzenie katalogu roboczego
mkdir -p ~/llm-classifier && cd ~/llm-classifier

# Wirtualne środowisko
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Instalacja vLLM (dla CUDA 11.8)
pip install vllm

# Weryfikacja instalacji
python -c "import vllm; print(vllm.__version__)"
```

#### Krok 2: Pobranie modelu

```bash
# Przykład: Bielik 11B (nie wymaga akceptacji licencji)
from huggingface_hub import snapshot_download
model_path = snapshot_download(
    repo_id="speakleash/Bielik-11B-v2.3-Instruct",
    local_dir="./models/Bielik-11B-v2.3-Instruct"
)

# Przykład: Bielik 7B (wymaga tokena HF)
import os
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxxxx"
from huggingface_hub import snapshot_download
model_path = snapshot_download(
    repo_id="speakleash/Bielik-7B-Instruct",
    local_dir="./models/Bielik-7B-Instruct",
    token=os.environ["HF_TOKEN"]
)
```

#### Krok 3: Uruchomienie serwera vLLM

```bash
# Skrypt startowy — launch-vllm.sh
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
# Uruchom w tmux (serwer działa w tle)
tmux new -s vllm './launch-vllm.sh'
```

> **Wskazówka:** Serwer vLLM nasłuchuje na `http://0.0.0.0:7000/v1/chat/completions`.
> Test:
`curl http://localhost:7000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"speakleash/Bielik-11B-v2.3-Instruct","messages":[{"role":"user","content":"Test"}],"max_tokens":10}'`

---

### 3.2 llama.cpp

llama.cpp to engine C++ z bind Python, zoptymalizowany pod CPU i GPU.

#### Wymagania

- Ubuntu 20.04+
- gcc/g++ (min. wersja 7)
- Python 3.10+
- Opcjonalnie: CUDA dla akceleracji GPU

#### Krok 1: Instalacja

```bash
mkdir -p ~/llm-classifier && cd ~/llm-classifier

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Klonowanie i budowa llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake ..
make -j$(nproc)  # kompilacja z wszystkimi rdzeniami

# Budowa server (server.cpp)
make server -j$(nproc)

# Weryfikacja
./server --version
```

#### Krok 2: Pobranie i konwersja modelu

```bash
cd ~/llm-classifier
# Pobranie modelu GGUF (już skwantizowanego)
huggingface-cli download speakleash/Bielik-11B-v2.3-Instruct \
    --include "*.gguf" \
    --local-dir ./models/

# Lub pobranie i konwersja z formatu safetensors → GGUF
python3 ../llama.cpp/convert.py ../models/Bielik-11B-v2.3-Instruct/ \
    --outtype f16 \
    --outfile ./models/Bielik-11B-v2.3-Instruct-q8_0.gguf
```

#### Krok 3: Konwersja do formatu KV-cache (GGUF Q4_K_M)

```bash
# Rekomendowana kwantyzacja dla dobrej balansu jakości/czasu
python3 ../llama.cpp/quantize ../models/Bielik-11B-v2.3-Instruct-q8_0.gguf \
    ../models/Bielik-11B-v2.3-Instruct-q4_k_m.gguf \
    Q4_K_M
```

#### Krok 4: Uruchomienie serwera llama.cpp

```bash
# launch-llama.sh — start serwera llama.cpp
#!/bin/bash
cd ~/llm-classifier/llama.cpp/build

./server \
    -m ../models/Bielik-11B-v2.3-Instruct-q4_k_m.gguf \
    --port 7001 \
    --host 0.0.0.0 \
    -ngl 35  # liczba warstw do przeniesienia na GPU (0 = CPU only)
    --ctx-size 32768 \
    --n-predict 1024
```

```bash
chmod +x launch-llama.sh
tmux new -s llama './launch-llama.sh'
```

> **Wskazówka:** Serwer llama.cpp udostępnia endpoint w formacie OpenAI-compatible na
`http://localhost:7001/v1/chat/completions`.

---

### 3.3 Ollama

Ollama to prosty w obsłudze runner modeli LLM, idealny do testów i prototypowania.

#### Wymagania

- Ubuntu 20.04+
- Python 3.10+

#### Krok 1: Instalacja Ollama

```bash
# Instalacja Ollama (oficjalny skrypt instalacyjny)
curl -fsSL https://ollama.com/install.sh | sh

# Start usługi
systemctl start ollama
systemctl enable ollama

# Weryfikacja
ollama --version
```

#### Krok 2: Pobranie modelu

```bash
# Pobranie modelu Bielik
ollama pull speakleash/bielik-11b-v2.3-instruct

# Lub dla mniejszego modelu (szybszy)
ollama pull speakleash/bielik-7b-instruct
```

#### Krok 3: Weryfikacja

```bash
# Lista pobranych modeli
ollama list

# Interaktywne testowanie
ollama run speakleash/bielik-11b-v2.3-instruct "Cześć, jak się masz?"
```

> **Wskazówka:** Ollama automatycznie wystawia API na `http://localhost:11434`.
> Endpoint: `http://localhost:11434/v1/chat/completions` w formacie OpenAI-compatible.

---

## 4. Konfiguracja LLM Router

LLM Router działa jako brama, która równoważy obciążenie między modelami uruchomionymi na vLLM, llama.cpp i Ollama.

### 4.1 Instalacja Router-a

```bash
cd ~/llm-classifier

python3 -m venv .venv
source .venv/bin/activate

# Instalacja routera i bibliotek
git clone https://github.com/radlab-dev-group/llm-router.git
cd llm-router
pip install .[api]

# Wymagane zależności
pip install redis

# Weryfikacja
python3 -m llm_router_api.rest_api --help
```

### 4.2 Plik konfiguracyjny `models-config.json`

Przykładowy konfiguracja z wieloma providerami:

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

### 4.3 Uruchomienie Router-a

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

> **Weryfikacja:** `curl http://localhost:8080/ping` powinien zwrócić `"pong"`.
> Sprawdzić dostępne modele: `curl http://localhost:8080/models`

---

## 5. Utility genai-classifier

### 5.1 Instalacja

```bash
cd ~/llm-classifier

# Instalacja z git
pip install git+https://github.com/radlab-dev-group/llm-router-utils.git

# Lub instalacja lokalna
git clone https://github.com/radlab-dev-group/llm-router-utils.git
cd llm-router-utils
pip install .[llm-router]
```

### 5.2 Argumenty CLI

```bash
genai-classifier --help
```

| Argument            | Typ     | Domyślna wartość             | Opis                                                        |
|---------------------|---------|------------------------------|-------------------------------------------------------------|
| `--dataset-dir`     | `Path`  | **(wymagany)**               | Katalog ze sczytanymi danymi HF lub XLSX                    |
| `--prompts-dir`     | `Path`  | **(wymagany)**               | Katalog z plikami promptów `.prompt`                        |
| `--llm-router-url`  | `str`   | `http://192.168.100.65:8080` | URL LLM Router-a                                            |
| `--model-name`      | `str`   | `gpt-oss:120b`               | Nazwa modelu do użycia (przesyłana do routera)              |
| `--temperature`     | `float` | `0.0`                        | Temperatura generacji (0 = deterministyczna, 1 = kreatywna) |
| `--num-workers`     | `int`   | `2`                          | Liczba wątków równoległych (i klientow LLM)                 |
| `--batch-save-size` | `int`   | `5`                          | Ilość rekordów zapisywanych na dysk naraz                   |
| `--n-sample`        | `int`   | `50`                         | Liczba losowych próbek na pole (0 = wszystko)               |
| `--dry-run`         | `flag`  | `false`                      | Przetwarzaj bez zapisywania plików                          |
| `--output-dir`      | `Path`  | *(dataset-dir)*              | Katalog wyjściowy                                           |
| `--verbose`         | `flag`  | `false`                      | DEBUG logging                                               |
| `--export-xlsx`     | `flag`  | `true`                       | Eksportuj wyniki do Excela                                  |
| `--no-export-xlsx`  | `flag`  | `false`                      | Wyłącz eksport XLSX                                         |

### 5.3 Przygotowanie promptów

Prompty to pliki tekstowe z rozszerzeniem `.prompt`, gdzie nazwa pliku to **nazwa klasy**.

Przykładowa struktura katalogu:

```
prompts/
├── mowa_nienawistna.prompt
├── sugerowanie_siebie.prompt
├── pytanie.prompt
└── odpowiedź.prompt
```

Plik `mowa_nienawistna.prompt`:

```
Jesteś analitykiem treści. Sprawdź czy poniższy tekst zawiera mowę nienawistną.

Format wyjścia (JSON):
{
  "exists": true|false,
  "confidence": 0.0-1.0,
  "reason": "Krótka przyczyna decyzji"
}

Zachowanie:
- Zwróć JEDNO poprawne JSON-owe wyjście
- Nie dodawaj żadnego tekstu przed lub po JSON
```

### 5.4 Przygotowanie danych

#### Z HuggingFace

```python
from llm_router_utils.core.hf_dataset_handler import HfDatasetHandler
from pathlib import Path

# Pobieranie datasetu z HF
data_dir = Path("./datasets")
HfDatasetHandler.download_and_save_dataset(
    dataset_id="nbertagnolli/counsel-chat",
    data_dir=data_dir,
)
```

#### Z XLSX

```python
import pandas as pd

# Dane z pliku XLSX są automatycznie ładowane przez GenAIClassifierApp
# Zapisz plik w dataset_dir — zostanie wykryty automatycznie
```

---

## 6. Kompletny przykład end-to-end

### Krok 1: Przygotowanie środowiska (vLLM)

```bash
# 1. Środowisko
mkdir -p ~/llm-classifier && cd ~/llm-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 2. vLLM + biblioteki routera
pip install vllm
pip install git+https://github.com/radlab-dev-group/llm-router.git
pip install git+https://github.com/radlab-dev-group/llm-router-utils.git
pip install redis

# 3. Pobranie modelu
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('speakleash/Bielik-11B-v2.3-Instruct', local_dir='./models/Bielik')
"
```

### Krok 2: Uruchomienie komponentów

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

# Terminal 2 — LLM Router (w osobnym venv)
cd ~/llm-classifier/llm-router
export LLM_ROUTER_MODELS_CONFIG="$PWD/resources/configs/models-config.json"
export LLM_ROUTER_PROMPTS_DIR="$PWD/resources/prompts"
export LLM_ROUTER_SERVER_PORT="8080"
export LLM_ROUTER_BALANCE_STRATEGY="balanced"
./run-rest-api.sh

# Terminal 3 — Redis (jeśli używasz first_available)
redis-server
```

### Krok 3: Przygotowanie promptów i danych

```bash
# 1. Utworzenie katalogu z promptami
mkdir -p ~/llm-classifier/prompts

# Przykładowy prompt — detekcja kategorii tematycznej
cat > ~/llm-classifier/prompts/tematyka_ medyczna.prompt << 'EOF'
Jesteś analitykiem tekstów medycznych. Określ czy tekst należy do kategorii "medycyna".

Format wyjścia (JSON):
{
  "exists": true|false,
  "confidence": 0.0-1.0,
  "reason": "Krótka przyczyna decyzji"
}
EOF

# Przykładowy prompt — detekcja mowy nienawistnej
cat > ~/llm-classifier/prompts/mowa_nienawistna.prompt << 'EOF'
Sprawdź czy tekst zawiera mowę nienawistną.

Format wyjścia (JSON):
{
  "exists": true|false,
  "confidence": 0.0-1.0,
  "reason": "Krótka przyczyna decyzji"
}
EOF

# 2. Przygotowanie danych — pobranie z HF
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

### Krok 4: Uruchomienie genai-classifier

```bash
# Klasyfikacja — przetworzenie 100 próbek z danego pola
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

# Klasyfikacja WSZYSTKICH danych (bez limitu n-sample)
genai-classifier \
    --dataset-dir ~/llm-classifier/datasets \
    --prompts-dir ~/llm-classifier/prompts \
    --llm-router-url http://localhost:8080 \
    --model-name "speakleash/Bielik-11B-v2.3-Instruct" \
    --temperature 0.0 \
    --num-workers 4 \
    --n-sample 0 \
    --output-dir ~/llm-classifier/output \
    --dry-run  # test bez zapisywania
```

### Krok 5: Sprawdzanie wyników

```bash
# Podgląd JSONL (pierwsze 5 wierszy)
head -5 ~/llm-classifier/output/usham__mental-health-companion-new.jsonl

# Pełna konwersja do Excela (domyślna, dzieje się automatycznie)
ls ~/llm-classifier/output/usham__mental-health-companion-new.xlsx
```

---

## 7. Format wyjściowy

### 7.1 JSONL (JSON Lines)

Każdy wiersz to jeden rekord z klasami dla danego pola:

```json
{
  "text": "Czy mogę przyjąć aspirynę w ciąży?",
  "field": "input",
  "features": [
    {
      "name": "tematyka_medyczna",
      "response": {
        "exists": true,
        "confidence": 0.95,
        "reason": "Tekst zawiera termin medyczny 'aspiryna' i kontekst 'ciąża'."
      }
    },
    {
      "name": "mowa_nienawistna",
      "response": {
        "exists": false,
        "confidence": 0.99,
        "reason": "Tekst jest pytaniem medycznym, nie zawiera treści obraźliwych."
      }
    }
  ]
}
```

### 7.2 Struktura katalogu wyjściowego

```
output/
├── dataset_name__field1__field2.jsonl
├── dataset_name__field1__field2.xlsx       ← Excel z formatowaniem
└── ...
```

### 7.3 Excel (XLSX)

Każda karta arkusza odpowiada jednemu polu (`field`). Kolumny:

| Kolumna                      | Opis                                           |
|------------------------------|------------------------------------------------|
| `text`                       | Tekst źródłowy (szerokość kolumny ≈ 55 znaków) |
| `<nazwa_promptu>`            | Nazwa klasy (zawsze prawdziwa)                 |
| `<nazwa_promptu>-class`      | Klasy (jeśli dostępna)                         |
| `<nazwa_promptu>-confidence` | Pewność (0.0–1.0)                              |
| `<nazwa_promptu>-exists`     | Czy klasa występuje (1/True, 0/False)          |
| `<nazwa_promptu>-reason`     | Przyczyna decyzji modelu                       |

**Formatowanie**:

- Nagłówki: pogrubione, wyśrodkowane, niebieskie tło
- Wiersze parzyste: szare tło (zebra striping)
- Wiersze z `exists=1`: zielone tło
- Wiersze z `exists=0`: niebieskie tło
- Tekst: zawijanie (wrap)

---

## 8. Troubleshooting

### Problem: `RuntimeError: Redis is mandatory`

**Rozwiązanie:**

```bash
# Zainstaluj Redis
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server

# Lub zmień strategię balansıowania na "balanced" (nie wymaga Redisa)
export LLM_ROUTER_BALANCE_STRATEGY="balanced"
```

### Problem: `ValueError: Unknown model 'xyz'`

**Rozwiązanie:** Upewnij się, że `--model-name` zgadza się z modelem dostępnym u providerów w konfiguracji. Sprawdź w
`models-config.json`.

### Problem: Ollama zwraca `error: model not found`

**Rozwiązanie:**

```bash
# Sprawdź dostępne modele
ollama list

# Sprawdź czy model jest pobrany
ollama pull speakleash/bielik-11b-v2.3-instruct

# Sprawdź czy usługa działa
curl http://localhost:11434/api/tags
```

### Problem: vLLM o niskim VRAM

**Rozwiązanie:** Zmniejsz `--gpu-memory-utilization` lub zwiększ `--quantization`:

```bash
vllm serve ... \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --gpu-memory-utilization=0.7
```

### Problem: `genai-classifier` timeoutuje

**Rozwiązanie:** Zmniejsz `--num-workers` lub zwiększ timeout routera:

```bash
export LLM_ROUTER_TIMEOUT=120
export LLM_ROUTER_EXTERNAL_TIMEOUT=300
```

### Problem: Niepoprawny JSON z LLM

**Rozwiązanie:** genai-classifier automatycznie retryuje (max 5+3=8 prób). Jeśli problem się powtarza:

- Sprawdź czy prompt zawiera instrukcję "Zwróć JEDNO poprawne JSON-owe wyjście"
- Ustaw `--temperature 0.0` dla deterministycznej odpowiedzi
- Sprawdź czy w pliku `.prompt` jest sekcja `Format wyjścia (JSON)`

### Problem: `ImportError: No module named 'rdl_ml_utils'`

**Rozwiązanie:** PromptHandler pochodzi z pakietu `rdl_ml_utils`:

```bash
pip install rdl-ml-utils
```

### Problem: Błędy CUDA

**Rozwiązanie:**

```bash
# Sprawdź czy CUDA jest widoczne
nvidia-smi

# Sprawdź wersję CUDA w Python
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Jeśli vLLM nie wykrywa GPU — zainstaluj z odpowiednim tagiem CUDA
pip uninstall vllm -y
pip install vllm --no-cache-dir
```

---

## Przydatne polecenia

### Batch translation (translate-texts)

```bash
translate-texts \
    --llm-router-host http://localhost:8080 \
    --model "speakleash/Bielik-11B-v2.3-Instruct" \
    --dataset-path ./datasets.json \
    --accept-field questionText \
    --num-workers 4 \
    --batch-size 8
```

### Pełna konwersja JSONL → XLSX (ręczna)

```python
from pathlib import Path
from llm_router_utils.core.jsonl_to_xlsx import convert_jsonl_to_xlsx

convert_jsonl_to_xlsx(
    jsonl_path=Path("output/dataset.jsonl"),
    xlsx_path=Path("output/dataset.xlsx"),
)
```

---

## Podsumowanie

Pełny pipeline:

```
1. Instalacja venv + vLLM/llama.cpp/Ollama → model lokalny
2. Instalacja LLM Router → brama z load balancingiem
3. Instalacja llm-router-utils → CLI tools
4. Przygotowanie prompty (*.prompt) w katalogu
5. Pobranie danych (HF lub XLSX)
6. Uruchomienie genai-classifier
7. Weryfikacja wyników (JSONL / XLSX)
```
