# Ekosystem modeli Speakleash w LLM Router

Dokument zawiera opis konfiguracji modeli z rodziny
[Speakleash](https://huggingface.co/speakleash) w [LLM Router](https://llm-router.cloud/).
Podpięcie modelu
[Bielik-11B-v2.3-Instruct](https://huggingface.co/speakleash/Bielik-11B-v2.3-Instruct)
oraz
[Bielik-Guard-0.1B-v1.0](https://huggingface.co/speakleash/Bielik-Guard-0.1B-v1.0)
w połaczeniu z `fast_maskerem` z `llm-routera`, to kompletne rozwiązanie, które z powodzeniem można odpalić
jako brama GenAI z ochroną treści wysyłanych oraz maskowaniem danych wrażliwych.

---

## Konfiiguracja modeli Speakleash z rodziny Bielik

Pełen opis instalacji Bielika na lokalnym vLLM znajduje się pod adresem
[llm-router/examples](https://github.com/radlab-dev-group/llm-router/tree/main/examples/quickstart/speakleash-bielik-11b-v2_3-Instruct)

Ten katalog zawiera pełną konfigurację dla modeli Speakleash, zarówno Bielika jak i Bielika-Guard.
Oba modele można podłączyć do `llm-routera` i stworzyć kompletne środowiśko uruchomieniowe,
w którym jeden z modeli `Bielik-Guard` pilnuje treści, zaś `Bielik-Instruct` służy do generowania treści.

---

### Skrypty VLLM - fizyczne uruchomienie Bielika

**UWAGA!** W przykłądzie wykorzystany jest model `Bielik-11B-v2.3-Instruct`, który **nie wymaga** akceptacji
licencji podczas pobierania. Nowsze modele Speakleash wymagają zaakceptowania regulaminu
oraz pobrania modelu z wygenerowanym tokenem na [huggingface](https://huggingface.co/).

W tym katalogu znajdują się skrypty uruchomioniowe do VLLM, które ładują na GPU: `0, 1, 2`
model

- [run-bielik-11b-v2_3-vllm_0.sh](run-bielik-11b-v2_3-vllm_0.sh) --
  uruchamia model `speakleash/Bielik-11B-v2.3-Instruct` na `cuda:0`
- [run-bielik-11b-v2_3-vllm_1.sh](run-bielik-11b-v2_3-vllm_1.sh) --
  uruchamia model `speakleash/Bielik-11B-v2.3-Instruct` na `cuda:1`
- [run-bielik-11b-v2_3-vllm_2.sh](run-bielik-11b-v2_3-vllm_2.sh) --
  uruchamia model `speakleash/Bielik-11B-v2.3-Instruct` na `cuda:2`

Skrypty można uruchamiać na dowolnej liczbie maszyn, które następnie trzeba podpiąć do pliku konfiguracyjnego
[speakleash-models.json](configs/speakleash-models.json) w uruchomionej instancji `lm-router`.

---

### LM Router Serwis - Fizyczne uruchomienie Bielik-Guard

**UWAGA!** Bielik-Guard wymaga podania tokenu podczas pobierania modelu. W tym celu nalezy założyć konto
na [huggingface](https://huggingface.co/) i pobrać model z wygenerowanym tokenem.

W [run_sojka_guardrail.sh](./run-sojka-guardrail.sh) znajduje się skrypt uruchomieniowy
[LMM Router Services](https://github.com/radlab-dev-group/llm-router-services)
z wystawionym modelem
[Bielik-Guard-0.1B-v1.0](https://huggingface.co/speakleash/Bielik-Guard-0.1B-v1.0).

Aby `run_sojka_guardrail.sh` działało, trzeba zainstalować bibliotekę `llm-router-services`:

```bash
pip install git+https://github.com/radlab-dev-group/llm-router-services.git
```

Dowiązujemy pobrany model `speakleash/Bielik-Guard-0.1B-v1.0` do lokalizacji, gdzie uruchamiamy model
-- tam gdzie leży skrypt uruchamiający `run-sojka-guardrail.sh`. Jeżeli model jest ściągnięty
i znajduje się pod inną ścieżką, wystarczy utworzyć dowiązanie symboliczne:

```bash
mkdir ./speakleash
ln -s /mnt/data2/llms/models/community/speakleash/Bielik-Guard-0.1B-v1.0/ ./speakleash/
```

Wtedy w katalogu `speakleash/Bielik-Guard-0.1B-v1.0` powinny pokazać się pliki:

```bash
segfault:community $ ls -la speakleash/Bielik-Guard-0.1B-v1.0
razem 492356
drwxrwxr-x 2 pkedzia pkedzia      4096 lis 24 12:55 .
drwxrwxr-x 6 pkedzia pkedzia      4096 gru 13 12:37 ..
-rw-rw-r-- 1 pkedzia pkedzia       948 lis 24 12:53 config.json
-rw-rw-r-- 1 pkedzia pkedzia      1519 lis 24 12:53 gitattributes
-rw-rw-r-- 1 pkedzia pkedzia 497811044 lis 24 12:54 model.safetensors
-rw-rw-r-- 1 pkedzia pkedzia     15577 lis 24 12:53 README.md
-rw-rw-r-- 1 pkedzia pkedzia       964 lis 24 12:53 special_tokens_map.json
-rw-rw-r-- 1 pkedzia pkedzia      1468 lis 24 12:53 tokenizer_config.json
-rw-rw-r-- 1 pkedzia pkedzia   3355765 lis 24 12:53 tokenizer.json
-rw-rw-r-- 1 pkedzia pkedzia   2953979 lis 24 12:53 unigram.json
```

**UWAGA** `run-sojka-guardrail.sh` uruchamia dwa workery gunicorna z modelem ładowanym na `cuda:0`.

---

## Konfiguracja LLM Router

W plilku [speakleash-models.json](configs/speakleash-models.json) znajduje się konfiguracja, w której
Bielik uruchomiony jest na **8 dostawcach** w sieci lokalnmej (w tym przypadku vLLM, 1GPU == 1 dostawca).

Przykładowy wpis z konfiguracji:

```json
{
  "speakleash_models": {
    "speakleash/Bielik-11B-v2.3-Instruct": {
      "providers": [
        {
          "id": "bielik-11B_v2_3-vllm-local_71:7000",
          "api_host": "http://192.168.100.71:7000/",
          "api_token": "",
          "api_type": "vllm",
          "input_size": 32768,
          "model_path": "",
          "weight": 1.0,
          "keep_alive": ""
        },
        ...
      ]
    },
    ...
  },
  ...
}
```

Czyli na hoście `http://192.168.100.71` na porcie `7000` uruchomiony jest model `speakleash/Bielik-11B-v2.3-Instruct`
na dostawcy `vllm` z ustawionym `max_tokens=32768`. Całość znajduje się w sekcji `providers`, w której podawane
są namiary na wszystkich dostawców modelu. `llm-router` podczas działania balansuje obciążenie na tych właśnie
dostawców.

Unikalny identyfikator providera, np. `bielik-11B_v2_3-vllm-local_70:7002` to klucz, którym posługują się strategie
korzystające z Redisa. W llm-routerze założenie jest proste, jeden model może być dostępny u wielu dostawców,
dlagego równoważenie/balansowanie odbywa się w kontekście konkretnego modelu, a nie wszystkich modeli w konfiguracji.
Każdy model z konfiguracji traktowany jest jako osobny _byt_ do równoważenia obciążenia.

W przykładach model Bielika uruchomiony jest na 8 hostach:

- http://192.168.100.71:7000/ (vLLM na porcie 7000)
- http://192.168.100.71:7001/ (vLLM na porcie 7001)
- http://192.168.100.70:7000/ (vLLM na porcie 7000)
- http://192.168.100.70:7001/ (vLLM na porcie 7001)
- http://192.168.100.70:7002/ (vLLM na porcie 7002)
- http://192.168.100.66:7000/ (vLLM na porcie 7000)
- http://192.168.100.66:7001/ (vLLM na porcie 7001)
- http://192.168.100.66:7002/ (vLLM na porcie 7002)

---

## Uruchomienie LLM Routera

W pliku [run-rest-api-gunicorn](./run-rest-api-gunicorn.sh) znajduje się kompletna konfiguracja routera:

- konfiguracja modeli znajduje się w pliku `LLM_ROUTER_MODELS_CONFIG` (`resources/configs/models-config.json`)
  względem uruchomionego api LLM Routera z predefiniowanymi promptami w `resources/prompts`
  (`LLM_ROUTER_PROMPTS_DIR`) -- ścieżka również względem uruchomionego LLM Routera
- usługa dostępna jest na porcie `8080` (`LLM_ROUTER_SERVER_PORT`)
- odpalana za pomocą gunicorna (`LLM_ROUTER_SERVER_TYPE`) na 4 (`LLM_ROUTER_SERVER_WORKERS_COUNT`)
  workerach z 16 (`LLM_ROUTER_SERVER_THREADS_COUNT`) wątkami
- ze strategią `first_available` (`LLM_ROUTER_BALANCE_STRATEGY`) z połaczeniem do Redisa na hoście
  `192.168.100.67` (`LLM_ROUTER_REDIS_HOST`) i porcie `6379` (`LLM_ROUTER_REDIS_PORT`)
- z włączonym wymuszonym maskowaniem `LLM_ROUTER_FORCE_MASKING=1` i audytowaniem maskowania
  `LLM_ROUTER_MASKING_WITH_AUDIT=1` z wykorzystaniem jednoelementowego pipelinu: `[fast_masker]`
- z włączoną obsługą typu Guardrail `LLM_ROUTER_FORCE_GUARDRAIL_REQUEST=1` audytowaniem tych incydentów
  `LLM_ROUTER_GUARDRAIL_WITH_AUDIT_REQUEST=1` do zaszyfrowanych logów, w pipelinie guardrails wykorzystany
  jest plugin do połączenia z modelem Sójki `sojka_guard`, to jednoelementowy pipeline ustawiany za pomocą
  zmiennej `LLM_ROUTER_GUARDRAIL_STRATEGY_PIPELINE_REQUEST`
- Sójka dostępna jest na hoście definiowanym poprzez zmienna `LLM_ROUTER_GUARDRAIL_SOJKA_GUARD_HOST_EP`
  w przykładzie jest to `http://192.168.100.71:5001` czyli lokalny host

**UWAGA!** Redis jest wymagany do poprawnego działania strategii `first_available`! Jezeli nie posiadasz
Redisa i chciałbyś przetestować rozwiązanie, wystarczy zmienić strategię np. na `balanced`, która równomiernie
obiąża providerów. Wtedy w skrypcie uruchomioniowym [run-rest-api-gunicorn](./run-rest-api-gunicorn.sh)
wystarczy zmienić linię

```bash
export LLM_ROUTER_MASKING_STRATEGY_PIPELINE=${LLM_ROUTER_MASKING_STRATEGY_PIPELINE:-"fast_masker"}
```

na

```bash
export LLM_ROUTER_MASKING_STRATEGY_PIPELINE=${LLM_ROUTER_MASKING_STRATEGY_PIPELINE:-"balanced"}
```

---

**Udanego generowania i ochrony treści!** Niech Twoje treści nie wypływają z Twojej organizacji!
