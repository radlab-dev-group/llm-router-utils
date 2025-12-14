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
w którym jeden z modeli `Bielik-Guard` pilnuje treści, zaś `Bielik` służy do generowania treści.

---

### Skrypty VLLM - fizyczne uruchomienie Bielika

**UWAGA!** W przykłądzie wykorzystany jest model `Bielik-11B-v2.3-Instruct`, który nie wymaga akceptacji
licencji podczas pobierania. Nowsze modele Speakleash wymagają zaakceptowania regulaminu
oraz pobrania modelu z wygenerowanym tokenem na [huggingface](https://huggingface.co/).

W tym katalogu znajdują się skrypty uruchomioniowe do VLLM, które ładują na karty: `0, 1, 2`
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

### Konfiguracja llm-router

W plilku [speakleash-models.json](configs/speakleash-models.json) znajduje się konfiguracja, w której
Bielik uruchomiony jest na 8 dostawcach w sieci lokalnmej (w tym przypadku vLLM, 1GPU == 1 dostawca).

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
          "input_size": 56000,
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
na dostawcy `vllm` z ustawionym `max_tokens=56000`. Całość znajduje się w sekcji `providers`, w której podawane
są namiary na wszystkich dostawców modelu. `llm-router` podczas działania balansuje obciążenie na tych właśnie
dostawców.

W przykładach model Bielika uruchomiony jest na 8 hostach:
 - http://192.168.100.71:7000/ (vLLM na porcie 7000)
 - http://192.168.100.71:7001/ (vLLM na porcie 7001)
 - http://192.168.100.70:7000/ (vLLM na porcie 7000)
 - http://192.168.100.70:7001/ (vLLM na porcie 7001)
 - http://192.168.100.70:7002/ (vLLM na porcie 7002)
 - http://192.168.100.66:7000/ (vLLM na porcie 7000)
 - http://192.168.100.66:7001/ (vLLM na porcie 7001)
 - http://192.168.100.66:7002/ (vLLM na porcie 7002)


### Uruchamianie Bielik-Guard

