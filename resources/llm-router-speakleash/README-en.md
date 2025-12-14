**# Speakleash Model Ecosystem in LLM Router**

This document describes the configuration of models from the [Speakleash](https://huggingface.co/speakleash)
family in [LLM Router](https://llm-router.cloud/). Connecting the model
[Bielik-11B-v2.3-Instruct](https://huggingface.co/speakleash/Bielik-11B-v2.3-Instruct)
and
[Bielik-Guard-0.1B-v1.0](https://huggingface.co/speakleash/Bielik-Guard-0.1B-v1.0)
together with the `fast_masker` from `llm-router` gives a complete solution that can be run as a GenAI gateway
with content protection and sensitive‑data masking.

---

## Configuration of Speakleash Models – the Bielik Family

A full installation guide for Bielik on a local vLLM can be found at  
[llm-router/examples](https://github.com/radlab-dev-group/llm-router/tree/main/examples/quickstart/speakleash-bielik-11b-v2_3-Instruct).

This directory contains the complete configuration for Speakleash models, both Bielik and Bielik‑Guard.  
Both models can be attached to `llm-router` to create a full runtime environment where the `Bielik‑Guard`
model guards the content, while `Bielik-Instruct` generates it.

---

### VLLM Scripts – Physical Launch of Bielik

**NOTE!** The example uses the model `Bielik‑11B‑v2.3‑Instruct`, which does **not** require license acceptance
during download. Newer Speakleash models require you to accept the licence terms and download the model
with a generated token from [huggingface](https://huggingface.co/).

In this directory you will find launch scripts for VLLM that load the model onto GPUs `0, 1, 2`:

- [run-bielik-11b-v2_3-vllm_0.sh](run-bielik-11b-v2_3-vllm_0.sh) –
  runs `speakleash/Bielik-11B-v2.3-Instruct` on `cuda:0`
- [run-bielik-11b-v2_3-vllm_1.sh](run-bielik-11b-v2_3-vllm_1.sh) – runs the same model on `cuda:1`
- [run-bielik-11b-v2_3-vllm_2.sh](run-bielik-11b-v2_3-vllm_2.sh) – runs the same model on `cuda:2`

The scripts can be executed on any number of machines; those machines then need to be referenced in
the configuration file [speakleash-models.json](configs/speakleash-models.json)
of the running `lm-router` instance.

---

### LM Router Service – Physical Launch of Bielik‑Guard

**NOTE!** Bielik‑Guard requires a token when downloading the model. To obtain it, create an account
on [huggingface](https://huggingface.co/) and download the model with the generated token.

The script [run_sojka_guardrail.sh](./run-sojka-guardrail.sh) launches the
[LMM Router Services](https://github.com/radlab-dev-group/llm-router-services)
with the exposed model
[Bielik-Guard-0.1B-v1.0](https://huggingface.co/speakleash/Bielik-Guard-0.1B-v1.0).

To get `run_sojka_guardrail.sh` working, you need to install the `llm-router-services` library:

```shell script
pip install git+https://github.com/radlab-dev-group/llm-router-services.git
```

Link the downloaded `speakleash/Bielik-Guard-0.1B-v1.0` to the directory where you run
`run-sojka-guardrail.sh`. If the model is stored elsewhere, create a symbolic link:

```shell script
mkdir ./speakleash
ln -s /mnt/data2/llms/models/community/speakleash/Bielik-Guard-0.1B-v1.0/ ./speakleash/
```

Now the directory `speakleash/Bielik-Guard-0.1B-v1.0` should contain the following files:

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

**NOTE** `run-sojka-guardrail.sh` starts two Gunicorn workers with the model loaded on `cuda:0`.

---

## LLM Router Configuration

The file [speakleash-models.json](configs/speakleash-models.json) contains a configuration
where Bielik is deployed on **8 providers** in the local network (in this case vLLM, 1 GPU = 1 provider).

Example entry:

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

Thus, on host `http://192.168.100.71` port `7000` the model `speakleash/Bielik-11B-v2.3-Instruct` is running
on a vLLM provider with `max_tokens=56000`. All providers are listed under the `providers` section,
and `llm-router` balances the load across them.

The unique provider identifier (e.g., `bielik-11B_v2_3-vllm-local_70:7002`) is the key used by strategies that
rely on Redis. In `llm-router`, the assumption is simple: a single model can be available from many providers,
and load‑balancing is performed **per model**, not across all models in the configuration.
Each model entry is treated as an independent _entity_ for load balancing.

In the examples, the Bielik model is launched on 8 hosts:

- `http://192.168.100.71:7000/` (vLLM on port 7000)
- `http://192.168.100.71:7001/` (vLLM on port 7001)
- `http://192.168.100.70:7000/` (vLLM on port 7000)
- `http://192.168.100.70:7001/` (vLLM on port 7001)
- `http://192.168.100.70:7002/` (vLLM on port 7002)
- `http://192.168.100.66:7000/` (vLLM on port 7000)
- `http://192.168.100.66:7001/` (vLLM on port 7001)
- `http://192.168.100.66:7002/` (vLLM on port 7002)

---

**Happy content generation and protection!** May your content stay securely within your organization!
