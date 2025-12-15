# Speakleash Model Ecosystem in LLM Router

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

**NOTE** `run-sojka-guardrail.sh` launches five Gunicorn workers with the model loaded on `cuda:0`.
This means that the model is loaded five times per GPU; the minimum GPU memory required is 6 GB.

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

## Running the LLM Router

In the file [run-rest-api-gunicorn](./run-rest-api-gunicorn.sh) you’ll find the full router configuration:

- The model configuration is stored in `LLM_ROUTER_MODELS_CONFIG` (`resources/configs/models-config.json`) relative to
  the running LLM Router API, with predefined prompts in `resources/prompts` (`LLM_ROUTER_PROMPTS_DIR`) – the path is
  also relative to the running LLM Router.
- The service is available on port `8080` (`LLM_ROUTER_SERVER_PORT`).
- It is started with **gunicorn** (`LLM_ROUTER_SERVER_TYPE`) using **4** workers (`LLM_ROUTER_SERVER_WORKERS_COUNT`),
  each with **16** threads (`LLM_ROUTER_SERVER_THREADS_COUNT`).
- It uses the `first_available` balancing strategy (`LLM_ROUTER_BALANCE_STRATEGY`) and connects to Redis at host
  `192.168.100.67` (`LLM_ROUTER_REDIS_HOST`) on port `6379` (`LLM_ROUTER_REDIS_PORT`).
- Forced masking is enabled (`LLM_ROUTER_FORCE_MASKING=1`) together with masking audit (
  `LLM_ROUTER_MASKING_WITH_AUDIT=1`) using a single‑element pipeline: `[fast_masker]`.
- Guardrail support is turned on (`LLM_ROUTER_FORCE_GUARDRAIL_REQUEST=1`) with audit of those incidents (
  `LLM_ROUTER_GUARDRAIL_WITH_AUDIT_REQUEST=1`) written to encrypted logs. In the guardrail pipeline, the **sojka_guard**
  plugin (a connector to the Sójka model) is used; this is a single‑element pipeline set via the variable
  `LLM_ROUTER_GUARDRAIL_STRATEGY_PIPELINE_REQUEST`.
- The Sójka model is available on the host defined by the `LLM_ROUTER_GUARDRAIL_SOJKA_GUARD_HOST_EP` variable.
  In the example, it is set to http://192.168.100.71:5001, which corresponds to a local host.

**NOTE!** Redis is required for the `first_available` strategy to work correctly! If you don’t have Redis and still want
to test the solution, simply switch the strategy, e.g., to `balanced`, which distributes the load evenly across
providers. In that case, in the startup script [run-rest-api-gunicorn](./run-rest-api-gunicorn.sh) you only need to
change the line:

```bash
export LLM_ROUTER_MASKING_STRATEGY_PIPELINE=${LLM_ROUTER_MASKING_STRATEGY_PIPELINE:-"fast_masker"}
```

to

```bash
export LLM_ROUTER_MASKING_STRATEGY_PIPELINE=${LLM_ROUTER_MASKING_STRATEGY_PIPELINE:-"balanced"}
```

---

**Happy content generation and protection!** May your content stay securely within your organization!
