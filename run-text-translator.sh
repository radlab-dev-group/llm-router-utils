#!/bin/bash

LLM_ROUTER_HOST="http://192.168.100.65:8080"
MODEL="speakleash/Bielik-11B-v2.3-Instruct"
DATASET="/mnt/data2/data/datasets/huggingface/marmikpandya/mental-health/data.jsonl"

python3 -m llm_router_utils.cli.translate_texts \
  --llm-router-host=LLM_ROUTER_HOST \
  --model=MODEL \
  --dataset-type jsonl \
  --dataset-path=DATASET \
  --accept-field input \
  --accept-field output \
  --num-workers=10 \
  --batch-size=3
