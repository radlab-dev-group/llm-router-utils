#!/bin/bash

python3 -m llm_router_utils.cli.translate_texts \
  --llm-router-host="http://192.168.100.65:8080" \
  --model="speakleash/Bielik-11B-v2.3-Instruct" \
  --dataset-type jsonl \
  --dataset-path="/mnt/data2/data/datasets/huggingface/marmikpandya/mental-health/data.jsonl" \
  --accept-field input \
  --accept-field output \
  --num-workers=10 \
  --batch-size=3
