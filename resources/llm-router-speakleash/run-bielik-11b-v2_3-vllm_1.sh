#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

MODEL_PATH=speakleash/Bielik-11B-v2.3-Instruct

vllm serve \
	"${MODEL_PATH}" \
        --port 7001 \
        --host 0.0.0.0 \
        --quantization bitsandbytes \
        --load-format bitsandbytes \
        --max-model-len=32768 \
        --max_num_seqs=8 \
        --gpu-memory-utilization=0.90
