#!/bin/bash

CUDA_VISIBLE_DEVICES="7" \
    deepspeed --master_port=23212 examples/evaluation.py \
    --model_name_or_path pinkmanlove/llama-13b-hf \
    --dataset_path /home/zhangyihan/projects/LMFlow/data/wikitext-2-raw-v1/test \
    --deepspeed examples/ds_config.json \
    --metric layer_attention_importance