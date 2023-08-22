#!/bin/bash

CUDA_VISIBLE_DEVICES=7 \
    deepspeed examples/evaluation.py \
    --answer_type text \
    --model_name_or_path $1 \
    --arch_type  pruned_decoder_only\
    --pruned_model $2 \
    --dataset_path data/wikitext-103-raw-v1/test \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric ppl
