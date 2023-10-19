#!/bin/bash

deepspeed --master_port=12151 --include localhost:0,1,2,3,4,5,6,7 examples/evaluation.py \
    --model_name_or_path pinkmanlove/llama-13b-hf \
    --arch_type my_llama\
    --dataset_path /home/zhangyihan/LMFlow/data/red_test_mini \
    --deepspeed configs/ds_config_zero3_for_eval.json \
    --metric layer_importance \
    --evaluate_block_size 512
    --batch_size 16
