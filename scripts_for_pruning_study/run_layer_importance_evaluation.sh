#!/bin/bash

# deepspeed --master_port=12251 --include localhost:0,1,2,3,4,5,6,7 examples/evaluation.py \
#     --model_name_or_path pinkmanlove/llama-7b-hf \
#     --arch_type my_llama \
#     --dataset_path /home/zhangyihan/LMFlow/data/wikitext-2-raw-v1/test \
#     --deepspeed examples/ds_config.json \
#     --metric layer_importance \
#     --evaluate_block_size 1024 \
#     --batch_size 8 \
#     --torch_dtype bfloat16 \

#     deepspeed --master_port=12251 --include localhost:0,1,2,3,4,5,6,7 examples/evaluation.py \
#     --model_name_or_path pinkmanlove/llama-7b-hf \
#     --arch_type my_llama \
#     --dataset_path /home/zhangyihan/LMFlow/data/red_test_mini \
#     --deepspeed examples/ds_config.json \
#     --metric layer_importance \
#     --evaluate_block_size 1024 \
#     --batch_size 8 \
#     --torch_dtype bfloat16 \

    # deepspeed --master_port=12251 --include localhost:0,1,2,3,4,5,6,7 examples/evaluation.py \
    # --model_name_or_path pinkmanlove/llama-13b-hf \
    # --arch_type my_llama \
    # --dataset_path /home/zhangyihan/LMFlow/data/wikitext-2-raw-v1/test \
    # --deepspeed examples/ds_config.json \
    # --metric layer_importance \
    # --evaluate_block_size 1024 \
    # --batch_size 2 \
    # --torch_dtype bfloat16 \

    deepspeed --master_port=12251 --include localhost:0,1,2,3,4,5,6,7 examples/evaluation.py \
    --model_name_or_path princeton-nlp/Sheared-LLaMA-2.7B \
    --arch_type my_llama \
    --dataset_path /home/zhangyihan/LMFlow/data/red_test_mini \
    --deepspeed examples/ds_config.json \
    --metric layer_importance \
    --evaluate_block_size 1024 \
    --batch_size 2 \
    --torch_dtype bfloat16 \
