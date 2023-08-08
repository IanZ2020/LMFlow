#!/bin/bash

project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/llama-7b-mha

CUDA_VISIBLE_DEVICES='' python scripts_for_pruning_study/prune_mha.py \
    --model_name_or_path pinkmanlove/llama-7b-hf\
    --torch_dtype bfloat16 \
    --layers_to_be_pruned "{31:[0,1,2,3,4,5,6,7,8],30:[0,1,2,3,4,5,6,7,8],29:[0,1,2,3,4,5,6,7,8],28:[0,1,2,3,4,5,6,7,8]}" \
    --output_model_path ${output_dir}
    