#!/bin/bash

project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/llama-13b-last10merge

CUDA_VISIBLE_DEVICES='' python /home/zhangyihan/projects/LMFlow/scripts_for_pruning_study/layer_average.py \
    --model_name_or_path pinkmanlove/llama-13b-hf\
    --torch_dtype bfloat16 \
    --layers_to_merge "39,37,35,33,31,29,27,25,23,21" \
    --output_model_path ${output_dir}
    