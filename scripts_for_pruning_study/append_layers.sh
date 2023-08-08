#!/bin/bash

project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/llama-7b-odd+3

CUDA_VISIBLE_DEVICES='0' python scripts_for_pruning_study/append_layers.py \
    --model_name_or_path pinkmanlove/llama-7b-hf\
    --torch_dtype bfloat16 \
    --layers_to_append "30,28,26" \
    --output_model_path ${output_dir}
    