#!/bin/bash

project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/llama-7b-odd-2-reinit

CUDA_VISIBLE_DEVICES='0' python scripts_for_pruning_study/reinit_layers.py \
    --model_name_or_path pinkmanlove/llama-7b-hf\
    --torch_dtype bfloat16 \
    --layers_to_be_pruned "30,28" \
    --output_model_path ${output_dir}
    