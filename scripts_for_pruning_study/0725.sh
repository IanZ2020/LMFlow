#!/bin/bash

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' /home/zhangyihan/projects/LMFlow/scripts/run_finetune.sh pinkmanlove/llama-13b-hf
CUDA_VISIBLE_DEVICES='7' /home/zhangyihan/projects/LMFlow/scripts/run_finetune_with_pruned_model.sh pinkmanlove/llama-7b-hf /home/zhangyihan/LLM-Pruner/prune_log/llama_13b_prune/pytorch_model.bin
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' /home/zhangyihan/projects/LMFlow/scripts/run_finetune.sh output_models/llama-13b-catcontribution-10
