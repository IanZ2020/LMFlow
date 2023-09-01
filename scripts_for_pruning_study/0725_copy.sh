#!/bin/bash

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' /home/zhangyihan/projects/LMFlow/scripts/run_finetune.sh pinkmanlove/llama-13b-hf
# CUDA_VISIBLE_DEVICES='7' /home/zhangyihan/projects/LMFlow/scripts/run_finetune_with_pruned_model.sh /home/zhangyihan/projects/LMFlow/prune_log/llama_13b_prune
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' /home/zhangyihan/projects/LMFlow/scripts/run_finetune.sh output_models/llama-13b-catcontribution-10

CUDA_VISIBLE_DEVICES=0,1 bash /home/zhangyihan/projects/LMFlow/scripts_for_pruning_study/run_evaluation_with_pruned_model.sh "--master_port=13200" prune_log/llama_13b_prune_mlp_4-39_global_0.125 data/red_test_mini