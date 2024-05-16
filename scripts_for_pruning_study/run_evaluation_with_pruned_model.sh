#!/bin/bash
deepspeed_args="--master_port=13100"      # Default argument
# if [ $# -ge 1 ]; then
#   deepspeed_args="$1"
# fi

deepspeed --master_port=12151 --include localhost:0,1,2,3,4,5,6,7 examples/evaluation.py \
    --answer_type text \
    --model_name_or_path $1 \
    --arch_type  pruned_decoder_only\
    --dataset_path $2 \
    --deepspeed examples/ds_config.json \
    --metric ppl \
    --evaluate_block_size 1024 \
    --batch_size 4
