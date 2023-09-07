#!/bin/bash
deepspeed_args="--master_port=13100"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

deepspeed ${deepspeed_args} examples/evaluation.py \
    --answer_type text \
    --model_name_or_path $2 \
    --arch_type  pruned_decoder_only\
    --dataset_path $3 \
    --deepspeed examples/ds_config.json \
    --metric ppl \
    --evaluate_block_size 1024 \
    --batch_size 4
