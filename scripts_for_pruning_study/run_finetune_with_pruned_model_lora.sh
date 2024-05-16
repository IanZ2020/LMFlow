#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

deepspeed_args="--master_port=13100"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

exp_id=pruning_study_pretraining_wiki_lr3e-4/$2
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --arch_type  pruned_decoder_only\
    --model_name_or_path $2 \
    --dataset_path $3 \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --block_size 2048 \
    --per_device_train_batch_size 16 \
    --deepspeed configs/ds_config_zero3.json \
    --fp16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 100 \
    --dataloader_num_workers 1 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --gradient_accumulation_steps 4 \
    --use_flash_attention True\
    --use_lora 1 \
    --lora_r 64 \
    --save_aggregated_lora 1\
    --lora_target_modules q_proj v_proj k_proj o_proj up_proj gate_proj down_proj \
    --lora_dropout 0 \
    --lora_alpha 100 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err