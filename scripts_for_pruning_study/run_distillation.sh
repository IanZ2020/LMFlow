#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

deepspeed_args="--master_port=13100"      # Default argument
# if [ $# -ge 1 ]; then
#   deepspeed_args="$1"
# fi

exp_id=distill_with_klt2.0_klw0.5_msew_0.5_redpajama_$1
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/distill.py \
    --arch_type  pruned_decoder_only\
    --model_name_or_path $1 \
    --ref_model pinkmanlove/llama-13b-hf \
    --kl_t 2.0 \
    --kl_w 0.5 \
    --mse_w 0.5 \
    --hard_w 1.0 \
    --dataset_path data/redpajama_mini_formatted \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --block_size 2048 \
    --per_device_train_batch_size 4 \
    --deepspeed configs/ds_config_zero3.json \
    --fp16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 300 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 4\
    --dataloader_num_workers 1 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err