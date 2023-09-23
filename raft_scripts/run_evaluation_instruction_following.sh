#!/bin/bash
# bash scripts/llama3B/evaluation_instruct/run_evaluation.sh
# exp_id=openllama3b_finetune_medmcqa_ep5
exp_id=proxy_rm_raft
if [ $# -ge 1 ]; then
  exp_id="$1"
fi

model_path=4000
if [ $# -ge 2 ]; then
  model_path="$2"
fi

gpu_idx=2
if [ $# -ge 3 ]; then
  gpu_idx="$3"
fi

master_port=11001
if [ $# -ge 4 ]; then
  master_port="$4"
fi

# exp_id=openllama3b_finetune_medmcqa_ep5
project_dir=/home/linhangyu/Projects/LLMs/LMFlow
output_dir=${project_dir}/output_models/${exp_id}/${ckpt_tag}

# MedQA-USMLE MedMCQA
task_names=("alpaca" "gpt4_instruction_en_eval" "gpt4_instruction_zh_eval" "lmflow_en_eval" "lmflow_cn_eval")
eval_data_paths=(
  "data/alpaca/test" 
  "data/gpt4_instruction_en_eval/"
  "data/gpt4_instruction_zh_eval/"
  "data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/"
  "data/lmflow_chat_cn_dialog_multiturn_single_nll_text2text/"
)
answer_types=("text_only" "text2text" "text2text" "text2text" "text2text")
prompts=(
  "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### {input}\n\n### Response:" 
  "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### {input}\n\n### Response:" 
  "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### {input}\n\n### Response:" 
  "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### {input}\n\n### Response:" 
  "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### {input}\n\n### Response:" 
)


for i in "${!task_names[@]}" 
do
    task_name=${task_names[i]}
    answer_type=${answer_types[i]}
    eval_data_path=${eval_data_paths[i]}
    prompt=${prompts[i]}
    log_dir=${project_dir}/eval_log/${exp_id}/${task_name}
    if [ ! -d "${log_dir}" ];
    then 
        mkdir -p ${log_dir}
    fi

    deepspeed_args="--master_port=${master_port} --include localhost:${gpu_idx} "
    echo ${deepspeed_args}
    # --model_name_or_path ${output_dir} \
    deepspeed ${deepspeed_args} \
        examples/evaluation.py \
        --answer_type=${answer_type} \
        --model_name_or_path=${model_path} \
        --prompt_structure="\"${prompt}\"" \
        --dataset_path="${eval_data_path}" \
        --deepspeed="examples/ds_config.json" \
        --inference_batch_size_per_device=64 \
        --metric="nll" \
        | tee ${log_dir}/evaluation_${ckpt_idx}.log \
        2> ${log_dir}/evaluation_${ckpt_idx}.err
done