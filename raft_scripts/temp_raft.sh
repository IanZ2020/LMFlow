#!/bin/bash

# 定义要运行的次数
count=7


base_dir="raft_output/proxy_openllama_3b_relabel"
mkdir $base_dir
sft_model="raft_output/0715_relabel_sft_llama_7b_2e-5_1epoch"

x=0
y=1
model_dir="${base_dir}/model${x}"
tmp_model_dir="${base_dir}/model${y}"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" ./raft_scripts/run_finetune_raft.sh ${sft_model} $tmp_model_dir ${model_dir}/filtered_set

old_model_dir=$tmp_model_dir

for (( i=2; i<=$count; i++ )); do
  model_dir="${base_dir}/model${i}"
  CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" ./raft_scripts/run_finetune_raft.sh $old_model_dir $model_dir ${old_model_dir}/filtered_set
  old_model_dir=$model_dir
done