# bash /home/zhangyihan/LMFlow/scripts_for_pruning_study/run_finetune_with_pruned_model.sh "--launcher pdsh --master_addr=192.168.126.23 --hostfile hostfile3 --master_port=11001 --include 192.168.126.23:0,1,2,3,4,5,6,7@192.168.126.11:0,1,2,3,4,5,6,7" prune_log/llama-13b-mha-2-38-0.25-mlp-0.25-wikitext-512exm-1024--first data/redpajama_mini_formatted

# bash /home/zhangyihan/LMFlow/scripts_for_pruning_study/run_finetune_with_pruned_model.sh --master_port=11001 prune_log/llama-13b-mha-2-38-0.25-mlp-0.25-wikitext-512exm-1024--first data/red_plus

bash /home/zhangyihan/LMFlow/scripts_for_pruning_study/run_finetune_with_pruned_model_lora.sh --master_port=11001 prune_log/llama-13b-mha-2-38-0.25-mlp-0.25-wikitext-512exm-1024--first data/red_plus