# prune_ckpt_path='llama_13b_prune_local_even_0.2'
# tune_ckpt_path='llama_0.2'
# echo "[START] - Start Pruning Model"
# CUDA_VISIBLE_DEVICES=0 python scripts_for_pruning_study/hf_prune.py --base_model pinkmanlove/llama-13b-hf --pruning_ratio 0.20 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 40 --block_attention_layer_start 0 --block_attention_layer_end 40 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_mix --save_model
# echo "[FINISH] - Finish Pruning Model"

# prune_ckpt_path='llama_13b_prune_global_attn_only_0.8'
# tune_ckpt_path='llama_0.2'
# echo "[START] - Start Pruning Model"
# CUDA_VISIBLE_DEVICES=1 python scripts_for_pruning_study/hf_prune.py --base_model pinkmanlove/llama-13b-hf --pruning_ratio 0.80 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 40 --block_attention_layer_start 0 --block_attention_layer_end 40 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_mix --save_model --global_pruning
# echo "[FINISH] - Finish Pruning Model"

prune_ckpt_path='llama_13b_prune_global_0-40_0.2_NumWeight_edgrouping_strategy_mean'
tune_ckpt_path='llama_0.2'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=0 python scripts_for_pruning_study/hf_prune.py --base_model pinkmanlove/llama-13b-hf --pruning_ratio 0.2 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 40 --block_attention_layer_start 0 --block_attention_layer_end 40 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_mix --save_model --global_pruning --grouping_strategy mean
echo "[FINISH] - Finish Pruning Model"

# echo "[START] - Start Tuning"
# CUDA_VISIBLE_DEVICES=3 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
# echo "[FINISH] - Finish Prune and Post-Training."
# echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

# echo "You can use the command:"
# echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
# echo "to use the pruned model"
