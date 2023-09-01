prune_ckpt_path='llama-13b-mha-0-40-0.25-8exm-paramfirst'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port=43477 examples/prune_model.py --model_name_or_path pinkmanlove/llama-13b-hf --pruning_ratio 0.25 --block_wise --device gpu --block_mlp_layer_start 0 --block_mlp_layer_end 0 --block_attention_layer_start 0 --block_attention_layer_end 40 --output_dir $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_first --save_model --global_pruning --grouping_strategy sum --num_examples 8 --prune_block_size 1024 --block_size 1024 --prune_batch_size 1 --torch_dtype float16 --use_flash_attention --deepspeed configs/ds_config_zero3_for_eval.json --dataset_path data/wikitext-2-raw-v1/test
#--gradient_checkpointing
echo "[FINISH] - Finish Pruning Model"

prune_ckpt_path='llama-13b-mha-0-40-0.25-1024exm-paramfirst'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port=43477 examples/prune_model.py --model_name_or_path pinkmanlove/llama-13b-hf --pruning_ratio 0.25 --block_wise --device gpu --block_mlp_layer_start 0 --block_mlp_layer_end 0 --block_attention_layer_start 0 --block_attention_layer_end 40 --output_dir $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_first --save_model --global_pruning --grouping_strategy sum --num_examples 1024 --prune_block_size 1024 --block_size 1024 --prune_batch_size 1 --torch_dtype float16 --use_flash_attention --deepspeed configs/ds_config_zero3_for_eval.json --dataset_path data/wikitext-2-raw-v1/test
#--gradient_checkpointing
echo "[FINISH] - Finish Pruning Model"