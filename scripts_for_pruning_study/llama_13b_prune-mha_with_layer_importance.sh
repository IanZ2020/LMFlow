# prune_ckpt_path='13b-mha/llama-13b-mha-2-38-0.5-8192Redpajama-local_layer_imp-second-exp_t0.1'
# echo "[START] - Start Pruning Model"
# deepspeed --master_port=43475 --include localhost:0,1,2,3,4,5,6,7 examples/prune_with_layer_importance.py --model_name_or_path /home/zhangyihan/.cache/huggingface/hub/models--pinkmanlove--llama-13b-hf/snapshots/e7f2b4560f73910c9c6cea51cd7bd132d4745882 --pruning_ratio 0.5 --block_wise --device gpu --block_mlp_layer_start 2 --block_mlp_layer_end 2 --block_attention_layer_start 2 --block_attention_layer_end 38 --output_dir $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_second --save_model --grouping_strategy sum --num_examples 8192 --prune_block_size 1024 --block_size 1024 --prune_batch_size 1 --torch_dtype float16 --use_flash_attention --deepspeed configs/ds_config_zero3_for_eval.json --dataset_path data/wikitext-103-raw-v1/test --pruning_dataset wikitext_cat --layer_importance "3.7188,0.8945,0.7773,0.6484,0.625,0.5859,0.6484,0.5547,0.5273,0.5,0.4922,0.4883,0.4844,0.4746,0.4492,0.4414,0.3867,0.3926,0.3848,0.3672,0.3477,0.3438,0.334,0.3203,0.2812,0.2754,0.2441,0.2402,0.2344,0.2041,0.2109,0.2031,0.1875,0.1963,0.2051,0.2031,0.1963,0.2344,0.3223,0.8164" --layer_importance_weighting_type exp --grad_info_path /home/zhangyihan/LMFlow/grad_info/13b_8192red_second_grad_info.bin --exp_t 0.1
# echo "[FINISH] - Finish Pruning Model"

# prune_ckpt_path='13b-mha/llama-13b-mha-2-38-0.5-8192Redpajama-global-second-layer-weight-exp_t_4.0'
# echo "[START] - Start Pruning Model"
# deepspeed --master_port=43475 --include localhost:0,1,2,3,4,5,6,7 examples/prune_model.py --model_name_or_path pinkmanlove/llama-13b-hf --pruning_ratio 0.5 --block_wise --device gpu --block_mlp_layer_start 0 --block_mlp_layer_end 0 --block_attention_layer_start 2 --block_attention_layer_end 38 --output_dir $prune_ckpt_path --pruner_type weighted_taylor --global_pruning --test_after_train --taylor param_second --save_model --grouping_strategy sum --num_examples 512 --prune_block_size 1024 --block_size 1024 --prune_batch_size 1 --torch_dtype float16 --use_flash_attention --deepspeed configs/ds_config_zero3_for_eval.json --dataset_path data/wikitext-103-raw-v1/test --firstabs --pruning_dataset wikitext_cat --layer_importance "3.7188,0.8945,0.7773,0.6484,0.625,0.5859,0.6484,0.5547,0.5273,0.5,0.4922,0.4883,0.4844,0.4746,0.4492,0.4414,0.3867,0.3926,0.3848,0.3672,0.3477,0.3438,0.334,0.3203,0.2812,0.2754,0.2441,0.2402,0.2344,0.2041,0.2109,0.2031,0.1875,0.1963,0.2051,0.2031,0.1963,0.2344,0.3223,0.8164" --grad_info_path /home/zhangyihan/LMFlow/grad_info/13b_8192red_second_grad_info.bin --exp_t 4.0
# echo "[FINISH] - Finish Pruning Model"


# prune_ckpt_path='13b-mha/llama-13b-mha-2-38-0.5-8192Redpajama-global-second'
# echo "[START] - Start Pruning Model"
# deepspeed --master_port=43475 --include localhost:0,1,2,3,4,5,6,7 examples/prune_model.py --model_name_or_path pinkmanlove/llama-13b-hf --pruning_ratio 0.5 --block_wise --device gpu --block_mlp_layer_start 0 --block_mlp_layer_end 0 --block_attention_layer_start 2 --block_attention_layer_end 38 --output_dir $prune_ckpt_path --pruner_type taylor --global_pruning --test_after_train --taylor param_second --save_model --grouping_strategy sum --num_examples 512 --prune_block_size 1024 --block_size 1024 --prune_batch_size 1 --torch_dtype float16 --use_flash_attention --deepspeed configs/ds_config_zero3_for_eval.json --dataset_path data/wikitext-103-raw-v1/test --firstabs --pruning_dataset wikitext_cat --layer_importance "3.7188,0.8945,0.7773,0.6484,0.625,0.5859,0.6484,0.5547,0.5273,0.5,0.4922,0.4883,0.4844,0.4746,0.4492,0.4414,0.3867,0.3926,0.3848,0.3672,0.3477,0.3438,0.334,0.3203,0.2812,0.2754,0.2441,0.2402,0.2344,0.2041,0.2109,0.2031,0.1875,0.1963,0.2051,0.2031,0.1963,0.2344,0.3223,0.8164" --grad_info_path /home/zhangyihan/LMFlow/grad_info/13b_8192red_second_grad_info.bin --exp_t 8.0
# echo "[FINISH] - Finish Pruning Model"

prune_ckpt_path='13b-mha/llama-13b-mha-2-38-0.5-8192Redpajama-local_layer_imp-first-exp_t0.1'
echo "[START] - Start Pruning Model"
deepspeed --master_port=43475 --include localhost:0,1,2,3,4,5,6,7 examples/prune_with_layer_importance.py --model_name_or_path /home/zhangyihan/.cache/huggingface/hub/models--pinkmanlove--llama-13b-hf/snapshots/e7f2b4560f73910c9c6cea51cd7bd132d4745882 --pruning_ratio 0.5 --block_wise --device gpu --block_mlp_layer_start 2 --block_mlp_layer_end 2 --block_attention_layer_start 2 --block_attention_layer_end 38 --output_dir $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_first --save_model --grouping_strategy sum --num_examples 8192 --prune_block_size 1024 --block_size 1024 --prune_batch_size 1 --torch_dtype float16 --use_flash_attention --deepspeed configs/ds_config_zero3_for_eval.json --dataset_path data/wikitext-103-raw-v1/test --pruning_dataset wikitext_cat --layer_importance "3.7188,0.8945,0.7773,0.6484,0.625,0.5859,0.6484,0.5547,0.5273,0.5,0.4922,0.4883,0.4844,0.4746,0.4492,0.4414,0.3867,0.3926,0.3848,0.3672,0.3477,0.3438,0.334,0.3203,0.2812,0.2754,0.2441,0.2402,0.2344,0.2041,0.2109,0.2031,0.1875,0.1963,0.2051,0.2031,0.1963,0.2344,0.3223,0.8164" --layer_importance_weighting_type exp --grad_info_path /home/zhangyihan/LMFlow/grad_info/13b_8192red_first_grad_info.bin --exp_t 0.1
echo "[FINISH] - Finish Pruning Model"

# prune_ckpt_path='13b-mha/llama-13b-mha-2-38-0.5-8192Redpajama-global-first-layer-weight-exp_t_4.0'
# echo "[START] - Start Pruning Model"
# deepspeed --master_port=43475 --include localhost:0,1,2,3,4,5,6,7 examples/prune_model.py --model_name_or_path pinkmanlove/llama-13b-hf --pruning_ratio 0.5 --block_wise --device gpu --block_mlp_layer_start 0 --block_mlp_layer_end 0 --block_attention_layer_start 2 --block_attention_layer_end 38 --output_dir $prune_ckpt_path --pruner_type weighted_taylor --global_pruning --test_after_train --taylor param_first --save_model --grouping_strategy sum --num_examples 512 --prune_block_size 1024 --block_size 1024 --prune_batch_size 1 --torch_dtype float16 --use_flash_attention --deepspeed configs/ds_config_zero3_for_eval.json --dataset_path data/wikitext-103-raw-v1/test --firstabs --pruning_dataset wikitext_cat --layer_importance "3.7188,0.8945,0.7773,0.6484,0.625,0.5859,0.6484,0.5547,0.5273,0.5,0.4922,0.4883,0.4844,0.4746,0.4492,0.4414,0.3867,0.3926,0.3848,0.3672,0.3477,0.3438,0.334,0.3203,0.2812,0.2754,0.2441,0.2402,0.2344,0.2041,0.2109,0.2031,0.1875,0.1963,0.2051,0.2031,0.1963,0.2344,0.3223,0.8164" --grad_info_path /home/zhangyihan/LMFlow/grad_info/13b_8192red_first_grad_info.bin --exp_t 4.0
# echo "[FINISH] - Finish Pruning Model"


# prune_ckpt_path='13b-mha/llama-13b-mha-2-38-0.5-8192Redpajama-global-first'
# echo "[START] - Start Pruning Model"
# deepspeed --master_port=43475 --include localhost:0,1,2,3,4,5,6,7 examples/prune_model.py --model_name_or_path pinkmanlove/llama-13b-hf --pruning_ratio 0.5 --block_wise --device gpu --block_mlp_layer_start 0 --block_mlp_layer_end 0 --block_attention_layer_start 2 --block_attention_layer_end 38 --output_dir $prune_ckpt_path --pruner_type taylor --global_pruning --test_after_train --taylor param_first --save_model --grouping_strategy sum --num_examples 512 --prune_block_size 1024 --block_size 1024 --prune_batch_size 1 --torch_dtype float16 --use_flash_attention --deepspeed configs/ds_config_zero3_for_eval.json --dataset_path data/wikitext-103-raw-v1/test --firstabs --pruning_dataset wikitext_cat --layer_importance "3.7188,0.8945,0.7773,0.6484,0.625,0.5859,0.6484,0.5547,0.5273,0.5,0.4922,0.4883,0.4844,0.4746,0.4492,0.4414,0.3867,0.3926,0.3848,0.3672,0.3477,0.3438,0.334,0.3203,0.2812,0.2754,0.2441,0.2402,0.2344,0.2041,0.2109,0.2031,0.1875,0.1963,0.2051,0.2031,0.1963,0.2344,0.3223,0.8164" --grad_info_path /home/zhangyihan/LMFlow/grad_info/13b_8192red_first_grad_info.bin --exp_t 8.0
# echo "[FINISH] - Finish Pruning Model"