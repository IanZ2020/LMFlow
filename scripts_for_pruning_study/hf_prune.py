import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple

# import deepspeed

import torch
import numpy as np
from transformers import LlamaTokenizer
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP
from LLMPruner.models.hf_llama import (
    LlamaConfig,
    PrunedLlamaForCausalLM, 
    PrunedLlamaConfig
)
import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts

MODELS_SUPPORT_FLASH_ATTENTION = [
    "PrunedLlamaForCausalLM",
    "LlamaForCausalLM",
    "GPTNeoForCausalLM",
    "GPT2ForCausalLM",
    "BloomForCausalLM"
]

GPU_SUPPORT_FLASH_ATTENTION = {
    "A100": ["PrunedLlamaForCausalLM","LlamaForCausalLM", "GPTNeoForCausalLM", "GPT2ForCausalLM", "BloomForCausalLM"],
    "A40": ["GPTNeoForCausalLM", "GPT2ForCausalLM", "BloomForCausalLM"]
}

try:
    import flash_attn
    if int(flash_attn.__version__.split(".")[0]) == 2:
        GPU_SUPPORT_FLASH_ATTENTION = {
            "A100": ["PrunedLlamaForCausalLM","LlamaForCausalLM", "GPTNeoForCausalLM", "GPT2ForCausalLM", "BloomForCausalLM"],
            "A40": ["PrunedLlamaForCausalLM","LlamaForCausalLM","GPTNeoForCausalLM", "GPT2ForCausalLM", "BloomForCausalLM"]
        }
except:
    pass

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

    torch_dtype = (
            args.torch_dtype
            if args.torch_dtype in ["auto", None]
            else getattr(torch, args.torch_dtype)
    )

    if args.is_base_model_pruned:
        config = PrunedLlamaConfig.from_pretrained(args.base_model)
    else: 
        config = LlamaConfig.from_pretrained(args.base_model)

    supported_gpu_device = None
    for gpu in GPU_SUPPORT_FLASH_ATTENTION:
        if gpu in torch.cuda.get_device_name():
            supported_gpu_device = gpu
    if args.use_flash_attention:
        if not any(model_supported in config.architectures
                    for model_supported in MODELS_SUPPORT_FLASH_ATTENTION):
            logger.warning(
                f"Model \"{config.architectures}\" does not support"
                " flash attention, use normal attention layer instead"
            )
        elif supported_gpu_device is None:
            logger.warning(
                f"Your decice \"{torch.cuda.get_device_name()}\""
                " does not support flash attention, it will"
                " automatically use normal attention layer"
            )
        else:
            
            supported_models = GPU_SUPPORT_FLASH_ATTENTION[supported_gpu_device]
            
            config.use_cache = False
            if ("LlamaForCausalLM" in config.architectures and "LlamaForCausalLM" in supported_models) or ("PrunedLlamaForCausalLM" in config.architectures and "PrunedLlamaForCausalLM" in supported_models):
                from lmflow.utils.flash_attention.llama_flash_attention import (
                    replace_llama_attn_with_flash_attn,
                )
                replace_llama_attn_with_flash_attn()
            elif "GPTNeoForCausalLM" in config.architectures and "GPTNeoForCausalLM" in supported_models:
                from lmflow.utils.flash_attention.gpt_neo_flash_attention import (
                    replace_gpt_neo_attn_with_flash_attn,
                )
                replace_gpt_neo_attn_with_flash_attn()
            elif "GPT2ForCausalLM" in config.architectures and "GPT2ForCausalLM" in supported_models:
                from lmflow.utils.flash_attention.gpt2_flash_attention import (
                    replace_gpt2_attn_with_flash_attn,
                )
                replace_gpt2_attn_with_flash_attn()
            elif "BloomForCausalLM" in config.architectures and "BloomForCausalLM" in supported_models:
                from lmflow.utils.flash_attention.bloom_flash_attention import (
                    replace_bloom_attn_with_flash_attn
                )
                replace_bloom_attn_with_flash_attn()
            else:
                raise ValueError(
                    f"Model \"{config.architectures}\" with GPU {supported_gpu_device} does not support"
                    " flash attention, use normal attention layer instead"
                )

    if args.is_base_model_pruned:
        model = PrunedLlamaForCausalLM.from_pretrained(
            args.base_model,
            config = config,
            torch_dtype = torch_dtype
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            low_cpu_mem_usage=True if args.torch_version >=1.9 else False,
            config = config
        )

    # deepspeed initialization
    # if args.device == "gpu":
    #     deepspeed.init_distributed()
    #     ds_engine = deepspeed.initialize(model=model, config_params=args.ds_config)[0]
    #     ds_engine.module.eval()

    if args.test_before_train:
        logger.log("\n==================Generation Results before Pruning================\n")
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
    
        ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.device)
        logger.log("PPL before pruning: {}".format(ppl))

    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = llama_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
    else:
        raise NotImplementedError

    logger.log("Use {} pruner...".format(pruner_type))
    
    if args.block_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio, 
            "ignored_layers":[],
            "channel_groups": {
            },
            "consecutive_groups": {
                layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
            },
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
            },
            "root_module_types": None, 
            "root_instances": [model.model.layers[i].self_attn.q_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +
                              [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
        }
        logger.log("Pruning Attention Layer = {}".format(list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
        logger.log("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))

        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            **kwargs
        )
        model.zero_grad()

        logger.log("Start Pruning")
        for i in range(args.iterative_steps):

            if pruner_type in ['taylor']:
                example_prompts = get_examples('bookcorpus', tokenizer, args.num_examples, seq_len = args.block_size).to(args.device)
                batch_num = args.num_examples // args.batch_size
                example_prompts = example_prompts[0: args.batch_size * batch_num].view(args.num_of_batch, args.batch_size, args.block_size)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                if args.taylor in ['param_mix', 'param_second']:
                    for j in range(batch_num):
                        batch_input = example_prompts[j]
                        loss = model(batch_input, labels=batch_input).loss
                        logger.log(f'batch{j}, loss: {loss}')
                        loss.backward()
                        for module_param in model.parameters():
                            module_param.grad = module_param.grad * module_param.grad / args.num_examples
                            if hasattr(module_param, 'acc_grad'):
                                module_param.acc_grad += module_param.grad
                            else:
                                module_param.acc_grad = copy.deepcopy(module_param.grad)
                        model.zero_grad()
                        del loss.grad
                    
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))
        
            # modify inferece-related attributes
            for layer in model.model.layers:
                layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        del pruner

    elif args.channel_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            "ignored_layers":[],
            #"round_to": model.config.num_attention_heads * 2,
            "channel_groups": {
                #layer.self_attn: layer.self_attn.num_heads for layer in model.model.layers
            },
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
                #LlamaAttention: llama_pruner.hf_attention_pruner,
            },
            "root_module_types": [LlamaRMSNorm, LlamaAttention],
        }

        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            **kwargs
        )
        model.zero_grad()
        
        logger.log("Start Pruning")
        for i in range(args.iterative_steps):

            if pruner_type in ['taylor']:
                example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len = 64)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        # modify inferece-related attributes
        model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
        model.zero_grad()
        
        del pruner
            
    elif args.layer_wise:
        model.model.layers = model.model.layers[:args.layer]
        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        raise NotImplementedError
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
    gc.collect()
    torch.cuda.empty_cache()

    if args.save_model:
        model.half()
        config = PrunedLlamaConfig()
        config.from_llama_model(model)
        model.config = config
        model.save_pretrained(logger.best_checkpoint_path)

        tokenizer.save_pretrained(logger.best_checkpoint_path)
    if args.eval_device != "cpu":
        model.half()
    model.to(args.eval_device)

    model.config.pad_token_id = tokenizer.pad_token_id = 0 
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if args.test_after_train:
        logger.log("\n==================Generation Results After Pruning================\n")
        
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
        
        logger.log("\n==================Finish================\n")
    
    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    logger.log("PPL after pruning: {}".format(ppl))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--batch_size', type=int, default=4, help='deepspeed config')
    parser.add_argument('--block_size', type=int, default=1024, help='deepspeed config')
    parser.add_argument('--ds_config', type=str, default="configs/ds_config_zero2.json", help='deepspeed config')
    parser.add_argument('--torch_dtype', type=str, default="float16", help='torch dtype')
    parser.add_argument('--is_base_model_pruned', action='store_true', help='whether the base model is pruned')
    parser.add_argument('--use_flash_attention', action='store_true', help='whether to use flash attention')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=10)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
