import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple
import deepspeed
import torch.distributed as dist
from lmflow.datasets.dataset import Dataset
from transformers.deepspeed import HfDeepSpeedConfig,is_deepspeed_zero3_enabled, deepspeed_config
import torch
import numpy as np
from transformers import LlamaTokenizer, HfArgumentParser
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP
from LLMPruner.models.hf_llama import (
    LlamaConfig,
    PrunedLlamaForCausalLM, 
    PrunedLlamaConfig
)
from lmflow.args import (
    ModelArguments,
    AutoArguments,
    DatasetArguments
)
from copy import deepcopy
from lmflow import save_zero3_model
import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric, evaluate_ppl
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts
from deepspeed.runtime.zero import ZeroParamStatus

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

def get_grad_info(model, path_to_grad):
    grad_info = torch.load(path_to_grad, map_location='cpu')
    for name, param in model.named_parameters():
        param.offload_grad = grad_info[name]

def acc_grad(model, firstabs = False, second_grad = False):
    for param in model.parameters():
        if hasattr(param, 'offload_grad') and param.offload_grad is not None:
            if firstabs:
                param.offload_grad += param.grad.data.detach().to('cpu').abs()
            else:
                param.offload_grad += param.grad.data.detach().to('cpu')
        else:
            if firstabs:
                param.offload_grad = param.grad.data.detach().to('cpu').abs()
            else:
                param.offload_grad = param.grad.data.detach().to('cpu')
        if second_grad:
            if hasattr(param, 'acc_grad') and param.acc_grad is not None:
                param.acc_grad += (param.grad.data * param.grad.data).detach().to('cpu')
            else:
                param.acc_grad = (param.grad.data * param.grad.data).detach().to('cpu')
        param.grad = None
        torch.cuda.empty_cache()

def average_gradients(model, second_grad = False):
    size = float(dist.get_world_size())
    for param in model.parameters():
        param.offload_grad = param.offload_grad.to(dist.get_rank())
        dist.all_reduce(param.offload_grad.data, op=dist.ReduceOp.SUM)
        param.offload_grad = param.offload_grad.to('cpu')
        torch.cuda.empty_cache()
        if second_grad:
            param.acc_grad = param.acc_grad.to(dist.get_rank())
            dist.all_reduce(param.acc_grad.data, op=dist.ReduceOp.SUM)
            param.acc_grad = param.acc_grad.to('cpu')
        torch.cuda.empty_cache()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_layer_prune_ratio(pruning_ratio, layer_importance, layer_importance_weighting_type='linear'):
    layer_importance = np.array(layer_importance)
    num_of_layers = len(layer_importance)
    if layer_importance_weighting_type == 'linear':
        weight_factor =  (layer_importance.max() - layer_importance)
        weight_factor = weight_factor / weight_factor.mean()
        pruning_ratio_weighted = np.full((num_of_layers), pruning_ratio) * weight_factor
        return pruning_ratio_weighted
    elif layer_importance_weighting_type == 'exp':
        layer_importance = np.exp(layer_importance.max()-layer_importance)
        weight_factor = layer_importance / layer_importance.mean()
        pruning_ratio_weighted = np.full((num_of_layers), pruning_ratio) * weight_factor
        return pruning_ratio_weighted
    else:
        raise NotImplementedError(f"Weighting type {layer_importance_weighting_type} not implemented.")

def main(model_args, data_args, args):
    set_random_seed(args.seed)
    
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)

    with open (args.deepspeed, "r") as f:
        ds_config = json.load(f)
    dschf = HfDeepSpeedConfig(ds_config)
    deepspeed.init_distributed()

    logger = LoggerWithDepth(
        env_name="{}".format(args.output_dir), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True,
        local_rank = local_rank
    )

    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)

    torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
    )

    if model_args.arch_type == 'pruned_decoder_only':
        config = PrunedLlamaConfig.from_pretrained(model_args.model_name_or_path)
    else: 
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)

    supported_gpu_device = None
    for gpu in GPU_SUPPORT_FLASH_ATTENTION:
        if gpu in torch.cuda.get_device_name():
            supported_gpu_device = gpu
    if model_args.use_flash_attention:
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

    if model_args.arch_type == 'pruned_decoder_only':
        model = PrunedLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config = config,
            torch_dtype = torch_dtype
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype = torch_dtype,
            config = config
        )
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]

    if args.test_before_train:
        logger.log("\n==================Generation Results before Pruning================\n")
        ds_engine.module.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(local_rank)

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

        ppl = evaluate_ppl(ds_engine.module, tokenizer, dataset = Dataset(data_args), block_size = data_args.block_size)
        logger.log("PPL before pruning: {}".format(ppl))

    pruner_type = args.pruner_type

    for param in model.parameters():
        param.requires_grad_(True)
    
    if is_deepspeed_zero3_enabled():
        before_pruning_parameters = sum(p.ds_numel for p in model.parameters() if p.requires_grad)
    else:
        before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(device = local_rank) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.
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
    
    if 'first' not in args.taylor:
        second_grad = True
    else:
        second_grad = False

    ###########################
    args.layer_importance = [float(x) for x in args.layer_importance.split(',')]
    layer_prune_ratio = get_layer_prune_ratio(args.pruning_ratio ,args.layer_importance, args.layer_importance_weighting_type)
    logger.log("Layer Pruning Ratio: {}".format(layer_prune_ratio))
    ch_sparsity_dict = {model.base_model.layers[i]:layer_prune_ratio[i] for i in range(len(layer_prune_ratio))}
    ###########################


    if args.block_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity_dict": ch_sparsity_dict, 
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
            ds_engine.module,
            forward_prompts,
            **kwargs
        )
        ds_engine.module.zero_grad()
        ds_engine.module.train()
        logger.log("Start Pruning")
        for i in range(args.iterative_steps):
            if args.grad_info_path is not None and i == 0:
                get_grad_info(model, args.grad_info_path)
            if pruner_type in ['taylor']:
                example_prompts = get_examples(args.pruning_dataset, tokenizer, args.num_examples, seq_len = args.prune_block_size).to(device = local_rank)
                batch_num = args.num_examples // args.prune_batch_size
                batch_num_per_device = batch_num // world_size
                batch_num = batch_num_per_device * world_size

                example_prompts = example_prompts[0: args.prune_batch_size * batch_num].view(batch_num, args.prune_batch_size, args.prune_block_size)
                current_batch = example_prompts[local_rank*batch_num_per_device : (local_rank+1)*batch_num_per_device]

                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                for j in range(batch_num_per_device):
                    batch_input = current_batch[j]
                    loss = ds_engine.module(batch_input, labels=batch_input).loss
                    logger.log(f'batch{j}, loss: {loss}')
                    loss.backward()
                    acc_grad(model, args.firstabs, second_grad)
                average_gradients(model, second_grad)
                del loss.grad
                    
            pruner.step()

            if is_deepspeed_zero3_enabled():
                after_pruning_parameters = 0
                for name,p in model.named_parameters():
                    if p.ds_status == ZeroParamStatus.AVAILABLE: 
                        after_pruning_parameters += p.numel()
                    else:
                        with deepspeed.zero.GatheredParameters(p):
                            if p.requires_grad:
                                after_pruning_parameters += p.numel()
            else:
                after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))
        
            # modify inferece-related attributes
            for layer in model.model.layers:
                layer.self_attn.num_heads = layer.self_attn.q_proj.out_features // layer.self_attn.head_dim
                layer.self_attn.num_key_value_heads = layer.self_attn.num_heads
                print(layer.self_attn.q_proj.out_features, layer.self_attn.o_proj.in_features, layer.self_attn.num_heads)
            
            #evaluate the model after each step of pruning
            dataset = Dataset(data_args)
            ppl = evaluate_ppl(ds_engine.module, tokenizer, dataset = dataset, block_size = data_args.block_size)
            logger.log("PPL after pruning: {}".format(ppl))
            logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

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
            ds_engine.module,
            forward_prompts,
            **kwargs
        )
        ds_engine.module.zero_grad()
        
        logger.log("Start Pruning")
        for i in range(args.iterative_steps):

            if pruner_type in ['taylor']:
                example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len = 64)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()
                acc_grad(model)
            average_gradients(model)
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

    model.half()
    config = PrunedLlamaConfig()
    config.from_llama_model(model)
    print('Get config')
    model.config = config
    model.config.use_cache = True
    if args.save_model:
        print('saving model')
        if is_deepspeed_zero3_enabled():
            save_zero3_model.save_zero3_model(model, logger.best_checkpoint_path)
        else:
            model.save_pretrained(logger.best_checkpoint_path)
        print('done')
        if not dist.is_initialized() or dist.get_rank() == 0:
            tokenizer.save_pretrained(logger.best_checkpoint_path)
        
    dist.barrier()

if __name__ == "__main__":
    ## Prepare training_args
    pipeline_name = "pruner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)
    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
    model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, pipeline_args)