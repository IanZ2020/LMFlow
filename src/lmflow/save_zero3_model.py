from transformers.modeling_utils import shard_checkpoint
import json
import re
import os
import deepspeed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from collections import OrderedDict, namedtuple
from copy import deepcopy
import torch
import torch.distributed as dist

def add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name

def save_to_state_dict(model, device, destination, prefix, keep_vars):
    for hook in model._state_dict_pre_hooks.values():
        hook(model, prefix, keep_vars)

    for name, param in model._parameters.items():
        if param is not None:
            if prefix + name == 'model.embed_tokens.weight': 
                destination[prefix + name] = param.detach().to(device)
            else:
                with deepspeed.zero.GatheredParameters(param):
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        p = param.detach().to(device)
                        destination[prefix + name] = p
    for name, buf in model._buffers.items():
        if buf is not None and name not in model._non_persistent_buffers_set:
            if not dist.is_initialized() or dist.get_rank() == 0:
                destination[prefix + name] = buf if keep_vars else buf.detach()

def get_state_dict_for_zero3_model(model, device='cpu',destination=None, prefix='', keep_vars = False):
    if destination is None:
        destination = OrderedDict()
    
    local_metadata = dict(version=model._version)
    if hasattr(destination, "_metadata"):
        destination._metadata[prefix[:-1]] = local_metadata

    save_to_state_dict(model, device, destination, prefix, keep_vars)

    for name, module in model._modules.items():
        if module is not None:
            get_state_dict_for_zero3_model(module, device, destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
    for hook in model._state_dict_hooks.values():
        hook_result = hook(model, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result

    return destination

def save_zero3_model(model_to_save, save_directory, device = 'cpu', variant = None):
    model_to_save.config.save_pretrained(save_directory)
    if model_to_save.can_generate():
        model_to_save.generation_config.save_pretrained(save_directory)

    state_dict = get_state_dict_for_zero3_model(model_to_save, device=device)
    if not dist.is_initialized() or dist.get_rank() == 0:
        weights_name = 'pytorch_model.bin'
        max_shard_size = '5GB'

        shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)

        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
                and reg.fullmatch(filename_no_suffix) is not None
            ):
                os.remove(full_filename)

        for shard_file, shard in shards.items():
            torch.save(shard, os.path.join(save_directory, shard_file))

        if index is None:
            path_to_weights = os.path.join(save_directory, add_variant(WEIGHTS_NAME, variant))
            print(f"Model weights saved in {path_to_weights}")
        else:
            save_index_file = WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            print(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )
