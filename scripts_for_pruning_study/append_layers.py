# Load model directly

import sys
import os
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))

from transformers import HfArgumentParser
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

from lmflow.args import (
    ModelArguments,
    AutoArguments,
)

from lmflow.models.auto_model import AutoModel

import copy

@dataclass
class LayerDropArguments:
    layers_to_append: Optional[str] = field(
        default = None,
        metadata={
            "help": (
                "The indexes of transformer layers you want to drop"
            )
        }
    )

    output_model_path: Optional[str] = field(
        default = None,
        metadata={
            "help": (
                "The indexes of transformer layers you want to drop"
            )
        }
    )

def reinit_layer(layer):
    for weight_name, parameter in layer.named_parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=6e-3)

def main():
    parser = HfArgumentParser((ModelArguments, LayerDropArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, layer_drop_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, layer_drop_args = parser.parse_args_into_dataclasses()

    model = AutoModel.get_model(model_args)

    layers_to_append = [int(eval(idx)) for idx in layer_drop_args.layers_to_append.split(',')]

    num_of_new_layers = len(layers_to_append)

    new_layers = [copy.deepcopy(model.get_backend_model().base_model.layers[0]) for i in range(num_of_new_layers)]

    [reinit_layer(layer) for layer in new_layers]

    for i, j in enumerate(sorted(layers_to_append, reverse=True)):
        model.get_backend_model().base_model.layers.insert(j, new_layers[i])

    print(len(model.get_backend_model().base_model.layers))
    model.get_backend_model().config.num_hidden_layers += len(layers_to_append)
    
    model.save(layer_drop_args.output_model_path, save_full_model=True)
    

if __name__ == '__main__':
    main()