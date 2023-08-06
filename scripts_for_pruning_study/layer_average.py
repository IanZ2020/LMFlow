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


@dataclass
class LayerDropArguments:
    layers_to_merge: Optional[str] = field(
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


def merge_layers(layer, layer_tomerge):
    for weight_name, parameter_tomerge in list(layer_tomerge.named_parameters()):
        parameter = layer.get_parameter(weight_name)
        parameter_tomerge.data = (parameter_tomerge.data + parameter.data) / 2

def main():
    parser = HfArgumentParser((ModelArguments, LayerDropArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, layer_drop_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, layer_drop_args = parser.parse_args_into_dataclasses()

    model = AutoModel.get_model(model_args)

    layers_tomerge = [int(eval(idx)) for idx in layer_drop_args.layers_to_merge.split(',')]
    layers = [i-1 for i in layers_tomerge]

    for i, i_tomerge in zip(layers, layers_tomerge):
        merge_layers(model.get_backend_model().base_model.layers[i], model.get_backend_model().base_model.layers[i_tomerge])
    for i in sorted(layers, reverse=True):
        del model.get_backend_model().base_model.layers[i]
    model.get_backend_model().config.num_hidden_layers -= len(layers)
    
    
    model.save(layer_drop_args.output_model_path, save_full_model=True)
    

if __name__ == '__main__':
    main()