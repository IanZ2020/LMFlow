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
import ast

from lmflow.args import (
    ModelArguments,
    AutoArguments,
)

from lmflow.models.auto_model import AutoModel


@dataclass
class LayerDropArguments:
    layers_to_be_pruned: Optional[str] = field(
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

def main():
    parser = HfArgumentParser((ModelArguments, LayerDropArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, layer_drop_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, layer_drop_args = parser.parse_args_into_dataclasses()

    model = AutoModel.get_model(model_args)

    model.get_backend_model().prune_heads(ast.literal_eval(layer_drop_args.layers_to_be_pruned))

    
    model.save(layer_drop_args.output_model_path, save_full_model=True)
    

if __name__ == '__main__':
    main()