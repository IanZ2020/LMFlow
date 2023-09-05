#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
import torch
import deepspeed
import sys
import os
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from transformers import HfArgumentParser,AutoModelForCausalLM, AutoConfig

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline


def main():
	# Parses arguments
    pipeline_name = "distiller"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()


    world_size = int(os.getenv("WORLD_SIZE", "1"))
    config = AutoConfig.from_pretrained(pipeline_args.ref_model)
    model_hidden_size = config.hidden_size
    train_batch_size = 1 * world_size
    ds_config = {
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
    }
    ref_model = AutoModelForCausalLM.from_pretrained(
            pipeline_args.ref_model,
            torch_dtype=torch.float16,
        )

    ds_engine = deepspeed.initialize(model=ref_model, config_params=ds_config)[0]
    ds_engine.module.eval()
    # Initialization
    finetuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    dataset = Dataset(data_args)
    model = AutoModel.get_model(model_args)

    # Finetuning
    tuned_model = finetuner.tune(model=model, ref_model=ds_engine.module, dataset=dataset)


if __name__ == '__main__':
    main()
