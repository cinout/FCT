#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Created on Wednesday, September 28, 2022

This script is a simplified version of the training script in detectron2/tools.

@author: Guangxing Han
"""

import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)

from FCT.config import get_cfg
from FCT.data.build import build_detection_train_loader, build_detection_test_loader
from FCT.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator
from FCT.solver import build_optimizer
import FCT.modeling.fsod.pvt_roi_heads

import bisect
import copy
import itertools
import logging
import numpy as np
import operator
import pickle
import torch.utils.data
import torch

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if "coco" in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return PascalVOCDetectionEvaluator(dataset_name)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="FCT")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print(
        f">>>> device names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}",
    )
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
