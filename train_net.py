#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Created on Wednesday, September 28, 2022

This script is a simplified version of the training script in detectron2/tools.

@author: Guangxing Han
"""

import os
import logging
from collections import OrderedDict

import torch.utils.data

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
)

from sct_net.config import get_cfg
from sct_net.data import DatasetMapperWithSupportCOCO, DatasetMapperWithSupportVOC
from sct_net.data.build import build_detection_train_loader, build_detection_test_loader
from sct_net.solver import build_optimizer
from sct_net.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator,DIOREvaluator, DOTAEvaluator, PASCALEvaluator




class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'pascalvoc' in dataset_name:
            return PascalVOCDetectionEvaluator(dataset_name)
        elif 'coco' in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif 'dota' in dataset_name:
            return DOTAEvaluator(dataset_name, cfg, True, output_folder)
        elif 'dior' in dataset_name:
            return DIOREvaluator(dataset_name, cfg, True, output_folder)


    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

class FsodTrainer(Trainer):
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        if 'pascalvoc' in cfg.DATASETS.TRAIN[0]:
            mapper = DatasetMapperWithSupportVOC(cfg)
        else:
            mapper = DatasetMapperWithSupportCOCO(cfg)
        return build_detection_train_loader(cfg, mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warning(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue

            test_seeds = cfg.DATASETS.SEEDS
            test_shots = cfg.DATASETS.TEST_SHOTS
            cur_test_shots_set = set(test_shots)
            if 'pascalvoc' in cfg.DATASETS.TRAIN[0]:
                evaluation_dataset = 'voc'
                voc_test_shots_set = set([1,2,3,5,10])
                test_shots_join = cur_test_shots_set.intersection(voc_test_shots_set)
                test_keepclasses = cfg.DATASETS.TEST_KEEPCLASSES
            else:
                evaluation_dataset = 'coco'
                coco_test_shots_set = set([1,2,3,5,10,30])
                test_shots_join = cur_test_shots_set.intersection(coco_test_shots_set)
                test_keepclasses = cfg.DATASETS.TEST_KEEPCLASSES

            if cfg.INPUT.FS.FEW_SHOT:
                test_shots = [cfg.INPUT.FS.SUPPORT_SHOT + 1] # add one due to few shot: query + k-1 support
                test_shots_join = set(test_shots)

            print("================== test_shots_join=", test_shots_join)
            for shot in test_shots_join:
                print("evaluating {}.{} for {} shot".format(evaluation_dataset, test_keepclasses, shot))
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model.module.init_support_features(evaluation_dataset, shot, test_keepclasses, test_seeds)
                else:
                    model.init_support_features(evaluation_dataset, shot, test_keepclasses, test_seeds)

                results_i = inference_on_dataset(model, data_loader, evaluator)
                results[dataset_name] = results_i
                if comm.is_main_process():
                    assert isinstance(
                        results_i, dict
                    ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                        results_i
                    )
                    logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                    print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
    
    
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    
    if args.additional_configs:
        for additional_cfg_path in args.additional_configs:
            cfg.merge_from_file(additional_cfg_path)
    
    if args.opts:
        if args.opts[0] == '--':
            args.opts = args.opts[1:]  # Remove the '--' delimiter
        cfg.merge_from_list(args.opts)
    
    check_fewshot(cfg)

    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="sct-net")

    return cfg

def check_fewshot(cfg):
    if cfg.INPUT.FS.FEW_SHOT:
        assert cfg.INPUT.FS.ENABLED, "Few-shot learning requires enabling few-shot setting"

    if cfg.INPUT.FS.ENABLED:
        assert cfg.INPUT.FS.SUPPORT_SHOT > 0, "SUPPORT_SHOT should be larger than 0"
        assert cfg.INPUT.FS.SUPPORT_WAY > 0, "SUPPORT_WAY should be larger than 0"


def main(args):
    cfg = setup(args)

    if cfg.INPUT.FS.ENABLED:
        trainer = FsodTrainer
    else:
        trainer = Trainer

    if args.eval_only:
        model = trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = trainer.test(cfg, model)
        return res

    trainer = trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

def get_custom_argument_parser():
    parser = default_argument_parser()
    parser.add_argument(
        "--additional-configs",
        nargs='+',  # This allows you to pass multiple paths
        help="List of additional config file paths to merge"
    )
    return parser

if __name__ == "__main__":
    args = get_custom_argument_parser().parse_args()
    try:
        import idr_torch
        from sct_net.engine import launch

        # Use idr_torch to set up environment variables
        rank = idr_torch.rank
        local_rank = idr_torch.local_rank
        size = idr_torch.size
        cpus_per_task = idr_torch.cpus_per_task
        world_size = size
        os.environ['MASTER_ADDR'] = os.environ['MASTER_ADDR']
        os.environ['MASTER_PORT'] = os.environ['MASTER_PORT']

        num_nodes = len(idr_torch.hostnames)
        num_gpus_per_node = torch.cuda.device_count()

        args.num_machines = num_nodes
        args.machine_rank = rank
        args.local_rank = local_rank
        args.dist_url = 'env://'
        args.num_gpus = num_gpus_per_node
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            local_rank=args.local_rank,
            args=(args,),
        )
    except ModuleNotFoundError:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )

    print("Command Line Args:", args)
    
    