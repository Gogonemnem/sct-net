# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified on Wednesday, September 28, 2022

@author: Guangxing Han
"""

import os
import tempfile
from collections import OrderedDict, defaultdict

import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.logger import create_small_table

from detectron2.evaluation.pascal_voc_evaluation import (
    PascalVOCDetectionEvaluator as detectron2PascalVOCDetectionEvaluator,
    voc_eval,
)


class PascalVOCDetectionEvaluator(detectron2PascalVOCDetectionEvaluator):

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        super().__init__(dataset_name)
        meta = MetadataCatalog.get(dataset_name)
        # add this two terms for calculating the mAP of different subset
        self._base_classes = meta.base_classes
        self._novel_classes = meta.novel_classes

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}

        # Calculate mAP for base and novel classes if available
        base_indices = [i for i, cls in enumerate(self._class_names) if cls in self._base_classes]
        novel_indices = [i for i, cls in enumerate(self._class_names) if cls in self._novel_classes]

        if base_indices:
            mAP_base = {iou: np.mean([aps[iou][i] for i in base_indices]) for iou in aps.keys()}
            ret["bbox"].update(
                {
                    "bAP": np.mean(list(mAP_base.values())),
                    "bAP50": mAP_base[50],
                    "bAP75": mAP_base[75]
                }
            )

        if novel_indices:
            mAP_novel = {iou: np.mean([aps[iou][i] for i in novel_indices]) for iou in aps.keys()}
            ret["bbox"].update(
                {
                    "nAP": np.mean(list(mAP_novel.values())),
                    "nAP50": mAP_novel[50],
                    "nAP75": mAP_novel[75]
                }
            )

        # Log per-class mAP for AP50
        per_class_res = {self._class_names[idx]: ap for idx, ap in enumerate(aps[50])}
        self._logger.info("Evaluate per-class mAP50:\n" + create_small_table(per_class_res))
        self._logger.info("Evaluate overall bbox:\n" + create_small_table(ret["bbox"]))

        return ret
