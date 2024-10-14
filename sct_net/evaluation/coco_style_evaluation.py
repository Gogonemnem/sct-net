import numpy as np

from detectron2.utils.logger import create_small_table
from detectron2.evaluation.coco_evaluation import (
    COCOEvaluator as detectron2COCOEvaluator,
)

class COCOStyleEvaluator(detectron2COCOEvaluator):
    CLASS_NAMES = []

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.
        Additional to the detectron2 COCOEvaluator, distinguish between novel and base classes

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        results = super()._derive_coco_results(coco_eval, iou_type, class_names)

        idx_novel = []
        idx_base = []

        for idx, name in enumerate(class_names):
            if name in self.CLASS_NAMES:
                idx_novel.append(idx)
            else:
                idx_base.append(idx)

        precisions = coco_eval.eval["precision"]
        # Note this next operation reshapes precisions to size (cls, iou, recall, area range,)
        novel_precision = precisions[:, :, idx_novel, :, -1]
        base_precision = precisions[:, :, idx_base, :, -1]

        metric_slices = {
            "AP":   (slice(None), slice(None), slice(None), 0),
            "AP50": (slice(None),           0, slice(None), 0),
            "AP75": (slice(None),           5, slice(None), 0),
            "APs":  (slice(None), slice(None), slice(None), 1),
            "APm":  (slice(None), slice(None), slice(None), 2),
            "APl":  (slice(None), slice(None), slice(None), 3),
        }
        
        results_novel = {}
        results_base = {}
        for metric in metrics:
            metric_slice = metric_slices[metric]

            novel_precisions_slice = novel_precision[metric_slice]
            valid_novel_precisions = novel_precisions_slice[novel_precisions_slice > -1]
            results_novel[f"novel_{metric}"] = np.mean(valid_novel_precisions) * 100 if valid_novel_precisions.size else float("nan")

            base_precisions_slice = base_precision[metric_slice]
            valid_base_precisions = base_precisions_slice[base_precisions_slice > -1]
            results_base[f"base_{metric}"] = np.mean(valid_base_precisions) * 100 if valid_base_precisions.size else float("nan")

        results.update(results_novel)
        self._logger.info(
            f"Evaluation results for {len(idx_novel)} Novel classes {iou_type}: \n" + create_small_table(results_novel)
        )

        results.update(results_base)
        self._logger.info(
            f"Evaluation results for {len(idx_base)} Base classes {iou_type}: \n" + create_small_table(results_base)
        )
        return results
    

class COCOEvaluator(COCOStyleEvaluator):
    CLASS_NAMES = [
        "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "dining table", "dog", "horse", "motorcycle", "person",
        "potted plant", "sheep", "couch", "train", "tv",
    ]

class DOTAEvaluator(COCOStyleEvaluator):
    CLASS_NAMES = ["storage-tank", "tennis-court", "soccer-ball-field"]

class DIOREvaluator(COCOStyleEvaluator):
    CLASS_NAMES = [ "Airplane ", "Baseball field ", "Tennis court ", "Train station ", "Wind mill"]

class PASCALEvaluator(COCOStyleEvaluator):
    CLASS_NAMES = ['bird', 'bus', 'cow', 'motorbike', 'sofa']
