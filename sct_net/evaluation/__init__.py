"""
Created on Wednesday, September 28, 2022

@author: Guangxing Han
"""
from .pascal_voc_evaluation import PascalVOCDetectionEvaluator

from .coco_style_evaluation import (
    COCOEvaluator,
    DOTAEvaluator,
    DIOREvaluator,
    PASCALEvaluator,
)

__all__ = [k for k in globals().keys() if not k.startswith("_")]
