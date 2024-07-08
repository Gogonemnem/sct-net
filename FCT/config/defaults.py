from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# PVT options
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.PVT = CN()

_C.MODEL.PVT.GLOBAL_POOL = 'avg'
_C.MODEL.PVT.DEPTHS = [3, 4, 6, 3]
_C.MODEL.PVT.EMBED_DIMS = [64, 128, 320, 512]
_C.MODEL.PVT.NUM_HEADS = [1, 2, 5, 8]
_C.MODEL.PVT.SR_RATIOS = [8, 4, 2, 1]
_C.MODEL.PVT.MLP_RATIOS = [8, 8, 4, 4]
_C.MODEL.PVT.QKV_BIAS = True
_C.MODEL.PVT.LINEAR = True
_C.MODEL.PVT.DROP_RATE = 0.0
_C.MODEL.PVT.PROJ_DROP_RATE = 0.0
_C.MODEL.PVT.ATTN_DROP_RATE = 0.0
_C.MODEL.PVT.DROP_PATH_RATE = 0.1
_C.MODEL.PVT.NORM_LAYER = "LN"
# _C.MODEL.PVT.OUT_FEATURES = ["pvt2", "pvt3", "pvt4"]
# _C.MODEL.PVT.OUT_FEATURES = ["pvt3", "pvt4"]
_C.MODEL.PVT.OUT_FEATURES = ["pvt4"]


# ---------------------------------------------------------------------------- #
# Multi-Relation Box Head Detector Options
# ---------------------------------------------------------------------------- #
_C.MODEL.MULTI_RELATION = CN()

_C.MODEL.MULTI_RELATION.GLOBAL_RELATION = True
_C.MODEL.MULTI_RELATION.LOCAL_CORRELATION = True
_C.MODEL.MULTI_RELATION.PATCH_RELATION = True

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./output"

# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.SOLVER.HEAD_LR_FACTOR = 1.0
_C.SOLVER.SOLVER_TYPE = "adamw"

# ---------------------------------------------------------------------------- #
# Few shot setting
# ---------------------------------------------------------------------------- #
_C.INPUT.FS = CN()
# Whether to enable two-branch 'few-shot' setting
_C.INPUT.FS.ENABLED = False
# Whether to enable the actual few-shot examples setting
_C.INPUT.FS.FEW_SHOT = False
_C.INPUT.FS.SUPPORT_WAY = 2
_C.INPUT.FS.SUPPORT_SHOT = 0
_C.INPUT.FS.SUPPORT_EXCLUDE_QUERY = False

# _C.DATASETS.TRAIN_KEEPCLASSES = 'all'
_C.DATASETS.TEST_KEEPCLASSES = ''
_C.DATASETS.TEST_SHOTS = (1,2,3,5,10,30)
_C.DATASETS.SEEDS = 0

_C.MODEL.BACKBONE.ONLY_TRAIN_NORM = False
_C.MODEL.BACKBONE.BRANCH_EMBED = True
_C.MODEL.BACKBONE.TRAIN_BRANCH_EMBED = True
_C.MODEL.RPN.FREEZE_RPN = False
_C.MODEL.ROI_HEADS.FREEZE_ROI_FEATURE_EXTRACTOR = False
_C.MODEL.ROI_HEADS.ONLY_TRAIN_NORM = False