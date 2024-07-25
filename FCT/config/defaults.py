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
_C.MODEL.PVT.OUT_FEATURES = ["pvt4"]


# ---------------------------------------------------------------------------- #
# Swin Transformer options
# ---------------------------------------------------------------------------- #
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.IMG_SIZE = 256
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.GLOBAL_POOL = 'avg'
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 16
_C.MODEL.SWIN.MLP_RATIO = 4.0
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.DROP_RATE = 0.0
_C.MODEL.SWIN.PROJ_DROP_RATE = 0.0
_C.MODEL.SWIN.ATTN_DROP_RATE = 0.0
_C.MODEL.SWIN.DROP_PATH_RATE = 0.1
_C.MODEL.SWIN.ACT_LAYER = 'gelu'
_C.MODEL.SWIN.NORM_LAYER = "LN"
_C.MODEL.SWIN.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWIN.OUT_FEATURES = ["stage4"]

# ---------------------------------------------------------------------------- #
# Twins options
# ---------------------------------------------------------------------------- #
_C.MODEL.TWINS = CN()
_C.MODEL.TWINS.IMG_SIZE = 224
_C.MODEL.TWINS.PATCH_SIZE = 4
_C.MODEL.TWINS.GLOBAL_POOL = 'avg'
_C.MODEL.TWINS.EMBED_DIMS = [64, 128, 256, 512]
_C.MODEL.TWINS.NUM_HEADS = [2, 4, 8, 16]
_C.MODEL.TWINS.MLP_RATIOS = [4, 4, 4, 4]
_C.MODEL.TWINS.DEPTHS = [2, 2, 10, 4]
_C.MODEL.TWINS.SR_RATIOS = [8, 4, 2, 1]
_C.MODEL.TWINS.WSS = [7, 7, 7, 7]
_C.MODEL.TWINS.DROP_RATE = 0.0
_C.MODEL.TWINS.POS_DROP_RATE = 0.0
_C.MODEL.TWINS.PROJ_DROP_RATE = 0.0
_C.MODEL.TWINS.ATTN_DROP_RATE = 0.0
_C.MODEL.TWINS.DROP_PATH_RATE = 0.0
# _C.MODEL.TWINS.NORM_LAYER = "LN"
_C.MODEL.TWINS.OUT_FEATURES = ["stage4"]

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN.BBOX_REG_LOSS_TYPE = "giou"

# ---------------------------------------------------------------------------- #
# Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "giou"

# ---------------------------------------------------------------------------- #
# Multi-Relation Box Head Detector Options
# ---------------------------------------------------------------------------- #
_C.MODEL.MULTI_RELATION = CN()

_C.MODEL.MULTI_RELATION.GLOBAL_RELATION = True
_C.MODEL.MULTI_RELATION.LOCAL_CORRELATION = True
_C.MODEL.MULTI_RELATION.PATCH_RELATION = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY  # None means following WEIGHT_DECAY

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./output"
_C.VIS_PERIOD = 20

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
_C.MODEL.BACKBONE.CROSS_ATTN = (True, True, True, True)
_C.MODEL.RPN.FREEZE_RPN = False
_C.MODEL.ROI_HEADS.FREEZE_ROI_FEATURE_EXTRACTOR = False
_C.MODEL.ROI_HEADS.ONLY_TRAIN_NORM = False