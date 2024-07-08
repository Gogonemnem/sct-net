import math

from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelP6P7
from detectron2.config import configurable

from .pvt_v2 import PyramidVisionTransformerV2
from .fsod_pvt_v2 import FsodPyramidVisionTransformerV2


class FsodFPN(FPN):
    def __init__(self, *args, freeze_at=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._freeze_stages(freeze_at)

    def _freeze_stages(self, freeze_at=0):
        input_shapes = self.bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in self.in_features]
        stages = [int(math.log2(s)) for s in strides]

        module_names = ["fpn_lateral", "fpn_output"]

        for stage in stages:
            if freeze_at >= stage:
                for module_name in module_names:
                    module = getattr(self, f"{module_name}{stage}")
                    if module is not None:
                        for param in module.parameters():
                            param.requires_grad = False
                    else:
                        raise ValueError(f"Module {module_name}{stage} does not exist")

        if self.top_block is not None:
            for idx, module in enumerate(self.top_block.children(), start=stages[-1] + 1):
                if freeze_at >= idx:
                    for param in module.parameters():
                        param.requires_grad = False
    
    def forward(self, x, y):
        x, y = self.bottom_up(x, y)
        x = self._fpn_forward(x)
        y = self._fpn_forward(y)
        return x, y

    def _fpn_forward(self, x):
        results = []
        prev_features = self.lateral_convs[0](x[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = x[features]
                ## Instead of scale factor x2, we use the size instead. Makes the fpn more robust to uneven input sizes
                top_down_features = F.interpolate(prev_features, size=features.shape[-2:], mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in x:
                top_block_in_feature = x[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}
    
@BACKBONE_REGISTRY.register()
def build_retinanet_pvtv2_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = PyramidVisionTransformerV2(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    last_in_feature = in_features[-1]
    in_channels_p6p7 = bottom_up.output_shape()[last_in_feature].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels, in_feature=last_in_feature),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_retinanet_fsod_pvtv2_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = FsodPyramidVisionTransformerV2(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    last_in_feature = in_features[-1]
    in_channels_p6p7 = bottom_up.output_shape()[last_in_feature].channels
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    backbone = FsodFPN(
        freeze_at=freeze_at,
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels, in_feature=last_in_feature),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone