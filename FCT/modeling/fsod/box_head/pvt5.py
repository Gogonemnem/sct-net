from functools import partial

import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY

from ..backbone.pvt_v2 import PyramidVisionTransformerStage, get_norm


@ROI_BOX_HEAD_REGISTRY.register()
class PVT5BoxHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        depths=(3, 4, 6, 3),
        embed_dims=(64, 128, 256, 512),
        num_heads=(1, 2, 4, 8),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratios=(8., 8., 4., 4.),
        qkv_bias=True,
        linear=False,
        proj_drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer="LN",
        branch_embed=True,
        cross_attn=(True, True, True, True),
    ):
        super().__init__()

        norm_layer = partial(get_norm, norm_layer)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
    
        i = -1 # because we take last stage in PVTv2
        self.stage = PyramidVisionTransformerStage(
                dim=input_shape.channels,
                dim_out=embed_dims[i],
                depth=depths[i],
                downsample=i != 0,
                num_heads=num_heads[i],
                sr_ratio=sr_ratios[i],
                mlp_ratio=mlp_ratios[i],
                linear_attn=linear,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                branch_embed=branch_embed and cross_attn[i], # TODO: branch embed needed at all?
                cross_attn=cross_attn[i],
            )
        self.feature_info = dict(num_chs=embed_dims[i], reduction=4 * 2**i, module=f'stages.{i}')

        self._output_size = ShapeSpec(channels=embed_dims[i], height=input_shape.height // 2, width=input_shape.width // 2)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "depths": cfg.MODEL.PVT.DEPTHS,
            "embed_dims": cfg.MODEL.PVT.EMBED_DIMS,
            "num_heads": cfg.MODEL.PVT.NUM_HEADS,
            "sr_ratios": cfg.MODEL.PVT.SR_RATIOS,
            "mlp_ratios": cfg.MODEL.PVT.MLP_RATIOS,
            "qkv_bias": cfg.MODEL.PVT.QKV_BIAS,
            "linear": cfg.MODEL.PVT.LINEAR,
            "proj_drop_rate": cfg.MODEL.PVT.PROJ_DROP_RATE,
            "attn_drop_rate": cfg.MODEL.PVT.ATTN_DROP_RATE,
            "drop_path_rate": cfg.MODEL.PVT.DROP_PATH_RATE,
            "norm_layer": cfg.MODEL.PVT.NORM_LAYER,
            "branch_embed": cfg.MODEL.BACKBONE.BRANCH_EMBED and cfg.INPUT.FS.ENABLED,
            "cross_attn": cfg.MODEL.BACKBONE.CROSS_ATTN,
        }

    def forward(self, x, y=None):
        return self.stage(x, y)

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return o
        
# TODO: keys are now reusable, but is a bit hacky.
# Is it even "good" to have both pvt4 and multi_relation in the same head?
@ROI_BOX_HEAD_REGISTRY.register()
class PVT5MultiRelationBoxHead(nn.Module):
    @configurable
    def __init__(self, pvt_stage, multi_relation):
        super().__init__()
        self.stage = pvt_stage
        self.multi_relation = multi_relation

    @classmethod
    def from_config(cls, cfg, input_shape):
        pvt_box_head = ROI_BOX_HEAD_REGISTRY.get("PVT5BoxHead")(cfg, input_shape)
        multi_relation = ROI_BOX_HEAD_REGISTRY.get("MultiRelationBoxHead")(cfg, pvt_box_head.output_shape)
        return {
            "pvt_stage": pvt_box_head.stage,
            "multi_relation": multi_relation,
        }

    def forward(self, x, y):
        x, y = self.stage(x, y)
        return self.multi_relation(x, y)
    
    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        return self.multi_relation._output_size