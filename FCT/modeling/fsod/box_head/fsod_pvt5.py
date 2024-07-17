# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY

from ..backbone.fsod_pvt_v2 import FsodPyramidVisionTransformerStage
from ..backbone.pvt_v2 import get_norm
from functools import partial


# To get torchscript support, we make the head a subclass of `nn.Module`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_BOX_HEAD_REGISTRY.register()
class FsodPVT5BoxHead(nn.Module):
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
    ):
        super().__init__()

        norm_layer = partial(get_norm, norm_layer)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        
        i = -1 # because we take last stage in PVTv2
        self.stage = FsodPyramidVisionTransformerStage(
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
        }

    def forward(self, x, y):

        x, y = self.stage(x, y)
        return x, y

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        return self._output_size
        
@ROI_BOX_HEAD_REGISTRY.register()
class MultiRelationBoxHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *args,
        global_relation: bool=True,
        local_correlation: bool=True,
        patch_relation: bool=True,
        **kwargs
        ):
        super().__init__(*args, **kwargs)

        input_size = input_shape.channels
        self._output_size = ShapeSpec(channels=input_size)
        
        self.global_relation = global_relation
        self.local_correlation = local_correlation
        self.patch_relation = patch_relation

        if global_relation:
            self.avgpool_fc = nn.AvgPool2d(7)
            self.fc_1 = nn.Linear(input_size * 2, input_size)
            self.fc_2 = nn.Linear(input_size, input_size)

            for l in [self.fc_1, self.fc_2]:
                nn.init.constant_(l.bias, 0)
                nn.init.normal_(l.weight, std=0.01)
        
        if local_correlation:
            self.conv_cor = nn.Conv2d(input_size, input_size, 1, padding=0)
            
            for l in [self.conv_cor]:
                nn.init.constant_(l.bias, 0)
                nn.init.normal_(l.weight, std=0.01)
        
        if patch_relation:
            self.avgpool = nn.AvgPool2d(kernel_size=3,stride=1)
            self.conv_1 = nn.Conv2d(input_size*2, int(input_size/4), 1, padding=0)
            self.conv_2 = nn.Conv2d(int(input_size/4), int(input_size/4), 3, padding=0)
            self.conv_3 = nn.Conv2d(int(input_size/4), input_size, 1, padding=0)
            
            for l in [self.conv_1, self.conv_2, self.conv_3]:
                nn.init.constant_(l.bias, 0)
                nn.init.normal_(l.weight, std=0.01)
        
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "global_relation": cfg.MODEL.MULTI_RELATION.GLOBAL_RELATION,
            "local_correlation": cfg.MODEL.MULTI_RELATION.LOCAL_CORRELATION,
            "patch_relation": cfg.MODEL.MULTI_RELATION.PATCH_RELATION,
        }
    
    def forward(self, x_query, x_support):
        x_fc, x_cor, x_pr = 0, 0, 0
        # fc
        if self.global_relation:
            x_query_fc = x_query.mean(dim=(2, 3))
            support_fc = x_support.mean(dim=(2, 3)).expand_as(x_query_fc)
            x_fc = torch.cat((x_query_fc, support_fc), 1)
            x_fc = F.relu(self.fc_1(x_fc), inplace=True)
            x_fc = F.relu(self.fc_2(x_fc), inplace=True)

        # correlation
        if self.local_correlation:
            x_query_cor = self.conv_cor(x_query)
            support_cor = self.conv_cor(x_support)
            x_cor = F.relu(F.conv2d(x_query_cor, support_cor.permute(1,0,2,3), groups=self._output_size.channels), inplace=True).squeeze(3).squeeze(2)

        # relation
        if self.patch_relation:
            support_relation = x_support.expand_as(x_query)
            x_pr = torch.cat((x_query, support_relation), 1)
            x_pr = F.relu(self.conv_1(x_pr), inplace=True) # 5x5
            x_pr = self.avgpool(x_pr)
            x_pr = F.relu(self.conv_2(x_pr), inplace=True) # 3x3
            x_pr = F.relu(self.conv_3(x_pr), inplace=True) # 3x3
            x_pr = x_pr.mean(dim=(2, 3))

        return x_fc + x_cor + x_pr
    
    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        return self._output_size

# TODO: keys are now reusable, but is a bit hacky.
# Is it even "good" to have both pvt4 and multi_relation in the same head?
@ROI_BOX_HEAD_REGISTRY.register()
class FsodPVT5MultiRelationBoxHead(nn.Module):
    @configurable
    def __init__(self, pvt4_stage, multi_relation):
        super().__init__()
        self.stage = pvt4_stage
        self.multi_relation = multi_relation

    @classmethod
    def from_config(cls, cfg, input_shape):
        pvt4_box_head = ROI_BOX_HEAD_REGISTRY.get("FsodPVT4BoxHead")(cfg, input_shape)
        multi_relation = ROI_BOX_HEAD_REGISTRY.get("MultiRelationBoxHead")(cfg, pvt4_box_head.output_shape)
        return {
            "pvt4_stage": pvt4_box_head.stage,
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
