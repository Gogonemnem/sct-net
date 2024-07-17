import torch
from torch import nn

from timm.layers import to_2tuple

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY
from ..backbone.wip_swin import SwinTransformerV2Stage

from ..backbone.pvt_v2 import get_norm
from functools import partial

@ROI_BOX_HEAD_REGISTRY.register()
class Swin5BoxHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        img_size=224,
        patch_size=4,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        act_layer='gelu',
        norm_layer="LN",
        pretrained_window_size=(0, 0, 0, 0),
    ):
        super().__init__()

        norm_layer = partial(get_norm, norm_layer)
        if not isinstance(embed_dim, (tuple, list)):
            num_layers = len(depths)
            embed_dim = [int(embed_dim * 2 ** i) for i in range(num_layers)]

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]

        i = -1

        ith_stage = len(depths[:i])
        img_size = to_2tuple(img_size)
        input_resolution = (
            img_size[0] // (patch_size * 2 ** (ith_stage - 1)),
            img_size[1] // (patch_size * 2 ** (ith_stage - 1))
        )

        self.layer = SwinTransformerV2Stage(
            dim=input_shape.channels,
            out_dim=embed_dim[i],
            input_resolution=input_resolution,
            depth=depths[i],
            num_heads=num_heads[i],
            window_size=window_size,
            downsample=i != 0,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[i],
            act_layer=act_layer,
            norm_layer=norm_layer,
            pretrained_window_size=pretrained_window_size[i],
        )
        self.feature_info = dict(num_chs=embed_dim[i], reduction=patch_size * (2 ** i), module=f'layers.{i}')

        self._output_size = ShapeSpec(channels=embed_dim[i], height=input_shape.height // 2, width=input_shape.width // 2)

    
    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "img_size": cfg.MODEL.SWIN.IMG_SIZE,
            "patch_size": cfg.MODEL.SWIN.PATCH_SIZE,
            "embed_dim": cfg.MODEL.SWIN.EMBED_DIM,
            "depths": cfg.MODEL.SWIN.DEPTHS,
            "num_heads": cfg.MODEL.SWIN.NUM_HEADS,
            "window_size": cfg.MODEL.SWIN.WINDOW_SIZE,
            "mlp_ratio": cfg.MODEL.SWIN.MLP_RATIO,
            "qkv_bias": cfg.MODEL.SWIN.QKV_BIAS,
            "proj_drop_rate": cfg.MODEL.SWIN.PROJ_DROP_RATE,
            "attn_drop_rate": cfg.MODEL.SWIN.ATTN_DROP_RATE,
            "drop_path_rate": cfg.MODEL.SWIN.DROP_PATH_RATE,
            "act_layer": cfg.MODEL.SWIN.ACT_LAYER,
            "norm_layer": cfg.MODEL.SWIN.NORM_LAYER,
            "pretrained_window_size": cfg.MODEL.SWIN.PRETRAINED_WINDOW_SIZES,
        }

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.layer(x)
        return x.permute(0, 3, 1, 2).contiguous()

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