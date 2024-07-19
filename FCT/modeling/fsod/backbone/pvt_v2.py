from typing import Callable, List, Union
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

from detectron2.config import configurable
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.backbone.fpn import LastLevelP6P7
from detectron2.layers import ShapeSpec

from timm.layers import to_ntuple
from timm.layers.norm import GroupNorm, GroupNorm1, LayerNorm, LayerNorm2d, LayerNormExp2d, RmsNorm
from timm.models.pvt_v2 import (
    MlpWithDepthwiseConv as _MlpWithDepthwiseConv,
    Attention as _Attention,
    Block as _Block,
    PyramidVisionTransformerStage as _PyramidVisionTransformerStage,
    PyramidVisionTransformerV2 as _PyramidVisionTransformerV2
)

from .fpn import FPN

def get_norm(norm_layer: str, out_channels: int, **kwargs):
    """
    Args:
        norm_layer (str): the name of the normalization layer
        out_channels (int): the number of channels of the input feature
        kwargs: other parameters for the normalization layers

    Returns:
        nn.Module: the normalization layer
    """
    if norm_layer == "LN":
        return LayerNorm(out_channels, **kwargs)
    else:
        raise NotImplementedError(f"Norm type {norm_layer} is not supported (in this code)")

class MlpWithDepthwiseConv(_MlpWithDepthwiseConv):
    def forward(self, x, feat_size: List[int]):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, feat_size[0], feat_size[1]).contiguous()
        x = self.relu(x)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(_Attention):
    def forward_single(self, x, feat_size: List[int]):
        B, N, C = x.shape
        H, W = feat_size
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        if self.pool is not None:
            x = x.permute(0, 2, 1).view(B, C, H, W).contiguous()
            x = self.sr(self.pool(x)).view(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            x = self.act(x)
            kv = self.kv(x).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            if self.sr is not None:
                x = x.permute(0, 2, 1).view(B, C, H, W).contiguous()
                x = self.sr(x).view(B, C, -1).permute(0, 2, 1)
                x = self.norm(x)
                kv = self.kv(x).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv.unbind(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x, feat_size_query: List[int], y=None, feat_size_support: List[int]=None):
        if y is None:
            return self.forward_single(x, feat_size_query)
        # B, N, C = x.shape
        # H, W = feat_size
        if x.shape[0] == 1:
            reverse = False
            xs = [x, y]
            feat_sizes = [feat_size_query, feat_size_support]
        elif y.shape[0] == 1:
            reverse = True
            xs = [y, x]
            feat_sizes = [feat_size_support, feat_size_query]
        else:
            raise ValueError('Either the query or support tensor should have a batch size of 1')
        del x, y, feat_size_query, feat_size_support
        shapes = [x.shape for x in xs]

        qs = [self.q(x).reshape(*x.shape[:2], self.num_heads, -1).permute(0, 2, 1, 3) for x in xs]

        ks, vs = [], []
        for x, feat_size in zip(xs, feat_sizes):
            B, N, C = x.shape
            H, W = feat_size

            if self.pool is not None:
                x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
                x = self.sr(self.pool(x)).reshape(B, C, -1).permute(0, 2, 1)
                x = self.norm(x)
                x = self.act(x)
            elif self.sr is not None:
                x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
                x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
                x = self.norm(x)
            
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
            ks.append(kv[0])
            vs.append(kv[1])
        
        del xs, kv

        k_expanded = ks[0].expand(ks[1].shape[0], -1, -1, -1)
        k_mean = ks[1].mean(dim=0, keepdim=True)

        v_expanded = vs[0].expand(vs[1].shape[0], -1, -1, -1)
        v_mean = vs[1].mean(dim=0, keepdim=True)

        if not reverse:
            k_cats = [torch.cat((ks[0], k_mean), dim=2), torch.cat((k_expanded, ks[1]), dim=2)]
            v_cats = [torch.cat((vs[0], v_mean), dim=2), torch.cat((v_expanded, vs[1]), dim=2)]
        else:
            k_cats = [torch.cat((k_mean, ks[0]), dim=2), torch.cat((ks[1], k_expanded), dim=2)]
            v_cats = [torch.cat((v_mean, vs[0]), dim=2), torch.cat((vs[1], v_expanded), dim=2)]

        outputs = []
        for shape, feat_size, q, k_cat, v_cat in zip(shapes, feat_sizes, qs, k_cats, v_cats):
            H, W = feat_size
            
            if self.fused_attn:
                x = F.scaled_dot_product_attention(q, k_cat, v_cat, dropout_p=self.attn_drop.p if self.training else 0.)
            else:
                q = q * self.scale
                attn = q @ k_cat.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v_cat

            x = x.transpose(1, 2).reshape(*shape)
            x = self.proj(x)
            x = self.proj_drop(x)
            outputs.append(x)
        
        if reverse:
            return reversed(outputs)
        return tuple(outputs)
    
class Block(_Block):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            sr_ratio=1,
            linear_attn=False,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=LayerNorm,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            sr_ratio=sr_ratio,
            linear_attn=linear_attn,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            linear_attn=linear_attn,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.mlp = MlpWithDepthwiseConv(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            extra_relu=linear_attn,
        )

    def forward(self, x, feat_size_query: List[int], y=None, feat_size_support: List[int]=None):
        if y is None:
            return super().forward(x, feat_size_query)

        x, y = self.attn(self.norm1(x), feat_size_query, self.norm1(y), feat_size_support)
        x = x + self.drop_path1(x)
        y = y + self.drop_path1(y)

        x = x + self.drop_path2(self.mlp(self.norm2(x), feat_size_query))
        y = y + self.drop_path2(self.mlp(self.norm2(y), feat_size_support))
        return x, y
    
class PyramidVisionTransformerStage(_PyramidVisionTransformerStage):
    def __init__(
            self, 
            dim: int,
            dim_out: int,
            depth: int,
            branch_embed: bool = True,
            downsample: bool = True,
            num_heads: int = 8,
            sr_ratio: int = 1,
            linear_attn: bool = False,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[List[float], float] = 0.0,
            norm_layer: Callable = LayerNorm,
            ):
        super().__init__(
            dim=dim,
            dim_out=dim_out,
            depth=depth,
            downsample=downsample,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            linear_attn=linear_attn,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
        )
        self.blocks = nn.ModuleList([Block(
            dim=dim_out,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            linear_attn=linear_attn,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
        ) for i in range(depth)])
        
        # branch_embed = False
        if branch_embed:
            num_branches = 2
            self.branch_embedding = nn.Embedding(num_branches, dim_out)
            # self.branch_embedding.apply(self._init_weights)
        self.branch_embed = branch_embed

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Embedding):
    #         m.weight.data.normal_(0, 0.02)


    def forward(self, x, y=None):
        if y is None:
            return super().forward(x)

        if self.downsample is not None:
            # input to downsample is B, C, H, W
            x = self.downsample(x)  # output B, H, W, C
            y = self.downsample(y)

        if self.branch_embed:
            x_branch_embed = torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device)
            x = x + self.branch_embedding(x_branch_embed)

            y_branch_embed = torch.ones(y.shape[:-1], dtype=torch.long, device=y.device)
            y = y + self.branch_embedding(y_branch_embed)

        Bq, Hq, Wq, Cq = x.shape
        Bs, Hs, Ws, Cs = y.shape
        feat_size_query = (Hq, Wq)
        feat_size_support = (Hs, Ws)
        x = x.reshape(Bq, -1, Cq)
        y = y.reshape(Bs, -1, Cs)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x, y = checkpoint.checkpoint(blk, x, feat_size_query, y, feat_size_support)
            else:
                x, y = blk(x, feat_size_query, y, feat_size_support)
        x = self.norm(x)
        y = self.norm(y)
        x = x.reshape(Bq, feat_size_query[0], feat_size_query[1], -1).permute(0, 3, 1, 2).contiguous()
        y = y.reshape(Bs, feat_size_support[0], feat_size_support[1], -1).permute(0, 3, 1, 2).contiguous()
        return x, y

@BACKBONE_REGISTRY.register()
class PyramidVisionTransformerV2(_PyramidVisionTransformerV2, Backbone):
    
    @configurable
    def __init__(
            self,
            *,
            in_chans=3,
            num_classes=None,
            global_pool='avg',
            depths=(3, 4, 6, 3),
            embed_dims=(64, 128, 256, 512),
            num_heads=(1, 2, 4, 8),
            sr_ratios=(8, 4, 2, 1),
            mlp_ratios=(8., 8., 4., 4.),
            qkv_bias=True,
            linear=False,
            drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer="LayerNorm",
            out_features=None,
            freeze_at=0,
            only_train_norm=False,
            branch_embed=True,
            
    ):
        
        num_stages = len(depths)
        mlp_ratios = to_ntuple(num_stages)(mlp_ratios)
        num_heads = to_ntuple(num_stages)(num_heads)
        sr_ratios = to_ntuple(num_stages)(sr_ratios)

        norm_layer = partial(get_norm, norm_layer)

        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes if num_classes is not None else 0,
            global_pool=global_pool,
            depths=depths,
            embed_dims=embed_dims,
            num_heads=num_heads,
            sr_ratios=sr_ratios,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            linear=linear,
            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer
            )
        
        self.stages = nn.ModuleList()
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_dim = embed_dims[0]
        for i in range(num_stages):
            stage = PyramidVisionTransformerStage(
                dim=prev_dim,
                dim_out=embed_dims[i],
                depth=depths[i],
                branch_embed=branch_embed,
                downsample=i > 0,
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
            self.stages.append(stage)
            prev_dim = embed_dims[i]

        self._out_feature_strides = {"patch_embed": 4}
        self._out_feature_channels = {"patch_embed": embed_dims[0]}
        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max(
                [{"pvt2": 1, "pvt3": 2, "pvt4": 3, "pvt5": 4}.get(f, 0) for f in out_features]
            )
            self.stages = self.stages[:num_stages]
        self.stage_names = tuple(["pvt" + str(i + 2) for i in range(num_stages)])
        self._out_feature_strides.update({name: 2 ** (i + 2) for i, name in enumerate(self.stage_names)})
        self._out_feature_channels.update({name: embed_dims[i] for i, name in enumerate(self.stage_names)})
        
        if out_features is None:
            if num_classes is not None:
                out_features = ["linear"]
            else:
                out_features = ["pvt" + str(num_stages + 1)]
        self._out_features = out_features
        assert len(self._out_features)

        self._freeze_stages(freeze_at, only_train_norm)
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "in_chans": input_shape.channels,
            "global_pool": cfg.MODEL.PVT.GLOBAL_POOL,
            "depths": cfg.MODEL.PVT.DEPTHS,
            "embed_dims": cfg.MODEL.PVT.EMBED_DIMS,
            "num_heads": cfg.MODEL.PVT.NUM_HEADS,
            "sr_ratios": cfg.MODEL.PVT.SR_RATIOS,
            "mlp_ratios": cfg.MODEL.PVT.MLP_RATIOS,
            "qkv_bias": cfg.MODEL.PVT.QKV_BIAS,
            "linear": cfg.MODEL.PVT.LINEAR,
            "drop_rate": cfg.MODEL.PVT.DROP_RATE,
            "attn_drop_rate": cfg.MODEL.PVT.ATTN_DROP_RATE,
            "drop_path_rate": cfg.MODEL.PVT.DROP_PATH_RATE,
            "proj_drop_rate": cfg.MODEL.PVT.PROJ_DROP_RATE,
            "norm_layer": cfg.MODEL.PVT.NORM_LAYER,
            "out_features": cfg.MODEL.PVT.OUT_FEATURES,
            "only_train_norm": cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM,
            "freeze_at": cfg.MODEL.BACKBONE.FREEZE_AT,
            "branch_embed": cfg.MODEL.BACKBONE.BRANCH_EMBED,
        }

    def forward_single(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"PVTv2 takes an input of shape (N, C, H, W). Got {x.shape} instead!"
    
        outputs = {}

        x = self.patch_embed(x)
        if "patch_embed" in self._out_features:
            outputs["patch_embed"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        
        if self.num_classes:
            if "linear" in self._out_features:
                outputs["linear"] = self.forward_head(x)
        return outputs

    def forward(self, x, y=None):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        if y is None:
            return self.forward_single(x)

        assert x.dim() == 4, f"PVTv2 takes a query input of shape (N, C, H, W). Got {x.shape} instead!"
        assert y.dim() == 4, f"PVTv2 takes a support input of shape (N, C, H, W). Got {y.shape} instead!"
        outputs_query = {}
        outputs_support = {}

        x = self.patch_embed(x)
        y = self.patch_embed(y)
        if "patch_embed" in self._out_features:
            outputs_query["patch_embed"] = x
            outputs_support["patch_embed"] = y
        for name, stage in zip(self.stage_names, self.stages):
            x, y = stage(x, y)
            if name in self._out_features:
                outputs_query[name] = x
                outputs_support[name] = y
        
        if self.num_classes:
            if "linear" in self._out_features:
                outputs_query["linear"] = self.forward_head(x)

        return outputs_query, outputs_support

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
    
    def _freeze_stages(self, freeze_at=0, only_train_norm=False):
        if freeze_at >= 1:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        
        module_names = ["downsample", "blocks", "norm"]
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for module_name in module_names:
                    module = getattr(stage, module_name)
                    if module is None:
                        continue

                    for param in module.parameters():
                        param.requires_grad = False

                    if not only_train_norm:
                        continue

                    for name, param in module.named_parameters():
                        if 'norm' in name:
                            param.requires_grad = True

        module_names = ["branch_embedding"]
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for module_name in module_names:
                    module = getattr(stage, module_name, None)
                    if module is None:
                        continue

                    for param in module.parameters():
                        param.requires_grad = False
                        
                        # Zero out the branch embeddings to avoid randomness
                        param.data.fill_(0)


@BACKBONE_REGISTRY.register()
def build_retinanet_pvtv2_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    pvt_args = PyramidVisionTransformerV2.from_config(cfg, input_shape)
    bottom_up = PyramidVisionTransformerV2(**pvt_args)
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
