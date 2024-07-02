from typing import Callable, List, Union
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

from detectron2.config import configurable
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

from .pvt_v2 import Attention, Block, PyramidVisionTransformerStage
from timm.layers import to_ntuple, LayerNorm

from .pvt_v2 import PyramidVisionTransformerV2, get_norm

class CrossAttention(Attention):
    def forward(self, x, y, feat_size_query: List[int], feat_size_support: List[int]):
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
        # B_x, N_x, C_x = x.shape
        # B_y, N_y, C_y = y.shape
        # H_x, W_x = feat_size_query
        # H_y, W_y = feat_size_support
        # assert B_x == 1 or B_y == 1
        # # assert B_y == 1
        # q_x = self.q(x).reshape(B_x, N_x, self.num_heads, C_x // self.num_heads).permute(0, 2, 1, 3)
        # q_y = self.q(y).reshape(B_y, N_y, self.num_heads, C_y // self.num_heads).permute(0, 2, 1, 3)
        # # print("q_x.shape={}, q_y.shape={}".format(q_x.shape, q_y.shape))

        # if self.pool is None:
        #     if self.sr is not None:
        #         x_ = x.permute(0, 2, 1).reshape(B_x, C_x, H_x, W_x)
        #         x_ = self.sr(x_).reshape(B_x, C_x, -1).permute(0, 2, 1)
        #         x_ = self.norm(x_)
        #         kv_x = self.kv(x_).reshape(B_x, -1, 2, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1, 4)

        #         y_ = y.permute(0, 2, 1).reshape(B_y, C_y, H_y, W_y)
        #         y_ = self.sr(y_).reshape(B_y, C_y, -1).permute(0, 2, 1)
        #         y_ = self.norm(y_)
        #         kv_y = self.kv(y_).reshape(B_y, -1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1, 4)
        #     else:
        #         kv_x = self.kv(x).reshape(B_x, -1, 2, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1, 4)
        #         kv_y = self.kv(y).reshape(B_y, -1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1, 4)
        # else:
        #     x_ = x.permute(0, 2, 1).reshape(B_x, C_x, H_x, W_x)
        #     x_ = self.sr(self.pool(x_)).reshape(B_x, C_x, -1).permute(0, 2, 1)
        #     x_ = self.norm(x_)
        #     x_ = self.act(x_)
        #     kv_x = self.kv(x_).reshape(B_x, -1, 2, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1, 4)

        #     y_ = y.permute(0, 2, 1).reshape(B_y, C_y, H_y, W_y)
        #     y_ = self.sr(self.pool(y_)).reshape(B_y, C_y, -1).permute(0, 2, 1)
        #     y_ = self.norm(y_)
        #     y_ = self.act(y_)
        #     kv_y = self.kv(y_).reshape(B_y, -1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1, 4)

        # k_x, v_x = kv_x[0], kv_x[1]
        # k_y, v_y = kv_y[0], kv_y[1]
        # # print("k_x.shape={}, k_y.shape={}".format(k_x.shape, k_y.shape))
        # # print("v_x.shape={}, v_y.shape={}".format(v_x.shape, v_y.shape))

        # if B_x == 1:
        #     k_y_avg = k_y.mean(0, True)
        #     v_y_avg = v_y.mean(0, True)
        #     k_cat_x = torch.cat((k_x, k_y_avg), dim=2)
        #     v_cat_x = torch.cat((v_x, v_y_avg), dim=2)
        # elif B_y == 1:
        #     k_y_ext = k_y.repeat(B_x, 1, 1, 1)
        #     v_y_ext = v_y.repeat(B_x, 1, 1, 1)
        #     k_cat_x = torch.cat((k_x, k_y_ext), dim=2)
        #     v_cat_x = torch.cat((v_x, v_y_ext), dim=2)

        # # print("k_cat.shape={}, v_cat.shape={}".format(k_cat.shape, v_cat.shape))

        # attn_x = (q_x @ k_cat_x.transpose(-2, -1)) * self.scale
        # attn_x = attn_x.softmax(dim=-1)
        # attn_x = self.attn_drop(attn_x)

        # x = (attn_x @ v_cat_x).transpose(1, 2).reshape(B_x, N_x, C_x)
        # x = self.proj(x)
        # x = self.proj_drop(x)

        # if B_x == 1:
        #     k_x_ext = k_x.repeat(B_y, 1, 1, 1)
        #     v_x_ext = v_x.repeat(B_y, 1, 1, 1)
        #     k_cat_y = torch.cat((k_x_ext, k_y), dim=2)
        #     v_cat_y = torch.cat((v_x_ext, v_y), dim=2)
        # elif B_y == 1:
        #     k_x_avg = k_x.mean(0, True)
        #     v_x_avg = v_x.mean(0, True)
        #     k_cat_y = torch.cat((k_x_avg, k_y), dim=2)
        #     v_cat_y = torch.cat((v_x_avg, v_y), dim=2)

        # attn_y = (q_y @ k_cat_y.transpose(-2, -1)) * self.scale
        # attn_y = attn_y.softmax(dim=-1)
        # attn_y = self.attn_drop(attn_y)

        # y = (attn_y @ v_cat_y).transpose(1, 2).reshape(B_y, N_y, C_y)
        # y = self.proj(y)
        # y = self.proj_drop(y)

        # return x, y
    
class FsodBlock(Block):
    def __init__(
            self,
            dim,
            num_heads,
            sr_ratio=1,
            linear_attn=False,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            **kwargs,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            linear_attn=linear_attn,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            **kwargs,
        )
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            linear_attn=linear_attn,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x, y, feat_size_query: List[int], feat_size_support: List[int]):
        res_x, res_y = self.attn(self.norm1(x), self.norm1(y), feat_size_query, feat_size_support)
        x = x + self.drop_path1(res_x)
        y = y + self.drop_path1(res_y)

        x = x + self.drop_path2(self.mlp(self.norm2(x), feat_size_query))
        y = y + self.drop_path2(self.mlp(self.norm2(y), feat_size_support))
        return x, y
    
class FsodPyramidVisionTransformerStage(PyramidVisionTransformerStage):
    def __init__(
            self,
            dim_out: int,
            depth: int,
            branch_embed: bool = True,
            num_heads: int = 8,
            sr_ratio: int = 1,
            linear_attn: bool = False,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[List[float], float] = 0.0,
            norm_layer: Callable = LayerNorm,
            **kwargs,
            ):
        super().__init__(
            dim_out=dim_out,
            depth=depth,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            linear_attn=linear_attn,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            **kwargs,
        )
        self.blocks = nn.ModuleList([FsodBlock(
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


    def forward(self, x, y):
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
                x, y = checkpoint.checkpoint(blk, x, y, feat_size_query, feat_size_support)
            else:
                x, y = blk(x, y, feat_size_query, feat_size_support)
        x = self.norm(x)
        y = self.norm(y)
        x = x.reshape(Bq, feat_size_query[0], feat_size_query[1], -1).permute(0, 3, 1, 2).contiguous()
        y = y.reshape(Bs, feat_size_support[0], feat_size_support[1], -1).permute(0, 3, 1, 2).contiguous()
        return x, y

@BACKBONE_REGISTRY.register()
class FsodPyramidVisionTransformerV2(PyramidVisionTransformerV2):
    
    @configurable
    def __init__(
            self,
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
            freeze_at=0,
            only_train_norm=False,
            branch_embed=True,
            **kwargs,
            
    ):
        super().__init__(
            depths=depths,
            embed_dims=embed_dims,
            num_heads=num_heads,
            sr_ratios=sr_ratios,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            linear=linear,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            freeze_at=freeze_at,
            only_train_norm=only_train_norm,
            **kwargs,
        )
        num_stages = len(self.stages)
        mlp_ratios = to_ntuple(num_stages)(mlp_ratios)
        num_heads = to_ntuple(num_stages)(num_heads)
        sr_ratios = to_ntuple(num_stages)(sr_ratios)

        norm_layer = partial(get_norm, norm_layer)
        
        self.stages = nn.ModuleList()
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_dim = embed_dims[0]
        for i in range(num_stages):
            stage = FsodPyramidVisionTransformerStage(
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
        self._freeze_stages(freeze_at, only_train_norm)
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["branch_embed"] = cfg.MODEL.BACKBONE.BRANCH_EMBED
        return ret

    def forward(self, x, y):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"PVTv2 takes an input of shape (N, C, H, W). Got {x.shape} instead!"
    
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
    
    def _freeze_stages(self, freeze_at=0, only_train_norm=False):
        super()._freeze_stages(freeze_at, only_train_norm)
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