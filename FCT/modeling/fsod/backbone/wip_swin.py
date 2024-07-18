# import math
# from typing import Callable, List, Optional, Tuple, Union
# from functools import partial

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint

# from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, get_act_layer, LayerType
# from timm.models.swin_transformer_v2 import (
#     SwinTransformerV2 as _SwinTransformerV2,
#     SwinTransformerV2Stage as _SwinTransformerV2Stage,
#     WindowAttention,
#     window_partition,
#     window_reverse,
# )
# _int_or_tuple_2_t = Union[int, Tuple[int, int]]


# from detectron2.modeling.backbone import BACKBONE_REGISTRY, Backbone
# from detectron2.layers import ShapeSpec
# from detectron2.config import configurable


# from .pvt_v2 import get_norm
# # from detectron2.layers import get_norm

# class SwinTransformerV2Block(nn.Module):
#     def __init__(
#             self,
#             dim: int,
#             input_resolution: _int_or_tuple_2_t,
#             num_heads: int,
#             window_size: _int_or_tuple_2_t = 7,
#             shift_size: _int_or_tuple_2_t = 0,
#             mlp_ratio: float = 4.,
#             qkv_bias: bool = True,
#             proj_drop: float = 0.,
#             attn_drop: float = 0.,
#             drop_path: float = 0.,
#             act_layer: LayerType = "gelu",
#             norm_layer: nn.Module = nn.LayerNorm,
#             pretrained_window_size: _int_or_tuple_2_t = 0,
#     ):
#         """
#         Args:
#             dim: Number of input channels.
#             input_resolution: Input resolution.
#             num_heads: Number of attention heads.
#             window_size: Window size.
#             shift_size: Shift size for SW-MSA.
#             mlp_ratio: Ratio of mlp hidden dim to embedding dim.
#             qkv_bias: If True, add a learnable bias to query, key, value.
#             proj_drop: Dropout rate.
#             attn_drop: Attention dropout rate.
#             drop_path: Stochastic depth rate.
#             act_layer: Activation layer.
#             norm_layer: Normalization layer.
#             pretrained_window_size: Window size in pretraining.
#         """
#         nn.Module.__init__(self)
#         self.dim = dim
#         self.input_resolution = to_2tuple(input_resolution)
#         self.num_heads = num_heads
#         ws, ss = self._calc_window_shift(window_size, shift_size)
#         self.window_size: Tuple[int, int] = ws
#         self.shift_size: Tuple[int, int] = ss
#         self.window_area = self.window_size[0] * self.window_size[1]
#         self.mlp_ratio = mlp_ratio
#         act_layer = get_act_layer(act_layer)

#         self.attn = WindowAttention(
#             dim,
#             window_size=to_2tuple(self.window_size),
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             pretrained_window_size=to_2tuple(pretrained_window_size),
#         )
#         self.norm1 = norm_layer(dim)
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.mlp = Mlp(
#             in_features=dim,
#             hidden_features=int(dim * mlp_ratio),
#             act_layer=act_layer,
#             drop=proj_drop,
#         )
#         self.norm2 = norm_layer(dim)
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
#     def _calc_window_shift(self,
#                            target_window_size: _int_or_tuple_2_t,
#                            target_shift_size: _int_or_tuple_2_t) -> Tuple[Tuple[int, int], Tuple[int, int]]:
#         target_window_size = to_2tuple(target_window_size)
#         target_shift_size = to_2tuple(target_shift_size)
#         window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
#         shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
#         return tuple(window_size), tuple(shift_size)

#     def _attn(self, x: torch.Tensor) -> torch.Tensor:
#         B, H, W, C = x.shape

#         # cyclic shift [+ attention mask (different way of same timm implementation)]
#         has_shift = any(self.shift_size)
#         if has_shift:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
#             attn_mask = self._create_attn_mask(x)
#         else:
#             shifted_x = x
#             attn_mask = None

#         # partition windows
#         x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
#         x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C

#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
#         shifted_x = window_reverse(attn_windows, self.window_size, (H, W))  # B H' W' C

#         # reverse cyclic shift
#         if has_shift:
#             x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
#         else:
#             x = shifted_x
#         return x

#     def _create_attn_mask(self, x: torch.Tensor) -> torch.Tensor:
#         _, H, W, _ = x.shape
#         img_mask = torch.zeros((1, H, W, 1), device=x.device)  # 1 H W 1
#         cnt = 0
#         for h in (
#                 slice(0, -self.window_size[0]),
#                 slice(-self.window_size[0], -self.shift_size[0]),
#                 slice(-self.shift_size[0], None)):
#             for w in (
#                     slice(0, -self.window_size[1]),
#                     slice(-self.window_size[1], -self.shift_size[1]),
#                     slice(-self.shift_size[1], None)):
#                 img_mask[:, h, w, :] = cnt
#                 cnt += 1
#         mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#         mask_windows = mask_windows.view(-1, self.window_area)
#         attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#         attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#         return attn_mask

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, H, W, C = x.shape

#         # Check if the height and width are divisible by the window size, if not pad the input
#         pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
#         pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
#         if pad_h > 0 or pad_w > 0:
#             x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
#         x = x + self.drop_path1(self.norm1(self._attn(x)))

#         x = x[:, :H, :W, :].contiguous()  # Remove padding if added

#         x = x.reshape(B, -1, C)
#         x = x + self.drop_path2(self.norm2(self.mlp(x)))
#         x = x.reshape(B, H, W, C)
#         return x


# class SwinTransformerV2Stage(_SwinTransformerV2Stage):
#     def __init__(
#             self,
#             out_dim: int,
#             input_resolution: _int_or_tuple_2_t,
#             depth: int,
#             num_heads: int,
#             window_size: _int_or_tuple_2_t,
#             mlp_ratio: float = 4.,
#             qkv_bias: bool = True,
#             proj_drop: float = 0.,
#             attn_drop: float = 0.,
#             drop_path: float = 0.,
#             act_layer: Union[str, Callable] = 'gelu',
#             norm_layer: nn.Module = nn.LayerNorm,
#             pretrained_window_size: _int_or_tuple_2_t = 0,
#             **kwargs
#     ):
#         super().__init__(
#             out_dim=out_dim,
#             input_resolution=input_resolution,
#             depth=depth,
#             num_heads=num_heads,
#             window_size=window_size,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             proj_drop=proj_drop,
#             attn_drop=attn_drop,
#             drop_path=drop_path,
#             act_layer=act_layer,
#             norm_layer=norm_layer,
#             pretrained_window_size=pretrained_window_size,
#             **kwargs,
#         )
#         window_size = to_2tuple(window_size)
#         shift_size = tuple([w // 2 for w in window_size])

#         # build blocks
#         self.blocks = nn.ModuleList([
#             SwinTransformerV2Block(
#                 dim=out_dim,
#                 input_resolution=input_resolution,
#                 num_heads=num_heads,
#                 window_size=window_size,
#                 shift_size=0 if (i % 2 == 0) else shift_size,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 proj_drop=proj_drop,
#                 attn_drop=attn_drop,
#                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                 act_layer=act_layer,
#                 norm_layer=norm_layer,
#                 pretrained_window_size=pretrained_window_size,
#             )
#             for i in range(depth)])
#         self.norm = norm_layer(out_dim)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = super().forward(x)
#         x = self.norm(x) # added
#         return x

# @BACKBONE_REGISTRY.register()
# class SwinTransformerV2(_SwinTransformerV2, Backbone):
#     @configurable
#     def __init__(
#             self,
#             img_size: _int_or_tuple_2_t = 224,
#             patch_size: int = 4,
#             in_chans: int = 3,
#             num_classes: Optional[int] = None,
#             global_pool: str = 'avg',
#             embed_dim: int = 96,
#             depths: Tuple[int, ...] = (2, 2, 6, 2),
#             num_heads: Tuple[int, ...] = (3, 6, 12, 24),
#             window_size: _int_or_tuple_2_t = 7,
#             mlp_ratio: float = 4.,
#             qkv_bias: bool = True,
#             drop_rate: float = 0.,
#             proj_drop_rate: float = 0.,
#             attn_drop_rate: float = 0.,
#             drop_path_rate: float = 0.1,
#             act_layer: Union[str, Callable] = 'gelu',
#             norm_layer: Union[str, Callable] = nn.LayerNorm,
#             pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0, 0),
#             out_features=None,
#             freeze_at=0,
#             **kwargs,
#     ):
#         if isinstance(norm_layer, str):
#             norm_layer = partial(get_norm, norm_layer)

#         super().__init__(
#             img_size=img_size,
#             patch_size=patch_size,
#             in_chans=in_chans,
#             num_classes=num_classes if num_classes is not None else 0,
#             global_pool=global_pool,
#             embed_dim=embed_dim,
#             depths=depths,
#             num_heads=num_heads,
#             window_size=window_size,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             drop_rate=drop_rate,
#             proj_drop_rate=proj_drop_rate,
#             attn_drop_rate=attn_drop_rate,
#             drop_path_rate=drop_path_rate,
#             act_layer=act_layer,
#             norm_layer=norm_layer,
#             pretrained_window_sizes=pretrained_window_sizes,
#             **kwargs,
#         )
        
#         if not isinstance(embed_dim, (tuple, list)):
#             embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

#         dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
#         layers = []
#         in_dim = embed_dim[0]
#         scale = 1
#         grid_size = tuple([s // p for s, p in zip(to_2tuple(img_size), to_2tuple(patch_size))])
#         for i in range(self.num_layers):
#             out_dim = embed_dim[i]
#             layers += [SwinTransformerV2Stage(
#                 dim=in_dim,
#                 out_dim=out_dim,
#                 input_resolution=(
#                     grid_size[0] // scale,
#                     grid_size[1] // scale),
#                 depth=depths[i],
#                 downsample=i > 0,
#                 num_heads=num_heads[i],
#                 window_size=window_size,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 proj_drop=proj_drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_path=dpr[i],
#                 act_layer=act_layer,
#                 norm_layer=norm_layer,
#                 pretrained_window_size=pretrained_window_sizes[i],
#             )]
#             in_dim = out_dim
#             if i > 0:
#                 scale *= 2
#             self.feature_info += [dict(num_chs=out_dim, reduction=patch_size * scale, module=f'layers.{i}')]

#         self.layers = nn.Sequential(*layers)

#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             img_size=None,
#             patch_size=patch_size,
#             in_chans=in_chans,
#             embed_dim=embed_dim[0],
#             norm_layer=norm_layer,
#             output_fmt='NHWC',
#         )

#         self.apply(self._init_weights)
#         for bly in self.layers:
#             bly._init_respostnorm()

#         self._out_feature_strides = {"patch_embed": patch_size}
#         self._out_feature_channels = {"patch_embed": embed_dim[0]}

#         if out_features is not None:
#             # Avoid keeping unused layers in this module. They consume extra memory
#             # and may cause allreduce to fail
#             num_layers = max(
#                 [{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4}.get(f, 0) for f in out_features]
#             )
#             self.layers = self.layers[:num_layers]
#         else:
#             num_layers = len(self.layers)
        
#         self.layer_names = tuple(["stage" + str(i + 2) for i in range(num_layers)])
#         self._out_feature_strides.update({name: patch_size * (2 ** i) for i, name in enumerate(self.layer_names)})
#         self._out_feature_channels.update({name: embed_dim[i] for i, name in enumerate(self.layer_names)})

#         if out_features is None:
#             if num_classes is not None:
#                 out_features = ["linear"]
#             else:
#                 out_features = ["stage" + str(num_layers)]
#         self._out_features = out_features
#         assert len(self._out_features)
        
#         self._freeze_layers(freeze_at)

#         self._size_divisibility = patch_size * 2 ** (self.num_layers - 1)

#     @classmethod
#     def from_config(cls, cfg, input_shape):
#         return {
#             "img_size": cfg.MODEL.SWIN.IMG_SIZE,
#             "patch_size": cfg.MODEL.SWIN.PATCH_SIZE,
#             "in_chans": input_shape.channels,
#             "global_pool": cfg.MODEL.SWIN.GLOBAL_POOL,
#             "embed_dim": cfg.MODEL.SWIN.EMBED_DIM,
#             "depths": cfg.MODEL.SWIN.DEPTHS,
#             "num_heads": cfg.MODEL.SWIN.NUM_HEADS,
#             "window_size": cfg.MODEL.SWIN.WINDOW_SIZE,
#             "mlp_ratio": cfg.MODEL.SWIN.MLP_RATIO,
#             "qkv_bias": cfg.MODEL.SWIN.QKV_BIAS,
#             "drop_rate": cfg.MODEL.SWIN.DROP_RATE,
#             "proj_drop_rate": cfg.MODEL.SWIN.PROJ_DROP_RATE,
#             "attn_drop_rate": cfg.MODEL.SWIN.ATTN_DROP_RATE,
#             "drop_path_rate": cfg.MODEL.SWIN.DROP_PATH_RATE,
#             "act_layer": cfg.MODEL.SWIN.ACT_LAYER,
#             "norm_layer": cfg.MODEL.SWIN.NORM_LAYER,
#             "pretrained_window_sizes": cfg.MODEL.SWIN.PRETRAINED_WINDOW_SIZES,
#             "out_features": cfg.MODEL.SWIN.OUT_FEATURES,
#             "freeze_at": cfg.MODEL.BACKBONE.FREEZE_AT,
#         }
    
#     def forward(self, x):
#         """
#         Args:
#             x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

#         Returns:
#             dict[str->Tensor]: names and the corresponding features
#         """
#         assert x.dim() == 4, f"PVTv2 takes an input of shape (N, C, H, W). Got {x.shape} instead!"
    
#         outputs = {}

#         x = self.patch_embed(x)
#         if "patch_embed" in self._out_features:
#             outputs["patch_embed"] = x
#         for name, layer in zip(self.layer_names, self.layers):
#             x = layer(x)
#             if name in self._out_features:
#                 outputs[name] = x.permute(0, 3, 1, 2).contiguous() # added like this, better elsewhere

#         if self.num_classes:
#             if "linear" in self._out_features:
#                 outputs["linear"] = self.forward_head(x)
#         return outputs
    
#     def output_shape(self):
#         return {
#             name: ShapeSpec(
#                 channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
#             )
#             for name in self._out_features
#         }

#     def _freeze_layers(self, freeze_at=0):
#         if freeze_at >= 1:
#             for param in self.patch_embed.parameters():
#                 param.requires_grad = False
        
#         module_names = ["downsample", "blocks", "norm"]
#         for idx, layer in enumerate(self.layers, start=2):
#             if freeze_at >= idx:
#                 for module_name in module_names:
#                     module = getattr(layer, module_name)
#                     if module is not None:
#                         for param in module.parameters():
#                             param.requires_grad = False

#     @property
#     def size_divisibility(self) -> int:
#         return self._size_divisibility


# class CrossWindowAttention(WindowAttention):
#     def forward(self,
#                 x: torch.Tensor, y: torch.Tensor,
#                 mask_query: Optional[torch.Tensor] = None,
#                 mask_support: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
#         # B_, N, C = x.shape
#         # if x.shape[0] == 1:
#         reverse = False
#         xs = [x, y]
#         masks = [mask_query, mask_support]
#         # elif y.shape[0] == 1:
#         #     reverse = True
#         #     xs = [y, x]
#         #     masks = [mask_support, mask_query]
#         # else:
#         #     raise ValueError('Either the query or support tensor should have a batch size of 1')
#         shapes = [x.shape for x in xs]

#         qkv_bias = None
#         if self.q_bias is not None:
#             qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))

#         qs, ks, vs = [], [], []
#         for x, (B_, N, C) in zip(xs, shapes):
#             qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
#             qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#             qs.append(qkv[0])
#             ks.append(qkv[1])
#             vs.append(qkv[2])

#         relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
#         relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

#         xs = []
#         ks = reversed(ks)
#         vs = reversed(vs)
#         for q, k_other, v_other, mask, (B_, N, C) in zip(qs, ks, vs, masks, shapes):
#             raise Exception(q.shape, k_other.transpose(-2, -1).shape)
#             # cosine attention
#             attn = (F.normalize(q, dim=-1) @ F.normalize(k_other, dim=-1).transpose(-2, -1))
#             raise Exception(attn.shape)
#             logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
#             attn = attn * logit_scale
#             attn = attn + relative_position_bias.unsqueeze(0)

#             if mask is not None:
#                 num_win = mask.shape[0]
#                 attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#                 attn = attn.view(-1, self.num_heads, N, N)
            
#             attn = self.softmax(attn)
#             attn = self.attn_drop(attn)
            
#             x = (attn @ v_other).transpose(1, 2).reshape(B_, N, C)
#             x = self.proj(x)
#             x = self.proj_drop(x)
#             xs.append(x)
        
#         if reverse:
#             return reversed(xs)
#         return tuple(xs)


# class FsodSwinTransformerV2Block(SwinTransformerV2Block):
#     def __init__(
#             self,
#             dim: int,
#             num_heads: int,
#             qkv_bias: bool = True,
#             proj_drop: float = 0.,
#             attn_drop: float = 0.,
#             pretrained_window_size: _int_or_tuple_2_t = 0,
#             **kwargs
#             ):
#         super().__init__(
#             dim=dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             proj_drop=proj_drop,
#             attn_drop=attn_drop,
#             pretrained_window_size=pretrained_window_size,
#             **kwargs)

#         self.attn = CrossWindowAttention(
#             dim,
#             window_size=to_2tuple(self.window_size),
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             pretrained_window_size=to_2tuple(pretrained_window_size),
#         )

#     def _attn(self, x: torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor]:
#         xs = [x, y]
#         xs_windows, attn_masks = [], []
#         for x in xs:
#             B, H, W, C = x.shape

#             # cyclic shift [+ attention mask (different way of same timm implementation)]
#             has_shift = any(self.shift_size)
#             if has_shift:
#                 shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
#                 attn_mask = self._create_attn_mask(x)

#             else:
#                 shifted_x = x
#                 attn_mask = None

#             attn_masks.append(attn_mask)

#             # partition windows
#             x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
#             x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
#             xs_windows.append(x_windows)

#         # W-MSA/SW-MSA
#         attn_windows = self.attn(xs_windows[0], xs_windows[1],
#                                  mask_query=attn_masks[0],
#                                  mask_support=attn_masks[1])
#         # nW*B, window_size*window_size, C

#         outs = []
#         for attn_window, x in zip(attn_windows, xs):
#             B, H, W, C = x.shape

#             # merge windows
#             attn_window = attn_window.view(-1, self.window_size[0], self.window_size[1], C)
#             shifted_x = window_reverse(attn_window, self.window_size, (H, W))  # B H' W' C

#             # reverse cyclic shift
#             if has_shift:
#                 x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
#             else:
#                 x = shifted_x
#             outs.append(x)
#         return tuple(outs)
    
#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor]:
#         xs = [x, y]
#         del x, y

#         shapes = [x.shape for x in xs]
#         for idx, (x, (B, H, W, C)) in enumerate(zip(xs, shapes)):
#             # Check if the height and width are divisible by the window size, if not pad the input
#             pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
#             pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
#             if pad_h > 0 or pad_w > 0:
#                 xs[idx] = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

#         xs = self._attn(*xs)
#         outs = []

#         for idx, (x, (B, H, W, C)) in enumerate(zip(xs, shapes)):
#             x = x + self.drop_path1(self.norm1(x))

#             x = x[:, :H, :W, :].contiguous()  # Remove padding if added

#             x = x.reshape(B, -1, C)
#             x = x + self.drop_path2(self.norm2(self.mlp(x)))
#             x = x.reshape(B, H, W, C)
#             outs.append(x)
#         return tuple(outs)


# class FsodSwinTransformerV2Stage(SwinTransformerV2Stage):
#     def __init__(
#             self,
#             out_dim: int,
#             input_resolution: _int_or_tuple_2_t,
#             depth: int,
#             num_heads: int,
#             window_size: _int_or_tuple_2_t,
#             mlp_ratio: float = 4.,
#             qkv_bias: bool = True,
#             proj_drop: float = 0.,
#             attn_drop: float = 0.,
#             drop_path: float = 0.,
#             act_layer: Union[str, Callable] = 'gelu',
#             norm_layer: nn.Module = nn.LayerNorm,
#             pretrained_window_size: _int_or_tuple_2_t = 0,
#             **kwargs
#     ):
#         super().__init__(
#             out_dim=out_dim,
#             input_resolution=input_resolution,
#             depth=depth,
#             num_heads=num_heads,
#             window_size=window_size,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             proj_drop=proj_drop,
#             attn_drop=attn_drop,
#             drop_path=drop_path,
#             act_layer=act_layer,
#             norm_layer=norm_layer,
#             pretrained_window_size=pretrained_window_size,
#             **kwargs,
#         )
#         window_size = to_2tuple(window_size)
#         shift_size = tuple([w // 2 for w in window_size])

#         # build blocks
#         self.blocks = nn.ModuleList([
#             FsodSwinTransformerV2Block(
#                 dim=out_dim,
#                 input_resolution=input_resolution,
#                 num_heads=num_heads,
#                 window_size=window_size,
#                 shift_size=0 if (i % 2 == 0) else shift_size,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 proj_drop=proj_drop,
#                 attn_drop=attn_drop,
#                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                 act_layer=act_layer,
#                 norm_layer=norm_layer,
#                 pretrained_window_size=pretrained_window_size,
#             )
#             for i in range(depth)])
    
#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor]:
#         x = self.downsample(x)
#         y = self.downsample(y)

#         for blk in self.blocks:
#             if self.grad_checkpointing and not torch.jit.is_scripting():
#                 x, y = checkpoint.checkpoint(blk, x, y)
#             else:
#                 x, y = blk(x, y)
#         x = self.norm(x) # added
#         y = self.norm(y)
#         return x, y
    

# @BACKBONE_REGISTRY.register()
# class FsodSwinTransformerV2(SwinTransformerV2):
#     @configurable
#     def __init__(
#             self,
#             img_size: _int_or_tuple_2_t = 224,
#             patch_size: int = 4,
#             embed_dim: int = 96,
#             depths: Tuple[int, ...] = (2, 2, 6, 2),
#             num_heads: Tuple[int, ...] = (3, 6, 12, 24),
#             window_size: _int_or_tuple_2_t = 7,
#             mlp_ratio: float = 4.,
#             qkv_bias: bool = True,
#             proj_drop_rate: float = 0.,
#             attn_drop_rate: float = 0.,
#             drop_path_rate: float = 0.1,
#             act_layer: Union[str, Callable] = 'gelu',
#             norm_layer: Union[str, Callable] = nn.LayerNorm,
#             pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0, 0),
#             out_features=None,
#             freeze_at=0,
#             **kwargs,
#     ):
        
#         if isinstance(norm_layer, str):
#             norm_layer = partial(get_norm, norm_layer)

#         super().__init__(
#             img_size=img_size,
#             patch_size=patch_size,
#             embed_dim=embed_dim,
#             depths=depths,
#             num_heads=num_heads,
#             window_size=window_size,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             proj_drop_rate=proj_drop_rate,
#             attn_drop_rate=attn_drop_rate,
#             drop_path_rate=drop_path_rate,
#             act_layer=act_layer,
#             norm_layer=norm_layer,
#             pretrained_window_sizes=pretrained_window_sizes,
#             out_features=out_features,
#             freeze_at=freeze_at,
#             **kwargs,
#         )
#         if not isinstance(embed_dim, (tuple, list)):
#             embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

#         dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
#         layers = []
#         in_dim = embed_dim[0]
#         scale = 1
#         grid_size = tuple([s // p for s, p in zip(to_2tuple(img_size), to_2tuple(patch_size))])
#         for i in range(self.num_layers):
#             out_dim = embed_dim[i]
#             layers += [FsodSwinTransformerV2Stage(
#                 dim=in_dim,
#                 out_dim=out_dim,
#                 input_resolution=(
#                     grid_size[0] // scale,
#                     grid_size[1] // scale),
#                 depth=depths[i],
#                 downsample=i > 0,
#                 num_heads=num_heads[i],
#                 window_size=window_size,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 proj_drop=proj_drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_path=dpr[i],
#                 act_layer=act_layer,
#                 norm_layer=norm_layer,
#                 pretrained_window_size=pretrained_window_sizes[i],
#             )]
#             in_dim = out_dim
#             if i > 0:
#                 scale *= 2

#         self.layers = nn.Sequential(*layers)

#         self.apply(self._init_weights)
#         for bly in self.layers:
#             bly._init_respostnorm()

#         if out_features is not None:
#             # Avoid keeping unused layers in this module. They consume extra memory
#             # and may cause allreduce to fail
#             num_layers = max(
#                 [{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4}.get(f, 0) for f in out_features]
#             )
#             self.layers = self.layers[:num_layers]
#         else:
#             num_layers = len(self.layers)
        
#         self._freeze_layers(freeze_at)
    
#     @classmethod
#     def from_config(cls, cfg, input_shape):
#         return super().from_config(cfg, input_shape)

#     def forward(self, x, y):
#         """
#         Args:
#             x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

#         Returns:
#             dict[str->Tensor]: names and the corresponding features
#         """
#         assert x.dim() == 4, f"SWINv2 takes an input of shape (N, C, H, W). Got {x.shape} instead!"
    
#         outputs_query = {}
#         outputs_support = {}

#         x = self.patch_embed(x)
#         y = self.patch_embed(y)
#         if "patch_embed" in self._out_features:
#             outputs_query["patch_embed"] = x
#             outputs_support["patch_embed"] = y
#         for name, layer in zip(self.layer_names, self.layers):
#             x, y = layer(x, y)
#             if name in self._out_features:
#                 outputs_query[name] = x.permute(0, 3, 1, 2).contiguous() # added like this, better elsewhere
#                 outputs_support[name] = y.permute(0, 3, 1, 2).contiguous()

#         if self.num_classes:
#             if "linear" in self._out_features:
#                 outputs_query["linear"] = self.forward_head(x)

#         return outputs_query, outputs_support
