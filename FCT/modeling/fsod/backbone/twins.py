from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from timm.models.twins import (
    Twins as _Twins,
    Block as _Block,
    LocallyGroupedAttn as _LocallyGroupedAttn,
    GlobalSubSampleAttn as _GlobalSubSampleAttn,
)

from timm.models.vision_transformer import Attention as _Attention

from detectron2.config import configurable
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec


@BACKBONE_REGISTRY.register()
class Twins(_Twins, Backbone):
    @configurable
    def __init__(self, *, patch_size=4, num_classes=None, out_features=None, freeze_at=0, **kwargs):
        super().__init__(patch_size=patch_size, num_classes=num_classes if num_classes is not None else 0, **kwargs)

        self._out_feature_strides = {"patch_embed": patch_size}
        self._out_feature_channels = {"patch_embed": self.embed_dims[0]}

        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            stage_names = {f"stage{idx+1}": idx for idx in range(1, len(self.blocks) + 1)}
            num_layers = max(
                [stage_names.get(f, 0) for f in out_features]
            )
            self.patch_embeds = self.patch_embeds[:num_layers]
            self.pos_drops = self.pos_drops[:num_layers]
            self.blocks = self.blocks[:num_layers]
            self.pos_block = self.pos_block[:num_layers]
        else:
            num_layers = len(self.blocks)
        
        self.layer_names = tuple(["stage" + str(i + 2) for i in range(num_layers)])
        self._out_feature_strides.update({name: patch_size * (2 ** i) for i, name in enumerate(self.layer_names)})
        self._out_feature_channels.update({name: self.embed_dims[i] for i, name in enumerate(self.layer_names)})

        if out_features is None:
            if num_classes is not None:
                out_features = ["linear"]
            else:
                out_features = ["stage" + str(num_layers)]
        self._out_features = out_features
        assert len(self._out_features)
        
        self._freeze_layers(freeze_at)
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        # norm_layer = cfg.MODEL.TWINS.NORM_LAYER
        # norm_layer = partial(get_norm, norm_layer)
        norm_layer = nn.LayerNorm
                    
        return {
            'img_size': cfg.MODEL.TWINS.IMG_SIZE,
            'patch_size': cfg.MODEL.TWINS.PATCH_SIZE,
            'in_chans': input_shape.channels,
            'num_classes': None,
            'global_pool': cfg.MODEL.TWINS.GLOBAL_POOL,
            'embed_dims': cfg.MODEL.TWINS.EMBED_DIMS,
            'num_heads': cfg.MODEL.TWINS.NUM_HEADS,
            'mlp_ratios': cfg.MODEL.TWINS.MLP_RATIOS,
            'depths': cfg.MODEL.TWINS.DEPTHS,
            'sr_ratios': cfg.MODEL.TWINS.SR_RATIOS,
            'wss': cfg.MODEL.TWINS.WSS,
            'drop_rate': cfg.MODEL.TWINS.DROP_RATE,
            'pos_drop_rate': cfg.MODEL.TWINS.POS_DROP_RATE,
            'proj_drop_rate': cfg.MODEL.TWINS.PROJ_DROP_RATE,
            'attn_drop_rate': cfg.MODEL.TWINS.ATTN_DROP_RATE,
            'drop_path_rate': cfg.MODEL.TWINS.DROP_PATH_RATE,
            'norm_layer': norm_layer,
            'block_cls': _Block, # we allow just one block class for now,
            'out_features': cfg.MODEL.TWINS.OUT_FEATURES,
            'freeze_at': cfg.MODEL.BACKBONE.FREEZE_AT,
        }

    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"Twins takes an input of shape (N, C, H, W). Got {x.shape} instead!"
    
        outputs = {}

        B, _, height, width = x.shape
        for i, (name, embed, drop, blocks, pos_blk) in enumerate(zip(
            self.layer_names, self.patch_embeds, self.pos_drops, self.blocks, self.pos_block
            )):
            x, size = embed(x)
            x = drop(x)
            if i == 0 and "patch_embed" in self._out_features:
                outputs["patch_embed"] = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                
                if j == 0:
                    x = pos_blk(x, size)  # PEG here
            
            if name == 'stage5': # last stage
                x = self.norm(x)
                if self.num_classes and "linear" in self._out_features:
                    outputs["linear"] = self.forward_head(x)

            x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            if name in self._out_features:
                outputs[name] = x

        return outputs
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def _freeze_layers(self, freeze_at):
        if freeze_at >= 1:
            for param in self.patch_embeds[0].parameters():
                param.requires_grad = False
        
        # TODO: variable name: layer is not great
        module_names = ["patch_embeds", "pos_drops", "blocks", "pos_block"]
        for module_name in module_names:
            module = getattr(self, module_name)
            for idx in range(len(self.layer_names)):

                layer = module[idx]
                if freeze_at >= idx:
                    if layer is not None:
                        for param in layer.parameters():
                            param.requires_grad = False


class Attention(_Attention):
    def forward(self, x, y):
        if x.shape[0] == 1:
            reverse = False
            xs = [x, y]
        elif y.shape[0] == 1:
            reverse = True
            xs = [y, x]
        else:
            raise ValueError('Either the query or support tensor should have a batch size of 1')
        del x, y
        shapes = [x.shape for x in xs]

        qs, ks, vs = [], [], []
        for x, shape in zip(xs, shapes):
            B, N, C = x.shape

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            
            qs.append(q)
            ks.append(k)
            vs.append(v)

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
        for shape, q, k_cat, v_cat in zip(shapes, qs, k_cats, v_cats):
            B, N, C = shape        
            if self.fused_attn:
                x = F.scaled_dot_product_attention(q, k_cat, v_cat, dropout_p=self.attn_drop.p if self.training else 0.)
            else:
                q = q * self.scale
                attn = q @ k_cat.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v_cat

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            outputs.append(x)
        
        if reverse:
            return reversed(outputs)
        return tuple(outputs)


class LocallyGroupedAttn(_LocallyGroupedAttn):
    def forward(self, x, y, size_query, size_support):
        # forward_mask not implemented
        if size_query != size_support:
            if size_query[0] * size_query[1] > size_support[0] * size_support[1]:
                y = y.view(-1, size_support[0], size_support[1], y.shape[-1]).permute(0, 3, 1, 2)
                y = F.interpolate(y, size=size_query, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                y = y.view(-1, size_query[0] * size_query[1], y.shape[-1])
                resize_back = True
            else:
                x = x.view(-1, size_query[0], size_query[1], x.shape[-1]).permute(0, 3, 1, 2)
                x = F.interpolate(x, size=size_support, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                x = x.view(-1, size_support[0] * size_support[1], x.shape[-1])
                resize_back = True
        else:
            resize_back = False

        if x.shape[0] == 1:
            reverse = False
            xs = [x, y]
            og_sizes = [size_query, size_support]
            sizes = [size_query, size_query]
        elif y.shape[0] == 1:
            reverse = True
            xs = [y, x]
            og_sizes = [size_support, size_query]
            sizes = [size_support, size_support]
        else:
            raise ValueError('Either the query or support tensor should have a batch size of 1')
        del x, y, size_query, size_support
        shapes = [x.shape for x in xs]

        qs, ks, vs = [], [], []
        padded_sizes = []
        for x, size in zip(xs, sizes):
            B, N, C = x.shape
            H, W = size
            x = x.view(B, H, W, C)
            pad_l = pad_t = 0
            pad_r = (self.ws - W % self.ws) % self.ws
            pad_b = (self.ws - H % self.ws) % self.ws
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x.shape
            _h, _w = Hp // self.ws, Wp // self.ws
            x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
            qkv = self.qkv(x).reshape(
                B, _h * _w, self.ws * self.ws, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
            q, k, v = qkv.unbind(0)
            qs.append(q)
            ks.append(k)
            vs.append(v)
            padded_sizes.append((_h, _w))
        
        k_expanded = ks[0].expand(ks[1].shape[0], -1, -1, -1, -1)
        k_mean = ks[1].mean(dim=0, keepdim=True)

        v_expanded = vs[0].expand(vs[1].shape[0], -1, -1, -1, -1)
        v_mean = vs[1].mean(dim=0, keepdim=True)

        if not reverse:
            k_cats = [torch.cat((ks[0], k_mean), dim=3), torch.cat((k_expanded, ks[1]), dim=3)]
            v_cats = [torch.cat((vs[0], v_mean), dim=3), torch.cat((v_expanded, vs[1]), dim=3)]
        else:
            k_cats = [torch.cat((k_mean, ks[0]), dim=3), torch.cat((ks[1], k_expanded), dim=3)]
            v_cats = [torch.cat((v_mean, vs[0]), dim=3), torch.cat((vs[1], v_expanded), dim=3)]

        outputs = []
        for shape, size, q, k_cat, v_cat, padded_size in zip(shapes, sizes, qs, k_cats, v_cats, padded_sizes):
            H, W = size
            B, N, C = shape
            _h, _w = padded_size

            if self.fused_attn:
                x = F.scaled_dot_product_attention(q, k_cat, v_cat, dropout_p=self.attn_drop.p if self.training else 0.)
            else:
                q = q * self.scale
                attn = q @ k_cat.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v_cat

            x = x.transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
            x = x.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
            if size != padded_size:
                x = x[:, :H, :W, :].contiguous()
            x = x.reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            outputs.append(x)

        if resize_back:
            outputs[1] = outputs[1].view(-1, og_sizes[0][0], og_sizes[0][1], outputs[1].shape[-1]).permute(0, 3, 1, 2)
            outputs[1] = F.interpolate(outputs[1], size=og_sizes[1], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            outputs[1] = outputs[1].view(-1, og_sizes[1][0] * og_sizes[1][1], outputs[1].shape[-1])

        outputs = reversed(outputs) if reverse else outputs
        return tuple(outputs)
    

class GlobalSubSampleAttn(_GlobalSubSampleAttn):
    def forward(self, x, y, size_query, size_support):
        if x.shape[0] == 1:
            reverse = False
            xs = [x, y]
            sizes = [size_query, size_support]
        elif y.shape[0] == 1:
            reverse = True
            xs = [y, x]
            sizes = [size_support, size_query]
        else:
            raise ValueError('Either the query or support tensor should have a batch size of 1')
        del x, y, size_query, size_support
        shapes = [x.shape for x in xs]

        qs, ks, vs = [], [], []
        for x, size in zip(xs, sizes):
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            if self.sr is not None:
                x = x.permute(0, 2, 1).reshape(B, C, *size)
                x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
                x = self.norm(x)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)

            qs.append(q)
            ks.append(k)
            vs.append(v)

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
        for shape, q, k_cat, v_cat in zip(shapes, qs, k_cats, v_cats):
            B, N, C = shape        
            if self.fused_attn:
                x = F.scaled_dot_product_attention(q, k_cat, v_cat, dropout_p=self.attn_drop.p if self.training else 0.)
            else:
                q = q * self.scale
                attn = q @ k_cat.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v_cat

            x = x.transpose(1, 2).reshape(B, N, C)
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
            proj_drop=0.,
            attn_drop=0.,
            norm_layer=nn.LayerNorm,
            sr_ratio=1,
            ws=None,
            **kwargs,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            sr_ratio=sr_ratio,
            ws=ws,
            **kwargs,
            )
        self.norm1 = norm_layer(dim)
        if ws is None:
            self.attn = Attention(dim, num_heads, False, None, attn_drop, proj_drop)
        elif ws == 1:
            self.attn = GlobalSubSampleAttn(dim, num_heads, attn_drop, proj_drop, sr_ratio)
        else:
            self.attn = LocallyGroupedAttn(dim, num_heads, attn_drop, proj_drop, ws)
        
    def forward(self, x, y, size_query, size_support):
        x, y = self.attn(self.norm1(x), self.norm1(y), size_query, size_support)
        x = x + self.drop_path1(x)
        y = y + self.drop_path2(y)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        y = y + self.drop_path2(self.mlp(self.norm2(y)))
        return x, y
    

@BACKBONE_REGISTRY.register()
class FsodTwins(Twins):
    @configurable
    def __init__(
        self,
        *,
        patch_size=4,
        embed_dims=(64, 128, 256, 512),
        num_heads=(1, 2, 4, 8),
        mlp_ratios=(4, 4, 4, 4),
        depths=(3, 4, 6, 3),
        sr_ratios=(8, 4, 2, 1),
        wss=None,
        proj_drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_cls=Block,
        num_classes=None,
        out_features=None,
        freeze_at=0,
        **kwargs,
    ):
        super().__init__(
            patch_size=patch_size,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            depths=depths,
            sr_ratios=sr_ratios,
            wss=wss,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            block_cls=block_cls,
            num_classes=num_classes,
            out_features=out_features,
            freeze_at=freeze_at,
            **kwargs
            )

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k],
                num_heads=num_heads[k],
                mlp_ratio=mlp_ratios[k],
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[k],
                ws=1 if wss is None or i % 2 == 1 else wss[k]) for i in range(depths[k])],
            )
            self.blocks.append(_block)
            cur += depths[k]

        self._out_feature_strides = {"patch_embed": patch_size}
        self._out_feature_channels = {"patch_embed": self.embed_dims[0]}

        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            stage_names = {f"stage{idx+1}": idx for idx in range(1, len(self.blocks) + 1)}
            num_layers = max(
                [stage_names.get(f, 0) for f in out_features]
            )
            self.patch_embeds = self.patch_embeds[:num_layers]
            self.pos_drops = self.pos_drops[:num_layers]
            self.blocks = self.blocks[:num_layers]
            self.pos_block = self.pos_block[:num_layers]
        else:
            num_layers = len(self.blocks)
        
        self.layer_names = tuple(["stage" + str(i + 2) for i in range(num_layers)])
        self._out_feature_strides.update({name: patch_size * (2 ** i) for i, name in enumerate(self.layer_names)})
        self._out_feature_channels.update({name: self.embed_dims[i] for i, name in enumerate(self.layer_names)})

        if out_features is None:
            if num_classes is not None:
                out_features = ["linear"]
            else:
                out_features = ["stage" + str(num_layers)]
        self._out_features = out_features
        assert len(self._out_features)

        # init weights
        self.apply(self._init_weights)
        self._freeze_layers(freeze_at)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['block_cls'] = Block
        return ret

    def forward(self, x, y):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"Twins takes an input of query shape (N, C, H, W). Got {x.shape} instead!"
        assert y.dim() == 4, f"Twins takes an input of support shape (N, C, H, W). Got {y.shape} instead!"
    
        outputs_query = {}
        outputs_support = {}

        B_x, _, _, _ = x.shape
        B_y, _, _, _ = y.shape
        for i, (name, embed, drop, blocks, pos_blk) in enumerate(zip(
            self.layer_names, self.patch_embeds, self.pos_drops, self.blocks, self.pos_block
            )):
            x, size_query = embed(x)
            x = drop(x)

            y, size_support = embed(y)
            y = drop(y)

            if i == 0 and "patch_embed" in self._out_features:
                outputs_query["patch_embed"] = x.reshape(B_x, *size_query, -1).permute(0, 3, 1, 2).contiguous()
                outputs_support["patch_embed"] = y.reshape(B_y, *size_support, -1).permute(0, 3, 1, 2).contiguous()
            
            for j, blk in enumerate(blocks):
                x, y = blk(x, y, size_query, size_support)
                
                if j == 0:
                    x = pos_blk(x, size_query)  # PEG here
                    y = pos_blk(y, size_support)
            
            if name == 'stage5': # last stage
                x = self.norm(x)
                if self.num_classes and "linear" in self._out_features:
                    outputs_query["linear"] = self.forward_head(x)

            x = x.reshape(B_x, *size_query, -1).permute(0, 3, 1, 2).contiguous()
            y = y.reshape(B_y, *size_support, -1).permute(0, 3, 1, 2).contiguous()
            if name in self._out_features:
                outputs_query[name] = x
                outputs_support[name] = y

        return outputs_query, outputs_support