from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from functools import partial
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.modeling.poolers import ROIPooler

from timm.layers.norm import LayerNorm
from timm.models.layers import Mlp

from .backbone.pvt_v2 import Attention, Block, get_norm


@PROPOSAL_GENERATOR_REGISTRY.register()
class FsodRPN(RPN):
    @configurable
    def __init__(
            self,
            *,
            per_level_roi_poolers,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.per_level_roi_poolers = per_level_roi_poolers

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update(cls._init_per_level_poolers(cfg, input_shape))
        return ret
    
    @classmethod
    def _init_per_level_poolers(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.RPN.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        poolers = {
            f: ROIPooler(
                output_size=pooler_resolution,
                scales=(1.0 / input_shape[f].stride,),
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            ) for f in in_features
        }
        return {"per_level_roi_poolers": poolers}
    
    def per_level_roi_pooling(
        self,
        features: Dict[str, torch.Tensor],
        boxes: List[Boxes]
    ):
        box_features = {}
        for in_feature in self.in_features:
            level_features = [features[in_feature]]
            pooler = self.per_level_roi_poolers[in_feature]
            box_features[in_feature] = pooler(level_features, boxes)
        return box_features

    def forward(
        self,
        images: ImageList,
        features: Dict[int, Dict[str, torch.Tensor]],
        gt_instances: Optional[Instances] = None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        proposals_dict = {}

        combined_pred_objectness_logits = defaultdict(list)
        combined_pred_anchor_deltas = defaultdict(list)

        combined_gt_boxes = []
        combined_gt_labels = []

        for idx, cls_id in enumerate(features.keys()):
            cls_id_features = [features[cls_id][f] for f in self.in_features]
            
            if idx == 0:
                # same anchor generator, so all anchors are same
                anchors = self.anchor_generator(cls_id_features)
                
                if self.training:
                    assert gt_instances is not None, "RPN requires gt_instances in training!"
                    gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            

            pred_objectness_logits, pred_anchor_deltas = self.rpn_head(cls_id_features)
            # Transpose the Hi*Wi*A dimension to the middle:
            pred_objectness_logits = [
                # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
                score.permute(0, 2, 3, 1).flatten(1)
                for score in pred_objectness_logits
            ]
            pred_anchor_deltas = [
                # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
                x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .flatten(1, -2)
                for x in pred_anchor_deltas
            ]

            proposals_dict[cls_id] = self.predict_proposals(
                anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
            )

            if not self.training:
                continue

            # Adjust labels for negative samples
            cls_id_gt_labels = deepcopy(gt_labels)
            if idx != 0: # only first class is positive
                for item in cls_id_gt_labels:
                    item[item == 1] = 0  # Adjust labels for negative samples

            for idx in range(len(self.in_features)):
                combined_pred_objectness_logits[idx].append(pred_objectness_logits[idx])
                combined_pred_anchor_deltas[idx].append(pred_anchor_deltas[idx])

            combined_gt_boxes.extend(gt_boxes)
            combined_gt_labels.extend(cls_id_gt_labels)

        if self.training:
            outputs_pred_objectness_logits = [torch.cat(logit) for logit in combined_pred_objectness_logits.values()]
            outputs_pred_anchor_deltas = [torch.cat(delta) for delta in combined_pred_anchor_deltas.values()]
            outputs_gt_boxes = combined_gt_boxes
            outputs_gt_labels = combined_gt_labels
            losses = self.losses(
                anchors, outputs_pred_objectness_logits, outputs_gt_labels, outputs_pred_anchor_deltas, outputs_gt_boxes
            )
        else:
            losses = {}
        return proposals_dict, losses

@PROPOSAL_GENERATOR_REGISTRY.register()
class AttentionRPN(FsodRPN):
    def forward(
        self,
        images: ImageList,
        query_features: Dict[int, Dict[str, torch.Tensor]],
        support_features: Dict[int, Dict[str, torch.Tensor]],
        support_boxes: Dict[int, List[Boxes]],
        gt_instances: Optional[List[Instances]] = None,
    ):
        rpn_features_dict = {}
        for cls_id in query_features.keys():
            support_box_features = self.per_level_roi_pooling(support_features[cls_id], support_boxes[cls_id])
            support_box_features = {in_feature: features.mean(dim=[0, 2, 3], keepdim=True) for in_feature, features in support_box_features.items()}
            rpn_features_dict[cls_id] = {
                in_feature: F.conv2d(
                    query_features[cls_id][in_feature],
                    support_box_features[in_feature].permute(1,0,2,3),
                    groups=query_features[cls_id][in_feature].shape[1]
                    ) for in_feature in support_box_features.keys()
                }

        return super().forward(images, rpn_features_dict, gt_instances) # standard rpn
    

@PROPOSAL_GENERATOR_REGISTRY.register()
class CrossScalesRPN(FsodRPN):
    @configurable
    def __init__(
            self,
            block,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.block = block

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        
        # All inputs must have same channels: take the first input shape
        first_input_shape = input_shape[cfg.MODEL.RPN.IN_FEATURES[0]]

        ret['block'] = CrossScaleBlock(
            dim=first_input_shape.channels,
            num_heads=cfg.MODEL.PVT.NUM_HEADS[-1],
            mlp_ratio=cfg.MODEL.PVT.MLP_RATIOS[-1],
            qkv_bias=cfg.MODEL.PVT.QKV_BIAS,
            proj_drop=cfg.MODEL.PVT.PROJ_DROP_RATE,
            attn_drop=cfg.MODEL.PVT.ATTN_DROP_RATE,
            drop_path=cfg.MODEL.PVT.DROP_PATH_RATE,
            norm_layer=partial(get_norm, cfg.MODEL.PVT.NORM_LAYER),
        )
        return ret

    def forward(
        self,
        images: ImageList,
        query_features: Dict[str, torch.Tensor],
        support_features: Dict[str, torch.Tensor],
        support_boxes: List[Boxes],
        gt_instances: Optional[List[Instances]] = None,
    ):
        rpn_features_dict = {}
        for cls_id in query_features.keys():
            # same shape for all classes? if so, we can move this outside the loop
            query_feat_sizes = {in_feature: query_features[cls_id][in_feature].shape for in_feature in self.in_features}
            query_flattened = {in_feature: query_features[cls_id][in_feature].flatten(start_dim=2) for in_feature in self.in_features}
            query_flattened_sizes = [query_flattened[in_feature].shape[2] for in_feature in self.in_features]
            query_cat = torch.cat([query_flattened[in_feature] for in_feature in self.in_features], dim=2).permute(0, 2, 1)

            support_box_features = self.per_level_roi_pooling(support_features[cls_id], support_boxes[cls_id])
            # Similar to AAF take the average of the shots (also computation reasons)
            # But different from AttentionRPN this does not pool the spatial dimensions
            support_box_features = {in_feature: features.mean(dim=[0], keepdim=True) for in_feature, features in support_box_features.items()} # 2, 3
            support_cat = torch.cat([support_box_features[in_feature].flatten(start_dim=2) for in_feature in self.in_features], dim=2).permute(0, 2, 1)

            features = self.block(query_cat, support_cat).permute(0, 2, 1)
            features = torch.split(features, query_flattened_sizes, dim=2)

            # Reshape the split tensors back to their original shapes using the stored shapes
            rpn_features = {}
            for i, key in enumerate(self.in_features):
                shape = query_feat_sizes[key]
                batch_size, num_channels, *spatial_dims = shape
                rpn_features[key] = features[i].reshape(batch_size, -1, *spatial_dims)
            rpn_features_dict[cls_id] = rpn_features

        return super().forward(images, rpn_features_dict, gt_instances) # almost standard rpn

class CrossScaleAttention(Attention): ### PVTv2 VARIANT
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__(
            dim,
            num_heads=num_heads,
            sr_ratio=1,
            linear_attn=False,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop
            )
        
    def forward(self, x, y):
        q_x = self.q(x).reshape(*x.shape[:2], self.num_heads, -1).permute(0, 2, 1, 3)
        
        B, N, C = x.shape
        # no pool/sr for this layer
        kv = self.kv(y).reshape(y.shape[0], -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        k_y, v_y = kv.unbind(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q_x, k_y, v_y, dropout_p=self.attn_drop.p if self.training else 0.)
        else:
            q_x = q_x * self.scale
            attn = q_x @ k_y.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v_y

        x = x.transpose(1, 2).view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossScaleBlock(Block):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
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
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.attn = CrossScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x, y):
        x = x + self.drop_path1(self.attn(self.norm1(x), self.norm1(y)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x
