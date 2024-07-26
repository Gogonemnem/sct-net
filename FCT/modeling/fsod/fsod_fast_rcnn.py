from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import cat, cross_entropy
from detectron2.structures import Instances
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, _log_classification_stats, fast_rcnn_inference


__all__ = ["FsodFastRCNNOutputLayers"]


class FsodFastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(self, num_proposals, **kwargs):
        super().__init__(**kwargs)
        self.num_proposals = num_proposals
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['num_proposals'] = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        return ret

    def forward(self, x, y=None):
        x = super().forward(x)
        return x

    def losses(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions

        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses =  {
            "loss_cls": self.softmax_cross_entropy_loss(scores, gt_classes),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def softmax_cross_entropy_loss(self, scores: torch.Tensor, gt_classes: torch.Tensor):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        n_instances_per_class = self.num_proposals

        # TODO: use configs to figure it out
        background_ratio = 2
        negative_ratio = 1

        fg_inds = (gt_classes == 0).nonzero().squeeze(-1)
        bg_inds = (gt_classes == 1).nonzero().squeeze(-1)

        n_positives_foreground = fg_inds.shape[0]
        if n_positives_foreground == 0:
            n_positives_background = 1
            n_negatives = 1
        else:
            n_positives_background = n_positives_foreground * background_ratio
            n_negatives = n_positives_foreground * negative_ratio

            if  n_positives_background + n_positives_foreground > n_instances_per_class:
                n_positives_background = n_instances_per_class - n_positives_foreground

                # anchor the ratio to positive background, same logic as in the original implementation
                # maybe not necessary as the ratio should be anchored to positive foreground?
                # n_negatives = n_positives_background

        normed_scores = F.softmax(scores, dim=1)
        bg_cls_scores = normed_scores[bg_inds, 0]

        _, sorted_bg_inds = torch.sort(bg_cls_scores, descending=True)
        sorted_bg_inds        = bg_inds[sorted_bg_inds]
        pos_bg_inds = sorted_bg_inds[:n_positives_background]
        neg_bg_inds = sorted_bg_inds[n_instances_per_class - n_positives_foreground:n_instances_per_class - n_positives_foreground + n_negatives]

        topk_inds = torch.cat([fg_inds, pos_bg_inds, neg_bg_inds], dim=0)

        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores[topk_inds], gt_classes[topk_inds])
        else:
            loss_cls = cross_entropy(scores[topk_inds], gt_classes[topk_inds], reduction="mean")
        
        return loss_cls

    def inference(self, pred_cls: List[int], predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            pred_cls (List[int]): list of class ids for each class
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        num_classes = len(pred_cls)
        full_scores = []
        full_boxes = []
        for idx in range(len(proposals)):
            score = scores[idx]
            class_score = score[:, :-1].reshape(num_classes, -1).permute(1, 0)
            background_holder = torch.zeros(score.size(0) // num_classes, 1, device=score.device)
            full_score = torch.cat([class_score, background_holder], dim=1)
            full_scores.append(full_score)

            box = boxes[idx]
            class_box = box.reshape(num_classes, -1, 4).permute(1, 0, 2).reshape(-1, num_classes*4)
            full_boxes.append(class_box)
        
        instances, kept_indices = fast_rcnn_inference(
            full_boxes,
            full_scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

        # Create a tensor mapping from index to original class values
        class_map = torch.tensor(pred_cls).to(instances[0].pred_classes.device)

        for instance in instances:
            instance.pred_classes = class_map[instance.pred_classes]

        return instances, kept_indices
