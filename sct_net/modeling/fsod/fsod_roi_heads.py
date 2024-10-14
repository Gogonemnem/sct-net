from typing import Dict, List, Optional, Tuple, Union

import torch
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads

from .fsod_fast_rcnn import FsodFastRCNNOutputLayers

@ROI_HEADS_REGISTRY.register()
class FsodStandardROIHeads(StandardROIHeads):
    """
    Standard ROIHeads with few-shot object detection (Fsod) functionality.
    """
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        shape = ret['box_head'].output_shape
        ret["box_predictor"] = FsodFastRCNNOutputLayers(
            cfg, shape
        )
        return ret

    def forward(
        self,
        images: ImageList,
        query_features_dict: Dict[int, Dict[str, torch.Tensor]],
        support_features_dict: Dict[int, Dict[str, torch.Tensor]],
        support_boxes_dict: Dict[int, List[Boxes]],
        proposals_dict: Dict[int, List[Instances]],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            for cls_id, proposals in proposals_dict.items():
                proposals_dict[cls_id] = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(query_features_dict, support_features_dict, support_boxes_dict, proposals_dict)
            # TODO: implement mask & keypoint
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            # losses.update(self._forward_mask(query_features_dict, proposals_dict))
            # losses.update(self._forward_keypoint(query_features_dict, proposals_dict))
            return proposals_dict, losses
        else:
            pred_instances = self._forward_box(query_features_dict, support_features_dict, support_boxes_dict, proposals_dict)
            # TODO: implement mask & keypoint
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            # pred_instances = self.forward_with_given_boxes(query_features_dict, pred_instances)
            return pred_instances, {}

    def _forward_box(self, query_features_dict, support_features_dict, support_boxes_dict, proposals_dict):
        """
        See :meth:`ROIHeads.forward`.
        """
        combined_proposals = []
        combined_class_logits = []
        combined_proposal_deltas = []

        for idx, cls_id in enumerate(query_features_dict.keys()):
            query_features = [query_features_dict[cls_id][f] for f in self.box_in_features]
            proposal_boxes = [x.proposal_boxes for x in proposals_dict[cls_id]]
            query_box_features = self.box_pooler(query_features, proposal_boxes)

            support_features = [support_features_dict[cls_id][f] for f in self.box_in_features]
            support_box_features = self.box_pooler(support_features, support_boxes_dict[cls_id])
            support_box_features = support_box_features.mean(0, True)

            features = self.box_head(query_box_features, support_box_features)
            
            if isinstance(features, Union[List, Tuple]):
                query_features, support_features = features
                class_logits, proposal_deltas = self.box_predictor(query_features, support_features)
            else:
                class_logits, proposal_deltas = self.box_predictor(features)

            if idx != 0 and self.training: # only first class is positive
                for item in proposals_dict[cls_id]:
                    item.gt_classes = torch.full_like(item.gt_classes, 1)

            combined_class_logits.append(class_logits)
            combined_proposal_deltas.append(proposal_deltas)
            combined_proposals.extend(proposals_dict[cls_id])

        combined_class_logits = torch.cat(combined_class_logits)
        combined_proposal_deltas = torch.cat(combined_proposal_deltas)
        proposals = [Instances.cat(combined_proposals)]

        predictions = combined_class_logits, combined_proposal_deltas
        
        if self.training:
            del query_features_dict
            losses = self.box_predictor.losses(predictions, proposals)
            
            # TODO: copied from StandardROIHeads.forward, probably does not work
            # if self.train_on_pred_boxes:
            #     with torch.no_grad():
            #         pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
            #             predictions, proposals
            #         )
            #         for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
            #             proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_cls = list(proposals_dict.keys())
            pred_instances, _ = self.box_predictor.inference(pred_cls, predictions, proposals) # numclasses
            return pred_instances
