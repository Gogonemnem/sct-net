from collections import defaultdict

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.structures import ImageList, Boxes
from detectron2.utils.events import get_event_storage

__all__ = ["FsodRCNN"]


@META_ARCH_REGISTRY.register()
class FsodRCNN(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        support_way,
        support_shot,
        **kwargs):
        super().__init__(**kwargs)

        self.support_way = support_way
        self.support_shot = support_shot

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update(
            {
                "support_way": cfg.FEWSHOT.SUPPORT_WAY,
                "support_shot": cfg.FEWSHOT.SUPPORT_SHOT,
            }
        )
        return ret

    def visualize_training(self, batched_inputs, proposals_dict):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        input = batched_inputs[0] # only visualize one image in a batch
        img = input["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
        anno_img = v_gt.get_image()

        imgs = [anno_img]
        for proposals in proposals_dict.values():
            prop = proposals[0] # only visualize one image in a batch
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)

            boxes = prop.proposal_boxes[0:box_size].tensor.cpu().numpy()

            probs = torch.sigmoid(prop.objectness_logits[0:box_size]).cpu().numpy()
            gt_classes = prop.gt_classes[0:box_size]
            labels = [f"{prob:.2f}, {cls}" for prob, cls in zip(probs, gt_classes)]

            v_pred = v_pred.overlay_instances(
                boxes=boxes,
                labels=labels,
            )
            prop_img = v_pred.get_image()
            imgs.append(prop_img)

        vis_img = np.concatenate(imgs, axis=1)
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images, support_images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            for x in batched_inputs:
                x['instances'].set('gt_classes', torch.full_like(x['instances'].get('gt_classes'), 0))

            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        B, N, C, H, W = support_images.tensor.shape
        assert N == self.support_way * self.support_shot

        # Split support images based on support way and shot
        support_images = support_images.tensor.view(B, self.support_way, self.support_shot, C, H, W)

        losses = defaultdict(float)

        # TODO: support better handling of batch-wise training
        for i in range(B):
            # Query
            query_gt_instances = [gt_instances[i]]  # One query gt instances
            query_images = ImageList.from_tensors([images[i]])  # One query image

            query_features_dict = {}
            support_features_dict = {}
            support_boxes_dict = {}

            for way in range(self.support_way):
                support_image_set = support_images[i, way, :, :, :, :]
                query_features, support_features = self.backbone(images.tensor[i].unsqueeze(0), support_image_set)
                query_features_dict[way] = {key: query_features[key] for key in query_features.keys()}
                support_features_dict[way] = {key: support_features[key] for key in support_features.keys()}

                support_boxes = batched_inputs[i]['support_bboxes'][way * self.support_shot:(way + 1) * self.support_shot]
                support_boxes = [Boxes(box[np.newaxis, :]).to(self.device) for box in support_boxes]

                support_boxes_dict[way] = support_boxes

            proposals_dict, proposal_losses = self.proposal_generator(
                query_images,
                query_features_dict,
                support_features_dict,
                support_boxes_dict,
                query_gt_instances
                )

            _, detector_losses = self.roi_heads(
                query_images,
                query_features_dict,
                support_features_dict,
                support_boxes_dict,
                proposals_dict,
                query_gt_instances
                )

            if self.vis_period > 0 and i == 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_dict)

            for loss_type, value in detector_losses.items():
                losses[loss_type] += value

            for loss_type, value in proposal_losses.items():
                losses[loss_type] += value

        losses = {loss_type: value / B for loss_type, value in losses.items()}
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images, support_images = self.preprocess_image(batched_inputs)

        B, C, H, W = images.tensor.shape
        assert B == 1 # only support 1 query image in test
        assert len(images) == 1

        # TODO: make possible for arbitrary batch
        # for i in range(B):
        i = 0
        batch_item = batched_inputs[i]
        query_images = ImageList.from_tensors([images[i]]) # one query image
        support_images = support_images[i] # one support image

        # Get support images and shots per class from batched inputs
        shots_per_class = batch_item["shots_per_class"]

        query_features_dict = {}
        support_features_dict = {}
        support_boxes_dict = {}

        # Initialize an offset to track where the data for each class starts
        current_offset = 0

        # Iterate over each class and use smart slicing to get support data
        for cls_id, num_shots in shots_per_class.items():
            # Calculate the start and end indices for slicing
            start_idx = current_offset
            end_idx = current_offset + num_shots

            # Slice the support images and bounding boxes for the given class
            support_images_cls = support_images[start_idx:end_idx]

            query_features, support_features = self.backbone(query_images.tensor, support_images_cls)
            query_features_dict[cls_id] = {key: query_features[key] for key in query_features.keys()}
            support_features_dict[cls_id] = {key: support_features[key] for key in support_features.keys()}

            support_boxes = batch_item["support_bboxes"][start_idx:end_idx]
            support_boxes = [Boxes(box[np.newaxis, :]).to(self.device) for box in support_boxes]

            support_boxes_dict[cls_id] = support_boxes

            # Update the current offset for the next class
            current_offset = end_idx


        proposals_dict, _ = self.proposal_generator(
            query_images,
            query_features_dict,
            support_features_dict,
            support_boxes_dict,
            None
        )

        results, _ = self.roi_heads(
            query_images,
            query_features_dict,
            support_features_dict,
            support_boxes_dict,
            proposals_dict
        )

        if do_postprocess:
            return FsodRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        query_images = super().preprocess_image(batched_inputs)

        support_images = [self._move_to_current_device(x["support_images"]) for x in batched_inputs]
        support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
        support_images = ImageList.from_tensors(
            support_images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return query_images, support_images
