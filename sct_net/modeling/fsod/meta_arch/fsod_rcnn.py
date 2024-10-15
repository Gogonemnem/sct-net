import os
from collections import defaultdict

import pandas as pd
import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.data.catalog import MetadataCatalog
from detectron2.data import detection_utils as utils
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
        data_dir,
        dataset,
        **kwargs):
        super().__init__(**kwargs)

        self.support_way = support_way
        self.support_shot = support_shot
        self.data_dir = data_dir
        self.dataset = dataset

        self.evaluation_dataset = 'voc'
        self.evaluation_shot = 10
        self.keepclasses = 'all1'
        self.test_seeds = 0

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update(
            {
                "support_way": cfg.INPUT.FS.SUPPORT_WAY,
                "support_shot": cfg.INPUT.FS.SUPPORT_SHOT,
                "data_dir": cfg.DATA_DIR,
                "dataset": cfg.DATASETS.TRAIN[0],
            }
        )
        return ret

    def init_support_features(self, evaluation_dataset, evaluation_shot, keepclasses, test_seeds):
        self.evaluation_dataset = evaluation_dataset
        self.evaluation_shot = evaluation_shot
        self.keepclasses = keepclasses
        self.test_seeds = test_seeds

        if self.evaluation_dataset == 'voc':
            self.init_model_voc()
        elif self.evaluation_dataset == 'coco':
            self.init_model_coco()
    
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

        # support branches
        support_bboxes_ls = []
        for item in batched_inputs:
            bboxes = item['support_bboxes']
            for box in bboxes:
                box = Boxes(box[np.newaxis, :])
                support_bboxes_ls.append(box.to(self.device))
        
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

    def init_model_voc(self):
        if 1:
            if self.test_seeds == 0:
                support_path = os.path.join(self.data_dir, 'pascal_voc/voc_2007_trainval_{}_{}shot.json'.format(self.keepclasses, self.evaluation_shot))
            elif self.test_seeds >= 0:
                support_path = os.path.join(self.data_dir, 'pascal_voc/seed{}/voc_2007_trainval_{}_{}shot.json'.format(self.test_seeds, self.keepclasses, self.evaluation_shot))

            support_df = pd.read_json(support_path, orient='records', lines=True)

            min_shot = self.evaluation_shot
            max_shot = self.evaluation_shot
            self.support_dict = {'image': {}, 'box': {}}
            for cls in support_df['category_id'].unique():
                support_cls_df = support_df.loc[support_df['category_id'] == cls, :].reset_index()
                support_data_all = []
                support_box_all = []

                for index, support_img_df in support_cls_df.iterrows():
                    img_path = os.path.join(self.data_dir, 'pascal_voc', support_img_df['file_path'])
                    support_data = utils.read_image(img_path, format='BGR')
                    support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                    support_data_all.append(support_data)

                    support_box = support_img_df['support_box']
                    support_box_all.append(Boxes([support_box]).to(self.device))

                min_shot = min(min_shot, len(support_box_all))
                max_shot = max(max_shot, len(support_box_all))
                # support images
                support_images = [x.to(self.device) for x in support_data_all]
                support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
                support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)
                self.support_dict['image'][cls] = support_images
                self.support_dict['box'][cls] = support_box_all

            print("min_shot={}, max_shot={}".format(min_shot, max_shot))


    def init_model_coco(self):
        if 1:
            if self.keepclasses == 'all':
                if self.test_seeds == 0:
                    support_path = os.path.join(self.data_dir, 'full_class_{}_shot_support_df.json'.format(self.evaluation_shot))##(self.data_dir, 'coco/full_class_{}_shot_support_df.json'.format(self.evaluation_shot))
                elif self.test_seeds > 0:
                    support_path = os.path.join(self.data_dir, 'seed{}/full_class_{}_shot_support_df.json'.format(self.test_seeds, self.evaluation_shot))
            else:
                if self.test_seeds == 0:
                    support_path = os.path.join(self.data_dir, '{}_shot_support_df.json'.format(self.evaluation_shot))
                elif self.test_seeds > 0:
                    support_path = os.path.join(self.data_dir, 'seed{}/{}_shot_support_df.json'.format(self.test_seeds, self.evaluation_shot))

            support_df = pd.read_json(support_path, orient='records', lines=True)
            if 'coco' in self.dataset:
                metadata = MetadataCatalog.get('coco_2014_train')
            else:
                metadata = MetadataCatalog.get(self.dataset)##MetadataCatalog.get('coco_2014_train')  ##HACK
            # unmap the category mapping ids for COCO
            reverse_id_mapper = lambda dataset_id: metadata.thing_dataset_id_to_contiguous_id[dataset_id]  # noqa
            support_df['category_id'] = support_df['category_id'].map(reverse_id_mapper)

            min_shot = self.evaluation_shot
            max_shot = self.evaluation_shot
            self.support_dict = {'image': {}, 'box': {}}
            for cls in support_df['category_id'].unique():
                support_cls_df = support_df.loc[support_df['category_id'] == cls, :].reset_index()
                support_data_all = []
                support_box_all = []

                for index, support_img_df in support_cls_df.iterrows():
                    img_path = os.path.join(self.data_dir, support_img_df['file_path'])
                    support_data = utils.read_image(img_path, format='BGR')
                    support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                    support_data_all.append(support_data)

                    support_box = support_img_df['support_box']
                    support_box_all.append(Boxes([support_box]).to(self.device))

                min_shot = min(min_shot, len(support_box_all))
                max_shot = max(max_shot, len(support_box_all))
                # support images
                support_images = [x.to(self.device) for x in support_data_all]
                support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
                support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)
                self.support_dict['image'][cls] = support_images
                self.support_dict['box'][cls] = support_box_all

            print("min_shot={}, max_shot={}".format(min_shot, max_shot))


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
        
        images = self.preprocess_image(batched_inputs)

        B, C, H, W = images.tensor.shape
        assert B == 1 # only support 1 query image in test
        assert len(images) == 1

        # TODO: make possible for arbitrary batch
        # for i in range(B):
        i = 0
        query_images = ImageList.from_tensors([images[i]]) # one query image

        query_features_dict = {}
        support_features_dict = {}
        support_boxes_dict = {}

        for cls_id, support_images in self.support_dict['image'].items():
            query_features, support_features = self.backbone(query_images.tensor, support_images.tensor)
            query_features_dict[cls_id] = {key: query_features[key] for key in query_features.keys()}
            support_features_dict[cls_id] = {key: support_features[key] for key in support_features.keys()}
            support_boxes_dict[cls_id] = self.support_dict['box'][cls_id]

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
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        if not self.training:
            return images

        support_images = [x['support_images'].to(self.device) for x in batched_inputs]
        support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
        support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)

        return images, support_images
