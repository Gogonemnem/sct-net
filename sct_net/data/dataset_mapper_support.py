import os
import logging
from typing import Tuple

import pandas as pd
import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper
from detectron2.data.catalog import MetadataCatalog

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithSupport"]


class DatasetMapperWithSupport(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and maps it into a format used by the model, including support data for few-shot learning.

    This class inherits from the default DatasetMapper and adds functionality to include
    support images and annotations required for few-shot object detection.

    It can handle both COCO and Pascal VOC datasets based on configuration.
    """

    @configurable
    def __init__(
        self,
        *,
        support_on: bool = True,
        few_shot: bool = True,
        support_way: int = 1,
        support_shot: int = 5,
        test_shot: int = 10,
        data_dir: str = "",
        seeds: int = 0,
        dataset_name: str = "",
        dataset_names: Tuple[str] = [],
        test_all_classes: bool = True,
        support_exclude_query: bool = False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.support_on = support_on
        self.few_shot = few_shot
        self.support_way = support_way
        self.support_shot = support_shot
        self.test_shot = test_shot
        self.data_dir = data_dir
        self.seeds = seeds
        self.test_all_classes = test_all_classes
        self.support_exclude_query = support_exclude_query

        self.dataset_name = dataset_name
        self.dataset_names = dataset_names
        self.dataset_type = self._get_dataset_type(self.dataset_name)
        if self.support_on:
            self._load_support_dataframe()
            self._prepare_metadata()
            if not self.is_train:
                # Load support data for inference if in evaluation mode
                self.support_data_for_inference, self.shots_per_class = self._load_all_support_for_inference()


    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        # Additional configuration parameters for support data
        ret.update({
            "support_on":   cfg.FEWSHOT.ENABLED,  # Enable support data during training
            "few_shot":     cfg.FEWSHOT.FEW_SHOT,
            "support_way":  cfg.FEWSHOT.SUPPORT_WAY,
            "support_shot": cfg.FEWSHOT.SUPPORT_SHOT,
            "test_shot":    cfg.FEWSHOT.TEST_SHOT,
            "data_dir":     cfg.DATA_DIR,
            "seeds":        cfg.DATASETS.SEEDS,
            "dataset_name": cfg.DATASETS.TRAIN[0],
            "dataset_names": cfg.DATASETS.TRAIN,
            "test_all_classes": cfg.DATASETS.TEST_ALL_CLASSES,
            "support_exclude_query": cfg.FEWSHOT.SUPPORT_EXCLUDE_QUERY,
        })
        return ret

    def _get_dataset_type(self, dataset_name):
        if 'coco' in dataset_name.lower():
            return 'coco'
        elif 'voc' in dataset_name.lower() or 'pascal' in dataset_name.lower():
            return 'voc'
        else:
            return 'coco'
            # raise ValueError(f"Unsupported dataset type in {dataset_name}")

    def _load_support_dataframe(self):
        if self.dataset_type == 'coco':
            self._load_support_dataframe_coco()
        elif self.dataset_type == 'voc':
            self._load_support_dataframe_voc()

    def _load_support_dataframe_coco(self):
        # Construct the support dataset path based on configuration
        prefix = ""
        if self.seeds != 0:
            prefix += f"seeds{self.seeds}/"
        
        train_full_classes = 'full' in self.dataset_name and self.is_train
        test_full_classes = self.test_all_classes and not self.is_train
        if train_full_classes or test_full_classes:
            prefix += "full_class_"

        if self.few_shot:
            path = prefix + f"{self.support_shot+1}_shot_support_df.json"
        elif not self.is_train:
            path = prefix + f"{self.test_shot}_shot_support_df.json"
        else:
            path = "train_support_df.json"

        support_dataset_path = os.path.join(self.data_dir, path)
        self.support_df = pd.read_json(support_dataset_path, orient='records', lines=True)
        logging.getLogger(__name__).info(f"Loaded support dataframe from {support_dataset_path}")

    def _load_support_dataframe_voc(self):
        # Load support_df for VOC dataset
        prefix = ""
        if self.seeds != 0:
            prefix += f"seed{self.seeds}/"
        if self.few_shot:
            path = prefix + f"{self.dataset_name}.pkl"
        else:
            # Combine multiple datasets if necessary
            support_dfs = []
            for dataset_name in self.dataset_names:
                path = os.path.join(self.data_dir, f"{dataset_name}.pkl")
                support_df_tmp = pd.read_pickle(path)
                support_dfs.append(support_df_tmp)
                logging.getLogger(__name__).info(f"Loaded support dataframe from {path}")

            self.support_df = pd.concat(support_dfs, ignore_index=True)
            return

        self.support_df = pd.read_pickle(path)
        logging.getLogger(__name__).info(f"Loaded support dataframe from {path}")

    def _prepare_metadata(self):
        # Prepare metadata and mappings
        metadata = MetadataCatalog.get(self.dataset_name)
        if self.dataset_type == 'coco':
            # Map category IDs to contiguous IDs
            reverse_id_mapper = lambda dataset_id: metadata.thing_dataset_id_to_contiguous_id[dataset_id]
            self.support_df['category_id'] = self.support_df['category_id'].map(reverse_id_mapper)
        # For VOC, IDs are usually already contiguous
    
    def _load_all_support_for_inference(self):
        """
        Preloads all support data for inference. This data will be reused during inference.

        Returns:
            Tuple of (support_images, support_bboxes, support_cls) and shots per class.
        """
        support_images_list = []
        support_bboxes_list = []
        support_cls_list = []
        shots_per_class = {}

        # Unique classes for consistent ordering
        unique_classes = self.support_df['category_id'].unique()

        for cls_id in unique_classes:
            cls_df = self.support_df[self.support_df['category_id'] == cls_id]
            shots_per_class[cls_id] = len(cls_df)
            for _, row in cls_df.iterrows():
                support_image, support_box, support_cls, _ = self._load_support_data(row)
                support_images_list.append(support_image)
                support_bboxes_list.append(support_box)
                support_cls_list.append(support_cls)

        support_images = np.stack(support_images_list, axis=0)
        support_bboxes = np.stack(support_bboxes_list, axis=0)
        support_cls = support_cls_list

        return (support_images, support_bboxes, support_cls), shots_per_class

    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)
        if self.support_on:
            if self.is_train:
                # Generate support data and add to dataset_dict during training
                support_images, support_bboxes, support_cls = self.generate_support(dataset_dict)
            else:
                # Use preloaded support data during inference
                support_images, support_bboxes, support_cls = self.support_data_for_inference
                dataset_dict['shots_per_class'] = self.shots_per_class

            dataset_dict['support_images'] = torch.as_tensor(support_images)
            dataset_dict['support_bboxes'] = torch.as_tensor(support_bboxes)
            dataset_dict['support_cls'] = torch.as_tensor(support_cls, dtype=torch.long)
        return dataset_dict

    def generate_support(self, dataset_dict):
        support_way = self.support_way #2
        support_shot = self.support_shot #5

        query_instances = dataset_dict.get('instances', None)
        if query_instances is None or len(query_instances) == 0:
            # If no annotations, return empty support data
            num_support = support_way * support_shot
            support_images = np.zeros((num_support, 3, 320, 320), dtype=np.float32)
            support_bboxes = np.zeros((num_support, 4), dtype=np.float32)
            support_cls = []
            return support_images, support_bboxes, support_cls

        # Get query class(es) and image ID
        query_img_id = dataset_dict['image_id']
        used_image_ids = [query_img_id] if self.support_exclude_query else []
        query_classes = query_instances.gt_classes.unique().tolist() # its only one class for now anyways

        all_classes_in_query = self.support_df.loc[self.support_df['image_id']==query_img_id, 'category_id'].tolist()

        used_ids = set()
        # used_category_ids = set(query_classes) all_cls also takes into account other support classes for contrastive loss
        used_category_ids = set(all_classes_in_query)

        support_images_list = []
        support_bboxes_list = []
        support_cls_list = []

        for query_cls in query_classes:
            for _ in range(support_shot):
                support_sample = self._sample_support(query_cls, used_ids, used_image_ids)
                if support_sample is None:
                    continue

                image, bbox, cls, support_id = support_sample
                support_images_list.append(image)
                support_bboxes_list.append(bbox)
                support_cls_list.append(cls)
                used_ids.add(support_id)

        # Prepare support data for other classes if support_way > number of query classes
        num_other_classes = support_way - len(query_classes)
        if num_other_classes > 0:
            all_classes = set(self.support_df['category_id'].unique())
            other_classes = all_classes - used_category_ids
            other_classes = list(other_classes)
            np.random.shuffle(other_classes)
            for other_cls in other_classes[:num_other_classes]:
                used_category_ids.add(other_cls)
                for _ in range(support_shot):
                    support_sample = self._sample_support(other_cls, used_ids, used_image_ids)
                    if support_sample is None:
                        continue

                    image, bbox, cls, support_id = support_sample
                    support_images_list.append(image)
                    support_bboxes_list.append(bbox)
                    support_cls_list.append(cls)
                    used_ids.add(support_id)

        # Stack support data
        support_images = np.stack(support_images_list, axis=0)
        support_bboxes = np.stack(support_bboxes_list, axis=0)
        support_cls = support_cls_list

        return support_images, support_bboxes, support_cls
    
    def _sample_support_metadata(self, category_id, used_ids, used_image_ids):
        """
        Samples metadata for a support example for a given category ID.

        Args:
            category_id (int): The category ID to sample.
            used_ids (set): Set of already used support IDs.
            used_image_ids (list): List of image IDs to exclude.

        Returns:
            A dictionary with metadata of a support sample, or None if no sample is available.
        """
        df = self.support_df
        if self.support_exclude_query:
            available_supports = df[
                (df['category_id'] == category_id) &
                (~df['image_id'].isin(used_image_ids)) &
                (~df['id'].isin(used_ids))
            ]
        else:
            available_supports = df[
                (df['category_id'] == category_id) &
                (~df['id'].isin(used_ids))
            ]

        if available_supports.empty:
            return None

        support_sample_metadata = available_supports.sample(n=1).iloc[0]
        return support_sample_metadata


    def _load_support_data(self, support_sample_metadata):
        """
        Loads support data (image, bbox, class_id) from metadata.

        Args:
            support_sample_metadata (pd.Series): Metadata of the support sample.

        Returns:
            Tuple of (image, bbox, class_id, support_id).
        """
        support_id = support_sample_metadata['id']
        support_cls = support_sample_metadata['category_id']
        file_path = support_sample_metadata['file_path']

        # Construct the full file path to the support image
        file_path = os.path.join(self.data_dir, file_path)

        # Read the support image and convert to tensor
        support_image = utils.read_image(file_path, format=self.image_format)
        support_image = torch.as_tensor(np.ascontiguousarray(support_image.transpose(2, 0, 1)))  # Convert to CxHxW

        # Read the bounding box
        support_box = np.array(support_sample_metadata['support_box'], dtype=np.float32)

        return support_image, support_box, support_cls, support_id
    
    def _sample_support(self, category_id, used_ids, used_image_ids):
        """
        Samples a support example for a given category ID.

        Args:
            category_id (int): The category ID to sample.
            used_ids (set): Set of already used support IDs.
            used_image_ids (list): List of image IDs to exclude.

        Returns:
            Tuple of (image, bbox, class_id, support_id) or None if not found.
        """
        # First, sample the metadata for the support example
        support_sample_metadata = self._sample_support_metadata(category_id, used_ids, used_image_ids)
        if support_sample_metadata is None:
            return None

        # Load the actual support data (image, bbox, etc.) using the metadata
        return self._load_support_data(support_sample_metadata)