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
        data_dir: str = "",
        seeds: int = 0,
        dataset_name: str = "",
        dataset_names: Tuple[str] = [],
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
        self.data_dir = data_dir
        self.seeds = seeds
        self.support_exclude_query = support_exclude_query

        self.dataset_name = dataset_name
        self.dataset_names = dataset_names
        self.dataset_type = self._get_dataset_type(self.dataset_name)
        if self.support_on:
            self._load_support_dataframe()
            self._prepare_metadata()

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        # Additional configuration parameters for support data
        ret.update({
            "support_on":   is_train,  # Enable support data during training
            "few_shot":     cfg.INPUT.FS.FEW_SHOT,
            "support_way":  cfg.INPUT.FS.SUPPORT_WAY,
            "support_shot": cfg.INPUT.FS.SUPPORT_SHOT,
            "data_dir":     cfg.DATA_DIR,
            "seeds":        cfg.DATASETS.SEEDS,
            "dataset_name": cfg.DATASETS.TRAIN[0],
            "dataset_names": cfg.DATASETS.TRAIN,
            "support_exclude_query": cfg.INPUT.FS.SUPPORT_EXCLUDE_QUERY,
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
        if 'full' in self.dataset_name:
            prefix += "full_class_"
        if self.few_shot:
            path = prefix + f"{self.support_shot+1}_shot_support_df.json"
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

    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)
        if self.support_on:
            # Generate support data and add to dataset_dict
            support_images, support_bboxes, support_cls = self.generate_support(dataset_dict)
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

        support_sample = available_supports.sample(n=1).iloc[0]
        support_id = support_sample['id']
        support_cls = support_sample['category_id']
        file_path = support_sample['file_path']

        file_path = os.path.join(self.data_dir, file_path)
        support_image = utils.read_image(file_path, format=self.image_format)
        support_image = torch.as_tensor(np.ascontiguousarray(support_image.transpose(2, 0, 1)))  # Convert to CxHxW
        support_box = np.array(support_sample['support_box'], dtype=np.float32)
        return support_image, support_box, support_cls, support_id
