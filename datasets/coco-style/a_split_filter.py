#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functionality for filtering and processing annotation splits.
It includes functions for loading annotations, filtering based on class and size criteria, combining 
annotations and saving processed annotations back to disk.
"""
import os
import json
from pycocotools.coco import COCO

from sct_net.data.datasets.builtin import CATEGORIES

FILTER_SMALL = False
NOVEL_CLASSES = {
    "dota": ["storage-tank", "tennis-court", "soccer-ball-field"],
    "dior": ["Airplane ", "Baseball field ", "Tennis court ", "Train station ", "Wind mill"]
}
DATA_TYPES = {
    "dota": ['train2017', 'test2017'],
    "dior": ['train2017', 'test2017'],
}

DATASET_NAME = "dior"
ALL_CLASSES = [cls['name'] for cls in CATEGORIES[DATASET_NAME]]
ROOT_PATH = os.path.join(os.getcwd(), 'datasets', 'data', DATASET_NAME)
SPLIT_DIRECTORY = os.path.join(ROOT_PATH, 'split', 'seed1729')
SHOTS = [10]


def get_novel_ids(novel_classes, categories):
    """Get the IDs of the novel classes."""
    novel_ids = set()

    for category in categories:
        if category['name'].strip().lower().replace(' ', '-') in novel_classes:
            novel_ids.add(category['id'])
        elif category['name'] in novel_classes:
            novel_ids.add(category['id'])
    return novel_ids

def load_json(json_file):
    """Load a JSON file."""
    with open(json_file, 'r', encoding='utf-8') as load_f:
        data = json.load(load_f)
    return data

def filter_annotations(coco, cls_split, filter_small):
    """Filter annotations based on class split and area size."""
    new_anns = []
    all_cls_dict = {}
    for img_id, id in enumerate(coco.imgs):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None))
        if len(anns) == 0:
            continue
        for ann in anns:
            area = ann['area']
            category_id = ann['category_id']
            if category_id in cls_split:
                if filter_small and area < 32 * 32:
                    continue
                new_anns.append(ann)
                if category_id in all_cls_dict.keys():
                    all_cls_dict[category_id] += 1
                else:
                    all_cls_dict[category_id] = 1
    return new_anns, all_cls_dict

def combine_annotations(file_ids):
    """Combine annotations from multiple files."""
    combined_imgs = []
    seen_img_ids = set()
    combined_anns = []
    for _, fileids_ in file_ids.items():
        for img_dict, anno_dict_list in fileids_:
            if img_dict['id'] not in seen_img_ids:
                combined_imgs.append(img_dict)
                seen_img_ids.add(img_dict['id'])
            combined_anns.extend(anno_dict_list)
    return combined_imgs, combined_anns

def save_annotations(dataset, data_type, prefix="", suffix="", root_path=""):
    """Save annotations to a JSON file."""
    new_annotations_path = os.path.join(root_path, 'new_annotations')
    if not os.path.isdir(new_annotations_path):
        os.mkdir(new_annotations_path)

    file = os.path.join(new_annotations_path, f'{prefix}_instances_{data_type}{suffix}.json')

    with open(file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
    print("Total number of boxes is ", len(dataset['annotations']))

def process_base_split(data_type, root_path, filter_small, novel_ids):
    """Process the base split annotations."""
    data_dir = os.path.join(root_path, 'annotations')
    ann_file = os.path.join(data_dir, f'instances_{data_type}.json')
    dataset = load_json(ann_file)
    base_ids, novel_names, base_names = split_categories(dataset, novel_ids)
    coco = COCO(ann_file)

    base_split_annotations, all_cls_dict = filter_annotations(coco, base_ids, filter_small)
    dataset['annotations'] = base_split_annotations
    save_annotations(dataset, data_type, prefix="base_split", suffix='' if filter_small else '_with_small', root_path=root_path)
    print_split_info(novel_names, base_names, all_cls_dict)

def process_few_shot(data_type, classes, shot, prefix, root_path, split_directory):
    """Process the few-shot annotations."""
    file_ids = {}
    for idx, cls in enumerate(classes):
        json_file = os.path.join(split_directory, f'full_box_{shot}shot_{cls}_trainval.json')
        coco_api = COCO(json_file)

        img_ids = coco_api.imgs.keys()
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        file_ids[idx] = list(zip(imgs, anns))
        last_dataset = load_json(json_file)

    combined_imgs, combined_anns = combine_annotations(file_ids)
    dataset = {
        'info': last_dataset['info'],
        'licenses': last_dataset['licenses'],
        'images': combined_imgs,
        'annotations': combined_anns,
        'categories': last_dataset['categories']
    }
    save_annotations(dataset, data_type, prefix=f"{prefix}{shot}_shot", root_path=root_path)

def split_categories(dataset, novel_ids):
    """Split categories into base and novel sets."""
    all_category_ids = set(category['id'] for category in dataset['categories'])
    id_to_name = {category['id']: category['name'] for category in dataset['categories']}
    base_ids = all_category_ids - novel_ids
    novel_names = [id_to_name[id] for id in novel_ids]
    base_names = [id_to_name[id] for id in base_ids]
    return base_ids, novel_names, base_names

def print_split_info(novel_names, base_names, all_cls_dict):
    """Print information about the category splits."""
    print('Novel Split: {} classes'.format(len(novel_names)))
    for name in novel_names:
        print('\t', name)
    print('Base Split: {} classes'.format(len(base_names)))
    for name in base_names:
        print('\t', name)
    print("Number of instances in each class")
    for id, cnt in sorted(all_cls_dict.items()):
        print(f"Class {id}: {cnt}")

def main():
    """
    Main function to process and filter annotations for both base and few-shot splits.
    
    This function iterates through the specified data types, processes the base split annotations,
    and then processes the few-shot annotations for both novel and full class splits.
    """
    for data_type in DATA_TYPES[DATASET_NAME]:
        novel_ids = get_novel_ids(NOVEL_CLASSES[DATASET_NAME], CATEGORIES[DATASET_NAME])
        process_base_split(data_type, ROOT_PATH, FILTER_SMALL, novel_ids)

        for classes in [NOVEL_CLASSES[DATASET_NAME], ALL_CLASSES]:
            for shot in SHOTS:
                prefix = "novel_split_" if classes == NOVEL_CLASSES[DATASET_NAME] else "full_class_"
                process_few_shot(data_type, classes, shot, prefix, ROOT_PATH, SPLIT_DIRECTORY)

if __name__ == "__main__":
    main()
