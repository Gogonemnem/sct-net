"""
Created on Wednesday, September 28, 2022

@author: Guangxing Han
"""
import os

from .register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import load_coco_json
from .builtin_meta_pascal_voc import _get_builtin_metadata_pascal_voc
from .meta_pascal_voc import register_meta_pascal_voc
from detectron2.data import MetadataCatalog



# ==== Predefined datasets and splits for PASCAL VOC in COCO format ==========

_PREDEFINED_SPLITS_pascal = {}
_PREDEFINED_SPLITS_pascal["pascal"] = {
    "pascal_2014_train_nonvoc": ("coco/train", "coco/new_annotations/final_split_non_voc_instances_train2014.json"), # by default no_smaller_32
    "pascal_2014_train_nonvoc_with_small": ("coco/train", "coco/new_annotations/final_split_non_voc_instances_train2014_with_small.json"), # includeing all boxes
    "pascal_2014_val2": ("coco/test", "/home/bibahaduri/pascalvoc/coco/annotations/instances_test.json"),
    "pascal_2014_train_voc_10_shot": ("coco/train", "coco/new_annotations/final_split_voc_10_shot_instances_train2014.json"),
    "pascal_2014_train_voc_1_shot": ("coco/train", "coco/new_annotations/final_split_voc_1_shot_instances_train2014.json"),
    # "coco_2014_train_voc_2_shot": ("coco/train", "coco/new_annotations/final_split_voc_2_shot_instances_train2014.json"),
    "pascal_2014_train_voc_3_shot": ("coco/train", "coco/new_annotations/final_split_voc_3_shot_instances_train2014.json"),
    "pascal_2014_train_voc_5_shot": ("coco/train", "coco/new_annotations/final_split_voc_5_shot_instances_train2014.json"),
    # "coco_2014_train_voc_30_shot": ("coco/train", "coco/new_annotations/final_split_voc_30_shot_instances_train2014.json"),

    "pascal_2014_train_full_10_shot": ("coco/train", "coco/new_annotations/full_class_10_shot_instances_train2014.json"),
    "pascal_2014_train_full_1_shot": ("coco/train", "coco/new_annotations/full_class_1_shot_instances_train2014.json"),
    # "coco_2014_train_full_2_shot": ("coco/trainval2014", "coco/new_annotations/full_class_2_shot_instances_train2014.json"),
    "pascal_2014_train_full_3_shot": ("coco/train", "coco/new_annotations/full_class_3_shot_instances_train2014.json"),
    "pascal_2014_train_full_5_shot": ("coco/train", "coco/new_annotations/full_class_5_shot_instances_train2014.json"),
    # "coco_2014_train_full_30_shot": ("coco/trainval2014", "coco/new_annotations/full_class_30_shot_instances_train2014.json"),
}

PASCAL_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "aeroplane"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "bird"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "boat"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "bottle"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "car"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "cat"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "chair"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "cow"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "diningtable"},
    {"color": [220, 220, 0], "isthing": 1, "id": 12, "name": "dog"},
    {"color": [175, 116, 175], "isthing": 1, "id": 13, "name": "horse"},
    {"color": [250, 0, 30], "isthing": 1, "id": 14, "name": "motorbike"},
    {"color": [165, 42, 42], "isthing": 1, "id": 15, "name": "person"},
    {"color": [255, 77, 255], "isthing": 1, "id": 16, "name": "pottedplant"},
    {"color": [0, 226, 252], "isthing": 1, "id": 17, "name": "sheep"},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "sofa"},
    {"color": [0, 82, 0], "isthing": 1, "id": 19, "name": "train"},
    {"color": [120, 166, 157], "isthing": 1, "id": 20, "name": "tvmonitor"},
]

def _get_pascal_instances_meta():
    thing_ids = [k["id"] for k in PASCAL_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in PASCAL_CATEGORIES if k["isthing"] == 1]
    #assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in PASCAL_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret



def register_all_pascal(root):
    # for prefix in ["novel",]: #"all", 
    for shot in [1, 2, 3, 5, 10, 30]:
        for seed in range(1, 10):
            name = "pascal_2014_train_voc_{}_shot_seed{}".format(shot, seed)
            _PREDEFINED_SPLITS_pascal["pascal"][name] = ("coco/train", "coco/new_annotations/seed{}/{}_shot_instances_train2014.json".format(seed, shot))

            name = "pascal_2014_train_full_{}_shot_seed{}".format(shot, seed)
            _PREDEFINED_SPLITS_pascal["pascal"][name] = ("coco/train", "coco/new_annotations/seed{}/full_class_{}_shot_instances_train2014.json".format(seed, shot))

    dataset_metadata = _get_pascal_instances_meta()

    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_pascal.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                dataset_metadata,##_get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )




# ==== Predefined datasets and splits for DIOR ==========

_PREDEFINED_SPLITS_DIOR = {}
_PREDEFINED_SPLITS_DIOR["dior"] = {
    "dior_2014_train_nonvoc": ("coco/train2017", "coco/new_annotations/final_split_non_voc_instances_train2014.json"), # by default no_smaller_32
    "dior_2014_train_nonvoc_with_small": ("coco/train2017", "coco/new_annotations/final_split_non_voc_instances_train2014_with_small.json"), # includeing all boxes
    "dior_2014_val2": ("coco/test2017", "/home/bibahaduri/DIOR/coco/annotations/instances_test2017.json"),
    "dior_2014_train_voc_10_shot": ("coco/train2017", "coco/new_annotations/final_split_voc_10_shot_instances_train2014.json"),
    # "coco_2014_train_voc_1_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_1_shot_instances_train2014.json"),
    # "coco_2014_train_voc_2_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_2_shot_instances_train2014.json"),
    # "coco_2014_train_voc_3_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_3_shot_instances_train2014.json"),
    # "coco_2014_train_voc_5_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_5_shot_instances_train2014.json"),
    # "coco_2014_train_voc_30_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_30_shot_instances_train2014.json"),

    "dior_2014_train_full_10_shot": ("coco/train2017", "coco/new_annotations/full_class_10_shot_instances_train2014.json"),
    "dior_2014_test_full_10_shot": ("coco/test2017", "coco/new_annotations/full_class_10_shot_instances_test2014.json"),
    # "coco_2014_train_full_1_shot": ("coco/trainval2014", "coco/new_annotations/full_class_1_shot_instances_train2014.json"),
    # "coco_2014_train_full_2_shot": ("coco/trainval2014", "coco/new_annotations/full_class_2_shot_instances_train2014.json"),
    # "coco_2014_train_full_3_shot": ("coco/trainval2014", "coco/new_annotations/full_class_3_shot_instances_train2014.json"),
    # "coco_2014_train_full_5_shot": ("coco/trainval2014", "coco/new_annotations/full_class_5_shot_instances_train2014.json"),
    # "coco_2014_train_full_30_shot": ("coco/trainval2014", "coco/new_annotations/full_class_30_shot_instances_train2014.json"),
}

DIOR_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "Airplane "},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "Airport "},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "Baseball field "},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "Basketball court "},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "Bridge "},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "Chimney "},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "Dam "},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "Expressway service area "},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "Expressway toll station "},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "Golf course"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "Ground track field "},
    {"color": [220, 220, 0], "isthing": 1, "id": 12, "name": "Harbor "},
    {"color": [175, 116, 175], "isthing": 1, "id": 13, "name": "Overpass "},
    {"color": [250, 0, 30], "isthing": 1, "id": 14, "name": "Ship "},
    {"color": [165, 42, 42], "isthing": 1, "id": 15, "name": "Stadium "},
    {"color": [255, 77, 255], "isthing": 1, "id": 16, "name": "Storage tank "},
    {"color": [0, 226, 252], "isthing": 1, "id": 17, "name": "Tennis court "},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "Train station "},
    {"color": [0, 82, 0], "isthing": 1, "id": 19, "name": "Vehicle "},
    {"color": [120, 166, 157], "isthing": 1, "id": 20, "name": "Wind mill"},
]

def _get_dior_instances_meta():
    thing_ids = [k["id"] for k in DIOR_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DIOR_CATEGORIES if k["isthing"] == 1]
    #assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DIOR_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret



def register_all_dior(root):
    # for prefix in ["novel",]: #"all", 
    for shot in [1, 2, 3, 5, 10, 30]:
        for seed in range(1, 10):
            name = "dior_2014_train_voc_{}_shot_seed{}".format(shot, seed)
            _PREDEFINED_SPLITS_DIOR["dior"][name] = ("coco/train2017", "coco/new_annotations/seed{}/{}_shot_instances_train2014.json".format(seed, shot))

            name = "dior_2014_train_full_{}_shot_seed{}".format(shot, seed)
            _PREDEFINED_SPLITS_DIOR["dior"][name] = ("coco/train2017", "coco/new_annotations/seed{}/full_class_{}_shot_instances_train2014.json".format(seed, shot))

    dataset_metadata = _get_dior_instances_meta()

    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_DIOR.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                dataset_metadata,##_get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined datasets and splits for DOTA ==========

_PREDEFINED_SPLITS_DOTA = {}
_PREDEFINED_SPLITS_DOTA["dota"] = {
    "dota_2014_train_nonvoc": ("coco/train2017", "coco/new_annotations/final_split_non_voc_instances_train2014.json"), # by default no_smaller_32
    "dota_2014_train_nonvoc_with_small": ("coco/train2017", "coco/new_annotations/final_split_non_voc_instances_train2014_with_small.json"), # includeing all boxes
    "dota_2014_val2": ("coco/test2017", "/home/bibahaduri/dota_dataset/coco/annotations/instances_test2017.json"),
    "dota_2014_train_voc_10_shot": ("coco/train2017", "coco/new_annotations/final_split_voc_10_shot_instances_train2014.json"),
    # "coco_2014_train_voc_1_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_1_shot_instances_train2014.json"),
    # "coco_2014_train_voc_2_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_2_shot_instances_train2014.json"),
    # "coco_2014_train_voc_3_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_3_shot_instances_train2014.json"),
    # "coco_2014_train_voc_5_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_5_shot_instances_train2014.json"),
    # "coco_2014_train_voc_30_shot": ("coco/trainval2014", "coco/new_annotations/final_split_voc_30_shot_instances_train2014.json"),

    "dota_2014_train_full_10_shot": ("coco/train2017", "coco/new_annotations/full_class_10_shot_instances_train2014.json"),
    "dota_2014_test_full_10_shot": ("coco/test2017", "coco/new_annotations/full_class_10_shot_instances_test2014.json"),
    # "coco_2014_train_full_1_shot": ("coco/trainval2014", "coco/new_annotations/full_class_1_shot_instances_train2014.json"),
    # "coco_2014_train_full_2_shot": ("coco/trainval2014", "coco/new_annotations/full_class_2_shot_instances_train2014.json"),
    # "coco_2014_train_full_3_shot": ("coco/trainval2014", "coco/new_annotations/full_class_3_shot_instances_train2014.json"),
    # "coco_2014_train_full_5_shot": ("coco/trainval2014", "coco/new_annotations/full_class_5_shot_instances_train2014.json"),
    # "coco_2014_train_full_30_shot": ("coco/trainval2014", "coco/new_annotations/full_class_30_shot_instances_train2014.json"),
}

DOTA_CATEGORIES = [
    {"color": [133, 129, 255], "isthing": 1, "id": 1, "name": "plane"},
    {"color": [220, 20, 60], "isthing": 1, "id": 2, "name": "ship"},
    {"color": [119, 11, 32], "isthing": 1, "id": 3, "name": "storage-tank"},
    {"color": [0, 0, 142], "isthing": 1, "id": 4, "name": "baseball-diamond"},
    {"color": [0, 0, 230], "isthing": 1, "id": 5, "name": "tennis-court"},
    {"color": [106, 0, 228], "isthing": 1, "id": 6, "name": "basketball-court"},
    {"color": [0, 60, 100], "isthing": 1, "id": 7, "name": "ground-track-field"},
    {"color": [0, 80, 100], "isthing": 1, "id": 8, "name": "harbor"},
    {"color": [0, 0, 70], "isthing": 1, "id": 9, "name": "bridge"},
    {"color": [0, 0, 192], "isthing": 1, "id": 10, "name": "small-vehicle"},
    {"color": [250, 170, 30], "isthing": 1, "id": 11, "name": "large-vehicle"},
    {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "roundabout"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "swimming-pool"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "helicopter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "soccer-ball-field"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "container-crane"},
]


def _get_dota_instances_meta():
    thing_ids = [k["id"] for k in DOTA_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DOTA_CATEGORIES if k["isthing"] == 1]
    #assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DOTA_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret



def register_all_dota(root):
    # for prefix in ["novel",]: #"all", 
    for shot in [1, 2, 3, 5, 10, 30]:
        for seed in range(1, 10):
            name = "dota_2014_train_voc_{}_shot_seed{}".format(shot, seed)
            _PREDEFINED_SPLITS_DOTA["dota"][name] = ("coco/train2017", "coco/new_annotations/seed{}/{}_shot_instances_train2014.json".format(seed, shot))

            name = "dota_2014_train_full_{}_shot_seed{}".format(shot, seed)
            _PREDEFINED_SPLITS_DOTA["dota"][name] = ("coco/train2017", "coco/new_annotations/seed{}/full_class_{}_shot_instances_train2014.json".format(seed, shot))

    dataset_metadata = _get_dota_instances_meta()

    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_DOTA.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                dataset_metadata,##_get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root="datasets"):
    # register meta datasets
    METASPLITS = [
        ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
        ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
        ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
        ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(100):
                        seed = '' if seed == 0 else '_seed{}'.format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed)
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid)
                        keepclasses = "base_novel_{}".format(sid) \
                            if prefix == 'all' else "novel{}".format(sid)
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid))

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_pascal_voc(name,
                                 _get_builtin_metadata_pascal_voc("pascal_voc_fewshot"),
                                 os.path.join(root, dirname), split, year,
                                 keepclasses, sid)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DOTA", "/home/bibahaduri/dota_dataset")
register_all_dota(_root)
_root = os.getenv("DETECTRON2_DIOR", "/home/bibahaduri/DIOR")
register_all_dior(_root)
_root = os.getenv("DETECTRON2_PASCAL", "/home/bibahaduri/pascalvoc")
register_all_pascal(_root)

# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_all_coco(_root)
#_root = os.getenv("DETECTRON2_DATASETS", "datasets/pascal_voc")
#register_all_pascal_voc(_root)
