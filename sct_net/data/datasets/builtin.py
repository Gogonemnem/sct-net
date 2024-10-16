import os

from detectron2.data import MetadataCatalog

from .register_coco import register_coco_instances
from .builtin_meta_pascal_voc import _get_builtin_metadata_pascal_voc
from .meta_pascal_voc import register_meta_pascal_voc

CATEGORIES = {
    "pascal": [
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
    ],
    "dior": [
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
    ],
    "dota": [
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
    ],
    "coco": [
        {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
        {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
        {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
        {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
        {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
        {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
        {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
        {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
        {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
        {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
        {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
        {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
        {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
        {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
        {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
        {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
        {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
        {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
        {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
        {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
        {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
        {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
        {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
        {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
        {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
        {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
        {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
        {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
        {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
        {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
        {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
        {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
        {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
        {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
        {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
        {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
        {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
        {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
        {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
        {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
        {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
        {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
        {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
        {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
        {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
        {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
        {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
        {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
        {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
        {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
        {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
        {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
        {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
        {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
        {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
        {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
        {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
        {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
        {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
        {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
        {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
        {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
        {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
        {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
        {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
        {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
        {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
        {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
        {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
        {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
        {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
        {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
        {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
        {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
        {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
        {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
        {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
        {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
        {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
        {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    ],
}

# ==== Predefined datasets and splits for PASCAL VOC in COCO format ==========

_PREDEFINED_SPLITS = {}
_PREDEFINED_SPLITS["pascal"] = {
    "pascal_train_base":            ("train", "new_annotations/base_split_instances_train.json"), # by default no_smaller_32
    "pascal_train_base_with_small": ("train", "new_annotations/base_split_instances_train_with_small.json"), # includeing all boxes
    "pascal_test":                    ("test",  "annotations/instances_test.json"),
}

# ==== Predefined datasets and splits for DIOR ==========
_PREDEFINED_SPLITS["dior"] = {
    "dior_train2017_base":             ("train2017", "new_annotations/base_split_instances_train2017.json"), # by default no_smaller_32
    "dior_train2017_base_with_small":  ("train2017", "new_annotations/base_split_instances_train2017_with_small.json"), # includeing all boxes
    "dior_test2017":                     ("test2017",  "annotations/instances_test2017.json"),
}

# ==== Predefined datasets and splits for DOTA ==========
_PREDEFINED_SPLITS["dota"] = {
        "dota_train2017_base":            ("train2017", "new_annotations/base_split_instances_train2017.json"), # by default no_smaller_32
        "dota_train2017_base_with_small": ("train2017", "new_annotations/base_split_instances_train2017_with_small.json"), # includeing all boxes
        "dota_test2017":                    ("test2017",  "annotations/instances_test2017.json"),
    }

# ==== Predefined datasets and splits for COCO ==========#
_PREDEFINED_SPLITS["coco"] = {
    "coco_train2017_base":            ("train2017", "new_annotations/final_split_non_voc_instances_train2017.json"), # by default no_smaller_32
    "coco_train2017_base_with_small": ("train2017", "new_annotations/final_split_non_voc_instances_train2017_with_small.json"), # includeing all boxes
    "coco_val2017":                     ("val2017",   "annotations/instances_val2017.json"),
}


# Define functions to get dataset metadata
def _get_instances_meta(dataset_name):
    categories = CATEGORIES[dataset_name]
    thing_ids = [k["id"] for k in categories if k["isthing"] == 1]
    thing_colors = [k["color"] for k in categories if k["isthing"] == 1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in categories if k["isthing"] == 1]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }


def register_all(root, dataset_name):
    dataset_splits = _PREDEFINED_SPLITS[dataset_name].items()
    
    data_types = set()
    for split_name, (image_root, json_file) in dataset_splits:
        data_types.add(image_root)

    # TODO: shots configurable?
    for shot in [1, 2, 3, 5, 10, 30]:
        for data_type in data_types:
            name = f"{dataset_name}_{data_type}_voc_{shot}shot"
            _PREDEFINED_SPLITS[dataset_name][name] = (data_type, f"new_annotations/novel_split_{shot}_shot_instances_{data_type}.json")

            name = f"{dataset_name}_{data_type}_full_{shot}shot"
            _PREDEFINED_SPLITS[dataset_name][name] = (data_type, f"new_annotations/full_class_{shot}_shot_instances_{data_type}.json")
            
            # TODO: seeds configurable?
            for seed in range(1, 10):
                name = f"{dataset_name}_{data_type}_voc_{shot}_shot_{seed}_seed"
                _PREDEFINED_SPLITS[dataset_name][name] = (data_type, f"new_annotations/seed{seed}/{shot}_shot_instances_{data_type}.json")

                name = f"{dataset_name}_{data_type}_full_{shot}_shot_{seed}_seed"
                _PREDEFINED_SPLITS[dataset_name][name] = (data_type, f"new_annotations/seed{seed}/full_class_{shot}_shot_instances_{data_type}.json")

    for split_name, (image_root, json_file) in dataset_splits:
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            split_name,
            _get_instances_meta(dataset_name),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

# Register them all under "./datasets"
root_dir = "./" # CHANGE THIS
_root = os.getenv("DETECTRON2_DOTA", root_dir + "datasets/data/dota")
register_all(_root, "dota")
_root = os.getenv("DETECTRON2_DIOR", root_dir + "datasets/data/dior")
register_all(_root, "dior")
_root = os.getenv("DETECTRON2_PASCAL", root_dir + "datasets/data/pascal")
register_all(_root, "pascal")
_root = os.getenv("DETECTRON2_COCO", root_dir + "datasets/data/coco")
register_all(_root, "coco")


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

#_root = os.getenv("DETECTRON2_DATASETS", "datasets/pascal_voc")
#register_all_pascal_voc(_root)
