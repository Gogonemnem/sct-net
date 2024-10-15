import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union, Optional, Dict

from fvcore.common.file_io import PathManager
import numpy as np

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


__all__ = ["register_meta_pascal_voc"]


def load_voc_instances(
    dirname: str,
    split: str,
    class_names: Union[List[str], Tuple[str, ...]],
    fileids: Union[None, List[str], Dict[str, List[str]]] = None,
    shot: Optional[int] = None,
    keep_difficult: bool = False,
):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname (str): Directory containing "Annotations", "ImageSets", "JPEGImages".
        split (str): One of "train", "test", "val", "trainval".
        class_names (list or tuple): List of class names to include.
        fileids (list of str or dict of class_name to list of str, optional): File IDs to process.
            If None, reads from the split file.
            If dict, keys are class names, values are lists of file IDs.
        shot (int, optional): Number of instances per class to include.
        keep_difficult (bool): Whether to include difficult instances.
    """
    dicts = []
    if isinstance(fileids, dict):
        # Few-shot setting, process per class
        for cls, cls_fileids in fileids.items():
            total_instances = 0
            for fileid in cls_fileids:
                year = "2012" if "_" in fileid else "2007"
                dirname_year = os.path.join(dirname, f"VOC{year}")
                anno_file = os.path.join(dirname_year, "Annotations", fileid + ".xml")
                jpeg_file = os.path.join(dirname_year, "JPEGImages", fileid + ".jpg")

                with PathManager.open(anno_file) as f:
                    tree = ET.parse(f)

                r = {
                    "file_name": jpeg_file,
                    "image_id": fileid,
                    "height": int(tree.find("./size/height").text),
                    "width": int(tree.find("./size/width").text),
                }

                instances = []
                for obj in tree.findall("object"):
                    obj_cls = obj.find("name").text
                    if obj_cls != cls:
                        continue

                    difficult = int(obj.find("difficult").text)
                    if not keep_difficult and difficult == 1:
                        continue

                    bbox = obj.find("bndbox")
                    bbox = [float(bbox.find(x).text) - 1.0 for x in ["xmin", "ymin", "xmax", "ymax"]]

                    instances.append({
                        "category_id": class_names.index(cls),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                    })

                if not instances:
                    continue

                # Enforce shot limit per class
                if shot is not None:
                    if total_instances + len(instances) > shot:
                        instances = instances[: shot - total_instances]
                        total_instances = shot
                    else:
                        total_instances += len(instances)

                r["annotations"] = instances
                dicts.append(r)

                if shot is not None and total_instances >= shot:
                    break
    else:
        # Standard setting or few-shot without per-class file IDs
        if fileids is None:
            with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
                fileids = np.loadtxt(f, dtype=str)

        annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))

        for fileid in fileids:
            anno_file = os.path.join(annotation_dirname, fileid + ".xml")
            jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

            with PathManager.open(anno_file) as f:
                tree = ET.parse(f)

            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.find("./size/height").text),
                "width": int(tree.find("./size/width").text),
            }
            instances = []

            for obj in tree.findall("object"):
                cls = obj.find("name").text
                if cls not in class_names:
                    continue

                difficult = int(obj.find("difficult").text)
                if not keep_difficult and difficult == 1:
                    continue

                bbox = obj.find("bndbox")
                bbox = [float(bbox.find(x).text) - 1.0 for x in ["xmin", "ymin", "xmax", "ymax"]]

                instances.append({
                    "category_id": class_names.index(cls),
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                })

            if instances:
                r["annotations"] = instances
                dicts.append(r)

    return dicts


def get_few_shot_fileids(name: str, class_names: List[str], shot: int, seed: Optional[int] = None):
    """
    Get per-class file IDs for the few-shot setting.

    Args:
        name (str): Dataset name.
        class_names (list of str): List of class names.
        shot (int): Number of shots per class.
        seed (int, optional): Seed for the dataset split.

    Returns:
        Dict[str, List[str]]: Dictionary mapping class names to file IDs.
    """
    fileids = {}
    split_dir = os.path.join("datasets", "pascal_voc", "vocsplit")
    if seed is not None:
        split_dir = os.path.join(split_dir, f"seed{seed}")

    for cls in class_names:
        shot_file = os.path.join(split_dir, f"box_{shot}shot_{cls}_train.txt")
        with PathManager.open(shot_file) as f:
            ids = np.loadtxt(f, dtype=str).tolist()
            if isinstance(ids, str):
                ids = [ids]
            ids = [os.path.splitext(os.path.basename(fid))[0] for fid in ids]
            fileids[cls] = ids

    return fileids



def register_meta_pascal_voc(
    name: str,
    metadata: dict,
    dirname: str,
    split: str,
    year: int,
    keepclasses: str,
    sid: int,
):
    """
    Register the Pascal VOC dataset with support for meta-learning experiments.

    Args:
        name (str): Dataset name.
        metadata (dict): Metadata containing class information.
        dirname (str): Root directory of the dataset.
        split (str): Dataset split ("train", "test", etc.).
        year (int): Dataset year (2007 or 2012).
        keepclasses (str): Classes to keep ("base", "novel", etc.).
        sid (int): Split ID.
    """
    if keepclasses.startswith('base_novel'):
        thing_classes = metadata["thing_classes"][sid]
    elif keepclasses.startswith('base'):
        thing_classes = metadata["base_classes"][sid]
    elif keepclasses.startswith('novel'):
        thing_classes = metadata["novel_classes"][sid]
    else:
        thing_classes = metadata["thing_classes"][sid]

    is_shots = "shot" in name
    shot = None
    seed = None

    if is_shots:
        # Parse shot and seed from the dataset name
        parts = name.split('_')
        if "seed" in name:
            shot_part = [p for p in parts if 'shot' in p][0]
            shot = int(shot_part.replace('shot', ''))
            seed_part = [p for p in parts if 'seed' in p][0]
            seed = int(seed_part.replace('seed', ''))
        else:
            shot_part = [p for p in parts if 'shot' in p][0]
            shot = int(shot_part.replace('shot', ''))

    def dataset_func():
        if is_shots:
            fileids = get_few_shot_fileids(name, thing_classes, shot, seed)
            return load_voc_instances(
                dirname=dirname,
                split=split,
                class_names=thing_classes,
                fileids=fileids,
                shot=shot,
                keep_difficult=False,
            )
        else:
            return load_voc_instances(
                dirname=dirname,
                split=split,
                class_names=thing_classes,
                keep_difficult=False,
            )

    DatasetCatalog.register(name, dataset_func)

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        dirname=dirname,
        year=year,
        split=split,
        base_classes=metadata["base_classes"][sid],
        novel_classes=metadata["novel_classes"][sid],
    )
