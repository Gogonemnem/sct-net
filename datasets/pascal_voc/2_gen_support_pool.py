#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday, April 14, 2022

@author: Guangxing Han
"""

import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
import sys
import time
import os
import pandas as pd
import sys

import xml.etree.ElementTree as ET

from sct_net.data.datasets.builtin_meta_pascal_voc import *
from datasets.coco_style.b_gen_support_pool import crop_support


def main(root_path, year, split, keepclasses, sid):
    dirname = "VOC{}".format(year)

    if keepclasses == 'all':
        classnames = PASCAL_VOC_ALL_CATEGORIES[sid]
    elif keepclasses == 'base':
        classnames = PASCAL_VOC_BASE_CATEGORIES[sid]
    elif keepclasses == 'novel':
        classnames = PASCAL_VOC_NOVEL_CATEGORIES[sid]

    f = open(os.path.join(dirname, "ImageSets", "Main", split + ".txt"))
    fileids = np.loadtxt(f, dtype=np.str)

    support_dict = {}
    support_dict['support_box'] = []
    support_dict['category_id'] = []
    support_dict['image_id'] = []
    support_dict['id'] = []
    support_dict['file_path'] = []

    support_path = os.path.join(root_path, 'voc_{}_{}_{}{}'.format(year, split, keepclasses, sid))
    if not isdir(support_path):
        mkdir(support_path)

    box_id = 0
    for img_id, fileid in enumerate(fileids):
        if img_id % 100 == 0:
            print(img_id)
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        frame_crop_base_path = join(support_path, fileid)
        if not isdir(frame_crop_base_path):
            makedirs(frame_crop_base_path)

        im = cv2.imread(jpeg_file)
        tree = ET.parse(anno_file)
        count = 0

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if not (cls in classnames):
                continue
            difficult = int(obj.find("difficult").text)
            if difficult == 1: 
                continue

            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0

            support_img, support_box = crop_support(im, bbox)

            file_path = join(frame_crop_base_path, '{:04d}.jpg'.format(count))
            cv2.imwrite(file_path, support_img)
            #print(file_path)
            support_dict['support_box'].append(support_box.tolist())
            support_dict['category_id'].append(classnames.index(cls))
            support_dict['image_id'].append(fileid)
            support_dict['id'].append(box_id)
            support_dict['file_path'].append(file_path)
            box_id += 1
            count += 1

    support_df = pd.DataFrame.from_dict(support_dict)
    return support_df


if __name__ == '__main__':
    year = 2007
    split = 'trainval'
    keepclasses = 'base'
    sid = 1
    root_path = sys.argv[1]

    for year in [2007, 2012]:
        for sid in [1, 2, 3]:
            since = time.time()
            support_df = main(root_path, year, split, keepclasses, sid)
            support_df.to_pickle("voc_{}_{}_{}{}.pkl".format(year, split, keepclasses, sid))

            time_elapsed = time.time() - since
            print('Total complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

