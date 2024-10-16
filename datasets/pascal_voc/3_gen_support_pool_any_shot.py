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


def main(root_path, year, split, keepclasses, sid, shot):
    dirname = "VOC{}".format(year)

    # classnames_all = PASCAL_VOC_ALL_CATEGORIES[sid]
    if keepclasses == 'all':
        classnames = PASCAL_VOC_ALL_CATEGORIES[sid]
    elif keepclasses == 'base':
        classnames = PASCAL_VOC_BASE_CATEGORIES[sid]
    elif keepclasses == 'novel':
        classnames = PASCAL_VOC_NOVEL_CATEGORIES[sid]

    fileids = {}
    for cls in classnames:
        with open(os.path.join("vocsplit", "box_{}shot_{}_train.txt".format(shot, cls))) as f:
            fileids_ = np.loadtxt(f, dtype=np.str).tolist()
            if isinstance(fileids_, str):
                fileids_ = [fileids_]
            fileids_ = [fid.split('/')[-1].split('.jpg')[0] \
                            for fid in fileids_]
            fileids[cls] = fileids_

    support_dict = {}
    support_dict['support_box'] = []
    support_dict['category_id'] = []
    support_dict['image_id'] = []
    support_dict['id'] = []
    support_dict['file_path'] = []

    support_path = os.path.join(root_path, 'voc_{}_{}_{}{}_{}shot'.format(year, split, keepclasses, sid, shot))
    if not isdir(support_path):
        mkdir(support_path)

    box_id = 0
    vis = {}
    for cls, fileids_ in fileids.items():
        for fileid in fileids_:
            if fileid in vis:
                continue
            else:
                vis[fileid] = True
            year = "2012" if "_" in fileid else "2007"
            dirname = "VOC{}".format(year)
            anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
            jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

            frame_crop_base_path = join(support_path, fileid)
            if not isdir(frame_crop_base_path):
                makedirs(frame_crop_base_path)

            im = cv2.imread(jpeg_file)
            tree = ET.parse(anno_file)
            count = 0

            for obj in tree.findall("object"):
                cls_ = obj.find("name").text
                if not (cls_ in classnames):
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
                support_dict['category_id'].append(classnames.index(cls_)) #(classnames_all.index(cls_))
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
    keepclasses = 'all'
    sid = 2
    shot = 3
    root_path = sys.argv[1]


    for year in [2007, 2012]:
        for keepclasses in ['all',]: #'novel']:
            for sid in [1,2,3]:
                for shot in [1,2,3,5,10]:
                    print("*******************keepclasses={}, sid={}, shot={}".format(keepclasses, sid, shot))

                    since = time.time()
                    support_df = main(root_path, year, split, keepclasses, sid, shot)
                    support_df.to_pickle("./voc_{}_{}_{}{}_{}shot.pkl".format(year, split, keepclasses, sid, shot))

                    time_elapsed = time.time() - since
                    print('Total complete in {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))
