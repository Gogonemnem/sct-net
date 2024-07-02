#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import concurrent.futures

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pycocotools.coco import COCO

def vis_image(im, bboxs, im_name):
    dpi = 300
    fig, ax = plt.subplots() 
    ax.imshow(im, aspect='equal') 
    plt.axis('off') 
    height, width, channels = im.shape 
    fig.set_size_inches(width/100.0/3.0, height/100.0/3.0) 
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0,0)
    # Show box (off by default, box_alpha=0.0)
    for bbox in bboxs:
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='r',
                          linewidth=0.5, alpha=1))
    output_name = os.path.basename(im_name)
    plt.savefig(im_name, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close('all')


def crop_support(img, bbox):
    image_shape = img.shape[:2]  # h, w
    data_height, data_width = image_shape
    
    img = img.transpose(2, 0, 1)

    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    
    width = x2 - x1
    height = y2 - y1
    context_pixel = 16 #int(16 * im_scale)
    
    new_x1 = 0
    new_y1 = 0
    new_x2 = width
    new_y2 = height
    target_size = (320, 320) #(384, 384)
 
    if width >= height:
        crop_x1 = x1 - context_pixel
        crop_x2 = x2 + context_pixel
   
        # New_x1 and new_x2 will change when crop context or overflow
        new_x1 = new_x1 + context_pixel
        new_x2 = new_x1 + width
        if crop_x1 < 0:
            new_x1 = new_x1 + crop_x1
            new_x2 = new_x1 + width
            crop_x1 = 0
        if crop_x2 > data_width:
            crop_x2 = data_width
            
        short_size = height
        long_size = crop_x2 - crop_x1
        y_center = int((y2+y1) / 2) #math.ceil((y2 + y1) / 2)
        crop_y1 = int(y_center - (long_size / 2)) #int(y_center - math.ceil(long_size / 2))
        crop_y2 = int(y_center + (long_size / 2)) #int(y_center + math.floor(long_size / 2))
        
        # New_y1 and new_y2 will change when crop context or overflow
        new_y1 = new_y1 + math.ceil((long_size - short_size) / 2)
        new_y2 = new_y1 + height
        if crop_y1 < 0:
            new_y1 = new_y1 + crop_y1
            new_y2 = new_y1 + height
            crop_y1 = 0
        if crop_y2 > data_height:
            crop_y2 = data_height
        
        crop_short_size = crop_y2 - crop_y1
        crop_long_size = crop_x2 - crop_x1
        square = np.zeros((3, crop_long_size, crop_long_size), dtype = np.uint8)
        delta = int((crop_long_size - crop_short_size) / 2) #int(math.ceil((crop_long_size - crop_short_size) / 2))
        square_y1 = delta
        square_y2 = delta + crop_short_size

        new_y1 = new_y1 + delta
        new_y2 = new_y2 + delta
        
        crop_box = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        square[:, square_y1:square_y2, :] = crop_box

        #show_square = np.zeros((crop_long_size, crop_long_size, 3))#, dtype=np.int16)
        #show_crop_box = original_img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        #show_square[square_y1:square_y2, :, :] = show_crop_box
        #show_square = show_square.astype(np.int16)
    else:
        crop_y1 = y1 - context_pixel
        crop_y2 = y2 + context_pixel
   
        # New_y1 and new_y2 will change when crop context or overflow
        new_y1 = new_y1 + context_pixel
        new_y2 = new_y1 + height
        if crop_y1 < 0:
            new_y1 = new_y1 + crop_y1
            new_y2 = new_y1 + height
            crop_y1 = 0
        if crop_y2 > data_height:
            crop_y2 = data_height
            
        short_size = width
        long_size = crop_y2 - crop_y1
        x_center = int((x2 + x1) / 2) #math.ceil((x2 + x1) / 2)
        crop_x1 = int(x_center - (long_size / 2)) #int(x_center - math.ceil(long_size / 2))
        crop_x2 = int(x_center + (long_size / 2)) #int(x_center + math.floor(long_size / 2))

        # New_x1 and new_x2 will change when crop context or overflow
        new_x1 = new_x1 + math.ceil((long_size - short_size) / 2)
        new_x2 = new_x1 + width
        if crop_x1 < 0:
            new_x1 = new_x1 + crop_x1
            new_x2 = new_x1 + width
            crop_x1 = 0
        if crop_x2 > data_width:
            crop_x2 = data_width

        crop_short_size = crop_x2 - crop_x1
        crop_long_size = crop_y2 - crop_y1
        square = np.zeros((3, crop_long_size, crop_long_size), dtype = np.uint8)
        delta = int((crop_long_size - crop_short_size) / 2) #int(math.ceil((crop_long_size - crop_short_size) / 2))
        square_x1 = delta
        square_x2 = delta + crop_short_size

        new_x1 = new_x1 + delta
        new_x2 = new_x2 + delta
        crop_box = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        square[:, :, square_x1:square_x2] = crop_box

        #show_square = np.zeros((crop_long_size, crop_long_size, 3)) #, dtype=np.int16)
        #show_crop_box = original_img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        #show_square[:, square_x1:square_x2, :] = show_crop_box
        #show_square = show_square.astype(np.int16)
    #print(crop_y2 - crop_y1, crop_x2 - crop_x1, bbox, data_height, data_width)

    square = square.astype(np.float32, copy=False)
    square_scale = float(target_size[0]) / long_size
    square = square.transpose(1,2,0)
    square = cv2.resize(square, target_size, interpolation=cv2.INTER_LINEAR) # None, None, fx=square_scale, fy=square_scale, interpolation=cv2.INTER_LINEAR)
    #square = square.transpose(2,0,1)
    square = square.astype(np.uint8)

    new_x1 = int(new_x1 * square_scale)
    new_y1 = int(new_y1 * square_scale)
    new_x2 = int(new_x2 * square_scale)
    new_y2 = int(new_y2 * square_scale)

    # For test
    #show_square = cv2.resize(show_square, target_size, interpolation=cv2.INTER_LINEAR) # None, None, fx=square_scale, fy=square_scale, interpolation=cv2.INTER_LINEAR)
    #self.vis_image(show_square, [new_x1, new_y1, new_x2, new_y2], img_path.split('/')[-1][:-4]+'_crop.jpg', './test')

    support_data = square
    support_box = np.array([new_x1, new_y1, new_x2, new_y2]).astype(np.float32)
    return support_data, support_box
        


def process_image(img_id, id, coco, crop_directory, img_directory, support_dict):
    if img_id % 100 == 0:
        print(img_id)

    img = coco.loadImgs(id)[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None))

    if len(anns) == 0:
        return
    
    file_ext_split = os.path.splitext(img['file_name'])
    frame_crop_directory = os.path.join(crop_directory, file_ext_split[0])
    if not os.path.isdir(frame_crop_directory):
        os.makedirs(frame_crop_directory)
        
    full_image = cv2.imread(os.path.join(img_directory, img['file_name']))

    for item_id, ann in enumerate(anns):
        rect = ann['bbox']
        x, y, w, h = rect
        x2, y2 = x + w, y + h
        bbox = [x, y, x2, y2]
        support_img, support_box = crop_support(full_image, bbox)

        if rect[2] <= 0 or rect[3] <=0:
            print(rect)
            continue

        file_path = os.path.join(frame_crop_directory, f"{item_id:04d}.jpg")
        cv2.imwrite(file_path, support_img)

        support_dict['support_box'].append(support_box.tolist())
        support_dict['category_id'].append(ann['category_id'])
        support_dict['image_id'].append(ann['image_id'])
        support_dict['id'].append(ann['id'])
        support_dict['file_path'].append(file_path)

def main(shot: int=0, full_class: bool=False):
    if not shot:
        directory_name = 'support'
    elif full_class:
        directory_name = f'full_class_{shot}_shot_support'
    else:
        directory_name = f'{shot}_shot_support'

    root_path = os.path.join(os.getcwd(), 'datasets', 'data', 'dota')

    support_path = os.path.join(root_path, directory_name)
    if not os.path.isdir(support_path):
        os.mkdir(support_path)

    support_dict = {
        'support_box': [],
        'category_id': [],
        'image_id': [],
        'id': [],
        'file_path': []
    }

    for dataType in ['train2017']:
        crop_directory = os.path.join(support_path, dataType)
        img_directory = os.path.join(root_path, dataType)

        if not shot:
            annotation_name = f'base_split_instances_{dataType}_with_small.json'
        elif full_class:
            annotation_name = f'full_class_{shot}_shot_instances_{dataType}.json'
        else:
            annotation_name = f'novel_split_{shot}_shot_instances_{dataType}.json'
        
        annFile = os.path.join(root_path, 'new_annotations', annotation_name)

        with open(annFile,'r') as load_f:
            dataset = json.load(load_f)
            print(dataset.keys())

        coco = COCO(annFile)
        
        max_workers = int(os.cpu_count() * 0.67)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for img_id, id in enumerate(coco.imgs):
                futures.append(executor.submit(process_image, img_id, id, coco, crop_directory, img_directory, support_dict))
            
            for future in concurrent.futures.as_completed(futures):
                pass

        support_df = pd.DataFrame.from_dict(support_dict)
        print("Total number of boxes is ", len(support_dict['id']))
        
        if not shot:
            pickle_name = "train_support_df.pkl"
        elif full_class:
            pickle_name = f"full_class_{shot}_shot_support_df.pkl"
        else:
            pickle_name = f"{shot}_shot_support_df.pkl"
        pickle_file = os.path.join(root_path, pickle_name)
        support_df.to_pickle(pickle_file)
        
    return support_df


if __name__ == '__main__':
    for shot in [0, 10]: # 0,
        for full_class in [True, False]:
            if not shot and not full_class:
                continue # skip redundant with full training set

            since = time.time()
            support_df = main(shot, full_class)

            time_elapsed = time.time() - since
            print(f'Total complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')