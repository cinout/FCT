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
from concurrent import futures
import sys
import time
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import shutil
import sys
import xml.etree.ElementTree as ET

MVTECVOC_ALL_CATEGORIES = [
    "aeroplane",
    "bicycle",
    "boat",
    "bottle",
    "car",
    "cat",
    "chair",
    "diningtable",
    "dog",
    "horse",
    "person",
    "pottedplant",
    "sheep",
    "train",
    "tvmonitor",
    # "bird",
    # "bus",
    # "cow",
    # "motorbike",
    # "sofa",
    # FIXME[DONE]: update
    "nectarine",
    "orange",
    "cereal",
    "almond_mix",
    "short_screw",
    "long_screw",
    "washer",
    "screw_nut",
    "pushpin",
    "clamp_2",
    "cable_yellow",
    "clamp_3",
    "cable_blue",
    "clamp_5",
    "cable_red",
    "juice_banana",
    "label_banana",
    "juice_orange",
    "label_orange",
    "juice_cherry",
    "label_cherry",
    "label_100",
]


def vis_image(im, bboxs, im_name):
    dpi = 300
    fig, ax = plt.subplots()
    ax.imshow(im, aspect="equal")
    plt.axis("off")
    height, width, channels = im.shape
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    # Show box (off by default, box_alpha=0.0)
    for bbox in bboxs:
        ax.add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor="r",
                linewidth=0.5,
                alpha=1,
            )
        )
    output_name = os.path.basename(im_name)
    plt.savefig(im_name, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close("all")


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
    context_pixel = 16  # int(16 * im_scale)

    new_x1 = 0
    new_y1 = 0
    new_x2 = width
    new_y2 = height
    target_size = (320, 320)  # (384, 384)

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
        y_center = int((y2 + y1) / 2)  # math.ceil((y2 + y1) / 2)
        crop_y1 = int(
            y_center - (long_size / 2)
        )  # int(y_center - math.ceil(long_size / 2))
        crop_y2 = int(
            y_center + (long_size / 2)
        )  # int(y_center + math.floor(long_size / 2))

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
        square = np.zeros((3, crop_long_size, crop_long_size), dtype=np.uint8)
        delta = int(
            (crop_long_size - crop_short_size) / 2
        )  # int(math.ceil((crop_long_size - crop_short_size) / 2))
        square_y1 = delta
        square_y2 = delta + crop_short_size

        new_y1 = new_y1 + delta
        new_y2 = new_y2 + delta

        crop_box = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        square[:, square_y1:square_y2, :] = crop_box

        # show_square = np.zeros((crop_long_size, crop_long_size, 3))#, dtype=np.int16)
        # show_crop_box = original_img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        # show_square[square_y1:square_y2, :, :] = show_crop_box
        # show_square = show_square.astype(np.int16)
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
        x_center = int((x2 + x1) / 2)  # math.ceil((x2 + x1) / 2)
        crop_x1 = int(
            x_center - (long_size / 2)
        )  # int(x_center - math.ceil(long_size / 2))
        crop_x2 = int(
            x_center + (long_size / 2)
        )  # int(x_center + math.floor(long_size / 2))

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
        square = np.zeros((3, crop_long_size, crop_long_size), dtype=np.uint8)
        delta = int(
            (crop_long_size - crop_short_size) / 2
        )  # int(math.ceil((crop_long_size - crop_short_size) / 2))
        square_x1 = delta
        square_x2 = delta + crop_short_size

        new_x1 = new_x1 + delta
        new_x2 = new_x2 + delta
        crop_box = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        square[:, :, square_x1:square_x2] = crop_box

        # show_square = np.zeros((crop_long_size, crop_long_size, 3)) #, dtype=np.int16)
        # show_crop_box = original_img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        # show_square[:, square_x1:square_x2, :] = show_crop_box
        # show_square = show_square.astype(np.int16)
    # print(crop_y2 - crop_y1, crop_x2 - crop_x1, bbox, data_height, data_width)

    square = square.astype(np.float32, copy=False)
    square_scale = float(target_size[0]) / long_size
    square = square.transpose(1, 2, 0)
    square = cv2.resize(
        square, target_size, interpolation=cv2.INTER_LINEAR
    )  # None, None, fx=square_scale, fy=square_scale, interpolation=cv2.INTER_LINEAR)
    # square = square.transpose(2,0,1)
    square = square.astype(np.uint8)

    new_x1 = int(new_x1 * square_scale)
    new_y1 = int(new_y1 * square_scale)
    new_x2 = int(new_x2 * square_scale)
    new_y2 = int(new_y2 * square_scale)

    # For test
    # show_square = cv2.resize(show_square, target_size, interpolation=cv2.INTER_LINEAR) # None, None, fx=square_scale, fy=square_scale, interpolation=cv2.INTER_LINEAR)
    # self.vis_image(show_square, [new_x1, new_y1, new_x2, new_y2], img_path.split('/')[-1][:-4]+'_crop.jpg', './test')

    support_data = square
    support_box = np.array([new_x1, new_y1, new_x2, new_y2]).astype(np.float32)
    return support_data, support_box


def main(split_path, split, keepclasses, shot):
    """
    Args:
        split_path: 'datasets/mvtecvoc'
        split: "trainval"
        keepclasses: "all"
        shot: 1, 3, ...
    """
    dirname = "datasets/mvtecvoc"

    classnames = MVTECVOC_ALL_CATEGORIES

    fileids = {}
    for cls in classnames:
        with open(
            os.path.join(
                split_path, "mvtecvocsplit", "box_{}shot_{}_train.txt".format(shot, cls)
            )
        ) as f:
            fileids_ = np.loadtxt(f, dtype=np.str).tolist()
            if isinstance(fileids_, str):
                fileids_ = [fileids_]
            fileids_ = [
                fid.split("/")[-1].split(".jpg" if fid.endswith("jpg") else ".png")[0]
                for fid in fileids_
            ]
            fileids[cls] = fileids_  # dictionary, with "key" of classname

    support_dict = {}
    support_dict["support_box"] = []
    support_dict["category_id"] = []
    support_dict["image_id"] = []
    support_dict["id"] = []
    support_dict["file_path"] = []

    support_path = os.path.join(
        split_path,
        "mvtecvocsplit",
        "mvtecvoc_{}_{}_{}shot".format(split, keepclasses, shot),
    )  # e.g., 'datasets/mvtecvoc/mvtecvocsplit/mvtecvoc_trainval_all_1shot'
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

            anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
            jpeg_file = os.path.join(
                dirname,
                "JPEGImages",
                fileid + (".png" if fileid.startswith("mvtec") else ".jpg"),
            )

            frame_crop_base_path = join(
                support_path, fileid
            )  # e.g., 'datasets/mvtecvoc/mvtecvocsplit/mvtecvoc_trainval_all_1shot/58900'
            if not isdir(frame_crop_base_path):
                makedirs(frame_crop_base_path)

            im = cv2.imread(jpeg_file)
            tree = ET.parse(anno_file)
            count = 0

            for obj in tree.findall("object"):
                cls_ = obj.find("name").text
                if not (cls_ in classnames):
                    continue

                if obj.find("difficult") is not None:
                    difficult = int(obj.find("difficult").text)
                    if difficult == 1:
                        continue

                bbox = obj.find("bndbox")
                bbox = [
                    float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]
                ]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                support_img, support_box = crop_support(im, bbox)

                file_path = join(frame_crop_base_path, "{:04d}.jpg".format(count))
                cv2.imwrite(file_path, support_img)

                support_dict["support_box"].append(support_box.tolist())
                support_dict["category_id"].append(
                    classnames.index(cls_)
                )  # (classnames_all.index(cls_))
                support_dict["image_id"].append(fileid)
                support_dict["id"].append(box_id)
                support_dict["file_path"].append(file_path)
                box_id += 1
                count += 1

    support_df = pd.DataFrame.from_dict(support_dict)
    return support_df


if __name__ == "__main__":
    split = "trainval"
    keepclasses = "all"
    split_path = "datasets/mvtecvoc"

    for shot in [1, 2, 3, 5, 15]:  # FIXME[DONE]: 10
        print(">>> keepclasses={},  shot={}".format(keepclasses, shot))

        since = time.time()
        support_df = main(split_path, split, keepclasses, shot)
        support_df.to_pickle(
            os.path.join(
                split_path,
                "./mvtecvoc_{}_{}_{}shot.pkl".format(split, keepclasses, shot),
            )
        )

        time_elapsed = time.time() - since
        print(
            "Total complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
