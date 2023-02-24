# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Created on Wednesday, September 28, 2022

@author: Guangxing Han
"""


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
    "nectarine",
    "orange",
    "cereal",
    "almond_mix",
    # FIXME: update
    "short_screw",
    "long_screw",
    "washer",
    "screw_nut",
    "pushpin",
    "clamp_2l",
    "clamp_2r",
    "cable_yellow",
    "clamp_3l",
    "clamp_3r",
    "cable_blue",
    "clamp_5l",
    "clamp_5r",
    "cable_red",
    "juice_banana",
    "label_banana",
    "juice_orange",
    "label_orange",
    "juice_cherry",
    "label_cherry",
    "label_100",
]

# FIXME: update
MVTECVOC_NOVEL_CATEGORIES = [
    # "bird", "bus", "cow", "motorbike", "sofa"
    "nectarine",
    "orange",
    "cereal",
    "almond_mix",
]

MVTECVOC_BASE_CATEGORIES = [
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
]


def _get_mvtecvoc_fewshot_instances_meta():
    ret = {
        "thing_classes": MVTECVOC_ALL_CATEGORIES,
        "novel_classes": MVTECVOC_NOVEL_CATEGORIES,
        "base_classes": MVTECVOC_BASE_CATEGORIES,
    }
    return ret


def _get_builtin_metadata_mvtecvoc(dataset_name):
    if dataset_name == "mvtecvoc_fewshot":
        return _get_mvtecvoc_fewshot_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
