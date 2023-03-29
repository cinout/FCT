# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Created on Wednesday, September 28, 2022

@author: Guangxing Han
"""


MVTECVOC_ALL_CATEGORIES = {
    "breakfast_box": [
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
    ],
    "juice_bottle": [
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
        "juice_banana",
        "label_banana",
        "juice_orange",
        "label_orange",
        "juice_cherry",
        "label_cherry",
        "label_100",
    ],
    "pushpins": [
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
        "motorbike",
        "sofa",
        "pushpin",
    ],
    "screw_bag": [
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
        "short_screw",
        "long_screw",
        "washer",
        "screw_nut",
        "tools_bag",
    ],
    "splicing_connectors": [
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
        "clamp_2",
        "cable_yellow",
        "clamp_3",
        "cable_blue",
        "clamp_5",
        "cable_red",
    ],
}


MVTECVOC_NOVEL_CATEGORIES = {
    "breakfast_box": [
        "nectarine",
        "orange",
        "cereal",
        "almond_mix",
    ],
    "juice_bottle": [
        "juice_banana",
        "label_banana",
        "juice_orange",
        "label_orange",
        "juice_cherry",
        "label_cherry",
        "label_100",
    ],
    "pushpins": [
        "pushpin",
        "motorbike",  # to create at least 2 ways (for pos/neg branches)
        "sofa",
    ],
    "screw_bag": [
        "short_screw",
        "long_screw",
        "washer",
        "screw_nut",
        "tools_bag",
    ],
    "splicing_connectors": [
        "clamp_2",
        "cable_yellow",
        "clamp_3",
        "cable_blue",
        "clamp_5",
        "cable_red",
    ],
}

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


def _get_mvtecvoc_fewshot_instances_meta(category):
    ret = {
        "thing_classes": MVTECVOC_ALL_CATEGORIES[category],
        "novel_classes": MVTECVOC_NOVEL_CATEGORIES[category],
        "base_classes": MVTECVOC_BASE_CATEGORIES,
    }
    return ret


def _get_builtin_metadata_mvtecvoc(dataset_name, category):
    if dataset_name == "mvtecvoc_fewshot":
        return _get_mvtecvoc_fewshot_instances_meta(category)
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
