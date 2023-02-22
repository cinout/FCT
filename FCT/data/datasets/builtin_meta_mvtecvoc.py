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
    # FIXME: update
    # "bird",
    # "bus",
    # "cow",
    # "motorbike",
    # "sofa",
    "nectarine",
    "orange",
    "cereal",
    "almond_mix",
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
