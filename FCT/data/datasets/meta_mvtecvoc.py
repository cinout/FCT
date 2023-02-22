# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified on Wednesday, September 28, 2022

@author: Guangxing Han
"""

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


__all__ = ["register_meta_mvtecvoc"]


def load_filtered_mvtecvoc_instances(
    name: str, dirname: str, split: str, classnames: str
):
    """
    to Detectron2 format.
    Args:
        name: e.g., mvtecvoc_trainval_novel_1shot
        dirname: e.g., datasets/mvtecvoc
        split (str): one of "train", "test", "val", "trainval", only used in base training
        classnames: a list of class names (all/base/novel), defined in "meta_mvtecvoc.py"
    """
    is_shots = "shot" in name
    if is_shots:
        fileids = {}
        split_dir = os.path.join("datasets", "mvtecvoc", "mvtecvocsplit")
        if "seed" in name:
            shot = name.split("_")[-2].split("shot")[0]
            seed = int(name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            shot = name.split("_")[-1].split("shot")[0]
        for cls in classnames:
            with PathManager.open(
                os.path.join(split_dir, "box_{}shot_{}_train.txt".format(shot, cls))
            ) as f:
                fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [
                    fid.split("/")[-1].split(".jpg" if fid.endswith("jpg") else ".png")[
                        0
                    ]
                    for fid in fileids_
                ]  # list of image file ids without .jpg/.png suffix
                fileids[cls] = fileids_  # dictionary, with "key" of classname

    else:
        if name != "mvtecvoc_test_all":  # FIXME: update
            with PathManager.open(
                os.path.join(dirname, "ImageSets", "Main", split + ".txt")
            ) as f:
                fileids = np.loadtxt(f, dtype=np.str)  # image file ids

    dicts = []
    if is_shots:
        for cls, fileids_ in fileids.items():
            dicts_ = []
            tot_instance = 0
            for fileid in fileids_:
                dirname = os.path.join("datasets", "mvtecvoc")
                anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
                jpeg_file = os.path.join(
                    dirname,
                    "JPEGImages",
                    fileid + (".png" if fileid.startswith("mvtec") else ".jpg"),
                )

                tree = ET.parse(anno_file)
                r = {
                    "file_name": jpeg_file,
                    "image_id": fileid,
                    "height": int(tree.findall("./size/height")[0].text),
                    "width": int(tree.findall("./size/width")[0].text),
                }

                instances = []
                for obj in tree.findall("object"):
                    cls_ = obj.find("name").text

                    difficult = obj.find("difficult")
                    if difficult is not None:
                        difficult = int(difficult.text)
                        if difficult == 1:
                            continue
                    if cls != cls_:
                        continue
                    bbox = obj.find("bndbox")
                    bbox = [
                        float(bbox.find(x).text)
                        for x in ["xmin", "ymin", "xmax", "ymax"]
                    ]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    instances.append(
                        {
                            "category_id": classnames.index(cls_),
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                        }
                    )
                if tot_instance + len(instances) <= int(shot):
                    r["annotations"] = instances
                    tot_instance += len(instances)
                else:
                    r["annotations"] = instances[: int(shot) - tot_instance]
                    tot_instance = int(shot)
                dicts_.append(r)
                if tot_instance >= int(shot):
                    break
            # if len(dicts_) > int(shot):
            #     dicts_ = np.random.choice(dicts_, int(shot), replace=False)
            dicts.extend(dicts_)
    else:
        if name == "mvtecvoc_test_all":  # FIXME: update
            test_samples_path = os.path.join(
                dirname, "mvtec_novel_samples"
            )  # dirname is datasets/mvtecvoc
            for file in os.listdir(test_samples_path):
                file_name = os.path.join(test_samples_path, file)
                image_id = file.split(".png")[0]
                r = {
                    "file_name": file_name,
                    "image_id": image_id,
                    "height": int(1280),  # FIXME: update: breakfast:1280, screw:1100
                    "width": int(1600),  # FIXME: update: breakfast/screw:1600
                    "annotations": [],
                }
                dicts.append(r)
        else:
            for fileid in fileids:
                anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
                jpeg_file = os.path.join(
                    dirname,
                    "JPEGImages",
                    fileid + (".png" if fileid.startswith("mvtec") else ".jpg"),
                )

                tree = ET.parse(anno_file)

                r = {
                    "file_name": jpeg_file,
                    "image_id": fileid,
                    "height": int(tree.findall("./size/height")[0].text),
                    "width": int(tree.findall("./size/width")[0].text),
                }
                instances = []

                for obj in tree.findall("object"):
                    cls = obj.find("name").text
                    if not (cls in classnames):
                        continue
                    difficult = obj.find("difficult")
                    if difficult is not None:
                        difficult = int(difficult.text)
                        if difficult == 1:
                            continue  # voc do not use difficult objects to calculate mAP
                    bbox = obj.find("bndbox")
                    bbox = [
                        float(bbox.find(x).text)
                        for x in ["xmin", "ymin", "xmax", "ymax"]
                    ]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    instances.append(
                        {
                            "category_id": classnames.index(cls),
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                        }
                    )
                r["annotations"] = instances
                dicts.append(r)
    return dicts


def register_meta_mvtecvoc(name, metadata, dirname, split, keepclasses):
    if keepclasses.startswith("base_novel"):
        thing_classes = metadata["thing_classes"]
    elif keepclasses.startswith("base"):
        thing_classes = metadata["base_classes"]
    elif keepclasses.startswith("novel"):
        thing_classes = metadata["novel_classes"]

    DatasetCatalog.register(
        name,
        lambda: load_filtered_mvtecvoc_instances(name, dirname, split, thing_classes),
    )

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        dirname=dirname,
        year=2007,
        split=split,
        base_classes=metadata["base_classes"],
        novel_classes=metadata["novel_classes"],
    )
