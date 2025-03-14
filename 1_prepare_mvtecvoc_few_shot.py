import argparse
import copy
import os
import random
import xml.etree.ElementTree as ET
import numpy as np

# from fsdet.utils.file_io import PathManager

from iopath.common.file_io import (
    HTTPURLHandler,
    OneDrivePathHandler,
    PathHandler,
    PathManager as PathManagerBase,
)

__all__ = ["PathManager", "PathHandler"]


PathManager = PathManagerBase()
"""
This is a detectron2 project-specific PathManager.
We try to stay away from global PathManager in fvcore as it
introduces potential conflicts among other libraries.
"""


class Detectron2Handler(PathHandler):
    """
    Resolve anything that's hosted under detectron2's namespace.
    """

    PREFIX = "detectron2://"
    S3_DETECTRON2_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.S3_DETECTRON2_PREFIX + name)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


class FsDetHandler(PathHandler):
    """
    Resolve anything that's in FsDet model zoo.
    """

    PREFIX = "fsdet://"
    URL_PREFIX = "http://dl.yf.io/fs-det/models/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.URL_PREFIX + name)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())
PathManager.register_handler(Detectron2Handler())
PathManager.register_handler(FsDetHandler())


MVTECVOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    "nectarine",
    "orange",
    "cereal",
    "almond_mix",
    "short_screw",
    "long_screw",
    "washer",
    "screw_nut",
    "tools_bag",
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
    # FIXME: update (append instead of replacing)
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0], help="Range of seeds"
    )
    args = parser.parse_args()
    return args


def generate_seeds():
    data = []
    data_per_cat = {c: [] for c in MVTECVOC_CLASSES}

    with PathManager.open("datasets/mvtecvoc/ImageSets/Main/trainval.txt") as f:
        fileids = np.loadtxt(f, dtype=np.str).tolist()
        data.extend(fileids)

    for fileid in data:
        anno_file = os.path.join("datasets/mvtecvoc", "Annotations", fileid + ".xml")
        tree = ET.parse(anno_file)
        clses = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            clses.append(cls)  # find all classes of object in this anno_file
        for cls in set(clses):
            data_per_cat[cls].append(
                anno_file
            )  # all annotation file paths by classname

    result = {cls: {} for cls in data_per_cat.keys()}
    shots = [1, 2, 3, 5, 10, 15, 20, 30]  # FIXME:

    # we don't care seeds here, refer to prepare_pascol_xxx.py if you want to use different seeds
    random.seed(0)
    for c in data_per_cat.keys():  # for each classname
        for shot in shots:  # for each shot number
            c_data = []

            if len(data_per_cat[c]) < shot:
                shots_c = data_per_cat[c]
            else:
                shots_c = random.sample(
                    data_per_cat[c], shot
                )  # anno file paths for shots

            for s in shots_c:
                tree = ET.parse(s)
                file = tree.find("filename").text  # contains suffix
                name = "datasets/mvtecvoc/JPEGImages/{}".format(file)  # image file path
                c_data.append(name)

            result[c][shot] = c_data  # image file paths by (1) classname (2) #shot

    save_path = "datasets/mvtecvoc/mvtecvocsplit"
    os.makedirs(save_path, exist_ok=True)
    for c in result.keys():
        for shot in result[c].keys():
            filename = "box_{}shot_{}_train.txt".format(shot, c)
            with open(os.path.join(save_path, filename), "w") as fp:
                fp.write("\n".join(result[c][shot]) + "\n")


if __name__ == "__main__":
    generate_seeds()
