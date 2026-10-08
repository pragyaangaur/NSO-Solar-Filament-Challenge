"""Shared paths, data loading and cross-validation folds."""
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
# The competition data can live anywhere. Set FILAMENT_DATA to point at the
# extracted MAGFiLO_1.0_Kaggle_2026 folder if it is not in data/.
DATA = Path(os.environ.get("FILAMENT_DATA", PROJECT / "data" / "MAGFiLO_1.0_Kaggle_2026"))
TRAIN_IMAGES = DATA / "train" / "train_images"
TEST_IMAGES = DATA / "test" / "test_images"
TRAIN_JSON = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
CACHE = PROJECT / "data" / "cache"
OUT = PROJECT / "outputs"
SIZE = 2048
N_FOLDS = 5


def load_train_index():
    """Return {observation: {"file": name, "annotators": {annotator_image_id: [polygons]}}}.

    In the COCO file every image id has the form "<annotator>-<observation>", so
    one observation can appear several times, once per annotator.
    """
    coco = json.load(open(TRAIN_JSON))
    polys = defaultdict(list)
    for a in coco["annotations"]:
        polys[a["image_id"]].append(a["segmentation"])
    obs = {}
    for im in coco["images"]:
        o = im["id"].split("-", 1)[1]
        obs.setdefault(o, {"file": im["file_name"], "annotators": {}})
        obs[o]["annotators"][im["id"]] = polys[im["id"]]
    return dict(sorted(obs.items()))


def fold_of(observation: str) -> int:
    """Deterministic fold per observation, so all annotators of one image stay in the same fold."""
    return int(hashlib.md5(observation.encode()).hexdigest(), 16) % N_FOLDS


def read_gray(path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
