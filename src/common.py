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


def find_disk(gray: np.ndarray):
    """Centre (cx, cy) and radius of the solar disk, from the largest bright region."""
    m = (gray > 30).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return float(cx), float(cy), float(r)


def disk_mask(shape, disk) -> np.ndarray:
    cx, cy, r = disk
    yy, xx = np.ogrid[0:shape[0], 0:shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))


def image_planes(gray: np.ndarray, disk):
    """uint8 planes (gray, CLAHE) and the mean and spread of the brightness on the disk."""
    vals = gray[disk_mask(gray.shape, disk)]
    stats = (float(vals.mean()), float(vals.std()) + 1e-6)
    return np.stack([gray, _clahe.apply(gray)]), stats


def make_channels(planes: np.ndarray, disk, stats, x0=0, y0=0) -> np.ndarray:
    """3 x h x w float32 network input for a window of the image that starts at (x0, y0).

    The channels are the image normalised by the disk brightness, the CLAHE
    image and a disk mask. Pixels off the disk are set to zero in the first channel.
    """
    cx, cy, r = disk
    mu, sd = stats
    h, w = planes.shape[1:]
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float32)
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2).astype(np.float32)
    a = (planes[0].astype(np.float32) - mu) / sd * inside
    b = planes[1].astype(np.float32) / 127.5 - 1
    return np.stack([a, b, inside - 0.5]).astype(np.float32)


def rasterize(polygons, shape=(SIZE, SIZE)) -> np.ndarray:
    """Union mask of a list of COCO polygon segmentations."""
    m = np.zeros(shape, np.uint8)
    for seg in polygons:
        for p in seg:
            pts = np.round(np.array(p).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
    return m
