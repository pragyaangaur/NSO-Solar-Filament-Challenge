"""Read the competition data: the train index, the ground truth as RLEs, and the solar-disk geometry.

One train image (an observation) is annotated by one to three annotators. The COCO file has one
image entry per annotator, with id "<annotator>-<observation>". Everything here is keyed by observation.
"""
import json
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from pycocotools import mask as mu

from common import SIZE, find_disk, fold_of, read_gray

TRAIN_JSON = "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def load_index(data_dir):
    """{observation: {"file": name, "annotators": {annotator_image_id: [annotations]}}}, sorted by observation."""
    coco = json.load(open(Path(data_dir) / TRAIN_JSON))
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    obs = {}
    for im in coco["images"]:
        o = im["id"].split("-", 1)[1]
        obs.setdefault(o, {"file": im["file_name"], "annotators": {}})
        obs[o]["annotators"][im["id"]] = anns[im["id"]]
    return dict(sorted(obs.items()))


def test_images(data_dir):
    return sorted(p.stem for p in (Path(data_dir) / "test" / "test_images").glob("*.jpeg"))


def image_path(data_dir, key, index=None):
    """Path of a train observation (index given) or a test image."""
    if index is not None and key in index:
        return Path(data_dir) / "train" / "train_images" / Path(index[key]["file"]).name
    return Path(data_dir) / "test" / "test_images" / f"{key}.jpeg"


def to_rle(ann):
    """One annotation's polygons as one compressed RLE string, the official ground-truth format."""
    return mu.merge(mu.frPyObjects(ann["segmentation"], SIZE, SIZE))["counts"].decode("ascii")


def gt_table(index, observations=None):
    """Ground truth in the submission format: filament_id "<annotator>-<observation>_<annotation id>".
    Annotator-images with no filaments have no rows, as in the official ground truth."""
    rows = []
    for o in observations or index:
        for aid, anns in index[o]["annotators"].items():
            rows += [{"filament_id": f"{aid}_{a['id']}", "segmentation_rle": to_rle(a)} for a in anns]
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def gt_rles(gt_df):
    """{observation: [one list of RLE dicts per annotator]} from a ground-truth table."""
    gt = defaultdict(dict)
    key = gt_df.filament_id.str.split("_", n=1).str[0]
    for k, counts in zip(key, gt_df.segmentation_rle):
        ann, obs = k.split("-", 1)
        gt[obs].setdefault(ann, []).append({"size": [SIZE, SIZE], "counts": counts.encode()})
    return {o: list(v.values()) for o, v in gt.items()}


def _disk(key, data_dir, index):
    return key, find_disk(read_gray(image_path(data_dir, key, index)))


def build_meta(data_dir, index, train_keys, test_keys, workers=4):
    """Disk geometry of every image, detected afresh. {"train": {obs: {...}}, "test": {img: {...}}}."""
    with Pool(workers) as p:
        tr = dict(p.map(partial(_disk, data_dir=data_dir, index=index), train_keys, chunksize=8))
        te = dict(p.map(partial(_disk, data_dir=data_dir, index=None), test_keys, chunksize=8))
    return {"train": {o: {"file": index[o]["file"], "disk": tr[o], "fold": fold_of(o),
                          "n_annotators": len(index[o]["annotators"])} for o in train_keys},
            "test": {k: {"file": f"{k}.jpeg", "disk": te[k]} for k in test_keys}}
