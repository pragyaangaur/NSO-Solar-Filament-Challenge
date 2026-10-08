"""Summary statistics of the training annotations.

It also writes the ground truth in the official submission format
(data/train_gt_rle.parquet), which is useful for scoring local predictions with
the organisers' self-evaluation notebook.
"""
from collections import Counter, defaultdict
import json

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from common import DATA, PROJECT, SIZE, TRAIN_JSON


def to_rle(ann):
    rle = mask_util.merge(mask_util.frPyObjects(ann["segmentation"], SIZE, SIZE))
    return rle["counts"].decode("ascii")


def main():
    coco = json.load(open(TRAIN_JSON))
    images = {im["id"]: im for im in coco["images"]}
    print("data folder:", DATA)
    print("annotator-images:", len(images), "| annotations:", len(coco["annotations"]))

    obs_to_ann = defaultdict(list)
    for iid in images:
        obs_to_ann[iid.split("-", 1)[1]].append(iid)
    print("unique observations:", len(obs_to_ann))
    print("annotators per observation:", dict(Counter(len(v) for v in obs_to_ann.values())))

    per_img = Counter(a["image_id"] for a in coco["annotations"])
    counts = np.array([per_img.get(i, 0) for i in images])
    print(f"filaments per annotator-image: mean {counts.mean():.2f}, median {np.median(counts):.0f}, max {counts.max()}")
    areas = np.array([a["area"] for a in coco["annotations"]])
    print("filament area percentiles 5/25/50/75/95:", np.percentile(areas, [5, 25, 50, 75, 95]).round(0).tolist())
    # The last two characters of an observation name are the GONG site code.
    print("observations per site:", dict(Counter(o[-2:] for o in obs_to_ann)))
    print("observations per year:", dict(sorted(Counter(o[:4] for o in obs_to_ann).items())))

    rows = [{"filament_id": f'{a["image_id"]}_{a["id"]}', "segmentation_rle": to_rle(a)} for a in coco["annotations"]]
    out = PROJECT / "data" / "train_gt_rle.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
