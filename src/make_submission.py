"""Turn averaged probability maps into a submission of filament instances.

Each connected region above the threshold becomes one filament. Regions smaller
than --min-area pixels are dropped as noise. Every test image gets at least one
instance, because the metric gives nothing for an image with no prediction.

Usage: python make_submission.py [--name unet_baseline] [--thr 0.5] [--min-area 300]
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from common import CACHE, OUT


def encode(mask: np.ndarray) -> str:
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def instances(prob: np.ndarray, thr: float, min_area: int):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((prob >= thr).astype(np.uint8), connectivity=8)
    keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= min_area]
    # Most confident regions first.
    keep.sort(key=lambda k: -prob[labels == k].mean())
    return [labels == k for k in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="unet_baseline")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=300)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    base = OUT / args.name
    meta = json.load(open(CACHE / "meta.json"))["test"]
    fold_dirs = sorted(base.glob("test_fold*"))
    print("averaging", [d.name for d in fold_dirs])

    rows = []
    for img in sorted(meta):
        prob = np.mean([np.load(d / f"{img}.npy").astype(np.float32) for d in fold_dirs], 0) / 255
        masks = instances(prob, args.thr, args.min_area)
        thr, area = args.thr, args.min_area
        while not masks and thr > 0.1:  # relax step by step until something is found
            thr, area = thr - 0.1, max(area // 2, 30)
            masks = instances(prob, thr, area)
        for k, m in enumerate(masks, start=1):
            rows.append({"filament_id": f"{img}_{k}", "segmentation_rle": encode(m)})

    sub = pd.DataFrame(rows)
    sub.to_csv(base / args.out, index=False)
    print(f"wrote {base / args.out}: {len(sub)} filaments over {sub.filament_id.str.rsplit('_', n=1).str[0].nunique()} images")


if __name__ == "__main__":
    main()
