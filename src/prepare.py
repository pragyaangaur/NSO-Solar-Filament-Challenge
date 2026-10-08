"""Cache image planes, training targets and disk geometry for every observation.

The training target is the union of all annotators' masks for an observation.
Everything is written to data/cache together with meta.json, which the other
scripts read.

Usage: python prepare.py [--limit N]
"""
import argparse
import json

import numpy as np
from tqdm import tqdm

from common import CACHE, TEST_IMAGES, TRAIN_IMAGES, find_disk, fold_of, image_planes, load_train_index, rasterize, read_gray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only cache the first N observations of each split (for quick tests)")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    obs = list(load_train_index().items())
    test = sorted(TEST_IMAGES.glob("*.jpeg"))
    if args.limit:
        obs, test = obs[:args.limit], test[:args.limit]

    meta = {}
    for o, info in tqdm(obs, desc="train", mininterval=30):
        gray = read_gray(TRAIN_IMAGES / info["file"])
        disk = find_disk(gray)
        planes, stats = image_planes(gray, disk)
        np.save(CACHE / f"{o}_planes.npy", planes)
        target = np.max([rasterize(p) for p in info["annotators"].values()], axis=0)
        np.save(CACHE / f"{o}_target.npy", target * 255)
        meta[o] = {"file": info["file"], "disk": disk, "stats": stats, "fold": fold_of(o)}

    test_meta = {}
    for p in tqdm(test, desc="test", mininterval=30):
        gray = read_gray(p)
        disk = find_disk(gray)
        planes, stats = image_planes(gray, disk)
        np.save(CACHE / f"{p.stem}_planes.npy", planes)
        test_meta[p.stem] = {"file": p.name, "disk": disk, "stats": stats}

    json.dump({"train": meta, "test": test_meta}, open(CACHE / "meta.json", "w"), indent=1)
    print("observations per fold:", np.bincount([m["fold"] for m in meta.values()]).tolist())


if __name__ == "__main__":
    main()
