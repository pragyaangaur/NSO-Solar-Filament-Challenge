"""Full-resolution probability maps for the test images from one trained fold.

The whole 2048 x 2048 image goes through the network in one pass, averaged
over horizontal and vertical flips. Maps are saved as uint8 (probability x 255).

Usage: python predict.py --fold 0 [--name unet_baseline] [--tta 2]
"""
import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from common import CACHE, OUT, make_channels
from train import autocast, build_model, device


@torch.no_grad()
def predict_map(model, planes, disk, stats, dev, tta=2):
    x = torch.from_numpy(make_channels(planes, disk, stats))[None].to(dev)
    flips = [[], [3], [2], [2, 3]][:tta]
    acc = torch.zeros(1, 1, *x.shape[2:], device=dev)
    for f in flips:
        xi = torch.flip(x, f) if f else x
        with autocast(dev):
            y = torch.sigmoid(model(xi).float())
        acc += torch.flip(y, f) if f else y
    return (acc / len(flips))[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--name", default="unet_baseline")
    ap.add_argument("--tta", type=int, default=2, help="number of flip variants to average (1 to 4)")
    args = ap.parse_args()

    base = OUT / args.name
    meta = json.load(open(CACHE / "meta.json"))["test"]
    dev = device()
    model = build_model(json.load(open(base / "model.json"))["encoder"], pretrained=False).to(dev).eval()
    model.load_state_dict(torch.load(base / f"fold{args.fold}" / "last.pt", map_location=dev))

    out = base / f"test_fold{args.fold}"
    out.mkdir(exist_ok=True)
    for img, m in tqdm(meta.items(), desc=f"fold {args.fold}", mininterval=30):
        p = predict_map(model, np.load(CACHE / f"{img}_planes.npy"), m["disk"], m["stats"], dev, args.tta)
        np.save(out / f"{img}.npy", (p * 255).round().astype(np.uint8))


if __name__ == "__main__":
    main()
