"""Train one cross-validation fold of a U-Net on random 512 x 512 crops.

Usage: python train.py --fold 0 [--encoder resnet34] [--epochs 30] [--steps 250]
"""
import argparse
import json
import math
import random
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import CACHE, OUT, SIZE, make_channels

CROP = 512


class CropDataset(Dataset):
    """Random crops from the cached observations. Half of them are centred near a filament."""

    def __init__(self, observations, meta, length, pos_frac=0.5, augment=True, seed=None):
        self.obs, self.meta, self.length = list(observations), meta, length
        self.pos_frac, self.augment, self.seed = pos_frac, augment, seed
        self.pos = {}
        for o in self.obs:
            t = np.load(CACHE / f"{o}_target.npy", mmap_mode="r")
            ys, xs = np.nonzero(t[::4, ::4])
            self.pos[o] = np.stack([ys * 4, xs * 4], 1)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.seed is None else self.seed + i)
        o = self.obs[rng.integers(len(self.obs))]
        m = self.meta[o]
        if rng.random() < self.pos_frac and len(self.pos[o]):
            y, x = self.pos[o][rng.integers(len(self.pos[o]))] + rng.integers(-200, 201, 2)
        else:
            y, x = rng.integers(0, SIZE, 2)
        y0 = int(np.clip(y - CROP // 2, 0, SIZE - CROP))
        x0 = int(np.clip(x - CROP // 2, 0, SIZE - CROP))
        planes = np.load(CACHE / f"{o}_planes.npy", mmap_mode="r")[:, y0:y0 + CROP, x0:x0 + CROP]
        target = np.load(CACHE / f"{o}_target.npy", mmap_mode="r")[y0:y0 + CROP, x0:x0 + CROP]
        x_in = make_channels(np.asarray(planes), m["disk"], m["stats"], x0, y0)
        t = target.astype(np.float32)[None] / 255.0
        if self.augment:
            k = rng.integers(4)
            x_in, t = np.rot90(x_in, k, (1, 2)), np.rot90(t, k, (1, 2))
            if rng.random() < 0.5:
                x_in, t = x_in[:, :, ::-1], t[:, :, ::-1]
            x_in = x_in.copy()
            x_in[0] = x_in[0] * rng.uniform(0.8, 1.25) + rng.uniform(-0.2, 0.2)
        return torch.from_numpy(np.ascontiguousarray(x_in)), torch.from_numpy(np.ascontiguousarray(t))


def build_model(encoder="resnet34", pretrained=True):
    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)


def loss_fn(logits, target):
    """Binary cross-entropy plus soft Dice."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    denom = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (denom + 1)
    return bce + dice.mean()


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
