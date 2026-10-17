"""Shared constants, folds, solar-disk detection and probability-map input and output."""
import hashlib
from pathlib import Path

import cv2
import numpy as np

SIZE = 2048          # native image size
MAP_SIZE = 1024      # probability maps are stored at 1024 and stretched back to 2048 (the 1024 round trip)
N_FOLDS = 5


def fold_of(obs: str) -> int:
    """Model fold of an observation. All annotators of one image share a fold."""
    return int(hashlib.md5(obs.encode()).hexdigest(), 16) % N_FOLDS


def cv_group(obs: str, k: int = 5) -> int:
    """Cross-fitting group of the quality head, independent of the model folds."""
    return int(hashlib.md5(("head" + obs).encode()).hexdigest(), 16) % k


def read_gray(path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def find_disk(gray: np.ndarray):
    """Centre (cx, cy) and radius of the solar disk in pixels, as the minimum enclosing circle of the
    largest region brighter than 30.

    This simple detector produced the frozen geometry in data/meta.json that every result was scored
    with. On many GONG frames it fits the grey halo around the Sun, so its radius can run up to about
    15 per cent too large. The pipeline only uses the disk to drop candidates whose centre lies outside
    it (limb_max 1.0), so the larger radius drops almost nothing.
    """
    m = (gray > 30).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return float(cx), float(cy), float(r)


def save_map(path, prob_2048: np.ndarray):
    """Save a uint8 2048 map as a lossless 1024 PNG (area downsampling, the first half of the round trip)."""
    p = Path(path).with_suffix(".png")
    p.parent.mkdir(parents=True, exist_ok=True)
    small = cv2.resize(np.asarray(prob_2048, np.uint8), (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(p), small)


def load_map(path) -> np.ndarray:
    """Load a uint8 probability map at 2048. Maps stored at 1024 are stretched back with bilinear
    interpolation. Maps already stored at 2048 (the research folder keeps them that way) load as they are.
    Older research runs wrote .npy files, which are accepted too."""
    p = Path(path)
    for cand in (p.with_suffix(".png"), p.with_suffix(".npy")):
        if cand.exists():
            a = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE) if cand.suffix == ".png" else np.load(cand)
            if a.shape[0] != SIZE:
                a = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            return a
    raise FileNotFoundError(p)


def map_keys(folder) -> list:
    """Image names of the maps in a folder."""
    return sorted({p.stem for p in Path(folder).glob("*.png")} | {p.stem for p in Path(folder).glob("*.npy")})
