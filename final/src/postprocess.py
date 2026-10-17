"""Turn a probability map into candidate filaments.

The steps follow the MAGFiLO annotation protocol: each filament is one region with no holes,
nearby fragments of one filament are grouped into one instance, and tiny regions are dropped.
The final pipeline uses threshold 178 (0.7), grouping 9 pixels and minimum area 50 pixels. The
candidates are not filtered here. The CNN quality head and the fusion decide which ones to keep.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi

MAX_REGIONS = 150  # real images hold at most about 26 filaments, so only the largest regions can matter


@dataclass(frozen=True)
class Params:
    thr: int = 178          # probability threshold * 255
    group_dist: int = 9     # fragments closer than this many pixels become one instance
    min_area: int = 50      # drop instances smaller than this (pixels)
    limb_max: float = 1.0   # drop instances whose centroid |x - cx| / R is above this


PARAMS = Params()


def instances(prob: np.ndarray, disk, params: Params = PARAMS):
    """Candidate filaments of one map as a list of RLE dicts, in a fixed order."""
    cx, cy, R = disk
    fg = prob >= params.thr
    if params.group_dist > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.group_dist, params.group_dist))
        grouped = cv2.dilate(fg.astype(np.uint8), k)
        lab, n = ndi.label(grouped, structure=np.ones((3, 3)))
        lab = lab * fg
    else:
        lab, n = ndi.label(fg, structure=np.ones((3, 3)))
    objects = ndi.find_objects(lab)
    sizes = ndi.sum_labels(np.ones_like(lab, dtype=np.uint8), lab, index=np.arange(1, len(objects) + 1))
    allowed = set((np.argsort(-sizes)[:MAX_REGIONS] + 1).tolist())
    full = np.zeros(prob.shape, np.uint8, order="F")
    out = []
    for i, sl in enumerate(objects, start=1):
        if sl is None or i not in allowed or sizes[i - 1] < params.min_area:
            continue
        pad = 3
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, prob.shape[0])
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, prob.shape[1])
        m = lab[y0:y1, x0:x1] == i
        if int(m.sum()) < params.min_area:
            continue
        m = ndi.binary_fill_holes(m)  # one region with no holes
        ys, xs = np.nonzero(m)
        if abs((xs.mean() + x0 - cx) / R) > params.limb_max:
            continue
        full[y0:y1, x0:x1] = m
        out.append(mu.encode(full))
        full[y0:y1, x0:x1] = 0
    return out
