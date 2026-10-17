"""Filament-level fusion: ensemble the models by voting on their candidates instead of averaging their maps.

Averaging probability maps softens their edges, so the 0.7 threshold cuts different shapes and the
quality head (trained on single-model maps) scores the candidates lower. Here each member map gets its
own candidates and CNN scores. Candidates of different members that overlap with IoU > 0.5 form a
cluster, which scores the mean of its members' scores (0 for a member with no candidate there) and is
represented by its best-scored member. Clusters above the threshold lambda are kept, best first,
skipping any that overlaps a kept one by more than half of the smaller area.

On OOF there are three members (one map per model). On test there are fifteen (one map per model and
fold), and lambda is chosen so the test fusion keeps as many filaments per image as the OOF fusion did.
"""
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pycocotools import mask as mu

from cnn_head import candidates_and_crops, score
from common import SIZE
from fast_pq import pq

LAMBDAS = np.round(np.arange(0.10, 0.40, 0.02), 2)


def fuse(cands, lam, ovmax=0.5):
    """cands: one (rles, scores) pair per member. Returns the kept RLEs."""
    pool = [(m, r, q) for m, (rl, qs) in enumerate(cands) for r, q in zip(rl, qs)]
    if not pool:
        return []
    rles = [p[1] for p in pool]
    iou = mu.iou(rles, rles, [0] * len(rles))
    used = np.zeros(len(pool), bool)
    clusters = []
    for i in np.argsort([-p[2] for p in pool]):
        if used[i]:
            continue
        members, used[i] = [i], True
        for m in range(len(cands)):
            if m == pool[i][0]:
                continue
            js = [j for j in range(len(pool)) if pool[j][0] == m and not used[j] and iou[i, j] > 0.5]
            if js:
                j = max(js, key=lambda j: iou[i, j])
                members.append(j)
                used[j] = True
        clusters.append((sum(pool[j][2] for j in members) / len(cands), i))
    kept = []
    for score_, i in sorted(clusters, key=lambda c: -c[0]):
        if score_ <= lam:
            break
        if kept:
            inter = mu.iou([rles[i]], kept, [1] * len(kept))[0] * mu.area(rles[i])
            if np.any(inter / np.minimum(mu.area(rles[i]), mu.area(kept)) > ovmax):
                continue
        kept.append(rles[i])
    return kept


def _one(item):
    key, map_path, gray_path, disk = item
    return candidates_and_crops(key, map_path, gray_path, disk)


def extract(items, workers=4, chunksize=4):
    """items: (key, map path, image path, disk) tuples. Returns {key: (rles, crops, sizes)}."""
    with Pool(workers) as p:
        return {k: (r, x, s) for k, r, x, s in p.imap(_one, items, chunksize=chunksize)}


def score_member(ext, heads, oof):
    """CNN scores for extracted candidates. {key: (rles, scores)}."""
    keys = list(ext)
    x = np.concatenate([ext[k][1] for k in keys])
    sides = np.concatenate([ext[k][2] for k in keys])
    owner = [k for k in keys for _ in ext[k][0]]
    q = score(heads, x, sides, keys=owner if oof else None)
    out, i = {}, 0
    for k in keys:
        n = len(ext[k][0])
        out[k] = (ext[k][0], q[i:i + n])
        i += n
    return out


def member(items, heads, oof, workers=4):
    """Candidates and CNN scores of one member map. {key: (rles, scores)}."""
    return score_member(extract(items, workers), heads, oof)


def check(per, gt, lambdas=LAMBDAS, log=print):
    """PQ of the OOF fusion at every lambda. per: one {key: (rles, scores)} per member."""
    keys = sorted(gt)
    res = []
    for lam in lambdas:
        keep = {o: fuse([p[o] for p in per], lam) for o in keys}
        res.append((pq(gt, keep), float(lam), float(np.mean([len(v) for v in keep.values()]))))
        log(f"lambda {lam:.2f}: PQ {res[-1][0]:.4f}, kept per image {res[-1][2]:.2f}")
    best = max(res)
    return {"pq": best[0], "lambda": best[1], "kept_per_image": best[2], "curve": res}


def keep_at(per, keys, lam):
    return {o: fuse([p[o] for p in per], lam) for o in keys}


def match_rate(per, keys, kept_per_image, lambdas=LAMBDAS):
    """The lambda whose test fusion keeps the closest number of filaments per image to the OOF fusion."""
    rates = {float(lam): np.mean([len(fuse([p[o] for p in per], lam)) for o in keys]) for lam in lambdas}
    lam = min(rates, key=lambda l: abs(rates[l] - kept_per_image))
    return lam, rates[lam]


EMPTY_RLE = mu.encode(np.zeros((SIZE, SIZE), np.uint8, order="F"))["counts"].decode()


def submission(per, keys, lam, fallback=None):
    """Submission table. An image where nothing passes lambda gets the best candidate of any member.
    An image where no member has any candidate at all gets the rows of the fallback table if one is
    given (to reproduce an earlier file exactly), otherwise one empty mask, so every image has a row."""
    rows, notes = [], {"best_candidate": [], "fallback": [], "empty": []}
    for img in keys:
        kept = fuse([p[img] for p in per], lam)
        if not kept:
            allc = [(q, r) for p in per for r, q in zip(*p[img])]
            if allc:
                kept = [max(allc, key=lambda c: c[0])[1]]
                notes["best_candidate"].append(img)
        if kept:
            rows += [{"filament_id": f"{img}_{k}", "segmentation_rle": r["counts"].decode()} for k, r in enumerate(kept, start=1)]
            continue
        fb = [] if fallback is None else list(fallback[fallback.filament_id.str.rsplit("_", n=1).str[0] == img].segmentation_rle)
        if fb:
            rows += [{"filament_id": f"{img}_{k}", "segmentation_rle": c} for k, c in enumerate(fb, start=1)]
            notes["fallback"].append(img)
        else:
            rows.append({"filament_id": f"{img}_1", "segmentation_rle": EMPTY_RLE})
            notes["empty"].append(img)
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    assert sub.filament_id.is_unique
    assert set(sub.filament_id.str.rsplit("_", n=1).str[0]) == set(keys)
    return sub, notes
