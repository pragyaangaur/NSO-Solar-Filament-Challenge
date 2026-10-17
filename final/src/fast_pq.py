"""Fast PQ on RLEs, numerically identical to the official metric.

The official code decodes every mask to dense 2048 x 2048 tensors. Here IoUs come from
pycocotools on RLEs, which is far faster inside search loops. It also returns the matched
IoU of every prediction, which is the training target of the quality head.
"""
import numpy as np
from pycocotools import mask as mu


def pq(gt, preds, return_details=False):
    """gt: {obs: [annotator RLE lists]}, preds: {obs: [RLE]}.

    Scores every annotator-image of every observation in gt, as the official metric does.
    Predictions for observations not in gt are ignored.
    """
    s_iou, n_tp, n_fp, n_fn = 0.0, 0, 0, 0
    details = {}
    for obs, annotators in gt.items():
        p = preds.get(obs, [])
        per_pred = np.zeros((len(annotators), len(p)))
        for a, g in enumerate(annotators):
            if not g:
                n_fp += len(p)
                continue
            if not p:
                n_fn += len(g)
                continue
            iou = mu.iou(p, g, [0] * len(g)).T  # (n_gt, n_pred)
            hit = iou > 0.5
            s_iou += iou[hit].sum()
            n_tp += hit.sum()
            n_fp += (hit.sum(0) == 0).sum()
            n_fn += (hit.sum(1) == 0).sum()
            per_pred[a] = (iou * hit).max(0)
        details[obs] = per_pred
    denom = n_tp + 0.5 * n_fp + 0.5 * n_fn
    score = s_iou / denom if denom else 0.0
    if return_details:
        return score, {"tp": int(n_tp), "fp": int(n_fp), "fn": int(n_fn), "sum_iou": float(s_iou)}, details
    return score
