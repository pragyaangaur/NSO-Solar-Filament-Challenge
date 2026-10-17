"""Self-evaluation with the official scorer: PQ, its parts, Dice and IoU summaries, and per-image numbers.

The overlap table comes from the organisers' get_overlap_df, so every number here uses the same IoU and
Dice matrices as the official score. The plots are in organiser_plots.py.
"""
import numpy as np
import pandas as pd

from official_metric import get_overlap_df, get_pq_score


def to_frame(preds):
    """{image: [RLE dicts]} to a submission-format table."""
    rows = [{"filament_id": f"{img}_{k}", "segmentation_rle": r["counts"].decode() if isinstance(r["counts"], bytes) else r["counts"]}
            for img, rl in preds.items() for k, r in enumerate(rl, start=1)]
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def summary(overlap_df, thr=0.5):
    """Numbers for the quantitative rubric, all from the official overlap matrices."""
    tp_iou, tp_dice, best_dice_gt, best_iou_gt = [], [], [], []
    fp = fn = 0
    m2o = o2m = 0  # predictions that touch more than one GT filament, GT filaments touched by more than one prediction
    per_image = []
    for row in overlap_df.itertuples(index=False):
        iou, dice = np.asarray(row.iou_matrix, float), np.asarray(row.dice_matrix, float)
        n_gt, n_pred = row.n_gt, row.n_pred
        if n_gt == 0:
            fp += n_pred
            per_image.append((row.annotator_image, 0.0, 0, n_pred, 0))
            continue
        if n_pred == 0:
            fn += n_gt
            best_dice_gt += [0.0] * n_gt
            best_iou_gt += [0.0] * n_gt
            per_image.append((row.annotator_image, 0.0, 0, 0, n_gt))
            continue
        hit = iou > thr
        tp_iou += iou[hit].tolist()
        tp_dice += dice[hit].tolist()
        f_p, f_n = int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum())
        fp += f_p
        fn += f_n
        best_dice_gt += dice.max(1).tolist()
        best_iou_gt += iou.max(1).tolist()
        o2m += int(((iou > 0).sum(1) > 1).sum())
        m2o += int(((iou > 0).sum(0) > 1).sum())
        t = int(hit.sum())
        den = t + 0.5 * f_p + 0.5 * f_n
        per_image.append((row.annotator_image, float(iou[hit].sum() / den) if den else 0.0, t, f_p, f_n))
    tp = len(tp_iou)
    den = tp + 0.5 * fp + 0.5 * fn
    out = {
        "PQ (official get_pq_score)": get_pq_score(overlap_df),
        "PQ (recomputed)": sum(tp_iou) / den if den else 0.0,
        "SQ (mean IoU of matches)": float(np.mean(tp_iou)) if tp else 0.0,
        "RQ (F1 of matches)": tp / den if den else 0.0,
        "TP": tp, "FP": fp, "FN": fn,
        "mean Dice of matches": float(np.mean(tp_dice)) if tp else 0.0,
        "mean best Dice per GT filament": float(np.mean(best_dice_gt)) if best_dice_gt else 0.0,
        "mean best IoU per GT filament": float(np.mean(best_iou_gt)) if best_iou_gt else 0.0,
        "GT filaments touched by more than one prediction (1:n)": o2m,
        "predictions touching more than one GT filament (n:1)": m2o,
    }
    per = pd.DataFrame(per_image, columns=["annotator_image", "pq", "tp", "fp", "fn"])
    return out, per


def evaluate(gt_df, pred_df):
    overlap = get_overlap_df(gt_df, pred_df)
    stats, per = summary(overlap)
    return overlap, stats, per
