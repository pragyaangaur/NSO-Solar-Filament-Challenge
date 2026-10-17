"""Plots from the organisers' self-evaluation notebook (azimahmadzadeh/self-evaluation-notebook),
copied without changes so the judges see the same IoU, Dice and relation plots they use."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.patches import Patch


def plot_distribution(matrices: pd.Series, title: str, x_label: str):
    """
    Plots histogram of pairwise overlap scores between GT and predicted filaments
    (scores > 0 only).

    Flattens all per-image score matrices into a single sample, drops
    non-overlapping (zero) pairs, and plots normalized bar frequencies with
    a mean line.

    Args:
        matrices: Iterable of per-image score matrices, e.g. the ``iou_matrix``
            or ``dice_matrix`` column of the dataframe returned by
            ``get_overlap_df``.
        title: Plot title.
        x_label: X-axis label.
    """

    # every entry is an (n_gt, n_pred) tensor with its own shape, so they cannot be stacked
    scores = np.concatenate(
        [np.asarray(m, dtype=float).ravel() for m in matrices] or [np.zeros(0)]
    )
    # non-overlapping pairs carry no information and would swamp the zero bin
    scores = scores[scores > 0]

    x_max = 1.0 if scores.size == 0 or scores.max() <= 1.0 else scores.max()

    if x_max <= 1.0:
        n_bins = 50
        bins = np.linspace(0.0, x_max, n_bins + 1)
        bar_width = 0.02
    else:
        bins = np.arange(0, int(x_max) + 2)
        bar_width = 0.8

    counts, edges = np.histogram(scores, bins=bins)
    probabilities = counts / counts.sum() if counts.sum() > 0 else counts
    centers = (edges[:-1] + edges[1:]) / 2.0

    plt.bar(centers, probabilities, width=bar_width, edgecolor="black", alpha=0.7, color="navy")
    plt.grid(axis="y", linestyle="--")

    if scores.size > 0:
        mean_score = scores.mean()
        plt.axvline(
            mean_score,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean = {mean_score:.3f}",
        )
        plt.legend(fontsize=8)
    plt.xlabel(x_label)
    plt.ylabel("Probability")
    plt.title(title)


def plot_m2n_counts(overlap_df: pd.DataFrame, title: str = "Many to One and One to Many Distribution"):
    """
    Plots frequency of GT filaments and predictions matching multiple partners (IoU > 0).

    For each image, counts overlapping GT→pred (1:n) and pred→GT (n:1) links,
    then draws a split bar chart (GT matches on the right, prediction matches
    on the left).

    Args:
        overlap_df: Per-image overlap table as produced by ``get_overlap_df``;
            must contain columns ``iou_matrix``, ``n_gt``, and ``n_pred``.
        title: Plot title.
    """
    offset = 0.5

    # per-GT number of matched predictions (1:n) and per-prediction number of matched GTs (n:1)
    gt_match_counts: list[int] = []
    pred_match_counts: list[int] = []

    for row in overlap_df.itertuples(index=False):
        iou_matrix = row.iou_matrix
        n_gt = row.n_gt
        n_pred = row.n_pred

        if n_gt == 0:
            pred_match_counts.extend([0] * n_pred)
            continue

        if n_pred == 0:
            gt_match_counts.extend([0] * n_gt)
            continue

        hit_matrix = iou_matrix > 0

        gt_match_counts.extend(hit_matrix.sum(dim=1).tolist())
        pred_match_counts.extend(hit_matrix.sum(dim=0).tolist())

    gt_degree_counts = Counter(gt_match_counts)
    pred_degree_counts = Counter(pred_match_counts)

    gt_miss_count = gt_degree_counts.get(0, 0)
    gt_hit_count = sum(count for degree, count in gt_degree_counts.items() if degree > 0)
    pred_miss_count = pred_degree_counts.get(0, 0)
    pred_hit_count = sum(count for degree, count in pred_degree_counts.items() if degree > 0)

    color_gt_miss = "crimson"
    color_gt_hit = "forestgreen"
    color_pred_miss = "darkorange"
    color_pred_hit = "steelblue"

    gt_bars = {degree + offset: count for degree, count in gt_degree_counts.items()}
    pred_bars = {-(degree + offset): count for degree, count in pred_degree_counts.items()}

    gt_positions = sorted(gt_bars)
    pred_positions = sorted(pred_bars)
    bar_width = 0.8

    for pos in gt_positions:
        degree = int(round(pos - offset))
        plt.bar(
            pos,
            gt_bars[pos],
            width=bar_width,
            color=color_gt_miss if degree == 0 else color_gt_hit,
            edgecolor="black",
            alpha=0.7,
        )

    for pos in pred_positions:
        degree = int(round(-pos - offset))
        plt.bar(
            pos,
            pred_bars[pos],
            width=bar_width,
            color=color_pred_miss if degree == 0 else color_pred_hit,
            edgecolor="black",
            alpha=0.7,
        )

    plt.axvline(0, color="black", linewidth=0.8)
    plt.grid(axis="y", linestyle="--")
    plt.xlabel("Matching Case")
    plt.ylabel("Frequency")
    tick_positions = np.concatenate([gt_positions, pred_positions])
    tick_labels = [
        f"{abs(int(round(v + offset)))}:1" if v < 0 else f"1:{int(round(v - offset))}"
        for v in tick_positions
    ]
    plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=9)
    plt.title(title)

    max_pos = max([abs(k) for k in gt_positions + pred_positions], default=offset)
    plt.xlim(-max_pos - bar_width, max_pos + bar_width)

    legend_handles = [
        Patch(facecolor=color_gt_hit, edgecolor="black", alpha=0.7, label=f"GT Hits (1:n, n>0): {gt_hit_count}"),
        Patch(facecolor=color_gt_miss, edgecolor="black", alpha=0.7, label=f"GT Misses (1:0): {gt_miss_count}"),
        Patch(facecolor=color_pred_miss, edgecolor="black", alpha=0.7, label=f"Prediction Misses (0:1): {pred_miss_count}"),
        Patch(facecolor=color_pred_hit, edgecolor="black", alpha=0.7, label=f"Prediction Hits (n:1, n>0): {pred_hit_count}"),
    ]
    plt.legend(handles=legend_handles, fontsize=9)
