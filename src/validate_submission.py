"""Check a submission file before uploading it to Kaggle.

The checks are the two required columns, unique filament ids, ids that refer
to real test images, coverage of every test image, and RLE strings that decode
to non-empty 2048 x 2048 masks.

Usage: python validate_submission.py path/to/submission.csv
"""
import sys

import pandas as pd
from pycocotools import mask as mask_util

from common import SIZE, TEST_IMAGES


def main():
    path = sys.argv[1]
    sub = pd.read_csv(path)
    problems = []
    if list(sub.columns) != ["filament_id", "segmentation_rle"]:
        problems.append(f"columns are {list(sub.columns)}, expected ['filament_id', 'segmentation_rle']")
    if not sub.filament_id.is_unique:
        problems.append(f"{sub.filament_id.duplicated().sum()} duplicated filament ids")

    test_ids = {p.stem for p in TEST_IMAGES.glob("*.jpeg")}
    images = set(sub.filament_id.str.rsplit("_", n=1).str[0])
    if test_ids:
        if images - test_ids:
            problems.append(f"{len(images - test_ids)} ids do not match a test image")
        if test_ids - images:
            problems.append(f"{len(test_ids - images)} test images have no prediction")

    bad = 0
    for counts in sub.segmentation_rle:
        try:
            rle = {"size": [SIZE, SIZE], "counts": counts.encode()}
            if mask_util.decode(rle).shape != (SIZE, SIZE) or mask_util.area(rle) == 0:
                bad += 1
        except Exception:
            bad += 1
    if bad:
        problems.append(f"{bad} RLE strings are invalid or empty")

    print(f"{path}: {len(sub)} filaments over {len(images)} images")
    for p in problems:
        print("PROBLEM:", p)
    print("OK" if not problems else f"{len(problems)} problem(s) found")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
