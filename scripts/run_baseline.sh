#!/usr/bin/env bash
# Full baseline pipeline: cache the data, train and predict all 5 folds, then write
# and validate a submission. Set PYTHON to choose the interpreter (default: python).
# Extra arguments are passed to train.py, for example --encoder tu-convnext_tiny.
set -euo pipefail
cd "$(dirname "$0")/../src"
PY=${PYTHON:-python}
NAME=${NAME:-unet_baseline}

[ -f ../data/cache/meta.json ] || $PY prepare.py
for f in 0 1 2 3 4; do
  $PY train.py --fold "$f" --name "$NAME" "$@"
  $PY predict.py --fold "$f" --name "$NAME"
done
$PY make_submission.py --name "$NAME"
$PY validate_submission.py "../outputs/$NAME/submission.csv"
