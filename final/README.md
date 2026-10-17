# Solar filament segmentation: final pipeline

This folder holds the final pipeline of our entry to the Filament Segmentation Challenge 2026. `pipeline.ipynb` runs everything from the raw competition data to `submission.csv`, and its out-of-fold result is 0.4571 PQ with the organisers' scorer. The code it imports is in `src/`, and `data/meta.json` freezes the solar-disk geometry of all 887 images.

## How to run

Install the packages with `pip install -r requirements.txt` (Python 3.13) and open `pipeline.ipynb` from this folder. The competition data is expected at `../data/MAGFiLO_1.0_Kaggle_2026`, or wherever `FIL_DATA_DIR` points. Every setting in the first code cell can also be set with an environment variable that has the prefix `FIL_`.

The notebook has three switches. `RUN_TRAINING` trains the three U-Nets with five folds each, which takes about 30 GPU hours on T4 cards. `RUN_INFERENCE` predicts the probability maps from the 15 checkpoints on a CUDA GPU, which takes about 3 hours on a Kaggle 2xT4 kernel. `TRAIN_CNN_HEAD` retrains the quality head, which takes about 35 minutes on an Apple GPU. With all three off, the notebook starts from saved maps and runs on a laptop CPU.

## Files in `src/`

| File | What it does |
| --- | --- |
| `global_unet.py` | Training, the 2048 fine-tune and 8-way inference of the whole-disk ConvNeXt-Tiny U-Net, one fold per process |
| `data.py`, `common.py` | Data index, ground truth in the official RLE format, folds, disk detection and map input and output |
| `postprocess.py` | Cuts a probability map into candidate filaments |
| `cnn_head.py` | The CNN quality head that scores each candidate |
| `fusion.py` | Filament-level fusion of the models and the submission table |
| `fast_pq.py` | PQ on RLEs, identical to the official score and much faster |
| `official_metric.py`, `organiser_plots.py` | The organisers' scorer and plots, copied without changes |
| `self_eval.py` | PQ, SQ, RQ, Dice and relation counts from the official overlap matrices |

## Where the weights come from

The 15 U-Net checkpoints are outputs of private Kaggle kernels of the account `pragyaangaur`, listed in section 2 of the notebook. Add those kernels as inputs of a Kaggle notebook and the notebook finds the files by their paths. The ten quality-head files (`cnn_s<seed>_g<group>.pt`, about 43 MB each) were trained locally and must be shipped with the repository or as a Kaggle dataset, with `FIL_HEAD_DIR` pointing to them.
