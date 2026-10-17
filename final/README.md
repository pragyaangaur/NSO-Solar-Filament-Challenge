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

All weights are public in the Kaggle dataset [pragyaangaur/filament-segmentation-2026-weights](https://www.kaggle.com/datasets/pragyaangaur/filament-segmentation-2026-weights). It holds the 15 U-Net checkpoints as `<model>_fold<k>.pt` (float16 EMA state dicts, about 64 MB each) and the 10 CNN quality heads as `cnn_s<seed>_g<group>.pt` (about 45 MB each). On Kaggle, add the dataset as an input of the notebook and it finds the files under `/kaggle/input`. On another machine, download the dataset into `weights/` next to the notebook or point `FIL_WEIGHTS_DIR` to it.

To run the whole pipeline from the checkpoints on Kaggle, attach the competition data and the weights dataset, enable the internet, choose the 2xT4 accelerator and set `FIL_RUN_INFERENCE=1`.

## Reproduction run

On 17 October 2026 the notebook was run end to end on a Kaggle kernel with two T4 GPUs, from a fresh clone of this repository and the public weights dataset, with `FIL_RUN_INFERENCE=1`. It took 1.7 hours. The executed notebook and its plots are in `results/`.

The run scored 0.4568 out-of-fold PQ on all 707 train observations, and the organisers' scorer gave the same value. The research run that produced the submitted file scored 0.4571. The 95 per cent bootstrap interval over observations is 0.448 to 0.466, and the five folds score between 0.450 and 0.463, so the two runs agree within noise. The test submission of the run has 1,175 filaments, and 1,160 of them match a filament of the submitted file (Kaggle reference 57096818) with IoU above 0.5, at a median IoU of 0.9996. The small differences come from running inference again with other GPU and library versions.

The U-Nets were not trained again in this run, because training needs about 30 GPU hours. `FIL_RUN_TRAINING=1` runs that step, and section 2 of the notebook lists the recipe of each model.
