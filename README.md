# NSO Solar Filament Challenge

This repository contains a deep learning pipeline for instance segmentation of solar filaments in full-disk H-alpha images. It was built for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026) on Kaggle.

## The task

Solar filaments are clouds of cool, dense plasma that are held above the solar surface by magnetic fields. In H-alpha images they appear as dark and often curved structures on the solar disk. Their eruptions can lead to coronal mass ejections, so a reliable catalogue of filaments is useful for space weather research.

The competition uses the MAGFiLO dataset. Each image is a 2048 x 2048 full-disk H-alpha observation from the Global Oscillation Network Group (GONG), a network of six ground stations operated by the National Solar Observatory. The training set has 707 observations. Some of them were labelled by one annotator and some by two or three, which gives 1154 annotated images with about 8200 filament polygons in total. The test set has 180 observations.

Every filament must be predicted as a separate instance. Submissions are scored with Panoptic Quality (PQ) at an IoU threshold of 0.5, and each annotator's labels for an image are scored separately.

## Status

The final pipeline is in [`final/`](final/README.md). Its notebook `final/pipeline.ipynb` runs everything from the raw competition data to `submission.csv`, and it reaches an out-of-fold PQ of 0.4571 with the organisers' scorer. The pipeline fuses three whole-disk ConvNeXt-Tiny U-Nets at the level of single filaments and keeps each filament only when a CNN quality head expects it to raise PQ. The technical report is [`report/report.pdf`](report/report.pdf), and all model weights are public in the Kaggle dataset [pragyaangaur/filament-segmentation-2026-weights](https://www.kaggle.com/datasets/pragyaangaur/filament-segmentation-2026-weights). The rest of this repository is the first baseline, which is kept for comparison.

## Repository layout

```
src/
  common.py               paths, annotation index, folds, disk geometry and input channels
  eda.py                  dataset statistics and ground truth export in the submission format
  prepare.py              caches image planes, training targets and disk geometry
  train.py                trains one cross-validation fold of a U-Net
  predict.py              full-resolution test predictions with flip averaging
  make_submission.py      turns probability maps into filament instances and writes the CSV
  validate_submission.py  checks a submission file before upload
scripts/
  run_baseline.sh         runs the whole pipeline from data to a validated submission
tools/
  memwatch.py             keeps long training runs under a memory limit on macOS
```

## Setup

The code needs Python 3.10 or newer. Install the dependencies with pip.

```bash
pip install -r requirements.txt
```

Download the competition data from the Kaggle competition page and extract it so that the folder `data/MAGFiLO_1.0_Kaggle_2026` exists in the repository root. If the data is stored somewhere else, set the `FILAMENT_DATA` environment variable to the path of the `MAGFiLO_1.0_Kaggle_2026` folder. The data itself is not included in this repository.

## Usage

The whole baseline runs with one command. It prepares the data and trains five fold models, and then it writes and checks `outputs/unet_baseline/submission.csv`.

```bash
./scripts/run_baseline.sh
```

Each step can also be run on its own from the `src` folder.

```bash
cd src
python eda.py
python prepare.py
python train.py --fold 0 --encoder resnet34
python predict.py --fold 0
python make_submission.py
python validate_submission.py ../outputs/unet_baseline/submission.csv
```

Any encoder from `segmentation_models_pytorch` can be used with `--encoder`. For example, `--encoder tu-convnext_tiny` selects a ConvNeXt-Tiny backbone from `timm`.

## Baseline method

The images are cached once with their solar disk geometry. The network input has three channels:

- the image normalised by the mean brightness of the disk,
- a contrast-enhanced (CLAHE) copy of the image,
- a mask of the solar disk.

The training target for an observation is the union of the masks drawn by all of its annotators.

A U-Net with an ImageNet-pretrained encoder is trained on random 512 x 512 crops at full resolution. Half of the crops are centred near a filament so that the network sees enough positive pixels. The loss is the sum of binary cross-entropy and soft Dice. Cross-validation uses five folds that are assigned per observation, so the annotations of one image never appear in both training and validation.

At test time the whole 2048 x 2048 image is processed in one pass and averaged over flips and over the five fold models. Connected regions above a probability threshold become filament instances, and very small regions are removed.

## Evaluation

The organisers provide a [self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) with the official PQ implementation. `src/eda.py` writes the training labels in the same format as a submission (`data/train_gt_rle.parquet`), so predictions on the training images can be scored locally with that notebook.

## License

The code is released under the MIT License. The competition data is subject to the competition rules and is not redistributed here.
