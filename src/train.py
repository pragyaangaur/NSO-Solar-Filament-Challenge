"""Train one cross-validation fold of a U-Net on random 512 x 512 crops.

Usage: python train.py --fold 0 [--encoder resnet34] [--epochs 30] [--steps 250]
"""
import argparse
import json
import math
import random
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import CACHE, OUT, SIZE, make_channels

CROP = 512


class CropDataset(Dataset):
    """Random crops from the cached observations. Half of them are centred near a filament."""

    def __init__(self, observations, meta, length, pos_frac=0.5, augment=True, seed=None):
        self.obs, self.meta, self.length = list(observations), meta, length
        self.pos_frac, self.augment, self.seed = pos_frac, augment, seed
        self.pos = {}
        for o in self.obs:
            t = np.load(CACHE / f"{o}_target.npy", mmap_mode="r")
            ys, xs = np.nonzero(t[::4, ::4])
            self.pos[o] = np.stack([ys * 4, xs * 4], 1)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.seed is None else self.seed + i)
        o = self.obs[rng.integers(len(self.obs))]
        m = self.meta[o]
        if rng.random() < self.pos_frac and len(self.pos[o]):
            y, x = self.pos[o][rng.integers(len(self.pos[o]))] + rng.integers(-200, 201, 2)
        else:
            y, x = rng.integers(0, SIZE, 2)
        y0 = int(np.clip(y - CROP // 2, 0, SIZE - CROP))
        x0 = int(np.clip(x - CROP // 2, 0, SIZE - CROP))
        planes = np.load(CACHE / f"{o}_planes.npy", mmap_mode="r")[:, y0:y0 + CROP, x0:x0 + CROP]
        target = np.load(CACHE / f"{o}_target.npy", mmap_mode="r")[y0:y0 + CROP, x0:x0 + CROP]
        x_in = make_channels(np.asarray(planes), m["disk"], m["stats"], x0, y0)
        t = target.astype(np.float32)[None] / 255.0
        if self.augment:
            k = rng.integers(4)
            x_in, t = np.rot90(x_in, k, (1, 2)), np.rot90(t, k, (1, 2))
            if rng.random() < 0.5:
                x_in, t = x_in[:, :, ::-1], t[:, :, ::-1]
            x_in = x_in.copy()
            x_in[0] = x_in[0] * rng.uniform(0.8, 1.25) + rng.uniform(-0.2, 0.2)
        return torch.from_numpy(np.ascontiguousarray(x_in)), torch.from_numpy(np.ascontiguousarray(t))


def build_model(encoder="resnet34", pretrained=True):
    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)


def loss_fn(logits, target):
    """Binary cross-entropy plus soft Dice."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    denom = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (denom + 1)
    return bce + dice.mean()


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(dev):
    """bfloat16 mixed precision on GPUs, plain float32 on the CPU."""
    if dev.type == "cpu":
        return torch.autocast("cpu", enabled=False)
    return torch.autocast(dev.type, dtype=torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--encoder", default="resnet34", help="any segmentation_models_pytorch encoder")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=250, help="batches per epoch")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--name", default="unet_baseline")
    args = ap.parse_args()

    torch.manual_seed(args.fold)
    random.seed(args.fold)
    meta = json.load(open(CACHE / "meta.json"))["train"]
    train_obs = [o for o, m in meta.items() if m["fold"] != args.fold]
    val_obs = [o for o, m in meta.items() if m["fold"] == args.fold]
    print(f"fold {args.fold}: {len(train_obs)} train and {len(val_obs)} validation observations", flush=True)
    out = OUT / args.name / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"encoder": args.encoder}, open(OUT / args.name / "model.json", "w"))

    dev = device()
    model = build_model(args.encoder).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * args.steps
    # Linear warm-up for 300 steps, then cosine decay to zero.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    tl = DataLoader(CropDataset(train_obs, meta, total * args.bs), batch_size=args.bs,
                    num_workers=args.workers, persistent_workers=args.workers > 0)
    vl = DataLoader(CropDataset(val_obs, meta, 128, augment=False, seed=12345), batch_size=args.bs)

    log = open(out / "log.txt", "a")
    step, run, t0 = 0, 0.0, time.time()
    model.train()
    for x, t in tl:
        x, t = x.to(dev), t.to(dev)
        with autocast(dev):
            logits = model(x)
        loss = loss_fn(logits.float(), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run += loss.item()
        step += 1
        if step % args.steps == 0:
            model.eval()
            vloss = 0.0
            with torch.no_grad():
                for vx, vt in vl:
                    with autocast(dev):
                        vloss += loss_fn(model(vx.to(dev)).float(), vt.to(dev)).item()
            model.train()
            msg = (f"fold {args.fold} epoch {step // args.steps}/{args.epochs} train {run / args.steps:.4f} "
                   f"val {vloss / len(vl):.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.1f} min")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            run = 0.0
            torch.save(model.state_dict(), out / "last.pt")
    print(f"done, checkpoint in {out / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
