"""Whole-disk ConvNeXt-Tiny U-Net: training, the 2048 fine-tune and 8-way inference, one fold per process.

This is the code of the Kaggle kernels that trained the final models (research file
work/db1_global/template.py with the edits made by build_cv.py, build_s2048.py, build_repredict.py and
build_perfold.py), turned into one script with command-line options. The training loop, model, loss,
augmentation and schedules are unchanged.

  train     stage 1: 30 epochs at 1024 (lr 3e-4), then stage 2: 24 epochs at 1536 (lr 1e-4), starting
            from the EMA weights of stage 1. Saves ema_stage2_1536.pt. About 2 hours per fold on one T4.
  finetune  stage 3: 10 epochs at 2048 (lr 5e-5) from a fold's ema_stage2_1536.pt. Saves
            ema_stage3_2048.pt. About 1.2 hours per fold on one T4.
  predict   8-way (rotations and flips) logit average at the model's resolution, sigmoid, bilinear to
            2048, then area-downsampled to 1024 and saved as PNG. Writes the fold's OOF maps to <out>/oof
            and its test maps to <out>/test_fold<k>.

Every model trains on one random annotator per sample (no consensus target). The two 1536 models differ
by seed and encoder weights: seed 0 with convnext_tiny.in12k_ft_in1k, and seed 7 with
convnext_tiny.fb_in22k_ft_in1k. The 2048 model fine-tunes the seed-0 checkpoints.

Usage (one GPU per process, choose it with CUDA_VISIBLE_DEVICES):
  python global_unet.py train    --data D --fold 0 --out W/g1536a/fold0 [--seed 7 --weights convnext_tiny.fb_in22k_ft_in1k]
  python global_unet.py finetune --data D --fold 0 --out W/g2048/fold0 --ckpt W/g1536a/fold0/ema_stage2_1536.pt
  python global_unet.py predict  --data D --fold 0 --out W/maps/g2048 --ckpt .../ema_stage3_2048.pt --res 2048
Add --smoke for a tiny run on CPU that only checks the code path (4 images, 2 steps, 256 pixels).
"""
import argparse
import copy
import glob
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIZE = 2048
DEFAULT_WEIGHTS = ["convnext_tiny.in12k_ft_in1k", "convnext_tiny.fb_in1k", "convnext_tiny"]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["train", "finetune", "predict"])
ap.add_argument("--data", required=True, help="competition data folder with train/ and test/")
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--weights", default="", help="timm encoder tag tried first (second model: convnext_tiny.fb_in22k_ft_in1k)")
ap.add_argument("--ckpt", default="", help="checkpoint for finetune and predict")
ap.add_argument("--res", type=int, default=1536, help="inference resolution for predict (1536, or 2048 for the fine-tuned model)")
ap.add_argument("--no-test", action="store_true", help="predict: skip the test images")
ap.add_argument("--no-oof", action="store_true", help="predict: skip the fold's OOF images")
ap.add_argument("--smoke", action="store_true", help="tiny CPU run that only checks the code path")
A = ap.parse_args()

FOLD, OUT = A.fold, A.out
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
CUDA = dev == "cuda"
NGPU = 1  # one process per GPU
SMOKE = A.smoke
R1, R2, R3 = (256, 256, 256) if SMOKE else (1024, 1536, 2048)
E1, E2, E3 = (1, 1, 1) if SMOKE else (30, 24, 10)
STEPS_CAP = 2 if SMOKE else 0
MAX_TRAIN_SEC = 10.5 * 3600

random.seed(A.seed); np.random.seed(A.seed); torch.manual_seed(A.seed)
torch.backends.cudnn.benchmark = True
if CUDA:
    log("torch", torch.__version__, torch.cuda.get_device_name(0),
        "mem GB", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))


def autocast():
    return torch.autocast(dev, dtype=torch.float16, enabled=CUDA)


# ---------------------------------------------------------------- data
ROOT = A.data
ann_path = os.path.join(ROOT, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
IMG_DIR = os.path.join(ROOT, "train", "train_images")
coco = json.load(open(ann_path))
polys = defaultdict(list)
for a in coco["annotations"]:
    polys[a["image_id"]].append(a)
index = {}
for im in coco["images"]:
    o = im["id"].split("-", 1)[1]
    index.setdefault(o, {"file": im["file_name"], "ann": {}})
    index[o]["ann"][im["id"]] = polys[im["id"]]
index = dict(sorted(index.items()))


def fold_of(o):
    return int(hashlib.md5(o.encode()).hexdigest(), 16) % 5


tr_obs = [o for o in index if fold_of(o) != FOLD]
va_obs = [o for o in index if fold_of(o) == FOLD]
if SMOKE:
    tr_obs, va_obs = tr_obs[:4], va_obs[:2]
log("train obs", len(tr_obs), "val obs", len(va_obs))


def read_gray(o):
    g = cv2.imread(os.path.join(IMG_DIR, os.path.basename(index[o]["file"])), cv2.IMREAD_GRAYSCALE)
    assert g is not None and g.shape == (SIZE, SIZE), o
    return g


def build_packed(anns, res):
    """uint8 map: bit0 = foreground, bit1 = instance boundary, at res x res."""
    s = res / SIZE
    fg = np.zeros((res, res), np.uint8)
    bd = np.zeros((res, res), np.uint8)
    k = np.ones((3, 3), np.uint8)
    for a in anns:
        inst = np.zeros((res, res), np.uint8)
        for p in a["segmentation"]:
            pts = np.round(np.array(p).reshape(-1, 2) * s).astype(np.int32)
            cv2.fillPoly(inst, [pts], 1)
        fg |= inst
        bd |= cv2.dilate(inst, k) - cv2.erode(inst, k)
    return fg | (bd << 1)


def build_stage_data(res):
    t = time.time()
    imgs, masks = [], []
    for o in tr_obs:
        imgs.append(cv2.resize(read_gray(o), (res, res), interpolation=cv2.INTER_AREA))
        masks.append([build_packed(anns, res) for anns in index[o]["ann"].values()])
    log(f"stage data res {res} built in {time.time() - t:.0f}s")
    return imgs, masks


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, imgs, masks):
        self.imgs, self.masks = imgs, masks

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i]
        m = random.choice(self.masks[i])          # one random annotator, no consensus
        k = random.randint(0, 3)
        img, m = np.rot90(img, k), np.rot90(m, k)
        if random.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        if random.random() < 0.5:
            img, m = img[::-1], m[::-1]
        x = img.astype(np.float32)
        if random.random() < 0.8:                 # mild photometric
            x = x * random.uniform(0.85, 1.15) + random.uniform(-20, 20)
            if random.random() < 0.5:
                x = 255 * np.clip(x / 255, 0, 1) ** random.uniform(0.8, 1.25)
        if random.random() < 0.15:
            sg = random.uniform(0.5, 1.5)
            x = cv2.GaussianBlur(x, (0, 0), sg)
        if random.random() < 0.15:
            x = x + np.random.normal(0, random.uniform(2, 6), x.shape).astype(np.float32)
        x = np.clip(x, 0, 255).astype(np.uint8)
        return torch.from_numpy(np.ascontiguousarray(x))[None], torch.from_numpy(np.ascontiguousarray(m))[None]


# ---------------------------------------------------------------- model
import timm  # noqa: E402


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.net(x)


class Net(nn.Module):
    """ConvNeXt-Tiny encoder (strides 4/8/16/32) + U-Net decoder 256/128/64/32/16, 2 output logits
    (foreground and instance boundary)."""

    def __init__(self, pretrained=True):
        super().__init__()
        last = None
        names = ([A.weights, "convnext_tiny.in12k_ft_in1k"] if A.weights else DEFAULT_WEIGHTS)
        for name in names:
            try:
                self.enc = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
                log("encoder weights:", name, "(pretrained)" if pretrained else "(random init)")
                break
            except Exception as e:  # noqa
                last = e
                log("encoder load failed", name, repr(e)[:200])
        else:
            raise RuntimeError("no pretrained ConvNeXt-Tiny could be loaded") from last
        c = self.enc.feature_info.channels()      # 96, 192, 384, 768
        self.d1 = Up(c[3], c[2], 256)
        self.d2 = Up(256, c[1], 128)
        self.d3 = Up(128, c[0], 64)
        self.d4 = Up(64, 0, 32)
        self.d5 = Up(32, 0, 16)
        self.head = nn.Conv2d(16, 2, 1)
        self.register_buffer("mean", torch.tensor(0.485).view(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(0.229).view(1, 1, 1, 1))

    def forward(self, x):                          # x: B,1,H,W uint8-scaled float in [0,1]
        x = ((x - self.mean) / self.std).expand(-1, 3, -1, -1)
        f = self.enc(x)
        y = self.d1(f[3], f[2]); y = self.d2(y, f[1]); y = self.d3(y, f[0]); y = self.d4(y); y = self.d5(y)
        return self.head(y)


def enc_dec_params(model):
    enc = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    dec = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    return enc, dec


def loss_fn(logits, packed):
    t_fg = (packed & 1).float()
    t_bd = ((packed >> 1) & 1).float()
    lf, lb = logits[:, 0:1].float(), logits[:, 1:2].float()
    bce = F.binary_cross_entropy_with_logits(lf, t_fg)
    p = torch.sigmoid(lf)
    inter = (p * t_fg).flatten(1).sum(1)
    dice = 1 - (2 * inter + 1) / (p.flatten(1).sum(1) + t_fg.flatten(1).sum(1) + 1)
    bnd = F.binary_cross_entropy_with_logits(lb, t_bd, pos_weight=torch.tensor(3.0, device=lb.device))
    return bce + dice.mean() + 0.2 * bnd


@torch.no_grad()
def ema_update(ema, model, d=0.99):
    for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
        if pe.dtype.is_floating_point:
            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        else:
            pe.copy_(pm)


def choose_micro(model, res):
    """Largest micro-batch from (4, 2, 1) that fits with some headroom; returns (micro, peak GB list)."""
    if not CUDA:
        return 1, []
    cap = torch.cuda.get_device_properties(0).total_memory
    tried = []
    for micro in (4, 2, 1):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.rand(micro, 1, res, res, device=dev).contiguous(memory_format=torch.channels_last)
            t = (torch.rand(micro, 1, res, res, device=dev) > 0.9).to(torch.uint8)
            for _ in range(2):
                with autocast():
                    out = model(x)
                loss_fn(out, t).backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(0) / 2**30
            tried.append((micro, round(peak, 2)))
            if peak * 2**30 < 0.88 * cap:
                log(f"res {res}: micro-batch {micro} fits, peak {peak:.1f} GB", tried)
                return micro, tried
        except torch.cuda.OutOfMemoryError:
            tried.append((micro, "OOM"))
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    raise RuntimeError(f"nothing fits at res {res}: {tried}")


def train_stage(model, ema, res, epochs, lr, warm_ep, tag, imgs, masks, steps_cap=0, deadline=None):
    micro, tried = choose_micro(model, res)
    accum = 4 // micro
    ds = TrainSet(imgs, masks)
    workers = 0 if SMOKE else 2
    dl = torch.utils.data.DataLoader(ds, batch_size=micro, shuffle=True, drop_last=True, num_workers=workers,
                                     pin_memory=CUDA, persistent_workers=workers > 0,
                                     prefetch_factor=2 if workers else None)
    enc, dec = enc_dec_params(model)
    opt = torch.optim.AdamW([{"params": enc, "lr": lr / 5}, {"params": dec, "lr": lr}], weight_decay=0.01)
    base = [lr / 5, lr]
    steps_ep = len(dl) // accum
    if steps_cap:
        steps_ep = min(steps_ep, steps_cap)
    total, warm = epochs * steps_ep, int(warm_ep * steps_ep)
    scaler = torch.amp.GradScaler("cuda", enabled=CUDA)
    step, ep_times = 0, []
    for ep in range(epochs):
        model.train(); t = time.time(); run = 0.0; n = 0
        it = iter(dl)
        for s in range(steps_ep):
            frac = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * max(frac, 0.0)
            for _ in range(accum):
                x, m = next(it)
                x = (x.to(dev, non_blocking=True).float() / 255).contiguous(memory_format=torch.channels_last)
                m = m.to(dev, non_blocking=True)
                with autocast():
                    out = model(x)
                loss = loss_fn(out, m) / accum
                scaler.scale(loss).backward()
                run += loss.item() * accum; n += 1
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            ema_update(ema, model)
            step += 1
        dt = time.time() - t
        ep_times.append(dt)
        log(f"[{tag}] epoch {ep + 1}/{epochs} loss {run / max(n, 1):.4f} {dt:.0f}s lr {opt.param_groups[1]['lr']:.2e}")
        if deadline and time.time() > deadline:
            log("deadline reached, stopping stage early"); break
    del dl
    return dict(res=res, micro=micro, accum=accum, tried=tried, ep_times=ep_times, steps_ep=steps_ep)


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict_prob(model, gray, res):
    """8-way (4 rotations x horizontal flip) logit average -> sigmoid -> bilinear to 2048. Returns uint8 2048 x 2048."""
    model.eval()
    g = cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA)
    x = (torch.from_numpy(g)[None, None].to(dev).float() / 255).contiguous(memory_format=torch.channels_last)
    lg = 0
    for k in range(4):
        xs = torch.cat([torch.rot90(x, k, (2, 3)), torch.rot90(x, k, (2, 3)).flip(3)])
        with autocast():
            y = model(xs.contiguous(memory_format=torch.channels_last))[:, 0:1].float()
        lg = lg + torch.rot90(y[0:1], -k, (2, 3)) + torch.rot90(y[1:2].flip(3), -k, (2, 3))
    lg = lg / 8
    p = F.interpolate(torch.sigmoid(lg), size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
    return (p * 255).round().clamp(0, 255).byte().cpu().numpy()


def save_1024(path, p):
    cv2.imwrite(path, cv2.resize(p, (1024, 1024), interpolation=cv2.INTER_AREA))


def load_ckpt(net, path):
    sd = torch.load(path, map_location="cpu")
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    log("loaded", path)


def save_half(net, path):
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()}, path)


# ---------------------------------------------------------------- run
model = Net(pretrained=not SMOKE and A.mode != "predict").to(dev).to(memory_format=torch.channels_last)
ema = copy.deepcopy(model).eval()
for p in ema.parameters():
    p.requires_grad_(False)
log("params M", sum(p.numel() for p in model.parameters()) / 1e6, "device", dev)

if A.mode == "train":
    imgs, masks = build_stage_data(R1)
    s1 = train_stage(model, ema, R1, E1, 3e-4, 1.0, "s1-1024", imgs, masks, steps_cap=STEPS_CAP, deadline=T0 + MAX_TRAIN_SEC)
    del imgs, masks
    save_half(ema, f"{OUT}/ema_stage1_1024.pt")
    model.load_state_dict(ema.state_dict())       # stage 2 starts from the EMA weights
    imgs, masks = build_stage_data(R2)
    s2 = train_stage(model, ema, R2, E2, 1e-4, 0.5, "s2-1536", imgs, masks, steps_cap=STEPS_CAP, deadline=T0 + 11.2 * 3600)
    del imgs, masks
    save_half(ema, f"{OUT}/ema_stage2_1536.pt")
    json.dump(dict(fold=FOLD, seed=A.seed, stage1=s1, stage2=s2, hours=(time.time() - T0) / 3600),
              open(f"{OUT}/train.json", "w"), indent=1, default=str)
    log("saved", f"{OUT}/ema_stage2_1536.pt")

elif A.mode == "finetune":
    sd = torch.load(A.ckpt, map_location="cpu")
    sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    ema.load_state_dict(sd); model.load_state_dict(sd)
    log("loaded", A.ckpt)
    imgs, masks = build_stage_data(R3)
    s3 = train_stage(model, ema, R3, E3, 5e-5, 0.3, "s3-2048", imgs, masks, steps_cap=STEPS_CAP, deadline=T0 + 3.0 * 3600)
    del imgs, masks
    save_half(ema, f"{OUT}/ema_stage3_2048.pt")
    json.dump(dict(fold=FOLD, stage3=s3, hours=(time.time() - T0) / 3600), open(f"{OUT}/finetune.json", "w"), indent=1, default=str)
    log("saved", f"{OUT}/ema_stage3_2048.pt")

else:  # predict
    if A.ckpt:
        load_ckpt(ema, A.ckpt)
    elif not SMOKE:
        sys.exit("predict needs --ckpt")
    res = 256 if SMOKE else A.res
    n = 0
    if not A.no_oof:
        os.makedirs(f"{OUT}/oof", exist_ok=True)
        for o in va_obs:
            save_1024(f"{OUT}/oof/{o}.png", predict_prob(ema, read_gray(o), res)); n += 1
    if not A.no_test:
        os.makedirs(f"{OUT}/test_fold{FOLD}", exist_ok=True)
        tests = sorted(glob.glob(os.path.join(ROOT, "test", "test_images", "*.jpeg")))[: 2 if SMOKE else None]
        for f in tests:
            save_1024(f"{OUT}/test_fold{FOLD}/{os.path.basename(f)[:-5]}.png",
                      predict_prob(ema, cv2.imread(f, cv2.IMREAD_GRAYSCALE), res)); n += 1
    log(f"fold {FOLD}: {n} maps saved at 1024 under {OUT}, hours {(time.time() - T0) / 3600:.2f}")
