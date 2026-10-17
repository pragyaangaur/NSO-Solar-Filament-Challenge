"""CNN quality head: a small CNN that looks at each candidate and predicts how well it will match the annotators.

The input is a square crop around the candidate (sized to the candidate, resized to 128 pixels) with
four channels: the H-alpha image, the probability map, the candidate mask and the other candidates'
masks. The target is the candidate's matched IoU averaged over the annotators (0 when unmatched).
The head is a ResNet18 with the log crop size as an extra input. It is cross-fitted on five groups of
observations (common.cv_group) with two seeds, so every OOF candidate is scored by the two models that
never saw its observation, and every test candidate by the mean of all ten.

The head is trained once, on the OOF candidates of the first 1536 model, and then scores the
candidates of every fusion member.
"""
import cv2
import numpy as np
from pycocotools import mask as mu

from common import cv_group, load_map, read_gray
from postprocess import PARAMS, instances

CROP = 128


def candidates_and_crops(key, map_path, gray_path, disk, params=PARAMS):
    """Candidates of one image with their 4-channel uint8 crops and crop sizes."""
    prob = load_map(map_path)
    gray = read_gray(gray_path)
    rles = instances(prob, disk, params)
    masks = [mu.decode(r) for r in rles]
    union = np.zeros(prob.shape, np.uint8)
    for m in masks:
        union |= m
    crops, sides = [], []
    for m in masks:
        ys, xs = np.nonzero(m)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        side = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.5 + 32)
        y0, x0 = int(cy - side / 2), int(cx - side / 2)
        pad = side
        sl = lambda a: np.pad(a, pad)[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]  # noqa: E731
        other = (union & (1 - m)) * 255
        crops.append(np.stack([cv2.resize(sl(gray), (CROP, CROP), interpolation=cv2.INTER_AREA),
                               cv2.resize(sl(prob), (CROP, CROP), interpolation=cv2.INTER_AREA),
                               cv2.resize(sl(m * 255), (CROP, CROP), interpolation=cv2.INTER_AREA),
                               cv2.resize(sl(other), (CROP, CROP), interpolation=cv2.INTER_AREA)]))
        sides.append(side)
    x = np.stack(crops) if crops else np.zeros((0, 4, CROP, CROP), np.uint8)
    return key, rles, x, np.array(sides, np.float32)


def device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_model(pretrained=True):
    import timm
    import torch
    import torch.nn as nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            try:
                self.body = timm.create_model("resnet18", pretrained=pretrained, in_chans=4, num_classes=0)
            except Exception:  # offline: train from scratch
                self.body = timm.create_model("resnet18", pretrained=False, in_chans=4, num_classes=0)
            self.out = nn.Sequential(nn.Linear(512 + 1, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x, s):
            return self.out(torch.cat([self.body(x), s[:, None]], 1))[:, 0]

    return Head()


def size_feature(sides):
    return np.log(np.asarray(sides, np.float32) / 256).astype(np.float32)


def augment(x, rng):
    k = rng.integers(0, 4)
    x = np.rot90(x, k, axes=(2, 3))
    if rng.random() < 0.5:
        x = x[:, :, :, ::-1]
    return np.ascontiguousarray(x)


def predict(models, x, s, dev):
    """Mean sigmoid score of the models over the four rotations of each crop."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            xb = x[i:i + 256]
            acc = 0
            for k in range(4):
                xt = torch.from_numpy(np.ascontiguousarray(np.rot90(xb, k, axes=(2, 3)))).to(dev).float() / 255
                st = torch.from_numpy(s[i:i + 256]).to(dev).float()
                acc = acc + sum(torch.sigmoid(m(xt, st)) for m in models) / len(models)
            out.append((acc / 4).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def load_heads(head_dir, seeds=(0, 1), dev=None):
    """{group: [one model per seed]} from files cnn_s<seed>_g<group>.pt."""
    import torch
    dev = dev or device()
    heads = {}
    for g in range(5):
        heads[g] = []
        for sd in seeds:
            m = make_model(pretrained=False).to(dev)
            m.load_state_dict(torch.load(f"{head_dir}/cnn_s{sd}_g{g}.pt", map_location=dev))
            heads[g].append(m.eval())
    return heads


def score(heads, x, sides, keys=None, dev=None):
    """CNN scores. With keys (the observation of each OOF candidate) each candidate is scored only by
    the models of its own cross-fitting group. Without keys (test) all models are averaged."""
    dev = dev or device()
    s = size_feature(sides)
    if keys is None:
        return predict([m for g in sorted(heads) for m in heads[g]], x, s, dev)
    q = np.zeros(len(x), np.float32)
    grp = np.array([cv_group(k) for k in keys])
    for g in sorted(heads):
        i = np.where(grp == g)[0]
        if len(i):
            q[i] = predict(heads[g], x[i], s[i], dev)
    return q


def train_heads(x, sides, target, keys, out_dir, seeds=(0, 1), epochs=12, log=print):
    """Cross-fit the head on five observation groups for each seed and save cnn_s<seed>_g<group>.pt."""
    import torch
    import torch.nn.functional as F
    dev = device()
    s = size_feature(sides)
    y = np.asarray(target, np.float32)
    group = np.array([cv_group(k) for k in keys])
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        for g in range(5):
            tr = np.where(group != g)[0]
            model = make_model().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            steps = max(1, epochs * (len(tr) // 64))
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
            model.train()
            for ep in range(epochs):
                perm = rng.permutation(tr)
                tot = 0.0
                for i in range(0, len(perm) - 63, 64):
                    b = perm[i:i + 64]
                    xb = torch.from_numpy(augment(x[b], rng)).to(dev).float() / 255
                    loss = F.binary_cross_entropy_with_logits(model(xb, torch.from_numpy(s[b]).to(dev)),
                                                              torch.from_numpy(y[b]).to(dev))
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                    tot += loss.item() * len(b)
                log(f"seed {seed} group {g} epoch {ep + 1}/{epochs} loss {tot / max(len(perm), 1):.4f}")
            torch.save(model.eval().state_dict(), f"{out_dir}/cnn_s{seed}_g{g}.pt")
