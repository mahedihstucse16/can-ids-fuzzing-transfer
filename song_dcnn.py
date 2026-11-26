# -*- coding: utf-8 -*-
"""Song et al. (2020) DCNN, re-implemented in PyTorch, trained with and without the fuzzing
capture and tested on the KIA Soul (OTIDS) windows.

Reference: H.M. Song, J. Woo, H.K. Kim, In-vehicle network intrusion detection using deep
convolutional neural network, Vehicular Communications 21 (2020) 100198.

Input (as in the paper): 29 consecutive CAN identifiers, each written as its 29-bit (extended
format) binary representation, giving a 29 x 29 x 1 bit image; non-overlapping windows; a window
is an attack if it contains at least one injected frame.

Network (as described in the paper): a reduced Inception-ResNet for 29 x 29 x 1 input and two
classes, keeping the stem, Inception-ResNet-A, Reduction-A, Inception-ResNet-B and Reduction-B
blocks, followed by global average pooling, dropout and a softmax output; Inception-ResNet-C is
removed. Three A blocks and two B blocks, with the widths below, give 1,684,290 parameters,
close to the 1.69 million that Hoang and Kim (2022, Table V) report for their re-implementation of the original.

Training: Adam (learning rate 0.001), batch 128, class-weighted cross-entropy, at most 30 epochs,
early stopping on a 10% validation split of the training windows (patience 5), best epoch kept.
The training and held-out windows of every draw are those of the paper's other window-based
detectors (read from song_windows.npz), so the comparison is like for like.

Usage (Colab, GPU runtime):  python song_dcnn.py song_windows.npz results/
Writes results/SONG__<composition>__DCNN__seed<k>.json for both compositions and ten seeds.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NPZ, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CH = ['DoS', 'Fuzzy', 'gear', 'RPM']
COMPS = {'flooding_gear_rpm': ['DoS', 'gear', 'RPM'], 'all_four_with_fuzzing': CH}
Z = np.load(NPZ)
BITS = torch.tensor([1 << (28 - i) for i in range(29)], dtype=torch.int64)


def to_image(ids):
    """(n, 29) identifiers -> (n, 1, 29, 29) float bit images (29-bit representation)."""
    t = torch.as_tensor(ids.astype(np.int64))
    return ((t.unsqueeze(-1) & BITS) > 0).float().unsqueeze(1)


# ----------------------------------------------------------------------------- network
class ConvBN(nn.Sequential):
    def __init__(self, cin, cout, k, s=1, p=0):
        super().__init__(nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Stem(nn.Module):                      # 29 -> 25 x 25 x 128
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(ConvBN(1, 32, 3), ConvBN(32, 32, 3), ConvBN(32, 64, 3, p=1),
                               ConvBN(64, 128, 1), ConvBN(128, 128, 3, p=1))

    def forward(self, x):
        return self.f(x)


class BlockA(nn.Module):                    # Inception-ResNet-A, 128 channels
    def __init__(self, c=128, scale=0.17):
        super().__init__()
        self.scale = scale
        self.b0 = ConvBN(c, 32, 1)
        self.b1 = nn.Sequential(ConvBN(c, 32, 1), ConvBN(32, 32, 3, p=1))
        self.b2 = nn.Sequential(ConvBN(c, 32, 1), ConvBN(32, 48, 3, p=1), ConvBN(48, 64, 3, p=1))
        self.up = nn.Conv2d(128, c, 1)

    def forward(self, x):
        y = torch.cat([self.b0(x), self.b1(x), self.b2(x)], 1)
        return F.relu(x + self.scale * self.up(y))


class ReductionA(nn.Module):                # 25 -> 12, 128 -> 448
    def __init__(self, c=128):
        super().__init__()
        self.b0 = nn.MaxPool2d(3, 2)
        self.b1 = ConvBN(c, 192, 3, 2)
        self.b2 = nn.Sequential(ConvBN(c, 96, 1), ConvBN(96, 96, 3, p=1), ConvBN(96, 128, 3, 2))

    def forward(self, x):
        return torch.cat([self.b0(x), self.b1(x), self.b2(x)], 1)


class BlockB(nn.Module):                    # Inception-ResNet-B, 448 channels
    def __init__(self, c=448, scale=0.10):
        super().__init__()
        self.scale = scale
        self.b0 = ConvBN(c, 64, 1)
        self.b1 = nn.Sequential(ConvBN(c, 64, 1), ConvBN(64, 64, (1, 7), p=(0, 3)), ConvBN(64, 64, (7, 1), p=(3, 0)))
        self.up = nn.Conv2d(128, c, 1)

    def forward(self, x):
        y = torch.cat([self.b0(x), self.b1(x)], 1)
        return F.relu(x + self.scale * self.up(y))


class ReductionB(nn.Module):                # 12 -> 5, 448 -> 768
    def __init__(self, c=448):
        super().__init__()
        self.b0 = nn.MaxPool2d(3, 2)
        self.b1 = nn.Sequential(ConvBN(c, 128, 1), ConvBN(128, 192, 3, 2))
        self.b2 = nn.Sequential(ConvBN(c, 128, 1), ConvBN(128, 128, 3, 2))

    def forward(self, x):
        return torch.cat([self.b0(x), self.b1(x), self.b2(x)], 1)


class SongDCNN(nn.Module):
    def __init__(self, n_a=3, n_b=2, dropout=0.2):
        super().__init__()
        self.f = nn.Sequential(Stem(), *[BlockA() for _ in range(n_a)], ReductionA(),
                               *[BlockB() for _ in range(n_b)], ReductionB())
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        x = self.f(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(self.drop(x))


# ----------------------------------------------------------------------------- helpers
def metrics(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'n_test': tp + fp + fn + tn,
            'precision': 100 * pr, 'recall': 100 * rc, 'f1': 100 * (2 * pr * rc / (pr + rc) if pr + rc else 0.0),
            'fpr_pct': 100.0 * fp / max(1, fp + tn)}


@torch.no_grad()
def predict(model, ids, bs=2048):
    model.eval()
    out = []
    for a in range(0, len(ids), bs):
        out.append(model(to_image(ids[a:a + bs]).to(DEV)).argmax(1).cpu().numpy())
    return np.concatenate(out).astype(np.int8)


def fit(model, X, y, seed, epochs=30, bs=128, patience=5):
    g = np.random.RandomState(seed)
    p = g.permutation(len(y)); nv = len(y) // 10
    vi, ti = p[:nv], p[nv:]
    w = len(y) / (2.0 * np.bincount(y, minlength=2).clip(min=1))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEV))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xv, yv = to_image(X[vi]).to(DEV), torch.as_tensor(y[vi].astype(np.int64), device=DEV)
    best, bad, state, ep_run = np.inf, 0, None, 0
    for ep in range(epochs):
        model.train()
        order = ti[g.permutation(len(ti))]
        for a in range(0, len(order), bs):
            b = order[a:a + bs]
            xb = to_image(X[b]).to(DEV); yb = torch.as_tensor(y[b].astype(np.int64), device=DEV)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(crit(model(Xv), yv))
        ep_run = ep + 1
        if vl < best - 1e-4:
            best, bad = vl, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return ep_run


# ----------------------------------------------------------------------------- run
n_params = sum(p.numel() for p in SongDCNN().parameters())
print('device', DEV, '| parameters', n_params, flush=True)
Xte, yte = Z['ids_test'], Z['y_test']
for comp, caps in COMPS.items():
    for s in range(10):
        path = os.path.join(OUT, 'SONG__%s__DCNN__seed%d.json' % (comp, s))
        if os.path.exists(path):
            continue
        t0 = time.time()
        Xtr = np.concatenate([Z['ids_%s' % k][Z['tr_%s_%d_%s' % (comp, s, k)]] for k in caps])
        ytr = np.concatenate([Z['y_%s' % k][Z['tr_%s_%d_%s' % (comp, s, k)]] for k in caps])
        Xho = np.concatenate([Z['ids_%s' % k][Z['ho_%s_%d_%s' % (comp, s, k)]] for k in caps])
        yho = np.concatenate([Z['y_%s' % k][Z['ho_%s_%d_%s' % (comp, s, k)]] for k in caps])
        torch.manual_seed(s); np.random.seed(s)
        model = SongDCNN().to(DEV)
        ep = fit(model, Xtr, ytr, s)
        m = metrics(yte, predict(model, Xte))
        mh = metrics(yho, predict(model, Xho))
        m.update({'model': 'DCNN', 'composition': comp, 'seed': s, 'unit': 'window of 29 frames',
                  'n_train_windows': int(len(ytr)), 'n_train_attack_windows': int(ytr.sum()), 'epochs': ep,
                  'n_parameters': int(n_params), 'within_vehicle_f1': mh['f1'],
                  'within_vehicle_fpr_pct': mh['fpr_pct'], 'device': str(DEV),
                  'torch_version': torch.__version__, 'seconds': round(time.time() - t0, 1)})
        json.dump(m, open(path, 'w'), indent=1)
        print('%-24s seed %d  F1 %6.2f  prec %6.2f  rec %6.2f  FPR %6.2f  within F1 %6.2f  epochs %d  (%.0fs)' % (
            comp, s, m['f1'], m['precision'], m['recall'], m['fpr_pct'], mh['f1'], ep, m['seconds']), flush=True)
print('done')
