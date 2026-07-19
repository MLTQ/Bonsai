"""Trains a Growing Neural Cellular Automaton to grow (and regrow) the bonsai target.

Recipe follows Mordvintsev et al., "Growing Neural Cellular Automata" (distill.pub 2020):
sample-pool training for persistence, circular damage during training for regeneration.
Exports weights in the flat binary format the Swift/Metal runtime loads (see export()).

Usage: python3 train_nca.py [--iters 8000] [--out ../weights/bonsai.nca]
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from target import GRID, make_target

CH = 16        # state channels; 0-3 are RGBA (premultiplied), rest are hidden
HIDDEN = 128
FIRE_RATE = 0.5
POOL_SIZE = 1024
BATCH = 8
DAMAGE_N = 2   # samples per batch that get circular damage (after warmup)


class NCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Conv2d(CH * 3, HIDDEN, 1)
        self.w2 = nn.Conv2d(HIDDEN, CH, 1)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

        # Fixed perception kernels: identity, sobel-x, sobel-y per channel.
        # groups=CH conv => output ordering [c0*id, c0*sx, c0*sy, c1*id, ...],
        # which the Metal shader reproduces (see NCAShaders.swift).
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sy = sx.T.contiguous()
        kernels = torch.stack([ident, sx, sy])                    # (3,3,3)
        kernels = kernels.repeat(CH, 1, 1).unsqueeze(1)           # (CH*3,1,3,3)
        self.register_buffer("percept_w", kernels)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def forward(self, x):
        pre_life = self.alive(x)
        p = F.conv2d(x, self.percept_w, padding=1, groups=CH)
        dx = self.w2(F.relu(self.w1(p)))
        fire = (torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) <= FIRE_RATE).float()
        x = x + dx * fire
        life = (pre_life & self.alive(x)).float()
        return x * life


def make_seed(n, device):
    x = torch.zeros(n, CH, GRID, GRID, device=device)
    x[:, 3:, GRID // 2, GRID // 2] = 1.0
    return x


def damage(x):
    """Zero a random circle on each sample — teaches regeneration."""
    n, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij"
    )
    for i in range(n):
        r = np.random.uniform(6, 14)
        cx = np.random.uniform(w * 0.25, w * 0.75)
        cy = np.random.uniform(h * 0.25, h * 0.75)
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2 > r ** 2).float()
        x[i] *= mask
    return x


def export(model, fire_rate, path):
    """Flat little-endian binary: magic 'NCA1', int32 CH, int32 HIDDEN, float32 fire_rate,
    then float32 arrays w1[HIDDEN][CH*3], b1[HIDDEN], w2[CH][HIDDEN], b2[CH]."""
    with open(path, "wb") as f:
        f.write(b"NCA1")
        np.array([CH, HIDDEN], dtype="<i4").tofile(f)
        np.array([fire_rate], dtype="<f4").tofile(f)
        model.w1.weight.detach().cpu().numpy().reshape(HIDDEN, CH * 3).astype("<f4").tofile(f)
        model.w1.bias.detach().cpu().numpy().astype("<f4").tofile(f)
        model.w2.weight.detach().cpu().numpy().reshape(CH, HIDDEN).astype("<f4").tofile(f)
        model.w2.bias.detach().cpu().numpy().astype("<f4").tofile(f)


def save_preview(model, device, path, steps=220):
    """Grow from seed for `steps` and save an RGBA snapshot (training sanity check)."""
    from PIL import Image

    with torch.no_grad():
        x = make_seed(1, device)
        for _ in range(steps):
            x = model(x)
        rgba = x[0, :4].permute(1, 2, 0).cpu().numpy()
    img = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img, "RGBA").resize((GRID * 4, GRID * 4), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--out", default="../weights/bonsai.nca")
    ap.add_argument("--target", default=None,
                    help="ingested creature .npz (tools/ingest.py) instead of the built-in art")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    if args.target:
        data = np.load(args.target)
        assert str(data["kind"]) == "2d", "expected a 2d target npz"
        target = torch.from_numpy(data["target"].astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    else:
        target = torch.from_numpy(make_target()).permute(2, 0, 1).unsqueeze(0).to(device)
    model = NCA().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2000], gamma=0.1)

    pool = make_seed(POOL_SIZE, device)
    t0 = time.time()

    for it in range(1, args.iters + 1):
        idx = torch.randint(0, POOL_SIZE, (BATCH,))
        batch = pool[idx].clone()

        with torch.no_grad():
            losses = ((batch[:, :4] - target) ** 2).mean(dim=(1, 2, 3))
            order = losses.argsort(descending=True)
            batch = batch[order]
            batch[0] = make_seed(1, device)[0]     # worst sample restarts from seed
            if it > 500:
                batch[-DAMAGE_N:] = damage(batch[-DAMAGE_N:])

        for _ in range(np.random.randint(64, 97)):
            batch = model(batch)

        loss = ((batch[:, :4] - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8     # per-param grad normalization (paper trick)
        opt.step()
        sched.step()

        pool[idx[order.cpu()]] = batch.detach()

        if it % 50 == 0:
            rate = it / (time.time() - t0)
            print(f"iter {it:5d}  loss {loss.item():.5f}  {rate:.2f} it/s", flush=True)
        if it % 250 == 0 or it == args.iters:
            export(model, FIRE_RATE, args.out)
            save_preview(model, device, "train_preview.png")
            print(f"  checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
