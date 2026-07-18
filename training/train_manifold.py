"""Trains the behavior-manifold NCA: one network, a 10-D factor space of cycles.

Conditioning is split by timescale:
  - phase (sin th, cos th): appended to perception as channels, as in train_cyclic.py
    (fast timing signal, changes every step)
  - z (10 factors): FiLM modulation — z -> (gamma, beta) scaling/shifting the hidden
    layer (slow "mood", uniform across cells; a much stronger pathway than channels)

Pool training with per-slot z; mid-life z re-sampling teaches transitions between
arbitrary points of the manifold, not just within-cycle tracking.

Exports NCA3: NCA2 fields + zdim + FiLM matrices after the base weights.

Run (corpus first): python3 manifold_shoggoth.py --n 2048
                    python3 train_manifold.py --iters 60000
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CH = 16
HIDDEN = 192
ZDIM = 10
FRAMES = 12
GRID = 64
FIRE_RATE = 0.5
OMEGA = 2 * np.pi / 240.0
POOL_SIZE = 1024
BATCH = 16
DAMAGE_N = 3
ZSWITCH_P = 0.15


class ManifoldNCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Conv2d(CH * 3 + 2, HIDDEN, 1)   # +2: sin/cos phase channels
        self.w2 = nn.Conv2d(HIDDEN, CH, 1)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
        self.film = nn.Linear(ZDIM, 2 * HIDDEN)
        nn.init.normal_(self.film.weight, std=0.02)  # start ~unconditioned
        nn.init.zeros_(self.film.bias)
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sy = sx.T.contiguous()
        kernels = torch.stack([ident, sx, sy]).repeat(CH, 1, 1).unsqueeze(1)
        self.register_buffer("percept_w", kernels)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def forward(self, x, theta, z):
        pre_life = self.alive(x)
        p = F.conv2d(x, self.percept_w, padding=1, groups=CH)
        sc = torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)
        scmap = sc[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        h = self.w1(torch.cat([p, scmap], dim=1))
        gb = self.film(z)                                  # (B, 2*HIDDEN)
        gamma, beta = gb.chunk(2, dim=1)
        h = F.relu(h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None])
        dx = self.w2(h)
        fire = (torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) <= FIRE_RATE).float()
        x = x + dx * fire
        life = (pre_life & self.alive(x)).float()
        return x * life


def make_seed(n, device):
    x = torch.zeros(n, CH, GRID, GRID, device=device)
    x[:, 3:, GRID // 2, GRID // 2] = 1.0
    return x


def damage(x):
    n, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij"
    )
    for i in range(n):
        r = np.random.uniform(5, 12)
        cx = np.random.uniform(w * 0.3, w * 0.7)
        cy = np.random.uniform(h * 0.3, h * 0.7)
        x[i] *= (((xx - cx) ** 2 + (yy - cy) ** 2) > r ** 2).float()
    return x


class Corpus:
    """CPU-resident target cycles; phase-lerped targets gathered per batch."""

    def __init__(self, path, device):
        data = np.load(path)
        self.z = torch.from_numpy(data["z"]).float()                    # (N, ZDIM)
        self.frames = torch.from_numpy(data["frames"])                  # (N,F,H,W,4) f16
        self.frames = self.frames.permute(0, 1, 4, 2, 3).contiguous()   # (N,F,4,H,W)
        self.device = device
        self.n = self.z.shape[0]

    def target_at(self, idx, thetas):
        pos = (thetas.cpu() / (2 * np.pi) * FRAMES) % FRAMES
        f0 = pos.long() % FRAMES
        f1 = (f0 + 1) % FRAMES
        wgt = (pos - pos.floor())[:, None, None, None]
        t0 = self.frames[idx, f0].float()
        t1 = self.frames[idx, f1].float()
        return ((1 - wgt) * t0 + wgt * t1).to(self.device)

    def z_at(self, idx):
        return self.z[idx].to(self.device)


def export(model, path):
    with open(path, "wb") as f:
        f.write(b"NCA3")
        np.array([CH, HIDDEN, ZDIM], dtype="<i4").tofile(f)
        np.array([FIRE_RATE], dtype="<f4").tofile(f)
        model.w1.weight.detach().cpu().numpy().reshape(HIDDEN, CH * 3 + 2).astype("<f4").tofile(f)
        model.w1.bias.detach().cpu().numpy().astype("<f4").tofile(f)
        model.w2.weight.detach().cpu().numpy().reshape(CH, HIDDEN).astype("<f4").tofile(f)
        model.w2.bias.detach().cpu().numpy().astype("<f4").tofile(f)
        model.film.weight.detach().cpu().numpy().astype("<f4").tofile(f)   # (2H, ZDIM)
        model.film.bias.detach().cpu().numpy().astype("<f4").tofile(f)
    print(f"  exported {path}", flush=True)


def save_preview(model, device, path):
    from PIL import Image
    from manifold_shoggoth import ANCHORS

    names = ["idle", "walk", "sleep", "manic"]
    rows = []
    with torch.no_grad():
        for name in names:
            z = torch.tensor([ANCHORS[name]], dtype=torch.float32, device=device)
            x = make_seed(1, device)
            theta = torch.zeros(1, device=device)
            for _ in range(300):
                x = model(x, theta, z)
                theta += OMEGA
            shots = []
            for _ in range(6):
                for _ in range(40):
                    x = model(x, theta, z)
                    theta += OMEGA
                shots.append(x[0, :4].permute(1, 2, 0).cpu().numpy())
            rows.append(np.concatenate(shots, axis=1))
    sheet = np.concatenate(rows, axis=0)
    img = (np.clip(sheet, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img, "RGBA").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--corpus", default="corpus_shoggoth.npz")
    ap.add_argument("--out", default="shoggoth_manifold.nca")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    corpus = Corpus(args.corpus, device)
    print(f"corpus: {corpus.n} cycles, device {device}", flush=True)
    model = ManifoldNCA().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[20000, 40000], gamma=0.3)

    pool = make_seed(POOL_SIZE, device)
    pool_theta = torch.rand(POOL_SIZE, device=device) * 2 * np.pi
    pool_zidx = torch.randint(0, corpus.n, (POOL_SIZE,))
    t0 = time.time()

    for it in range(1, args.iters + 1):
        idx = torch.randint(0, POOL_SIZE, (BATCH,))
        x = pool[idx].clone()
        theta = pool_theta[idx].clone()
        zidx = pool_zidx[idx].clone()

        with torch.no_grad():
            tgt = corpus.target_at(zidx, theta)
            losses = ((x[:, :4] - tgt) ** 2).mean(dim=(1, 2, 3))
            order = losses.argsort(descending=True)
            order_cpu = order.cpu()  # idx/zidx are CPU tensors; cross-device indexing throws
            x, theta, zidx = x[order], theta[order], zidx[order_cpu]
            x[0] = make_seed(1, device)[0]
            theta[0] = torch.rand(1, device=device) * 2 * np.pi
            zidx[0] = torch.randint(0, corpus.n, (1,))
            switch = torch.rand(BATCH) < ZSWITCH_P
            switch[0] = False
            nswitch = int(switch.sum())
            if nswitch:
                zidx[switch] = torch.randint(0, corpus.n, (nswitch,))
            if it > 1000:
                x[-DAMAGE_N:] = damage(x[-DAMAGE_N:])

        z = corpus.z_at(zidx)
        T = np.random.randint(48, 81)
        loss = torch.zeros((), device=device)
        checkpoints = {T - 24: 0.5, T - 12: 0.75, T - 1: 1.0}
        for t in range(T):
            x = model(x, theta, z)
            theta = theta + OMEGA
            if t in checkpoints:
                loss = loss + checkpoints[t] * ((x[:, :4] - corpus.target_at(zidx, theta)) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        opt.step()
        sched.step()

        with torch.no_grad():
            slots = idx[order_cpu]
            pool[slots] = x.detach()
            pool_theta[slots] = theta % (2 * np.pi)
            pool_zidx[slots] = zidx

        if it % 100 == 0:
            rate = it / (time.time() - t0)
            print(f"iter {it:6d}  loss {loss.item():.5f}  {rate:.2f} it/s", flush=True)
        if it % 2000 == 0 or it == args.iters:
            export(model, args.out)
            save_preview(model, device, "manifold_preview.png")


if __name__ == "__main__":
    main()
