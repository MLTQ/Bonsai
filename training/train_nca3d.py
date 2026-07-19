"""Trains a 3D (volumetric) Growing NCA — rung 1 of the 3D ladder: static target.

Same recipe as train_nca.py lifted one dimension: perception is identity +
Sobel x/y/z per channel (4 kernels, 64 features), damage is a sphere, and the
sample pool + life mask + stochastic fire all carry over unchanged in kind.

The one new engineering piece: BPTT through 48-72 steps at 32^3 stores ~14 GB
of activations naively, so rollouts run through torch.utils.checkpoint in
8-step chunks (peak memory ~ batch x chunk, ~35% recompute overhead).

Exports NC3D: magic, i32 ch, i32 hidden, f32 fire, then w1[hidden][ch*4],
b1, w2[ch][hidden], b2 — same flat layout family as NCA1, wider perception.

Usage: python3 train_nca3d.py [--iters 20000] [--out ../weights/bonsai3d.nca]
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from target3d import GRID3, make_target3d

CH = 16
HIDDEN = 128
FIRE_RATE = 0.5
POOL_SIZE = 256
BATCH = 8
DAMAGE_N = 2
CHUNK = 8  # rollout steps per checkpoint segment; overridable via --chunk


class NCA3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Conv3d(CH * 4, HIDDEN, 1)
        self.w2 = nn.Conv3d(HIDDEN, CH, 1)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

        ident = torch.zeros(3, 3, 3)
        ident[1, 1, 1] = 1.0
        # 3D Sobel: smoothing (1,2,1) on two axes, derivative (-1,0,1) on the third.
        smooth = torch.tensor([1.0, 2.0, 1.0])
        deriv = torch.tensor([-1.0, 0.0, 1.0])
        # volume index order is (z, y, x)
        sz = torch.einsum("i,j,k->ijk", deriv, smooth, smooth) / 32.0
        sy = torch.einsum("i,j,k->ijk", smooth, deriv, smooth) / 32.0
        sx = torch.einsum("i,j,k->ijk", smooth, smooth, deriv) / 32.0
        kernels = torch.stack([ident, sx, sy, sz])              # (4,3,3,3)
        kernels = kernels.repeat(CH, 1, 1, 1).unsqueeze(1)      # (CH*4,1,3,3,3)
        self.register_buffer("percept_w", kernels)

    def alive(self, x):
        return F.max_pool3d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def forward(self, x):
        pre_life = self.alive(x)
        p = F.conv3d(x, self.percept_w, padding=1, groups=CH)
        dx = self.w2(F.relu(self.w1(p)))
        fire = (torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) <= FIRE_RATE).float()
        x = x + dx * fire
        life = (pre_life & self.alive(x)).float()
        return (x * life).clamp(-8.0, 8.0)

    def rollout(self, x, steps):
        """Checkpointed rollout: recompute chunks in backward instead of storing them."""
        def run_chunk(x0, n):
            for _ in range(int(n)):
                x0 = self.forward(x0)
            return x0

        done = 0
        while done < steps:
            n = min(CHUNK, steps - done)
            if self.training and x.requires_grad:
                x = checkpoint(run_chunk, x, torch.tensor(n), use_reentrant=False)
            else:
                x = run_chunk(x, n)
            done += n
        return x


def make_seed(n, device):
    x = torch.zeros(n, CH, GRID3, GRID3, GRID3, device=device)
    x[:, 3:, GRID3 // 2, GRID3 // 3, GRID3 // 2] = 1.0  # seed low-center (y ~ soil line)
    return x


def damage(x):
    """Zero a random sphere — in 3D the pokes are tunnels and craters."""
    n = x.shape[0]
    g = x.shape[2]
    zz, yy, xx = torch.meshgrid(*(torch.arange(g, device=x.device),) * 3, indexing="ij")
    for i in range(n):
        r = np.random.uniform(4, 9) * (g / 32.0)  # wound size scales with the body
        cz, cy, cx = (np.random.uniform(g * 0.25, g * 0.75) for _ in range(3))
        mask = (((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) > r ** 2).float()
        x[i] *= mask
    return x


def export(model, path):
    with open(path, "wb") as f:
        f.write(b"NC3D")
        np.array([CH, HIDDEN], dtype="<i4").tofile(f)
        np.array([FIRE_RATE], dtype="<f4").tofile(f)
        model.w1.weight.detach().cpu().numpy().reshape(HIDDEN, CH * 4).astype("<f4").tofile(f)
        model.w1.bias.detach().cpu().numpy().astype("<f4").tofile(f)
        model.w2.weight.detach().cpu().numpy().reshape(CH, HIDDEN).astype("<f4").tofile(f)
        model.w2.bias.detach().cpu().numpy().astype("<f4").tofile(f)


def save_preview(model, device, path, steps=None):
    if steps is None:
        steps = int(200 * max(1.0, GRID3 / 32.0))  # bigger bodies take longer to grow
    """Grow from seed, save front max-weighted projection."""
    from PIL import Image

    with torch.no_grad():
        x = make_seed(1, device)
        for _ in range(steps):
            x = model(x)
        vol = x[0, :4].permute(1, 2, 3, 0).cpu().numpy()  # (z,y,x,4)
    alpha = np.clip(vol[..., 3], 0, 1)
    w = alpha / (alpha.sum(axis=0, keepdims=True) + 1e-6)
    rgb = (np.clip(vol[..., :3], 0, 1) * w[..., None]).sum(axis=0)
    a = 1 - np.prod(1 - alpha * 0.9, axis=0)
    img = np.concatenate([rgb, a[..., None]], axis=-1)[::-1]
    png = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(png, "RGBA").resize((GRID3 * 8, GRID3 * 8), Image.NEAREST).save(path)


def main():
    global POOL_SIZE, BATCH, CHUNK
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--out", default="../weights/bonsai3d.nca")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--pool", type=int, default=POOL_SIZE)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()
    POOL_SIZE, BATCH, CHUNK = args.pool, args.batch, args.chunk
    print(f"grid {GRID3}^3, pool {POOL_SIZE}, batch {BATCH}, chunk {CHUNK}", flush=True)

    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    target = torch.from_numpy(make_target3d()).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    model = NCA3D().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[6000, 14000], gamma=0.3)

    pool = make_seed(POOL_SIZE, device)
    t0 = time.time()

    for it in range(1, args.iters + 1):
        idx = torch.randint(0, POOL_SIZE, (BATCH,))
        batch = pool[idx].clone()

        with torch.no_grad():
            losses = ((batch[:, :4] - target) ** 2).mean(dim=(1, 2, 3, 4))
            order = losses.argsort(descending=True)
            order_cpu = order.cpu()
            batch = batch[order]
            batch[0] = make_seed(1, device)[0]
            if it > 500:
                batch[-DAMAGE_N:] = damage(batch[-DAMAGE_N:])

        batch.requires_grad_(True)
        out = model.rollout(batch, int(np.random.randint(48, 73)))
        loss = ((out[:, :4] - target) ** 2).mean()

        if not torch.isfinite(loss):
            print(f"iter {it}: non-finite loss, batch discarded", flush=True)
            opt.zero_grad()
            continue

        opt.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        opt.step()
        sched.step()

        with torch.no_grad():
            out_d = out.detach()
            finite = torch.isfinite(out_d).flatten(1).all(dim=1)
            pool[idx[order_cpu][finite.cpu()]] = out_d[finite]

        if it % 50 == 0:
            rate = it / (time.time() - t0)
            print(f"iter {it:6d}  loss {loss.item():.5f}  {rate:.2f} it/s", flush=True)
        if it % 500 == 0 or it == args.iters:
            export(model, args.out)
            save_preview(model, device, "preview3d.png")
            print(f"  checkpoint -> {args.out}", flush=True)
        if it % 5000 == 0:
            export(model, f"{args.out}.it{it}")


if __name__ == "__main__":
    main()
