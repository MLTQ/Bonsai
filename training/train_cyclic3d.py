"""Rung 2: phase-conditioned cyclic NCA in 3D — Shoggoth Mk. III learns to churn.

train_nca3d.py's volumetric machinery + train_cyclic.py's moving-target recipe:
(sin th, cos th, behavior) appended to the 64 perception features, theta advancing
OMEGA per step during checkpointed rollouts, mid-life behavior flips for
idle<->walk transition training, sphere damage for regeneration.

Exports NC3C: magic, i32 ch, i32 hidden, i32 cond, f32 fire, then
w1[hidden][ch*4+cond], b1, w2[ch][hidden], b2.

Usage: python3 train_cyclic3d.py [--iters 24000] [--out ../weights/shoggoth3d.nca]
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from shoggoth3d import BEHAVIORS, FRAMES, make_frames3d
from target3d import GRID3

CH = 16
HIDDEN = 128
COND = 3
FIRE_RATE = 0.5
OMEGA = 2 * np.pi / 240.0
POOL_SIZE = 256
BATCH = 8
DAMAGE_N = 2
SWITCH_P = 0.15
CHUNK = 8


class CyclicNCA3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Conv3d(CH * 4 + COND, HIDDEN, 1)
        self.w2 = nn.Conv3d(HIDDEN, CH, 1)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
        ident = torch.zeros(3, 3, 3)
        ident[1, 1, 1] = 1.0
        smooth = torch.tensor([1.0, 2.0, 1.0])
        deriv = torch.tensor([-1.0, 0.0, 1.0])
        sz = torch.einsum("i,j,k->ijk", deriv, smooth, smooth) / 32.0
        sy = torch.einsum("i,j,k->ijk", smooth, deriv, smooth) / 32.0
        sx = torch.einsum("i,j,k->ijk", smooth, smooth, deriv) / 32.0
        kernels = torch.stack([ident, sx, sy, sz]).repeat(CH, 1, 1, 1).unsqueeze(1)
        self.register_buffer("percept_w", kernels)

    def alive(self, x):
        return F.max_pool3d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def forward(self, x, cond):
        pre_life = self.alive(x)
        p = F.conv3d(x, self.percept_w, padding=1, groups=CH)
        cmap = cond[:, :, None, None, None].expand(-1, -1, *x.shape[2:])
        dx = self.w2(F.relu(self.w1(torch.cat([p, cmap], dim=1))))
        fire = (torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) <= FIRE_RATE).float()
        x = x + dx * fire
        life = (pre_life & self.alive(x)).float()
        return (x * life).clamp(-8.0, 8.0)

    use_checkpoint = True
    step_fn = None  # optionally a torch.compile'd wrapper around forward

    def rollout(self, x, theta0, beh, steps):
        """Rollout; theta advances OMEGA per step. Checkpointed in CHUNK segments
        unless use_checkpoint is False (viable on 80GB cards, kills recompute)."""
        fwd = self.step_fn if self.step_fn is not None else self.forward

        def run_chunk(x0, th0, n):
            for i in range(int(n)):
                th = th0 + i * OMEGA
                cond = torch.stack([torch.sin(th), torch.cos(th), beh.float()], dim=1)
                x0 = fwd(x0, cond)
            return x0

        if not self.use_checkpoint:
            return run_chunk(x, theta0, steps)

        done = 0
        while done < steps:
            n = min(CHUNK, steps - done)
            th0 = theta0 + done * OMEGA
            if self.training and x.requires_grad:
                x = checkpoint(run_chunk, x, th0, torch.tensor(n), use_reentrant=False)
            else:
                x = run_chunk(x, th0, n)
            done += n
        return x


def target_at(frames_t, behaviors, thetas):
    pos = (thetas / (2 * np.pi) * FRAMES) % FRAMES
    f0 = pos.long() % FRAMES
    f1 = (f0 + 1) % FRAMES
    w = (pos - pos.floor())[:, None, None, None, None]
    return frames_t[behaviors, f0] * (1 - w) + frames_t[behaviors, f1] * w


def make_seed(n, device):
    x = torch.zeros(n, CH, GRID3, GRID3, GRID3, device=device)
    x[:, 3:, GRID3 // 2, GRID3 // 2, GRID3 // 2] = 1.0
    return x


def damage(x):
    n = x.shape[0]
    g = x.shape[2]
    zz, yy, xx = torch.meshgrid(*(torch.arange(g, device=x.device),) * 3, indexing="ij")
    for i in range(n):
        r = np.random.uniform(4, 9)
        cz, cy, cx = (np.random.uniform(g * 0.25, g * 0.75) for _ in range(3))
        x[i] *= (((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) > r ** 2).float()
    return x


def load_nc3c(model, path):
    """Warm-start from an exported NC3C file (continuation / restart-after-config-change)."""
    import struct

    with open(path, "rb") as f:
        assert f.read(4) == b"NC3C", "not an NC3C file"
        ch, hid, cond = struct.unpack("<3i", f.read(12))
        assert (ch, hid, cond) == (CH, HIDDEN, COND), "shape mismatch"
        f.read(4)
        def arr(n):
            return torch.from_numpy(np.frombuffer(f.read(n * 4), dtype="<f4").copy())
        with torch.no_grad():
            model.w1.weight.copy_(arr(HIDDEN * (CH * 4 + COND)).view(HIDDEN, CH * 4 + COND, 1, 1, 1))
            model.w1.bias.copy_(arr(HIDDEN))
            model.w2.weight.copy_(arr(CH * HIDDEN).view(CH, HIDDEN, 1, 1, 1))
            model.w2.bias.copy_(arr(CH))


def export(model, path):
    with open(path, "wb") as f:
        f.write(b"NC3C")
        np.array([CH, HIDDEN, COND], dtype="<i4").tofile(f)
        np.array([FIRE_RATE], dtype="<f4").tofile(f)
        model.w1.weight.detach().cpu().numpy().reshape(HIDDEN, CH * 4 + COND).astype("<f4").tofile(f)
        model.w1.bias.detach().cpu().numpy().astype("<f4").tofile(f)
        model.w2.weight.detach().cpu().numpy().reshape(CH, HIDDEN).astype("<f4").tofile(f)
        model.w2.bias.detach().cpu().numpy().astype("<f4").tofile(f)


def save_preview(model, frames_t, device, path):
    from PIL import Image

    with torch.no_grad():
        x = make_seed(1, device)
        theta = torch.zeros(1, device=device)
        beh = torch.ones(1, dtype=torch.long, device=device)
        x = model.rollout(x, theta, beh, 300)
        theta = theta + 300 * OMEGA
        shots = []
        for _ in range(4):
            x = model.rollout(x, theta, beh, 60)
            theta = theta + 60 * OMEGA
            vol = x[0, :4].permute(1, 2, 3, 0).cpu().numpy()
            alpha = np.clip(vol[..., 3], 0, 1)
            w = alpha / (alpha.sum(axis=0, keepdims=True) + 1e-6)
            rgb = (np.clip(vol[..., :3], 0, 1) * w[..., None]).sum(axis=0)
            a = 1 - np.prod(1 - alpha * 0.9, axis=0)
            shots.append(np.concatenate([rgb, a[..., None]], axis=-1)[::-1])
    sheet = np.concatenate(shots, axis=1)
    png = (np.clip(sheet, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(png, "RGBA").resize((GRID3 * 4 * 4, GRID3 * 4), Image.NEAREST).save(path)


def main():
    global POOL_SIZE, BATCH, CHUNK
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=24000)
    ap.add_argument("--out", default="../weights/shoggoth3d.nca")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--pool", type=int, default=POOL_SIZE)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--init", default=None, help="warm-start from an exported .nca")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(reduce-overhead): fuse + CUDA-graph each step")
    ap.add_argument("--no-ckpt", action="store_true",
                    help="disable gradient checkpointing (needs ~big VRAM; kills recompute)")
    ap.add_argument("--fixed-t", type=int, default=0,
                    help="fix rollout length (avoids recompiles under --compile)")
    args = ap.parse_args()
    POOL_SIZE, BATCH, CHUNK = args.pool, args.batch, args.chunk
    print(f"grid {GRID3}^3, pool {POOL_SIZE}, batch {BATCH}, chunk {CHUNK}, "
          f"compile={args.compile}, ckpt={not args.no_ckpt}, fixedT={args.fixed_t}", flush=True)

    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    print("generating frames...", flush=True)
    frames_t = torch.from_numpy(make_frames3d()).permute(0, 1, 5, 2, 3, 4).float().to(device)
    model = CyclicNCA3D().to(device)
    if args.init:
        load_nc3c(model, args.init)
        print(f"warm-started from {args.init}", flush=True)
    model.use_checkpoint = not args.no_ckpt
    if args.compile:
        model.step_fn = torch.compile(model.forward, mode="reduce-overhead")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[8000, 18000], gamma=0.3)

    pool = make_seed(POOL_SIZE, device)
    pool_theta = torch.rand(POOL_SIZE, device=device) * 2 * np.pi
    pool_beh = torch.randint(0, BEHAVIORS, (POOL_SIZE,), device=device)
    t0 = time.time()

    for it in range(1, args.iters + 1):
        idx = torch.randint(0, POOL_SIZE, (BATCH,))
        x = pool[idx].clone()
        theta = pool_theta[idx].clone()
        beh = pool_beh[idx].clone()

        with torch.no_grad():
            losses = ((x[:, :4] - target_at(frames_t, beh, theta)) ** 2).mean(dim=(1, 2, 3, 4))
            order = losses.argsort(descending=True)
            order_cpu = order.cpu()
            x, theta, beh = x[order], theta[order], beh[order]
            x[0] = make_seed(1, device)[0]
            theta[0] = torch.rand(1, device=device) * 2 * np.pi
            beh[0] = torch.randint(0, BEHAVIORS, (1,), device=device)
            switch = torch.rand(BATCH, device=device) < SWITCH_P
            switch[0] = False
            beh[switch] = 1 - beh[switch]
            if it > 1000:
                x[-DAMAGE_N:] = damage(x[-DAMAGE_N:])

        T = args.fixed_t if args.fixed_t else int(np.random.randint(48, 73))
        x.requires_grad_(True)
        out = model.rollout(x, theta, beh, T)
        loss = ((out[:, :4] - target_at(frames_t, beh, theta + T * OMEGA)) ** 2).mean()

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
            slots = idx[order_cpu][finite.cpu()]
            pool[slots] = out_d[finite]
            pool_theta[slots] = ((theta + T * OMEGA) % (2 * np.pi))[finite]
            pool_beh[slots] = beh[finite]

        if it % 50 == 0:
            rate = it / (time.time() - t0)
            print(f"iter {it:6d}  loss {loss.item():.5f}  {rate:.2f} it/s", flush=True)
        if it % 500 == 0 or it == args.iters:
            export(model, args.out)
            save_preview(model, frames_t, device, "preview3d_cyclic.png")
            print(f"  checkpoint -> {args.out}", flush=True)
        if it % 5000 == 0:
            export(model, f"{args.out}.it{it}")


if __name__ == "__main__":
    main()
