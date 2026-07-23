"""Trains a phase-conditioned cyclic NCA: an automaton whose target is a MOVING
point on a closed loop through sprite-space (an animation cycle), not a fixed image.

Conditioning channels appended to the perception vector: (sin th, cos th, behavior).
The phase th advances OMEGA per step *during* training rollouts, so the network
learns to track continuous motion — coherent animation, not frame lookup. Behavior
is re-rolled mid-life on some samples so still<->talking transitions are trained.

Exports NCA2 format: like NCA1 plus an int32 cond-channel count in the header.

Usage: python3 train_cyclic.py [--iters 12000] [--out ../weights/lain.nca]
"""

import argparse
import importlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Creature frame module is chosen via --creature; these are populated in main().
# lain.py and shoggoth.py share the interface: FRAMES, BEHAVIORS, GRID, make_frames().
BEHAVIORS = FRAMES = GRID = make_frames = None


def _load_creature(name):
    global BEHAVIORS, FRAMES, GRID, make_frames
    mod = importlib.import_module(name)
    BEHAVIORS, FRAMES, GRID = mod.BEHAVIORS, mod.FRAMES, mod.GRID
    make_frames = mod.make_frames

CH = 16
HIDDEN = 128
COND = 3            # sin(theta), cos(theta), behavior flag
FIRE_RATE = 0.5
OMEGA = 2 * np.pi / 240.0   # one animation cycle = 240 automaton steps
POOL_SIZE = 1024
BATCH = 8
DAMAGE_N = 2
SWITCH_P = 0.15     # chance a live sample's behavior is re-rolled (transition training)


class CyclicNCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Conv2d(CH * 3 + COND, HIDDEN, 1)
        self.w2 = nn.Conv2d(HIDDEN, CH, 1)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sy = sx.T.contiguous()
        kernels = torch.stack([ident, sx, sy]).repeat(CH, 1, 1).unsqueeze(1)
        self.register_buffer("percept_w", kernels)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def forward(self, x, cond):
        """cond: (B, COND) — broadcast to every cell, appended after perception."""
        pre_life = self.alive(x)
        p = F.conv2d(x, self.percept_w, padding=1, groups=CH)
        cmap = cond[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        dx = self.w2(F.relu(self.w1(torch.cat([p, cmap], dim=1))))
        fire = (torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) <= FIRE_RATE).float()
        x = x + dx * fire
        life = (pre_life & self.alive(x)).float()
        return x * life


def cond_for(thetas, behaviors):
    return torch.stack([torch.sin(thetas), torch.cos(thetas), behaviors.float()], dim=1)


def target_at(frames_t, behaviors, thetas):
    """Phase-lerped target: frames are 12 points on the cycle; theta lands between them."""
    pos = (thetas / (2 * np.pi) * FRAMES) % FRAMES
    f0 = pos.long() % FRAMES
    f1 = (f0 + 1) % FRAMES
    w = (pos - pos.floor())[:, None, None, None]
    return frames_t[behaviors, f0] * (1 - w) + frames_t[behaviors, f1] * w


def sample_error(prediction, target, foreground_weighted=False):
    """Return per-sample error for pool ranking."""
    if not foreground_weighted:
        return (prediction - target).square().mean(dim=(1, 2, 3))
    color_weight = 0.25 + target[:, 3:4]
    color = ((prediction[:, :3] - target[:, :3]).square() * color_weight)
    color = color.mean(dim=(1, 2, 3))
    alpha = (prediction[:, 3:4] - target[:, 3:4]).square().mean(dim=(1, 2, 3))
    return color + 2.0 * alpha


def make_seed(n, device):
    x = torch.zeros(n, CH, GRID, GRID, device=device)
    x[:, 3:, GRID // 2, GRID // 2] = 1.0
    return x


def make_mature_state(visible):
    """Embed premultiplied RGBA targets in live 16-channel NCA states."""
    state = visible.new_zeros(visible.shape[0], CH, *visible.shape[2:])
    state[:, :4] = visible
    state[:, 4:] = visible[:, 3:4]
    return state


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


def load_nca2(model, path):
    """Warm-start from an exported NCA2 file (continuation training)."""
    import struct

    with open(path, "rb") as f:
        assert f.read(4) == b"NCA2", "not an NCA2 file"
        ch, hid, cond = struct.unpack("<3i", f.read(12))
        assert (ch, hid, cond) == (CH, HIDDEN, COND), "shape mismatch"
        f.read(4)  # fire rate
        def arr(n):
            return np.frombuffer(f.read(n * 4), dtype="<f4").copy()
        with torch.no_grad():
            model.w1.weight.copy_(torch.from_numpy(arr(HIDDEN * (CH * 3 + COND))).view(HIDDEN, CH * 3 + COND, 1, 1))
            model.w1.bias.copy_(torch.from_numpy(arr(HIDDEN)))
            model.w2.weight.copy_(torch.from_numpy(arr(CH * HIDDEN)).view(CH, HIDDEN, 1, 1))
            model.w2.bias.copy_(torch.from_numpy(arr(CH)))


def export(model, path):
    with open(path, "wb") as f:
        f.write(b"NCA2")
        np.array([CH, HIDDEN, COND], dtype="<i4").tofile(f)
        np.array([FIRE_RATE], dtype="<f4").tofile(f)
        model.w1.weight.detach().cpu().numpy().reshape(HIDDEN, CH * 3 + COND).astype("<f4").tofile(f)
        model.w1.bias.detach().cpu().numpy().astype("<f4").tofile(f)
        model.w2.weight.detach().cpu().numpy().reshape(CH, HIDDEN).astype("<f4").tofile(f)
        model.w2.bias.detach().cpu().numpy().astype("<f4").tofile(f)


def save_preview(model, frames_t, device, path, init_mode="point"):
    """Grow while the phase runs, then snapshot one full cycle: 12 talk frames."""
    from PIL import Image

    with torch.no_grad():
        theta = torch.zeros(1, device=device)
        preview_behavior = 1 if BEHAVIORS > 1 else 0
        b = torch.full((1,), preview_behavior, dtype=torch.long, device=device)
        if init_mode == "target":
            x = make_mature_state(target_at(frames_t, b, theta))
        else:
            x = make_seed(1, device)
            for _ in range(300):
                x = model(x, cond_for(theta, b))
                theta += OMEGA
        shots = []
        for _ in range(FRAMES):
            for _ in range(240 // FRAMES):
                x = model(x, cond_for(theta, b))
                theta += OMEGA
            shots.append(x[0, :4].permute(1, 2, 0).cpu().numpy())
    sheet = np.concatenate(shots, axis=1)
    img = (np.clip(sheet, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img, "RGBA").resize((FRAMES * GRID * 2, GRID * 2), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creature", default="lain", choices=["lain", "shoggoth"])
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--init", default=None, help="warm-start from an exported .nca")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--pool", type=int, default=POOL_SIZE)
    ap.add_argument("--rollout-min", type=int, default=48)
    ap.add_argument("--rollout-max", type=int, default=80)
    ap.add_argument("--damage-after", type=int, default=500,
                    help="first damage iteration; 0 disables damage")
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--preview", default=None)
    ap.add_argument("--target", default=None,
                    help="ingested 2d_cycle .npz (tools/ingest.py states/sheet)")
    ap.add_argument(
        "--init-mode", choices=("point", "target"), default="point",
        help="target trains mature-state dynamics before point-seed growth",
    )
    args = ap.parse_args()
    if args.batch <= 0 or args.pool < args.batch:
        ap.error("--batch must be positive and --pool must be at least --batch")
    if args.rollout_min <= 0 or args.rollout_max < args.rollout_min:
        ap.error("rollout bounds must satisfy 0 < min <= max")
    if args.checkpoint_every <= 0:
        ap.error("--checkpoint-every must be positive")
    if args.init_mode == "target" and not args.target:
        ap.error("--init-mode target requires --target")
    global BEHAVIORS, FRAMES, GRID, make_frames
    if args.target:
        _npz = np.load(args.target, allow_pickle=True)
        assert str(_npz["kind"]) == "2d_cycle", "expected a 2d_cycle npz"
        _frames_np = _npz["frames"].astype(np.float32)
        BEHAVIORS, FRAMES, GRID = _frames_np.shape[0], _frames_np.shape[1], _frames_np.shape[2]
        assert BEHAVIORS <= 2, "binary behavior flag supports 2 states; use the manifold trainer for more"
        make_frames = lambda: _frames_np
        if args.out is None:
            args.out = "../weights/ingested_cycle.nca"
    else:
        _load_creature(args.creature)
        if args.out is None:
            args.out = f"../weights/{args.creature}.nca"

    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    frames_t = torch.from_numpy(make_frames()).permute(0, 1, 4, 2, 3).to(device)  # (B,F,4,H,W)
    visible_loss = None
    if args.target:
        from transport_targets2d import visible_objective
        visible_loss = visible_objective
    model = CyclicNCA().to(device)
    if args.init:
        load_nca2(model, args.init)
        print(f"warm-started from {args.init}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[4000], gamma=0.1)

    pool_theta = torch.rand(args.pool, device=device) * 2 * np.pi
    pool_beh = torch.randint(0, BEHAVIORS, (args.pool,), device=device)
    pool = (
        make_mature_state(target_at(frames_t, pool_beh, pool_theta))
        if args.init_mode == "target"
        else make_seed(args.pool, device)
    )
    preview_path = args.preview
    if preview_path is None:
        preview_path = (str(Path(args.out).with_suffix(".preview.png"))
                        if args.target else f"cyclic_preview_{args.creature}.png")
    print(
        f"grid {GRID} batch {args.batch} pool {args.pool} rollout "
        f"{args.rollout_min}-{args.rollout_max} damage-after "
        f"{args.damage_after or 'off'}",
        flush=True,
    )
    t0 = time.time()

    for it in range(1, args.iters + 1):
        idx = torch.randint(0, args.pool, (args.batch,))
        x = pool[idx].clone()
        theta = pool_theta[idx].clone()
        beh = pool_beh[idx].clone()

        with torch.no_grad():
            losses = sample_error(
                x[:, :4], target_at(frames_t, beh, theta),
                foreground_weighted=visible_loss is not None,
            )
            order = losses.argsort(descending=True)
            x, theta, beh = x[order], theta[order], beh[order]
            theta[0] = torch.rand(1, device=device) * 2 * np.pi
            beh[0] = torch.randint(0, BEHAVIORS, (1,), device=device)
            x[0] = (
                make_mature_state(
                    target_at(frames_t, beh[:1], theta[:1])
                )[0]
                if args.init_mode == "target"
                else make_seed(1, device)[0]
            )
            switch = torch.rand(args.batch, device=device) < SWITCH_P
            switch[0] = False
            if BEHAVIORS == 2:
                beh[switch] = 1 - beh[switch]      # mid-life behavior flip
            if args.damage_after > 0 and it >= args.damage_after:
                x[-DAMAGE_N:] = damage(x[-DAMAGE_N:])

        T = np.random.randint(args.rollout_min, args.rollout_max + 1)
        loss = torch.zeros((), device=device)
        checkpoints = {T - 24: 0.5, T - 12: 0.75, T - 1: 1.0}
        for t in range(T):
            x = model(x, cond_for(theta, beh))
            theta = theta + OMEGA
            if t in checkpoints:
                tgt = target_at(frames_t, beh, theta)
                checkpoint_loss = (
                    visible_loss(x[:, :4], tgt)[0]
                    if visible_loss is not None
                    else (x[:, :4] - tgt).square().mean()
                )
                loss = loss + checkpoints[t] * checkpoint_loss

        opt.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        opt.step()
        sched.step()

        with torch.no_grad():
            pool[idx[order.cpu()]] = x.detach()
            pool_theta[idx[order.cpu()]] = theta % (2 * np.pi)
            pool_beh[idx[order.cpu()]] = beh

        if it % 50 == 0:
            rate = it / (time.time() - t0)
            print(f"iter {it:5d}  loss {loss.item():.5f}  {rate:.2f} it/s", flush=True)
        if it % args.checkpoint_every == 0 or it == args.iters:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
            export(model, args.out)
            save_preview(model, frames_t, device, preview_path, args.init_mode)
            print(f"  checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
