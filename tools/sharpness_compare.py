"""Compare local (NCA2) and pooled (NCAP) creatures on sharpness and traversal.

`verify_constellation.py` renders through the Swift runtime, which cannot parse
NCAP yet. Measuring a pooled creature there and a local creature in Swift would
make the renderer a confound, so this runs *both* families through one PyTorch
harness and reports the same statistics for each.

Sharpness is mean gradient magnitude (Sobel) of the rendered RGB, which is what
"looks blurry" means numerically: a creature that reproduces colour but not
edges scores far below its targets. The target set's own sharpness is the
ceiling.

Usage:
  python3 sharpness_compare.py target.npz local.nca pooled.nca [--steps 1200]
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import train_states                                        # noqa: E402


def load_any(path, device):
    """Load NCA2 (strict-local) or NCAP (pooled); returns (model, kind)."""
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == b"NCAP":
        from trace_pooled import load_ncap
        model, npool = load_ncap(path, device)
        return model, f"pooled(npool={npool})"
    assert magic == b"NCA2", f"{path}: unknown magic {magic!r}"
    with open(path, "rb") as f:
        f.read(4)
        ch, hidden, cond = (int(v) for v in np.fromfile(f, dtype="<i4", count=3))
        np.fromfile(f, dtype="<f4", count=1)
        pin = ch * 3 + cond
        w1 = np.fromfile(f, dtype="<f4", count=hidden * pin).reshape(hidden, pin)
        b1 = np.fromfile(f, dtype="<f4", count=hidden)
        w2 = np.fromfile(f, dtype="<f4", count=ch * hidden).reshape(ch, hidden)
        b2 = np.fromfile(f, dtype="<f4", count=ch)
    train_states.HIDDEN = hidden
    model = train_states.StateNCA().to(device)
    model.w1.weight.data.copy_(torch.from_numpy(w1).to(device)[..., None, None])
    model.w1.bias.data.copy_(torch.from_numpy(b1).to(device))
    model.w2.weight.data.copy_(torch.from_numpy(w2).to(device)[..., None, None])
    model.w2.bias.data.copy_(torch.from_numpy(b2).to(device))
    return model, "local"


def sharpness(rgb):
    """Mean Sobel gradient magnitude of an (N,3,H,W) batch, alpha-independent."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=rgb.device)[None, None] / 8.0
    ky = kx.transpose(-1, -2).contiguous()
    n = rgb.shape[1]
    gx = F.conv2d(rgb, kx.repeat(n, 1, 1, 1), padding=1, groups=n)
    gy = F.conv2d(rgb, ky.repeat(n, 1, 1, 1), padding=1, groups=n)
    return float(torch.sqrt(gx ** 2 + gy ** 2 + 1e-12).mean())


def evaluate(model, poses_t, state, steps, device, sample_every=20):
    x = train_states.make_seed(1, device)
    st = torch.full((1,), state, device=device, dtype=torch.long)
    frames, ids = [], []
    with torch.no_grad():
        for i in range(steps):
            x = model(x, st)
            if i >= steps // 3 and i % sample_every == 0:
                frames.append(x[:, :3].clone())
                d = ((x[:, :4] - poses_t) ** 2).mean(dim=(1, 2, 3))
                ids.append(int(d.argmin()))
    rgb = torch.cat(frames, 0).clamp(0, 1)
    ids = np.array(ids)
    fwd = rev = 0
    n = len(poses_t)
    for a, b in zip(ids[:-1], ids[1:]):
        if a == b:
            continue
        step = (b - a) % n
        if step <= n // 2:
            fwd += 1
        else:
            rev += 1
    return sharpness(rgb), len(np.unique(ids)), fwd, rev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--state", type=int, default=0)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = np.load(args.target, allow_pickle=True)
    poses = data["poses"].astype(np.float32)
    pose_state = data["pose_state"].astype(np.int64)
    train_states.GRID = poses.shape[1]
    keep = np.where(pose_state == args.state)[0]
    poses_t = torch.from_numpy(poses[keep]).permute(0, 3, 1, 2).to(device)

    ceiling = sharpness(poses_t[:, :3].clamp(0, 1).to(device))
    print(f"target sharpness (ceiling): {ceiling:.5f}   "
          f"{len(keep)} poses, grid {train_states.GRID}\n")

    for path in args.checkpoints:
        model, kind = load_any(path, device)
        sh, seen, fwd, rev = [], [], [], []
        for t in range(args.trials):
            torch.manual_seed(500 + t)
            s, u, f_, r = evaluate(model, poses_t, args.state, args.steps, device)
            sh.append(s); seen.append(u); fwd.append(f_); rev.append(r)
        sh = np.array(sh)
        print(f"{os.path.basename(path):24s} [{kind}]")
        print(f"    sharpness {sh.mean():.5f} +/- {sh.std():.5f}  "
              f"= {sh.mean()/ceiling*100:.0f}% of target")
        print(f"    poses visited {np.mean(seen):.1f}/{len(keep)}   "
              f"forward {np.mean(fwd):.0f}  reverse {np.mean(rev):.0f}\n")


if __name__ == "__main__":
    main()
