"""Fixed-set evaluation of soul checkpoints — a convergence signal the training
loss cannot provide.

Training loss here samples 4 random (mood, phase) pairs per iteration out of a
192-cycle corpus, so consecutive readings measure *different things*. Its
variance across batches swamps the model's improvement (measured: noise 2.9x
the total drop), which is why the training curve looks like a random walk
whether or not the model is learning.

This evaluates every archived checkpoint against the SAME moods, the SAME
phases, and the same RNG seed. Differences between checkpoints are then model
quality and nothing else.

Usage: python3 eval_soul.py [--moods idle,serene,melancholy,dizzy,angry,cowboy]
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import train_manifold3d as T                              # noqa: E402
import manifold_claudeguy3d as M                          # noqa: E402


def load_nc3m(path, device):
    """Load an NC3M checkpoint (including its FiLM block) into a fresh model."""
    with open(path, "rb") as f:
        assert f.read(4) == b"NC3M", f"{path} is not NC3M"
        ch, hidden, zdim = (int(v) for v in np.fromfile(f, dtype="<i4", count=3))
        np.fromfile(f, dtype="<f4", count=1)              # fire rate
        w1in = ch * 4 + 2
        w1 = np.fromfile(f, dtype="<f4", count=hidden * w1in).reshape(hidden, w1in)
        b1 = np.fromfile(f, dtype="<f4", count=hidden)
        w2 = np.fromfile(f, dtype="<f4", count=ch * hidden).reshape(ch, hidden)
        b2 = np.fromfile(f, dtype="<f4", count=ch)
        fw = np.fromfile(f, dtype="<f4", count=2 * hidden * zdim).reshape(2 * hidden, zdim)
        fb = np.fromfile(f, dtype="<f4", count=2 * hidden)
    model = T.ManifoldNCA3D().to(device)
    model.w1.weight.data.copy_(torch.from_numpy(w1).to(device)[..., None, None, None])
    model.w1.bias.data.copy_(torch.from_numpy(b1).to(device))
    model.w2.weight.data.copy_(torch.from_numpy(w2).to(device)[..., None, None, None])
    model.w2.bias.data.copy_(torch.from_numpy(b2).to(device))
    model.film.weight.data.copy_(torch.from_numpy(fw).to(device))
    model.film.bias.data.copy_(torch.from_numpy(fb).to(device))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moods", default="idle,serene,melancholy,dizzy,angry,cowboy")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lineage", default="../weights/claudeguy_lineage")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    moods = [m.strip() for m in args.moods.split(",") if m.strip() in M.ANCHORS]
    print(f"device {device}, grid {T.GRID3}^3, {len(moods)} moods: {', '.join(moods)}")

    # Fixed targets: the corpus frame each mood should hold at the evaluated phase.
    phase_idx = 0                                   # frame 0 of the cycle
    targets, zs = [], []
    for name in moods:
        z = np.array(M.ANCHORS[name], dtype=np.float32)
        zs.append(z)
        targets.append(M.draw_mood(2 * np.pi * phase_idx / M.FRAMES, z))
    tgt = torch.from_numpy(np.stack(targets)).permute(0, 4, 1, 2, 3).float().to(device)
    zt = torch.from_numpy(np.stack(zs)).to(device)

    ckpts = []
    for p in glob.glob(os.path.join(args.lineage, "soul_it*.nca")):
        m = re.search(r"soul_it(\d+)\.nca", p)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort()
    print(f"{len(ckpts)} checkpoints: {[c[0] for c in ckpts]}\n")

    rows = []
    for it, path in ckpts:
        model = load_nc3m(path, device)
        # One mood at a time: six 64^3 rollouts at once exhausts MPS unified
        # memory. Per-mood seeding keeps the fire pattern identical across
        # checkpoints, which is what makes the comparison fair.
        per_mood = np.zeros(len(moods), dtype=np.float32)
        for j in range(len(moods)):
            torch.manual_seed(1234 + j)
            x = T.make_seed(1, device)
            theta = torch.zeros(1, device=device)
            with torch.no_grad():
                x = model.rollout(x, theta, zt[j:j + 1], args.steps)
            per_mood[j] = float(((x[:, :4] - tgt[j:j + 1]) ** 2).mean())
            del x
        rows.append((it, per_mood))
        print(f"  it {it:6d}  mean MSE {per_mood.mean():.5f}   "
              + "  ".join(f"{n[:4]} {v:.4f}" for n, v in zip(moods, per_mood)), flush=True)

    its = np.array([r[0] for r in rows])
    means = np.array([r[1].mean() for r in rows])
    print(f"\nfirst {means[0]:.5f} -> last {means[-1]:.5f} "
          f"({100*(means[0]-means[-1])/means[0]:+.1f}%)")
    if len(means) > 3:
        late = means[len(means) // 2:]
        print(f"second half: {late.min():.5f} min, {late.max():.5f} max, "
              f"spread {100*(late.max()-late.min())/late.mean():.1f}% of mean")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(moods):
        ax.plot(its, [r[1][i] for r in rows], "o-", ms=4, lw=1, alpha=0.7, label=name)
    ax.plot(its, means, "k-", lw=2.5, label="mean")
    ax.set_xlabel("training iteration"); ax.set_ylabel("MSE vs fixed target")
    ax.set_title("Soul checkpoints on a FIXED evaluation set")
    ax.legend(fontsize=8); fig.tight_layout()
    out = "../experiments/soul_fixed_eval.png"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
