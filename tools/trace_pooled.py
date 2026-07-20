"""Read out the global variable of a pooled NCA (NCAP checkpoint).

The pooled creature's whole premise is that g — the alive-masked spatial mean
broadcast to every cell — becomes a state variable with its own dynamics. The
failure mode is that training finds it easier to ignore g, driving it to a
constant; then we have paid for broken locality and bought nothing.

This tells the two apart. It grows a creature from seed, traces g over a free
rollout, and reports how much g actually moves relative to its own magnitude.

Usage: python3 trace_pooled.py ../weights/F_pooled.nca [--steps 600] [--state 0]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import train_states                                    # noqa: E402
from pooled_nca import PooledNCA                       # noqa: E402


def load_ncap(path, device):
    with open(path, "rb") as f:
        assert f.read(4) == b"NCAP", f"{path} is not an NCAP checkpoint"
        ch, hidden, cond, npool = (int(v) for v in np.fromfile(f, dtype="<i4", count=4))
        np.fromfile(f, dtype="<f4", count=1)
        pin = ch * 3 + cond + npool
        w1 = np.fromfile(f, dtype="<f4", count=hidden * pin).reshape(hidden, pin)
        b1 = np.fromfile(f, dtype="<f4", count=hidden)
        w2 = np.fromfile(f, dtype="<f4", count=ch * hidden).reshape(ch, hidden)
        b2 = np.fromfile(f, dtype="<f4", count=ch)
    train_states.HIDDEN = hidden
    model = PooledNCA(npool=npool).to(device)
    model.w1.weight.data.copy_(torch.from_numpy(w1).to(device)[..., None, None])
    model.w1.bias.data.copy_(torch.from_numpy(b1).to(device))
    model.w2.weight.data.copy_(torch.from_numpy(w2).to(device)[..., None, None])
    model.w2.bias.data.copy_(torch.from_numpy(b2).to(device))
    return model, npool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--state", type=int, default=0)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--plot", default=None, help="write a PNG of g over time")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    train_states.GRID = args.grid
    model, npool = load_ncap(args.checkpoint, device)
    print(f"{args.checkpoint}: npool {npool}, hidden {train_states.HIDDEN}, grid {args.grid}")

    x = train_states.make_seed(1, device)
    st = torch.full((1,), args.state, device=device, dtype=torch.long)
    from pooled_nca import trace_global
    g = trace_global(model, x, st, args.steps)[:, 0, :]     # (steps, npool)

    # Ignore the growth transient; we care whether g moves once the body exists.
    warm = g[args.steps // 3:]
    for c in range(npool):
        col = warm[:, c]
        span = col.max() - col.min()
        scale = max(abs(col.mean()), 1e-6)
        print(f"  g[{c}]: mean {col.mean():+.4f}  range {span:.4f}  "
              f"range/|mean| {span / scale:.3f}")
    # Amplitude alone cannot distinguish a state variable from a noise channel:
    # g is a mean over ~2000 stochastically-firing cells, so it jitters even if
    # the update rule ignores it entirely. Structured dynamics decorrelate over
    # many steps; sampling noise decorrelates in one.
    print("\n  autocorrelation of g (lag in steps):")
    tau = []
    for c in range(npool):
        col = warm[:, c] - warm[:, c].mean()
        denom = (col * col).sum()
        ac = [float((col[:-k] * col[k:]).sum() / denom) for k in (1, 5, 20, 100)]
        # decorrelation time: first lag where autocorrelation drops under 1/e
        t = next((k for k in range(1, len(col) // 2)
                  if (col[:-k] * col[k:]).sum() / denom < 0.3679), len(col) // 2)
        tau.append(t)
        print(f"    g[{c}]: lag1 {ac[0]:+.3f}  lag5 {ac[1]:+.3f}  "
              f"lag20 {ac[2]:+.3f}  lag100 {ac[3]:+.3f}   tau~{t} steps")

    total = (warm.max(0) - warm.min(0)).max()
    slowest = max(tau)
    print(f"\nverdict: post-transient swing {total:.4f}, slowest decorrelation "
          f"~{slowest} steps")
    if total <= 0.02:
        print("  -> g is effectively CONSTANT; pooling bought nothing")
    elif slowest < 5:
        print("  -> g moves but decorrelates immediately: this is a NOISE "
              "channel, not a state variable")
    else:
        print(f"  -> g is a slow state variable ({slowest} steps vs 1-step local "
              "dynamics): the slow-fast structure we wanted")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for c in range(npool):
            plt.plot(g[:, c], label=f"g[{c}]")
        plt.axvline(args.steps // 3, ls=":", c="k", lw=0.8)
        plt.xlabel("step"); plt.ylabel("global channel"); plt.legend(fontsize=7)
        plt.title(os.path.basename(args.checkpoint))
        plt.tight_layout(); plt.savefig(args.plot, dpi=120)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
