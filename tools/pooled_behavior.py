"""Does the pooled creature's global variable correspond to what the body does?

`trace_pooled.py` establishes that g is a slow state variable rather than noise.
That is necessary but not sufficient: a slow variable the body ignores is still
useless. This asks the next question — is g coupled to visible behaviour, and
does it lead or follow?

Method: free-run the creature, and at every step record both g and which target
pose the body is currently nearest to. Then two statistics.

**Coupling (eta^2).** Group g by the pose the body occupies and compute the
correlation ratio: the fraction of g's variance explained by pose identity.
eta^2 ~ 0 means g drifts independently of what the body is doing. High eta^2
means g and the body move together.

**Lead/lag.** Cross-correlate |dg/dt| against pose-transition events across
shifts. If g's excursions peak *before* transitions, g is driving the body —
which is what "nervous system" would mean. If after, g is a readout of a change
that happened for other reasons. This distinction is the whole question, and it
is easy to get backwards by eyeballing a plot.

Usage:
  python3 pooled_behavior.py ../weights/F_pooled.nca \\
      ../experiments/sweep_h100/spirit_ringB.npz [--steps 3000]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import train_states                                        # noqa: E402
from trace_pooled import load_ncap                         # noqa: E402


def run(model, poses_t, state, steps, device):
    """Free rollout; per step return g and the nearest pose index + distance."""
    x = train_states.make_seed(1, device)
    st = torch.full((1,), state, device=device, dtype=torch.long)
    gs, ids, ds = [], [], []
    with torch.no_grad():
        for _ in range(steps):
            g = model.pooled(x, model.alive(x).float()).flatten(1)[0]
            d = ((x[:, :4] - poses_t) ** 2).mean(dim=(1, 2, 3))   # (P,)
            k = int(d.argmin())
            gs.append(g.cpu().numpy())
            ids.append(k)
            ds.append(float(d[k]))
            x = model(x, st)
    return np.stack(gs), np.array(ids), np.array(ds)


def eta_squared(values, groups):
    """Fraction of variance in `values` explained by `groups` membership."""
    total = values.var()
    if total < 1e-12:
        return 0.0
    within = 0.0
    for gid in np.unique(groups):
        sel = values[groups == gid]
        within += len(sel) * sel.var()
    return float(max(0.0, 1.0 - within / len(values) / total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("target")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--state", type=int, default=0)
    ap.add_argument("--trials", type=int, default=5,
                    help="independent rollouts; the timing statistic is noisy "
                         "enough that a single run flips sign between seeds")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    data = np.load(args.target, allow_pickle=True)
    poses = data["poses"].astype(np.float32)
    pose_state = data["pose_state"].astype(np.int64)
    train_states.GRID = poses.shape[1]
    keep = np.where(pose_state == args.state)[0]
    poses_t = torch.from_numpy(poses[keep]).permute(0, 3, 1, 2).to(device)

    model, npool = load_ncap(args.checkpoint, device)
    print(f"{os.path.basename(args.checkpoint)}: npool {npool}, "
          f"{len(keep)} poses in state {args.state}, {args.steps} steps")

    LAGS = list(range(-40, 41, 8))
    warm = args.steps // 3
    etas_all, curves = [], []
    g = ids = dist = None
    for t in range(args.trials):
        torch.manual_seed(1000 + t)
        gt, it_, dt = run(model, poses_t, args.state, args.steps, device)
        gt, it_, dt = gt[warm:], it_[warm:], dt[warm:]
        if g is None:
            g, ids, dist = gt, it_, dt           # keep trial 0 for the plot
        etas_all.append([eta_squared(gt[:, c], it_) for c in range(npool)])

        dgt = np.abs(np.diff(gt, axis=0)).mean(axis=1)
        evt = (np.diff(it_) != 0).astype(float)
        dgt = (dgt - dgt.mean()) / (dgt.std() + 1e-9)
        evt = (evt - evt.mean()) / (evt.std() + 1e-9)
        row = []
        for lag in LAGS:
            if lag < 0:
                row.append(float((dgt[-lag:] * evt[:lag]).mean()))
            elif lag > 0:
                row.append(float((dgt[:-lag] * evt[lag:]).mean()))
            else:
                row.append(float((dgt * evt).mean()))
        curves.append(row)
    etas_all = np.array(etas_all)
    curves = np.array(curves)

    visited, counts = np.unique(ids, return_counts=True)
    print(f"\n  poses visited: {len(visited)} of {len(keep)}  "
          f"(dwell: min {counts.min()} max {counts.max()} steps)")
    trans = int((np.diff(ids) != 0).sum())
    print(f"  pose transitions: {trans} in {len(ids)} steps "
          f"(~1 per {len(ids)/max(trans,1):.0f} steps)")

    print(f"\n  coupling of g to pose identity (eta^2 over {args.trials} trials):")
    for c in range(npool):
        print(f"    g[{c}]: {etas_all[:, c].mean():.3f} +/- {etas_all[:, c].std():.3f}")

    print(f"\n  cross-correlation |dg/dt| vs pose transitions "
          f"(mean +/- sd over {args.trials} trials):")
    mean_c, sd_c = curves.mean(axis=0), curves.std(axis=0)
    for i, lag in enumerate(LAGS):
        flag = "  <-- peak" if i == int(mean_c.argmax()) else ""
        print(f"    lag {lag:+4d}: r {mean_c[i]:+.3f} +/- {sd_c[i]:.3f}{flag}")
    bi = int(mean_c.argmax())
    best, bestlag = float(mean_c[bi]), LAGS[bi]
    # A peak smaller than its own spread across seeds is not a peak.
    if best < 2.0 * sd_c[bi]:
        print(f"    (peak {best:+.3f} is within 2 sd of noise -- not significant)")
        best = 0.0
    # The two statistics answer different questions and must be read together.
    # eta^2 asks whether g's LEVEL tracks body configuration; the cross-
    # correlation asks whether g's MOTION anticipates transition events. A high
    # eta^2 with a flat cross-correlation is not a contradiction — it is the
    # signature of a proprioceptive variable rather than a command variable.
    peak_eta = float(etas_all.mean(axis=0).max())
    timed = best >= 0.05
    print(f"\n  peak eta^2 {peak_eta:.3f} (level coupling);  peak r {best:+.3f} "
          f"at lag {bestlag:+d} (transition timing)")
    if peak_eta < 0.1 and not timed:
        print("verdict: g drifts independently of the body — pooling bought "
              "an internal variable that does nothing")
    elif peak_eta >= 0.1 and not timed:
        print("verdict: g's LEVEL tracks body configuration, but its motion does "
              "not anticipate transitions.\n  g is PROPRIOCEPTIVE — it encodes "
              "what the body is doing, and does not yet drive what it does next.")
    elif timed and bestlag > 0:
        print(f"verdict: g moves ~{bestlag} steps BEFORE pose changes — g is "
              "DRIVING the body. This is the nervous-system result.")
    elif timed and bestlag < 0:
        print(f"verdict: g moves ~{-bestlag} steps AFTER pose changes — g is a "
              "readout of transitions, not their cause")
    else:
        print("verdict: g and pose change move together with no clear lead")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for c in range(npool):
            ax[0].plot(g[:, c], lw=0.9, label=f"g[{c}]")
        ax[0].set_ylabel("global channel"); ax[0].legend(fontsize=7, ncol=4)
        ax[1].step(range(len(ids)), ids, lw=0.9, c="k")
        ax[1].set_ylabel("nearest pose"); ax[1].set_xlabel("step")
        fig.suptitle(os.path.basename(args.checkpoint))
        fig.tight_layout(); fig.savefig(args.plot, dpi=120)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
