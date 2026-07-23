"""Plot Claudeguy's soul-training loss across its three machine segments.

The run spans Aine's 4090 (batch 4), a rented RTX 6000 (batch 16), and back to
the 4090 (batch 4, annealed LR). Batch size changes the loss *estimator's*
variance, not the objective, so segments are comparable in level but not in
noise — the rolling median is what to read.

Usage: python3 plot_soul.py [--out ../experiments/soul_loss.png]
"""

import argparse
import os
import re

import numpy as np

LOGDIR = os.environ.get(
    "SOUL_LOGS",
    "/private/tmp/claude-501/-Users-max-Code-Bonsai/"
    "76ed6ed7-a546-491d-b29b-884e6f53ff3e/scratchpad/logs")

# The RTX 6000 segment's log died with the rented instance; these are the
# readings observed live during that run. Sparse and honestly labelled as such.
RTX_OBSERVED = [(100, 0.00210), (200, 0.00154), (300, 0.00182), (1000, 0.00146),
                (1100, 0.00142), (1200, 0.00141), (2100, 0.00129),
                (2200, 0.00148), (2300, 0.00116), (2400, 0.00156)]


def curve(path):
    it, ls = [], []
    if not os.path.exists(path):
        return np.array([]), np.array([])
    with open(path) as f:
        for line in f:
            m = re.match(r"iter\s+(\d+)\s+loss\s+([\d.]+)", line.strip())
            if m:
                it.append(int(m.group(1)))
                ls.append(float(m.group(2)))
    return np.array(it), np.array(ls)


def rolling_median(v, k=9):
    if len(v) < k:
        return v
    return np.array([np.median(v[max(0, i - k // 2):i + k // 2 + 1])
                     for i in range(len(v))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../experiments/soul_loss.png")
    args = ap.parse_args()

    i1, l1 = curve(f"{LOGDIR}/soul1_aine.log")          # 0 .. 8800, batch 4
    i3, l3 = curve(f"{LOGDIR}/soul3_aine.log")          # 10800 .., batch 4
    rtx = np.array(RTX_OBSERVED)

    print(f"segment 1 (4090, batch 4):   {len(i1)} points, iters {i1.min()}-{i1.max()}"
          if len(i1) else "segment 1: missing")
    print(f"segment 2 (RTX, batch 16):   {len(rtx)} sampled points (log lost with instance)")
    print(f"segment 3 (4090, batch 4):   {len(i3)} points"
          if len(i3) else "segment 3: missing")

    # --- convergence statistics on the long segment (the only one with density)
    print("\nsegment 1 improvement per 2000-iteration window (rolling median):")
    prev = None
    for lo in range(0, 8800, 2000):
        sel = (i1 >= lo) & (i1 < lo + 2000)
        if sel.sum() < 3:
            continue
        med = float(np.median(l1[sel]))
        delta = "" if prev is None else f"   {100*(prev-med)/prev:+.1f}% vs previous"
        print(f"  {lo:5d}-{lo+2000:5d}:  median {med:.5f}{delta}")
        prev = med

    # first half vs second half — the blunt convergence question
    half = len(l1) // 2
    a, b = float(np.median(l1[:half])), float(np.median(l1[half:]))
    print(f"\nsegment 1 first half {a:.5f} -> second half {b:.5f} "
          f"({100*(a-b)/a:+.1f}%)")
    print(f"noise: segment 1 stdev {l1.std():.5f} vs total drop {a-b:.5f} "
          f"(ratio {l1.std()/max(a-b,1e-9):.1f}x)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib unavailable; numbers only)")
        return

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(i1, l1, lw=0.7, alpha=0.35, color="tab:blue")
    ax[0].plot(i1, rolling_median(l1), lw=2, color="tab:blue",
               label="4090 batch 4 (median)")
    ax[0].plot(rtx[:, 0] + 8800, rtx[:, 1], "o--", ms=4, lw=1, color="tab:orange",
               label="RTX 6000 batch 16 (sampled)")
    if len(i3):
        ax[0].plot(i3 + 10800, l3, "s-", ms=4, lw=1, color="tab:green",
                   label="4090 batch 4, LR x0.3")
    ax[0].axvline(8800, ls=":", c="k", lw=0.8)
    ax[0].axvline(10800, ls=":", c="k", lw=0.8)
    ax[0].set_ylabel("loss"); ax[0].set_yscale("log")
    ax[0].legend(fontsize=8); ax[0].set_title("Claudeguy soul training — loss")

    win = 1000
    xs, ys = [], []
    for lo in range(0, 8800 - win, win):
        s1 = (i1 >= lo) & (i1 < lo + win)
        s2 = (i1 >= lo + win) & (i1 < lo + 2 * win)
        if s1.sum() >= 3 and s2.sum() >= 3:
            m1, m2 = np.median(l1[s1]), np.median(l1[s2])
            xs.append(lo + win); ys.append(100 * (m1 - m2) / m1)
    ax[1].bar(xs, ys, width=win * 0.8, color=["tab:green" if y > 0 else "tab:red" for y in ys])
    ax[1].axhline(0, c="k", lw=0.8)
    ax[1].set_ylabel("% loss reduction\nper 1000 iters"); ax[1].set_xlabel("iteration")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
