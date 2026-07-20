import re, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = {
    "A · h256 (64px)":      "experiments/sweep_h100/A_base.log",
    "B · h256 (96px)":      "experiments/sweep_h100/B_res96.log",
    "C · h256 6wp long":    "experiments/sweep_h100/C_dense.log",
    "D · h128 (64px)":      "experiments/sweep_h100/D_h128.log",
    "E · procedural art":   "experiments/sweep_h100/E_procedural.log",
}
COLORS = ["#e07a5f", "#3d9970", "#6a8caf", "#d4a017", "#9b5de5"]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="#16161a")
for a in (ax, ax2):
    a.set_facecolor("#16161a")
    for s in a.spines.values(): s.set_color("#555")
    a.tick_params(colors="#bbb"); a.grid(alpha=0.15, color="#888")

def med(y, k=15):
    return np.array([np.median(y[max(0, i-k):i+1]) for i in range(len(y))])

for (label, path), c in zip(LOGS.items(), COLORS):
    txt = open(path).read()
    pts = re.findall(r"iter\s+(\d+)\s+loss\s+([\d.]+)", txt)
    if not pts: continue
    it = np.array([int(a) for a, _ in pts]); ls = np.array([float(b) for _, b in pts])
    ax.plot(it, ls, color=c, alpha=0.16, lw=0.8)
    ax.plot(it, med(ls), color=c, lw=2.0, label=f"{label}  (last {med(ls)[-1]:.4f})")
    # right panel: fractional improvement per 5k, to show whether it is still learning
    step = 5000
    xs, rate = [], []
    for s in range(step, it.max() + 1, step):
        prev = med(ls)[(it > s - step) & (it <= s - step // 2)]
        cur = med(ls)[(it > s - step // 2) & (it <= s)]
        if len(prev) and len(cur):
            xs.append(s); rate.append(100 * (prev.mean() - cur.mean()) / prev.mean())
    ax2.plot(xs, rate, color=c, lw=2.0, marker="o", ms=4, label=label)

ax.set_xlabel("iteration", color="#ccc"); ax.set_ylabel("loss (motion-weighted)", color="#ccc")
ax.set_yscale("log"); ax.set_title("Sweep loss curves (rolling median)", color="#eee")
ax.legend(facecolor="#20202a", edgecolor="#444", labelcolor="#ddd", fontsize=8)
ax2.axhline(0, color="#888", lw=1, ls="--")
ax2.set_xlabel("iteration", color="#ccc"); ax2.set_ylabel("% loss reduction per 2.5k window", color="#ccc")
ax2.set_title("Still learning? (improvement rate)", color="#eee")
ax2.legend(facecolor="#20202a", edgecolor="#444", labelcolor="#ddd", fontsize=8)
plt.tight_layout()
out = os.environ.get("OUT", "/tmp/sweep_curves.png")
plt.savefig(out, dpi=130, facecolor="#16161a")
print("wrote", out)
