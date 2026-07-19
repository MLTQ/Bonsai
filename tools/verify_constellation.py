"""Verify a constellation/heteroclinic creature: does it actually glide?

Renders a free-run through the Swift runtime, classifies each frame to its
nearest pose, and reports the diagnostics from guide §12:

  tape            nearest-pose sequence over the run
  traversal       forward / reverse / held step counts around the pose graph
  period          mean frames per full lap (if it laps)
  sharpness       render edge-energy vs the target poses
  signal/floor    adjacent-pose distance vs render error — must exceed ~3x

Usage:
  python3 tools/verify_constellation.py weights/x.nca /tmp/x.npz [--state 0]
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYMBOLS = "0123456789abcdefghij"


def render(weights, state, frames, stride, binary):
    out = tempfile.mkdtemp(prefix="verify_")
    env = dict(os.environ, BONSAI_STATE=str(state))
    subprocess.run([binary, "--render-seq", out, str(frames), str(stride), weights],
                   check=True, capture_output=True, env=env)
    paths = sorted(glob.glob(os.path.join(out, "frame_*.png")))
    return [np.asarray(Image.open(p).convert("RGBA"), np.float32) / 255.0 for p in paths]


def edge_energy(a):
    gy, gx = np.gradient(a[..., 3])
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("target", help="the 2d_constellation npz it was trained on")
    ap.add_argument("--state", type=int, default=0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--binary", default=os.path.join(REPO, ".build/debug/Bonsai"))
    args = ap.parse_args()

    d = np.load(args.target, allow_pickle=True)
    poses = d["poses"].astype(np.float32)
    own = np.where(d["pose_state"].astype(int) == args.state)[0]
    poses = poses[own]
    n = len(poses)
    grid = poses.shape[1]

    frames = render(args.weights, args.state, args.frames, args.stride, args.binary)
    frames = [np.asarray(Image.fromarray((f * 255).astype(np.uint8)).resize((grid, grid)),
                         np.float32) / 255.0 for f in frames]

    dists = np.array([[((f - poses[k]) ** 2).mean() for k in range(n)] for f in frames])
    idx = dists.argmin(axis=1)
    tape = "".join(SYMBOLS[i] for i in idx)

    steps = [(int(idx[i + 1]) - int(idx[i])) % n for i in range(len(idx) - 1)]
    fwd = sum(1 for s in steps if 1 <= s <= max(1, n // 4))
    rev = sum(1 for s in steps if n - max(1, n // 4) <= s <= n - 1)
    held = sum(1 for s in steps if s == 0)

    render_err = float(dists.min(axis=1).mean())
    adj = float(np.mean([((poses[k] - poses[(k + 1) % n]) ** 2).mean() for k in range(n)]))
    motion = float(np.mean([np.abs(frames[i + 1] - frames[i]).mean()
                            for i in range(len(frames) - 1)])) * 255

    print(f"tape        {tape}")
    print(f"traversal   forward {fwd}  reverse {rev}  held {held}"
          f"   (net {'forward' if fwd > rev else 'reverse' if rev > fwd else 'none'})")
    if fwd + rev:
        laps = (fwd - rev) / n
        if abs(laps) > 0.5:
            print(f"period      ~{len(frames) / abs(laps):.0f} frames/lap"
                  f" = ~{len(frames) * args.stride / abs(laps):.0f} automaton steps")
    print(f"sharpness   render {np.mean([edge_energy(f) for f in frames]):.4f}"
          f"   targets {np.mean([edge_energy(p) for p in poses]):.4f}")
    print(f"motion      {motion:.2f} mean abs delta per frame (0-255)")
    print(f"signal/floor  adjacent {adj:.5f} / render err {render_err:.5f}"
          f" = {adj / max(render_err, 1e-9):.1f}x"
          f"   {'OK' if adj / max(render_err, 1e-9) > 3 else 'TOO LOW (guide §12)'}")


if __name__ == "__main__":
    main()
