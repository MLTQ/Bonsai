"""Export a deterministic mature NCS1 seed for an NCA5 clockless creature."""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from hidden_momentum_nca import (POSITION_CH, STATE_CH, VISIBLE_CH,
                                 HiddenMomentumNCA, lift_state, load_nca5)
from train_autonomous import load_conditioned
from train_cyclic import (HIDDEN, OMEGA, CyclicNCA, _load_creature, cond_for,
                          make_seed)


ROOT = Path(__file__).resolve().parent.parent


def make_mature_state(teacher, steps, phase, device):
    """Grow the conditioned teacher in walk mode and retain hidden velocity."""
    x = make_seed(1, device)
    theta = torch.tensor([phase], dtype=torch.float32, device=device)
    behavior = torch.ones(1, dtype=torch.long, device=device)
    previous = x
    with torch.no_grad():
        for _ in range(steps):
            previous = x
            x = teacher(x, cond_for(theta, behavior))
            theta = theta + OMEGA
    return lift_state(x, (x - previous)[:, VISIBLE_CH:POSITION_CH])


def write_ncs1(state, path):
    """Write one NCHW state tensor as cell-major little-endian NCS1."""
    if state.shape[0] != 1 or state.shape[1] != STATE_CH:
        raise ValueError(f"expected one {STATE_CH}-channel state, got {tuple(state.shape)}")
    height, width = state.shape[2:]
    cell_major = state[0].permute(1, 2, 0).contiguous().cpu().numpy().astype("<f4")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<4s3i", b"NCS1", width, height, STATE_CH))
        cell_major.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=str(ROOT / "weights" / "shoggoth.nca"))
    ap.add_argument("--student", default=str(
        ROOT / "weights" / "shoggoth_auto_hidden_momentum.nca"))
    ap.add_argument("--out", default=str(
        ROOT / "weights" / "shoggoth_auto_hidden_momentum.ncs"))
    ap.add_argument("--grow-steps", type=int, default=480)
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--student-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=(
        "mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    _load_creature("shoggoth")

    teacher = CyclicNCA().to(device).eval()
    load_conditioned(args.teacher, teacher)
    student = HiddenMomentumNCA(cond=1, hidden=HIDDEN).to(device).eval()
    load_nca5(student, args.student)

    state = make_mature_state(teacher, args.grow_steps, args.phase, device)
    behavior_cond = torch.ones(1, 1, device=device)
    with torch.no_grad():
        for _ in range(args.student_steps):
            state = student(state, behavior_cond)
    if not torch.isfinite(state).all():
        raise RuntimeError("mature state contains non-finite values")
    write_ncs1(state, args.out)

    alive = int((state[:, 3:4] > 0.1).sum().item())
    velocity_rms = state[:, POSITION_CH:].square().mean().sqrt().item()
    print(f"wrote {args.out}: alive={alive}, hidden velocity rms={velocity_rms:.6f}")


if __name__ == "__main__":
    main()
