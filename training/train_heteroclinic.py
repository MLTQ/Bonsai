"""Heteroclinic cycling: Max's unstable-attractor gait. No clock anywhere.

Two pose targets (left-stride, right-stride) under ONE conditioning flag.
Training rule: from wherever the state is, the loss target is the pole
OPPOSITE the nearest one. Each pose is therefore unstable in the direction
of the other; walking emerges as perpetual transit between saddles, with
the period set by the learned dynamics rather than an external phase.

Usage: python3 train_heteroclinic.py [--iters 10000]
"""

import argparse
import time

import numpy as np
import torch

from train_cyclic import (BATCH, CH, DAMAGE_N, HIDDEN, POOL_SIZE,
                          _load_creature, damage)
from train_states import StateNCA, export, make_seed  # cond=1 model + NCA2 export


def main():
    global GRID
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--out", default="../weights/heteroclinic.nca")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    _load_creature("shoggoth")
    import train_cyclic
    frames = train_cyclic.make_frames()          # (2, 12, G, G, 4)
    import train_states
    train_states.GRID = frames.shape[2]

    device = torch.device(args.device)
    torch.manual_seed(0); np.random.seed(0)
    # the two saddle poses: opposite strides of the walk cycle
    poles = torch.from_numpy(np.stack([frames[1, 0], frames[1, 6]])) \
        .permute(0, 3, 1, 2).to(device)          # (2, 4, G, G)

    model = StateNCA().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[4000], gamma=0.1)

    pool = train_states.make_seed(POOL_SIZE, device)
    flag = torch.ones(BATCH, dtype=torch.long, device=device)  # "walking", always
    t0 = time.time()

    for it in range(1, args.iters + 1):
        idx = torch.randint(0, POOL_SIZE, (BATCH,))
        x = pool[idx].clone()

        with torch.no_grad():
            # nearest pole per sample -> target is the OTHER pole
            d0 = ((x[:, :4] - poles[0]) ** 2).mean(dim=(1, 2, 3))
            d1 = ((x[:, :4] - poles[1]) ** 2).mean(dim=(1, 2, 3))
            nearest = (d1 < d0).long()           # 1 if nearer pole B
            target = poles[1 - nearest]          # aim at the opposite saddle
            # worst-vs-own-pole sample restarts from seed (seed aims at pole A)
            worst = torch.maximum(d0, d1).argmax()
            x[worst] = train_states.make_seed(1, device)[0]
            target[worst] = poles[0]
            if it > 500:
                x[-DAMAGE_N:] = damage(x[-DAMAGE_N:])

        # transit horizon: roughly half a gait period of the clocked ancestor
        for _ in range(int(np.random.randint(36, 61))):
            x = model(x, flag)
        loss = ((x[:, :4] - target) ** 2).mean()

        if not torch.isfinite(loss):
            opt.zero_grad(); continue
        opt.zero_grad(); loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        opt.step(); sched.step()

        with torch.no_grad():
            finite = torch.isfinite(x).flatten(1).all(dim=1)
            pool[idx[finite.cpu()]] = x.detach()[finite]

        if it % 50 == 0:
            print(f"iter {it:5d}  loss {loss.item():.5f}  {it/(time.time()-t0):.2f} it/s", flush=True)
        if it % 250 == 0 or it == args.iters:
            export(model, args.out)
            print(f"  checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
