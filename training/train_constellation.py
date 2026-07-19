"""Constellation states: every mood is a set of semistable poses the NCA glides
among — the creature is never just an image (Max, 2026-07-19, mid-epiphany).

Manifest: {"state_name": {"poses": [img...], "transit": "walk"|"cycle"}, ...}
ingested to npz by tools/ingest.py constellation mode. Training rule per
rollout: find the nearest pose in the sample's state; target a DIFFERENT pose
("cycle": the next in order; "walk": uniform random other). Mid-life state
switches train metamorphosis between constellations. cond=1 flag (2 states).

Usage: python3 train_constellation.py --target spirit_constellation.npz
"""

import argparse
import time

import numpy as np
import torch

from train_states import StateNCA, damage, export, make_seed
import train_states

POOL_SIZE = 1024
BATCH = 8
DAMAGE_N = 2
SWITCH_P = 0.12
DWELL_P = 0.10   # low, and proximity-gated below: polish on arrival, never park


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--out", default="../weights/constellation.nca")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--horizon", type=int, nargs=2, default=[24, 48],
                    help="rollout steps per transit; scale DOWN as pose density goes UP")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--pool", type=int, default=POOL_SIZE)
    ap.add_argument("--hidden", type=int, default=None,
                    help="update-rule width (Swift parses it from the header; 128 default)")
    ap.add_argument("--growth-p", type=float, default=0.2,
                    help="fraction of rollouts using a 4x horizon (seed growth, persistence)")
    ap.add_argument("--dwell", type=float, default=None,
                    help="override DWELL_P (dense rings want 0: arrival is free)")
    args = ap.parse_args()

    global DWELL_P, BATCH, POOL_SIZE
    BATCH, POOL_SIZE = args.batch, args.pool
    if args.hidden:
        train_states.HIDDEN = args.hidden
    if args.dwell is not None:
        DWELL_P = args.dwell
    data = np.load(args.target, allow_pickle=True)
    assert str(data["kind"]) == "2d_constellation"
    poses = data["poses"].astype(np.float32)        # (P, G, G, 4) all poses, flat
    pose_state = data["pose_state"].astype(np.int64)  # (P,) owning state per pose
    transits = list(data["transits"])                # per-state "walk" | "cycle"
    edges = data["edges"] if "edges" in data.files else np.zeros((0, 2), int)
    successors = {}                                   # directed graph, if provided
    for a, b in edges:
        successors.setdefault(int(a), []).append(int(b))
    n_states = int(pose_state.max()) + 1
    assert n_states == 2, "cond=1 flag: 2 states (manifold trainer for more)"
    train_states.GRID = poses.shape[1]
    print(f"{poses.shape[0]} poses across {n_states} states, transits {transits}", flush=True)

    device = torch.device(args.device)
    torch.manual_seed(0); np.random.seed(0)
    poses_t = torch.from_numpy(poses).permute(0, 3, 1, 2).to(device)  # (P,4,G,G)
    state_pose_idx = [np.where(pose_state == s)[0] for s in range(n_states)]

    model = StateNCA().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(args.iters * 0.45), int(args.iters * 0.8)], gamma=0.25)

    pool = make_seed(POOL_SIZE, device)
    pool_state = torch.randint(0, n_states, (POOL_SIZE,), device=device)
    t0 = time.time()

    for it in range(1, args.iters + 1):
        idx = torch.randint(0, POOL_SIZE, (BATCH,))
        x = pool[idx].clone()
        st = pool_state[idx].clone()

        with torch.no_grad():
            # nearest pose within each sample's own state
            tgt = torch.empty(BATCH, 4, train_states.GRID, train_states.GRID, device=device)
            for b in range(BATCH):
                own = state_pose_idx[int(st[b])]
                d = ((x[b:b+1, :4] - poses_t[own]) ** 2).mean(dim=(1, 2, 3))
                near = int(d.argmin())
                near_global = int(own[near])
                arrived = float(d.min()) < 0.004        # already essentially at the pose
                if arrived and np.random.rand() < DWELL_P:
                    nxt = near_global                    # brief polish, then depart
                elif near_global in successors:          # directed graph wins:
                    outs = successors[near_global]       # waypoints, hysteresis loops
                    nxt = outs[np.random.randint(len(outs))]
                elif transits[int(st[b])] == "cycle" and len(own) > 1:
                    nxt = own[(near + 1) % len(own)]
                elif len(own) > 1:  # walk: any other star in the constellation
                    others = [p for k, p in enumerate(own) if k != near]
                    nxt = others[np.random.randint(len(others))]
                else:
                    nxt = own[0]
                tgt[b] = poses_t[nxt]
            x[0] = make_seed(1, device)[0]
            st[0] = torch.randint(0, n_states, (1,), device=device)
            switch = torch.rand(BATCH, device=device) < SWITCH_P
            switch[0] = False
            st[switch] = 1 - st[switch]
            # switched samples aim at their new state's nearest star instead
            for b in torch.nonzero(switch).flatten().tolist():
                own = state_pose_idx[int(st[b])]
                d = ((x[b:b+1, :4] - poses_t[own]) ** 2).mean(dim=(1, 2, 3))
                tgt[b] = poses_t[own[int(d.argmin())]]
            if it > 500:
                x[-DAMAGE_N:] = damage(x[-DAMAGE_N:])

        steps = int(np.random.randint(args.horizon[0], args.horizon[1] + 1))
        if np.random.rand() < args.growth_p:
            steps *= 4          # seeds need long rollouts to grow at all
        for _ in range(steps):
            x = model(x, st)
        loss = ((x[:, :4] - tgt) ** 2).mean()

        if not torch.isfinite(loss):
            opt.zero_grad(); continue
        opt.zero_grad(); loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        opt.step(); sched.step()

        with torch.no_grad():
            finite = torch.isfinite(x).flatten(1).all(dim=1)
            slots = idx[finite.cpu()]
            pool[slots] = x.detach()[finite]
            pool_state[slots] = st[finite]

        if it % 50 == 0:
            print(f"iter {it:5d}  loss {loss.item():.5f}  {it/(time.time()-t0):.2f} it/s", flush=True)
        if it % 1000 == 0:
            export(model, f"{args.out}.it{it}")
        if it % 250 == 0 or it == args.iters:
            export(model, args.out)
            print(f"  checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
