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


def warm_start(model, path):
    """Load weights from an exported NCA2 checkpoint so a finished run can carry on.

    Only the update rule is saved by export(), so a resumed run re-fills its sample
    pool from seed and restarts the LR schedule — pass --lr to pick up near where
    the previous schedule left off rather than jumping back to 2e-3.
    """
    with open(path, "rb") as f:
        assert f.read(4) == b"NCA2", f"{path} is not an NCA2 checkpoint"
        ch, hidden, cond = (int(v) for v in np.fromfile(f, dtype="<i4", count=3))
        np.fromfile(f, dtype="<f4", count=1)   # fire rate (fixed at train time)
        want = (train_states.CH, train_states.HIDDEN, train_states.COND)
        assert (ch, hidden, cond) == want, \
            f"checkpoint is ch/hidden/cond {(ch, hidden, cond)}, this run wants {want} " \
            f"(--hidden must match the run you are resuming)"
        pin = ch * 3 + cond
        w1 = np.fromfile(f, dtype="<f4", count=hidden * pin).reshape(hidden, pin)
        b1 = np.fromfile(f, dtype="<f4", count=hidden)
        w2 = np.fromfile(f, dtype="<f4", count=ch * hidden).reshape(ch, hidden)
        b2 = np.fromfile(f, dtype="<f4", count=ch)
    dev = model.w1.weight.device
    model.w1.weight.data.copy_(torch.from_numpy(w1).to(dev)[..., None, None])
    model.w1.bias.data.copy_(torch.from_numpy(b1).to(dev))
    model.w2.weight.data.copy_(torch.from_numpy(w2).to(dev)[..., None, None])
    model.w2.bias.data.copy_(torch.from_numpy(b2).to(dev))
    print(f"warm start <- {path}", flush=True)


def main():
    global DWELL_P, BATCH, POOL_SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--out", default="../weights/constellation.nca")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--horizon", type=int, nargs=2, default=[24, 48],
                    help="rollout steps per transit; scale DOWN as pose density goes UP")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--pool", type=int, default=POOL_SIZE)
    ap.add_argument("--hidden", type=int, default=None,
                    help="update-rule width (Swift parses it from the header; 128 default)")
    ap.add_argument("--fused", action="store_true",
                    help="Triton fused step (CUDA only; see fused_step.py). "
                         "Big win here is memory: enables much larger batches.")
    ap.add_argument("--motion-weight", type=float, default=0.0,
                    help="alpha: loss weight becomes 1 + alpha*normalized per-pixel "
                         "variance across a state's poses. Lets tiny motions train.")
    ap.add_argument("--waypoints", type=int, default=4,
                    help="targets advanced along the graph within one rollout "
                         "(1 = old single-hop behavior)")
    ap.add_argument("--growth-p", type=float, default=0.2,
                    help="fraction of rollouts using a 4x horizon (seed growth, persistence)")
    ap.add_argument("--dwell", type=float, default=None,
                    help="override DWELL_P (dense rings want 0: arrival is free)")
    ap.add_argument("--resume", default=None,
                    help="warm start from an exported .nca (weights only; pool restarts)")
    ap.add_argument("--pooled", type=int, default=0,
                    help="N globally-broadcast feedback channels (see pooled_nca.py). "
                         "0 = strict locality. Incompatible with --fused.")
    ap.add_argument("--lr", type=float, default=2e-3,
                    help="initial LR; drop it when resuming a run that already annealed")
    args = ap.parse_args()

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
    for e in edges:
        a, b = int(e[0]), int(e[1])
        weight = int(e[2]) if len(e) > 2 else 1      # repeat = probability weight
        successors.setdefault(a, []).extend([b] * max(1, weight))
    n_states = int(pose_state.max()) + 1
    assert n_states == 2, "cond=1 flag: 2 states (manifold trainer for more)"
    train_states.GRID = poses.shape[1]
    print(f"{poses.shape[0]} poses across {n_states} states, transits {transits}, "
          f"device {args.device}, batch {BATCH}, pool {POOL_SIZE}", flush=True)

    device = torch.device(args.device)
    torch.manual_seed(0); np.random.seed(0)
    poses_t = torch.from_numpy(poses).permute(0, 3, 1, 2).to(device)  # (P,4,G,G)
    pose_state_t = torch.from_numpy(pose_state).to(device)
    state_pose_idx = [np.where(pose_state == s)[0] for s in range(n_states)]

    # per-state motion mask: where does this state's pose set actually vary?
    motion_w = torch.ones(1, 1, train_states.GRID, train_states.GRID, device=device)
    if args.motion_weight > 0:
        v = np.zeros((train_states.GRID, train_states.GRID), np.float32)
        for s in range(n_states):
            ps = poses[state_pose_idx[s]]
            if len(ps) > 1:
                v = np.maximum(v, ps.var(axis=0).mean(axis=-1))
        v = v / (v.max() + 1e-9)
        motion_w = (1.0 + args.motion_weight *
                    torch.from_numpy(v)[None, None].to(device))
        print(f"motion mask: {float((v > 0.1).mean()) * 100:.0f}% of pixels move", flush=True)

    if args.pooled:
        assert not args.fused, "the fused kernel assumes strict-local perception"
        from pooled_nca import PooledNCA, export_pooled
        model = PooledNCA(npool=args.pooled).to(device)
        save = export_pooled
        print(f"pooled: {args.pooled} global feedback channels", flush=True)
    else:
        model = StateNCA().to(device)
        save = export
    if args.resume:
        warm_start(model, args.resume)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
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
            # Seed reset + state switches first, so target selection sees final states.
            x[0] = make_seed(1, device)[0]
            st[0] = torch.randint(0, n_states, (1,), device=device)
            switch = torch.rand(BATCH, device=device) < SWITCH_P
            switch[0] = False
            st[switch] = 1 - st[switch]

            # All-pairs distances in ONE kernel instead of a per-sample Python loop
            # (that loop forced ~70 host syncs per iteration and left the GPU idle).
            d_all = ((x[:, None, :4] - poses_t[None]) ** 2).mean(dim=(2, 3, 4))   # (B,P)
            wrong_state = pose_state_t[None, :] != st[:, None]
            d_masked = d_all.masked_fill(wrong_state, float("inf"))
            near_idx = d_masked.argmin(dim=1)
            near_cpu = near_idx.cpu().numpy()
            near_d = d_masked.gather(1, near_idx[:, None]).squeeze(1).cpu().numpy()
            st_cpu = st.cpu().numpy()
            switch_cpu = switch.cpu().numpy()

            def step_from(pose_idx, state):
                """One hop along the graph (or the state's transit rule)."""
                if pose_idx in successors:
                    outs = successors[pose_idx]
                    return int(outs[np.random.randint(len(outs))])
                own = state_pose_idx[state]
                if len(own) == 1:
                    return int(own[0])
                k = int(np.where(own == pose_idx)[0][0])
                if transits[state] == "cycle":
                    return int(own[(k + 1) % len(own)])
                others = [p for j, p in enumerate(own) if j != k]
                return int(others[np.random.randint(len(others))])

            # Waypoint CHAIN: the target advances along the graph *during* the
            # rollout, so the whole trajectory is supervised (not just its end).
            K = max(1, args.waypoints)
            chain = np.empty((BATCH, K), dtype=np.int64)
            for b in range(BATCH):
                cur = int(near_cpu[b])
                s = int(st_cpu[b])
                if switch_cpu[b]:
                    chain[b, 0] = cur              # metamorphosis: become the new form
                    start = 1
                else:
                    start = 0
                for j in range(start, K):
                    cur = step_from(cur, s)
                    chain[b, j] = cur
            chain_t = torch.from_numpy(chain).to(device)

            if it > 500:
                x[-DAMAGE_N:] = damage(x[-DAMAGE_N:])

        K = chain_t.shape[1]
        seg_scale = 4 if np.random.rand() < args.growth_p else 1   # occasional long haul
        loss = torch.zeros((), device=device)
        if args.fused:
            from fused_step import fused_nca_rollout
            H, C, CN = train_states.HIDDEN, train_states.CH, train_states.COND
            w1f = model.w1.weight.reshape(H, C * 3 + CN)
            w2f = model.w2.weight.reshape(C, H)
            cond_flag = st.float()[:, None]
        gstep = 0
        for j in range(K):
            seg = int(np.random.randint(args.horizon[0], args.horizon[1] + 1)) * seg_scale
            if args.fused:
                x = fused_nca_rollout(
                    x, w1f, model.w1.bias, w2f, model.w2.bias, seg,
                    cond=cond_flag, seed=it, step_offset=gstep,
                    fire_rate=train_states.FIRE_RATE, clamp=8.0)
            else:
                for _ in range(seg):
                    x = model(x, st)
            gstep += seg
            w = 0.6 + 0.4 * (j + 1) / K            # later waypoints weigh a little more
            err = (x[:, :4] - poses_t[chain_t[:, j]]) ** 2
            loss = loss + w * (err * motion_w).mean()
        loss = loss / K

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
            save(model, f"{args.out}.it{it}")
        if it % 250 == 0 or it == args.iters:
            save(model, args.out)
            print(f"  checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
