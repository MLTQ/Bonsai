"""Multi-state attractor NCA: dramatically different forms, learned metamorphosis.

Each state is a static target selected by a conditioning flag; mid-life state
switches during training teach the automaton to TRANSFORM its existing body
(the werewolf pattern: calm form <-> beast form). All motion lives in the NCA:
shimmer within states, metamorphosis between them.

Exports NCA2 with cond = 1 (state flag; 2 states). Targets from tools/ingest.py
states mode. Usage: python3 train_states.py --target spirit_states.npz
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CH, HIDDEN, COND = 16, 128, 1
FIRE_RATE = 0.5
POOL_SIZE = 1024
BATCH = 8
DAMAGE_N = 2
SWITCH_P = 0.15
GRID = 64


class StateNCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Conv2d(CH * 3 + COND, HIDDEN, 1)
        self.w2 = nn.Conv2d(HIDDEN, CH, 1)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sy = sx.T.contiguous()
        self.register_buffer("percept_w",
                             torch.stack([ident, sx, sy]).repeat(CH, 1, 1).unsqueeze(1))

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def forward(self, x, state):
        pre = self.alive(x)
        p = F.conv2d(x, self.percept_w, padding=1, groups=CH)
        smap = state.float()[:, None, None, None].expand(-1, 1, *x.shape[2:])
        dx = self.w2(F.relu(self.w1(torch.cat([p, smap], dim=1))))
        fire = (torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) <= FIRE_RATE).float()
        x = x + dx * fire
        return (x * (pre & self.alive(x)).float()).clamp(-8.0, 8.0)


def make_seed(n, device):
    x = torch.zeros(n, CH, GRID, GRID, device=device)
    x[:, 3:, GRID // 2, GRID // 2] = 1.0
    return x


def damage(x):
    n, _, h, w = x.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device),
                            torch.arange(w, device=x.device), indexing="ij")
    for i in range(n):
        r = np.random.uniform(6, 14)
        cx, cy = np.random.uniform(w * .25, w * .75), np.random.uniform(h * .25, h * .75)
        x[i] *= (((xx - cx) ** 2 + (yy - cy) ** 2) > r ** 2).float()
    return x


def export(model, path):
    with open(path, "wb") as f:
        f.write(b"NCA2")
        np.array([CH, HIDDEN, COND], dtype="<i4").tofile(f)
        np.array([FIRE_RATE], dtype="<f4").tofile(f)
        model.w1.weight.detach().cpu().numpy().reshape(HIDDEN, CH * 3 + COND).astype("<f4").tofile(f)
        model.w1.bias.detach().cpu().numpy().astype("<f4").tofile(f)
        model.w2.weight.detach().cpu().numpy().reshape(CH, HIDDEN).astype("<f4").tofile(f)
        model.w2.bias.detach().cpu().numpy().astype("<f4").tofile(f)


def main():
    global GRID
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--iters", type=int, default=9000)
    ap.add_argument("--out", default="../weights/states_creature.nca")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()

    data = np.load(args.target, allow_pickle=True)
    assert str(data["kind"]) == "2d_states"
    targets_np = data["targets"].astype(np.float32)
    n_states = targets_np.shape[0]
    assert n_states == 2, "cond=1 flag handles 2 states; more states -> manifold trainer"
    GRID = targets_np.shape[1]
    print(f"states: {list(data['state_names'])}, grid {GRID}", flush=True)

    device = torch.device(args.device)
    torch.manual_seed(0); np.random.seed(0)
    targets = torch.from_numpy(targets_np).permute(0, 3, 1, 2).to(device)  # (S,4,H,W)

    model = StateNCA().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[3000], gamma=0.1)

    pool = make_seed(POOL_SIZE, device)
    pool_state = torch.randint(0, n_states, (POOL_SIZE,), device=device)
    t0 = time.time()

    for it in range(1, args.iters + 1):
        idx = torch.randint(0, POOL_SIZE, (BATCH,))
        x = pool[idx].clone()
        st = pool_state[idx].clone()

        with torch.no_grad():
            losses = ((x[:, :4] - targets[st]) ** 2).mean(dim=(1, 2, 3))
            order = losses.argsort(descending=True)
            order_cpu = order.cpu()
            x, st = x[order], st[order]
            x[0] = make_seed(1, device)[0]
            st[0] = torch.randint(0, n_states, (1,), device=device)
            switch = torch.rand(BATCH, device=device) < SWITCH_P
            switch[0] = False
            st[switch] = 1 - st[switch]          # metamorphosis training
            if it > 500:
                x[-DAMAGE_N:] = damage(x[-DAMAGE_N:])

        for _ in range(int(np.random.randint(64, 97))):
            x = model(x, st)
        loss = ((x[:, :4] - targets[st]) ** 2).mean()

        if not torch.isfinite(loss):
            print(f"iter {it}: non-finite, discarded", flush=True)
            opt.zero_grad()
            continue
        opt.zero_grad(); loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        opt.step(); sched.step()

        with torch.no_grad():
            finite = torch.isfinite(x).flatten(1).all(dim=1)
            slots = idx[order_cpu][finite.cpu()]
            pool[slots] = x.detach()[finite]
            pool_state[slots] = st[finite]

        if it % 50 == 0:
            print(f"iter {it:5d}  loss {loss.item():.5f}  {it/(time.time()-t0):.2f} it/s", flush=True)
        if it % 250 == 0 or it == args.iters:
            export(model, args.out)
            print(f"  checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
