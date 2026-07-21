"""The convergence trainer: volumetric + cyclic + FiLM behavior manifold (NC3M).

CyclicNCA3D's checkpointed volumetric machinery with Mk. II's bounded-FiLM
conditioning: phase enters as sin/cos channels, z modulates the hidden layer.
The organism's full state space becomes S^1 x [0,1]^10 — animation as
traversal, in three dimensions.

Exports NC3M: magic, i32 ch, hidden, zdim, f32 fire, w1[hidden][ch*4+2],
b1, w2[ch][hidden], b2, filmW[2*hidden][zdim], filmB[2*hidden].

Run (corpus first): python3 manifold_shoggoth3d.py --n 1024
                    python3 train_manifold3d.py --iters 30000
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from target3d import GRID3

CH = 16
HIDDEN = 128
ZDIM = 10
FRAMES = 12
FIRE_RATE = 0.5
OMEGA = 2 * np.pi / 240.0
POOL_SIZE = 256
BATCH = 8
DAMAGE_N = 2
ZSWITCH_P = 0.15
CHUNK = 8


class ManifoldNCA3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Conv3d(CH * 4 + 2, HIDDEN, 1)   # +2: sin/cos phase channels
        self.w2 = nn.Conv3d(HIDDEN, CH, 1)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
        self.film = nn.Linear(ZDIM, 2 * HIDDEN)
        nn.init.normal_(self.film.weight, std=0.02)
        nn.init.zeros_(self.film.bias)
        ident = torch.zeros(3, 3, 3)
        ident[1, 1, 1] = 1.0
        smooth = torch.tensor([1.0, 2.0, 1.0])
        deriv = torch.tensor([-1.0, 0.0, 1.0])
        sz = torch.einsum("i,j,k->ijk", deriv, smooth, smooth) / 32.0
        sy = torch.einsum("i,j,k->ijk", smooth, deriv, smooth) / 32.0
        sx = torch.einsum("i,j,k->ijk", smooth, smooth, deriv) / 32.0
        kernels = torch.stack([ident, sx, sy, sz]).repeat(CH, 1, 1, 1).unsqueeze(1)
        self.register_buffer("percept_w", kernels)

    def alive(self, x):
        return F.max_pool3d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def forward(self, x, theta, z):
        pre_life = self.alive(x)
        p = F.conv3d(x, self.percept_w, padding=1, groups=CH)
        sc = torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)
        scmap = sc[:, :, None, None, None].expand(-1, -1, *x.shape[2:])
        h = self.w1(torch.cat([p, scmap], dim=1))
        gb = self.film(z)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = torch.tanh(gamma)  # bounded gain — non-negotiable (see iter-19.5k NaN)
        h = F.relu(h * (1 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None])
        dx = self.w2(h)
        fire = (torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) <= FIRE_RATE).float()
        x = x + dx * fire
        life = (pre_life & self.alive(x)).float()
        return (x * life).clamp(-8.0, 8.0)

    fused = False  # Triton fused step (CUDA); set from --fused

    def rollout(self, x, theta0, z, steps, seed=0):
        if self.fused:
            # Fused path replaces checkpointing outright: the fused Function
            # saves only each step's 16-ch input state and recomputes in
            # backward (fire mask regenerated from counter-based RNG).
            from fused_step import fused_nca_rollout
            gamma, beta = self.film(z).chunk(2, dim=1)
            gamma = torch.tanh(gamma)  # bounded gain — non-negotiable
            w1 = self.w1.weight.reshape(HIDDEN, CH * 4 + 2)
            w2 = self.w2.weight.reshape(CH, HIDDEN)
            ths = theta0[None, :] + torch.arange(int(steps), device=x.device)[:, None] * OMEGA
            conds = torch.stack([ths.sin(), ths.cos()], dim=2)  # (T, B, 2)
            return fused_nca_rollout(
                x, w1, self.w1.bias, w2, self.w2.bias, steps,
                cond=conds, gamma=gamma, beta=beta, seed=seed,
                fire_rate=FIRE_RATE, clamp=8.0)

        def run_chunk(x0, th0, n):
            for i in range(int(n)):
                x0 = self.forward(x0, th0 + i * OMEGA, z)
            return x0

        done = 0
        while done < steps:
            n = min(CHUNK, steps - done)
            th0 = theta0 + done * OMEGA
            if self.training and x.requires_grad:
                x = checkpoint(run_chunk, x, th0, torch.tensor(n), use_reentrant=False)
            else:
                x = run_chunk(x, th0, n)
            done += n
        return x


class Corpus:
    def __init__(self, path, device):
        data = np.load(path)
        self.z = torch.from_numpy(data["z"]).float()
        self.frames = torch.from_numpy(data["frames"]).permute(0, 1, 5, 2, 3, 4).contiguous()
        self.device = device
        self.n = self.z.shape[0]

    def target_at(self, idx, thetas):
        pos = (thetas.cpu() / (2 * np.pi) * FRAMES) % FRAMES
        f0 = pos.long() % FRAMES
        f1 = (f0 + 1) % FRAMES
        w = (pos - pos.floor())[:, None, None, None, None]
        return ((1 - w) * self.frames[idx, f0].float()
                + w * self.frames[idx, f1].float()).to(self.device)

    def z_at(self, idx):
        return self.z[idx].to(self.device)


def make_seed(n, device):
    x = torch.zeros(n, CH, GRID3, GRID3, GRID3, device=device)
    x[:, 3:, GRID3 // 2, GRID3 // 2, GRID3 // 2] = 1.0
    return x


def damage(x):
    n = x.shape[0]
    g = x.shape[2]
    zz, yy, xx = torch.meshgrid(*(torch.arange(g, device=x.device),) * 3, indexing="ij")
    for i in range(n):
        r = np.random.uniform(4, 9)
        cz, cy, cx = (np.random.uniform(g * 0.25, g * 0.75) for _ in range(3))
        x[i] *= (((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) > r ** 2).float()
    return x


def export(model, path):
    with open(path, "wb") as f:
        f.write(b"NC3M")
        np.array([CH, HIDDEN, ZDIM], dtype="<i4").tofile(f)
        np.array([FIRE_RATE], dtype="<f4").tofile(f)
        model.w1.weight.detach().cpu().numpy().reshape(HIDDEN, CH * 4 + 2).astype("<f4").tofile(f)
        model.w1.bias.detach().cpu().numpy().astype("<f4").tofile(f)
        model.w2.weight.detach().cpu().numpy().reshape(CH, HIDDEN).astype("<f4").tofile(f)
        model.w2.bias.detach().cpu().numpy().astype("<f4").tofile(f)
        model.film.weight.detach().cpu().numpy().astype("<f4").tofile(f)
        model.film.bias.detach().cpu().numpy().astype("<f4").tofile(f)
    print(f"  exported {path}", flush=True)


def save_preview(model, device, path, anchors=None, names=None):
    """Render a few named moods. The anchor set is per-creature: Claudeguy has
    no 'sleep'/'walk'/'manic', so preview names must come from whatever anchors
    this run actually trains on."""
    from PIL import Image
    if anchors is None:
        from manifold_shoggoth3d import ANCHORS as anchors
    if names is None:
        preferred = ["sleep", "walk", "manic", "melancholy", "delight", "mindblown"]
        names = [n for n in preferred if n in anchors][:3] or list(anchors)[:3]
    rows = []
    with torch.no_grad():
        for name in names:
            z = torch.tensor([anchors[name]], dtype=torch.float32, device=device)
            x = make_seed(1, device)
            theta = torch.zeros(1, device=device)
            x = model.rollout(x, theta, z, 300)
            theta = theta + 300 * OMEGA
            shots = []
            for _ in range(4):
                x = model.rollout(x, theta, z, 60)
                theta = theta + 60 * OMEGA
                vol = x[0, :4].permute(1, 2, 3, 0).cpu().numpy()
                a = np.clip(vol[..., 3], 0, 1)
                wgt = a / (a.sum(axis=0, keepdims=True) + 1e-6)
                rgb = (np.clip(vol[..., :3], 0, 1) * wgt[..., None]).sum(axis=0)
                al = 1 - np.prod(1 - a * 0.9, axis=0)
                shots.append(np.concatenate([rgb, al[..., None]], axis=-1)[::-1])
            rows.append(np.concatenate(shots, axis=1))
    sheet = np.concatenate(rows, axis=0)
    Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8), "RGBA").resize(
        (GRID3 * 4 * 4, GRID3 * len(names) * 4), Image.NEAREST).save(path)


def main():
    global BATCH, POOL_SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--corpus", default="corpus_shoggoth3d.npz")
    ap.add_argument("--out", default="shoggoth3d_manifold.nca")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--pool", type=int, default=POOL_SIZE)
    ap.add_argument("--anchors-module", default="manifold_shoggoth3d",
                    help="module supplying ANCHORS for preview renders "
                         "(e.g. manifold_claudeguy3d)")
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="multiply both LR groups (base 2e-3, film 2e-4); lower "
                         "when resuming a run that already annealed")
    ap.add_argument("--milestones", type=int, nargs="*", default=[10000, 22000],
                    help="LR decay iterations. When RESUMING, shift these by the "
                         "iterations already completed so the schedule continues "
                         "as if uninterrupted instead of restarting from scratch.")
    ap.add_argument("--init", default=None,
                    help="warm start from NC3C (cyclic) or NC3M weights. NC3C rows "
                         "are ch*4+3 wide (sin,cos,behavior); the behavior column "
                         "is dropped and FiLM is zeroed so iteration 0 behaves "
                         "exactly like the cyclic parent -- the lineage recipe.")
    ap.add_argument("--fused", action="store_true",
                    help="Triton fused step (CUDA only; see fused_step.py)")
    args = ap.parse_args()

    BATCH, POOL_SIZE = args.batch, args.pool
    try:
        preview_anchors = __import__(args.anchors_module).ANCHORS
    except Exception as e:            # previews are a nicety, never fatal
        print(f"preview anchors unavailable ({e}); previews disabled", flush=True)
        preview_anchors = None
    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    corpus = Corpus(args.corpus, device)
    print(f"corpus: {corpus.n} volumetric cycles, device {device}", flush=True)
    model = ManifoldNCA3D().to(device)
    if args.init:
        with open(args.init, "rb") as f:
            magic = f.read(4)
            hdr = np.fromfile(f, dtype="<i4", count=3 if magic == b"NC3M" else 2)
            if magic == b"NC3C":
                ch, hidden = int(hdr[0]), int(hdr[1])
                cond = int(np.fromfile(f, dtype="<i4", count=1)[0])
            else:
                assert magic == b"NC3M", f"cannot warm-start from {magic!r}"
                ch, hidden, _zd = (int(v) for v in hdr)
                cond = 2
            np.fromfile(f, dtype="<f4", count=1)  # fire rate
            w1in = ch * 4 + cond
            w1 = np.fromfile(f, dtype="<f4", count=hidden * w1in).reshape(hidden, w1in)
            b1 = np.fromfile(f, dtype="<f4", count=hidden)
            w2 = np.fromfile(f, dtype="<f4", count=ch * hidden).reshape(ch, hidden)
            b2 = np.fromfile(f, dtype="<f4", count=ch)
            fw = fb = None
            if magic == b"NC3M":
                # RESUMING a manifold: FiLM is the mood machinery itself. Zeroing
                # it here would silently discard every iteration of mood learning
                # (it did, for 90 seconds, before this was caught).
                fw = np.fromfile(f, dtype="<f4",
                                 count=2 * hidden * _zd).reshape(2 * hidden, _zd)
                fb = np.fromfile(f, dtype="<f4", count=2 * hidden)
                assert fb.size == 2 * hidden, "NC3M film block truncated"
        assert ch == CH and hidden == HIDDEN, f"shape mismatch {ch}/{hidden}"
        keep = CH * 4 + 2                      # perception + sin + cos
        model.w1.weight.data.copy_(
            torch.from_numpy(w1[:, :keep]).to(device)[..., None, None, None])
        model.w1.bias.data.copy_(torch.from_numpy(b1).to(device))
        model.w2.weight.data.copy_(torch.from_numpy(w2).to(device)[..., None, None, None])
        model.w2.bias.data.copy_(torch.from_numpy(b2).to(device))
        if fw is not None:
            assert _zd == ZDIM, f"zdim mismatch: checkpoint {_zd}, model {ZDIM}"
            model.film.weight.data.copy_(torch.from_numpy(fw).to(device))
            model.film.bias.data.copy_(torch.from_numpy(fb).to(device))
            note = f"film RESTORED (|W|={np.abs(fw).mean():.5f}) — manifold resume"
        else:
            # NC3C parent has no FiLM; zeroing makes iteration 0 behave exactly
            # like the cyclic creature, which is the lineage recipe.
            nn.init.zeros_(model.film.weight)
            nn.init.zeros_(model.film.bias)
            note = "film zeroed — starts as the cyclic parent"
        print(f"warm-started from {args.init} ({magic.decode()}); {note}", flush=True)
    model.fused = args.fused
    base_params = [p for n, p in model.named_parameters() if not n.startswith("film")]
    opt = torch.optim.Adam([
        {"params": base_params, "lr": 2e-3 * args.lr_scale},
        {"params": model.film.parameters(), "lr": 2e-4 * args.lr_scale},
    ])
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[m for m in args.milestones if m > 0], gamma=0.3)
    print(f"lr scale {args.lr_scale}, milestones {args.milestones}", flush=True)

    pool = make_seed(POOL_SIZE, device)
    pool_theta = torch.rand(POOL_SIZE, device=device) * 2 * np.pi
    pool_zidx = torch.randint(0, corpus.n, (POOL_SIZE,))
    t0 = time.time()

    for it in range(1, args.iters + 1):
        idx = torch.randint(0, POOL_SIZE, (BATCH,))
        x = pool[idx].clone()
        theta = pool_theta[idx].clone()
        zidx = pool_zidx[idx].clone()

        with torch.no_grad():
            tgt = corpus.target_at(zidx, theta)
            losses = ((x[:, :4] - tgt) ** 2).mean(dim=(1, 2, 3, 4))
            order = losses.argsort(descending=True)
            order_cpu = order.cpu()
            x, theta, zidx = x[order], theta[order], zidx[order_cpu]
            x[0] = make_seed(1, device)[0]
            theta[0] = torch.rand(1, device=device) * 2 * np.pi
            zidx[0] = torch.randint(0, corpus.n, (1,))
            switch = torch.rand(BATCH) < ZSWITCH_P
            switch[0] = False
            ns = int(switch.sum())
            if ns:
                zidx[switch] = torch.randint(0, corpus.n, (ns,))
            if it > 1000:
                x[-DAMAGE_N:] = damage(x[-DAMAGE_N:])

        z = corpus.z_at(zidx)
        T = int(np.random.randint(48, 73))
        x.requires_grad_(True)
        out = model.rollout(x, theta, z, T, seed=it)
        loss = ((out[:, :4] - corpus.target_at(zidx, theta + T * OMEGA)) ** 2).mean()

        if not torch.isfinite(loss):
            print(f"iter {it}: non-finite loss, batch discarded", flush=True)
            opt.zero_grad()
            continue

        opt.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        opt.step()
        sched.step()

        with torch.no_grad():
            out_d = out.detach()
            finite = torch.isfinite(out_d).flatten(1).all(dim=1)
            slots = idx[order_cpu][finite.cpu()]
            pool[slots] = out_d[finite]
            pool_theta[slots] = ((theta + T * OMEGA) % (2 * np.pi))[finite]
            pool_zidx[slots] = zidx[finite.cpu()]

        if it % 100 == 0:
            print(f"iter {it:6d}  loss {loss.item():.5f}  {it/(time.time()-t0):.2f} it/s", flush=True)
        if it % 1000 == 0 or it == args.iters:
            export(model, args.out)
            if preview_anchors is not None:
                save_preview(model, device, "preview_manifold3d.png",
                             anchors=preview_anchors)
        if it % 5000 == 0:
            export(model, f"{args.out}.it{it}")


if __name__ == "__main__":
    main()
