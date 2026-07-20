"""Fused Triton NCA step — the whole automaton step as two kernels.

Port of the proven Metal formulation (NCAShaders.swift / NCAShaders3D.swift) to
Triton for CUDA training, per docs/TRITON_KERNEL_PLAN.md. Replaces ~15 small
CUDA launches per step with:

  _nca_step_fwd  — perception + MLP + FiLM + fire + residual + clamp -> x_mid
  _nca_life_fwd  — 3x3(x3) alpha maxpool life gate over (pre, mid)   -> x_out

Perception is computed inline per channel (explicit tap sums over the 9/27
neighborhood — the tap coefficients are compile-time constants) and the first
MLP layer accumulates via outer-product FMAs, so the kernel does exactly the
eager path's FLOPs. With IEEE fp32 (mandatory for parity — no tf32) tensor
cores are off the table anyway, so tl.dot would buy nothing over FMAs here.

Fire mask uses counter-based RNG — tl.rand(seed, flat_cell_index) — so it is
deterministic given (seed, step): backward and torch.utils.checkpoint replays
recompute it exactly, and the standalone `fire_mask()` helper reproduces it
bit-for-bit for the eager reference in parity tests. Never torch's stateful RNG.

Backward (FusedNCAStep.backward) saves only each step's input state (16 ch —
8x smaller than hidden) and recomputes internals by replaying the forward
kernel with SAVE=True (one launch materializes percept, h_lin, x_mid in flat
layout), then backprops analytically: a fused gate kernel (life+clamp+fire),
cuBLAS mms for the MLP grads, and a perception-transpose kernel for the
input grad. No cuDNN grouped convs anywhere — their groups=16 engine paths
are catastrophically slow (measured: 74% of 2D backward, 60% of 3D).

The eager trainer models remain the permanent reference implementation; this
file, the eager models, and the Metal shaders form a three-way numerical
contract. Any math change updates all three.
"""

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # e.g. macOS / MPS boxes — trainers fall back to eager
    HAS_TRITON = False

CH = 16


# --------------------------------------------------------------------------
# Perception taps (must stay identical to the trainers and Metal shaders)
# --------------------------------------------------------------------------

def perception_taps(dims):
    """(n_kernels, taps) fp32: flattened fixed perception kernels.

    2D: [identity, sobelX, sobelY] as (3, 9), sobel / 8.
    3D: [identity, sobelX, sobelY, sobelZ] as (4, 27), smooth x smooth x deriv / 32.
    Flattening order matches neighborhood enumeration n = ((dz+1)*3+(dy+1))*3+(dx+1).
    """
    if dims == 2:
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sy = sx.T.contiguous()
        return torch.stack([ident, sx, sy]).reshape(3, 9)
    ident = torch.zeros(3, 3, 3)
    ident[1, 1, 1] = 1.0
    smooth = torch.tensor([1.0, 2.0, 1.0])
    deriv = torch.tensor([-1.0, 0.0, 1.0])
    sz = torch.einsum("i,j,k->ijk", deriv, smooth, smooth) / 32.0
    sy = torch.einsum("i,j,k->ijk", smooth, deriv, smooth) / 32.0
    sx = torch.einsum("i,j,k->ijk", smooth, smooth, deriv) / 32.0
    return torch.stack([ident, sx, sy, sz]).reshape(4, 27)


def perception_conv_weight(dims, device):
    """Grouped-conv weight (CH*nk, 1, 3[,3],3) matching the trainer buffers."""
    taps = perception_taps(dims)
    nk = taps.shape[0]
    shape = (3, 3) if dims == 2 else (3, 3, 3)
    return (taps.reshape(nk, *shape).repeat(CH, *(1,) * dims)
            .reshape(CH * nk, 1, *shape).to(device))


def _mix_seed(seed, step):
    return (seed * 1000003 ^ step * 2654435761) & 0x7FFFFFFF


# --------------------------------------------------------------------------
# Folded-weight cache (rebuilt when the param version bumps)
# --------------------------------------------------------------------------

_wcache = {}


def _folded(w1, w2, dims, cond_n):
    """W1eff (KPAD, H) with perception folded in, w1cond (cond_n, H), w2t (H, CH).

    W1eff[(c, n), h] = sum_k w1[h, c*nk+k] * tap_k[n]: perception + first MLP
    layer as ONE matmul over the raw 9/27-cell neighborhood. Exact up to fp32
    reassociation (parity gate: < 1e-5). The FLOP-exact alternative (explicit
    taps + outer-product FMA accumulation) was tried and is 30x SLOWER — the
    unrolled outer products spill registers and defeat Triton's codegen; the
    K-chunked tl.dot keeps the MLP on the fast path.
    """
    key = (w1.data_ptr(), dims, cond_n)
    ver = (w1._version, w2._version)
    hit = _wcache.get(key)
    if hit is not None and hit[0] == ver:
        return hit[1]
    with torch.no_grad():
        taps = perception_taps(dims).to(w1.device)          # (nk, NB)
        nk, nb = taps.shape
        hidden = w1.shape[0]
        k_real = CH * nb
        k_pad = -(-k_real // 32) * 32
        w1p = w1[:, : CH * nk].reshape(hidden, CH, nk)       # (H, CH, nk)
        w1eff = torch.zeros(k_pad, hidden, device=w1.device)
        w1eff[:k_real] = torch.einsum("hck,kn->cnh", w1p, taps).reshape(k_real, hidden)
        w1cond = (
            w1[:, CH * nk :].T.contiguous()
            if cond_n
            else torch.zeros(1, hidden, device=w1.device)
        )
        trip = (w1eff, w1cond, w2.T.contiguous())
    _wcache[key] = (ver, trip)
    return trip


if HAS_TRITON:

    # ----------------------------------------------------------------------
    # Kernels
    # ----------------------------------------------------------------------

    @triton.jit
    def _nca_step_fwd(
        x_ptr, out_ptr,
        w1eff_ptr, b1_ptr, w1cond_ptr, w2t_ptr, b2_ptr,
        cond_ptr, gamma_ptr, beta_ptr,
        percept_ptr, hlin_ptr,      # written only when SAVE (flat: row*BS+cell)
        N, S, BS, Wd, Hd, Dd,
        seed, fire_rate, clamp_v,
        DIMS: tl.constexpr, NB: tl.constexpr, NK: tl.constexpr,
        K: tl.constexpr, KPAD: tl.constexpr,
        DLO: tl.constexpr, DHI: tl.constexpr,
        HIDDEN: tl.constexpr, COND: tl.constexpr,
        FILM: tl.constexpr, CLAMP: tl.constexpr, SAVE: tl.constexpr,
        BLOCK: tl.constexpr, BK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        b = offs // S
        sp = offs % S
        xw = sp % Wd
        yh = (sp // Wd) % Hd
        zd = sp // (Wd * Hd)  # 0 everywhere when DIMS == 2 (Dd == 1)
        hr = tl.arange(0, HIDDEN)

        # hidden = neighborhood @ W1eff (perception folded into the weights)
        acc = tl.zeros((BLOCK, HIDDEN), dtype=tl.float32)
        for k0 in range(0, KPAD, BK):
            j = k0 + tl.arange(0, BK)
            c = j // NB
            n = j % NB
            dx = n % 3 - 1
            dy = (n // 3) % 3 - 1
            xx = xw[:, None] + dx[None, :]
            yy = yh[:, None] + dy[None, :]
            inb = (
                (xx >= 0) & (xx < Wd) & (yy >= 0) & (yy < Hd)
                & (j[None, :] < K) & m[:, None]
            )
            if DIMS == 3:
                dz = n // 9 - 1
                zz = zd[:, None] + dz[None, :]
                inb = inb & (zz >= 0) & (zz < Dd)
                plane = (zz * Hd + yy) * Wd + xx
            else:
                plane = yy * Wd + xx
            addr = (b[:, None] * CHANNELS + c[None, :]) * S + plane
            a = tl.load(x_ptr + addr, mask=inb, other=0.0)
            wblk = tl.load(w1eff_ptr + j[:, None] * HIDDEN + hr[None, :])
            acc += tl.dot(a, wblk, input_precision="ieee")
        acc += tl.load(b1_ptr + hr)[None, :]
        if COND > 0:
            for i in tl.static_range(COND):
                cv = tl.load(cond_ptr + b * COND + i, mask=m, other=0.0)
                wc = tl.load(w1cond_ptr + i * HIDDEN + hr)
                acc += cv[:, None] * wc[None, :]

        if SAVE:
            # backward replay: materialize h_lin (cheap 2D store; percept gets
            # its own small-register kernel — inlining its 432-load loop here
            # next to the (BLOCK,HIDDEN) accumulator made the kernel 5x slower)
            tl.store(hlin_ptr + hr[None, :] * BS + offs[:, None], acc, mask=m[:, None])
        if FILM:
            ga = tl.load(gamma_ptr + b[:, None] * HIDDEN + hr[None, :], mask=m[:, None], other=0.0)
            be = tl.load(beta_ptr + b[:, None] * HIDDEN + hr[None, :], mask=m[:, None], other=0.0)
            acc = acc * (1.0 + ga) + be
        h = tl.maximum(acc, 0.0)

        cr = tl.arange(0, CHANNELS)
        w2blk = tl.load(w2t_ptr + hr[:, None] * CHANNELS + cr[None, :])
        dxv = tl.dot(h, w2blk, input_precision="ieee") + tl.load(b2_ptr + cr)[None, :]

        fire = tl.rand(seed, offs) <= fire_rate
        xoff = (b[:, None] * CHANNELS + cr[None, :]) * S + sp[:, None]
        xin = tl.load(x_ptr + xoff, mask=m[:, None], other=0.0)
        res = xin + tl.where(fire[:, None], dxv, 0.0)
        if CLAMP:
            res = tl.minimum(tl.maximum(res, -clamp_v), clamp_v)
        tl.store(out_ptr + xoff, res, mask=m[:, None])

    @triton.jit
    def _nca_life_fwd(
        pre_ptr, mid_ptr, out_ptr,
        N, S, Wd, Hd, Dd,
        DIMS: tl.constexpr, DLO: tl.constexpr, DHI: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        b = offs // S
        sp = offs % S
        xw = sp % Wd
        yh = (sp // Wd) % Hd
        zd = sp // (Wd * Hd)

        mx_pre = tl.zeros((BLOCK,), dtype=tl.float32)
        mx_mid = tl.zeros((BLOCK,), dtype=tl.float32)
        for dz in tl.static_range(DLO, DHI):
            for dy in tl.static_range(-1, 2):
                for dx in tl.static_range(-1, 2):
                    xx = xw + dx
                    yy = yh + dy
                    inb = (xx >= 0) & (xx < Wd) & (yy >= 0) & (yy < Hd) & m
                    if DIMS == 3:
                        zz = zd + dz
                        inb = inb & (zz >= 0) & (zz < Dd)
                        plane = (zz * Hd + yy) * Wd + xx
                    else:
                        plane = yy * Wd + xx
                    addr = (b * CHANNELS + 3) * S + plane
                    mx_pre = tl.maximum(mx_pre, tl.load(pre_ptr + addr, mask=inb, other=0.0))
                    mx_mid = tl.maximum(mx_mid, tl.load(mid_ptr + addr, mask=inb, other=0.0))
        live = (mx_pre > 0.1) & (mx_mid > 0.1)

        cr = tl.arange(0, CHANNELS)
        xoff = (b[:, None] * CHANNELS + cr[None, :]) * S + sp[:, None]
        mid = tl.load(mid_ptr + xoff, mask=m[:, None], other=0.0)
        tl.store(out_ptr + xoff, tl.where(live[:, None], mid, 0.0), mask=m[:, None])

    @triton.jit
    def _nca_percept_fwd(
        x_ptr, out_ptr,             # out flat (PCH, P): row*BS + cell
        N, S, BS, Wd, Hd, Dd,
        DIMS: tl.constexpr, NK: tl.constexpr,
        DLO: tl.constexpr, DHI: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Perception alone (identity + sobels), small register footprint —
        used by backward to materialize percept for the dw1 matmul."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        b = offs // S
        sp = offs % S
        xw = sp % Wd
        yh = (sp // Wd) % Hd
        zd = sp // (Wd * Hd)

        for c in tl.static_range(CHANNELS):
            base_c = (b * CHANNELS + c) * S
            sid = tl.zeros((BLOCK,), dtype=tl.float32)
            s1 = tl.zeros((BLOCK,), dtype=tl.float32)
            s2 = tl.zeros((BLOCK,), dtype=tl.float32)
            s3 = tl.zeros((BLOCK,), dtype=tl.float32)
            for dz in tl.static_range(DLO, DHI):
                for dy in tl.static_range(-1, 2):
                    for dx in tl.static_range(-1, 2):
                        xx = xw + dx
                        yy = yh + dy
                        inb = (xx >= 0) & (xx < Wd) & (yy >= 0) & (yy < Hd) & m
                        if DIMS == 3:
                            zz = zd + dz
                            inb = inb & (zz >= 0) & (zz < Dd)
                            plane = (zz * Hd + yy) * Wd + xx
                        else:
                            plane = yy * Wd + xx
                        v = tl.load(x_ptr + base_c + plane, mask=inb, other=0.0)
                        # compile-time taps: smooth(d) = 2 - d*d, deriv = d
                        if dz == 0 and dy == 0 and dx == 0:
                            sid = v
                        if DIMS == 2:
                            c1 = dx * (2 - dy * dy) / 8.0
                            c2 = dy * (2 - dx * dx) / 8.0
                            c3 = 0.0
                        else:
                            c1 = dx * (2 - dy * dy) * (2 - dz * dz) / 32.0
                            c2 = dy * (2 - dx * dx) * (2 - dz * dz) / 32.0
                            c3 = dz * (2 - dx * dx) * (2 - dy * dy) / 32.0
                        if c1 != 0.0:
                            s1 += v * c1
                        if c2 != 0.0:
                            s2 += v * c2
                        if c3 != 0.0:
                            s3 += v * c3
            tl.store(out_ptr + (c * NK + 0) * BS + offs, sid, mask=m)
            tl.store(out_ptr + (c * NK + 1) * BS + offs, s1, mask=m)
            tl.store(out_ptr + (c * NK + 2) * BS + offs, s2, mask=m)
            if DIMS == 3:
                tl.store(out_ptr + (c * NK + 3) * BS + offs, s3, mask=m)

    @triton.jit
    def _nca_bwd_gates(
        x_ptr, mid_ptr, g_ptr,
        g1_ptr, ddx_ptr,            # outputs: g1 standard layout, ddx flat (CH, P)
        N, S, BS, Wd, Hd, Dd,
        seed, fire_rate, clamp_v,
        DIMS: tl.constexpr, DLO: tl.constexpr, DHI: tl.constexpr,
        CLAMP: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """dL/dx_mid in one launch: life gate (alpha maxpools over pre & mid),
        clamp gate, and the fire gate for ddx — replaces ~8 eager ops."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        b = offs // S
        sp = offs % S
        xw = sp % Wd
        yh = (sp // Wd) % Hd
        zd = sp // (Wd * Hd)

        mx_pre = tl.zeros((BLOCK,), dtype=tl.float32)
        mx_mid = tl.zeros((BLOCK,), dtype=tl.float32)
        for dz in tl.static_range(DLO, DHI):
            for dy in tl.static_range(-1, 2):
                for dx in tl.static_range(-1, 2):
                    xx = xw + dx
                    yy = yh + dy
                    inb = (xx >= 0) & (xx < Wd) & (yy >= 0) & (yy < Hd) & m
                    if DIMS == 3:
                        zz = zd + dz
                        inb = inb & (zz >= 0) & (zz < Dd)
                        plane = (zz * Hd + yy) * Wd + xx
                    else:
                        plane = yy * Wd + xx
                    addr = (b * CHANNELS + 3) * S + plane
                    mx_pre = tl.maximum(mx_pre, tl.load(x_ptr + addr, mask=inb, other=0.0))
                    mx_mid = tl.maximum(mx_mid, tl.load(mid_ptr + addr, mask=inb, other=0.0))
        live = (mx_pre > 0.1) & (mx_mid > 0.1)
        fire = tl.rand(seed, offs) <= fire_rate

        cr = tl.arange(0, CHANNELS)
        xoff = (b[:, None] * CHANNELS + cr[None, :]) * S + sp[:, None]
        g = tl.load(g_ptr + xoff, mask=m[:, None], other=0.0)
        g1 = tl.where(live[:, None], g, 0.0)
        if CLAMP:
            mid = tl.load(mid_ptr + xoff, mask=m[:, None], other=0.0)
            t = tl.where(live[:, None], mid, 0.0)
            g1 = tl.where((t >= -clamp_v) & (t <= clamp_v), g1, 0.0)
        tl.store(g1_ptr + xoff, g1, mask=m[:, None])
        ddx = tl.where(fire[:, None], g1, 0.0)
        tl.store(ddx_ptr + cr[None, :] * BS + offs[:, None], ddx, mask=m[:, None])

    @triton.jit
    def _nca_bwd_percept(
        dp_ptr, g1_ptr, out_ptr,    # dpercept flat (PCH, P); g1, out standard
        N, S, BS, Wd, Hd, Dd,
        DIMS: tl.constexpr, NK: tl.constexpr,
        DLO: tl.constexpr, DHI: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Perception transpose: dx[c] = g1[c] + sum_k corr(dpercept[c,k],
        flipped tap k). Gather formulation, no atomics. Tap coefficients are
        compile-time constants (flipped: coeff at -offset)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        b = offs // S
        sp = offs % S
        xw = sp % Wd
        yh = (sp // Wd) % Hd
        zd = sp // (Wd * Hd)

        for c in tl.static_range(CHANNELS):
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for dz in tl.static_range(DLO, DHI):
                for dy in tl.static_range(-1, 2):
                    for dx in tl.static_range(-1, 2):
                        xx = xw + dx
                        yy = yh + dy
                        inb = (xx >= 0) & (xx < Wd) & (yy >= 0) & (yy < Hd) & m
                        if DIMS == 3:
                            zz = zd + dz
                            inb = inb & (zz >= 0) & (zz < Dd)
                            plane = (zz * Hd + yy) * Wd + xx
                        else:
                            plane = yy * Wd + xx
                        nb = b * S + plane  # flat P index of the neighbor
                        # flipped taps: coefficient of tap k at offset -d
                        # (smooth is even, deriv is odd -> just negate deriv)
                        if DIMS == 2:
                            f1 = -dx * (2 - dy * dy) / 8.0
                            f2 = -dy * (2 - dx * dx) / 8.0
                            f3 = 0.0
                        else:
                            f1 = -dx * (2 - dy * dy) * (2 - dz * dz) / 32.0
                            f2 = -dy * (2 - dx * dx) * (2 - dz * dz) / 32.0
                            f3 = -dz * (2 - dx * dx) * (2 - dy * dy) / 32.0
                        if dz == 0 and dy == 0 and dx == 0:
                            acc += tl.load(dp_ptr + (c * NK + 0) * BS + nb,
                                           mask=inb, other=0.0)
                        if f1 != 0.0:
                            acc += f1 * tl.load(dp_ptr + (c * NK + 1) * BS + nb,
                                                mask=inb, other=0.0)
                        if f2 != 0.0:
                            acc += f2 * tl.load(dp_ptr + (c * NK + 2) * BS + nb,
                                                mask=inb, other=0.0)
                        if f3 != 0.0:
                            acc += f3 * tl.load(dp_ptr + (c * NK + 3) * BS + nb,
                                                mask=inb, other=0.0)
            addr = (b * CHANNELS + c) * S + sp
            acc += tl.load(g1_ptr + addr, mask=m, other=0.0)
            tl.store(out_ptr + addr, acc, mask=m)

    @triton.jit
    def _fire_mask_kernel(out_ptr, N, seed, fire_rate, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        r = tl.rand(seed, offs)
        tl.store(out_ptr + offs, tl.where(r <= fire_rate, 1.0, 0.0), mask=m)

    # Module-level constexpr baked into the kernels above.
    CHANNELS = tl.constexpr(CH)


def _fire_mask_mixed(x, mixed_seed, fire_rate):
    b, _, *spatial = x.shape
    n = b * math.prod(spatial)
    out = torch.empty(b, 1, *spatial, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(n, 1024),)
    _fire_mask_kernel[grid](out, n, mixed_seed, fire_rate, BLOCK=1024)
    return out


def fire_mask(x, seed, step, fire_rate):
    """(B, 1, *spatial) float 0/1 mask, bit-identical to the fused step's mask.

    Feed this to an eager model for parity tests. Counter-based: same
    (seed, step) -> same mask, always.
    """
    return _fire_mask_mixed(x, _mix_seed(seed, step), fire_rate)


def _geometry(x, dims):
    b = x.shape[0]
    spatial = x.shape[2:]
    s = math.prod(spatial)
    if dims == 2:
        hd, wd = spatial
        dd = 1
    else:
        dd, hd, wd = spatial
    return b, s, b * s, wd, hd, dd


def _launch_step(x, w1eff, b1, w1cond, w2t, b2, cond, gamma, beta,
                 seed_step, fire_rate, clamp, dims, cond_n, film,
                 save_bufs=None):
    b, s, n, wd, hd, dd = _geometry(x, dims)
    nk = 3 if dims == 2 else 4
    nb = 9 if dims == 2 else 27
    hidden = w2t.shape[0]
    dlo, dhi = (-1, 2) if dims == 3 else (0, 1)
    x_mid = torch.empty_like(x)
    dummy = b1  # dead pointer when a flag is off
    if save_bufs is not None:
        percept_buf, hlin_buf = save_bufs
    else:
        percept_buf = hlin_buf = dummy
    grid = (triton.cdiv(n, 64),)
    _nca_step_fwd[grid](
        x, x_mid,
        w1eff, b1, w1cond, w2t, b2,
        cond if cond_n else dummy,
        gamma if film else dummy,
        beta if film else dummy,
        percept_buf, hlin_buf,
        n, s, n, wd, hd, dd,
        seed_step, fire_rate, clamp if clamp is not None else 0.0,
        DIMS=dims, NB=nb, NK=nk, K=CH * nb, KPAD=w1eff.shape[0],
        DLO=dlo, DHI=dhi,
        HIDDEN=hidden, COND=cond_n,
        FILM=film, CLAMP=clamp is not None, SAVE=save_bufs is not None,
        BLOCK=64, BK=32, num_warps=8,
    )
    return x_mid


def _launch_life(x, x_mid, dims):
    b, s, n, wd, hd, dd = _geometry(x, dims)
    dlo, dhi = (-1, 2) if dims == 3 else (0, 1)
    x_out = torch.empty_like(x)
    grid = (triton.cdiv(n, 128),)
    _nca_life_fwd[grid](
        x, x_mid, x_out,
        n, s, wd, hd, dd,
        DIMS=dims, DLO=dlo, DHI=dhi,
        BLOCK=128, num_warps=4,
    )
    return x_out


# --------------------------------------------------------------------------
# Autograd
# --------------------------------------------------------------------------

class FusedNCAStep(torch.autograd.Function):
    """One NCA step: x_out = life(clamp(x + mlp(percept(x), cond, film) * fire)).

    Forward is the two fused Triton kernels. Backward saves only x_in,
    replays the forward kernel once with SAVE=True to materialize internals,
    then backprops analytically (Triton gate/transpose kernels + cuBLAS mms).
    """

    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2, cond, gamma, beta,
                seed_step, fire_rate, clamp, dims, cond_n, film):
        x = x.contiguous()
        w1eff, w1cond, w2t = _folded(w1, w2, dims, cond_n)
        cond_c = cond.contiguous() if cond_n else None
        gamma_c = gamma.contiguous() if film else None
        beta_c = beta.contiguous() if film else None
        x_mid = _launch_step(x, w1eff, b1, w1cond, w2t, b2, cond_c, gamma_c, beta_c,
                             seed_step, fire_rate, clamp, dims, cond_n, film)
        x_out = _launch_life(x, x_mid, dims)
        ctx.save_for_backward(x, w1, b1, w2, b2,
                              cond_c if cond_n else None,
                              gamma_c if film else None,
                              beta_c if film else None)
        ctx.cfg = (seed_step, fire_rate, clamp, dims, cond_n, film)
        return x_out

    @staticmethod
    def backward(ctx, g):
        x, w1, b1, w2, b2, cond, gamma, beta = ctx.saved_tensors
        seed_step, fire_rate, clamp, dims, cond_n, film = ctx.cfg
        hidden = w1.shape[0]
        nk = 3 if dims == 2 else 4
        pch = CH * nk
        bsz, s, p_total, wd, hd, dd = _geometry(x, dims)
        dlo, dhi = (-1, 2) if dims == 3 else (0, 1)
        dev = x.device

        with torch.no_grad():
            # --- replay forward once, materializing internals (flat layout) ---
            w1eff, w1cond, w2t = _folded(w1, w2, dims, cond_n)
            percept_f = torch.empty(pch, p_total, device=dev)
            h_lin_f = torch.empty(hidden, p_total, device=dev)
            # clamp=None: gates must see the RAW pre-clamp x_mid — the clamp
            # gate |x_mid*life| <= 8 is vacuous on already-clamped values
            # (gradient would leak at the boundary; eager blocks it there)
            x_mid = _launch_step(x, w1eff, b1, w1cond, w2t, b2, cond, gamma, beta,
                                 seed_step, fire_rate, None, dims, cond_n, film,
                                 save_bufs=(percept_f, h_lin_f))
            grid = (triton.cdiv(p_total, 128),)
            _nca_percept_fwd[grid](
                x, percept_f,
                p_total, s, p_total, wd, hd, dd,
                DIMS=dims, NK=nk, DLO=dlo, DHI=dhi,
                BLOCK=128, num_warps=4,
            )

            # --- fused gates: life + clamp + fire -> g1, ddx ---
            g = g.contiguous()
            g1 = torch.empty_like(x)
            ddx_f = torch.empty(CH, p_total, device=dev)
            grid = (triton.cdiv(p_total, 128),)
            _nca_bwd_gates[grid](
                x, x_mid, g, g1, ddx_f,
                p_total, s, p_total, wd, hd, dd,
                seed_step, fire_rate, clamp if clamp is not None else 0.0,
                DIMS=dims, DLO=dlo, DHI=dhi,
                CLAMP=clamp is not None, BLOCK=128, num_warps=4,
            )

            # --- MLP backward as flat cuBLAS ---
            if film:
                gam_f = gamma.T.reshape(hidden, bsz, 1)                  # (H, B, 1)
                h_f = F.relu((h_lin_f.reshape(hidden, bsz, -1) * (1 + gam_f)
                              + beta.T.reshape(hidden, bsz, 1))
                             .reshape(hidden, p_total))
            else:
                h_f = F.relu(h_lin_f)
            dw2 = ddx_f @ h_f.T
            db2 = ddx_f.sum(1)
            dh_pre_f = (w2.T @ ddx_f) * (h_f > 0).float()
            if film:
                prod = (dh_pre_f * h_lin_f).reshape(hidden, bsz, -1)
                dgamma = prod.sum(2).T.contiguous()                      # (B, H)
                dbeta = dh_pre_f.reshape(hidden, bsz, -1).sum(2).T.contiguous()
                dh_lin_f = (dh_pre_f.reshape(hidden, bsz, -1) * (1 + gam_f)
                            ).reshape(hidden, p_total)
            else:
                dgamma = dbeta = None
                dh_lin_f = dh_pre_f
            dw1 = torch.empty_like(w1)
            dw1[:, :pch] = dh_lin_f @ percept_f.T
            if cond_n:
                dw1[:, pch:] = dh_lin_f.reshape(hidden, bsz, -1).sum(2) @ cond
            db1 = dh_lin_f.sum(1)
            dpercept_f = w1[:, :pch].T @ dh_lin_f                        # (PCH, P)

            # --- perception transpose + residual, one gather kernel ---
            dx_in = torch.empty_like(x)
            grid = (triton.cdiv(p_total, 128),)
            _nca_bwd_percept[grid](
                dpercept_f.contiguous(), g1, dx_in,
                p_total, s, p_total, wd, hd, dd,
                DIMS=dims, NK=nk, DLO=dlo, DHI=dhi,
                BLOCK=128, num_warps=4,
            )

        return (dx_in, dw1, db1, dw2, db2, None, dgamma, dbeta,
                None, None, None, None, None, None)


def fused_nca_step(x, w1, b1, w2, b2, *, cond=None, gamma=None, beta=None,
                   seed=0, step=0, fire_rate=0.5, clamp=8.0):
    """One fused NCA step. Drop-in for the eager trainer forward()s.

    x: (B, 16, H, W) or (B, 16, D, H, W). w1: (HIDDEN, PCH + cond_n) — pass
    conv weight as w1.weight.reshape(HIDDEN, -1). w2: (16, HIDDEN). cond:
    (B, cond_n) per-sample scalars appended after perception. gamma/beta:
    (B, HIDDEN) FiLM (gamma already tanh-bounded by the caller). clamp: state
    bound (None = unclamped, e.g. the 2D bonsai/cyclic trainers).

    (seed, step) drive the counter-based fire RNG: pass a distinct step per
    NCA step, derived from loop indices (NOT a mutable counter) so that
    torch.utils.checkpoint replays regenerate identical masks.
    """
    if not HAS_TRITON:
        raise RuntimeError("triton not available — use the eager path")
    dims = x.dim() - 2
    cond_n = cond.shape[1] if cond is not None else 0
    film = gamma is not None
    return FusedNCAStep.apply(x, w1, b1, w2, b2, cond, gamma, beta,
                              _mix_seed(seed, step), fire_rate, clamp,
                              dims, cond_n, film)
