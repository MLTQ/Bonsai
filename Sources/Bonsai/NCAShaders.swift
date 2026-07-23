/// Metal source for the NCA, compiled at runtime (no .metal build step needed).
///
/// Generated per-creature: `cond` is the number of conditioning channels appended
/// after the 48 perception features (0 for static creatures like the bonsai; 3 for
/// the phase-conditioned Lain; 2 = sin/cos for NCA3 manifold creatures), `hidden`
/// is the update-rule width, and `useFilm` enables FiLM modulation of the hidden
/// layer (gamma/beta supplied per step in buffer 4, computed CPU-side from z).
///
/// Must stay numerically equivalent to the PyTorch models in training/: perception
/// ordering [identity, sobelX, sobelY] interleaved per channel, sobel / 8,
/// cross-correlation with zero padding, per-cell stochastic fire mask, and life
/// mask = alive(pre) AND alive(post) where alive is maxpool3x3(alpha) > 0.1.
func ncaMetalSource(cond: Int, hidden: Int = 128, useFilm: Bool = false, npool: Int = 0,
                    stateChannels: Int = 16, positionChannels: Int = 16,
                    momentumChannels: Int = 0) -> String {
    """
    #include <metal_stdlib>
    using namespace metal;

    constant int CH = \(stateChannels);
    constant int POSITION_CH = \(positionChannels);
    constant int MOMENTUM_CH = \(momentumChannels);
    constant int RESIDUAL_CH = POSITION_CH - MOMENTUM_CH;
    constant int OUTPUT_CH = POSITION_CH;
    constant int PCH = CH * 3;
    constant int COND = \(cond);
    constant int NPOOL = \(npool);  // globally-broadcast feedback channels (NCAP)
    constant int PIN = PCH + COND + NPOOL;   // w1 input width
    constant int HIDDEN = \(hidden);
    constant bool USE_FILM = \(useFilm);

    struct Uniforms {
        int   width;
        int   height;
        float fireRate;
        uint  step;
        int   damageActive;
        float damageX;
        float damageY;
        float damageRadius;
        float cond0;
        float cond1;
        float cond2;
        float cond3;
        int   style;    // render: 0 = plain, 1 = CRT scanlines
        int   flipX;    // render: mirror horizontally (creature facing)
        int   crisp;    // render: 1 = bilinear + alpha remap (sharp silhouette)
        float momentumDecay;  // NCA4 velocity retention; ignored by residual formats
    };

    inline float cellCh(const device float *s, int x, int y, int c, int W, int H) {
        if (x < 0 || y < 0 || x >= W || y >= H) return 0.0;
        return s[(y * W + x) * CH + c];
    }

    inline float rand01(uint x, uint y, uint step) {
        uint h = x * 1664525u ^ y * 1013904223u ^ step * 69069u;
        h ^= h >> 16; h *= 0x7feb352du;
        h ^= h >> 15; h *= 0x846ca68bu;
        h ^= h >> 16;
        return float(h & 0x00FFFFFFu) / 16777216.0;
    }

    kernel void nca_step(const device float *src      [[buffer(0)]],
                         device float *dst            [[buffer(1)]],
                         const device float *weights  [[buffer(2)]],
                         constant Uniforms &u         [[buffer(3)]],
                         const device float *film     [[buffer(4)]],   // gamma[HIDDEN], beta[HIDDEN]
                         const device float *pooled   [[buffer(5)]],   // g[NPOOL] from nca_pool
                         uint2 gid [[thread_position_in_grid]]) {
        int W = u.width, H = u.height;
        if ((int)gid.x >= W || (int)gid.y >= H) return;
        int x = gid.x, y = gid.y;

        float percept[PIN];
        for (int c = 0; c < CH; c++) {
            float tl = cellCh(src, x-1, y-1, c, W, H), t = cellCh(src, x, y-1, c, W, H), tr = cellCh(src, x+1, y-1, c, W, H);
            float l  = cellCh(src, x-1, y,   c, W, H),                                   r  = cellCh(src, x+1, y,   c, W, H);
            float bl = cellCh(src, x-1, y+1, c, W, H), b = cellCh(src, x, y+1, c, W, H), br = cellCh(src, x+1, y+1, c, W, H);
            percept[c*3 + 0] = cellCh(src, x, y, c, W, H);
            percept[c*3 + 1] = (-tl + tr - 2.0*l + 2.0*r - bl + br) / 8.0;  // sobel X
            percept[c*3 + 2] = (-tl - 2.0*t - tr + bl + 2.0*b + br) / 8.0;  // sobel Y
        }
        if (COND > 0) {
            float cv[4] = { u.cond0, u.cond1, u.cond2, u.cond3 };
            for (int i = 0; i < COND; i++) percept[PCH + i] = cv[i];
        }
        for (int i = 0; i < NPOOL; i++) percept[PCH + COND + i] = pooled[i];

        const device float *w1 = weights;
        const device float *b1 = w1 + HIDDEN * PIN;
        const device float *w2 = b1 + HIDDEN;
        const device float *b2 = w2 + OUTPUT_CH * HIDDEN;

        float hidden[HIDDEN];
        for (int h = 0; h < HIDDEN; h++) {
            float acc = b1[h];
            const device float *row = w1 + h * PIN;
            for (int i = 0; i < PIN; i++) acc += row[i] * percept[i];
            if (USE_FILM) acc = acc * (1.0f + film[h]) + film[HIDDEN + h];
            hidden[h] = max(acc, 0.0f);
        }

        int base = (y * W + x) * CH;
        bool fire = rand01(gid.x, gid.y, u.step) <= u.fireRate;
        for (int c = 0; c < OUTPUT_CH; c++) {
            float acc = b2[c];
            const device float *row = w2 + c * HIDDEN;
            for (int h = 0; h < HIDDEN; h++) acc += row[h] * hidden[h];
            if (c >= RESIDUAL_CH) {
                // Symplectic Euler: force is stochastic, but stored velocity keeps
                // advancing and damping on every cell step. NCA4 applies this to
                // every position channel; NCA5 applies it only to hidden channels.
                int velocityChannel = POSITION_CH + (c - RESIDUAL_CH);
                float velocity = u.momentumDecay * src[base + velocityChannel]
                               + (fire ? acc : 0.0f);
                dst[base + c] = clamp(src[base + c] + velocity, -8.0f, 8.0f);
                dst[base + velocityChannel] = clamp(velocity, -8.0f, 8.0f);
            } else {
                // +-8 state bound matches training; inert for healthy dynamics
                dst[base + c] = clamp(src[base + c] + (fire ? acc : 0.0f), -8.0f, 8.0f);
            }
        }

        if (u.damageActive != 0) {
            float ddx = float(x) - u.damageX, ddy = float(y) - u.damageY;
            if (ddx*ddx + ddy*ddy <= u.damageRadius * u.damageRadius) {
                for (int c = 0; c < CH; c++) dst[base + c] = 0.0f;
            }
        }
    }

    inline bool aliveAt(const device float *s, int x, int y, int W, int H) {
        float m = 0.0;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
                m = max(m, cellCh(s, x+dx, y+dy, 3, W, H));
        return m > 0.1;
    }

    // Life mask: cells not alive both before and after the update are zeroed.
    // Reads pre and post, writes a third buffer to avoid read/write races.
    // Alive-masked spatial mean of state channels [4, 4+NPOOL) — the global
    // variable of a pooled creature (training/pooled_nca.py). One threadgroup;
    // each thread strides the grid, then a shared-memory tree reduction.
    // Count clamps to >= 1, matching the PyTorch reference exactly.
    kernel void nca_pool(const device float *src   [[buffer(0)]],
                         device float *outPooled   [[buffer(1)]],
                         constant Uniforms &u      [[buffer(2)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tcount [[threads_per_threadgroup]]) {
        threadgroup float sums[256 * 9];   // NPOOL <= 8, +1 for the alive count
        int W = u.width, H = u.height;
        float acc[9];
        for (int i = 0; i <= NPOOL; i++) acc[i] = 0.0;
        for (int idx = tid; idx < W * H; idx += tcount) {
            int x = idx % W, y = idx / W;
            // alive = 3x3 max of alpha > 0.1 (same rule as nca_life's masks)
            float m = 0.0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    m = max(m, cellCh(src, x + dx, y + dy, 3, W, H));
            if (m > 0.1) {
                int base = idx * CH;
                for (int i = 0; i < NPOOL; i++) acc[i] += src[base + 4 + i];
                acc[NPOOL] += 1.0;
            }
        }
        for (int i = 0; i <= NPOOL; i++) sums[tid * 9 + i] = acc[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tcount / 2; stride > 0; stride /= 2) {
            if (tid < stride)
                for (int i = 0; i <= NPOOL; i++)
                    sums[tid * 9 + i] += sums[(tid + stride) * 9 + i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            float n = max(sums[NPOOL], 1.0f);
            for (int i = 0; i < NPOOL; i++) outPooled[i] = sums[i] / n;
        }
    }

    kernel void nca_life(const device float *pre   [[buffer(0)]],
                         const device float *post  [[buffer(1)]],
                         device float *outBuf      [[buffer(2)]],
                         constant Uniforms &u      [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
        int W = u.width, H = u.height;
        if ((int)gid.x >= W || (int)gid.y >= H) return;
        int x = gid.x, y = gid.y;
        bool live = aliveAt(pre, x, y, W, H) && aliveAt(post, x, y, W, H);
        int base = (y * W + x) * CH;
        for (int c = 0; c < CH; c++) outBuf[base + c] = live ? post[base + c] : 0.0f;
    }

    inline float4 stateRGBA(const device float *state, int x, int y, int W, int H) {
        x = clamp(x, 0, W - 1); y = clamp(y, 0, H - 1);
        int base = (y * W + x) * CH;
        float4 c = clamp(float4(state[base], state[base+1], state[base+2], state[base+3]), 0.0, 1.0);
        c.rgb = min(c.rgb, c.a);  // premultiplied-alpha invariant
        return c;
    }

    // Upscale of state RGBA into the drawable (premultiplied alpha).
    // crisp mode is the SDF-font trick: bilinear-sample the soft alpha field at
    // display resolution, then push it through a steep smoothstep. The 95% of
    // measured softness that lives in the silhouette's alpha ramp collapses to
    // a crisp antialiased edge; the creature's dynamics are untouched (display
    // only — headless verification still reads the raw state).
    kernel void nca_render(const device float *state [[buffer(0)]],
                           constant Uniforms &u      [[buffer(1)]],
                           texture2d<float, access::write> tex [[texture(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
        float4 rgba;
        if (u.crisp != 0) {
            float fx = (float(gid.x) + 0.5f) * float(u.width)  / float(tex.get_width())  - 0.5f;
            float fy = (float(gid.y) + 0.5f) * float(u.height) / float(tex.get_height()) - 0.5f;
            if (u.flipX != 0) fx = float(u.width - 1) - fx;
            int x0 = int(floor(fx)), y0 = int(floor(fy));
            float tx = fx - floor(fx), ty = fy - floor(fy);
            float4 c = mix(mix(stateRGBA(state, x0,     y0, u.width, u.height),
                               stateRGBA(state, x0 + 1, y0, u.width, u.height), tx),
                           mix(stateRGBA(state, x0,     y0 + 1, u.width, u.height),
                               stateRGBA(state, x0 + 1, y0 + 1, u.width, u.height), tx), ty);
            // Narrow the alpha ramp; re-premultiply so colour tracks the new edge.
            float a2 = smoothstep(0.08f, 0.60f, c.a);
            rgba = float4(c.rgb * (a2 / max(c.a, 1e-4f)), a2);
            rgba.rgb = min(rgba.rgb, rgba.a);
        } else {
            int sx = int(gid.x * (uint)u.width  / tex.get_width());
            int sy = int(gid.y * (uint)u.height / tex.get_height());
            if (u.flipX != 0) sx = u.width - 1 - sx;
            rgba = stateRGBA(state, sx, sy, u.width, u.height);
        }
        if (u.style == 1 && (gid.y % 3u) == 0u) rgba *= 0.86;  // faint CRT scanlines
        tex.write(rgba, gid);
    }
    """
}
