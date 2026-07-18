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
func ncaMetalSource(cond: Int, hidden: Int = 128, useFilm: Bool = false) -> String {
    """
    #include <metal_stdlib>
    using namespace metal;

    constant int CH = 16;
    constant int PCH = 48;
    constant int COND = \(cond);
    constant int PIN = PCH + COND;   // w1 input width
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

        const device float *w1 = weights;
        const device float *b1 = w1 + HIDDEN * PIN;
        const device float *w2 = b1 + HIDDEN;
        const device float *b2 = w2 + CH * HIDDEN;

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
        for (int c = 0; c < CH; c++) {
            float acc = b2[c];
            const device float *row = w2 + c * HIDDEN;
            for (int h = 0; h < HIDDEN; h++) acc += row[h] * hidden[h];
            // +-8 state bound matches training; inert for healthy dynamics
            dst[base + c] = clamp(src[base + c] + (fire ? acc : 0.0f), -8.0f, 8.0f);
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

    // Nearest-neighbor upscale of state RGBA into the drawable (premultiplied alpha).
    kernel void nca_render(const device float *state [[buffer(0)]],
                           constant Uniforms &u      [[buffer(1)]],
                           texture2d<float, access::write> tex [[texture(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
        int sx = int(gid.x * (uint)u.width  / tex.get_width());
        int sy = int(gid.y * (uint)u.height / tex.get_height());
        if (u.flipX != 0) sx = u.width - 1 - sx;
        int base = (sy * u.width + sx) * CH;
        float4 rgba = clamp(float4(state[base], state[base+1], state[base+2], state[base+3]), 0.0, 1.0);
        rgba.rgb = min(rgba.rgb, rgba.a);  // premultiplied-alpha invariant
        if (u.style == 1 && (gid.y % 3u) == 0u) rgba *= 0.86;  // faint CRT scanlines
        tex.write(rgba, gid);
    }
    """
}
