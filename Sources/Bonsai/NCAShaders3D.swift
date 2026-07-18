/// Metal source for the volumetric NCA: 3D step/life kernels plus an
/// emission-absorption raymarcher with gradient-based diffuse shading.
///
/// Numerical contract with training/train_nca3d.py and train_cyclic3d.py:
/// perception ordering [identity, sobelX, sobelY, sobelZ] interleaved per channel,
/// 3D Sobels = smooth(1,2,1) x smooth x deriv(-1,0,1) / 32, zero padding,
/// volume index order (z, y, x) -> flat idx ((z*G + y)*G + x)*CH + c,
/// life = alive(pre) AND alive(post) with 27-neighborhood maxpool alpha > 0.1,
/// per-voxel stochastic fire, state clamped to +-8.
func nca3dMetalSource(cond: Int, hidden: Int) -> String {
    """
    #include <metal_stdlib>
    using namespace metal;

    constant int CH = 16;
    constant int PCH = 64;              // identity + 3 sobels, per channel
    constant int COND = \(cond);
    constant int PIN = PCH + COND;
    constant int HIDDEN = \(hidden);

    struct Uniforms3D {
        int   grid;
        float fireRate;
        uint  step;
        int   damageActive;
        float damageX;   // grid-space (x, y, z)
        float damageY;
        float damageZ;
        float damageRadius;
        float cond0;
        float cond1;
        float cond2;
        float cond3;
        float azimuth;   // render: orbit angle (radians)
        float elevation; // render: camera tilt (radians)
    };

    inline int vidx(int x, int y, int z, int G) { return ((z * G + y) * G + x) * CH; }

    inline float vox(const device float *s, int x, int y, int z, int c, int G) {
        if (x < 0 || y < 0 || z < 0 || x >= G || y >= G || z >= G) return 0.0;
        return s[vidx(x, y, z, G) + c];
    }

    inline float rand01(uint x, uint y, uint z, uint step) {
        uint h = x * 1664525u ^ y * 1013904223u ^ z * 2654435761u ^ step * 69069u;
        h ^= h >> 16; h *= 0x7feb352du;
        h ^= h >> 15; h *= 0x846ca68bu;
        h ^= h >> 16;
        return float(h & 0x00FFFFFFu) / 16777216.0;
    }

    kernel void nca3d_step(const device float *src      [[buffer(0)]],
                           device float *dst            [[buffer(1)]],
                           const device float *weights  [[buffer(2)]],
                           constant Uniforms3D &u       [[buffer(3)]],
                           uint3 gid [[thread_position_in_grid]]) {
        int G = u.grid;
        if ((int)gid.x >= G || (int)gid.y >= G || (int)gid.z >= G) return;
        int x = gid.x, y = gid.y, z = gid.z;

        // Perception: identity + 3D sobel x/y/z (smooth-smooth-deriv / 32).
        // Weights of the separable kernels: w = s[a]*s[b]*d[c] enumerated inline.
        float percept[PIN];
        for (int c = 0; c < CH; c++) {
            float sxv = 0.0, syv = 0.0, szv = 0.0;
            for (int dz = -1; dz <= 1; dz++)
            for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                float v = vox(src, x+dx, y+dy, z+dz, c, G);
                if (v == 0.0) continue;
                float smx = (dx == 0) ? 2.0 : 1.0;
                float smy = (dy == 0) ? 2.0 : 1.0;
                float smz = (dz == 0) ? 2.0 : 1.0;
                sxv += v * float(dx) * smy * smz;
                syv += v * float(dy) * smx * smz;
                szv += v * float(dz) * smx * smy;
            }
            percept[c*4 + 0] = vox(src, x, y, z, c, G);
            percept[c*4 + 1] = sxv / 32.0;
            percept[c*4 + 2] = syv / 32.0;
            percept[c*4 + 3] = szv / 32.0;
        }
        if (COND > 0) {
            float cv[4] = { u.cond0, u.cond1, u.cond2, u.cond3 };
            for (int i = 0; i < COND; i++) percept[PCH + i] = cv[i];
        }

        const device float *w1 = weights;
        const device float *b1 = w1 + HIDDEN * PIN;
        const device float *w2 = b1 + HIDDEN;
        const device float *b2 = w2 + CH * HIDDEN;

        float hid[HIDDEN];
        for (int h = 0; h < HIDDEN; h++) {
            float acc = b1[h];
            const device float *row = w1 + h * PIN;
            for (int i = 0; i < PIN; i++) acc += row[i] * percept[i];
            hid[h] = max(acc, 0.0f);
        }

        int base = vidx(x, y, z, G);
        bool fire = rand01(gid.x, gid.y, gid.z, u.step) <= u.fireRate;
        for (int c = 0; c < CH; c++) {
            float acc = b2[c];
            const device float *row = w2 + c * HIDDEN;
            for (int h = 0; h < HIDDEN; h++) acc += row[h] * hid[h];
            dst[base + c] = clamp(src[base + c] + (fire ? acc : 0.0f), -8.0f, 8.0f);
        }

        if (u.damageActive != 0) {
            float ddx = float(x) - u.damageX, ddy = float(y) - u.damageY, ddz = float(z) - u.damageZ;
            if (ddx*ddx + ddy*ddy + ddz*ddz <= u.damageRadius * u.damageRadius) {
                for (int c = 0; c < CH; c++) dst[base + c] = 0.0f;
            }
        }
    }

    inline bool aliveAt3(const device float *s, int x, int y, int z, int G) {
        float m = 0.0;
        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            m = max(m, vox(s, x+dx, y+dy, z+dz, 3, G));
        return m > 0.1;
    }

    kernel void nca3d_life(const device float *pre   [[buffer(0)]],
                           const device float *post  [[buffer(1)]],
                           device float *outBuf      [[buffer(2)]],
                           constant Uniforms3D &u    [[buffer(3)]],
                           uint3 gid [[thread_position_in_grid]]) {
        int G = u.grid;
        if ((int)gid.x >= G || (int)gid.y >= G || (int)gid.z >= G) return;
        int x = gid.x, y = gid.y, z = gid.z;
        bool live = aliveAt3(pre, x, y, z, G) && aliveAt3(post, x, y, z, G);
        int base = vidx(x, y, z, G);
        for (int c = 0; c < CH; c++) outBuf[base + c] = live ? post[base + c] : 0.0f;
    }

    // --- Rendering -----------------------------------------------------------

    inline float4 sampleVol(const device float *s, float3 p, int G) {
        // Trilinear over the 8 surrounding voxels, RGBA only.
        float3 pf = p - 0.5;
        int3 i0 = int3(floor(pf));
        float3 f = pf - float3(i0);
        float4 acc = 0.0;
        for (int dz = 0; dz <= 1; dz++)
        for (int dy = 0; dy <= 1; dy++)
        for (int dx = 0; dx <= 1; dx++) {
            float w = (dx ? f.x : 1-f.x) * (dy ? f.y : 1-f.y) * (dz ? f.z : 1-f.z);
            if (w <= 0.0) continue;
            int xx = i0.x+dx, yy = i0.y+dy, zz = i0.z+dz;
            acc += w * float4(vox(s, xx, yy, zz, 0, G), vox(s, xx, yy, zz, 1, G),
                              vox(s, xx, yy, zz, 2, G), vox(s, xx, yy, zz, 3, G));
        }
        return acc;
    }

    inline float alphaAt(const device float *s, float3 p, int G) {
        return sampleVol(s, p, G).a;
    }

    kernel void nca3d_render(const device float *state [[buffer(0)]],
                             constant Uniforms3D &u    [[buffer(1)]],
                             texture2d<float, access::write> tex [[texture(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
        int G = u.grid;
        float3 center = float3(G) * 0.5;

        // Orbiting camera: azimuth around y, slight elevation, mild perspective.
        float ca = cos(u.azimuth), sa = sin(u.azimuth);
        float ce = cos(u.elevation), se = sin(u.elevation);
        float dist = 2.4 * float(G);
        float3 eye = center + dist * float3(sa * ce, se, ca * ce);
        float3 fwd = normalize(center - eye);
        float3 right = normalize(cross(float3(0, 1, 0), fwd));
        float3 up = cross(fwd, right);

        float2 ndc = float2((float(gid.x) + 0.5) / tex.get_width() - 0.5,
                            0.5 - (float(gid.y) + 0.5) / tex.get_height());
        float focal = 1.9;  // ~30 degree fov
        float3 dir = normalize(fwd * focal + right * ndc.x * 1.6 + up * ndc.y * 1.6);

        // Ray-box intersection with [0, G]^3
        float3 inv = 1.0 / dir;
        float3 t0s = (float3(0.0) - eye) * inv;
        float3 t1s = (float3(G) - eye) * inv;
        float3 tmin3 = min(t0s, t1s), tmax3 = max(t0s, t1s);
        float tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
        float tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
        if (tmax <= max(tmin, 0.0)) { tex.write(float4(0.0), gid); return; }

        float3 lightDir = normalize(float3(-0.5, 0.8, 0.6));
        float3 C = 0.0;
        float T = 1.0;
        float dt = 0.55;
        for (float t = max(tmin, 0.0) + dt * 0.5; t < tmax && T > 0.012; t += dt) {
            float3 p = eye + dir * t;
            float4 s = sampleVol(state, p, G);
            float a = clamp(s.a, 0.0, 1.0);
            if (a < 0.02) continue;
            float3 col = clamp(s.rgb, 0.0, 1.0) / max(a, 0.05);  // un-premultiply
            // Cheap diffuse from the density gradient
            float e = 1.0;
            float3 grad = float3(
                alphaAt(state, p + float3(e,0,0), G) - alphaAt(state, p - float3(e,0,0), G),
                alphaAt(state, p + float3(0,e,0), G) - alphaAt(state, p - float3(0,e,0), G),
                alphaAt(state, p + float3(0,0,e), G) - alphaAt(state, p - float3(0,0,e), G));
            float glen = length(grad);
            float diffuse = glen > 1e-4 ? (0.62 + 0.38 * max(dot(-grad / glen, lightDir), 0.0)) : 0.8;
            float sampleA = 1.0 - exp(-a * 1.9 * dt);
            C += T * sampleA * col * diffuse;
            T *= 1.0 - sampleA;
        }
        tex.write(float4(C, 1.0 - T), gid);  // premultiplied
    }
    """
}
