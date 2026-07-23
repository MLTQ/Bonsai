/// Metal kernels for the hard-routed fused pose/edge NCA.
func fusedNCAMetalSource(_ weights: FusedNCAWeights) -> String {
    let ch = weights.channels
    let experts = weights.experts
    let slots = weights.slots
    let poseHidden = weights.expertHidden
    let flowHidden = weights.flowHidden
    let frequencies = weights.positionFrequencies
    let coordinateChannels = weights.coordinateChannels
    let inputs = weights.ruleInputs

    let poseW1 = 0
    let poseB1 = poseW1 + experts * poseHidden * inputs
    let poseW2 = poseB1 + experts * poseHidden
    let poseB2 = poseW2 + experts * ch * poseHidden
    let flowW1 = poseB2 + experts * ch
    let flowB1 = flowW1 + experts * flowHidden * inputs
    let flowW2 = flowB1 + experts * flowHidden
    let flowB2 = flowW2 + experts * slots * 2 * flowHidden
    let slotW1 = flowB2 + experts * slots * 2
    let slotB1 = slotW1 + experts * flowHidden * inputs
    let slotW2 = slotB1 + experts * flowHidden
    let slotB2 = slotW2 + experts * slots * flowHidden
    let repairW1 = slotB2 + experts * slots
    let repairB1 = repairW1 + experts * poseHidden * inputs
    let repairW2 = repairB1 + experts * poseHidden
    let repairB2 = repairW2 + experts * ch * poseHidden

    return """
    #include <metal_stdlib>
    using namespace metal;

    constant int CH = \(ch);
    constant int EXPERTS = \(experts);
    constant int SLOTS = \(slots);
    constant int POSE_HIDDEN = \(poseHidden);
    constant int FLOW_HIDDEN = \(flowHidden);
    constant int POSITION_FREQUENCIES = \(frequencies);
    constant int COORD_CH = \(coordinateChannels);
    constant int PCH = CH * 3;
    constant int PIN = \(inputs);

    constant int POSE_W1 = \(poseW1);
    constant int POSE_B1 = \(poseB1);
    constant int POSE_W2 = \(poseW2);
    constant int POSE_B2 = \(poseB2);
    constant int FLOW_W1 = \(flowW1);
    constant int FLOW_B1 = \(flowB1);
    constant int FLOW_W2 = \(flowW2);
    constant int FLOW_B2 = \(flowB2);
    constant int SLOT_W1 = \(slotW1);
    constant int SLOT_B1 = \(slotB1);
    constant int SLOT_W2 = \(slotW2);
    constant int SLOT_B2 = \(slotB2);
    constant int REPAIR_W1 = \(repairW1);
    constant int REPAIR_B1 = \(repairB1);
    constant int REPAIR_W2 = \(repairW2);
    constant int REPAIR_B2 = \(repairB2);

    struct FusedUniforms {
        int width;
        int height;
        float fireRate;
        float maxFlow;
        uint step;
        int expert;
        int transition;
        float progress;
        int damageActive;
        float damageX;
        float damageY;
        float damageRadius;
        int crisp;
        int flipX;
    };

    inline float stateCh(const device float *state, int x, int y, int c,
                         int width, int height) {
        if (x < 0 || y < 0 || x >= width || y >= height) return 0.0f;
        return state[(y * width + x) * CH + c];
    }

    inline float predictedCh(const device float *state, int slot, int x, int y,
                             int c, int width, int height) {
        if (x < 0 || y < 0 || x >= width || y >= height) return 0.0f;
        return state[((slot * height + y) * width + x) * CH + c];
    }

    inline float bilinearState(const device float *state, float x, float y, int c,
                               int width, int height) {
        int x0 = int(floor(x)), y0 = int(floor(y));
        float tx = x - floor(x), ty = y - floor(y);
        return mix(
            mix(stateCh(state, x0, y0, c, width, height),
                stateCh(state, x0 + 1, y0, c, width, height), tx),
            mix(stateCh(state, x0, y0 + 1, c, width, height),
                stateCh(state, x0 + 1, y0 + 1, c, width, height), tx), ty);
    }

    inline float bilinearPredicted(const device float *state, int slot,
                                   float x, float y, int c,
                                   int width, int height) {
        int x0 = int(floor(x)), y0 = int(floor(y));
        float tx = x - floor(x), ty = y - floor(y);
        return mix(
            mix(predictedCh(state, slot, x0, y0, c, width, height),
                predictedCh(state, slot, x0 + 1, y0, c, width, height), tx),
            mix(predictedCh(state, slot, x0, y0 + 1, c, width, height),
                predictedCh(state, slot, x0 + 1, y0 + 1, c, width, height), tx), ty);
    }

    inline void makeRuleInput(const device float *state, int x, int y,
                              constant FusedUniforms &u, thread float *input) {
        for (int c = 0; c < CH; c++) {
            float tl = stateCh(state, x-1, y-1, c, u.width, u.height);
            float t  = stateCh(state, x,   y-1, c, u.width, u.height);
            float tr = stateCh(state, x+1, y-1, c, u.width, u.height);
            float l  = stateCh(state, x-1, y,   c, u.width, u.height);
            float r  = stateCh(state, x+1, y,   c, u.width, u.height);
            float bl = stateCh(state, x-1, y+1, c, u.width, u.height);
            float b  = stateCh(state, x,   y+1, c, u.width, u.height);
            float br = stateCh(state, x+1, y+1, c, u.width, u.height);
            input[c*3] = stateCh(state, x, y, c, u.width, u.height);
            input[c*3+1] = (-tl + tr - 2.0f*l + 2.0f*r - bl + br) / 8.0f;
            input[c*3+2] = (-tl - 2.0f*t - tr + bl + 2.0f*b + br) / 8.0f;
        }
        float xn = 2.0f * float(x) / float(max(u.width - 1, 1)) - 1.0f;
        float yn = 2.0f * float(y) / float(max(u.height - 1, 1)) - 1.0f;
        int offset = PCH;
        input[offset++] = xn;
        input[offset++] = yn;
        for (int frequency = 0; frequency < POSITION_FREQUENCIES; frequency++) {
            float scale = M_PI_F * float(1 << frequency);
            input[offset++] = sin(scale * xn);
            input[offset++] = cos(scale * xn);
            input[offset++] = sin(scale * yn);
            input[offset++] = cos(scale * yn);
        }
        input[offset] = u.progress;
    }

    inline float affineOutput(const device float *weights, int weightBase,
                              int biasBase, int expert, int outputs, int inputs,
                              int output, const thread float *values) {
        float result = weights[biasBase + expert * outputs + output];
        int row = weightBase + (expert * outputs + output) * inputs;
        for (int i = 0; i < inputs; i++) result += weights[row + i] * values[i];
        return result;
    }

    inline uint randHash(uint x, uint y, uint step) {
        uint h = x * 1664525u ^ y * 1013904223u ^ step * 69069u;
        h ^= h >> 16; h *= 0x7feb352du;
        h ^= h >> 15; h *= 0x846ca68bu;
        h ^= h >> 16;
        return h;
    }

    inline bool fires(uint x, uint y, uint step, float rate) {
        float value = float(randHash(x, y, step) & 0x00FFFFFFu) / 16777216.0f;
        return value <= rate;
    }

    kernel void fused_pose_step(const device float *src [[buffer(0)]],
                                device float *dst [[buffer(1)]],
                                const device float *weights [[buffer(2)]],
                                constant FusedUniforms &u [[buffer(3)]],
                                uint2 gid [[thread_position_in_grid]]) {
        if ((int)gid.x >= u.width || (int)gid.y >= u.height) return;
        int x = gid.x, y = gid.y;
        float input[PIN];
        float hidden[POSE_HIDDEN];
        makeRuleInput(src, x, y, u, input);
        for (int h = 0; h < POSE_HIDDEN; h++) {
            hidden[h] = max(affineOutput(
                weights, POSE_W1, POSE_B1, u.expert, POSE_HIDDEN, PIN, h, input), 0.0f);
        }
        bool fire = fires(gid.x, gid.y, u.step, u.fireRate);
        int cell = y * u.width + x;
        int base = cell * CH;
        bool damaged = u.damageActive != 0
            && pow(float(x) - u.damageX, 2.0f) + pow(float(y) - u.damageY, 2.0f)
               <= u.damageRadius * u.damageRadius;
        for (int c = 0; c < CH; c++) {
            float reaction = affineOutput(
                weights, POSE_W2, POSE_B2, u.expert, CH, POSE_HIDDEN, c, hidden);
            dst[base + c] = damaged ? 0.0f : src[base + c] + (fire ? reaction : 0.0f);
        }
    }

    kernel void fused_edge_fields(const device float *src [[buffer(0)]],
                                  device float *flow [[buffer(1)]],
                                  device uint *selectedSlot [[buffer(2)]],
                                  device float *reaction [[buffer(3)]],
                                  const device float *weights [[buffer(4)]],
                                  constant FusedUniforms &u [[buffer(5)]],
                                  uint2 gid [[thread_position_in_grid]]) {
        if ((int)gid.x >= u.width || (int)gid.y >= u.height) return;
        int x = gid.x, y = gid.y;
        int cell = y * u.width + x;
        float input[PIN];
        float flowHidden[FLOW_HIDDEN];
        float slotHidden[FLOW_HIDDEN];
        float repairHidden[POSE_HIDDEN];
        makeRuleInput(src, x, y, u, input);
        for (int h = 0; h < FLOW_HIDDEN; h++) {
            flowHidden[h] = max(affineOutput(
                weights, FLOW_W1, FLOW_B1, u.expert, FLOW_HIDDEN, PIN, h, input), 0.0f);
            slotHidden[h] = max(affineOutput(
                weights, SLOT_W1, SLOT_B1, u.expert, FLOW_HIDDEN, PIN, h, input), 0.0f);
        }
        for (int h = 0; h < POSE_HIDDEN; h++) {
            repairHidden[h] = max(affineOutput(
                weights, REPAIR_W1, REPAIR_B1, u.expert, POSE_HIDDEN, PIN, h, input), 0.0f);
        }
        float bestLogit = -INFINITY;
        uint bestSlot = 0;
        for (int slot = 0; slot < SLOTS; slot++) {
            float logit = affineOutput(
                weights, SLOT_W2, SLOT_B2, u.expert, SLOTS, FLOW_HIDDEN,
                slot, slotHidden);
            if (logit > bestLogit) { bestLogit = logit; bestSlot = uint(slot); }
            int flowBase = (cell * SLOTS + slot) * 2;
            for (int axis = 0; axis < 2; axis++) {
                float raw = affineOutput(
                    weights, FLOW_W2, FLOW_B2, u.expert, SLOTS * 2, FLOW_HIDDEN,
                    slot * 2 + axis, flowHidden);
                flow[flowBase + axis] = tanh(raw) * u.maxFlow;
            }
        }
        selectedSlot[cell] = bestSlot;
        for (int c = 0; c < CH; c++) {
            reaction[cell * CH + c] = affineOutput(
                weights, REPAIR_W2, REPAIR_B2, u.expert, CH, POSE_HIDDEN,
                c, repairHidden);
        }
    }

    kernel void fused_edge_predict(const device float *src [[buffer(0)]],
                                   const device float *flow [[buffer(1)]],
                                   device float *predicted [[buffer(2)]],
                                   constant FusedUniforms &u [[buffer(3)]],
                                   uint3 gid [[thread_position_in_grid]]) {
        if ((int)gid.x >= u.width || (int)gid.y >= u.height || (int)gid.z >= SLOTS) return;
        int x = gid.x, y = gid.y, slot = gid.z;
        int cell = y * u.width + x;
        int flowBase = (cell * SLOTS + slot) * 2;
        float sx = float(x) - flow[flowBase];
        float sy = float(y) - flow[flowBase + 1];
        int base = ((slot * u.height + y) * u.width + x) * CH;
        for (int c = 0; c < CH; c++) {
            predicted[base + c] = bilinearState(src, sx, sy, c, u.width, u.height);
        }
    }

    kernel void fused_edge_correct(const device float *src [[buffer(0)]],
                                   const device float *flow [[buffer(1)]],
                                   const device uint *selectedSlot [[buffer(2)]],
                                   const device float *predicted [[buffer(3)]],
                                   const device float *reaction [[buffer(4)]],
                                   device float *transported [[buffer(5)]],
                                   device float *dst [[buffer(6)]],
                                   constant FusedUniforms &u [[buffer(7)]],
                                   uint2 gid [[thread_position_in_grid]]) {
        if ((int)gid.x >= u.width || (int)gid.y >= u.height) return;
        int x = gid.x, y = gid.y;
        int cell = y * u.width + x;
        int slot = int(selectedSlot[cell]);
        int flowBase = (cell * SLOTS + slot) * 2;
        float reverseX = float(x) + flow[flowBase];
        float reverseY = float(y) + flow[flowBase + 1];
        bool fire = fires(gid.x, gid.y, u.step, u.fireRate);
        bool damaged = u.damageActive != 0
            && pow(float(x) - u.damageX, 2.0f) + pow(float(y) - u.damageY, 2.0f)
               <= u.damageRadius * u.damageRadius;
        int base = cell * CH;
        for (int c = 0; c < CH; c++) {
            float forward = predicted[((slot * u.height + y) * u.width + x) * CH + c];
            float reversed = bilinearPredicted(
                predicted, slot, reverseX, reverseY, c, u.width, u.height);
            float corrected = forward + 0.5f * (src[base + c] - reversed);
            float low = INFINITY, high = -INFINITY;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && ny >= 0 && nx < u.width && ny < u.height) {
                        float value = stateCh(src, nx, ny, c, u.width, u.height);
                        low = min(low, value); high = max(high, value);
                    }
                }
            }
            float moved = clamp(corrected, low, high);
            transported[base + c] = moved;
            dst[base + c] = damaged ? 0.0f
                : moved + (fire ? reaction[base + c] : 0.0f);
        }
    }

    inline bool aliveAt(const device float *state, int x, int y,
                        int width, int height) {
        float maximum = 0.0f;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
                maximum = max(maximum, stateCh(state, x + dx, y + dy, 3, width, height));
        return maximum > 0.1f;
    }

    kernel void fused_life(const device float *pre [[buffer(0)]],
                           const device float *post [[buffer(1)]],
                           device float *output [[buffer(2)]],
                           constant FusedUniforms &u [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
        if ((int)gid.x >= u.width || (int)gid.y >= u.height) return;
        bool live = aliveAt(pre, gid.x, gid.y, u.width, u.height)
            && aliveAt(post, gid.x, gid.y, u.width, u.height);
        int base = (gid.y * u.width + gid.x) * CH;
        for (int c = 0; c < CH; c++) {
            output[base + c] = live ? clamp(post[base + c], -8.0f, 8.0f) : 0.0f;
        }
    }

    inline float4 visibleRGBA(const device float *state, int x, int y,
                              int width, int height) {
        x = clamp(x, 0, width - 1); y = clamp(y, 0, height - 1);
        int base = (y * width + x) * CH;
        float4 value = clamp(float4(state[base], state[base+1], state[base+2],
                                    state[base+3]), 0.0f, 1.0f);
        value.rgb = min(value.rgb, value.a);
        return value;
    }

    kernel void fused_render(const device float *state [[buffer(0)]],
                             constant FusedUniforms &u [[buffer(1)]],
                             texture2d<float, access::write> texture [[texture(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= texture.get_width() || gid.y >= texture.get_height()) return;
        float fx = (float(gid.x) + 0.5f) * float(u.width) / float(texture.get_width()) - 0.5f;
        float fy = (float(gid.y) + 0.5f) * float(u.height) / float(texture.get_height()) - 0.5f;
        if (u.flipX != 0) fx = float(u.width - 1) - fx;
        float4 rgba;
        if (u.crisp != 0) {
            int x0 = int(floor(fx)), y0 = int(floor(fy));
            float tx = fx - floor(fx), ty = fy - floor(fy);
            float4 value = mix(
                mix(visibleRGBA(state, x0, y0, u.width, u.height),
                    visibleRGBA(state, x0 + 1, y0, u.width, u.height), tx),
                mix(visibleRGBA(state, x0, y0 + 1, u.width, u.height),
                    visibleRGBA(state, x0 + 1, y0 + 1, u.width, u.height), tx), ty);
            float alpha = smoothstep(0.08f, 0.60f, value.a);
            rgba = float4(value.rgb * (alpha / max(value.a, 1e-4f)), alpha);
            rgba.rgb = min(rgba.rgb, rgba.a);
        } else {
            rgba = visibleRGBA(state, int(round(fx)), int(round(fy)), u.width, u.height);
        }
        texture.write(rgba, gid);
    }
    """
}
