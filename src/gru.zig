const std = @import("std");
const testing = std.testing;

/// Quantized weight value: int8 with a shared scale factor.
/// Dequantized value = @as(f32, raw) * scale
pub const QuantizedWeights = struct {
    data: []const i8,
    scale: f32,

    pub fn get(self: *const QuantizedWeights, idx: usize) f32 {
        return @as(f32, @floatFromInt(self.data[idx])) * self.scale;
    }
};

/// GRU cell dimensions.
pub const GruConfig = struct {
    input_size: u32, // typically 128 for ASCII byte input
    hidden_size: u32, // e.g. 64 or 128
};

/// A single-layer GRU for character-level language modeling.
/// Forward-pass only (inference). Weights are int8 quantized.
///
/// GRU equations:
///   z = sigmoid(W_iz * x + W_hz * h)     -- update gate
///   r = sigmoid(W_ir * x + W_hr * h)     -- reset gate
///   n = tanh(W_in * x + r * (W_hn * h))  -- new gate
///   h' = (1 - z) * n + z * h             -- hidden state
pub const Gru = struct {
    config: GruConfig,

    // Input weights (3 * hidden_size × input_size): [W_iz | W_ir | W_in]
    w_ih: QuantizedWeights,
    // Hidden weights (3 * hidden_size × hidden_size): [W_hz | W_hr | W_hn]
    w_hh: QuantizedWeights,
    // Input biases (3 * hidden_size): [b_iz | b_ir | b_in]
    b_ih: QuantizedWeights,
    // Hidden biases (3 * hidden_size): [b_hz | b_hr | b_hn]
    b_hh: QuantizedWeights,
    // Output projection: hidden_size × input_size (for predicting next char logits)
    w_out: QuantizedWeights,
    b_out: QuantizedWeights,

    const MAX_HIDDEN: usize = 256;
    const MAX_INPUT: usize = 256;
    const ASCII_OFFSET: usize = 32;

    fn byteToIndex(self: *const Gru, byte: u8) usize {
        const is: usize = self.config.input_size;
        if (is == 0) return 0;

        // For printable-ASCII vocabulary (common case: 96), map by offset.
        if (is == 96) {
            const shifted: isize = @as(isize, @intCast(byte)) - @as(isize, ASCII_OFFSET);
            if (shifted <= 0) return 0;
            const idx: usize = @intCast(shifted);
            return if (idx >= is) is - 1 else idx;
        }

        // For larger vocabularies (e.g. 256), use direct byte index with clamp.
        const idx = @as(usize, byte);
        return if (idx >= is) is - 1 else idx;
    }

    /// Compute one GRU cell step: given input byte and hidden state, produce new hidden state.
    /// All computation on stack — zero allocation.
    pub fn step(
        self: *const Gru,
        input_byte: u8,
        hidden_in: []const f32,
        hidden_out: []f32,
    ) void {
        const hs: usize = self.config.hidden_size;
        const is: usize = self.config.input_size;

        // One-hot input (conceptually). We just index the column directly.
        const x_idx: usize = self.byteToIndex(input_byte);

        // Compute gates: z, r, n
        var z: [MAX_HIDDEN]f32 = undefined;
        var r: [MAX_HIDDEN]f32 = undefined;
        var n: [MAX_HIDDEN]f32 = undefined;

        for (0..hs) |j| {
            // z_j = sigmoid(W_iz[j, x] + b_iz[j] + W_hz[j, :] · h + b_hz[j])
            const iz_idx = j * is + x_idx;
            const z_val = self.w_ih.get(iz_idx) + self.b_ih.get(j);
            var hz_val = self.b_hh.get(j);
            for (0..hs) |k| {
                hz_val += self.w_hh.get(j * hs + k) * hidden_in[k];
            }
            z[j] = sigmoid(z_val + hz_val);

            // r_j = sigmoid(W_ir[j, x] + b_ir[j] + W_hr[j, :] · h + b_hr[j])
            const ir_idx = (hs + j) * is + x_idx;
            const r_val = self.w_ih.get(ir_idx) + self.b_ih.get(hs + j);
            var hr_val = self.b_hh.get(hs + j);
            for (0..hs) |k| {
                hr_val += self.w_hh.get((hs + j) * hs + k) * hidden_in[k];
            }
            r[j] = sigmoid(r_val + hr_val);

            // n_j = tanh(W_in[j, x] + b_in[j] + r_j * (W_hn[j, :] · h + b_hn[j]))
            const in_idx = (2 * hs + j) * is + x_idx;
            const n_val = self.w_ih.get(in_idx) + self.b_ih.get(2 * hs + j);
            var hn_val = self.b_hh.get(2 * hs + j);
            for (0..hs) |k| {
                hn_val += self.w_hh.get((2 * hs + j) * hs + k) * hidden_in[k];
            }
            n[j] = tanh_f32(n_val + r[j] * hn_val);

            // h'_j = (1 - z_j) * n_j + z_j * h_j
            hidden_out[j] = (1.0 - z[j]) * n[j] + z[j] * hidden_in[j];
        }
    }

    /// Score a word by computing total negative log-likelihood.
    /// Lower score = more natural word. Higher score = more suspicious.
    /// Returns the average NLL per character.
    pub fn scoreWord(self: *const Gru, word: []const u8) f32 {
        const hs: usize = self.config.hidden_size;

        // Hidden state buffers (ping-pong)
        var h0: [MAX_HIDDEN]f32 = [_]f32{0.0} ** MAX_HIDDEN;
        var h1: [MAX_HIDDEN]f32 = undefined;

        var total_nll: f32 = 0.0;

        for (word, 0..) |char, i| {
            const h_in = if (i % 2 == 0) h0[0..hs] else h1[0..hs];
            const h_out = if (i % 2 == 0) h1[0..hs] else h0[0..hs];

            self.step(char, h_in, h_out);

            // If there's a next character, compute its log-probability from h_out
            if (i + 1 < word.len) {
                const next_char = word[i + 1];
                const logp = self.outputLogProb(h_out, next_char);
                total_nll -= logp;
            }
        }

        if (word.len <= 1) return 0.0;
        return total_nll / @as(f32, @floatFromInt(word.len - 1));
    }

    /// Compute log-probability of a character given the hidden state.
    fn outputLogProb(self: *const Gru, hidden: []const f32, target: u8) f32 {
        const is: usize = self.config.input_size;
        const hs: usize = self.config.hidden_size;
        const target_idx = self.byteToIndex(target);

        // Compute logits for all characters
        var logits: [MAX_INPUT]f32 = undefined;
        for (0..is) |c| {
            var val = self.b_out.get(c);
            for (0..hs) |k| {
                val += self.w_out.get(c * hs + k) * hidden[k];
            }
            logits[c] = val;
        }

        // Log-softmax: log(exp(logit[target]) / sum(exp(logit[:])))
        // = logit[target] - log(sum(exp(logit[:])))
        // Use max subtraction for numerical stability
        var max_logit: f32 = logits[0];
        for (logits[1..is]) |l| {
            if (l > max_logit) max_logit = l;
        }
        var sum_exp: f32 = 0.0;
        for (logits[0..is]) |l| {
            sum_exp += @exp(l - max_logit);
        }
        return logits[target_idx] - max_logit - @log(sum_exp);
    }

    fn sigmoid(x: f32) f32 {
        return 1.0 / (1.0 + @exp(-x));
    }

    fn tanh_f32(x: f32) f32 {
        return std.math.tanh(x);
    }
};

// ============================================================================
// Tests
// ============================================================================

// Create a tiny GRU (input_size=4, hidden_size=2) with known weights for testing.
fn makeTestGru() Gru {
    // w_ih: 3*hidden_size × input_size = 3*2*4 = 24 weights
    const w_ih_data = [_]i8{
        // W_iz (2×4)
        10, 5,  -3, 2,
        -1, 8,  4,  -5,
        // W_ir (2×4)
        3,  -2, 7,  1,
        5,  3,  -4, 6,
        // W_in (2×4)
        -6, 4,  2,  -3,
        1,  -7, 5,  8,
    };
    // w_hh: 3*hidden_size × hidden_size = 3*2*2 = 12 weights
    const w_hh_data = [_]i8{
        // W_hz (2×2)
        4,  -2,
        3,  5,
        // W_hr (2×2)
        -1, 6,
        2,  -3,
        // W_hn (2×2)
        5,  1,
        -2, 4,
    };
    // Biases: 3*hidden_size = 6 each
    const b_ih_data = [_]i8{ 1, -1, 2, 0, -2, 3 };
    const b_hh_data = [_]i8{ 0, 1, -1, 2, 1, -1 };

    // Output projection: input_size × hidden_size = 4×2 = 8
    const w_out_data = [_]i8{ 5, -3, -2, 7, 4, 1, -5, 3 };
    const b_out_data = [_]i8{ 1, -1, 0, 2 };

    const scale: f32 = 0.1;
    return Gru{
        .config = .{ .input_size = 4, .hidden_size = 2 },
        .w_ih = .{ .data = &w_ih_data, .scale = scale },
        .w_hh = .{ .data = &w_hh_data, .scale = scale },
        .b_ih = .{ .data = &b_ih_data, .scale = scale },
        .b_hh = .{ .data = &b_hh_data, .scale = scale },
        .w_out = .{ .data = &w_out_data, .scale = scale },
        .b_out = .{ .data = &b_out_data, .scale = scale },
    };
}

test "gru step produces finite output" {
    const gru = makeTestGru();
    const h_in = [_]f32{ 0.0, 0.0 };
    var h_out: [2]f32 = undefined;

    gru.step(1, &h_in, &h_out);

    try testing.expect(std.math.isFinite(h_out[0]));
    try testing.expect(std.math.isFinite(h_out[1]));
}

test "gru step changes hidden state" {
    const gru = makeTestGru();
    const h_in = [_]f32{ 0.0, 0.0 };
    var h_out: [2]f32 = undefined;

    gru.step(2, &h_in, &h_out);

    // At least one hidden unit should change (with non-zero weights)
    const changed = (h_out[0] != 0.0) or (h_out[1] != 0.0);
    try testing.expect(changed);
}

test "gru step deterministic" {
    const gru = makeTestGru();
    const h_in = [_]f32{ 0.5, -0.3 };
    var h_out1: [2]f32 = undefined;
    var h_out2: [2]f32 = undefined;

    gru.step(3, &h_in, &h_out1);
    gru.step(3, &h_in, &h_out2);

    try testing.expectEqual(h_out1[0], h_out2[0]);
    try testing.expectEqual(h_out1[1], h_out2[1]);
}

test "gru hidden state stays bounded" {
    const gru = makeTestGru();
    var h: [2]f32 = [_]f32{ 0.0, 0.0 };
    var h_next: [2]f32 = undefined;

    // Run 100 steps — hidden state should remain bounded due to tanh
    for (0..100) |i| {
        const input: u8 = @truncate(i % 4);
        if (i % 2 == 0) {
            gru.step(input, &h, &h_next);
        } else {
            gru.step(input, &h_next, &h);
        }
    }
    const final = h;
    for (final[0..2]) |v| {
        try testing.expect(std.math.isFinite(v));
        try testing.expect(@abs(v) < 10.0);
    }
}

test "gru scoreWord returns finite value" {
    const gru = makeTestGru();
    // Using bytes 0-3 since input_size=4
    const word = [_]u8{ 1, 2, 3, 0, 1 };
    const score = gru.scoreWord(&word);
    try testing.expect(std.math.isFinite(score));
    try testing.expect(score >= 0.0); // NLL is non-negative
}

test "gru scoreWord different words get different scores" {
    const gru = makeTestGru();
    const word1 = [_]u8{ 0, 1, 2, 3 };
    const word2 = [_]u8{ 3, 2, 1, 0 };
    const s1 = gru.scoreWord(&word1);
    const s2 = gru.scoreWord(&word2);
    // Different sequences should (almost certainly) get different NLL scores
    try testing.expect(s1 != s2);
}

test "gru scoreWord single char returns zero" {
    const gru = makeTestGru();
    const word = [_]u8{1};
    const score = gru.scoreWord(&word);
    try testing.expectEqual(@as(f32, 0.0), score);
}
