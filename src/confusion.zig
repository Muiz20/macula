const std = @import("std");
const testing = std.testing;

/// A single OCR confusion rule: a source pattern that is visually confused
/// with a replacement pattern, along with a probability score.
pub const ConfusionPair = struct {
    /// Source byte pattern (what OCR might produce), e.g. "rn"
    from: []const u8,
    /// Target byte pattern (what was intended), e.g. "m"
    to: []const u8,
    /// Probability of this confusion (0.0 – 1.0). Higher = more likely.
    probability: f32,
};

/// OCR Confusion Matrix — a collection of known visual confusion patterns.
/// Read-only after construction. Zero heap allocation at query time.
pub const ConfusionMatrix = struct {
    pairs: []const ConfusionPair,

    /// Initialize from a static slice of pairs.
    pub fn init(pairs: []const ConfusionPair) ConfusionMatrix {
        return .{ .pairs = pairs };
    }

    /// Return all confusion pairs where `from` matches a substring at the
    /// given position in the word. Writes matches to the caller-provided buffer.
    /// Returns the number of matches written.
    pub fn matchAt(
        self: *const ConfusionMatrix,
        word: []const u8,
        pos: usize,
        out: []ConfusionPair,
    ) usize {
        var count: usize = 0;
        for (self.pairs) |pair| {
            if (pos + pair.from.len <= word.len) {
                if (std.mem.eql(u8, word[pos..][0..pair.from.len], pair.from)) {
                    if (count < out.len) {
                        out[count] = pair;
                        count += 1;
                    }
                }
            }
        }
        return count;
    }

    /// Count how many pairs are registered.
    pub fn pairCount(self: *const ConfusionMatrix) usize {
        return self.pairs.len;
    }
};

/// Default OCR confusion pairs — common visual confusions in printed text.
pub const DEFAULT_PAIRS = [_]ConfusionPair{
    .{ .from = "rn", .to = "m", .probability = 0.85 },
    .{ .from = "m", .to = "rn", .probability = 0.60 },
    .{ .from = "cl", .to = "d", .probability = 0.70 },
    .{ .from = "d", .to = "cl", .probability = 0.50 },
    .{ .from = "0", .to = "O", .probability = 0.80 },
    .{ .from = "O", .to = "0", .probability = 0.75 },
    .{ .from = "1", .to = "l", .probability = 0.80 },
    .{ .from = "l", .to = "1", .probability = 0.65 },
    .{ .from = "1", .to = "I", .probability = 0.70 },
    .{ .from = "I", .to = "1", .probability = 0.60 },
    .{ .from = "vv", .to = "w", .probability = 0.75 },
    .{ .from = "w", .to = "vv", .probability = 0.55 },
    .{ .from = "li", .to = "h", .probability = 0.40 },
    .{ .from = "ii", .to = "u", .probability = 0.35 },
    .{ .from = "nn", .to = "m", .probability = 0.30 },
    .{ .from = "ri", .to = "n", .probability = 0.30 },
};

// ============================================================================
// Tests
// ============================================================================

test "confusion matrix finds rn at position" {
    const cm = ConfusionMatrix.init(&DEFAULT_PAIRS);
    var out: [8]ConfusionPair = undefined;

    // "corn" — at position 2, "rn" matches
    const n = cm.matchAt("corn", 2, &out);
    try testing.expect(n >= 1);
    try testing.expectEqualStrings("rn", out[0].from);
    try testing.expectEqualStrings("m", out[0].to);
}

test "confusion matrix no match at wrong position" {
    const cm = ConfusionMatrix.init(&DEFAULT_PAIRS);
    var out: [8]ConfusionPair = undefined;

    // "corn" — at position 0, no confusion starts with "co"
    const n = cm.matchAt("corn", 0, &out);
    try testing.expectEqual(@as(usize, 0), n);
}

test "confusion matrix matches digit confusions" {
    const cm = ConfusionMatrix.init(&DEFAULT_PAIRS);
    var out: [8]ConfusionPair = undefined;

    // "f00d" — at position 1, "0" matches
    const n = cm.matchAt("f00d", 1, &out);
    try testing.expect(n >= 1);
    var found_o = false;
    for (out[0..n]) |p| {
        if (std.mem.eql(u8, p.to, "O")) found_o = true;
    }
    try testing.expect(found_o);
}

test "confusion matrix pair count" {
    const cm = ConfusionMatrix.init(&DEFAULT_PAIRS);
    try testing.expectEqual(@as(usize, 16), cm.pairCount());
}

test "confusion matrix empty" {
    const cm = ConfusionMatrix.init(&[_]ConfusionPair{});
    var out: [8]ConfusionPair = undefined;
    const n = cm.matchAt("hello", 0, &out);
    try testing.expectEqual(@as(usize, 0), n);
}

test "confusion matrix boundary — pattern at end of word" {
    const cm = ConfusionMatrix.init(&DEFAULT_PAIRS);
    var out: [8]ConfusionPair = undefined;

    // "barn" — "rn" at position 2 (len=4, pos+2=4 == len, valid)
    const n = cm.matchAt("barn", 2, &out);
    try testing.expect(n >= 1);
    try testing.expectEqualStrings("rn", out[0].from);
}

test "confusion matrix boundary — pattern would exceed word" {
    const cm = ConfusionMatrix.init(&DEFAULT_PAIRS);
    var out: [8]ConfusionPair = undefined;

    // "a" — can't fit "rn" at position 0
    const n = cm.matchAt("a", 0, &out);
    // Only single-char confusions could match
    for (out[0..n]) |p| {
        try testing.expect(p.from.len <= 1);
    }
}
