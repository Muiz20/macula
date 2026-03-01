const std = @import("std");
const testing = std.testing;
const TokenizerMod = @import("tokenizer.zig");
const Tokenizer = TokenizerMod.Tokenizer;
const Token = TokenizerMod.Token;
const MphfMod = @import("mphf.zig");
const Mphf = MphfMod.Mphf;
const ConfusionMod = @import("confusion.zig");
const ConfusionMatrix = ConfusionMod.ConfusionMatrix;
const CandidateMod = @import("candidate.zig");
const CandidateGenerator = CandidateMod.CandidateGenerator;
const Candidate = CandidateMod.Candidate;
const GruMod = @import("gru.zig");
const Gru = GruMod.Gru;

/// Result of detection for a single token.
pub const DetectionResult = struct {
    /// Byte offset of the token in the original buffer.
    start: usize,
    /// Length of the token in bytes.
    len: usize,
    /// Detection status.
    status: Status,
    /// Best correction candidate (if any). Null-terminated in buf.
    correction: ?[CandidateMod.MAX_WORD_LEN]u8,
    correction_len: usize,
    /// Confidence score (higher = more confident in the correction).
    confidence: f32,

    pub const Status = enum {
        /// Word found in dictionary — no error.
        valid,
        /// Word not in dictionary, but a plausible correction was found.
        corrected,
        /// Word not in dictionary, no plausible correction — flagged as error.
        error_detected,
    };

    pub fn tokenSlice(self: *const DetectionResult, source: []const u8) []const u8 {
        return source[self.start..][0..self.len];
    }

    pub fn correctionSlice(self: *const DetectionResult) ?[]const u8 {
        if (self.correction) |*buf| {
            return buf[0..self.correction_len];
        }
        return null;
    }
};

/// OCR Error Detector — the top-level pipeline.
///
/// Layers:
///   1. MPHF fast-path (O(1) known-word check)
///   2. Confusion expansion → MPHF validation
///   3. GRU scoring → threshold
pub const OcrDetector = struct {
    mphf: *const Mphf,
    confusion: *const ConfusionMatrix,
    candidate_gen: CandidateGenerator,
    gru: ?*const Gru,
    /// NLL threshold. Words scoring above this are flagged as errors.
    threshold: f32,

    pub fn init(
        mphf: *const Mphf,
        confusion: *const ConfusionMatrix,
        gru: ?*const Gru,
        threshold: f32,
    ) OcrDetector {
        return .{
            .mphf = mphf,
            .confusion = confusion,
            .candidate_gen = CandidateGenerator.init(confusion, mphf),
            .gru = gru,
            .threshold = threshold,
        };
    }

    /// Detect errors in a single word (already extracted and lowercased).
    pub fn detectWord(self: *const OcrDetector, word: []const u8) DetectionResult {
        // Layer 1: MPHF fast-path
        if (self.mphf.contains(word)) {
            return DetectionResult{
                .start = 0,
                .len = word.len,
                .status = .valid,
                .correction = null,
                .correction_len = 0,
                .confidence = 1.0,
            };
        }

        // Layer 2: Confusion expansion
        var candidates: [CandidateMod.MAX_CANDIDATES]Candidate = undefined;
        const n_candidates = self.candidate_gen.generate(word, &candidates);

        if (n_candidates > 0) {
            // Find best candidate by probability
            var best_idx: usize = 0;
            var best_prob: f32 = candidates[0].probability;
            for (candidates[1..n_candidates], 1..) |c, i| {
                if (c.probability > best_prob) {
                    best_prob = c.probability;
                    best_idx = i;
                }
            }

            // Optional Layer 3: GRU re-scoring
            var confidence = best_prob;
            if (self.gru) |gru| {
                const score = gru.scoreWord(candidates[best_idx].word());
                // Blend confusion probability with GRU score
                // Lower GRU score = better → higher confidence
                confidence = best_prob * @exp(-score * 0.1);
            }

            var result = DetectionResult{
                .start = 0,
                .len = word.len,
                .status = .corrected,
                .correction = undefined,
                .correction_len = candidates[best_idx].len,
                .confidence = confidence,
            };
            result.correction = [_]u8{0} ** CandidateMod.MAX_WORD_LEN;
            @memcpy(result.correction.?[0..candidates[best_idx].len], candidates[best_idx].word());
            return result;
        }

        // Layer 3: GRU-only scoring (no confusion candidates found)
        // We do NOT accept unknown words as valid from GRU alone.
        // GRU is used to calibrate confidence on error detection in this path.
        if (self.gru) |gru| {
            const score = gru.scoreWord(word);
            return DetectionResult{
                .start = 0,
                .len = word.len,
                .status = .error_detected,
                .correction = null,
                .correction_len = 0,
                .confidence = 1.0 - @exp(-score),
            };
        }

        // Nothing worked — flag as error
        return DetectionResult{
            .start = 0,
            .len = word.len,
            .status = .error_detected,
            .correction = null,
            .correction_len = 0,
            .confidence = 0.0,
        };
    }

    /// Process a full text buffer: tokenize, detect, and return results.
    /// Results are written to the caller-provided buffer.
    /// Returns the number of results written.
    pub fn processText(
        self: *const OcrDetector,
        text: []const u8,
        results: []DetectionResult,
    ) usize {
        var tok = Tokenizer.init(text);
        var count: usize = 0;
        var lower_buf: [CandidateMod.MAX_WORD_LEN]u8 = undefined;

        while (tok.next()) |token| {
            if (count >= results.len) break;

            // Lowercase normalize
            const word = token.slice(text);
            const norm_len = @min(word.len, lower_buf.len);
            for (word[0..norm_len], 0..) |b, i| {
                lower_buf[i] = TokenizerMod.toLowerAscii(b);
            }
            const lower_word = lower_buf[0..norm_len];

            var result = self.detectWord(lower_word);
            // Adjust offsets to point into original text
            result.start = token.start;
            result.len = token.len;

            results[count] = result;
            count += 1;
        }

        return count;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "detector — known word returns valid" {
    const words = [_][]const u8{ "hello", "world", "the", "cat" };
    var mphf = try Mphf.build(testing.allocator, &words);
    defer mphf.deinit(testing.allocator);

    const cm = ConfusionMatrix.init(&ConfusionMod.DEFAULT_PAIRS);
    const det = OcrDetector.init(&mphf, &cm, null, 5.0);

    const result = det.detectWord("hello");
    try testing.expectEqual(DetectionResult.Status.valid, result.status);
    try testing.expectEqual(@as(f32, 1.0), result.confidence);
}

test "detector — unknown word without confusion match returns error" {
    const words = [_][]const u8{ "hello", "world" };
    var mphf = try Mphf.build(testing.allocator, &words);
    defer mphf.deinit(testing.allocator);

    const cm = ConfusionMatrix.init(&[_]ConfusionMod.ConfusionPair{});
    const det = OcrDetector.init(&mphf, &cm, null, 5.0);

    const result = det.detectWord("xyzzy");
    try testing.expectEqual(DetectionResult.Status.error_detected, result.status);
}

test "detector — confusion corrects rn to m" {
    // Dictionary has "corn" but not "com"
    const words = [_][]const u8{ "corn", "dog", "cat" };
    var mphf = try Mphf.build(testing.allocator, &words);
    defer mphf.deinit(testing.allocator);

    const pairs = [_]ConfusionMod.ConfusionPair{
        .{ .from = "m", .to = "rn", .probability = 0.60 },
    };
    const cm = ConfusionMatrix.init(&pairs);
    const det = OcrDetector.init(&mphf, &cm, null, 5.0);

    // "com" → apply m→rn → "corn" is in dict → corrected
    const result = det.detectWord("com");
    try testing.expectEqual(DetectionResult.Status.corrected, result.status);
    try testing.expectEqualStrings("corn", result.correctionSlice().?);
}

test "detector — processText full pipeline" {
    const words = [_][]const u8{ "the", "quick", "brown", "fox" };
    var mphf = try Mphf.build(testing.allocator, &words);
    defer mphf.deinit(testing.allocator);

    const cm = ConfusionMatrix.init(&[_]ConfusionMod.ConfusionPair{});
    const det = OcrDetector.init(&mphf, &cm, null, 5.0);

    var results: [16]DetectionResult = undefined;
    const text = "the quick zzzz fox";
    const n = det.processText(text, &results);

    try testing.expectEqual(@as(usize, 4), n);
    try testing.expectEqual(DetectionResult.Status.valid, results[0].status); // "the"
    try testing.expectEqual(DetectionResult.Status.valid, results[1].status); // "quick"
    try testing.expectEqual(DetectionResult.Status.error_detected, results[2].status); // "zzzz"
    try testing.expectEqual(DetectionResult.Status.valid, results[3].status); // "fox"

    // Verify byte offsets
    try testing.expectEqualStrings("zzzz", results[2].tokenSlice(text));
}

test "detector — processText normalizes case" {
    const words = [_][]const u8{ "hello", "world" };
    var mphf = try Mphf.build(testing.allocator, &words);
    defer mphf.deinit(testing.allocator);

    const cm = ConfusionMatrix.init(&[_]ConfusionMod.ConfusionPair{});
    const det = OcrDetector.init(&mphf, &cm, null, 5.0);

    var results: [8]DetectionResult = undefined;
    const n = det.processText("HELLO World", &results);

    try testing.expectEqual(@as(usize, 2), n);
    try testing.expectEqual(DetectionResult.Status.valid, results[0].status);
    try testing.expectEqual(DetectionResult.Status.valid, results[1].status);
}

test "detector — processText empty input" {
    const words = [_][]const u8{"hello"};
    var mphf = try Mphf.build(testing.allocator, &words);
    defer mphf.deinit(testing.allocator);

    const cm = ConfusionMatrix.init(&[_]ConfusionMod.ConfusionPair{});
    const det = OcrDetector.init(&mphf, &cm, null, 5.0);

    var results: [8]DetectionResult = undefined;
    const n = det.processText("", &results);
    try testing.expectEqual(@as(usize, 0), n);
}

test "detector — gru does not accept unknown word without correction" {
    const words = [_][]const u8{ "hello", "world" };
    var mphf = try Mphf.build(testing.allocator, &words);
    defer mphf.deinit(testing.allocator);

    const cm = ConfusionMatrix.init(&[_]ConfusionMod.ConfusionPair{});

    const w_ih = [_]i8{0} ** (3 * 1 * 256);
    const w_hh = [_]i8{0} ** (3 * 1 * 1);
    const b_ih = [_]i8{0} ** (3 * 1);
    const b_hh = [_]i8{0} ** (3 * 1);
    const w_out = [_]i8{0} ** (256 * 1);
    const b_out = [_]i8{0} ** 256;

    const gru = Gru{
        .config = .{ .input_size = 256, .hidden_size = 1 },
        .w_ih = .{ .data = &w_ih, .scale = 0.01 },
        .w_hh = .{ .data = &w_hh, .scale = 0.01 },
        .b_ih = .{ .data = &b_ih, .scale = 0.01 },
        .b_hh = .{ .data = &b_hh, .scale = 0.01 },
        .w_out = .{ .data = &w_out, .scale = 0.01 },
        .b_out = .{ .data = &b_out, .scale = 0.01 },
    };

    const det = OcrDetector.init(&mphf, &cm, &gru, 10.0);
    const result = det.detectWord("xyzzy");
    try testing.expectEqual(DetectionResult.Status.error_detected, result.status);
}
