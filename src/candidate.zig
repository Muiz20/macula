const std = @import("std");
const testing = std.testing;
const MphfMod = @import("mphf.zig");
const Mphf = MphfMod.Mphf;
const ConfusionMod = @import("confusion.zig");
const ConfusionMatrix = ConfusionMod.ConfusionMatrix;
const ConfusionPair = ConfusionMod.ConfusionPair;

/// A generated correction candidate.
pub const Candidate = struct {
    /// The corrected word (stored in a fixed buffer).
    buf: [MAX_WORD_LEN]u8,
    len: usize,
    /// The confusion pair that generated this candidate.
    probability: f32,

    pub fn word(self: *const Candidate) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const MAX_WORD_LEN = 64;
pub const MAX_CANDIDATES = 64;

const MAX_INTERMEDIATE = 24;
const SECOND_EDIT_PENALTY: f32 = 0.90;

const Intermediate = struct {
    buf: [MAX_WORD_LEN]u8,
    len: usize,
    score: f32,

    pub fn word(self: *const Intermediate) []const u8 {
        return self.buf[0..self.len];
    }
};

/// Candidate Generator: given a word that failed MPHF lookup, apply
/// confusion patterns to generate plausible corrections, then validate
/// each against the MPHF dictionary.
pub const CandidateGenerator = struct {
    confusion: *const ConfusionMatrix,
    mphf: *const Mphf,

    pub fn init(confusion: *const ConfusionMatrix, mphf: *const Mphf) CandidateGenerator {
        return .{ .confusion = confusion, .mphf = mphf };
    }

    fn hasCandidate(out: []const Candidate, count: usize, word: []const u8) bool {
        for (out[0..count]) |c| {
            if (std.mem.eql(u8, c.word(), word)) return true;
        }
        return false;
    }

    fn addValidated(
        self: *const CandidateGenerator,
        word: []const u8,
        probability: f32,
        out: []Candidate,
        count: *usize,
    ) void {
        if (count.* >= out.len) return;
        if (!self.mphf.contains(word)) return;
        if (hasCandidate(out, count.*, word)) return;

        var cand: Candidate = undefined;
        cand.len = word.len;
        cand.probability = probability;
        @memcpy(cand.buf[0..cand.len], word);

        out[count.*] = cand;
        count.* += 1;
    }

    fn addIntermediate(
        word: []const u8,
        score: f32,
        beam: []Intermediate,
        beam_count: *usize,
    ) void {
        // Deduplicate by word; keep the highest score.
        for (beam[0..beam_count.*]) |*b| {
            if (std.mem.eql(u8, b.word(), word)) {
                if (score > b.score) b.score = score;
                return;
            }
        }

        if (beam_count.* < beam.len) {
            beam[beam_count.*].len = word.len;
            beam[beam_count.*].score = score;
            @memcpy(beam[beam_count.*].buf[0..word.len], word);
            beam_count.* += 1;
            return;
        }

        // Beam full: replace the current worst element if this one is better.
        var worst_idx: usize = 0;
        var worst_score: f32 = beam[0].score;
        for (beam[1..beam_count.*], 1..) |b, i| {
            if (b.score < worst_score) {
                worst_score = b.score;
                worst_idx = i;
            }
        }

        if (score > worst_score) {
            beam[worst_idx].len = word.len;
            beam[worst_idx].score = score;
            @memcpy(beam[worst_idx].buf[0..word.len], word);
        }
    }

    /// Generate correction candidates for the given word.
    /// Returns the number of valid candidates written to `out`.
    /// Only includes candidates that pass MPHF dictionary check.
    pub fn generate(
        self: *const CandidateGenerator,
        word: []const u8,
        out: []Candidate,
    ) usize {
        if (word.len == 0 or word.len > MAX_WORD_LEN) return 0;

        var count: usize = 0;
        var beam: [MAX_INTERMEDIATE]Intermediate = undefined;
        var beam_count: usize = 0;

        var pos: usize = 0;

        // Pass 1: single-edit expansion.
        while (pos < word.len and count < out.len) {
            // Try confusion patterns at this position
            var matches: [16]ConfusionPair = undefined;
            const num_matches = self.confusion.matchAt(word, pos, &matches);

            for (matches[0..num_matches]) |pair| {
                // Build candidate: prefix + replacement + suffix
                const prefix_len = pos;
                const suffix_start = pos + pair.from.len;
                const suffix_len = word.len - suffix_start;
                const new_len = prefix_len + pair.to.len + suffix_len;

                if (new_len > MAX_WORD_LEN or new_len == 0) continue;

                var buf: [MAX_WORD_LEN]u8 = undefined;

                // Copy prefix
                @memcpy(buf[0..prefix_len], word[0..prefix_len]);
                // Copy replacement
                @memcpy(buf[prefix_len..][0..pair.to.len], pair.to);
                // Copy suffix
                @memcpy(buf[prefix_len + pair.to.len ..][0..suffix_len], word[suffix_start..][0..suffix_len]);

                const cand_word = buf[0..new_len];
                self.addValidated(cand_word, pair.probability, out, &count);
                addIntermediate(cand_word, pair.probability, &beam, &beam_count);
            }

            pos += 1;
        }

        // Pass 2: bounded two-edit expansion from beam candidates.
        for (beam[0..beam_count]) |intermediate| {
            if (count >= out.len) break;

            var pos2: usize = 0;
            while (pos2 < intermediate.len and count < out.len) {
                var matches2: [16]ConfusionPair = undefined;
                const num_matches2 = self.confusion.matchAt(intermediate.word(), pos2, &matches2);

                for (matches2[0..num_matches2]) |pair2| {
                    const prefix_len = pos2;
                    const suffix_start = pos2 + pair2.from.len;
                    const suffix_len = intermediate.len - suffix_start;
                    const new_len = prefix_len + pair2.to.len + suffix_len;

                    if (new_len > MAX_WORD_LEN or new_len == 0) continue;

                    var buf2: [MAX_WORD_LEN]u8 = undefined;
                    @memcpy(buf2[0..prefix_len], intermediate.word()[0..prefix_len]);
                    @memcpy(buf2[prefix_len..][0..pair2.to.len], pair2.to);
                    @memcpy(buf2[prefix_len + pair2.to.len ..][0..suffix_len], intermediate.word()[suffix_start..][0..suffix_len]);

                    const cand2_word = buf2[0..new_len];
                    const score2 = intermediate.score * pair2.probability * SECOND_EDIT_PENALTY;
                    self.addValidated(cand2_word, score2, out, &count);
                }

                pos2 += 1;
            }
        }

        return count;
    }

    /// Generate candidates without MPHF validation (for testing confusion expansion alone).
    pub fn generateUnvalidated(
        self: *const CandidateGenerator,
        word: []const u8,
        out: []Candidate,
    ) usize {
        _ = self.mphf; // unused in this path
        if (word.len == 0 or word.len > MAX_WORD_LEN) return 0;

        var count: usize = 0;
        var pos: usize = 0;

        while (pos < word.len and count < out.len) {
            var matches: [16]ConfusionPair = undefined;
            const num_matches = self.confusion.matchAt(word, pos, &matches);

            for (matches[0..num_matches]) |pair| {
                const prefix_len = pos;
                const suffix_start = pos + pair.from.len;
                const suffix_len = word.len - suffix_start;
                const new_len = prefix_len + pair.to.len + suffix_len;

                if (new_len > MAX_WORD_LEN or new_len == 0) continue;
                if (count >= out.len) break;

                var cand: Candidate = undefined;
                cand.len = new_len;
                cand.probability = pair.probability;

                @memcpy(cand.buf[0..prefix_len], word[0..prefix_len]);
                @memcpy(cand.buf[prefix_len..][0..pair.to.len], pair.to);
                @memcpy(cand.buf[prefix_len + pair.to.len ..][0..suffix_len], word[suffix_start..][0..suffix_len]);

                out[count] = cand;
                count += 1;
            }

            pos += 1;
        }

        return count;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "candidate generator — rn to m expansion" {
    const pairs = [_]ConfusionPair{
        .{ .from = "rn", .to = "m", .probability = 0.85 },
    };
    const cm = ConfusionMatrix.init(&pairs);
    const empty_mphf = Mphf{ .seeds = &.{}, .fingerprints = &.{}, .n = 0, .bucket_size = 4 };
    const gen = CandidateGenerator.init(&cm, &empty_mphf);

    var out: [MAX_CANDIDATES]Candidate = undefined;
    const n = gen.generateUnvalidated("corn", &out);

    try testing.expect(n >= 1);
    // "corn" with rn→m at pos 2 should yield "com"
    var found_com = false;
    for (out[0..n]) |c| {
        if (std.mem.eql(u8, c.word(), "com")) found_com = true;
    }
    try testing.expect(found_com);
}

test "candidate generator — m to rn expansion" {
    const pairs = [_]ConfusionPair{
        .{ .from = "m", .to = "rn", .probability = 0.60 },
    };
    const cm = ConfusionMatrix.init(&pairs);
    const empty_mphf = Mphf{ .seeds = &.{}, .fingerprints = &.{}, .n = 0, .bucket_size = 4 };
    const gen = CandidateGenerator.init(&cm, &empty_mphf);

    var out: [MAX_CANDIDATES]Candidate = undefined;
    const n = gen.generateUnvalidated("com", &out);

    try testing.expect(n >= 1);
    var found_corn = false;
    for (out[0..n]) |c| {
        if (std.mem.eql(u8, c.word(), "corn")) found_corn = true;
    }
    // "com" → position 2 "m"→"rn" → "corn"
    try testing.expect(found_corn);
}

test "candidate generator — validated against mphf" {
    // Build an MPHF with "corn" in the dictionary
    const words = [_][]const u8{ "corn", "dog", "cat", "the" };
    var mphf = try Mphf.build(testing.allocator, &words);
    defer mphf.deinit(testing.allocator);

    const pairs = [_]ConfusionPair{
        .{ .from = "rn", .to = "m", .probability = 0.85 },
        .{ .from = "m", .to = "rn", .probability = 0.60 },
    };
    const cm = ConfusionMatrix.init(&pairs);
    const gen = CandidateGenerator.init(&cm, &mphf);

    var out: [MAX_CANDIDATES]Candidate = undefined;
    // "com" is not in dict, but "corn" is — so m→rn should yield "corn" (validated)
    const n = gen.generate("com", &out);

    try testing.expect(n >= 1);
    var found_corn = false;
    for (out[0..n]) |c| {
        if (std.mem.eql(u8, c.word(), "corn")) found_corn = true;
    }
    try testing.expect(found_corn);
}

test "candidate generator — no matches for clean word" {
    const pairs = [_]ConfusionPair{
        .{ .from = "rn", .to = "m", .probability = 0.85 },
    };
    const cm = ConfusionMatrix.init(&pairs);
    const empty_mphf = Mphf{ .seeds = &.{}, .fingerprints = &.{}, .n = 0, .bucket_size = 4 };
    const gen = CandidateGenerator.init(&cm, &empty_mphf);

    var out: [MAX_CANDIDATES]Candidate = undefined;
    const n = gen.generateUnvalidated("hello", &out);
    // "hello" has no "rn" substring
    try testing.expectEqual(@as(usize, 0), n);
}

test "candidate generator — empty word" {
    const cm = ConfusionMatrix.init(&ConfusionMod.DEFAULT_PAIRS);
    const empty_mphf = Mphf{ .seeds = &.{}, .fingerprints = &.{}, .n = 0, .bucket_size = 4 };
    const gen = CandidateGenerator.init(&cm, &empty_mphf);

    var out: [MAX_CANDIDATES]Candidate = undefined;
    const n = gen.generateUnvalidated("", &out);
    try testing.expectEqual(@as(usize, 0), n);
}

test "candidate generator — two-edit beam recovers brown from br0vvn" {
    const words = [_][]const u8{ "brown", "quick", "fox" };
    var mphf = try Mphf.build(testing.allocator, &words);
    defer mphf.deinit(testing.allocator);

    const pairs = [_]ConfusionPair{
        .{ .from = "0", .to = "o", .probability = 0.90 },
        .{ .from = "vv", .to = "w", .probability = 0.85 },
    };
    const cm = ConfusionMatrix.init(&pairs);
    const gen = CandidateGenerator.init(&cm, &mphf);

    var out: [MAX_CANDIDATES]Candidate = undefined;
    const n = gen.generate("br0vvn", &out);

    try testing.expect(n >= 1);
    var found_brown = false;
    for (out[0..n]) |c| {
        if (std.mem.eql(u8, c.word(), "brown")) found_brown = true;
    }
    try testing.expect(found_brown);
}
