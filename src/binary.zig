const std = @import("std");
const testing = std.testing;

/// Binary artifact format for the OCR detector.
///
/// Layout:
///   [Header]              — 32 bytes
///   [MPHF Seeds]          — header.mphf_seed_count × 4 bytes
///   [MPHF Fingerprints]   — header.mphf_key_count × 2 bytes
///   [Confusion Pairs]     — header.confusion_count × sizeof(BinaryConfusionPair)
///   [GRU Weights]         — header.gru_weight_bytes bytes
///
/// All multi-byte values are little-endian.
pub const MAGIC: u32 = 0x4F435244; // "OCRD"
pub const VERSION: u32 = 1;

pub const Header = extern struct {
    magic: u32 align(1),
    version: u32 align(1),
    /// Number of MPHF seed entries (= number of buckets)
    mphf_seed_count: u32 align(1),
    /// Number of MPHF keys (= number of fingerprints)
    mphf_key_count: u32 align(1),
    /// MPHF bucket size parameter
    mphf_bucket_size: u32 align(1),
    /// Number of confusion pairs
    confusion_count: u32 align(1),
    /// Total bytes of GRU weight data
    gru_weight_bytes: u32 align(1),
    /// Reserved for future use
    _reserved: u32 align(1),
};

pub const BinaryConfusionPair = extern struct {
    from_len: u8 align(1),
    from: [8]u8 align(1),
    to_len: u8 align(1),
    to: [8]u8 align(1),
    /// Probability × 10000, stored as u16
    probability_u16: u16 align(1),

    pub fn fromConfusionPair(from: []const u8, to: []const u8, prob: f32) BinaryConfusionPair {
        var bp: BinaryConfusionPair = std.mem.zeroes(BinaryConfusionPair);
        bp.from_len = @intCast(@min(from.len, 8));
        @memcpy(bp.from[0..bp.from_len], from[0..bp.from_len]);
        bp.to_len = @intCast(@min(to.len, 8));
        @memcpy(bp.to[0..bp.to_len], to[0..bp.to_len]);
        bp.probability_u16 = @intFromFloat(prob * 10000.0);
        return bp;
    }

    pub fn probability(self: *const BinaryConfusionPair) f32 {
        return @as(f32, @floatFromInt(self.probability_u16)) / 10000.0;
    }
};

/// Pack a header into bytes.
pub fn packHeader(header: Header) [@sizeOf(Header)]u8 {
    return @bitCast(header);
}

/// Unpack a header from bytes. Returns error if magic/version mismatch.
pub fn unpackHeader(bytes: []const u8) !Header {
    if (bytes.len < @sizeOf(Header)) return error.BufferTooSmall;
    const header: Header = @bitCast(bytes[0..@sizeOf(Header)].*);
    if (header.magic != MAGIC) return error.InvalidMagic;
    if (header.version != VERSION) return error.UnsupportedVersion;
    return header;
}

/// Calculate total artifact size from header.
pub fn artifactSize(header: Header) usize {
    var size: usize = @sizeOf(Header);
    size += @as(usize, header.mphf_seed_count) * 4; // u32 seeds
    size += @as(usize, header.mphf_key_count) * 2; // u16 fingerprints
    size += @as(usize, header.confusion_count) * @sizeOf(BinaryConfusionPair);
    size += @as(usize, header.gru_weight_bytes);
    return size;
}

// ============================================================================
// Tests
// ============================================================================

test "header pack and unpack round-trip" {
    const header = Header{
        .magic = MAGIC,
        .version = VERSION,
        .mphf_seed_count = 100,
        .mphf_key_count = 400,
        .mphf_bucket_size = 4,
        .confusion_count = 16,
        .gru_weight_bytes = 1024,
        ._reserved = 0,
    };

    const packed_bytes = packHeader(header);
    const unpacked = try unpackHeader(&packed_bytes);

    try testing.expectEqual(MAGIC, unpacked.magic);
    try testing.expectEqual(VERSION, unpacked.version);
    try testing.expectEqual(@as(u32, 100), unpacked.mphf_seed_count);
    try testing.expectEqual(@as(u32, 400), unpacked.mphf_key_count);
    try testing.expectEqual(@as(u32, 4), unpacked.mphf_bucket_size);
    try testing.expectEqual(@as(u32, 16), unpacked.confusion_count);
    try testing.expectEqual(@as(u32, 1024), unpacked.gru_weight_bytes);
}

test "header rejects bad magic" {
    var packed_bytes = packHeader(Header{
        .magic = MAGIC,
        .version = VERSION,
        .mphf_seed_count = 0,
        .mphf_key_count = 0,
        .mphf_bucket_size = 4,
        .confusion_count = 0,
        .gru_weight_bytes = 0,
        ._reserved = 0,
    });
    packed_bytes[0] = 0xFF; // corrupt magic
    try testing.expectError(error.InvalidMagic, unpackHeader(&packed_bytes));
}

test "header rejects bad version" {
    var packed_bytes = packHeader(Header{
        .magic = MAGIC,
        .version = VERSION,
        .mphf_seed_count = 0,
        .mphf_key_count = 0,
        .mphf_bucket_size = 4,
        .confusion_count = 0,
        .gru_weight_bytes = 0,
        ._reserved = 0,
    });
    packed_bytes[4] = 99; // corrupt version
    try testing.expectError(error.UnsupportedVersion, unpackHeader(&packed_bytes));
}

test "header rejects too-small buffer" {
    const small = [_]u8{ 1, 2, 3, 4 };
    try testing.expectError(error.BufferTooSmall, unpackHeader(&small));
}

test "artifact size calculation" {
    const header = Header{
        .magic = MAGIC,
        .version = VERSION,
        .mphf_seed_count = 10,
        .mphf_key_count = 40,
        .mphf_bucket_size = 4,
        .confusion_count = 5,
        .gru_weight_bytes = 200,
        ._reserved = 0,
    };
    const expected = @sizeOf(Header) + 10 * 4 + 40 * 2 + 5 * @sizeOf(BinaryConfusionPair) + 200;
    try testing.expectEqual(expected, artifactSize(header));
}

test "binary confusion pair round-trip" {
    const bp = BinaryConfusionPair.fromConfusionPair("rn", "m", 0.85);
    try testing.expectEqual(@as(u8, 2), bp.from_len);
    try testing.expectEqualStrings("rn", bp.from[0..bp.from_len]);
    try testing.expectEqual(@as(u8, 1), bp.to_len);
    try testing.expectEqualStrings("m", bp.to[0..bp.to_len]);
    // 0.85 * 10000 = 8500
    try testing.expectEqual(@as(u16, 8500), bp.probability_u16);
    // Round-trip probability
    try testing.expectApproxEqAbs(@as(f32, 0.85), bp.probability(), 0.001);
}
