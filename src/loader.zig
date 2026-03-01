const std = @import("std");
const testing = std.testing;
const BinaryMod = @import("binary.zig");
const Header = BinaryMod.Header;
const BinaryConfusionPair = BinaryMod.BinaryConfusionPair;
const MphfMod = @import("mphf.zig");
const Mphf = MphfMod.Mphf;
const ConfusionMod = @import("confusion.zig");
const ConfusionPair = ConfusionMod.ConfusionPair;
const ConfusionMatrix = ConfusionMod.ConfusionMatrix;
const GruMod = @import("gru.zig");
const Gru = GruMod.Gru;
const QuantizedWeights = GruMod.QuantizedWeights;
const DetectorMod = @import("detector.zig");
const OcrDetector = DetectorMod.OcrDetector;

/// Loaded artifact — owns all allocated memory.
/// Call `deinit` to free.
pub const LoadedArtifact = struct {
    mphf: Mphf,
    confusion_pairs: []ConfusionPair,
    confusion_matrix: ConfusionMatrix,
    gru: ?Gru,
    detector: OcrDetector,
    allocator: std.mem.Allocator,

    // Keep raw data alive for sliced references
    raw_data: []const u8,

    pub fn deinit(self: *LoadedArtifact) void {
        self.allocator.free(self.confusion_pairs);
        self.allocator.free(self.raw_data);
        self.* = undefined;
    }
};

/// Load an OCR detector artifact from a file path.
pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !LoadedArtifact {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const data = try file.readToEndAlloc(allocator, 50 * 1024 * 1024); // 50MB max
    return loadFromBytes(allocator, data);
}

/// Load an OCR detector artifact from a byte buffer.
/// The buffer is owned by the returned LoadedArtifact and freed on deinit.
pub fn loadFromBytes(allocator: std.mem.Allocator, data: []const u8) !LoadedArtifact {
    // Parse header
    if (data.len < @sizeOf(Header)) return error.BufferTooSmall;

    const header: Header = @bitCast(data[0..@sizeOf(Header)].*);
    if (header.magic != BinaryMod.MAGIC) return error.InvalidMagic;
    if (header.version != BinaryMod.VERSION) return error.UnsupportedVersion;

    var offset: usize = @sizeOf(Header);

    // --- MPHF Seeds ---
    const seed_bytes = @as(usize, header.mphf_seed_count) * 4;
    if (offset + seed_bytes > data.len) return error.BufferTooSmall;
    const seeds_ptr: [*]const u32 = @ptrCast(@alignCast(data[offset..].ptr));
    const seeds = seeds_ptr[0..header.mphf_seed_count];
    offset += seed_bytes;

    // --- MPHF Fingerprints ---
    const fp_bytes = @as(usize, header.mphf_key_count) * 2;
    if (offset + fp_bytes > data.len) return error.BufferTooSmall;
    const fps_ptr: [*]const u16 = @ptrCast(@alignCast(data[offset..].ptr));
    const fingerprints = fps_ptr[0..header.mphf_key_count];
    offset += fp_bytes;

    // --- Confusion Pairs ---
    const cp_size = @sizeOf(BinaryConfusionPair);
    const cp_bytes = @as(usize, header.confusion_count) * cp_size;
    if (offset + cp_bytes > data.len) return error.BufferTooSmall;

    // Decode binary confusion pairs into ConfusionPair structs.
    // IMPORTANT: .from and .to slices must reference the stable `data`
    // buffer (kept alive as raw_data), NOT a stack-local @bitCast copy.
    const confusion_pairs = try allocator.alloc(ConfusionPair, header.confusion_count);
    for (0..header.confusion_count) |i| {
        const bp_offset = offset + i * cp_size;
        const bp_bytes = data[bp_offset..][0..cp_size];

        // BinaryConfusionPair layout: from_len(1) + from(8) + to_len(1) + to(8) + prob(2)
        const from_len: usize = bp_bytes[0];
        const from_start = bp_offset + 1;
        const to_len: usize = bp_bytes[9];
        const to_start = bp_offset + 10;

        // Read probability from the local copy (safe, it's a value)
        const bp: BinaryConfusionPair = @bitCast(bp_bytes.*);

        confusion_pairs[i] = .{
            .from = data[from_start..][0..from_len],
            .to = data[to_start..][0..to_len],
            .probability = bp.probability(),
        };
    }
    offset += cp_bytes;

    // --- GRU Weights ---
    var gru: ?Gru = null;
    if (header.gru_weight_bytes > 0) {
        if (offset + header.gru_weight_bytes > data.len) return error.BufferTooSmall;

        const gru_data = data[offset..][0..header.gru_weight_bytes];

        // Parse GRU header: input_size(u32) + hidden_size(u32) + 6 scales(f32)
        const gru_header_size = 2 * 4 + 6 * 4; // 32 bytes
        if (gru_data.len < gru_header_size) return error.BufferTooSmall;

        const input_size = std.mem.readInt(u32, gru_data[0..4], .little);
        const hidden_size = std.mem.readInt(u32, gru_data[4..8], .little);

        const s_w_ih: f32 = @bitCast(std.mem.readInt(u32, gru_data[8..12], .little));
        const s_w_hh: f32 = @bitCast(std.mem.readInt(u32, gru_data[12..16], .little));
        const s_b_ih: f32 = @bitCast(std.mem.readInt(u32, gru_data[16..20], .little));
        const s_b_hh: f32 = @bitCast(std.mem.readInt(u32, gru_data[20..24], .little));
        const s_out_w: f32 = @bitCast(std.mem.readInt(u32, gru_data[24..28], .little));
        const s_out_b: f32 = @bitCast(std.mem.readInt(u32, gru_data[28..32], .little));

        const hs: usize = hidden_size;
        const is: usize = input_size;
        var g_off: usize = gru_header_size;

        // w_ih: 3*hs*is bytes
        const w_ih_len = 3 * hs * is;
        const w_ih_data: []const i8 = @ptrCast(gru_data[g_off..][0..w_ih_len]);
        g_off += w_ih_len;

        // w_hh: 3*hs*hs bytes
        const w_hh_len = 3 * hs * hs;
        const w_hh_data: []const i8 = @ptrCast(gru_data[g_off..][0..w_hh_len]);
        g_off += w_hh_len;

        // b_ih: 3*hs bytes
        const b_ih_len = 3 * hs;
        const b_ih_data: []const i8 = @ptrCast(gru_data[g_off..][0..b_ih_len]);
        g_off += b_ih_len;

        // b_hh: 3*hs bytes
        const b_hh_len = 3 * hs;
        const b_hh_data: []const i8 = @ptrCast(gru_data[g_off..][0..b_hh_len]);
        g_off += b_hh_len;

        // w_out: is*hs bytes
        const w_out_len = is * hs;
        const w_out_data: []const i8 = @ptrCast(gru_data[g_off..][0..w_out_len]);
        g_off += w_out_len;

        // b_out: is bytes
        const b_out_data: []const i8 = @ptrCast(gru_data[g_off..][0..is]);

        gru = Gru{
            .config = .{ .input_size = input_size, .hidden_size = hidden_size },
            .w_ih = QuantizedWeights{ .data = w_ih_data, .scale = s_w_ih },
            .w_hh = QuantizedWeights{ .data = w_hh_data, .scale = s_w_hh },
            .b_ih = QuantizedWeights{ .data = b_ih_data, .scale = s_b_ih },
            .b_hh = QuantizedWeights{ .data = b_hh_data, .scale = s_b_hh },
            .w_out = QuantizedWeights{ .data = w_out_data, .scale = s_out_w },
            .b_out = QuantizedWeights{ .data = b_out_data, .scale = s_out_b },
        };
    }

    const mphf = Mphf{
        .seeds = seeds,
        .fingerprints = fingerprints,
        .n = header.mphf_key_count,
        .bucket_size = header.mphf_bucket_size,
    };

    const confusion_matrix = ConfusionMatrix.init(confusion_pairs);

    // NOTE: detector is left as `undefined` here on purpose.
    // The caller MUST call OcrDetector.init() after storing the artifact
    // at its final memory location, so that the detector's internal
    // pointers reference stable memory (not this stack-local).
    return LoadedArtifact{
        .mphf = mphf,
        .confusion_pairs = confusion_pairs,
        .confusion_matrix = confusion_matrix,
        .gru = gru,
        .detector = undefined,
        .allocator = allocator,
        .raw_data = data,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "loader — round-trip with compiled artifact" {
    // Build a minimal artifact in memory
    const header = Header{
        .magic = BinaryMod.MAGIC,
        .version = BinaryMod.VERSION,
        .mphf_seed_count = 2,
        .mphf_key_count = 6,
        .mphf_bucket_size = 2,
        .confusion_count = 0,
        .gru_weight_bytes = 0,
        ._reserved = 0,
    };

    const header_bytes = BinaryMod.packHeader(header);
    const seeds = [_]u32{ 1, 2 };
    const fingerprints = [_]u16{ 0, 0, 0, 0, 0, 0 };

    var buf: [200]u8 = undefined;
    var pos: usize = 0;
    @memcpy(buf[pos..][0..@sizeOf(Header)], &header_bytes);
    pos += @sizeOf(Header);
    @memcpy(buf[pos..][0..8], std.mem.asBytes(&seeds));
    pos += 8;
    @memcpy(buf[pos..][0..12], std.mem.asBytes(&fingerprints));
    pos += 12;

    // Copy to allocated buffer
    const data = try testing.allocator.alloc(u8, pos);
    @memcpy(data, buf[0..pos]);

    var artifact = try loadFromBytes(testing.allocator, data);
    defer artifact.deinit();

    try testing.expectEqual(@as(u32, 2), @as(u32, @intCast(artifact.mphf.seeds.len)));
    try testing.expectEqual(@as(u32, 6), artifact.mphf.n);
}
