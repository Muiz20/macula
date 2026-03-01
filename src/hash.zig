const std = @import("std");
const testing = std.testing;

/// FNV-1a hash for byte sequences. Fast, simple, good distribution.
pub fn fnv1a(data: []const u8) u64 {
    const offset_basis: u64 = 0xcbf29ce484222325;
    const prime: u64 = 0x100000001b3;
    var h: u64 = offset_basis;
    for (data) |byte| {
        h ^= byte;
        h *%= prime;
    }
    return h;
}

/// Seeded FNV-1a: mix a u32 seed into the initial state before hashing.
/// Used by MPHF for displacement-based re-hashing.
pub fn fnv1aSeeded(data: []const u8, seed: u32) u64 {
    const offset_basis: u64 = 0xcbf29ce484222325;
    const prime: u64 = 0x100000001b3;
    // Mix seed into basis
    var h: u64 = offset_basis ^ @as(u64, seed) *% 0x9e3779b97f4a7c15;
    for (data) |byte| {
        h ^= byte;
        h *%= prime;
    }
    return h;
}

/// Compute a 16-bit fingerprint for false-positive rejection.
pub fn fingerprint16(data: []const u8) u16 {
    // Use a different seed than bucket hashing to get independent bits
    const h = fnv1aSeeded(data, 0xDEAD);
    return @truncate(h);
}

// ============================================================================
// Tests
// ============================================================================

test "fnv1a known vectors" {
    // Empty string
    try testing.expectEqual(@as(u64, 0xcbf29ce484222325), fnv1a(""));
    // "a"
    try testing.expectEqual(@as(u64, 0xaf63dc4c8601ec8c), fnv1a("a"));
    // "foobar"
    try testing.expectEqual(@as(u64, 0x85944171f73967e8), fnv1a("foobar"));
}

test "fnv1a different inputs produce different hashes" {
    const h1 = fnv1a("hello");
    const h2 = fnv1a("world");
    const h3 = fnv1a("hellp"); // 1-char diff
    try testing.expect(h1 != h2);
    try testing.expect(h1 != h3);
    try testing.expect(h2 != h3);
}

test "seeded hash changes with seed" {
    const h1 = fnv1aSeeded("hello", 0);
    const h2 = fnv1aSeeded("hello", 1);
    const h3 = fnv1aSeeded("hello", 42);
    try testing.expect(h1 != h2);
    try testing.expect(h1 != h3);
    try testing.expect(h2 != h3);
}

test "seeded hash deterministic" {
    const h1 = fnv1aSeeded("test", 99);
    const h2 = fnv1aSeeded("test", 99);
    try testing.expectEqual(h1, h2);
}

test "fingerprint16 deterministic" {
    const fp1 = fingerprint16("hello");
    const fp2 = fingerprint16("hello");
    try testing.expectEqual(fp1, fp2);
}

test "fingerprint16 different for different inputs" {
    const fp1 = fingerprint16("hello");
    const fp2 = fingerprint16("world");
    // Not guaranteed by spec but near-certain for different inputs
    try testing.expect(fp1 != fp2);
}
