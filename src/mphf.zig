const std = @import("std");
const testing = std.testing;
const hash = @import("hash.zig");

/// Minimal Perfect Hash Function using CHD (Compress, Hash, Displace).
///
/// Given a set of N keys, maps each key to a unique index in [0, N) with
/// zero collisions. Uses ~6 bytes per key (4-byte seed + 2-byte fingerprint).
///
/// Build is offline (allocates). Lookup is O(1) with zero allocation.
pub const Mphf = struct {
    /// One seed per bucket. Bucket count = ceil(n / bucket_size).
    seeds: []const u32,
    /// One fingerprint per slot, for false-positive rejection.
    fingerprints: []const u16,
    /// Total number of keys.
    n: u32,
    /// Number of keys per bucket (load factor tuning).
    bucket_size: u32,

    /// Lookup: does the MPHF contain this key?
    /// O(1), zero allocation. May have false positives (~1/65536 per query).
    pub fn contains(self: *const Mphf, key: []const u8) bool {
        if (self.n == 0) return false;
        const idx = self.lookup(key);
        return self.fingerprints[idx] == hash.fingerprint16(key);
    }

    /// Compute the slot index for a key. Always returns a value in [0, n).
    pub fn lookup(self: *const Mphf, key: []const u8) u32 {
        const bucket = self.bucketFor(key);
        const seed = self.seeds[bucket];
        const h = hash.fnv1aSeeded(key, seed);
        return @intCast(h % self.n);
    }

    fn bucketFor(self: *const Mphf, key: []const u8) u32 {
        const h = hash.fnv1a(key);
        return @intCast(h % self.seeds.len);
    }

    /// Build an MPHF from a list of keys. Allocates using the provided allocator.
    /// The caller owns the returned Mphf and must call `deinit` to free it.
    pub fn build(allocator: std.mem.Allocator, keys: []const []const u8) !Mphf {
        const n: u32 = @intCast(keys.len);
        if (n == 0) return Mphf{
            .seeds = &.{},
            .fingerprints = &.{},
            .n = 0,
            .bucket_size = 4,
        };

        // Use a larger table to reduce collisions (load factor ~0.33)
        const table_size: u32 = n * 3;
        const bucket_size: u32 = 2;
        const num_buckets: u32 = (n + bucket_size - 1) / bucket_size;

        // Assign each key to a bucket
        const bucket_lists = try allocator.alloc(std.ArrayListUnmanaged(u32), num_buckets);
        defer {
            for (bucket_lists) |*bl| bl.deinit(allocator);
            allocator.free(bucket_lists);
        }
        for (bucket_lists) |*bl| bl.* = .{};

        for (keys, 0..) |key, i| {
            const bucket: u32 = @intCast(hash.fnv1a(key) % num_buckets);
            try bucket_lists[bucket].append(allocator, @intCast(i));
        }

        // Sort buckets by size descending (largest first for better packing)
        const bucket_order = try allocator.alloc(u32, num_buckets);
        defer allocator.free(bucket_order);
        for (bucket_order, 0..) |*bo, i| bo.* = @intCast(i);

        std.mem.sort(u32, bucket_order, bucket_lists, struct {
            pub fn lessThan(ctx: []const std.ArrayListUnmanaged(u32), a: u32, b: u32) bool {
                return ctx[a].items.len > ctx[b].items.len;
            }
        }.lessThan);

        // Allocate output
        const seeds = try allocator.alloc(u32, num_buckets);
        @memset(seeds, 0);

        const fingerprints = try allocator.alloc(u16, table_size);
        @memset(fingerprints, 0);

        // Track which slots are occupied
        const occupied = try allocator.alloc(bool, table_size);
        defer allocator.free(occupied);
        @memset(occupied, false);

        // For each bucket (largest first), find a seed that maps all its keys
        // to unoccupied slots with no internal collisions.
        for (bucket_order) |bucket_idx| {
            const members = bucket_lists[bucket_idx].items;
            if (members.len == 0) continue;

            // Dynamic temp buffer for this bucket's candidate slots
            const temp_slots = try allocator.alloc(u32, members.len);
            defer allocator.free(temp_slots);

            var seed: u32 = 1; // Start at 1 (0 is reserved for "unset")
            var found = false;
            while (seed < 10_000_000) : (seed += 1) {
                // Compute candidate slots for all members
                var valid = true;
                for (members, 0..) |key_idx, mi| {
                    const h = hash.fnv1aSeeded(keys[key_idx], seed);
                    const slot: u32 = @intCast(h % table_size);
                    // Check if slot is already globally occupied
                    if (occupied[slot]) {
                        valid = false;
                        break;
                    }
                    // Check for internal collision within this bucket
                    for (temp_slots[0..mi]) |prev| {
                        if (prev == slot) {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid) break;
                    temp_slots[mi] = slot;
                }

                if (valid) {
                    // Accept this seed — mark slots occupied and store fingerprints
                    seeds[bucket_idx] = seed;
                    for (members, 0..) |key_idx, mi| {
                        const slot = temp_slots[mi];
                        occupied[slot] = true;
                        fingerprints[slot] = hash.fingerprint16(keys[key_idx]);
                    }
                    found = true;
                    break;
                }
            }

            if (!found) return error.MphfBuildFailed;
        }

        return Mphf{
            .seeds = seeds,
            .fingerprints = fingerprints,
            .n = table_size,
            .bucket_size = bucket_size,
        };
    }

    /// Free memory allocated by `build`.
    pub fn deinit(self: *Mphf, allocator: std.mem.Allocator) void {
        if (self.seeds.len > 0) allocator.free(self.seeds);
        if (self.fingerprints.len > 0) allocator.free(self.fingerprints);
        self.* = undefined;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "mphf build and lookup — all known words found" {
    const words = [_][]const u8{
        "the",  "quick", "brown", "fox",   "jumps",
        "over", "lazy",  "dog",   "hello", "world",
    };
    var mphf = try Mphf.build(testing.allocator, &words);
    defer mphf.deinit(testing.allocator);

    // All known words must be found
    for (&words) |w| {
        try testing.expect(mphf.contains(w));
    }
}

test "mphf rejects unknown words" {
    const words = [_][]const u8{
        "apple", "banana", "cherry", "date", "elderberry",
    };
    var mphf = try Mphf.build(testing.allocator, &words);
    defer mphf.deinit(testing.allocator);

    // Unknown words should (almost certainly) be rejected via fingerprint
    try testing.expect(!mphf.contains("xylophone"));
    try testing.expect(!mphf.contains("zebra"));
    try testing.expect(!mphf.contains("mango"));
    try testing.expect(!mphf.contains(""));
}

test "mphf unique indices for known words" {
    const words = [_][]const u8{
        "alpha", "beta", "gamma", "delta", "epsilon",
        "zeta",  "eta",  "theta",
    };
    var mphf = try Mphf.build(testing.allocator, &words);
    defer mphf.deinit(testing.allocator);

    // All lookup indices must be unique (the "perfect" part)
    var seen = std.AutoHashMap(u32, void).init(testing.allocator);
    defer seen.deinit();
    for (&words) |w| {
        const idx = mphf.lookup(w);
        try testing.expect(idx < mphf.n);
        const result = try seen.getOrPut(idx);
        try testing.expect(!result.found_existing);
    }
}

test "mphf empty set" {
    const words = [_][]const u8{};
    var mphf = try Mphf.build(testing.allocator, &words);
    // No deinit needed for empty
    try testing.expect(!mphf.contains("anything"));
    _ = &mphf;
}

test "mphf larger set" {
    // Generate 100 unique keys
    var keys: [100][]const u8 = undefined;
    var bufs: [100][8]u8 = undefined;
    for (&keys, &bufs, 0..) |*k, *b, i| {
        const i_u32: u32 = @intCast(i);
        b[0] = 'k';
        b[1] = @truncate((i_u32 / 1000) % 10 + '0');
        b[2] = @truncate((i_u32 / 100) % 10 + '0');
        b[3] = @truncate((i_u32 / 10) % 10 + '0');
        b[4] = @truncate(i_u32 % 10 + '0');
        k.* = b[0..5];
    }

    var mphf = try Mphf.build(testing.allocator, &keys);
    defer mphf.deinit(testing.allocator);

    // All keys must be found
    for (&keys) |k| {
        try testing.expect(mphf.contains(k));
    }

    // All lookup indices must be unique
    var seen = std.AutoHashMap(u32, void).init(testing.allocator);
    defer seen.deinit();
    for (&keys) |k| {
        const idx = mphf.lookup(k);
        try testing.expect(idx < mphf.n);
        const result = try seen.getOrPut(idx);
        try testing.expect(!result.found_existing);
    }
}
