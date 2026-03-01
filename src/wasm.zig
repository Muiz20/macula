const std = @import("std");
const LoaderMod = @import("loader.zig");
const DetectorMod = @import("detector.zig");

const DetectionResult = DetectorMod.DetectionResult;

var g_allocator = std.heap.wasm_allocator;
var g_artifact: ?LoaderMod.LoadedArtifact = null;
var g_output_storage: [512 * 1024]u8 = undefined;
var g_output_len: usize = 0;
var g_error: []u8 = &[_]u8{};
var g_results: [1024]DetectionResult = undefined;

pub fn main() void {}

fn freeSlice(slice: []u8) void {
    if (slice.len > 0) g_allocator.free(slice);
}

fn setError(msg: []const u8) void {
    freeSlice(g_error);
    g_error = g_allocator.alloc(u8, msg.len) catch {
        g_error = &[_]u8{};
        return;
    };
    @memcpy(g_error, msg);
}

fn clearError() void {
    freeSlice(g_error);
    g_error = &[_]u8{};
}

fn bytesFromPtrConst(ptr: u32, len: u32) []const u8 {
    if (len == 0) return &[_]u8{};
    const p: [*]const u8 = @ptrFromInt(ptr);
    return p[0..@as(usize, len)];
}

fn bytesFromPtrMut(ptr: u32, len: u32) []u8 {
    if (len == 0) return &[_]u8{};
    const p: [*]u8 = @ptrFromInt(ptr);
    return p[0..@as(usize, len)];
}

fn outAppendByte(b: u8) !void {
    if (g_output_len >= g_output_storage.len) return error.NoSpaceLeft;
    g_output_storage[g_output_len] = b;
    g_output_len += 1;
}

fn outAppendSlice(slice: []const u8) !void {
    if (g_output_len + slice.len > g_output_storage.len) return error.NoSpaceLeft;
    @memcpy(g_output_storage[g_output_len .. g_output_len + slice.len], slice);
    g_output_len += slice.len;
}

fn outAppendFmt(comptime fmt: []const u8, args: anytype) !void {
    const written = try std.fmt.bufPrint(g_output_storage[g_output_len..], fmt, args);
    g_output_len += written.len;
}

fn appendEscapedJsonString(value: []const u8) !void {
    try outAppendByte('"');
    for (value) |b| {
        switch (b) {
            '"' => try outAppendSlice("\\\""),
            '\\' => try outAppendSlice("\\\\"),
            '\n' => try outAppendSlice("\\n"),
            '\r' => try outAppendSlice("\\r"),
            '\t' => try outAppendSlice("\\t"),
            else => {
                if (b < 0x20) {
                    try outAppendFmt("\\u{X:0>4}", .{@as(u16, b)});
                } else {
                    try outAppendByte(b);
                }
            },
        }
    }
    try outAppendByte('"');
}

fn statusText(status: DetectionResult.Status) []const u8 {
    return switch (status) {
        .valid => "valid",
        .corrected => "corrected",
        .error_detected => "error_detected",
    };
}

fn serializeResults(text: []const u8, count: usize) !void {
    g_output_len = 0;
    try outAppendByte('[');

    for (g_results[0..count], 0..) |result, idx| {
        if (idx > 0) try outAppendByte(',');
        try outAppendByte('{');

        try outAppendSlice("\"start\":");
        try outAppendFmt("{d}", .{result.start});
        try outAppendSlice(",\"len\":");
        try outAppendFmt("{d}", .{result.len});

        try outAppendSlice(",\"status\":");
        try appendEscapedJsonString(statusText(result.status));

        try outAppendSlice(",\"confidence\":");
        try outAppendFmt("{d:.6}", .{result.confidence});

        try outAppendSlice(",\"token\":");
        try appendEscapedJsonString(result.tokenSlice(text));

        try outAppendSlice(",\"correction\":");
        if (result.correctionSlice()) |corr| {
            try appendEscapedJsonString(corr);
        } else {
            try outAppendSlice("null");
        }

        try outAppendByte('}');
    }

    try outAppendByte(']');
}

pub export fn wasm_alloc(size: u32) u32 {
    if (size == 0) return 0;
    const buf = g_allocator.alloc(u8, @as(usize, size)) catch return 0;
    return @intCast(@intFromPtr(buf.ptr));
}

pub export fn wasm_free(ptr: u32, size: u32) void {
    if (ptr == 0 or size == 0) return;
    const buf = bytesFromPtrMut(ptr, size);
    g_allocator.free(buf);
}

pub export fn ocr_deinit() void {
    clearError();
    g_output_len = 0;
    if (g_artifact) |*artifact| {
        artifact.deinit();
        g_artifact = null;
    }
}

pub export fn ocr_init(artifact_ptr: u32, artifact_len: u32, threshold: f32) i32 {
    ocr_deinit();
    if (artifact_ptr == 0 or artifact_len == 0) {
        setError("artifact bytes are empty");
        return -1;
    }

    const artifact_src = bytesFromPtrConst(artifact_ptr, artifact_len);
    const artifact_copy = g_allocator.alloc(u8, artifact_src.len) catch {
        setError("allocation failed for artifact copy");
        return -2;
    };
    @memcpy(artifact_copy, artifact_src);

    const loaded = LoaderMod.loadFromBytes(g_allocator, artifact_copy) catch |err| {
        g_allocator.free(artifact_copy);
        setError(@errorName(err));
        return -3;
    };

    // Store the artifact into the global FIRST, so that the detector's
    // internal pointers reference the stable global memory — not the
    // stack-local `loaded` which would be invalidated after return.
    g_artifact = loaded;

    // Now re-initialise the detector with pointers into g_artifact's own fields.
    // In Wasm demo mode, keep the detector on dictionary + confusion layers.
    // This avoids runtime traps from GRU model/data mismatches in browser builds.
    g_artifact.?.detector = DetectorMod.OcrDetector.init(
        &g_artifact.?.mphf,
        &g_artifact.?.confusion_matrix,
        null,
        threshold,
    );
    return 0;
}

pub export fn ocr_process(text_ptr: u32, text_len: u32) i32 {
    clearError();
    if (g_artifact == null) {
        setError("model is not initialized");
        return -1;
    }

    const text = bytesFromPtrConst(text_ptr, text_len);
    const count = g_artifact.?.detector.processText(text, &g_results);

    serializeResults(text, count) catch |err| {
        setError(@errorName(err));
        g_output_len = 0;
        return -2;
    };

    return @intCast(count);
}

pub export fn ocr_output_ptr() u32 {
    if (g_output_len == 0) return 0;
    return @intCast(@intFromPtr(&g_output_storage));
}

pub export fn ocr_output_len() u32 {
    return @intCast(g_output_len);
}

pub export fn ocr_last_error_ptr() u32 {
    if (g_error.len == 0) return 0;
    return @intCast(@intFromPtr(g_error.ptr));
}

pub export fn ocr_last_error_len() u32 {
    return @intCast(g_error.len);
}
