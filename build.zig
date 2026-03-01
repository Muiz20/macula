const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library module
    const lib_mod = b.addModule("ocr", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Library artifact
    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "ocr-detector",
        .root_module = lib_mod,
    });
    b.installArtifact(lib);

    // Tests — Zig 0.16 requires root_module for addTest
    const test_step = b.step("test", "Run unit tests");

    const test_modules = [_][]const u8{
        "src/tokenizer.zig",
        "src/hash.zig",
        "src/mphf.zig",
        "src/confusion.zig",
        "src/candidate.zig",
        "src/gru.zig",
        "src/detector.zig",
        "src/binary.zig",
        "src/loader.zig",
    };

    for (test_modules) |src| {
        const test_mod = b.createModule(.{
            .root_source_file = b.path(src),
            .target = target,
            .optimize = optimize,
        });
        const t = b.addTest(.{
            .root_module = test_mod,
        });
        const run = b.addRunArtifact(t);
        test_step.dependOn(&run.step);
    }

    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    const wasm_mod = b.createModule(.{
        .root_source_file = b.path("src/wasm.zig"),
        .target = wasm_target,
        .optimize = optimize,
    });

    const wasm = b.addExecutable(.{
        .name = "ocr_wasm",
        .root_module = wasm_mod,
    });
    wasm.rdynamic = true;

    const install_wasm = b.addInstallArtifact(wasm, .{});
    const wasm_step = b.step("wasm", "Build WebAssembly module");
    wasm_step.dependOn(&install_wasm.step);
}
