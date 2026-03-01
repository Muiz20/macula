//! OCR Error Detector — Architecture E (OCR-Aware Hybrid)
//!
//! A lightweight, embedded-friendly OCR error detection and correction engine.
//! Designed for phone and browser (Wasm) targets with a total binary budget < 10MB.
//!
//! Layers:
//!   1. MPHF Dictionary — fast-path O(1) known-word check
//!   2. OCR Confusion Matrix — visual confusion pattern model
//!   3. Candidate Generator — expand unknowns via confusion patterns
//!   4. GRU Scorer — int8 quantized character-level language model
//!   5. Pipeline — orchestrates all layers

pub const Tokenizer = @import("tokenizer.zig").Tokenizer;
pub const Token = @import("tokenizer.zig").Token;
pub const hash = @import("hash.zig");
pub const Mphf = @import("mphf.zig").Mphf;
pub const ConfusionMatrix = @import("confusion.zig").ConfusionMatrix;
pub const CandidateGenerator = @import("candidate.zig").CandidateGenerator;
pub const Gru = @import("gru.zig").Gru;
pub const OcrDetector = @import("detector.zig").OcrDetector;
pub const binary = @import("binary.zig");
pub const loader = @import("loader.zig");
