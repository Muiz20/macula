const std = @import("std");
const testing = std.testing;

/// A zero-copy token: byte offset + length into the original input buffer.
pub const Token = struct {
    /// Byte offset of the first character in the source buffer.
    start: usize,
    /// Length in bytes.
    len: usize,

    /// Return the slice from the original buffer.
    pub fn slice(self: Token, source: []const u8) []const u8 {
        return source[self.start..][0..self.len];
    }
};

/// Zero-copy tokenizer. Iterates over a byte buffer yielding word tokens.
/// Skips whitespace and punctuation. Does NOT allocate.
pub const Tokenizer = struct {
    source: []const u8,
    pos: usize,

    pub fn init(source: []const u8) Tokenizer {
        return .{ .source = source, .pos = 0 };
    }

    /// Advance to the next word token, or return null if exhausted.
    pub fn next(self: *Tokenizer) ?Token {
        // Skip non-word bytes
        while (self.pos < self.source.len and !isWordByte(self.source[self.pos])) {
            self.pos += 1;
        }
        if (self.pos >= self.source.len) return null;

        const start = self.pos;
        // Consume word bytes
        while (self.pos < self.source.len and isWordByte(self.source[self.pos])) {
            self.pos += 1;
        }
        return Token{ .start = start, .len = self.pos - start };
    }

    /// Reset the tokenizer to the beginning.
    pub fn reset(self: *Tokenizer) void {
        self.pos = 0;
    }

    /// Returns true if byte is part of a word (letter, digit, or apostrophe).
    fn isWordByte(b: u8) bool {
        return switch (b) {
            'a'...'z', 'A'...'Z', '0'...'9', '\'' => true,
            else => false,
        };
    }
};

/// Convert an ASCII byte to lowercase. Non-alpha bytes pass through unchanged.
pub fn toLowerAscii(b: u8) u8 {
    return if (b >= 'A' and b <= 'Z') b + 32 else b;
}

/// Lowercase-normalize a token's content into a caller-provided buffer.
/// Returns the slice of the output buffer that was written to.
pub fn normalizeLower(source: []const u8, token: Token, buf: []u8) []const u8 {
    const word = token.slice(source);
    const len = @min(word.len, buf.len);
    for (word[0..len], 0..) |b, i| {
        buf[i] = toLowerAscii(b);
    }
    return buf[0..len];
}

// ============================================================================
// Tests
// ============================================================================

test "tokenize simple words" {
    var tok = Tokenizer.init("hello world");
    const t1 = tok.next().?;
    try testing.expectEqualStrings("hello", t1.slice("hello world"));
    const t2 = tok.next().?;
    try testing.expectEqualStrings("world", t2.slice("hello world"));
    try testing.expect(tok.next() == null);
}

test "tokenize skips punctuation" {
    const input = "hello, world! foo-bar.";
    var tok = Tokenizer.init(input);
    try testing.expectEqualStrings("hello", tok.next().?.slice(input));
    try testing.expectEqualStrings("world", tok.next().?.slice(input));
    try testing.expectEqualStrings("foo", tok.next().?.slice(input));
    try testing.expectEqualStrings("bar", tok.next().?.slice(input));
    try testing.expect(tok.next() == null);
}

test "tokenize preserves apostrophes" {
    const input = "don't can't";
    var tok = Tokenizer.init(input);
    try testing.expectEqualStrings("don't", tok.next().?.slice(input));
    try testing.expectEqualStrings("can't", tok.next().?.slice(input));
    try testing.expect(tok.next() == null);
}

test "tokenize empty input" {
    var tok = Tokenizer.init("");
    try testing.expect(tok.next() == null);
}

test "tokenize only whitespace and punctuation" {
    var tok = Tokenizer.init("   ,.!?  ");
    try testing.expect(tok.next() == null);
}

test "tokenize multiple spaces between words" {
    const input = "  hello   world  ";
    var tok = Tokenizer.init(input);
    try testing.expectEqualStrings("hello", tok.next().?.slice(input));
    try testing.expectEqualStrings("world", tok.next().?.slice(input));
    try testing.expect(tok.next() == null);
}

test "tokenize reports correct byte offsets" {
    const input = "  abc def  ";
    var tok = Tokenizer.init(input);
    const t1 = tok.next().?;
    try testing.expectEqual(@as(usize, 2), t1.start);
    try testing.expectEqual(@as(usize, 3), t1.len);
    const t2 = tok.next().?;
    try testing.expectEqual(@as(usize, 6), t2.start);
    try testing.expectEqual(@as(usize, 3), t2.len);
}

test "normalize lowercase" {
    const input = "Hello WORLD";
    var tok = Tokenizer.init(input);
    var buf: [64]u8 = undefined;

    const t1 = tok.next().?;
    const lower1 = normalizeLower(input, t1, &buf);
    try testing.expectEqualStrings("hello", lower1);

    const t2 = tok.next().?;
    const lower2 = normalizeLower(input, t2, &buf);
    try testing.expectEqualStrings("world", lower2);
}

test "tokenize with digits" {
    const input = "page42 test123";
    var tok = Tokenizer.init(input);
    try testing.expectEqualStrings("page42", tok.next().?.slice(input));
    try testing.expectEqualStrings("test123", tok.next().?.slice(input));
    try testing.expect(tok.next() == null);
}

test "reset tokenizer" {
    const input = "aaa bbb";
    var tok = Tokenizer.init(input);
    _ = tok.next();
    _ = tok.next();
    try testing.expect(tok.next() == null);

    tok.reset();
    try testing.expectEqualStrings("aaa", tok.next().?.slice(input));
}
