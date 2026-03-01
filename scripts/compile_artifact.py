"""
OCR Error Detector — Binary Artifact Compiler

Packages wordlist, confusion pairs, and GRU weights into a single
binary artifact for the Zig runtime.

Format: see src/binary.zig for the layout specification.

Usage:
    python scripts/compile_artifact.py
    
Reads:
    data/processed/wordlist.txt
    data/processed/confusion.csv  
    data/processed/gru_weights.bin

Outputs:
    data/ocr_detector.bin
"""

import csv
import struct
import sys
from pathlib import Path

# Must match src/binary.zig
MAGIC = 0x4F435244  # "OCRD"
VERSION = 1

def build_mphf_offline(words: list[str]) -> tuple[list[int], list[int], int]:
    """
    Build MPHF seeds and fingerprints offline.
    Must match the algorithm in src/mphf.zig exactly.
    
    Returns (seeds, fingerprints, table_size)
    """
    n = len(words)
    if n == 0:
        return [], [], 0
    
    table_size = n * 3
    bucket_size = 2
    num_buckets = (n + bucket_size - 1) // bucket_size
    
    # FNV-1a 64-bit hash — must match src/hash.zig exactly
    MASK64 = (1 << 64) - 1
    FNV_OFFSET = 0xcbf29ce484222325
    FNV_PRIME  = 0x100000001b3

    def fnv1a(data: bytes) -> int:
        h = FNV_OFFSET
        for b in data:
            h ^= b
            h = (h * FNV_PRIME) & MASK64
        return h
    
    def fnv1a_seeded(data: bytes, seed: int) -> int:
        # Must match: offset_basis ^ (seed *% 0x9e3779b97f4a7c15)
        h = FNV_OFFSET ^ ((seed * 0x9e3779b97f4a7c15) & MASK64)
        for b in data:
            h ^= b
            h = (h * FNV_PRIME) & MASK64
        return h
    
    def fingerprint16(data: bytes) -> int:
        # Must match: fnv1aSeeded(data, 0xDEAD) truncated to u16
        h = fnv1a_seeded(data, 0xDEAD)
        return h & 0xFFFF
    
    # Assign keys to buckets
    buckets = [[] for _ in range(num_buckets)]
    for i, w in enumerate(words):
        bucket = fnv1a(w.encode()) % num_buckets
        buckets[bucket].append(i)
    
    # Sort by bucket size (largest first)
    bucket_order = sorted(range(num_buckets), key=lambda b: -len(buckets[b]))
    
    seeds = [0] * num_buckets
    fingerprints = [0] * table_size
    occupied = [False] * table_size
    
    for bucket_idx in bucket_order:
        members = buckets[bucket_idx]
        if not members:
            continue
        
        found = False
        for seed in range(1, 10_000_000):
            # Compute candidate slots
            temp_slots = []
            valid = True
            for key_idx in members:
                h = fnv1a_seeded(words[key_idx].encode(), seed)
                slot = h % table_size
                if occupied[slot] or slot in temp_slots:
                    valid = False
                    break
                temp_slots.append(slot)
            
            if valid:
                seeds[bucket_idx] = seed
                for key_idx, slot in zip(members, temp_slots):
                    occupied[slot] = True
                    fingerprints[slot] = fingerprint16(words[key_idx].encode())
                found = True
                break
        
        if not found:
            print(f"ERROR: MPHF build failed for bucket {bucket_idx} ({len(members)} members)")
            sys.exit(1)
    
    return seeds, fingerprints, table_size


def main():
    processed = Path("data/processed")
    out_path = Path("data/ocr_detector.bin")
    
    # --- Load wordlist ---
    wordlist_path = processed / "wordlist.txt"
    words = wordlist_path.read_text(encoding='utf-8').strip().split('\n')
    words = [w.strip() for w in words if w.strip()]
    print(f"Loaded {len(words)} words")
    
    # --- Build MPHF ---
    print("Building MPHF (this may take a moment)...")
    seeds, fingerprints, table_size = build_mphf_offline(words)
    print(f"  MPHF: {len(seeds)} buckets, {table_size} slots")
    
    # --- Load confusion pairs ---
    confusion_path = processed / "confusion.csv"
    confusion_pairs = []
    with open(confusion_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            from_pat = row['from']
            to_pat = row['to']
            prob = float(row['probability'])
            confusion_pairs.append((from_pat, to_pat, prob))
    print(f"Loaded {len(confusion_pairs)} confusion pairs")
    
    # --- Load GRU weights ---
    gru_path = processed / "gru_weights.bin"
    gru_bytes = gru_path.read_bytes()
    print(f"Loaded GRU weights: {len(gru_bytes)} bytes")
    
    # --- Write binary artifact ---
    with open(out_path, 'wb') as f:
        # Header (32 bytes)
        header = struct.pack('<IIIIIIII',
            MAGIC,
            VERSION,
            len(seeds),          # mphf_seed_count
            table_size,          # mphf_key_count (table_size, not word count)
            2,                   # mphf_bucket_size
            len(confusion_pairs), # confusion_count
            len(gru_bytes),      # gru_weight_bytes
            0,                   # reserved
        )
        f.write(header)
        
        # MPHF Seeds (u32 each)
        for s in seeds:
            f.write(struct.pack('<I', s))
        
        # MPHF Fingerprints (u16 each)
        for fp in fingerprints:
            f.write(struct.pack('<H', fp))
        
        # Confusion pairs (20 bytes each: u8 from_len, 8B from, u8 to_len, 8B to, u16 prob)
        for from_pat, to_pat, prob in confusion_pairs:
            from_bytes = from_pat.encode('utf-8')[:8]
            to_bytes = to_pat.encode('utf-8')[:8]
            prob_u16 = int(prob * 10000)
            
            f.write(struct.pack('B', len(from_bytes)))
            f.write(from_bytes.ljust(8, b'\x00'))
            f.write(struct.pack('B', len(to_bytes)))
            f.write(to_bytes.ljust(8, b'\x00'))
            f.write(struct.pack('<H', prob_u16))
        
        # GRU weights (raw binary)
        f.write(gru_bytes)
    
    total_size = out_path.stat().st_size
    print(f"\nWrote {out_path}")
    print(f"  Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print(f"  Breakdown:")
    print(f"    Header:       32 bytes")
    print(f"    MPHF seeds:   {len(seeds) * 4:,} bytes")
    print(f"    MPHF fps:     {table_size * 2:,} bytes")
    print(f"    Confusion:    {len(confusion_pairs) * 20:,} bytes")
    print(f"    GRU weights:  {len(gru_bytes):,} bytes")
    
    # Verify
    verify_artifact(out_path, words)

def verify_artifact(path: Path, words: list[str]):
    """Quick sanity check that the artifact can be parsed."""
    data = path.read_bytes()
    magic, version = struct.unpack_from('<II', data, 0)
    assert magic == MAGIC, f"Bad magic: {magic:#x}"
    assert version == VERSION, f"Bad version: {version}"
    
    seed_count, key_count, bucket_size, confusion_count, gru_bytes, _ = \
        struct.unpack_from('<IIIIII', data, 8)
    
    print(f"\nVerification:")
    print(f"  Magic: 0x{magic:08X} ✓")
    print(f"  Version: {version} ✓")
    print(f"  Seeds: {seed_count}, Keys: {key_count}, Bucket: {bucket_size}")
    print(f"  Confusion pairs: {confusion_count}")
    print(f"  GRU weight bytes: {gru_bytes}")
    
    # Verify MPHF lookups
    offset = 32
    seeds = struct.unpack_from(f'<{seed_count}I', data, offset)
    offset += seed_count * 4
    fingerprints = struct.unpack_from(f'<{key_count}H', data, offset)
    
    MASK64 = (1 << 64) - 1
    FNV_OFFSET = 0xcbf29ce484222325
    FNV_PRIME  = 0x100000001b3

    def fnv1a(d: bytes) -> int:
        h = FNV_OFFSET
        for b in d:
            h ^= b
            h = (h * FNV_PRIME) & MASK64
        return h
    
    def fnv1a_seeded(d: bytes, seed: int) -> int:
        h = FNV_OFFSET ^ ((seed * 0x9e3779b97f4a7c15) & MASK64)
        for b in d:
            h ^= b
            h = (h * FNV_PRIME) & MASK64
        return h
    
    def fp16(d: bytes) -> int:
        h = fnv1a_seeded(d, 0xDEAD)
        return h & 0xFFFF
    
    # Check a sample of words
    hits = 0
    for w in words[:100]:
        wb = w.encode()
        bucket = fnv1a(wb) % seed_count
        seed = seeds[bucket]
        slot = fnv1a_seeded(wb, seed) % key_count
        if fingerprints[slot] == fp16(wb):
            hits += 1
    
    print(f"  MPHF verification: {hits}/100 words found ✓" if hits >= 98 else 
          f"  MPHF verification: {hits}/100 — WARNING: low hit rate")

if __name__ == '__main__':
    main()
