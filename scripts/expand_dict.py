"""
Merge the ICDAR-derived wordlist with a large English dictionary,
rebuild the MPHF artifact with the expanded vocabulary.
"""

import re
import struct
import sys
from pathlib import Path


def normalize_word(word: str) -> str:
    word = word.lower().strip()
    return re.sub(r"[^a-z']", "", word)


def fnv1a(data: bytes) -> int:
    h = 0x811c9dc5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def fnv1a_seeded(data: bytes, seed: int) -> int:
    h = 0x811c9dc5 ^ seed
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def fingerprint16(data: bytes) -> int:
    h = fnv1a(data)
    return ((h >> 16) ^ (h & 0xFFFF)) & 0xFFFF


def build_mphf_offline(words: list[str]) -> tuple[list[int], list[int], int]:
    n = len(words)
    if n == 0:
        return [], [], 0

    table_size = n * 3
    bucket_size = 2
    num_buckets = (n + bucket_size - 1) // bucket_size

    buckets = [[] for _ in range(num_buckets)]
    for i, w in enumerate(words):
        bucket = fnv1a(w.encode()) % num_buckets
        buckets[bucket].append(i)

    bucket_order = sorted(range(num_buckets), key=lambda b: -len(buckets[b]))

    seeds = [0] * num_buckets
    fingerprints = [0] * table_size
    occupied = [False] * table_size

    for bi, bucket_idx in enumerate(bucket_order):
        members = buckets[bucket_idx]
        if not members:
            continue

        if bi % 5000 == 0 and bi > 0:
            print(f"  Processing bucket {bi}/{num_buckets}...")

        found = False
        for seed in range(1, 10_000_000):
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
            print(f"ERROR: MPHF build failed for bucket {bucket_idx}")
            sys.exit(1)

    return seeds, fingerprints, table_size


def main():
    processed = Path("data/processed")

    # Load ICDAR wordlist
    icdar_words = set(
        (processed / "wordlist.txt").read_text(encoding='utf-8').strip().split('\n')
    )
    print(f"ICDAR words: {len(icdar_words)}")

    # Load large English dictionary
    full_dict_path = Path("data/english_words_full.txt")
    raw = full_dict_path.read_text(encoding='utf-8').strip().split('\n')
    english_words = set()
    for w in raw:
        w = normalize_word(w)
        if len(w) >= 2 and w.isalpha():
            english_words.add(w)
    print(f"English dictionary: {len(english_words)} words")

    # Merge
    merged = sorted(icdar_words | english_words)
    print(f"Merged vocabulary: {len(merged)} words")

    # Save merged wordlist
    merged_path = processed / "wordlist_expanded.txt"
    merged_path.write_text('\n'.join(merged) + '\n', encoding='utf-8')
    print(f"Saved to {merged_path}")

    # Build MPHF
    print("Building MPHF (this will take a few minutes for large dictionaries)...")
    seeds, fingerprints, table_size = build_mphf_offline(merged)
    print(f"  MPHF: {len(seeds)} buckets, {table_size} slots")

    # Verify a sample
    hits = 0
    sample = merged[:200]
    for w in sample:
        wb = w.encode()
        bucket = fnv1a(wb) % len(seeds)
        seed = seeds[bucket]
        slot = fnv1a_seeded(wb, seed) % table_size
        if fingerprints[slot] == fingerprint16(wb):
            hits += 1
    print(f"  Verification: {hits}/{len(sample)} words found")

    # Load existing confusion pairs and GRU weights
    import csv
    confusion_path = processed / "confusion.csv"
    confusion_pairs = []
    with open(confusion_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            confusion_pairs.append((row['from'], row['to'], float(row['probability'])))
    
    gru_bytes = (processed / "gru_weights.bin").read_bytes()

    # Write expanded artifact
    MAGIC = 0x4F435244
    VERSION = 1
    out_path = Path("data/ocr_detector.bin")

    with open(out_path, 'wb') as f:
        header = struct.pack('<IIIIIIII',
            MAGIC, VERSION,
            len(seeds), table_size, 2,
            len(confusion_pairs), len(gru_bytes), 0,
        )
        f.write(header)
        for s in seeds:
            f.write(struct.pack('<I', s))
        for fp in fingerprints:
            f.write(struct.pack('<H', fp))
        for from_pat, to_pat, prob in confusion_pairs:
            from_bytes = from_pat.encode('utf-8')[:8]
            to_bytes = to_pat.encode('utf-8')[:8]
            f.write(struct.pack('B', len(from_bytes)))
            f.write(from_bytes.ljust(8, b'\x00'))
            f.write(struct.pack('B', len(to_bytes)))
            f.write(to_bytes.ljust(8, b'\x00'))
            f.write(struct.pack('<H', int(prob * 10000)))
        f.write(gru_bytes)

    total_size = out_path.stat().st_size
    print(f"\nWrote {out_path}: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print(f"  MPHF: {len(seeds)*4 + table_size*2:,} bytes")
    print(f"  Confusion: {len(confusion_pairs)*20:,} bytes")
    print(f"  GRU: {len(gru_bytes):,} bytes")


if __name__ == '__main__':
    main()
