"""
OCR Error Detector — Data Preprocessing Pipeline

Takes the ICDAR English Monograph OCR Dataset (preprocessed CSV)
and produces three outputs for the Zig runtime:

1. wordlist.txt     — Unique words from ground truth (→ MPHF dictionary)
2. confusion.csv    — Mined OCR confusion pairs with probabilities (→ ConfusionMatrix)
3. gru_train.txt    — Clean ground truth text for char-level GRU training
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

def normalize_word(word: str) -> str:
    """Lowercase, strip non-alpha except apostrophes."""
    word = word.lower().strip()
    # Keep only alpha + apostrophe
    word = re.sub(r"[^a-z']", "", word)
    return word

def extract_words(text: str) -> list[str]:
    """Split text into words, normalize."""
    raw = re.split(r'\s+', text)
    return [w for r in raw if (w := normalize_word(r)) and len(w) >= 2]

def mine_confusion_pairs(ocr_aligned: str, gs_aligned: str) -> list[tuple[str, str]]:
    """
    Walk the aligned strings character by character.
    When they differ, extract (ocr_pattern, gs_pattern) pairs.
    
    The alignment uses '@' for gaps (insertions/deletions).
    """
    pairs = []
    i = 0
    n = min(len(ocr_aligned), len(gs_aligned))
    
    while i < n:
        if ocr_aligned[i] == gs_aligned[i]:
            i += 1
            continue
        
        # Found a mismatch — extend to get the full error span
        j = i
        while j < n and ocr_aligned[j] != gs_aligned[j]:
            j += 1
        
        ocr_span = ocr_aligned[i:j].replace('@', '')
        gs_span = gs_aligned[i:j].replace('@', '')
        
        # Only keep short, meaningful patterns (1-4 chars each)
        if 0 < len(ocr_span) <= 4 and 0 < len(gs_span) <= 4:
            # Lowercase both
            pairs.append((ocr_span.lower(), gs_span.lower()))
        
        i = j
    
    return pairs

def main():
    data_dir = Path("data")
    csv_path = data_dir / "preprocessed_icdar_data_eng_mono.csv"
    out_dir = Path("data/processed")
    out_dir.mkdir(exist_ok=True)
    
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Download the dataset first.")
        sys.exit(1)
    
    print(f"Reading {csv_path}...")
    
    all_words = Counter()
    confusion_counts = Counter()
    total_chars = 0
    total_errors = 0
    gru_lines = []
    
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        row_count = 0
        
        for row in reader:
            row_count += 1
            ocr_text = row.get('OCR_toInput', '')
            ocr_aligned = row.get('OCR_aligned', '')
            gs_aligned = row.get('GS_aligned', '')
            
            # 1. Extract words from ground truth
            if gs_aligned:
                clean_gs = gs_aligned.replace('@', '')
                words = extract_words(clean_gs)
                for w in words:
                    all_words[w] += 1
                
                # GRU training text — clean ground truth
                gru_lines.append(clean_gs.lower())
            
            # 2. Mine confusion pairs
            if ocr_aligned and gs_aligned:
                total_chars += len(gs_aligned)
                pairs = mine_confusion_pairs(ocr_aligned, gs_aligned)
                total_errors += len(pairs)
                for pair in pairs:
                    confusion_counts[pair] += 1
    
    print(f"Processed {row_count} rows")
    print(f"Total chars: {total_chars}, errors: {total_errors} ({total_errors/max(total_chars,1)*100:.2f}%)")
    
    # --- Output 1: Word list ---
    wordlist_path = out_dir / "wordlist.txt"
    # Keep words that appear at least 2 times
    vocab = sorted([w for w, c in all_words.items() if c >= 2])
    with open(wordlist_path, 'w', encoding='utf-8') as f:
        for w in vocab:
            f.write(w + '\n')
    print(f"Wrote {len(vocab)} unique words to {wordlist_path}")
    print(f"  (filtered from {len(all_words)} total unique words)")
    
    # --- Output 2: Confusion pairs ---
    confusion_path = out_dir / "confusion.csv"
    # Sort by count descending, keep top 200
    sorted_pairs = sorted(confusion_counts.items(), key=lambda x: -x[1])
    with open(confusion_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['from', 'to', 'count', 'probability'])
        for (from_pat, to_pat), count in sorted_pairs[:200]:
            # Probability = count / total_errors (relative frequency among errors)
            prob = count / max(total_errors, 1)
            writer.writerow([from_pat, to_pat, count, f"{prob:.6f}"])
    print(f"Wrote {min(len(sorted_pairs), 200)} confusion pairs to {confusion_path}")
    print(f"  Top 10 confusion pairs:")
    for (from_pat, to_pat), count in sorted_pairs[:10]:
        prob = count / max(total_errors, 1)
        print(f"    '{from_pat}' -> '{to_pat}': {count} ({prob:.4f})")
    
    # --- Output 3: GRU training text ---
    gru_path = out_dir / "gru_train.txt"
    with open(gru_path, 'w', encoding='utf-8') as f:
        for line in gru_lines:
            f.write(line.strip() + '\n')
    total_gru_chars = sum(len(l) for l in gru_lines)
    print(f"Wrote {len(gru_lines)} lines ({total_gru_chars} chars) to {gru_path}")
    
    # --- Summary stats ---
    print(f"\n--- Summary ---")
    print(f"Vocabulary size: {len(vocab)} words (freq >= 2)")
    print(f"Confusion patterns: {len(sorted_pairs)} unique pairs")
    print(f"GRU training data: {total_gru_chars} characters")
    print(f"Estimated sizes:")
    print(f"  MPHF dict: ~{len(vocab) * 6 / 1024:.1f} KB ({len(vocab)} keys × 6 bytes)")
    print(f"  Confusion: ~{min(len(sorted_pairs), 200) * 20 / 1024:.1f} KB")
    print(f"  GRU model: ~300 KB (128 hidden, int8)")
    total_kb = len(vocab) * 6 / 1024 + 4 + 300
    print(f"  Total estimated: ~{total_kb:.0f} KB ({total_kb/1024:.1f} MB)")

if __name__ == '__main__':
    main()
