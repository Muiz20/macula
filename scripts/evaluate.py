"""
OCR Error Detector — Improved Evaluation (v4)

Three key improvements:
1. Uses CHARACTER-LEVEL alignment for ground truth (not word-set matching)
2. Filters confusion pairs to sub-word patterns only (removes whole-word subs like the↔and)
3. Sweeps GRU thresholds to find the optimal operating point

Also includes analysis of false positive/negative patterns.
"""

import csv
import re
import struct
import sys
import numpy as np
from pathlib import Path
from collections import Counter


def tokenize(text):
    words = []
    current = []
    for ch in text:
        if ch.isalpha() or ch == "'":
            current.append(ch.lower())
        else:
            if current:
                w = ''.join(current)
                if len(w) >= 2:
                    words.append(w)
                current = []
    if current:
        w = ''.join(current)
        if len(w) >= 2:
            words.append(w)
    return words


# --- MPHF ---
def fnv1a(data):
    h = 0x811c9dc5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def fnv1a_seeded(data, seed):
    h = 0x811c9dc5 ^ seed
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def fingerprint16(data):
    h = fnv1a(data)
    return ((h >> 16) ^ (h & 0xFFFF)) & 0xFFFF


class MphfLookup:
    def __init__(self, seeds, fps, n):
        self.seeds = seeds
        self.fps = fps
        self.n = n

    def contains(self, word):
        if self.n == 0: return False
        key = word.encode('utf-8')
        bucket = fnv1a(key) % len(self.seeds)
        seed = self.seeds[bucket]
        slot = fnv1a_seeded(key, seed) % self.n
        return self.fps[slot] == fingerprint16(key)


# --- GRU Scorer ---
CHAR_OFFSET = 32
VOCAB_SIZE = 96

class GruScorer:
    def __init__(self, is_, hs, w_ih, s1, w_hh, s2, b_ih, s3, b_hh, s4, w_out, s5, b_out, s6):
        self.hs = hs
        self.is_ = is_
        self.w_ih = w_ih.astype(np.float32).reshape(3*hs, is_) * s1
        self.w_hh = w_hh.astype(np.float32).reshape(3*hs, hs) * s2
        self.b_ih = b_ih.astype(np.float32) * s3
        self.b_hh = b_hh.astype(np.float32) * s4
        self.w_out = w_out.astype(np.float32).reshape(is_, hs) * s5
        self.b_out = b_out.astype(np.float32) * s6

    def score_word(self, word):
        if len(word) <= 1: return 0.0
        hs = self.hs
        h = np.zeros(hs, dtype=np.float32)
        total_nll = 0.0
        for i, ch in enumerate(word):
            x = max(0, min(ord(ch) - CHAR_OFFSET, VOCAB_SIZE - 1))
            z = 1.0 / (1 + np.exp(-np.clip(self.w_ih[:hs, x] + self.b_ih[:hs] + self.w_hh[:hs] @ h + self.b_hh[:hs], -50, 50)))
            r = 1.0 / (1 + np.exp(-np.clip(self.w_ih[hs:2*hs, x] + self.b_ih[hs:2*hs] + self.w_hh[hs:2*hs] @ h + self.b_hh[hs:2*hs], -50, 50)))
            n = np.tanh(self.w_ih[2*hs:3*hs, x] + self.b_ih[2*hs:3*hs] + r * (self.w_hh[2*hs:3*hs] @ h + self.b_hh[2*hs:3*hs]))
            h = (1 - z) * n + z * h
            if i + 1 < len(word):
                nx = max(0, min(ord(word[i+1]) - CHAR_OFFSET, VOCAB_SIZE - 1))
                logits = self.w_out @ h + self.b_out
                logits -= logits.max()
                total_nll -= logits[nx] - np.log(np.exp(logits).sum())
        return total_nll / (len(word) - 1)


def load_artifact(path):
    data = path.read_bytes()
    sc, kc, bs, cc, gb, _ = struct.unpack_from('<IIIIII', data, 8)
    off = 32
    seeds = list(struct.unpack_from(f'<{sc}I', data, off)); off += sc * 4
    fps = list(struct.unpack_from(f'<{kc}H', data, off)); off += kc * 2
    off += cc * 20  # skip confusion

    gru = None
    if gb > 0:
        gd = data[off:off+gb]
        is_ = struct.unpack_from('<I', gd, 0)[0]
        hs = struct.unpack_from('<I', gd, 4)[0]
        s = struct.unpack_from('<ffffff', gd, 8)
        o = 32
        w_ih = np.frombuffer(gd[o:o+3*hs*is_], dtype=np.int8); o += 3*hs*is_
        w_hh = np.frombuffer(gd[o:o+3*hs*hs], dtype=np.int8); o += 3*hs*hs
        b_ih = np.frombuffer(gd[o:o+3*hs], dtype=np.int8); o += 3*hs
        b_hh = np.frombuffer(gd[o:o+3*hs], dtype=np.int8); o += 3*hs
        w_out = np.frombuffer(gd[o:o+is_*hs], dtype=np.int8); o += is_*hs
        b_out = np.frombuffer(gd[o:o+is_], dtype=np.int8)
        gru = GruScorer(is_, hs, w_ih, s[0], w_hh, s[1], b_ih, s[2], b_hh, s[3], w_out, s[4], b_out, s[5])

    return MphfLookup(seeds, fps, kc), gru


def extract_word_labels_from_alignment(ocr_text, ocr_aligned, gs_aligned):
    """
    Extract per-word error labels using character-level alignment.
    
    Strategy: Walk the OCR text and for each word, find its characters in the
    aligned OCR and check if they differ from the GS aligned.
    
    For rows where alignment lengths don't match, fall back to comparing
    each OCR word against a ground truth word set from the GS text.
    """
    ocr_words = tokenize(ocr_text)

    if len(ocr_aligned) == len(gs_aligned) and len(ocr_aligned) > 10:
        # Good alignment — use char-level error detection
        # Build word error map from aligned strings
        error_words = set()
        correct_words = set()
        ocr_clean = ocr_aligned.replace('@', ' ')
        i = 0
        while i < len(ocr_clean):
            if ocr_clean[i].isalpha():
                j = i
                while j < len(ocr_clean) and ocr_clean[j].isalpha():
                    j += 1
                word = ocr_clean[i:j].lower()
                has_error = any(ocr_aligned[k] != gs_aligned[k] for k in range(i, j))
                if len(word) >= 2:
                    if has_error:
                        error_words.add(word)
                    else:
                        correct_words.add(word)
                i = j
            else:
                i += 1
        
        labels = {}
        for w in ocr_words:
            if w in error_words:
                labels[w] = True
            elif w in correct_words:
                labels[w] = False
            else:
                labels[w] = None  # ambiguous
        return labels
    
    # Fallback: word-set comparison (noisy but workable)
    gs_clean = gs_aligned.replace('@', '') if gs_aligned else ocr_aligned.replace('@', '')
    gs_word_set = set(tokenize(gs_clean))
    
    labels = {}
    for w in ocr_words:
        labels[w] = w not in gs_word_set
    return labels


def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    accuracy = (tp + tn) / max(total, 1)
    return precision, recall, f1, accuracy


def main():
    artifact_path = Path("data/ocr_detector.bin")
    csv_path = Path("data/preprocessed_icdar_data_eng_mono.csv")

    print("Loading artifact...")
    mphf, gru = load_artifact(artifact_path)

    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        rows = list(csv.DictReader(f))

    # Pre-compute GRU scores for all unique OCR words
    all_ocr_words = set()
    for row in rows:
        all_ocr_words.update(tokenize(row.get('OCR_toInput', '')))
    
    print(f"Unique OCR words: {len(all_ocr_words)}")
    print("Computing GRU scores for all words...")
    
    gru_scores = {}
    for i, w in enumerate(sorted(all_ocr_words)):
        if gru:
            gru_scores[w] = gru.score_word(w)
        if (i + 1) % 2000 == 0:
            print(f"  Scored {i+1}/{len(all_ocr_words)}...")

    # Extract labels
    print("Extracting word-level labels...")
    word_labels = {}  # word -> list of (is_error_bool)
    for row in rows:
        ocr_text = row.get('OCR_toInput', '')
        ocr_aligned = row.get('OCR_aligned', '')
        gs_aligned = row.get('GS_aligned', '')
        
        labels = extract_word_labels_from_alignment(ocr_text, ocr_aligned, gs_aligned)
        for w, label in labels.items():
            if label is not None:
                if w not in word_labels:
                    word_labels[w] = []
                word_labels[w].append(label)
    
    # For each word, majority vote on error label
    word_is_error = {}
    for w, labels in word_labels.items():
        error_count = sum(labels)
        word_is_error[w] = error_count > len(labels) / 2

    total_error = sum(word_is_error.values())
    total_correct = len(word_is_error) - total_error
    print(f"Labeled words: {len(word_is_error)} (errors: {total_error}, correct: {total_correct})")

    # Evaluate strategies
    print(f"\n{'='*70}")
    print(f"{'Strategy':<35} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Acc':>7}")
    print(f"{'='*70}")

    # Strategy 1: MPHF only
    tp = fp = tn = fn = 0
    for w, is_error in word_is_error.items():
        in_dict = mphf.contains(w)
        if is_error:
            if not in_dict: tp += 1
            else: fn += 1
        else:
            if in_dict: tn += 1
            else: fp += 1
    p, r, f1, acc = compute_metrics(tp, fp, tn, fn)
    print(f"{'MPHF only':<35} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {acc:>7.4f}")

    # Strategy 2: MPHF + GRU at various thresholds
    if gru:
        for thresh in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]:
            tp = fp = tn = fn = 0
            for w, is_error in word_is_error.items():
                score = gru_scores.get(w, 10.0)
                valid = mphf.contains(w) or score < thresh
                if is_error:
                    if not valid: tp += 1
                    else: fn += 1
                else:
                    if valid: tn += 1
                    else: fp += 1
            p, r, f1, acc = compute_metrics(tp, fp, tn, fn)
            print(f"{'MPHF + GRU (t=' + str(thresh) + ')':<35} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {acc:>7.4f}")

    # Strategy 3: GRU only (no dictionary)
    if gru:
        for thresh in [3.0, 4.0, 5.0]:
            tp = fp = tn = fn = 0
            for w, is_error in word_is_error.items():
                score = gru_scores.get(w, 10.0)
                valid = score < thresh
                if is_error:
                    if not valid: tp += 1
                    else: fn += 1
                else:
                    if valid: tn += 1
                    else: fp += 1
            p, r, f1, acc = compute_metrics(tp, fp, tn, fn)
            print(f"{'GRU only (t=' + str(thresh) + ')':<35} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {acc:>7.4f}")

    print(f"{'='*70}")

    # Error analysis: breakdown of false positives and false negatives
    print(f"\n=== Error Analysis (MPHF only) ===")
    fp_words = []
    fn_words = []
    for w, is_error in word_is_error.items():
        in_dict = mphf.contains(w)
        score = gru_scores.get(w, 10.0)
        if not is_error and not in_dict:
            fp_words.append((w, score))
        elif is_error and in_dict:
            fn_words.append((w, score))
    
    fp_words.sort(key=lambda x: -x[1])
    fn_words.sort(key=lambda x: x[1])
    
    print(f"\nFalse Positives (valid words flagged as errors): {len(fp_words)}")
    print(f"  Top 15 (could fix with GRU acceptance):")
    for w, s in fp_words[:15]:
        print(f"    '{w}' GRU={s:.2f} {'← GRU could accept' if s < 5 else ''}")
    
    print(f"\nFalse Negatives (error words in dictionary): {len(fn_words)}")
    print(f"  Top 15 (real-word errors, need context):")
    for w, s in fn_words[:15]:
        print(f"    '{w}' GRU={s:.2f}")


if __name__ == '__main__':
    main()
