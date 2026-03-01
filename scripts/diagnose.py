"""Quick diagnostic of the evaluation data to identify improvement opportunities."""
import csv

with open('data/preprocessed_icdar_data_eng_mono.csv', 'r', encoding='utf-8', errors='replace') as f:
    rows = list(csv.DictReader(f))

# 1. Check alignment lengths
same_len = sum(1 for r in rows if len(r.get('OCR_aligned','')) == len(r.get('GS_aligned','')))
print(f"Rows with same-length alignment: {same_len}/{len(rows)}")

# 2. Show first 3 aligned samples
for r in rows[:3]:
    ocr = r.get('OCR_aligned', '')[:60]
    gs  = r.get('GS_aligned', '')[:60]
    print(f"\nOCR: {ocr}")
    print(f"GS:  {gs}")

# 3. Extract word-level errors from character alignment
import re
def extract_word_errors(ocr_aligned, gs_aligned):
    """Use char alignment to find which words contain errors."""
    if len(ocr_aligned) != len(gs_aligned):
        return [], []
    
    # Build error mask
    error_mask = [ocr_aligned[i] != gs_aligned[i] for i in range(len(ocr_aligned))]
    
    # Extract OCR words with their error status
    ocr_clean = ocr_aligned.replace('@', ' ')
    words_with_errors = []
    words_correct = []
    
    i = 0
    while i < len(ocr_clean):
        if ocr_clean[i].isalpha():
            j = i
            while j < len(ocr_clean) and ocr_clean[j].isalpha():
                j += 1
            word = ocr_clean[i:j].lower()
            has_error = any(error_mask[i:j])
            if len(word) >= 2:
                if has_error:
                    words_with_errors.append(word)
                else:
                    words_correct.append(word)
            i = j
        else:
            i += 1
    return words_with_errors, words_correct

total_error_words = 0
total_correct_words = 0
for r in rows:
    errs, corrs = extract_word_errors(r.get('OCR_aligned',''), r.get('GS_aligned',''))
    total_error_words += len(errs)
    total_correct_words += len(corrs)

total = total_error_words + total_correct_words
print(f"\n\n=== Character-alignment based word errors ===")
print(f"Total words: {total}")
print(f"Words with char errors: {total_error_words} ({total_error_words/max(total,1)*100:.1f}%)")
print(f"Words correct: {total_correct_words} ({total_correct_words/max(total,1)*100:.1f}%)")
