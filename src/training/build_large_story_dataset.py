"""
Build Master Large Story Dataset for Mochi-88M
Combines and cleans ~30,000 high-quality prompt-story pairs from:
1. Gryphe/Opus-WritingPrompts (Claude 3 Opus master stories)
2. stindardlogic/creative-writing-sft-50k (curated flash fiction & short stories)
3. euclaise/writingprompts (Reddit WritingPrompts)

Outputs: data/master_story_sft_dataset.jsonl
"""

import os
import sys
import json
import re
import random
import pyarrow.parquet as pq

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "master_story_sft_dataset.jsonl")

def clean_text(text):
    if not text:
        return ""
    # Normalize unicode quotes and dashes
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2014', ' -- ').replace('\u2013', ' - ')
    # Remove weird OCR or markdown formatting artifacts
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_prompt(p):
    p = clean_text(p)
    p = re.sub(r'\[\s*(WP|wp|OT|ot|EU|eu|CW|cw|PI|pi|RF|rf|IP|ip)\s*\]', '', p).strip()
    p = re.sub(r'^(Please\s+)?write\s+a\s+story\s+(using\s+the\s+following\s+writing\s+prompt:?|about|where)\s*', '', p, flags=re.IGNORECASE).strip()
    if p and not p[0].isupper():
        p = p[0].upper() + p[1:]
    return p

def main():
    print("=" * 70)
    print(" Compiling Master Story Dataset (~30,000 Samples)")
    print("=" * 70)

    dataset = []

    # 1. Opus WritingPrompts (Top tier quality)
    opus_files = [
        os.path.join(DATA_DIR, "opus-writing-prompts-1-raw.jsonl"),
        os.path.join(DATA_DIR, "opus-writing-prompts-2-raw.jsonl")
    ]
    opus_count = 0
    for op_f in opus_files:
        if os.path.exists(op_f):
            print(f"Processing {os.path.basename(op_f)}...")
            with open(op_f, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        p = clean_prompt(row.get("prompt", ""))
                        s = clean_text(row.get("story", ""))
                        words = s.split()
                        if len(words) > 320:
                            s = ' '.join(words[:320])
                            last_punct = max(s.rfind('.'), s.rfind('!'), s.rfind('?'))
                            if last_punct != -1:
                                s = s[:last_punct+1]
                        words_count = len(s.split())
                        if p and 80 <= words_count <= 350:
                            dataset.append({"prompt": p, "story": s, "source": "opus"})
                            opus_count += 1
                    except Exception:
                        continue
    print(f"Loaded {opus_count:,} stories from Opus-WritingPrompts.")

    # 2. Creative Writing SFT 50k
    cw_file = os.path.join(DATA_DIR, "creative-writing-sft-50k.jsonl")
    cw_count = 0
    if os.path.exists(cw_file):
        print(f"Processing {os.path.basename(cw_file)}...")
        with open(cw_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    convs = row.get("conversations", [])
                    if len(convs) >= 2:
                        raw_p = convs[0].get("value", "")
                        raw_s = convs[1].get("value", "")
                        # Filter for stories only, avoid poems or haikus
                        lower_p = raw_p.lower()
                        if any(w in lower_p for w in ['story', 'fiction', 'scene', 'tale', 'narrative', 'chapter', 'dialogue', 'character']):
                            if not any(w in lower_p for w in ['poem', 'haiku', 'sonnet', 'rhyme']):
                                p = clean_prompt(raw_p)
                                s = clean_text(raw_s)
                                words = len(s.split())
                                if p and 80 <= words <= 450:
                                    dataset.append({"prompt": p, "story": s, "source": "cw_sft"})
                                    cw_count += 1
                                    if cw_count >= 12000:
                                        break
                except Exception:
                    continue
    print(f"Loaded {cw_count:,} stories from Creative-Writing-SFT-50k.")

    # 3. Reddit WritingPrompts Parquet
    wp_file = os.path.join(DATA_DIR, "writingprompts_shard0.parquet")
    wp_count = 0
    if os.path.exists(wp_file):
        print(f"Processing {os.path.basename(wp_file)}...")
        table = pq.read_table(wp_file, columns=["prompt", "story"])
        # Sample rows
        num_rows = len(table)
        step = max(1, num_rows // 25000)
        for i in range(0, num_rows, step):
            try:
                row_p = str(table["prompt"][i])
                row_s = str(table["story"][i])
                p = clean_prompt(row_p)
                s = clean_text(row_s)
                words = len(s.split())
                if p and 80 <= words <= 450 and not p.lower().startswith("deleted"):
                    dataset.append({"prompt": p, "story": s, "source": "reddit_wp"})
                    wp_count += 1
                    if wp_count >= 12000:
                        break
            except Exception:
                continue
    print(f"Loaded {wp_count:,} stories from Reddit WritingPrompts.")

    # Shuffle dataset
    random.seed(42)
    random.shuffle(dataset)

    print(f"\nTotal compiled samples: {len(dataset):,}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved master dataset to: {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE)/(1024*1024):.1f} MB")

if __name__ == "__main__":
    main()
