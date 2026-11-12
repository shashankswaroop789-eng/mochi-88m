"""
Scrub and Clean Romance Corpus
Purges all book titles, running headers, chapter numbers, and EPUB artifacts
from pure_modern_english_corpus.txt so the model never generates random book titles
or chapter numbers mid-story.
"""

import os
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DATA_DIR, "pure_modern_english_corpus.txt")
OUTPUT_FILE = os.path.join(DATA_DIR, "scrubbed_clean_romance_corpus.txt")

# Exact titles and patterns that pollute the model's text stream
TITLES_TO_PURGE = [
    r"The Matcha Maker Caf[eé]?",
    r"Matcha Maker Caf[eé]?",
    r"Bared to You(?:\s*-\s*A Crossfire Novel)?",
    r"Crossfire(?:\s+Novel)?",
    r"Twisted Lies",
    r"Twisted Games",
    r"Twisted Love",
    r"Twisted Hate",
    r"The Kiss Thief",
    r"Kiss Thief",
    r"Pretty Reckless",
    r"Ugly Love",
    r"Fifty Shades(?: of Grey)?",
    r"Sylvia Day",
    r"Ana Huang",
    r"Colleen Hoover",
    r"L\.?\s*J\.?\s*Shen",
    r"E\.?\s*L\.?\s*James",
    r"Lily Chen"
]

def scrub_corpus():
    print(f"Reading raw corpus: {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    print(f"Original size: {len(text):,} chars | {len(text.split()):,} words")

    # 1. Purge book titles and author names
    for title in TITLES_TO_PURGE:
        text = re.sub(title, "", text, flags=re.IGNORECASE)

    # 2. Purge standalone chapter headings (e.g., "Chapter 35", "CHAPTER IV", "Chapter Twelve")
    # Matches patterns like "Chapter 35", "Chapter 12 -", "CHAPTER ONE"
    text = re.sub(r'(?i)\bchapter\s+(?:\d+|[ivxldcm]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|\w+ty(?:\s+\w+)?)\b', '', text)
    
    # Purge prologue/epilogue headings
    text = re.sub(r'(?i)\b(?:prologue|epilogue|acknowledgments?|table of contents|copyright)\b', '', text)

    # 3. Clean line by line
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        # Skip empty lines or pure number lines (page numbers)
        if not line or re.match(r'^\d+$', line):
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        
        # If line starts with weird punctuation or residual tags
        line = re.sub(r'^[—–\-\*\#\.\,\s]+', '', line).strip()
        if len(line) < 3:
            continue
        
        cleaned_lines.append(line)

    reconstructed = "\n".join(cleaned_lines)
    # Collapse 3+ newlines to 2
    reconstructed = re.sub(r'\n{3,}', '\n\n', reconstructed)
    # Clean up double spaces left from stripped words
    reconstructed = re.sub(r'[ \t]{2,}', ' ', reconstructed)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(reconstructed)

    print(f"Scrubbed corpus saved to: {OUTPUT_FILE}")
    print(f"Clean size: {len(reconstructed):,} chars | {len(reconstructed.split()):,} words")

    # Verify purge
    residual_titles = 0
    for title in TITLES_TO_PURGE:
        count = len(re.findall(title, reconstructed, re.IGNORECASE))
        residual_titles += count
    residual_chapters = len(re.findall(r'(?i)\bchapter\s+\d+\b', reconstructed))

    print(f"Verification: Residual titles = {residual_titles} | Residual chapter headers = {residual_chapters}")

if __name__ == "__main__":
    scrub_corpus()
