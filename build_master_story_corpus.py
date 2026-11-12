"""
Assembles a world-class, balanced master training corpus for ModernGPT 88M:
1. 50 MB of TinyStories GPT-4 Clean (perfect grammar, pronoun tracking, spatial logic, dialogue conventions)
2. 50 MB of BookCorpus Fiction & Romance (adult novel pacing, romance, emotional tension, vocabulary)
3. 10 MB of Curated Contemporary Romance (Stella & Christian, Eva & Gideon, modern NYC setting)
Total: ~110 MB of pure, verified English narrative.
"""

import os
import re
import pyarrow.parquet as pq

OUTPUT_CORPUS = "master_story_corpus.txt"

def clean_bookcorpus_text(text):
    # Remove Smashwords standard license blurb
    text = re.sub(r'This ebook is licensed for your personal enjoyment.*?Thank you for respecting the hard work of this author\.', '', text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'Smashwords (?:Ebook )?Edition', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Discover other titles by .*? at Smashwords.*', '', text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'About the Author.*', '', text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'Connect with .*? online:.*', '', text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Strip leading metadata to start at chapter/prologue
    m = re.search(r'\b(Chapter\s+(?:One|1|I\b)|Prologue|Part\s+One)\b', text, re.IGNORECASE)
    if m and m.start() < 5000:
        text = text[m.start():]
        
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def main():
    print("=== Building Master Story Corpus for ModernGPT 88M ===")
    
    corpus_parts = []
    
    # 1. TinyStories GPT-4 Clean (50 MB)
    if os.path.exists("tinystories_50mb.txt"):
        print("[1/3] Loading TinyStories GPT-4 Clean...")
        with open("tinystories_50mb.txt", "r", encoding="utf-8", errors="ignore") as f:
            ts_text = f.read()
        print(f"  TinyStories text: {len(ts_text)/(1024*1024):.2f} MB")
        corpus_parts.append(ts_text)
    else:
        print("  WARNING: tinystories_50mb.txt not found!")

    # 2. BookCorpus Fiction & Romance (~50 MB target)
    if os.path.exists("bookcorpus_shard0.parquet"):
        print("[2/3] Extracting ~50 MB of cleaned fiction & romance novels from BookCorpus...")
        table = pq.read_table("bookcorpus_shard0.parquet")
        
        romance_keywords = [
            'romance', 'love', 'kiss', 'heart', 'desire', 'passion', 'bride', 'wedding',
            'affair', 'secret', 'darling', 'billionaire', 'prince', 'king', 'queen',
            'lover', 'girl', 'boy', 'woman', 'man', 'night', 'dark', 'shadow',
            'magic', 'vampire', 'wolf', 'shifter', 'forever', 'together', 'promise',
            'destiny', 'fate', 'touch', 'seduce', 'seduction', 'pleasure', 'sin', 'wicked',
            'danger', 'thriller', 'mystery', 'detective', 'case', 'blood',
            'novel', 'chronicles', 'tales', 'story', 'journey'
        ]
        skip_keywords = [
            'poem', 'poetry', 'recipe', 'diet', 'bible', 'obama', 'trump', 'essay', 
            'meeting', 'business', 'money', 'guide', 'exercise', 'sex-positions', 
            'hacking', 'programming', 'manual'
        ]
        
        bc_chars = 0
        target_bc_chars = 52 * 1024 * 1024  # ~52 MB
        bc_books_used = 0
        
        for i in range(len(table)):
            if bc_chars >= target_bc_chars:
                break
            row = table.slice(i, 1).to_pandas().iloc[0]
            title = row['title'].lower()
            text = row['text']
            
            if any(s in title for s in skip_keywords):
                continue
            if len(text) < 40000:
                continue
            if any(k in title for k in romance_keywords):
                cleaned = clean_bookcorpus_text(text)
                if len(cleaned) > 20000:
                    corpus_parts.append("\n\n<|endoftext|>\n\n" + cleaned)
                    bc_chars += len(cleaned)
                    bc_books_used += 1

        print(f"  Extracted {bc_books_used} novels from BookCorpus ({bc_chars/(1024*1024):.2f} MB)")
    else:
        print("  WARNING: bookcorpus_shard0.parquet not found!")

    # 3. Curated Modern Romance Books (~10 MB)
    if os.path.exists("pure_modern_english_corpus.txt"):
        print("[3/3] Loading Curated Contemporary Romance Books...")
        with open("pure_modern_english_corpus.txt", "r", encoding="utf-8", errors="ignore") as f:
            romance_text = f.read()
        print(f"  Curated romance text: {len(romance_text)/(1024*1024):.2f} MB")
        corpus_parts.append("\n\n<|endoftext|>\n\n" + romance_text)
    else:
        print("  WARNING: pure_modern_english_corpus.txt not found!")

    # Combine into single file
    print("Writing master corpus to disk...")
    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as out:
        for part in corpus_parts:
            out.write(part)
            out.write("\n\n")

    final_size_mb = os.path.getsize(OUTPUT_CORPUS) / (1024 * 1024)
    print(f"=== Successfully Created {OUTPUT_CORPUS} ({final_size_mb:.2f} MB) ===")

if __name__ == "__main__":
    main()
