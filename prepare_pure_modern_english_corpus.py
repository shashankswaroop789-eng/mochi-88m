import os
import sys
import re
import zipfile
import pypdf
import sentencepiece as spm

# Force utf-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR = os.path.join(DATA_DIR, "custom_books")
OUT_CORPUS = os.path.join(DATA_DIR, "pure_modern_english_corpus.txt")
TOKENIZER_PREFIX = os.path.join(DATA_DIR, "spm_modern")

# ONLY 100% Verified English Modern Romance Books
VERIFIED_ENGLISH_FILES = [
    'Twisted Lies (Ana Huang) (z-library.sk, 1lib.sk, z-lib.sk).epub',
    'Bared to You - A Crossfire Novel (Sylvia Day) (z-library.sk, 1lib.sk, z-lib.sk).epub',
    'The Matcha Maker Caf (Lily Chen) (z-library.sk, 1lib.sk, z-lib.sk).epub',
    'Ugly Love (Colleen Hoover) (z-library.sk, 1lib.sk, z-lib.sk).pdf',
    'Pretty-Reckless-_L.-J.-Shen_-_z-library.sk_-1lib.sk_-z-lib.sk_.txt',
    'The-Kiss-Thief-_L.-J.-Shen_-_z-library.sk_-1lib.sk_-z-lib.sk_.txt'
]

def clean_modern_text(text):
    # Strip HTML entities
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"').replace("&apos;", "'")
    text = text.replace("&#39;", "'").replace("&lt;", "<").replace("&gt;", ">")

    # Normalize weird quotes and dashes to standard quotes
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", " — ").replace("–", " - ")

    # Strip web download artifacts & boilerplate
    boilerplates = [
        r"z-library\.sk", r"1lib\.sk", r"z-lib\.sk", r"b-ok\.org",
        r"Downloaded by .*?@", r"All rights reserved\.", r"ISBN:? \d+[- ]?\d+[- ]?\d+",
        r"Published by .*?\n", r"Cover design by .*?\n"
    ]
    for bp in boilerplates:
        text = re.sub(bp, "", text, flags=re.IGNORECASE)

    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = []
    for line in lines:
        if re.match(r'^\d+$', line) or len(line) == 0:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        cleaned_lines.append(line)

    reconstructed = "\n".join(cleaned_lines)
    reconstructed = re.sub(r'\n{3,}', '\n\n', reconstructed)
    return reconstructed.strip()

def extract_epub(epub_path):
    print(f"  [EPUB] Extracting: {os.path.basename(epub_path)}...")
    full_text = []
    try:
        with zipfile.ZipFile(epub_path, 'r') as z:
            html_files = [f for f in z.namelist() if f.lower().endswith(('.html', '.xhtml', '.htm'))]
            html_files.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)])
            
            for hf in html_files:
                raw_bytes = z.read(hf)
                html_str = raw_bytes.decode('utf-8', errors='ignore')
                plain = re.sub(r'<[^>]+>', ' ', html_str)
                plain = re.sub(r'\s+', ' ', plain).strip()
                if len(plain) > 100:
                    full_text.append(plain)
        return "\n\n".join(full_text)
    except Exception as e:
        print(f"    - Error extracting EPUB: {e}")
        return ""

def extract_pdf(pdf_path):
    print(f"  [PDF]  Extracting: {os.path.basename(pdf_path)}...")
    full_text = []
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            txt = page.extract_text()
            if txt and len(txt.strip()) > 50:
                full_text.append(txt.strip())
        return "\n\n".join(full_text)
    except Exception as e:
        print(f"    - Error extracting PDF: {e}")
        return ""

def extract_txt(txt_path):
    print(f"  [TXT]  Reading:    {os.path.basename(txt_path)}...")
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"    - Error reading TXT: {e}")
        return ""

def assemble_pure_corpus():
    print("=" * 70)
    print("   ASSEMBLING 100% PURE ENGLISH CONTEMPORARY ROMANCE CORPUS")
    print("   (Zero Victorian | Zero German | Zero Portuguese | 100% Modern)")
    print("=" * 70)

    extracted_books = []
    total_chars = 0

    all_files = os.listdir(CUSTOM_DIR)
    for target in VERIFIED_ENGLISH_FILES:
        match = [f for f in all_files if f.startswith(target[:20])]
        if not match:
            continue
        fname = match[0]
        fpath = os.path.join(CUSTOM_DIR, fname)

        if fname.endswith('.epub'):
            raw = extract_epub(fpath)
        elif fname.endswith('.pdf'):
            raw = extract_pdf(fpath)
        elif fname.endswith('.txt'):
            raw = extract_txt(fpath)
        else:
            continue

        cleaned = clean_modern_text(raw)
        if len(cleaned) > 5000:
            extracted_books.append((fname, cleaned))
            total_chars += len(cleaned)
            print(f"    --> Cleaned English Text: {len(cleaned):>10,} characters")

    # Write 3 copies of each book to give dense contemporary training
    print("\n[Writing Combined Pure English Modern Romance Corpus (3x Multiplier)]")
    with open(OUT_CORPUS, 'w', encoding='utf-8') as out_f:
        for _ in range(3):
            for fname, book_text in extracted_books:
                out_f.write(book_text + "\n\n")

    final_size = os.path.getsize(OUT_CORPUS)
    print("-" * 70)
    print(f"Verified English Books:  {len(extracted_books)}")
    print(f"Total Unique Characters: {total_chars:,} ({total_chars / (1024*1024):.2f} MB)")
    print(f"Corpus Size with 3x:     {final_size:,} bytes ({final_size / (1024*1024):.2f} MB)")
    print(f"Saved to:                {OUT_CORPUS}")
    print("-" * 70)

def train_pure_tokenizer(vocab_size=4096):
    print(f"\n[Tokenizer] Training SentencePiece BPE model (vocab_size={vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=OUT_CORPUS,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        byte_fallback=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    print(f"[Tokenizer] Generated: {TOKENIZER_PREFIX}.model & {TOKENIZER_PREFIX}.vocab")

if __name__ == "__main__":
    assemble_pure_corpus()
    train_pure_tokenizer(vocab_size=4096)
