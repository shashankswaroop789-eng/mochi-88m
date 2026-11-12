"""
Trains a clean Google SentencePiece BPE tokenizer on the 113MB master_story_corpus.txt.
Vocab size: 4096
"""

import os
import sys
import sentencepiece as spm

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DATA_DIR, "master_story_corpus.txt")
MODEL_PREFIX = os.path.join(DATA_DIR, "spm_hf_master")

def train_spm(vocab_size=4096):
    print(f"--- Training Google SentencePiece on Master Story Corpus (Vocab: {vocab_size}) ---")
    spm.SentencePieceTrainer.train(
        input=INPUT_FILE,
        model_prefix=MODEL_PREFIX,
        vocab_size=vocab_size,
        model_type='bpe',
        byte_fallback=True,                 # 100% lossless with byte fallback
        normalization_rule_name='identity', # Exact punctuation, quotes, case
        input_sentence_size=2000000,        # Sample up to 2 million lines for rich subwords
        shuffle_input_sentence=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    print("Tokenizer training complete!")
    print(f"  Model saved to: {MODEL_PREFIX}.model")
    print(f"  Vocab saved to: {MODEL_PREFIX}.vocab")

def test_spm():
    sp = spm.SentencePieceProcessor()
    sp.load(f"{MODEL_PREFIX}.model")
    
    test_strings = [
        "Christian Harper stepped out of his penthouse elevator into the sleek lobby.",
        "\"You shouldn't be here,\" Stella whispered, her pulse racing as his gaze trapped her.",
        "Lily smiled and held his hand. \"Let's go to the park!\" she said happily."
    ]
    
    print("\n--- Testing Tokenizer on Modern Dialogue & Fiction ---")
    for s in test_strings:
        ids = sp.encode_as_ids(s)
        decoded = sp.decode(ids)
        print(f"Original: {s}")
        print(f"Tokens ({len(ids)}): {ids[:10]}... -> Exact Match: {s == decoded}")

if __name__ == "__main__":
    train_spm(vocab_size=4096)
    test_spm()
