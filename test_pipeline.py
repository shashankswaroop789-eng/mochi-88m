import sys
import os
import torch
import sentencepiece as spm

sys.path.insert(0, 'C:/Users/user/.gemini/antigravity/scratch/gpt_from_scratch')
from modern_transformer import ModernGPT
from test_all_scenes_continuity import clean_tail, generate_scene

sp = spm.SentencePieceProcessor()
sp.load('C:/Users/user/.gemini/antigravity/scratch/gpt_from_scratch/spm_novel_16k.model')
ckpt = torch.load('C:/Users/user/.gemini/antigravity/scratch/gpt_from_scratch/mochi_novel_300m.pt', map_location='cuda', weights_only=False)
model = ModernGPT(**dict(ckpt['config'], dropout=0.0)).to('cuda')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Let's run S1 -> S2 -> S3 exactly as test_all_scenes_continuity does
s1_prompt = "### Prompt:\nWrite a short story about Shashank, a proud young mechanic working in a highway tin shed, and Rajesh Khanna, a wealthy businessman whose luxury car breaks down on the desert road.\n\n### Story:\n"
s1_text, _, _ = generate_scene(model, sp, s1_prompt, target_words=800)
s1_tail = clean_tail(s1_text)

user_inst = "Rajesh Khanna's personal investigator uncovers that Shashank is the outcast grandson of Savita Rajput, and Rajesh decides to visit the Rajput mansion."
s2_prompt = f"### Prompt:\nPrevious scene ending:\n{s1_tail}\n\nNext scene:\n{user_inst}\n\n### Story:\n"
s2_text, _, _ = generate_scene(model, sp, s2_prompt, target_words=800)
s2_tail = clean_tail(s2_text)
print("S2 clean tail:\n", repr(s2_tail))

s3_inst = "Rajesh Khanna visits Rajput Sadan to drop the bombshell proposal, rejecting Karthik and demanding Shashank Rajput."
s3_prompt = f"### Prompt:\nPrevious scene ending:\n{s2_tail}\n\nNext scene:\n{s3_inst}\n\n### Story:\n"
s3_text, words, _ = generate_scene(model, sp, s3_prompt, target_words=800)
print(f"\nS3 words: {words}")
print("Contains 'Savita':", "Savita" in s3_text)
print("Contains 'Rajput Sadan':", "Rajput Sadan" in s3_text)
print("Contains 'Karthik':", "Karthik" in s3_text)
print("\nFirst 400 chars of S3:\n", s3_text[:400])
