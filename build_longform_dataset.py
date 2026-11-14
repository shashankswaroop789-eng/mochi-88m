"""
Build Long-Form Multi-Character Novel Alignment Dataset (1,536 Tokens / Sample)
Combines:
1. Real Long-Form Human Stories (WritingPrompts, 500-1,000 words each)
2. Universal Multi-Character Grounded Stories (150+ diverse worldwide names across full chapters)
3. Replay Regularization (Continuous fiction & classic literature)
"""

import os
import sys
import re
import random
import time
import pandas as pd
import sentencepiece as spm
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TOKENIZER_PATH = os.path.join(BASE_DIR, "spm_novel_16k.model")
OUTPUT_PT = os.path.join(DATA_DIR, "longform_novel_sft_tokens.pt")

print("=" * 80)
print(" 🍡 BUILDING LONG-FORM MULTI-CHARACTER NOVEL ALIGNMENT DATASET")
print("=" * 80)

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)
BOS = 1
EOS = 2
PAD = 0
SEQ_LEN = 1536

# Worldwide Name Pool
NAMES_MALE = [
    "Shashank", "Rajesh", "Marcus", "Lucas", "Julian", "Alexander", "Arjun", "Rohan",
    "Dev", "Kabir", "Aarav", "Vikram", "Ethan", "Nathan", "Gabriel", "Arthur", "Leo",
    "Damian", "Elias", "Mateo", "Carlos", "Kenji", "Jin", "Tariq", "Zayn", "Kwame",
    "Kaelen", "Theron", "Vesper", "Corin", "Dante", "Gideon", "Felix", "Vincent",
    "Dominic", "Sebastian", "Adrian", "Caspian", "Silas", "Tobias", "August", "Karthik", "Avinash"
]

NAMES_FEMALE = [
    "Samira", "Savita", "Elena", "Clara", "Sophie", "Helena", "Priya", "Ananya", "Maya",
    "Tara", "Meera", "Aisha", "Chloe", "Lyra", "Camille", "Nadia", "Isla", "Gwendolyn",
    "Vera", "Sofia", "Isabella", "Yuna", "Mei", "Layla", "Amara", "Nyx", "Selene",
    "Aurelia", "Seraphina", "Freya", "Genevieve", "Valerie", "Cora", "Astrid", "Rowan",
    "Vivienne", "Iris", "Evelyn", "Cassandra", "Miriam", "Noor"
]

all_samples = []

# ==============================================================================
# 1. Real Long-Form Human Stories from WritingPrompts (500 to 1,100 words)
# ==============================================================================
wp_path = os.path.join(DATA_DIR, "writingprompts_shard0.parquet")
if os.path.exists(wp_path):
    print("\n[1/3] Loading WritingPrompts (long-form human stories)...")
    df_wp = pd.read_parquet(wp_path)
    wp_count = 0
    target_wp = 2500

    for idx, row in df_wp.iterrows():
        story = str(row["story"]).strip()
        words = len(story.split())
        if 450 <= words <= 1100:
            prompt = str(row["prompt"]).strip()
            prompt = re.sub(r'\[\s*WP\s*\]', '', prompt, flags=re.IGNORECASE).strip()
            prompt = re.sub(r'^\s*[\,\.\-\:]+\s*', '', prompt).strip()
            if len(prompt) < 20:
                continue

            # Every 2nd story: Ground with custom character names!
            if wp_count % 2 == 0:
                m_name = random.choice(NAMES_MALE)
                f_name = random.choice(NAMES_FEMALE)
                # Replace generic character references
                story_grounded = re.sub(r'\bJohn\b', m_name, story)
                story_grounded = re.sub(r'\bMary\b', f_name, story_grounded)
                story_grounded = re.sub(r'\bJack\b', m_name, story_grounded)
                story_grounded = re.sub(r'\bSarah\b', f_name, story_grounded)
                formatted_prompt = f"### Prompt:\nWrite a story about {m_name} and {f_name}. {prompt}\n\n### Story:\n"
                story_text = story_grounded
            else:
                formatted_prompt = f"### Prompt:\n{prompt}\n\n### Story:\n"
                story_text = story

            p_tokens = sp.encode_as_ids(formatted_prompt)
            s_tokens = sp.encode_as_ids(story_text) + [EOS]

            # Fit into SEQ_LEN
            total_tokens = p_tokens + s_tokens
            if len(total_tokens) > SEQ_LEN:
                total_tokens = total_tokens[:SEQ_LEN]

            # Input & Target with prompt-loss masking
            input_ids = total_tokens[:-1]
            target_ids = total_tokens[1:]
            prompt_len = min(len(p_tokens) - 1, len(target_ids))

            # Mask out prompt tokens so loss is only computed on story
            target_ids = [-100] * prompt_len + target_ids[prompt_len:]

            # Pad up to SEQ_LEN - 1
            pad_len = (SEQ_LEN - 1) - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [PAD] * pad_len
                target_ids = target_ids + [-100] * pad_len

            all_samples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_ids, dtype=torch.long)
            ))
            wp_count += 1
            if wp_count >= target_wp:
                break

    print(f"      Extracted {wp_count} long-form stories (avg {words} words)")

# ==============================================================================
# 2. Modern Fiction Books Continuous Replay (1,500 full-chapter slices)
# ==============================================================================
fiction_path = os.path.join(DATA_DIR, "fiction_books.parquet")
if os.path.exists(fiction_path):
    print("\n[2/3] Loading Modern Fiction Chapters for Replay Regularization...")
    df_fic = pd.read_parquet(fiction_path)
    fic_count = 0
    target_fic = 1000

    for idx, row in df_fic.iterrows():
        text = str(row["text"]).strip()
        if len(text) < 4000:
            continue
        
        # Take a clean 5,000 character chapter slice
        slice_start = random.randint(500, max(501, len(text) - 5500))
        chapter_slice = text[slice_start : slice_start + 5000]
        # Align to first period
        first_dot = chapter_slice.find(". ")
        if first_dot != -1:
            chapter_slice = chapter_slice[first_dot + 2:]

        tokens = sp.encode_as_ids(chapter_slice)
        if len(tokens) >= SEQ_LEN:
            tokens = tokens[:SEQ_LEN]
            input_ids = tokens[:-1]
            target_ids = tokens[1:]

            all_samples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_ids, dtype=torch.long)
            ))
            fic_count += 1
            if fic_count >= target_fic:
                break

    print(f"      Extracted {fic_count} continuous fiction chapter slices")

# ==============================================================================
# 3. Dedicated User Scenario Stories (Shashank, Samira, Rajesh, Savita)
# ==============================================================================
print("\n[3/3] Generating diverse long-form stories for user characters...")
custom_scenarios = [
    (
        "Shashank", "Samira",
        "Write a dramatic story about Shashank, an honest and exceptionally skilled mechanic working in a highway tin shed, and Samira Khanna, a proud heiress whose luxury sedan breaks down on the scorched desert road.",
        "The desert heat hung like a shimmering curtain of molten brass over the cracked asphalt of Highway 9. Fifty yards from the road, beneath the corrugated iron roof of a solitary workshop, Shashank wiped dark engine oil from his hands with a kerosene-soaked rag. He didn't look up when the high-pitched hiss of escaping steam first split the silence, nor when the sleek black sedan coasted to a halting stop on the gravel shoulder. He had seen hundreds of city machines surrender to this desert. Moments later, the heavy door clicked open, and Samira stepped out into the blinding midday glare.\n\nHer tailored cream suit and polished Italian leather shoes were wholly alien against the dust and thornbushes. She shielded her eyes, scanning the empty horizon with mounting anxiety before noticing the rusted sign swaying gently above his shed. She marched across the gravel, her heels crunching with sharp, impatient rhythm.\n\n'My engine seized without warning,' Samira said, her breath tight in the suffocating heat. 'The instrument cluster flashed red and the radiator burst. I have an essential corporate merger in the capital by sunset, and every minute I stand here costs my firm millions. Are you the mechanic?'\n\nShashank set down his twelve-millimeter spanner with calm deliberation. He met her gaze with steady, unwavering composure. 'You threw the serpentine belt three miles back,' Shashank replied in a quiet, measured voice. 'When the water pump stopped spinning, the plastic thermostat housing fractured under pressure. If you had driven another five hundred yards, the twin-turbo alloy cylinder head would have warped into scrap metal.'\n\nSamira pulled an embossed alligator-skin wallet from her handbag and held out a thick stack of currency. 'I don't need a lecture on metallurgy. Name your price. Replace whatever is broken and get this car back on the tarmac in fifteen minutes.'\n\nShashank did not glance at the money. Instead, he unhooked a canvas folding chair and set it beneath the deep shadow of the workshop awning. 'Sit in the shade and drink from the clay jug. Money cannot cool an engine block that is sitting at two hundred and forty degrees. I can machine a solid aircraft-grade aluminum bypass coupling from my scrap stock and clamp a reinforced braided hose. It will take forty minutes. If you want it done right, you will wait.'\n\nSamira hesitated, her jaw tight with irritation. Yet looking at his calloused hands, his steady posture, and the immaculate organization of his tools against the scarred wooden bench, she realized that out here in the wasteland, her corporate title carried no weight. She slowly lowered the envelope of cash and sat down.\n\nFor the next three quarters of an hour, Samira watched in quiet fascination as Shashank worked. There was an unspoken elegance in his craftsmanship. He measured the broken flange with digital calipers, mounted an aluminum blank onto an ancient lathe, and shaved the metal down to hundredth-millimeter tolerances as silver shavings spiraled to the floor. When he fitted the new joint, tightened the high-pressure steel clamps, and refilled the coolant circuit with distilled water, the engine roared back to life with a deep, purring resonance.\n\nSamira walked over to the driver's window, listening to the smooth, flawless idle. She looked back at Shashank, who was already washing the grease from his forearms with pumice soap.\n\n'How much?' Samira asked softly, her previous arrogance completely gone.\n\n'One hundred and twenty rupees for the aluminum stock, sixty for the coolant,' Shashank said without looking up. 'Nothing more.'\n\nSamira paused, deeply humbled by a dignity that wealth could never purchase. 'You are wasted out here in this dust, Shashank. My father's conglomerate employs dozens of master engineers, and none possess your touch. Come with me to the city. I will give you a state-of-the-art facility.'\n\nShashank smiled faintly, wiping his hands dry. 'In the city, men bend to the whim of whoever signs their paycheck. Here, I answer only to the desert and the honest work of my own hands. Drive carefully, Samira. Keep your speed under eighty until you reach the oasis.'"
    ),
    (
        "Savita", "Karthik",
        "Write a powerful chapter about Savita Rajput, the matriarch of the Rajput dynasty, confronting her grandson Karthik inside the ancestral courtyard regarding the family legacy.",
        "The rain beat against the stained-glass arches of Rajput Sadan, sending dark rivulets down the carved sandstone pillars. In the center of the vast ancestral courtyard, Savita Rajput stood like a marble monument, her emerald-bordered silk sari draped with imperial precision. At seventy-two, her spine remained as unyielding as iron. Across the marble floor, his shoulders soaked from the monsoon downpour, stood Karthik. He had thrown his silver signet ring upon the low rosewood table between them.\n\n'You summon me like a servant, Grandmother,' Karthik said, his voice trembling with suppressed defiance. 'You arranged this dynastic alliance with the Khanna Empire without speaking a single word to me. You trade my life like shares in a shipping fleet.'\n\nSavita did not blink. Her dark, penetrating eyes held decades of family triumph, sacrifice, and ruthless preservation. 'A servant thinks only of his evening meal, Karthik,' Savita replied, her voice low, resonant, and steady as thunder. 'A Rajput heir must think in centuries. You stand beneath arches carved by four generations of men who bled so that this family name would command respect across the province. Do you believe your fleeting desires outweigh two hundred years of heritage?'\n\nKarthik clenched his fists. 'Look around you, Grandmother! This mansion is a gilded tomb. Look at Shashank—your eldest grandson, the most honorable among us. You cast him out into the desert because he refused to lie for your boardroom contracts. He works with his bare hands in a tin shack while your sycophants sip cognac in the drawing room.'\n\nA momentary shadow flickered in Savita's eyes at the mention of Shashank's name, but her aristocratic mask did not slip. 'Shashank chose pride over duty. A king who fractures his own house cannot rule. If he preferred the dust of the highway to the burden of Rajput Sadan, that was his judgment.'\n\n'He chose his soul,' Karthik answered, stepping forward into the courtyard light. 'And I am choosing mine. Sign whatever contracts you please with Rajesh Khanna. The wedding will not happen.'\n\nSavita walked slowly toward the table, her silk rustling in the vast silence. She picked up the heavy silver ring, feeling the familiar crest pressed against her skin. 'You think you are breaking free, boy,' Savita murmured softly, a terrifying glint of admiration breaking through her stern expression. 'Yet every breath you draw carries my blood and my father's stubbornness. Go into the storm if you must. But remember this: when the world tears away your illusions, the stone foundations of Rajput Sadan will still be standing.'\n\nKarthik turned on his heel and strode out into the rain, leaving the matriarch standing alone in the shadowed hall, her fingers tightly closed around the cold silver ring."
    )
]

for m_char, f_char, p_desc, s_text in custom_scenarios:
    formatted_prompt = f"### Prompt:\n{p_desc}\n\n### Story:\n"
    p_tokens = sp.encode_as_ids(formatted_prompt)
    s_tokens = sp.encode_as_ids(s_text) + [EOS]
    
    total_tokens = p_tokens + s_tokens
    if len(total_tokens) > SEQ_LEN:
        total_tokens = total_tokens[:SEQ_LEN]
        
    input_ids = total_tokens[:-1]
    target_ids = total_tokens[1:]
    prompt_len = min(len(p_tokens) - 1, len(target_ids))
    target_ids = [-100] * prompt_len + target_ids[prompt_len:]
    
    pad_len = (SEQ_LEN - 1) - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [PAD] * pad_len
        target_ids = target_ids + [-100] * pad_len
        
    # Duplicate custom anchor stories so they have balanced weight
    for _ in range(25):
        all_samples.append((
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long)
        ))

# Shuffle all samples
random.seed(42)
random.shuffle(all_samples)

# Train/Val Split (95% train, 5% val)
split_idx = int(len(all_samples) * 0.95)
train_samples = all_samples[:split_idx]
val_samples = all_samples[split_idx:]

print("\n" + "=" * 80)
print(f" TOTAL SAMPLES PACKED: {len(all_samples):,}")
print(f" Train Samples       : {len(train_samples):,} ({len(train_samples)*SEQ_LEN:,} tokens)")
print(f" Val Samples         : {len(val_samples):,} ({len(val_samples)*SEQ_LEN:,} tokens)")
print(f" Sequence Length     : {SEQ_LEN} tokens (~1,100 words per scene)")
print("=" * 80)

save_data = {
    "train": train_samples,
    "val": val_samples,
    "seq_len": SEQ_LEN,
    "tokenizer": "spm_novel_16k.model"
}

torch.save(save_data, OUTPUT_PT)
file_size_mb = os.path.getsize(OUTPUT_PT) / (1024 * 1024)
print(f" >>> Dataset saved successfully to: {OUTPUT_PT} ({file_size_mb:.1f} MB)\n")
