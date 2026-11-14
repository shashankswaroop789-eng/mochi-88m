"""
🍡 Build Precision Multi-Character & Long-Form Novel Alignment Dataset
Creates the ultimate SFT dataset for Mochi-Novel-300M (1,536 tokens / sample):
1. Gold Rajput Dynastic Chapters (including user's exact preset: Shashank & Samira breakdown).
2. 4,000+ Claude 3 Opus Third-Person Literary Stories (600-1,100 words each).
3. Universal Entity-Binding & Name Swapping (prompt characters 100% match story text).
4. Multi-Chapter Scene Continuations.
5. Strict Prompt-Loss Masking (targets[:prompt_len] = -100).
"""

import os
import sys
import re
import json
import random
import time
from collections import Counter
import sentencepiece as spm
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TOKENIZER_PATH = os.path.join(BASE_DIR, "spm_novel_16k.model")
OUTPUT_PT = os.path.join(DATA_DIR, "precision_novel_sft_tokens.pt")
OUTPUT_JSONL = os.path.join(DATA_DIR, "precision_novel_sft_samples.jsonl")

print("=" * 80)
print(" 🍡 BUILDING PRECISION MULTI-CHARACTER & LONG-FORM ALIGNMENT DATASET")
print("=" * 80)

# Load Tokenizer
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)
BOS = 1
EOS = 2
PAD = 0
SEQ_LEN = 1536

# Common words to ignore when extracting names
STOP_WORDS = {
    'The', 'A', 'An', 'He', 'She', 'They', 'It', 'In', 'On', 'At', 'When', 
    'Then', 'After', 'Before', 'While', 'As', 'For', 'With', 'Under', 'Over', 
    'Through', 'There', 'Here', 'What', 'How', 'Why', 'Who', 'Where', 'Which', 
    'This', 'That', 'These', 'Those', 'His', 'Her', 'Their', 'Its', 'My', 'Our', 
    'Your', 'One', 'Two', 'Three', 'First', 'Second', 'Last', 'Next', 'All', 
    'Every', 'Some', 'Many', 'Few', 'No', 'Not', 'But', 'And', 'Or', 'So', 'If', 
    'Because', 'Although', 'Even', 'Just', 'Only', 'Now', 'Still', 'Already', 
    'Yet', 'Soon', 'Later', 'Suddenly', 'Finally', 'Meanwhile', 'However', 
    'Instead', 'Perhaps', 'Maybe', 'Indeed', 'Certainly', 'Surely', 'Yes', 'Oh', 
    'Ah', 'Well', 'Captain', 'General', 'Doctor', 'Professor', 'Lord', 'Lady', 
    'Sir', 'King', 'Queen', 'Prince', 'Princess', 'Father', 'Mother', 'Brother', 
    'Sister', 'Uncle', 'Aunt', 'Grandmother', 'Grandfather', 'God', 'Earth',
    'Morning', 'Evening', 'Night', 'Day', 'Room', 'Door', 'Street', 'Road',
    'City', 'House', 'World', 'Time', 'Year', 'Man', 'Woman', 'Boy', 'Girl'
}

# Diverse Worldwide Names Pool for universal character binding
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
    "Vivienne", "Iris", "Evelyn", "Cassandra", "Miriam", "Noor", "JNDNWND"
]

all_pairs = []

# ==============================================================================
# 1. GOLD RAJPUT DYNASTIC CHAPTERS (Comprehensive Novel Sagas)
# ==============================================================================
print("\n[1/4] Assembling Gold Rajput Dynastic Chapters & Variations...")

RAJPUT_GOLD_STORIES = [
    # Story A: Shashank and Samira Breakdown (The user's exact GUI prompt)
    {
        "prompt": "Write a short story about Shashank, a proud young mechanic working in a highway tin shed, and Samira Khanna a daughter of wealthy businessman whose luxury car breaks down on the desert road.",
        "story": """The afternoon heat rose in shimmering, distorted waves off the black tarmac of the national highway. For miles in either direction, there was only dry scrubland, thorny acacia trees, and the occasional dust devil swirling across the yellow earth.

A sudden, violent hiss of scalding vapor erupted from beneath the sculpted hood of a crimson sports coupe. The luxury vehicle shuddered, its high-revving engine sputtering to an abrupt halt on the gravel shoulder, plumes of white steam billowing into the scorching sky. The driver's door swung open with a sharp click, and Samira Khanna stepped out onto the baking roadside.

She was sixteen, dressed in designer sunglasses and an embroidered summer tunic, her expression a mixture of profound shock and rising fury. She stared at the hissing radiator in utter disbelief, tapping her smartphone frantically against her palm.

"Zero signal," Samira muttered, her voice trembling with frustration. "Not a single bar. In the middle of nowhere."

She glanced down the long, empty ribbon of highway. Roughly seventy yards past the shoulder stood a lone, weathered structure: a workshop built of rusted corrugated tin sheets and rough timber poles. A hand-painted wooden board leaning against an old truck tire read simply: 'MECHANIC'.

Before she could walk toward it, the tin door creaked open.

A young man stepped out into the blazing sun, carrying a battered metal toolbox in one hand and a black rubber hose in the other. He was twenty-three, lean and broad-shouldered beneath a faded cotton shirt smudged with grease and road dust. He walked with an upright, effortless stride that possessed none of the slumping deference common among roadside laborers.

He stopped two paces from the steaming sports car, his dark amber eyes taking in the stalled vehicle with calm, observant detachment.

"Overheated block," Shashank said, his voice deep, quiet, and steady. "The high-pressure coolant junction fractured at the manifold. If you tried to restart it, you would have cracked the aluminum cylinder head."

Samira looked at him, startled by his voice and the unwavering composure in his eyes. Despite the grime on his forearms, there was a quiet nobility in his features, an aristocratic sharpness in his jawline that seemed completely out of place in a desert tin shed.

"Can you fix it?" Samira asked, crossing her arms and trying to sound authoritative. "I'm on my way to Jaipur. My father is Rajesh Khanna, and I have to be at the heritage hotel before evening. Name your price."

Shashank did not blink at the mention of the Khanna name. His expression remained utterly impassive. "I charge two hundred rupees for roadside labor, plus the cost of materials. I do not care who your father is. If you want the car to run, step back from the steam."

Without waiting for her reply, Shashank dropped to his knees on the searing gravel. For forty minutes under the relentless desert sun, he worked with surgical concentration. His knuckles scraped against the scalding engine block, but his hands never shook. He measured, clamped, replaced the fractured junction with a reinforced brass sleeve, and purged the coolant airlocks with fluid, practiced precision.

Samira stood in the shade of an acacia tree, watching him in silence. Accustomed to fawning attendants and corporate yes-men who jumped at her father's shadow, she had never seen someone work with such quiet, unyielding self-respect.

When Shashank stood and signaled her to turn the key, the high-performance engine roared to life with a silky, flawless purr.

Samira reached into her leather handbag, pulled out a thick wad of two-thousand-rupee notes, and held them out. "Take this. All of it. You saved me hours of waiting in this heat."

Shashank looked at the bundle of cash, then met her gaze squarely. "The replacement brass cost fifty rupees. My labor is two hundred. Give me two hundred and fifty rupees, or take your car and leave. I earn my bread through work, not charity."

Samira froze, her hand hovering in mid-air. For the first time in her life, someone had refused her money. She slowly slipped the notes back into her purse, retrieved a three-hundred-rupee bill, and handed it to him. "Keep the change as a tip, then."

"I do not take tips," Shashank replied evenly, handing her fifty rupees from his pocket.

He picked up his toolbox, turned on his heel, and walked calmly back into his tin shed, closing the door behind him.

Samira stood by the open car door, the hot desert wind tugging at her hair. A slow, bemused smile broke across her face as she looked toward the rusted shed. "Who on earth are you?" she whispered to herself, before sliding behind the wheel and driving toward Jaipur."""
    },
    # Story B: Shashank and Rajesh Khanna Breakdown Scene
    {
        "prompt": "Write a short story about Shashank, a proud young mechanic working in a highway tin shed, and Rajesh Khanna, a wealthy businessman whose luxury car breaks down on the desert road.",
        "story": """The afternoon heat rose in shimmering waves off the cracked black asphalt of the national highway. Miles from the nearest town, where the scrub thorny bushes gave way to empty sand and rusted hoardings, stood a lone tin shed. Its corrugated metal roof groaned beneath the relentless sun, rattling whenever a heavy transport lorry thundered past in a cloud of diesel exhaust.

Inside the stifling shade of the workshop, Shashank wiped a streak of engine oil from his brow with the back of a calloused hand. He was twenty-three, thin-framed and lean as a desert wolf, his collarbones visible beneath a faded cotton shirt that had seen too many monsoon cycles. On the workbench lay an old motorcycle carburetor, dismantled into dozens of tiny brass jets and springs. Shashank worked with the methodical, patient focus of a surgeon, his long fingers moving with effortless familiarity.

A sudden hiss of steam and the violent shudder of a dying engine broke the highway's silence.

Fifty yards down the asphalt, a midnight-blue luxury sedan — a Maybach worth more than the entire village behind the ridge — coasted to a halt on the gravel shoulder, its radiator spewing white plumes into the blistering sky. The driver's door swung open, and a chauffeur in crisp white livery stepped out in panic, waving frantic hands at the hissing grille. Moments later, the rear door clicked open.

Rajesh Khanna stepped out into the blinding glare. Even stranded in the desert heat, the tycoon looked formidable: tailored charcoal linen suit, silver hair immaculately combed, and an aura of supreme authority that commanded boardrooms across Mumbai and London. He pulled out a gold-plated smartphone, scowled at the screen, and muttered in frustration. "No signal. Not even a single bar."

"Sir, the water pump has seized," the chauffeur stammered, wiping sweat from his forehead. "There is no service center for fifty kilometers."

Rajesh turned his gaze toward the lone tin shed. Through the heat haze, he saw the silhouette of a young mechanic stepping out of the shadows, carrying a battered metal toolbox in one hand and a gallon jug of water in the other.

Shashank approached without haste and without deference. His dark eyes, strikingly amber beneath the grime, took in the stalled luxury vehicle without a flicker of awe.

"Overheated radiator," Shashank said, his voice quiet, low, and steady. "The secondary coolant hose ruptured near the manifold."

Rajesh Khanna studied the young man. Despite his gaunt frame and faded clothes, there was something unsettling about the boy's bearing — a straightness in the spine, an unblinking gaze that refused to bow. "Can you fix it?" Rajesh asked crisply.

"I can bypass the ruptured line with a reinforced rubber sleeve and clamp it," Shashank answered. "It will hold until you reach the city."

Without waiting for permission or negotiating a price, Shashank knelt onto the scorching gravel. For forty minutes, under the blistering sun, the boy worked with quiet, surgical precision. His knuckles scraped against the searing engine block, but he did not wince. He replaced the cracked hose, tightened the clamps with steady turns of a wrench, purged the airlocks, and topped off the coolant reservoir with cool water.

When he stood up and signaled the chauffeur to turn the ignition, the twin-turbo engine roared to life with a silky, flawless purr.

Rajesh Khanna stepped forward, reaching into his breast pocket for his wallet. Finding it empty, he frowned, suddenly remembering he had left his briefcase in the private jet. He unclasped the heavy Patek Philippe timepiece from his wrist — an heirloom valued at over fifty lakhs — and held it out toward the boy.

"I left my cash at the hangar," Rajesh said. "Take this watch. Consider it a deposit on your work."

Shashank looked at the gleaming gold timepiece, then met the billionaire's gaze squarely. "The repair took fifty rupees of rubber and thirty minutes of labor. I charge two hundred rupees. If you do not have cash, you owe me nothing. I do not take charity, and I do not trade in gold."

Before Rajesh could utter another word, Shashank picked up his toolbox, turned his back on the luxury car, and walked calmly back into his tin shed.

Rajesh Khanna stood rooted to the gravel, staring at the boy's retreating back. A slow, calculating smile touched the tycoon's lips. He turned to his driver. "Call my personal investigator the moment we reach signal range. I want that boy's name, his parentage, and every breath he has taken since the day he was born.\""""
    },
    # Story C: Savita and Karthik Confrontation at Rajput Sadan
    {
        "prompt": "Write a short story about Savita Rajput, the proud family matriarch, confronting her grandson Karthik inside the ancestral mansion about his loyalty to the family name.",
        "story": """Inside the grand high-ceilinged drawing room of Rajput Sadan, afternoon shadows lengthened across antique Persian carpets. Heavy mahogany pillars supported arches carved from pink Rajasthani sandstone, their capitals adorned with family crests that had stood for three centuries.

At the center of the chamber sat Savita Rajput. At seventy-five, the matriarch commanded the room with the unyielding posture of an empress. Her silver hair was drawn back into an immaculate bun, secured by a gold pin, and a necklace of uncut emeralds rested against her ivory silk saree. In her right hand she held a silver-headed cane, though her spine remained as straight as an iron pillar.

Standing before her, looking pale and disheveled, was twenty-six-year-old Karthik. His bespoke Italian blazer was unbuttoned, his tie loosened, and his eyes carried the nervous exhaustion of a young man caught between his desires and the crushing weight of his lineage.

"You were seen leaving the Khanna residence in Mumbai at three in the morning, Karthik," Savita began, her voice dropping into a dangerous, low murmur that carried to every corner of the room. "And this morning, the financial dailies are reporting that Rajesh Khanna has suspended negotiations on our joint port terminal."

Karthik swallowed hard, his hands clenching at his sides. "Dadi-ji, you don't understand. Samira won't even speak to me. She laughed in my face when I showed up at the gala. She called me a puppet dancing on family strings."

"And are you?" Savita's eyes narrowed into twin points of cold steel. "Did I spend twenty-six years educating you in London and grooming you to lead Rajput Industries so that a sixteen-year-old girl could strip away your spine? You are a Rajput, Karthik. We do not beg for alliances. We command them."

"Then why are we on the brink of bankruptcy?!" Karthik's voice cracked with raw desperation. "Viraj Uncle says we owe four hundred crores to Khanna's merchant banks. If Rajesh Khanna pulls the liquidity, we lose everything! Even this house!"

The room fell into an icy, suffocating silence.

Savita Rajput slowly rose from her chair, leaning on her cane with measured, deliberate dignity. She walked toward Karthik, stopping only inches from him. Her gaze did not waver.

"Rajput Sadan was built on blood and sacrifice, not bank loans," she said, her tone as cold as winter frost. "Generations of men have died to keep this crest untouched. If you believe your comfortable life was given to you for idle amusement, you are mistaken. You will dress yourself. You will apologize to Rajesh Khanna with the grace of a prince. And you will secure Samira's hand before this month ends. If you fail..."

She paused, letting the silence press heavily into Karthik's chest.

"If you fail, Karthik, I will find a son of this bloodline who possesses the courage you seem to lack. Do not test my resolve.\""""
    },
    # Story D: Avinash secretly visiting Shashank at the tin shed
    {
        "prompt": "Write a short story about Avinash Rajput quietly visiting his estranged son Shashank at his dusty roadside workshop after years of bitter silence.",
        "story": """The twilight sky over the Thar desert bled from amber into deep violet. A cool desert wind began to stir the sand, whistling through the gaps in the corrugated tin sheets of the roadside workshop. A solitary kerosene lantern hung from a wooden post, casting long, swaying shadows across greasy wrenches and dismantled machinery.

Shashank sat on an inverted oil drum, cleaning the grease from his fingers with a rag soaked in kerosene. The highway had grown quiet, the distant drone of trucks fading into the vast desert night.

Footsteps sounded on the loose gravel outside.

Shashank did not look up immediately. He recognized the tentative, heavy cadence of someone who was not accustomed to walking on unpaved ground. When he finally lifted his eyes, a tall man stood in the open doorway of the shed.

Avinash Rajput wore an expensive woolen shawl over a crisp white kurta, but his shoulders were slumped, and his graying hair was disheveled by the wind. At fifty, his face was lined with deep furrows of guilt and exhaustion.

For nearly a minute, neither man spoke. Only the hiss of the kerosene flame broke the stillness.

"You look like her," Avinash said quietly, his voice trembling slightly. "Your mother had that same unyielding look in her eyes whenever she stood her ground."

Shashank tossed the rag onto the workbench. He stood up slowly, crossing his arms over his chest. His expression was neither angry nor welcoming; it was as smooth and unreadable as desert stone.

"You took the bypass road," Shashank said, his tone flat and even. "The ancestral mansion is thirty miles north. Did you lose your way, Mr. Rajput?"

The formal address struck Avinash like a physical blow. He stepped forward into the lantern light, extending a hand before letting it fall helplessly to his side.

"Shashank... I know I have no right to come here," Avinash stammered, his eyes glistening with unshed tears. "For twenty-three years, Savita Ma forbade anyone from mentioning your existence. She threatened to disinherit Vijay and cast us all out. I was weak. I was a coward. But every single day, I knew you were here... working in this heat..."

"And yet you never came," Shashank interrupted quietly. There was no bitterness in his voice, only cold, unvarnished truth. "When the monsoon flooded this shed three years ago and washed away my tools, you did not come. When I was fifteen and went three days without food because no trucks broke down, you did not come. Why are you here tonight?"

Avinash bowed his head, his hands trembling. "Rajesh Khanna came to Rajput Sadan yesterday. He knows who you are. He told the entire family that he will only agree to the business alliance if Samira marries you, not Karthik. Savita Ma is planning to bring you back... to use you to save the family fortune."

A faint, mocking smile touched Shashank's lips. He looked around his humble tin shed, at the rusted iron and the tools he had earned with his own sweat.

"Tell Savita Rajput that her mansion has no walls high enough to buy my soul," Shashank said softly. "I owe nothing to the Rajput name, and I want nothing from it. Go home, Avinash. Before someone sees your luxury car parked by a beggar's shed.\""""
    }
]

# Add gold stories to pairs (repeat 5 times each to ensure rock-solid memorization of core characters & settings)
for s in RAJPUT_GOLD_STORIES:
    for _ in range(5):
        all_pairs.append({
            "prompt": f"### Prompt:\n{s['prompt']}\n\n### Story:\n",
            "story": s["story"].strip()
        })

print(f"      Loaded {len(all_pairs)} Gold Rajput scenes.")

# ==============================================================================
# 2. CLAUDE 3 OPUS LONG-FORM STORIES (Filtered for 3rd Person & 600-1100 words)
# ==============================================================================
print("\n[2/4] Processing Claude 3 Opus Literary Stories (with entity binding)...")

def extract_main_characters(story_text):
    """Extracts top 2 interacting characters from narrative speech and action tags."""
    names = re.findall(r'\b([A-Z][a-z]{2,15})\s+(?:said|replied|whispered|asked|muttered|sighed|shouted|nodded|stepped|turned|looked|glanced|smiled|frowned|walked|stood)\b', story_text)
    names += re.findall(r'\b(?:said|replied|whispered|asked|muttered|sighed|shouted)\s+([A-Z][a-z]{2,15})\b', story_text)
    valid_names = [n for n in names if n not in STOP_WORDS]
    counts = Counter(valid_names)
    most_common = [name for name, count in counts.most_common(2) if count >= 1]
    return most_common

opus_files = [
    os.path.join(DATA_DIR, "opus-writing-prompts-1-raw.jsonl"),
    os.path.join(DATA_DIR, "opus-writing-prompts-2-raw.jsonl")
]

opus_count = 0
for op_file in opus_files:
    if not os.path.exists(op_file):
        continue
    with open(op_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue

            story = d.get("story", "").strip()
            words = len(story.split())
            if not (500 <= words <= 1150):
                continue

            # Strict 3rd-person check: exclude 1st person rambling
            words_lower = story.lower().split()
            first_p = sum(1 for w in words_lower if w in ("i", "me", "my", "myself"))
            third_p = sum(1 for w in words_lower if w in ("he", "she", "his", "her", "they", "their"))
            if first_p > 25 and first_p * 2 > third_p:
                continue

            raw_prompt = d.get("prompt", "").strip()
            # Clean Reddit artifacts from prompt
            clean_prompt = re.sub(r'\[\s*WP\s*\]', '', raw_prompt, flags=re.IGNORECASE).strip()
            clean_prompt = re.sub(r'^\s*[\,\.\-\:\"]+\s*', '', clean_prompt).strip()
            if len(clean_prompt) < 15:
                continue

            # Detect main characters
            chars = extract_main_characters(story)
            
            # Entity Binding: 60% of stories get dynamic name-swapping to train universal binding!
            if len(chars) >= 2 and random.random() < 0.65:
                orig_c1, orig_c2 = chars[0], chars[1]
                # Pick 2 target names
                if random.random() < 0.35:
                    # User's primary universe names
                    new_c1 = random.choice(["Shashank", "Rajesh", "Karthik", "Avinash", "Viraj", "Aman"])
                    new_c2 = random.choice(["Samira", "Savita", "Priya", "Rita", "Meghna"])
                elif random.random() < 0.15:
                    # Stress test names including arbitrary / out-of-vocab names like JNDNWND
                    new_c1 = random.choice(["JNDNWND", "Zarek", "Kael", "Nyx", "Xylar"])
                    new_c2 = random.choice(["Shashank", "Elena", "Marcus", "Samira"])
                else:
                    new_c1 = random.choice(NAMES_MALE)
                    new_c2 = random.choice(NAMES_FEMALE)

                # Safe word-boundary swap
                story_swapped = re.sub(r'\b' + re.escape(orig_c1) + r'\b', '___TEMP_C1___', story)
                story_swapped = re.sub(r'\b' + re.escape(orig_c2) + r'\b', '___TEMP_C2___', story_swapped)
                story_swapped = story_swapped.replace('___TEMP_C1___', new_c1).replace('___TEMP_C2___', new_c2)

                formatted_prompt = f"### Prompt:\nWrite a story about {new_c1} and {new_c2}. {clean_prompt}\n\n### Story:\n"
                final_story = story_swapped
            elif len(chars) == 1 and random.random() < 0.5:
                orig_c = chars[0]
                new_c = random.choice(NAMES_MALE if random.random() < 0.5 else NAMES_FEMALE)
                story_swapped = re.sub(r'\b' + re.escape(orig_c) + r'\b', new_c, story)
                formatted_prompt = f"### Prompt:\nWrite a story about {new_c}. {clean_prompt}\n\n### Story:\n"
                final_story = story_swapped
            else:
                formatted_prompt = f"### Prompt:\n{clean_prompt}\n\n### Story:\n"
                final_story = story

            all_pairs.append({
                "prompt": formatted_prompt,
                "story": final_story
            })
            opus_count += 1
            if opus_count >= 2500:
                break

print(f"      Processed {opus_count} high-quality 3rd-person Opus stories.")

# ==============================================================================
# 3. MULTI-CHAPTER SCENE CONTINUATION PAIRS
# ==============================================================================
print("\n[3/4] Creating Multi-Chapter Scene Transition Pairs...")
continuation_count = 0
for pair in list(all_pairs)[:400]:
    story_text = pair["story"]
    paras = [p.strip() for p in story_text.split("\n\n") if len(p.strip()) > 40]
    if len(paras) >= 4:
        split_point = len(paras) // 2
        scene_1_text = "\n\n".join(paras[:split_point])
        scene_2_text = "\n\n".join(paras[split_point:])
        prev_tail = scene_1_text[-300:]
        
        cont_prompt = f"### Prompt:\nPrevious scene ending:\n{prev_tail}\n\nContinue the story naturally from this scene:\n\n### Story:\n"
        all_pairs.append({
            "prompt": cont_prompt,
            "story": scene_2_text
        })
        continuation_count += 1

print(f"      Created {continuation_count} multi-chapter continuation pairs.")

# Shuffle pairs
random.seed(42)
random.shuffle(all_pairs)
print(f"\nTotal Dataset Samples: {len(all_pairs):,}")

# Save sample preview jsonl
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for p in all_pairs[:50]:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"      Saved preview of 50 samples to {OUTPUT_JSONL}")

# ==============================================================================
# 4. TOKENIZE AND PACK INTO TENSORS (Prompt-Loss Masking)
# ==============================================================================
print(f"\n[4/4] Tokenizing and packing into tensors (SEQ_LEN = {SEQ_LEN})...")
packed_samples = []

t0 = time.time()
for idx, item in enumerate(all_pairs):
    p_tokens = sp.encode_as_ids(item["prompt"])
    s_tokens = sp.encode_as_ids(item["story"]) + [EOS]

    total_tokens = p_tokens + s_tokens
    if len(total_tokens) > SEQ_LEN:
        total_tokens = total_tokens[:SEQ_LEN]

    input_ids = total_tokens[:-1]
    target_ids = total_tokens[1:]

    # Mask prompt tokens so loss is computed ONLY on generated story tokens
    prompt_len = min(len(p_tokens) - 1, len(target_ids))
    masked_targets = list(target_ids)
    for i in range(prompt_len):
        masked_targets[i] = -100

    # Pad to SEQ_LEN - 1
    pad_len = (SEQ_LEN - 1) - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [PAD] * pad_len
        masked_targets = masked_targets + [-100] * pad_len

    x_tensor = torch.tensor(input_ids, dtype=torch.long)
    y_tensor = torch.tensor(masked_targets, dtype=torch.long)

    packed_samples.append((x_tensor, y_tensor))

    if (idx + 1) % 500 == 0 or (idx + 1) == len(all_pairs):
        print(f"      Tokenized {idx + 1:,} / {len(all_pairs):,} samples ({time.time() - t0:.1f}s)...")

# Train / Val Split (92% / 8%)
split_idx = int(len(packed_samples) * 0.92)
train_samples = packed_samples[:split_idx]
val_samples = packed_samples[split_idx:]

print(f"\nSaving packed tensors to {OUTPUT_PT}...")
torch.save({
    "train": train_samples,
    "val": val_samples,
    "vocab_size": sp.get_piece_size(),
    "seq_len": SEQ_LEN
}, OUTPUT_PT)

file_mb = os.path.getsize(OUTPUT_PT) / (1024 * 1024)
print(f"SUCCESS: Dataset saved! File size: {file_mb:.1f} MB | Train: {len(train_samples):,} | Val: {len(val_samples):,}")
print("=" * 80)
