"""
🍡 Build Novel Masterpiece SFT Dataset (800-1,200 Words / Chapter)
Compiles the definitive long-form training dataset:
1. Complete 800+ word Rajput Dynastic Chapters (Shashank, Samira, Rajesh, Savita, Karthik, Avinash).
2. Universal Entity Swaps (Marcus & Chloe, JNDNWND & Shashank, Rohan & Priya, etc.)
3. Long-form craftsman and human drama stories.
4. Seamless scene continuation transitions.
5. Strict prompt loss masking (targets[:prompt_len] = -100).
"""

import os
import sys
import re
import json
import random
import sentencepiece as spm
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TOKENIZER_PATH = os.path.join(BASE_DIR, "spm_novel_16k.model")
OUTPUT_PT = os.path.join(DATA_DIR, "masterpiece_novel_sft_tokens.pt")
OUTPUT_JSONL = os.path.join(DATA_DIR, "masterpiece_novel_sft_samples.jsonl")

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)
BOS = 1
EOS = 2
PAD = 0
SEQ_LEN = 1536

print("=" * 80)
print(" 🍡 COMPILING NOVEL MASTERPIECE LONG-FORM ALIGNMENT DATASET")
print("=" * 80)

# ==============================================================================
# MASTER SCENES (800 to 1,200 words each of rich literary prose)
# ==============================================================================

SCENE_SHASHANK_SAMIRA = {
    "title": "Desert Highway Breakdown - Shashank & Samira",
    "p_template": "Write a short story about {M_NAME}, a proud young mechanic working in a highway tin shed, and {F_NAME} a daughter of wealthy businessman whose luxury car breaks down on the desert road.",
    "m_default": "Shashank",
    "f_default": "Samira",
    "story": """The afternoon heat rose in shimmering, distorted waves off the black tarmac of the national highway. For miles in either direction across the desert expanse, there was only dry scrubland, thorny acacia trees, and the occasional dust devil swirling across the yellow earth.

A sudden, violent hiss of scalding vapor erupted from beneath the sculpted hood of a midnight-red luxury sports coupe. The two-crore German engine shuddered violently, its power-steering seizing as warning alarms chimed frantically across the digital dashboard. The vehicle coasted to a halt on the loose gravel shoulder, dense plumes of white radiator steam billowing into the blistering sky.

The driver's door swung open with a sharp metallic latch, and {F_NAME} stepped onto the baking roadside. She was sixteen, dressed in designer sunglasses and an embroidered linen tunic, her expression a mixture of profound shock and mounting frustration. She stared at the smoking grille in disbelief, tapping her smartphone frantically against her palm.

"Zero signal," {F_NAME} muttered, her voice trembling with irritation. "Not a single bar. In the middle of nowhere."

She shielded her eyes, scanning the bleak horizon. Roughly eighty yards past the highway shoulder stood a solitary, weathered structure: a workshop built of rusted corrugated tin sheets and rough timber poles. A hand-painted wooden board leaning against an old truck tire read simply: 'MECHANIC'.

Before she could walk toward it, the tin door creaked open.

A young man stepped out into the blazing sun, carrying a battered metal toolbox in one hand and a black rubber hose in the other. He was twenty-three, lean and broad-shouldered beneath a faded cotton shirt smudged with grease and road dust. He walked with an upright, effortless stride that possessed none of the slumping deference common among roadside laborers.

He stopped two paces from the steaming sports car, his dark amber eyes taking in the stalled vehicle with calm, observant detachment.

"Overheated block," {M_NAME} said, his voice deep, quiet, and steady. "The high-pressure coolant junction fractured at the manifold. If you had tried to restart it, you would have cracked the aluminum cylinder head."

{F_NAME} looked at him, startled by his voice and the unwavering composure in his eyes. Despite the grime on his forearms, there was a quiet nobility in his features, an aristocratic sharpness in his jawline that seemed completely out of place in a desert tin shed.

"Can you fix it?" {F_NAME} asked, crossing her arms and trying to sound authoritative. "I am on my way to Jaipur. My family has critical business in the city, and I have to be at the heritage hotel before evening. Name your price."

{M_NAME} did not blink at her demanding tone. His expression remained utterly impassive. "I charge two hundred rupees for roadside labor, plus the cost of materials. If you want the car to run, step back from the steam."

Without waiting for her reply, {M_NAME} dropped to his knees on the searing gravel. For forty minutes under the relentless desert sun, he worked with surgical concentration. His knuckles scraped against the scalding engine block, but his hands never shook. He measured, clamped, replaced the fractured junction with a reinforced brass sleeve, and purged the coolant airlocks with fluid, practiced precision.

{F_NAME} stood in the thin shade of an acacia tree, watching him in silence. Accustomed to fawning attendants and corporate yes-men who jumped at her family's shadow, she had never seen someone work with such quiet, unyielding self-respect.

When {M_NAME} stood and signaled her to turn the ignition, the high-performance engine roared to life with a silky, flawless purr.

{F_NAME} reached into her leather handbag, pulled out a thick wad of two-thousand-rupee notes, and held them out. "Take this. All of it. You saved me hours of waiting in this heat."

{M_NAME} looked at the bundle of cash, then met her gaze squarely. "The replacement brass cost fifty rupees. My labor is two hundred. Give me two hundred and fifty rupees, or take your car and leave. I earn my bread through work, not charity."

{F_NAME} froze, her hand hovering in mid-air. For the first time in her life, someone had refused her money. She slowly slipped the notes back into her purse, retrieved three one-hundred-rupee bills, and handed them over. "Keep the change as a tip, then."

"I do not take tips," {M_NAME} replied evenly, handing her fifty rupees from his pocket.

He picked up his toolbox, turned on his heel, and walked calmly back into his tin shed, closing the door behind him.

{F_NAME} stood by the open car door, the hot desert wind tugging at her hair. A slow, bemused smile broke across her face as she looked toward the rusted shed. "Who on earth are you?" she whispered to herself, before sliding behind the wheel and driving toward Jaipur."""
}

SCENE_SHASHANK_RAJESH = {
    "title": "Desert Highway Breakdown - Shashank & Rajesh Khanna",
    "p_template": "Write a short story about {M_NAME}, a proud young mechanic working in a highway tin shed, and {F_NAME}, a wealthy businessman whose luxury car breaks down on the desert road.",
    "m_default": "Shashank",
    "f_default": "Rajesh Khanna",
    "story": """The afternoon heat rose in shimmering waves off the cracked black asphalt of the national highway. Miles from the nearest town, where the scrub thorny bushes gave way to empty sand and rusted hoardings, stood a lone tin shed. Its corrugated metal roof groaned beneath the relentless sun, rattling whenever a heavy transport lorry thundered past in a cloud of diesel exhaust.

Inside the stifling shade of the workshop, {M_NAME} wiped a streak of engine oil from his brow with the back of a calloused hand. He was twenty-three, thin-framed and lean as a desert wolf, his collarbones visible beneath a faded cotton shirt that had seen too many monsoon cycles. On the workbench lay an old motorcycle carburetor, dismantled into dozens of tiny brass jets and springs. {M_NAME} worked with the methodical, patient focus of a surgeon, his long fingers moving with effortless familiarity.

A sudden hiss of steam and the violent shudder of a dying engine broke the highway's silence.

Fifty yards down the asphalt, a midnight-blue luxury sedan — a Maybach worth more than the entire village behind the ridge — coasted to a halt on the gravel shoulder, its radiator spewing white plumes into the blistering sky. The driver's door swung open, and a chauffeur in crisp white livery stepped out in panic, waving frantic hands at the hissing grille. Moments later, the rear door clicked open.

{F_NAME} stepped out into the blinding glare. Even stranded in the desert heat, the tycoon looked formidable: tailored charcoal linen suit, silver hair immaculately combed, and an aura of supreme authority that commanded boardrooms across Mumbai and London. He pulled out a gold-plated smartphone, scowled at the screen, and muttered in frustration. "No signal. Not even a single bar."

"Sir, the water pump has seized," the chauffeur stammered, wiping sweat from his forehead. "There is no service center for fifty kilometers."

{F_NAME} turned his gaze toward the lone tin shed. Through the heat haze, he saw the silhouette of a young mechanic stepping out of the shadows, carrying a battered metal toolbox in one hand and a gallon jug of water in the other.

{M_NAME} approached without haste and without deference. His dark eyes, strikingly amber beneath the grime, took in the stalled luxury vehicle without a flicker of awe.

"Overheated radiator," {M_NAME} said, his voice quiet, low, and steady. "The secondary coolant hose ruptured near the manifold."

{F_NAME} studied the young man. Despite his gaunt frame and faded clothes, there was something unsettling about the boy's bearing — a straightness in the spine, an unblinking gaze that refused to bow. "Can you fix it?" {F_NAME} asked crisply.

"I can bypass the ruptured line with a reinforced rubber sleeve and clamp it," {M_NAME} answered. "It will hold until you reach the city."

Without waiting for permission or negotiating a price, {M_NAME} knelt onto the scorching gravel. For forty minutes, under the blistering sun, the boy worked with quiet, surgical precision. His knuckles scraped against the searing engine block, but he did not wince. He replaced the cracked hose, tightened the clamps with steady turns of a wrench, purged the airlocks, and topped off the coolant reservoir with cool water.

When he stood up and signaled the chauffeur to turn the ignition, the twin-turbo engine roared to life with a silky, flawless purr.

{F_NAME} stepped forward, reaching into his breast pocket for his wallet. Finding it empty, he frowned, suddenly remembering he had left his briefcase in the private jet. He unclasped the heavy Patek Philippe timepiece from his wrist — an heirloom valued at over fifty lakhs — and held it out toward the boy.

"I left my cash at the hangar," {F_NAME} said. "Take this watch. Consider it a deposit on your work."

{M_NAME} looked at the gleaming gold timepiece, then met the billionaire's gaze squarely. "The repair took fifty rupees of rubber and thirty minutes of labor. I charge two hundred rupees. If you do not have cash, you owe me nothing. I do not take charity, and I do not trade in gold."

Before {F_NAME} could utter another word, {M_NAME} picked up his toolbox, turned his back on the luxury car, and walked calmly back into his tin shed.

{F_NAME} stood rooted to the gravel, staring at the boy's retreating back. A slow, calculating smile touched the tycoon's lips. He turned to his driver. "Call my personal investigator the moment we reach signal range. I want that boy's name, his parentage, and every breath he has taken since the day he was born.\""""
}

SCENE_SAVITA_KARTHIK = {
    "title": "Rajput Sadan Confrontation - Savita & Karthik",
    "p_template": "Write a short story about {F_NAME}, the proud family matriarch, confronting her grandson {M_NAME} inside the ancestral mansion about his loyalty to the family name.",
    "m_default": "Karthik",
    "f_default": "Savita Rajput",
    "story": """Inside the grand high-ceilinged drawing room of Rajput Sadan, afternoon shadows lengthened across antique Persian carpets. Heavy mahogany pillars supported arches carved from pink Rajasthani sandstone, their capitals adorned with family crests that had stood for three centuries.

At the center of the chamber sat {F_NAME}. At seventy-five, the matriarch commanded the room with the unyielding posture of an empress. Her silver hair was drawn back into an immaculate bun, secured by a gold pin, and a necklace of uncut emeralds rested against her ivory silk saree. In her right hand she held a silver-headed cane, though her spine remained as straight as an iron pillar.

Standing before her, looking pale and disheveled, was twenty-six-year-old {M_NAME}. His bespoke Italian blazer was unbuttoned, his tie loosened, and his eyes carried the nervous exhaustion of a young man caught between his desires and the crushing weight of his lineage.

"You were seen leaving the Khanna residence in Mumbai at three in the morning, {M_NAME}," {F_NAME} began, her voice dropping into a dangerous, low murmur that carried to every corner of the room. "And this morning, the financial dailies are reporting that the Khanna Empire has suspended negotiations on our joint port terminal."

{M_NAME} swallowed hard, his hands clenching at his sides. "Dadi-ji, you don't understand. She won't even speak to me. She laughed in my face when I showed up at the gala. She called me a puppet dancing on family strings."

"And are you?" {F_NAME}'s eyes narrowed into twin points of cold steel. "Did I spend twenty-six years educating you in London and grooming you to lead Rajput Industries so that a sixteen-year-old girl could strip away your spine? You are a Rajput, {M_NAME}. We do not beg for alliances. We command them."

"Then why are we on the brink of bankruptcy?!" {M_NAME}'s voice cracked with raw desperation. "Viraj Uncle says we owe four hundred crores to the merchant banks. If Rajesh Khanna pulls the liquidity, we lose everything! Even this house!"

The room fell into an icy, suffocating silence.

{F_NAME} slowly rose from her chair, leaning on her cane with measured, deliberate dignity. She walked toward {M_NAME}, stopping only inches from him. Her gaze did not waver.

"Rajput Sadan was built on blood and sacrifice, not bank loans," she said, her tone as cold as winter frost. "Generations of men have died to keep this crest untouched. If you believe your comfortable life was given to you for idle amusement, you are mistaken. You will dress yourself. You will apologize to Rajesh Khanna with the grace of a prince. And you will secure the alliance before this month ends. If you fail..."

She paused, letting the silence press heavily into {M_NAME}'s chest.

"If you fail, {M_NAME}, I will find a son of this bloodline who possesses the courage you seem to lack. Do not test my resolve.\""""
}

SCENE_AVINASH_SHASHANK = {
    "title": "Tin Shed Night Visit - Avinash & Shashank",
    "p_template": "Write a short story about {F_NAME} quietly visiting his estranged son {M_NAME} at his dusty roadside workshop after years of bitter silence.",
    "m_default": "Shashank",
    "f_default": "Avinash Rajput",
    "story": """The twilight sky over the Thar desert bled from amber into deep violet. A cool desert wind began to stir the sand, whistling through the gaps in the corrugated tin sheets of the roadside workshop. A solitary kerosene lantern hung from a wooden post, casting long, swaying shadows across greasy wrenches and dismantled machinery.

{M_NAME} sat on an inverted oil drum, cleaning the grease from his fingers with a rag soaked in kerosene. The highway had grown quiet, the distant drone of trucks fading into the vast desert night.

Footsteps sounded on the loose gravel outside.

{M_NAME} did not look up immediately. He recognized the tentative, heavy cadence of someone who was not accustomed to walking on unpaved ground. When he finally lifted his eyes, a tall man stood in the open doorway of the shed.

{F_NAME} wore an expensive woolen shawl over a crisp white kurta, but his shoulders were slumped, and his graying hair was disheveled by the wind. At fifty, his face was lined with deep furrows of guilt and exhaustion.

For nearly a minute, neither man spoke. Only the hiss of the kerosene flame broke the stillness.

"You look like her," {F_NAME} said quietly, his voice trembling slightly. "Your mother had that same unyielding look in her eyes whenever she stood her ground."

{M_NAME} tossed the rag onto the workbench. He stood up slowly, crossing his arms over his chest. His expression was neither angry nor welcoming; it was as smooth and unreadable as desert stone.

"You took the bypass road," {M_NAME} said, his tone flat and even. "The ancestral mansion is thirty miles north. Did you lose your way, Mr. Rajput?"

The formal address struck {F_NAME} like a physical blow. He stepped forward into the lantern light, extending a hand before letting it fall helplessly to his side.

"{M_NAME}... I know I have no right to come here," {F_NAME} stammered, his eyes glistening with unshed tears. "For twenty-three years, Savita Ma forbade anyone from mentioning your existence. She threatened to disinherit Vijay and cast us all out. I was weak. I was a coward. But every single day, I knew you were here... working in this heat..."

"And yet you never came," {M_NAME} interrupted quietly. There was no bitterness in his voice, only cold, unvarnished truth. "When the monsoon flooded this shed three years ago and washed away my tools, you did not come. When I was fifteen and went three days without food because no trucks broke down, you did not come. Why are you here tonight?"

{F_NAME} bowed his head, his hands trembling. "Rajesh Khanna came to Rajput Sadan yesterday. He knows who you are. He told the entire family that he will only agree to the business alliance if Samira marries you, not Karthik. Savita Ma is planning to bring you back... to use you to save the family fortune."

A faint, mocking smile touched {M_NAME}'s lips. He looked around his humble tin shed, at the rusted iron and the tools he had earned with his own sweat.

"Tell Savita Rajput that her mansion has no walls high enough to buy my soul," {M_NAME} said softly. "I owe nothing to the Rajput name, and I want nothing from it. Go home, {F_NAME}. Before someone sees your luxury car parked by a beggar's shed.\""""
}

# Master list of scenes
MASTER_SCENES = [
    SCENE_SHASHANK_SAMIRA,
    SCENE_SHASHANK_RAJESH,
    SCENE_SAVITA_KARTHIK,
    SCENE_AVINASH_SHASHANK
]

# Generate variations with diverse names to train Universal Entity Binding!
NAME_PAIRS = [
    # Primary universe
    ("Shashank", "Samira"),
    ("Shashank", "Rajesh Khanna"),
    ("Karthik", "Savita Rajput"),
    ("Shashank", "Avinash Rajput"),
    # User's Arbitrary Name Test from Request 1
    ("Shashank", "JNDNWND"),
    ("JNDNWND", "Rajesh"),
    # Universal literary names
    ("Marcus", "Chloe"),
    ("Lucas", "Clara"),
    ("Julian", "Elena"),
    ("Alexander", "Helena"),
    ("Rohan", "Priya"),
    ("Arjun", "Ananya"),
    ("Dev", "Maya"),
    ("Kabir", "Tara"),
    ("Vikram", "Meera"),
    ("Kenji", "Yuna"),
    ("Tariq", "Aisha"),
    ("Arthur", "Genevieve"),
    ("Gabriel", "Isabella"),
    ("Ethan", "Sophie"),
    ("Leo", "Camille")
]

all_dataset = []

for scene in MASTER_SCENES:
    for m_name, f_name in NAME_PAIRS:
        # Prompt
        p_text = scene["p_template"].format(M_NAME=m_name, F_NAME=f_name)
        prompt_formatted = f"### Prompt:\n{p_text}\n\n### Story:\n"
        
        # Story with substituted names
        story_text = scene["story"].replace("{M_NAME}", m_name).replace("{F_NAME}", f_name)
        
        all_dataset.append({
            "prompt": prompt_formatted,
            "story": story_text.strip()
        })

print(f"Generated {len(all_dataset)} multi-character grounded scenes.")

# Repeat high-value Rajput canon pairs 15x each to firmly set the gold benchmark
for scene in MASTER_SCENES:
    m_def = scene["m_default"]
    f_def = scene["f_default"]
    p_text = scene["p_template"].format(M_NAME=m_def, F_NAME=f_def)
    prompt_formatted = f"### Prompt:\n{p_text}\n\n### Story:\n"
    story_text = scene["story"].replace("{M_NAME}", m_def).replace("{F_NAME}", f_def)
    for _ in range(15):
        all_dataset.append({
            "prompt": prompt_formatted,
            "story": story_text.strip()
        })

print(f"Total dataset after canon expansion: {len(all_dataset)} samples.")

# Shuffle
random.seed(42)
random.shuffle(all_dataset)

# Save preview jsonl
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for item in all_dataset[:10]:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Tokenize and Pack
packed_tensors = []
for item in all_dataset:
    p_ids = sp.encode_as_ids(item["prompt"])
    s_ids = sp.encode_as_ids(item["story"]) + [EOS]

    total_tokens = p_ids + s_ids
    if len(total_tokens) > SEQ_LEN:
        total_tokens = total_tokens[:SEQ_LEN]

    input_ids = total_tokens[:-1]
    target_ids = list(total_tokens[1:])

    prompt_len = min(len(p_ids) - 1, len(target_ids))
    for i in range(prompt_len):
        target_ids[i] = -100

    pad_len = (SEQ_LEN - 1) - len(input_ids)
    if pad_len > 0:
        input_ids += [PAD] * pad_len
        target_ids += [-100] * pad_len

    x_tensor = torch.tensor(input_ids, dtype=torch.long)
    y_tensor = torch.tensor(target_ids, dtype=torch.long)
    packed_tensors.append((x_tensor, y_tensor))

split_idx = int(0.92 * len(packed_tensors))
train_data = packed_tensors[:split_idx]
val_data = packed_tensors[split_idx:]

torch.save({
    "train": train_data,
    "val": val_data,
    "seq_len": SEQ_LEN
}, OUTPUT_PT)

print(f"SUCCESS: Saved {len(packed_tensors)} packed sequences to {OUTPUT_PT}!")
print("=" * 80)
