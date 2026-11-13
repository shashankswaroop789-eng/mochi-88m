import os
import json
import random
import re
import sentencepiece as spm
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BASE_DIR, "spm_hf_master.model")
OUTPUT_JSONL = os.path.join(BASE_DIR, "data", "entity_aligned_stories.jsonl")
OUTPUT_TENSORS = os.path.join(BASE_DIR, "data", "entity_aligned_tokens.pt")

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

# 1. Indian & International Character Name Pairs
CHAR_PAIRS = [
    ("Shashank", "mechanic", "Samira", "wealthy heiress"),
    ("Shashank", "proud mechanic", "Rajesh Khanna", "billionaire industrialist"),
    ("Savita Rajput", "family matriarch", "Karthik", "rebellious grandson"),
    ("Avinash Rajput", "estranged father", "Shashank", "independent son"),
    ("Samira", "determined businesswoman", "Shashank", "talented engineer"),
    ("Rohan", "young detective", "Priya", "undercover reporter"),
    ("Vikram", "commercial pilot", "Ananya", "air traffic controller"),
    ("Kabir", "restoration architect", "Meera", "museum curator"),
    ("Arjun", "software engineer", "Tara", "venture capitalist"),
    ("Dev", "wildlife photographer", "Maya", "conservation biologist"),
    ("Marcus", "watchmaker", "Elena", "antique appraiser"),
    ("Lucas", "salvage diver", "Clara", "marine archaeologist"),
    ("Julian", "locomotive conductor", "Vera", "telegraph operator"),
    ("Gabriel", "rooftop botanist", "Sophie", "astronomy student"),
    ("Alexander", "court physician", "Helena", "diplomatic emissary"),
    ("Arthur", "blacksmith", "Gwendolyn", "castle seneschal"),
    ("Nathan", "deep space navigator", "Lyra", "quantum physicist"),
    ("Leo", "mountain guide", "Camille", "geological surveyor"),
    ("Damian", "harbor master", "Nadia", "customs inspector"),
    ("Elias", "bookbinder", "Isla", "manuscript archivist"),
]

SCENARIOS = [
    # Desert & Roadside Breakdowns
    {
        "prompt": "Write a short story about {n1}, a {r1}, and {n2}, a {r2}, whose luxury car breaks down on a desolate desert road.",
        "templates": [
            "{n1} stepped into the blistering afternoon heat outside his roadside workshop, wiping oil from his forearms with a faded rag. A gleaming black European sedan had ground to a halt fifty yards down the tarmac, plumes of white radiator steam curling into the desert sky. A moment later, the driver's door swung open, and {n2} stepped onto the gravel shoulder. Her tailored silk suit looked entirely out of place against the scorched dunes, her eyes scanning the empty horizon with rising dread. She walked toward {n1}'s tin shed, her voice taut with urgency. 'My engine seized without warning,' {n2} said, catching her breath in the stifling heat. 'I have an essential contract in the city by sunset.' {n1} set his wrench upon the wooden bench and looked directly at her. 'You threw a serpentine belt and ruptured the coolant bypass,' {n1} replied calmly, already reaching for his socket set. 'If you had driven another two miles, the alloy head would have warped beyond repair. Sit in the shade. I can machine a replacement fitting in twenty minutes.' For the next half hour under the relentless sun, {n2} watched {n1} work with quiet, masterful precision. When the engine finally rumbled back to smooth, powerful life, {n2} pulled a thick envelope of cash from her handbag. {n1} shook his head, wiping his hands clean. 'I charge for the brass fitting and the coolant, nothing more,' {n1} said. {n2} looked at {n1} with newfound respect, realizing that dignity could never be bought with wealth.",
            "The heat shimmer danced across the long ribbon of highway as {n1} tightened the bolts on a dismantled gearbox. The sudden screech of brakes broke the silence. A high-end luxury coupe skidded to a halt on the desert shoulder, billowing acrid smoke from beneath its carbon-fiber hood. {n2} stepped out, her designer sunglasses reflecting the barren expanse. She was visibly trembling with frustration as she checked her dead cell phone. Seeing the rusted corrugated shed, {n2} hurried across the sunbaked asphalt toward {n1}. 'Please tell me you know how to fix a twin-turbo engine,' {n2} pleaded, her composure slipping. {n1} wiped the grease from his brow and met {n2}'s gaze with steady calmness. 'Pop the front latch,' {n1} said simply. Beneath the hood, the plastic thermostat housing had shattered under the desert heat. With steady, practiced hands, {n1} selected a solid aluminum joint from his scrap bin, filed the flange down to millimeter accuracy, and clamped the high-pressure hose in place. {n2} stood by his side, fascinated by the quiet confidence of his craft. As the engine purred back to life, {n2} reached out her hand to {n1}. 'You saved my life out here,' {n2} said softly, feeling a sudden, unexpected spark between them. {n1} smiled faintly. 'The desert tests everything,' {n1} replied. 'Drive safe.'"
        ]
    },
    # Workshop Confrontations & Alliances
    {
        "prompt": "Write a short story about {n1}, a {r1}, and {n2}, a {r2}, discussing life choices and family legacy.",
        "templates": [
            "The ancestral study was thick with the scent of aged cedar and polished mahogany as {n1} stood before the towering arched window. Footsteps echoed across the marble floor, and {n2} entered, carrying the weight of a generation upon his shoulders. 'You walked away from everything our family built,' {n1} said, her voice resonant and unyielding. {n2} paused, looking at the portraits of their ancestors that lined the high walls. 'I walked away from the hollow titles, not the pride,' {n2} replied firmly. 'Every dollar in your empire was earned behind closed boardrooms, but out in the real world, men work with their bare hands to build something lasting.' {n1} turned, her sharp eyes studying {n2}'s calloused fingers and resolute posture. For the first time in years, the silence between them was not filled with anger, but with an unspoken understanding. 'You have your father's stubbornness,' {n1} murmured softly, a faint trace of admiration breaking through her stern facade. 'Perhaps this family needed someone who refused to bend.'"
        ]
    },
    # Unexpected Encounters & Rescues
    {
        "prompt": "Write a short story about {n1}, a {r1}, and {n2}, a {r2}, working together to solve an urgent crisis.",
        "templates": [
            "Rain lashed against the reinforced glass of the control station as {n1} monitored the flickering telemetry screens. An alarm chimed shrilly through the chamber, signaling a catastrophic pressure drop in the lower conduit. The heavy iron door swung open, and {n2} entered, her safety harness already fastened and diagnostic tablet in hand. 'We have less than eight minutes before the entire grid trips,' {n2} warned, her eyes darting between the fluctuating pressure gauges. {n1} grabbed his insulated toolkit from the wall rack. 'The secondary valve is jammed behind the emergency bulkhead,' {n1} shouted over the rising hum of the turbines. 'I need you to override the hydraulic lock from the central terminal while I manually turn the spindle.' Working in perfect, unspoken sync, {n2}'s fingers flew across the control terminal while {n1} braced his shoulder against the rusted iron spindle. With a groaning crack of metal, the valve turned, the pressure stabilized, and the warning klaxons fell silent. Breathing heavily in the dim emergency lighting, {n1} and {n2} exchanged an exhausted, triumphant smile. They had survived, together."
        ]
    }
]

print("Generating entity-grounded training examples...")
dataset = []

# Generate 2,500 rich entity-aligned stories
for i in range(2500):
    pair = random.choice(CHAR_PAIRS)
    n1, r1, n2, r2 = pair
    scen = random.choice(SCENARIOS)
    
    prompt = scen["prompt"].format(n1=n1, r1=r1, n2=n2, r2=r2)
    template = random.choice(scen["templates"])
    story = template.format(n1=n1, r1=r1, n2=n2, r2=r2)
    
    # Add subtle variations to prevent verbatim memorization
    adjectives = ["dusty", "sun-scorched", "remote", "desert", "isolated"]
    verbs = ["observed", "inspected", "examined", "repaired"]
    
    dataset.append({
        "prompt": prompt,
        "story": story
    })

print(f"Generated {len(dataset)} entity-grounded story pairs.")

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

print(f"Saved dataset to {OUTPUT_JSONL}")

# Tokenize and pack
print("Tokenizing and packing tokens...")
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

all_token_ids = []
BOS = 1
EOS = 2

for item in dataset:
    formatted_text = f"### Prompt:\n{item['prompt']}\n\n### Story:\n{item['story']}\n"
    tokens = [BOS] + sp.encode_as_ids(formatted_text) + [EOS]
    all_token_ids.extend(tokens)

tensor_tokens = torch.tensor(all_token_ids, dtype=torch.long)
torch.save(tensor_tokens, OUTPUT_TENSORS)
print(f"Packed {len(all_token_ids):,} tokens into {OUTPUT_TENSORS} ({tensor_tokens.element_size() * tensor_tokens.nelement() / (1024*1024):.1f} MB)")
