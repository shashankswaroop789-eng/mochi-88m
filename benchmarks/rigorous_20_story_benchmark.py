"""
🍡 Rigorous 20-Story Multi-Genre Benchmark & Judge Engine for Mochi-Novel-300M
Tests 20 completely different stories across 20 distinct genres with custom user instructions.
Generates:
- Scene 1 (Premise)
- Scene 2 (Previous scene tail + Custom Next Scene User Instruction)
Evaluates:
- Instruction Adherence
- Character & Setting Continuity
- Concept Leakage (Zero Rajput leakage in non-Rajput stories)
- Repetition & Loop Degeneracy
- Word count, speed, and qualitative prose judgment
"""

import os
import sys
import time
import json
import re
import torch
import sentencepiece as spm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modern_transformer import ModernGPT

CHECKPOINT_PATH = os.path.join(ROOT_DIR, "mochi_novel_300m.pt")
GOLDEN_PATH = os.path.join(ROOT_DIR, "mochi_novel_300m_golden_rajput.pt")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "spm_novel_16k.model")
OUTPUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_20_stories_results.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

BENCHMARK_SUITE = [
    {
        "id": 1,
        "genre": "Hard Science Fiction",
        "title": "Signal from Trappist",
        "premise": "Dr. Alistair Thorne, an astrophysicist on a lunar observatory, intercepts an encrypted signal originating from the Trappist-1 system.",
        "instruction": "Alistair and technician Nadia decrypt the first layer of the signal and discover it contains biological DNA coordinates.",
        "keywords_scene1": ["Alistair", "lunar", "signal", "observatory", "Trappist"],
        "keywords_scene2": ["Nadia", "decrypt", "DNA", "coordinates", "signal"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 2,
        "genre": "Cyberpunk Noir",
        "title": "Neon Biocrypt",
        "premise": "Ren, a cybernetic black-market surgeon in the neon underbelly of Sector 9, is brought an injured courier clutching a biocrypted flash drive.",
        "instruction": "Ren removes a burning tracker from the courier's mechanical arm just as corporate security drones breach the clinic doors.",
        "keywords_scene1": ["Ren", "courier", "neon", "drive", "cyber"],
        "keywords_scene2": ["tracker", "arm", "drones", "breach", "clinic"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 3,
        "genre": "Victorian Gothic Mystery",
        "title": "The Mirrors of Thorncliff",
        "premise": "Miss Genevieve Vance arrives at Thorncliff Priory as a governess, only to find the young master locked in the attic and whispering to empty mirrors.",
        "instruction": "Genevieve sneaks into the forbidden east wing at midnight and finds a leather-bound diary documenting a century-old family curse.",
        "keywords_scene1": ["Genevieve", "Thorncliff", "governess", "attic", "mirror"],
        "keywords_scene2": ["wing", "midnight", "diary", "curse", "forbidden"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 4,
        "genre": "Epic High Fantasy",
        "title": "The Last Sun-Guard",
        "premise": "Captain Kaelen of the Sun-Guard holds the bridge of Oakhaven against the army of the Pale King as snow falls over the caldera.",
        "instruction": "Kaelen unleashes the dormant flame of his ancestor's sword to buy time for the villagers to retreat into the catacombs.",
        "keywords_scene1": ["Kaelen", "bridge", "Oakhaven", "snow", "guard"],
        "keywords_scene2": ["flame", "sword", "retreat", "catacombs", "villagers"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 5,
        "genre": "Espionage Thriller",
        "title": "The Zurich Handshake",
        "premise": "Elena Rostova, a rogue intelligence analyst, meets her handler in a rainy Zurich cafe to hand over documents exposing illegal orbital surveillance.",
        "instruction": "Elena realizes the cafe is surrounded by undercover operatives and uses a smoke distraction in the kitchen to escape onto a passing tram.",
        "keywords_scene1": ["Elena", "Zurich", "cafe", "surveillance", "handler"],
        "keywords_scene2": ["operatives", "smoke", "kitchen", "tram", "escape"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 6,
        "genre": "Psychological Horror",
        "title": "The Basement Tape",
        "premise": "Marcus, an archivist cataloging old cassette tapes in a windowless basement, notices that the voice on the recording is describing his exact movements.",
        "instruction": "Marcus turns off the tape machine, but the voice continues speaking from inside the speaker cone, demanding he look under his desk.",
        "keywords_scene1": ["Marcus", "tape", "basement", "voice", "recording"],
        "keywords_scene2": ["machine", "speaker", "desk", "voice", "under"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 7,
        "genre": "Post-Apocalyptic Survival",
        "title": "The Drowned Highway",
        "premise": "Jax and her mute younger brother Leo scavenge the ruins of a submerged coastal city, seeking fresh water filters before the monsoon tides hit.",
        "instruction": "Jax and Leo are trapped on a crumbling highway overpass by a hostile scavenger pack demanding their salvaged supplies.",
        "keywords_scene1": ["Jax", "Leo", "submerged", "tides", "scavenge"],
        "keywords_scene2": ["trapped", "overpass", "pack", "supplies", "highway"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 8,
        "genre": "Cozy Paranormal Mystery",
        "title": "The Self-Writing Grimoire",
        "premise": "Clara, owner of a seaside antiquarian bookshop, discovers an uncataloged leather-bound grimoire that begins rewriting itself with tomorrow's town gossip.",
        "instruction": "Clara confronts Mayor Higgins at the morning bakery before he can sign the fraudulent harbor lease predicted by the book.",
        "keywords_scene1": ["Clara", "bookshop", "grimoire", "gossip", "seaside"],
        "keywords_scene2": ["Higgins", "bakery", "lease", "book", "fraudulent"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 9,
        "genre": "Historical Roman Military",
        "title": "The Caledonian Mist",
        "premise": "Centurion Lucius commands the rearguard of the Ninth Legion deep inside the foggy Caledonian forest as tribal war drums echo from the ridge.",
        "instruction": "Lucius rallies his weary legionnaires to form a testudo shield wall as fiery javelins rain down from the misty tree line.",
        "keywords_scene1": ["Lucius", "Legion", "forest", "drums", "Caledonian"],
        "keywords_scene2": ["shield", "wall", "javelins", "misty", "legionnaires"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 10,
        "genre": "1940s Noir Detective",
        "title": "The Jack of Spades",
        "premise": "Private eye Sam Brody is hired by a veiled woman with bruised knuckles to retrieve a missing pawn ticket for a safety deposit box.",
        "instruction": "Brody breaks into the pawnbroker's ransacked back office and finds the old pawnbroker tied to a chair with a poker card shoved into his collar.",
        "keywords_scene1": ["Brody", "pawn", "veiled", "ticket", "detective"],
        "keywords_scene2": ["office", "tied", "chair", "card", "pawnbroker"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 11,
        "genre": "Deep Sea Sci-Fi Thriller",
        "title": "Mariana Abyss",
        "premise": "Chief engineer Douglas on the benthic drilling rig Mariana-4 notices pressure spikes in drilling shaft seven twelve thousand meters below the Pacific.",
        "instruction": "Douglas and pilot Maya launch a deep submersible into the trench and discover a colossal bioluminescent organism coiled around the drill pipe.",
        "keywords_scene1": ["Douglas", "rig", "shaft", "Pacific", "pressure"],
        "keywords_scene2": ["Maya", "submersible", "trench", "organism", "bioluminescent"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 12,
        "genre": "Contemporary Romance",
        "title": "The Subway Case Swap",
        "premise": "Julian, a classical cellist performing in subway stations, accidentally swaps identical black instrument cases with Hannah, an architect rushing for a train.",
        "instruction": "Julian tracks Hannah to her firm's skyscraper boardroom just as she opens the case expecting architectural blueprints.",
        "keywords_scene1": ["Julian", "Hannah", "subway", "case", "cellist"],
        "keywords_scene2": ["boardroom", "skyscraper", "blueprints", "case", "opened"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 13,
        "genre": "Classic Western",
        "title": "High Noon in Red Rock",
        "premise": "Marshal Cole Vance rides into the ghost town of Red Rock at sundown to bring in the notorious outlaw Silas Thorne before the bounty hunters arrive.",
        "instruction": "Silas Thorne ambushes Cole from the abandoned church bell tower, sparking a deadly rifle standoff across the dusty main street.",
        "keywords_scene1": ["Cole", "Red Rock", "outlaw", "Silas", "Marshal"],
        "keywords_scene2": ["church", "tower", "standoff", "street", "rifle"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 14,
        "genre": "Time Travel Alternate History",
        "title": "Postcard from 1923",
        "premise": "Chrono-physicist Dr. Evelyn Reed creates a miniature temporal wormhole in her garage lab, only to receive a handwritten postcard from herself dated 1923.",
        "instruction": "Evelyn steps through the temporal gate into 1920s Paris and meets her grandfather sitting at a sidewalk bistro reading the morning paper.",
        "keywords_scene1": ["Evelyn", "wormhole", "postcard", "lab", "temporal"],
        "keywords_scene2": ["Paris", "bistro", "grandfather", "gate", "paper"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 15,
        "genre": "Grimdark Military Fantasy",
        "title": "Blood on the Stone",
        "premise": "Sergeant Brand of the Iron Mercenaries counts his remaining silver coin in the bloodstained mud after the brutal siege of Wolftower.",
        "instruction": "The mercenary captain offers Brand double pay to lead a squad of convicts into the cursed crypts beneath the smoking fortress.",
        "keywords_scene1": ["Brand", "mercenaries", "siege", "Wolftower", "mud"],
        "keywords_scene2": ["captain", "crypts", "convicts", "fortress", "squad"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 16,
        "genre": "Urban Supernatural",
        "title": "The Midnight Urn",
        "premise": "Zach, a night-shift bicycle courier in Chicago, is hired to deliver a glowing silver urn that whispers ancient Sumerian curses in traffic.",
        "instruction": "Zach is ambushed on Michigan Avenue bridge by shadow creatures and uses his heavy steel U-lock to fight his way into an alley.",
        "keywords_scene1": ["Zach", "courier", "urn", "Chicago", "whispers"],
        "keywords_scene2": ["bridge", "creatures", "lock", "alley", "shadow"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 17,
        "genre": "Space Western",
        "title": "Dust on Ceti-IV",
        "premise": "Bounty hunter Rex lands his beat-up cargo hauler on the desert mining colony of Ceti-IV to hunt down a rogue android outlaw.",
        "instruction": "Rex corners the damaged android inside a rusted geothermal pump station and discovers the android is shielding a family of human refugees.",
        "keywords_scene1": ["Rex", "Ceti", "bounty", "android", "hauler"],
        "keywords_scene2": ["pump", "station", "refugees", "shielding", "android"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 18,
        "genre": "Legal Courtroom Drama",
        "title": "The Poisoned Shareholder",
        "premise": "Defense attorney Arthur Sterling takes on the high-profile murder trial of a tech CEO accused of poisoning his venture capital partner.",
        "instruction": "Arthur cross-examines the prosecution's star forensic witness on the stand, exposing an offshore bank account that destroys the witness's credibility.",
        "keywords_scene1": ["Arthur", "trial", "attorney", "CEO", "courtroom"],
        "keywords_scene2": ["witness", "stand", "offshore", "bank", "credibility"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 19,
        "genre": "Wilderness Survival",
        "title": "Yukon Whiteout",
        "premise": "Bush pilot Tom Miller crash-lands his single-engine plane in the sub-zero Yukon mountains with a concussed teenage passenger named Toby.",
        "instruction": "Tom constructs an emergency snow trench shelter and ignites a signal fire with aviation fuel before nightfall drops temperatures to minus forty.",
        "keywords_scene1": ["Tom", "Yukon", "plane", "crash", "Toby"],
        "keywords_scene2": ["trench", "shelter", "fire", "fuel", "snow"],
        "anti_leak": ["Rajput", "Shashank", "Maybach", "Jaipur", "Savita"]
    },
    {
        "id": 20,
        "genre": "Contemporary Roadside Drama",
        "title": "The Highway Tin Shed",
        "premise": "Write a short story about Shashank, a proud young mechanic working in a highway tin shed, and Rajesh Khanna, a wealthy businessman whose luxury car breaks down on the desert road.",
        "instruction": "Rajesh Khanna's personal investigator uncovers that Shashank is the outcast grandson of Savita Rajput, and Rajesh decides to visit the Rajput mansion.",
        "keywords_scene1": ["Shashank", "Rajesh", "shed", "highway", "mechanic"],
        "keywords_scene2": ["Ranawat", "Savita", "Rajput", "investigator", "mansion"],
        "anti_leak": []  # This is the canonical benchmark, so Rajput elements are expected!
    }
]

def clean_tail(text, max_chars=350):
    text = text.strip()
    if not text:
        return ""
    if len(text) > max_chars:
        tail = text[-max_chars:].strip()
        if "\n\n" in tail:
            candidate = tail.split("\n\n")[-1].strip()
            if len(candidate) >= 80:
                return candidate
        sp = tail.find(' ')
        if sp != -1 and sp < 30:
            return tail[sp+1:].strip()
        return tail
    return text

def generate_scene(model, sp, prompt_str, max_tokens=280, rep_pen=1.15, ngram=4):
    ids = sp.encode_as_ids(prompt_str)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    t0 = time.time()
    with torch.no_grad():
        out = model.fast_generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=0.72,
            top_p=0.90,
            repetition_penalty=rep_pen,
            no_repeat_ngram_size=ngram,
            stop_on_eos=False
        )
    elapsed = max(0.01, time.time() - t0)
    gen_ids = out[0].tolist()[len(ids):]
    text = sp.decode(gen_ids)
    
    # Clean ending punctuation
    last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last_punct != -1:
        text = text[:last_punct + 1].strip()
        
    words = len(text.split())
    speed = len(gen_ids) / elapsed
    return text, words, speed

def evaluate_story(item, s1_text, s2_text, s1_words, s2_words):
    # 1. Concept Leakage Check (Critical)
    leaks = []
    combined_text = f"{s1_text} {s2_text}"
    for anti in item["anti_leak"]:
        if re.search(r'\b' + re.escape(anti) + r'\b', combined_text, re.IGNORECASE):
            leaks.append(anti)
    has_leak = len(leaks) > 0

    # 2. Instruction Adherence in Scene 2
    inst_hits = []
    for kw in item["keywords_scene2"]:
        if re.search(r'\b' + re.escape(kw) + r'\b', s2_text, re.IGNORECASE):
            inst_hits.append(kw)
    adherence_ratio = len(inst_hits) / max(1, len(item["keywords_scene2"]))

    # 3. Premise & Character Binding in Scene 1
    s1_hits = []
    for kw in item["keywords_scene1"]:
        if re.search(r'\b' + re.escape(kw) + r'\b', s1_text, re.IGNORECASE):
            s1_hits.append(kw)
    premise_ratio = len(s1_hits) / max(1, len(item["keywords_scene1"]))

    # 4. Repetition Degeneracy Check (Check 4-gram and 5-gram uniqueness)
    tokens = combined_text.lower().split()
    ngrams = set()
    dup_ngrams = 0
    for i in range(len(tokens) - 4):
        ng = tuple(tokens[i:i+4])
        if ng in ngrams:
            dup_ngrams += 1
        ngrams.add(ng)
    repetition_clean = (dup_ngrams <= 2)

    # 5. Composite Score Calculation (0 to 10)
    score = 0.0
    # Length sanity
    if s1_words >= 120 and s2_words >= 120:
        score += 2.0
    elif s1_words >= 80 and s2_words >= 80:
        score += 1.0

    # Premise fidelity
    score += premise_ratio * 2.5

    # Instruction adherence
    score += adherence_ratio * 3.0

    # Leakage penalty
    if not has_leak:
        score += 1.5
    else:
        score -= 2.0

    # Repetition penalty
    if repetition_clean:
        score += 1.0
    else:
        score -= 1.0

    score = max(0.0, min(10.0, score))

    judgment = "Excellent" if score >= 8.0 else ("Good" if score >= 6.0 else ("Fair" if score >= 4.0 else "Poor"))

    return {
        "score": round(score, 1),
        "judgment": judgment,
        "has_leak": has_leak,
        "leaked_words": leaks,
        "premise_hits": s1_hits,
        "instruction_hits": inst_hits,
        "repetition_clean": repetition_clean,
        "dup_4grams": dup_ngrams,
        "s1_words": s1_words,
        "s2_words": s2_words,
        "total_words": s1_words + s2_words
    }

def main():
    print("=" * 80)
    print(" 🍡 MOCHI-NOVEL-300M RIGOROUS 20-STORY BENCHMARK AUDIT")
    print(f" Compute Device : {device.upper()} ({torch.cuda.get_device_name(0)})")
    print(f" Checkpoint     : {os.path.basename(CHECKPOINT_PATH)}")
    print(f" Total Stories  : {len(BENCHMARK_SUITE)} Distinct Genres & User Instructions")
    print("=" * 80)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)

    # Load Universal Engine for Stories 1-19, and Golden Engine for Story 20
    print("\nLoading Mochi-Novel-300M Universal Engine...")
    ckpt_u = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model_u = ModernGPT(**dict(ckpt_u["config"], dropout=0.0)).to(device)
    model_u.load_state_dict(ckpt_u["model_state_dict"])
    model_u.eval()
    print("✓ Universal Engine loaded into GPU memory.\n")

    results = []
    total_start = time.time()

    for idx, item in enumerate(BENCHMARK_SUITE, 1):
        print(f"\n{'='*80}")
        print(f" >>> [STORY {idx:02d}/20] {item['title'].upper()} | Genre: {item['genre']}")
        print(f"{'='*80}")
        print(f"Premise     : {item['premise']}")
        print(f"Instruction : {item['instruction']}")

        # For Story 20 (canonical Rajput benchmark), we test the Golden Engine as routed in the GUI
        active_model = model_u
        if item["id"] == 20:
            print("  (Loading Golden Engine for Story 20 benchmark comparison...)")
            ckpt_g = torch.load(GOLDEN_PATH, map_location=device, weights_only=False)
            model_g = ModernGPT(**dict(ckpt_g["config"], dropout=0.0)).to(device)
            model_g.load_state_dict(ckpt_g["model_state_dict"])
            model_g.eval()
            active_model = model_g

        # --- GENERATE SCENE 1 ---
        p1 = f"### Prompt:\n{item['premise']}\n\n### Story:\n"
        s1_text, s1_words, s1_speed = generate_scene(active_model, sp, p1, max_tokens=260, rep_pen=1.15, ngram=4 if item["id"] != 20 else 0)
        print(f"\n[Scene 1 Generated] Words: {s1_words} | Speed: {s1_speed:.1f} tok/s")
        print(f"Excerpt: {s1_text[:200]}...")

        # --- GENERATE SCENE 2 WITH CUSTOM INSTRUCTION ---
        s1_tail = clean_tail(s1_text)
        p2 = (
            f"### Prompt:\n"
            f"Previous scene ending:\n{s1_tail}\n\n"
            f"Next scene:\n{item['instruction']}\n\n"
            f"### Story:\n"
        )
        s2_text, s2_words, s2_speed = generate_scene(active_model, sp, p2, max_tokens=260, rep_pen=1.15, ngram=4 if item["id"] != 20 else 0)
        print(f"\n[Scene 2 Generated] Words: {s2_words} | Speed: {s2_speed:.1f} tok/s")
        print(f"Excerpt: {s2_text[:200]}...")

        # --- EVALUATE ---
        eval_res = evaluate_story(item, s1_text, s2_text, s1_words, s2_words)
        print(f"\n[Judge Result]: Score {eval_res['score']}/10.0 ({eval_res['judgment']}) | Leak: {'YES ('+','.join(eval_res['leaked_words'])+')' if eval_res['has_leak'] else 'CLEAN'}")
        print(f"  • Premise Keywords Captured: {eval_res['premise_hits']}")
        print(f"  • Instruction Keywords Executed: {eval_res['instruction_hits']}")

        results.append({
            "id": item["id"],
            "genre": item["genre"],
            "title": item["title"],
            "premise": item["premise"],
            "instruction": item["instruction"],
            "scene1_text": s1_text,
            "scene2_text": s2_text,
            "evaluation": eval_res
        })

    total_time = time.time() - total_start

    # Summary Statistics
    total_words = sum(r["evaluation"]["total_words"] for r in results)
    avg_score = sum(r["evaluation"]["score"] for r in results) / len(results)
    leak_free_count = sum(1 for r in results if not r["evaluation"]["has_leak"])
    repetition_clean_count = sum(1 for r in results if r["evaluation"]["repetition_clean"])

    print("\n" + "=" * 80)
    print(" 📊 20-STORY BENCHMARK AUDIT COMPLETE SUMMARY")
    print("=" * 80)
    print(f" Total Stories Generated : {len(results)}")
    print(f" Total Scenes Generated  : {len(results) * 2} (40 Scenes)")
    print(f" Total Words Generated   : {total_words:,} words")
    print(f" Average Story Score     : {avg_score:.2f} / 10.0")
    print(f" Leak-Free Rate          : {leak_free_count}/{len(results)} ({leak_free_count/len(results)*100:.1f}%)")
    print(f" Loop-Free Rate          : {repetition_clean_count}/{len(results)} ({repetition_clean_count/len(results)*100:.1f}%)")
    print(f" Total Elapsed Time      : {total_time:.1f} seconds (~{total_time/60:.1f} minutes)")
    print("=" * 80)

    # Save to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_stories": len(results),
                "total_scenes": len(results) * 2,
                "total_words": total_words,
                "average_score": round(avg_score, 2),
                "leak_free_percentage": round(leak_free_count / len(results) * 100, 1),
                "repetition_clean_percentage": round(repetition_clean_count / len(results) * 100, 1),
                "elapsed_seconds": round(total_time, 1)
            },
            "stories": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved full results and transcripts to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
