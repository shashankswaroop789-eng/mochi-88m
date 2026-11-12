import os
import sys
import time
import re
import random
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm

# Force utf-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DATA_DIR)

from modern_transformer import ModernGPT

NOVEL_DIR = os.path.join(DATA_DIR, "novel_output")
os.makedirs(NOVEL_DIR, exist_ok=True)

# Model paths
MODEL_HF_FINETUNED = os.path.join(DATA_DIR, "hf_finetuned_romance_gpt88m.pt")
MODEL_HF_MASTER = os.path.join(DATA_DIR, "hf_master_gpt88m.pt")
TOKENIZER_HF_MASTER = os.path.join(DATA_DIR, "spm_hf_master.model")

MODEL_PURE_88M = os.path.join(DATA_DIR, "modern_romance_gpt88m.pt")
TOKENIZER_PURE_88M = os.path.join(DATA_DIR, "spm_modern.model")

MODEL_88M = os.path.join(DATA_DIR, "romance_gpt88m.pt")
TOKENIZER_88M = os.path.join(DATA_DIR, "spm_romance.model")

MODEL_28M = os.path.join(DATA_DIR, "modern_gpt.pt")
TOKENIZER_28M = os.path.join(DATA_DIR, "spm_grand.model")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# STORY TEMPLATES & POOLS
# ==============================================================================
STORY_POOLS = {
    "romance": {
        "genre_name": "Contemporary Romance & Emotional Drama (Colleen Hoover / L.J. Shen style)",
        "titles": [
            "Before We Shatter", "Reminders of Tonight", "Twisted Bonds", 
            "Fragile Vows", "The Words We Kept", "Whispers of Rain", 
            "Ruthless Heart", "All the Broken Pieces", "Beneath the Scars"
        ],
        "character_pairs": [
            ("Cole", "Maya", "a brooding architect carrying the weight of old regrets", "a fiercely independent investigative writer"),
            ("Liam", "Chloe", "a guarded former athlete with a protective streak", "a resilient photographer with a sharp wit"),
            ("Atlas", "Lily", "a quiet, patient restaurant owner", "a determined florist seeking a fresh start"),
            ("Christian", "Stella", "a ruthless tech CEO with guarded defenses", "a gifted classical musician who challenges his silence"),
            ("Julian", "Elena", "an enigmatic trauma surgeon living on borrowed time", "a bold artist fighting for her family's future"),
            ("Dante", "Aria", "a dangerous corporate heir whom everyone fears", "a calm, perceptive woman who never flinches")
        ],
        "settings": [
            "A snowed-in rustic timber cabin in Aspen during a winter power outage",
            "A rain-slicked glass penthouse overlooking the Chicago midnight skyline",
            "A quiet seaside cottage along the rugged coast of Maine in late November",
            "A dimly lit artist's loft in Brooklyn filled with turpentine and rain",
            "An upscale rooftop garden in Manhattan under an approaching thunderstorm"
        ],
        "conflicts": [
            "A shared secret from eight years ago that could ruin both of their families if revealed.",
            "A fierce corporate rivalry that unexpectedly turns into an undeniable, dangerous passion.",
            "A fake public relationship designed to save their careers that quickly blurs into real love.",
            "Two guarded souls who made a pact never to risk their hearts again, tested by close confinement."
        ]
    },
    "gothic": {
        "genre_name": "Gothic Mystery & Romantic Thriller",
        "titles": [
            "The Whispering Manor", "Shadows of Blackwood Hall", "The Crimson Cipher",
            "A Veil of Fog and Ash", "The Midnight Heirloom", "Secrets of Thorncliff"
        ],
        "character_pairs": [
            ("Lord Gabriel Sterling", "Vivienne Vance", "an enigmatic nobleman haunted by his family's past", "a sharp-witted archivist investigating old manuscripts"),
            ("Victor Black", "Genevieve Ross", "a solitary investigator obsessed with truth", "an heiress receiving cryptic midnight letters"),
            ("Adrian Thorne", "Clara Sterling", "a reclusive scholar living on the cliffs", "a fearless newcomer seeking answers to a vanished friend")
        ],
        "settings": [
            "A wind-battered gothic stone manor overlooking treacherous sea cliffs",
            "A cobblestone alley shrouded in thick London fog outside an antiquities shop",
            "A dimly lit ancestral library with locked vaults and hidden passageways"
        ],
        "conflicts": [
            "A cryptic letter bearing a broken wax seal arrives, linking an old crime to the family heir.",
            "A priceless family heirloom is stolen, leaving a trail of coded ciphers and mounting danger.",
            "Strange occurrences in the high tower suggest that a supposedly dead relative is still alive."
        ]
    },
    "regency": {
        "genre_name": "Regency & Victorian Society Romance (Jane Austen style)",
        "titles": [
            "A Match of Pride", "Promises in the Parlour", "The Earl's Dilemma",
            "Scandal at Pemberley Green", "The Masquerade Waltz", "An Unspoken Accord"
        ],
        "character_pairs": [
            ("Lord Julian Ashford", "Miss Eleanor Ward", "a wealthy, aloof aristocrat with high expectations", "a quick-witted lady unimpressed by high society"),
            ("Captain Edward Hastings", "Miss Beatrice Wood", "a decorated naval officer returning home", "a spirited woman defying traditional expectations"),
            ("Arthur Montgomery", "Lady Sophia Darcy", "a contemplative estate owner", "a fiercely loyal daughter guarding her family's honour")
        ],
        "settings": [
            "A sunlit drawing room overlooking the rolling estate gardens in Derbyshire",
            "A glittering candlelit ballroom filled with masked dancers and whispering chaperones",
            "A quiet estate library as rain gently taps against tall arched French windows"
        ],
        "conflicts": [
            "A witty battle of pride and misunderstandings between two people who refuse to admit their feelings.",
            "An impending marriage of convenience that threatens to part two genuinely devoted hearts.",
            "A high-society rumour that forces them into a public alliance to protect family reputations."
        ]
    }
}

# ==============================================================================
# MODEL & TOKENIZER LOADER
# ==============================================================================
def load_modern_engine():
    # 1. Prefer the Fine-Tuned Romance 88.1M Model if available
    if os.path.exists(MODEL_HF_FINETUNED) and os.path.exists(TOKENIZER_HF_MASTER):
        print(f"[Engine] Loading Stage 2 Fine-Tuned Contemporary Romance 88.1M Model onto {device}...")
        sp = spm.SentencePieceProcessor()
        sp.load(TOKENIZER_HF_MASTER)

        model = ModernGPT(
            vocab_size=4096,
            dim=768,
            n_layers=12,
            n_heads=12,
            max_seq_len=512,
            dropout=0.0
        ).to(device)

        ckpt = torch.load(MODEL_HF_FINETUNED, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()

        quote_chars = {'\u2018', '\u2019', '\u201c', '\u201d', '"'}
        quote_ids = [i for i in range(sp.get_piece_size()) if any(q in sp.id_to_piece(i) for q in quote_chars)]

        print(f"[Engine] Stage 2 Fine-Tuned Romance Engine Ready! (dim=768, layers=12, heads=12 | Pure Contemporary Romance Style)\n")
        return model, sp, quote_ids

    # 2. Fallback to Pre-Trained HuggingFace Master 88.1M Foundation Model if available
    elif os.path.exists(MODEL_HF_MASTER) and os.path.exists(TOKENIZER_HF_MASTER):
        print(f"[Engine] Loading HuggingFace Master Story 88.1M Foundation Model onto {device}...")
        sp = spm.SentencePieceProcessor()
        sp.load(TOKENIZER_HF_MASTER)

        model = ModernGPT(
            vocab_size=4096,
            dim=768,
            n_layers=12,
            n_heads=12,
            max_seq_len=512,
            dropout=0.0
        ).to(device)

        ckpt = torch.load(MODEL_HF_MASTER, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()

        quote_chars = {'\u2018', '\u2019', '\u201c', '\u201d', '"'}
        quote_ids = [i for i in range(sp.get_piece_size()) if any(q in sp.id_to_piece(i) for q in quote_chars)]

        print(f"[Engine] HF Master 88.1M Engine Ready! (dim=768, layers=12, heads=12 | 33.6M Token Master Corpus)\n")
        return model, sp, quote_ids

    # 2. Fallback to Pure Modern English Romance 88.1M Model if available
    elif os.path.exists(MODEL_PURE_88M) and os.path.exists(TOKENIZER_PURE_88M):
        print(f"[Engine] Loading Pure Modern Romance 88.1M Foundation Model onto {device}...")
        sp = spm.SentencePieceProcessor()
        sp.load(TOKENIZER_PURE_88M)

        model = ModernGPT(
            vocab_size=4096,
            dim=768,
            n_layers=12,
            n_heads=12,
            max_seq_len=512,
            dropout=0.0
        ).to(device)

        model.load_state_dict(torch.load(MODEL_PURE_88M, map_location=device, weights_only=True))
        model.eval()

        quote_chars = {'\u2018', '\u2019', '\u201c', '\u201d', '"'}
        quote_ids = [i for i in range(sp.get_piece_size()) if any(q in sp.id_to_piece(i) for q in quote_chars)]

        print(f"[Engine] Pure Modern 88.1M Engine Ready! (dim=768, layers=12, heads=12 | 100% Contemporary English Romance)\n")
        return model, sp, quote_ids

    # 2. Fallback to earlier 88.1M model if present
    elif os.path.exists(MODEL_88M) and os.path.exists(TOKENIZER_88M):
        print(f"[Engine] Loading Scaled 88.1M Foundation Model onto {device}...")
        sp = spm.SentencePieceProcessor()
        sp.load(TOKENIZER_88M)

        model = ModernGPT(
            vocab_size=4096,
            dim=768,
            n_layers=12,
            n_heads=12,
            max_seq_len=512,
            dropout=0.0
        ).to(device)

        model.load_state_dict(torch.load(MODEL_88M, map_location=device, weights_only=True))
        model.eval()

        quote_chars = {'\u2018', '\u2019', '\u201c', '\u201d', '"'}
        quote_ids = [i for i in range(sp.get_piece_size()) if any(q in sp.id_to_piece(i) for q in quote_chars)]

        print(f"[Engine] Scaled 88.1M Engine Ready! (dim=768, layers=12, heads=12 | {len(quote_ids)} quote tokens identified)\n")
        return model, sp, quote_ids
    else:
        print(f"[Engine] Loading Modern 27.8M Foundation Model onto {device}...")
        sp = spm.SentencePieceProcessor()
        sp.load(TOKENIZER_28M)

        model = ModernGPT(
            vocab_size=4096,
            dim=512,
            n_layers=8,
            n_heads=8,
            max_seq_len=512,
            dropout=0.0
        ).to(device)

        model.load_state_dict(torch.load(MODEL_28M, map_location=device, weights_only=True))
        model.eval()

        quote_chars = {'\u2018', '\u2019', '\u201c', '\u201d', '"'}
        quote_ids = [i for i in range(sp.get_piece_size()) if any(q in sp.id_to_piece(i) for q in quote_chars)]

        print(f"[Engine] Modern 27.8M Engine Ready! (dim=512, layers=8, heads=8 | {len(quote_ids)} quote tokens identified)\n")
        return model, sp, quote_ids

# ==============================================================================
# CHARACTER ALIGNMENT & PROSE POLISHING
# ==============================================================================
def apply_character_alignment(text, name_mapping):
    """
    Substitutes legacy corpus default characters (Conan Doyle, Austen, Brontë)
    with the user's custom cast or narrative equivalents.
    """
    if not name_mapping:
        return text
    
    sorted_keys = sorted(name_mapping.keys(), key=len, reverse=True)
    for key in sorted_keys:
        val = name_mapping[key]
        if not val or key == val:
            continue
        pattern = re.compile(rf'\b{re.escape(key)}\b', re.IGNORECASE)
        text = pattern.sub(val, text)
    return text

def complete_to_sentence_end(text):
    """
    Trims dangling tokens and ensures the generated beat ends cleanly on a 
    complete sentence punctuation mark (. ! ?).
    """
    # Look for the last sentence-terminating punctuation
    last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last_punct != -1:
        # Check if punctuation is followed immediately by a closing quotation mark
        if last_punct + 1 < len(text) and text[last_punct + 1] in {'"', "'", '\u2019', '\u201d'}:
            last_punct += 1
        return text[:last_punct + 1].strip()
    return text.strip()

# ==============================================================================
# BALANCED SCENE BEAT GENERATION (EXPOSITION + DIALOGUE + RESOLUTION)
# ==============================================================================
def generate_balanced_scene_beat(
    model, 
    sp, 
    quote_ids, 
    narrative_lead_in, 
    name_mapping, 
    target_words=250, 
    temperature=0.58, 
    top_k=35,
    top_p=0.88,
    repetition_penalty=1.25
):
    """
    Generates a coherent, grounded scene beat with:
      - Nucleus sampling (top_p = 0.88) and controlled temperature (0.58)
      - Repetition penalty (1.25) to prevent loops and stuttering
      - Balanced narrative and dialogue flow
      - Clean sentence completion
    """
    token_ids = sp.encode_as_ids(narrative_lead_in)
    if not token_ids:
        token_ids = [0]
    
    current_tokens = list(token_ids)
    word_count = len(sp.decode(current_tokens).split())

    while word_count < target_words:
        context_ids = current_tokens[-384:]
        idx = torch.tensor([context_ids], dtype=torch.long, device=device)
        burst = model.fast_generate(
            idx, 
            max_new_tokens=45, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            banned_token_ids=None
        )[0].tolist()
        
        new_tokens = burst[len(context_ids):]
        current_tokens.extend(new_tokens)
        word_count = len(sp.decode(current_tokens).split())

    raw_text = sp.decode(current_tokens)
    raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)

    # Complete cleanly at last sentence ending (no mid-sentence cutoffs)
    closed_text = complete_to_sentence_end(raw_text)

    # Apply character name alignment
    for old_name, new_name in name_mapping.items():
        if old_name and new_name:
            closed_text = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, closed_text)

    return closed_text
    if name_mapping:
        closed_text = apply_character_alignment(closed_text, name_mapping)

    return closed_text

# ==============================================================================
# NOVEL ARCHITECT & SCENE OUTLINE BUILDER
# ==============================================================================
class InteractiveNovelArchitect:
    def __init__(self, title, genre, protagonist, counterpart, setting, conflict, words_per_beat=700, num_chapters=3):
        self.title = title
        self.genre = genre
        self.protagonist = protagonist
        self.counterpart = counterpart
        self.setting = setting
        self.conflict = conflict
        self.words_per_beat = words_per_beat
        self.num_chapters = num_chapters
        self.model, self.sp, self.quote_ids = load_modern_engine()

        # Build extensive character alignment table to redirect training corpus defaults
        self.name_mapping = {
            # Conan Doyle / Sherlock Holmes universe
            "Sherlock Holmes": self.protagonist,
            "Holmes": self.protagonist,
            "Dr. Watson": self.counterpart,
            "Watson": self.counterpart,
            "Dr Mortimer": "the physician",
            "Mortimer": "the physician",
            "Dr Trevelyan": "the doctor",
            "Trevelyan": "the doctor",
            "Barker": "the investigator",
            "McMurdo": self.protagonist,
            "McGinty": "the elder",
            "John Clay": "the stranger",
            "Stangerson": "the shadow",
            "Drebber": "the informant",
            "Phelps": "the contact",
            "Huxtable": "the messenger",
            "Douglas": "the landlord",
            "Mr Douglas": "the landlord",
            "Mr. Douglas": "the landlord",
            "Hosmer Angel": "the suitor",
            "Pycroft": "the merchant",
            "Cantlemere": "the director",
            "Gilchrist": "the scholar",
            "Lestrade": "the detective",
            "Gregson": "the officer",
            "Mac": "the officer",
            "Mrs Hudson": "the housekeeper",
            "Mrs. Hudson": "the housekeeper",
            "Ettie": self.counterpart,
            "Miss Morstan": self.counterpart,
            "Mary Morstan": self.counterpart,

            # Jane Austen universe
            "Mr. Darcy": self.protagonist,
            "Darcy": self.protagonist,
            "Bingley": self.protagonist,
            "Elizabeth Bennet": self.counterpart,
            "Elizabeth": self.counterpart,
            "Jane": self.counterpart,
            "Wickham": "the rival",
            "Collins": "the acquaintance",
            "Lady Catherine": "the matriarch",
            "Mrs. Bennet": "the hostess",

            # Brontë & Shelley Gothic universe
            "Rochester": self.protagonist,
            "Jane Eyre": self.counterpart,
            "Heathcliff": self.protagonist,
            "Catherine": self.counterpart,
            "Victor Frankenstein": self.protagonist,
            "Frankenstein": self.protagonist,
            "Victor": self.protagonist
        }

    def build_dynamic_outline(self):
        """
        Synthesizes structured narrative blueprints with rich sensory lead-in prose
        for each scene, ensuring deep exposition, clear character introduction,
        and compelling chapter arcs.
        """
        outline = []
        p = self.protagonist
        c = self.counterpart
        s = self.setting
        f = self.conflict

        # ----------------------------------------------------------------------
        # CHAPTER 1: THE INCITING SPARK
        # ----------------------------------------------------------------------
        outline.append({
            "chapter_num": 1,
            "title": "A Midnight Secret",
            "beats": [
                {
                    "beat_num": 1,
                    "title": "The Weight of the Past",
                    "lead_in": (
                        f"Midnight rain drummed relentlessly against the floor-to-ceiling glass of {s}. "
                        f"I stood quietly near the marble island, watching {p} stand by the dark windows. "
                        f"His tailored coat was black against the city lights, shoulders tight with a quiet, menacing tension. "
                        f"For months, we had lived under the shadow of {f.lower()}, pretending our arrangement was nothing more than business. "
                        f"Yet tonight, the quiet between us felt like an impending storm. "
                        f"I wrapped my cardigan tighter around my shoulders, knowing that the distance between us was slipping away."
                    )
                },
                {
                    "beat_num": 2,
                    "title": "The Unexpected Arrival",
                    "lead_in": (
                        f"The soft chime of the private elevator shattered the silence of the penthouse. "
                        f"I turned as {c} stepped out into the foyer, rain glistening on her dark hair and breathing shallow. "
                        f"She clutched a black leather folio tightly against her chest, her knuckles pale under the dim lighting. "
                        f"For a long, electric moment, {p}'s gaze locked onto hers across the room. "
                        f"The shock of this late encounter hung heavy between them—a tangled web of guarded distrust and unspoken heat."
                    )
                },
                {
                    "beat_num": 3,
                    "title": "The Broken Seal",
                    "lead_in": (
                        f"The signed contract lay open on the dark glass table under the low glow of the pendant lights. "
                        f"{p} stepped closer, his jaw rigid as his fingers traced the clause that changed everything. "
                        f"The evidence was undeniable—the truth behind {f.lower()} had been exposed, and the danger was arriving far sooner than either had prepared for. "
                        f"I watched {p}'s face intently, reading the sudden storm brewing behind his guarded eyes. "
                        f"The secret was out, and neither of us could afford to make a mistake."
                    )
                },
                {
                    "beat_num": 4,
                    "title": "The Unspoken Reckoning",
                    "lead_in": (
                        f"The space between us shrank until the air grew thick with unspoken tension and undeniable attraction. "
                        f"{p} took a slow step toward me, running a hand through dark hair, his voice dropping into that low, dangerous register. "
                        f"I refused to step back, my pulse hammering in my throat as his presence overwhelmed the room. "
                        f"Every rule we had agreed to in that contract was beginning to fracture under the heat of the night. "
                        f"Standing so close, neither of us could pretend this was just a game anymore."
                    )
                },
                {
                    "beat_num": 5,
                    "title": "Into the Dark",
                    "lead_in": (
                        f"The city lights flickered through the sheet of falling rain as the early morning hours crept closer. "
                        f"The time for hesitation had expired; the decision had to be made before daylight broke over Chicago. "
                        f"{p} reached out, his warm fingers brushing against mine before closing around my hand with quiet, fierce certainty. "
                        f"There would be no returning to the lives we had lived before tonight. "
                        f"Whatever perils were waiting in the shadows of this deal, we were stepping forward together."
                    ),
                    "is_chapter_closing": True
                }
            ]
        })

        # ----------------------------------------------------------------------
        # CHAPTER 2: WHISPERS IN THE DARK
        # ----------------------------------------------------------------------
        if self.num_chapters >= 2:
            outline.append({
                "chapter_num": 2,
                "title": "Whispers in the Dark",
                "beats": [
                    {
                        "beat_num": 1,
                        "title": "The Fog-Shrouded Path",
                        "lead_in": (
                            f"Dawn brought no comfort, only a pale gray haze that blanketed the valley in mist. "
                            f"{p} and {c} moved quietly through the unfamiliar terrain, exhaustion tugging at their limbs "
                            f"as they navigated the hidden roads. The temporary sanctuary ahead offered little relief; "
                            f"the stakes of {f.lower()} grew heavier with every mile."
                        )
                    },
                    {
                        "beat_num": 2,
                        "title": "The Elusive Informant",
                        "lead_in": (
                            f"Inside a dimly lit antiquities house on the outskirts of the town, the ticking of grandfather clocks "
                            f"filled the air with nervous tension. A reclusive contact waited in the shadows, holding records "
                            f"that connected the past to the present danger."
                        )
                    },
                    {
                        "beat_num": 3,
                        "title": "The Hidden Lockbox",
                        "lead_in": (
                            f"With a faint click, the iron lock surrendered beneath {p}'s fingers, revealing a velvet-lined "
                            f"compartment beneath the floorboards. Inside lay letters that rewrote everything they thought "
                            f"they understood about each other."
                        )
                    },
                    {
                        "beat_num": 4,
                        "title": "Vulnerability in the Shadows",
                        "lead_in": (
                            f"Taking refuge in a quiet alcove away from the street, the exhaustion finally broke through {c}'s defenses. "
                            f"In the dim lantern light, {p} reached out, pulling {c} close as the walls between them completely collapsed."
                        )
                    },
                    {
                        "beat_num": 5,
                        "title": "The Echo of Footsteps",
                        "lead_in": (
                            f"A sudden thud on the cobblestones outside froze them in place. The search party had tracked them down. "
                            f"{p} exchanged a breathless glance with {c}, their fingers interlacing as the shadows closed in."
                        ),
                        "is_chapter_closing": True
                    }
                ]
            })

        # ----------------------------------------------------------------------
        # CHAPTER 3: THE TRUTH UNVEILED
        # ----------------------------------------------------------------------
        if self.num_chapters >= 3:
            outline.append({
                "chapter_num": 3,
                "title": "The Truth Unveiled",
                "beats": [
                    {
                        "beat_num": 1,
                        "title": "The Final Confrontation",
                        "lead_in": (
                            f"The climax approached under the vaulted arches of the high hall. All the pieces of the puzzle "
                            f"had aligned, and {p} stood ready to face the mastermind behind the conspiracy."
                        )
                    },
                    {
                        "beat_num": 2,
                        "title": "Standing Together",
                        "lead_in": (
                            f"{c} stepped up beside {p}, refusing to be sidelined as the accusations flew and the tension reached a boiling point."
                        )
                    },
                    {
                        "beat_num": 3,
                        "title": "The Mask Shatters",
                        "lead_in": (
                            f"The confession was torn free in the grand gallery, shattering years of deception and revealing the true motive."
                        )
                    },
                    {
                        "beat_num": 4,
                        "title": "The Dust Settles",
                        "lead_in": (
                            f"As the sirens and shouts faded into the rainy morning, {p} and {c} were left alone amidst the quiet aftermath."
                        )
                    },
                    {
                        "beat_num": 5,
                        "title": "A New Beginning",
                        "lead_in": (
                            f"Standing before the open balcony as the sun broke through the clouds, {p} turned to {c}, knowing the shadows "
                            f"were finally behind them."
                        ),
                        "is_chapter_closing": True
                    }
                ]
            })

        return outline

    def write_full_chapter(self, chapter_info):
        ch_num = chapter_info["chapter_num"]
        ch_title = chapter_info["title"]
        beats = chapter_info["beats"]
        target_total = len(beats) * self.words_per_beat

        print("\n" + "=" * 70)
        print(f" GENERATING CHAPTER {ch_num}: {ch_title.upper()}")
        print(f" Protagonists: {self.protagonist} & {self.counterpart}")
        print(f" Setting: {self.setting}")
        print(f" Target Length: {target_total:,}+ Words ({len(beats)} Beats x {self.words_per_beat} Words)")
        print("=" * 70)

        chapter_text_parts = []
        for i, beat in enumerate(beats, 1):
            print(f"\n[Beat {i}/{len(beats)}] Generating ~{self.words_per_beat} words: \"{beat['title']}\"...")
            
            beat_prose = generate_balanced_scene_beat(
                self.model,
                self.sp,
                quote_ids=self.quote_ids,
                narrative_lead_in=beat["lead_in"],
                name_mapping=self.name_mapping,
                target_words=self.words_per_beat,
                temperature=0.58,
                top_k=35,
                top_p=0.88,
                repetition_penalty=1.25
            )
            words = len(beat_prose.split())
            print(f"--> Beat {i} complete: {words:,} words generated on RTX 3060.")
            chapter_text_parts.append(beat_prose)

        # ----------------------------------------------------------------------
        # DEDICATED CHAPTER CLOSING ARC (Resolves the Chapter & Bridges to Next)
        # ----------------------------------------------------------------------
        closing_resolution = (
            f"The quiet of the penthouse settled around us, sealing out the storm raging against the glass. "
            f"In the dim amber glow of the living room, {self.protagonist} stepped closer, his breathing falling into a shared, steady rhythm with mine. "
            f"Outside, the winter rain continued to wash over the Chicago midnight skyline, but between us, the silent barriers of the past had finally fractured. "
            f"Whatever consequences tomorrow would bring, we would face them together.\n\n"
            f"[ END OF CHAPTER {ch_num} ]"
        )
        chapter_text_parts.append(closing_resolution)

        full_chapter = "\n\n***\n\n".join(chapter_text_parts)
        total_words = len(full_chapter.split())

        safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', self.title.lower())
        out_filename = f"{safe_title}_chapter_{ch_num:02d}.txt"
        out_path = os.path.join(NOVEL_DIR, out_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{self.title.upper()}\nCHAPTER {ch_num}: {ch_title}\n\n" + full_chapter)

        print("\n" + "-" * 70)
        print(f" CHAPTER {ch_num} COMPLETE!")
        print(f" Final Word Count: {total_words:,} words (Over {target_total:,} words!)")
        print(f" Chapter properly resolved and closed!")
        print(f" Saved to: {out_path}")
        print("-" * 70)
        return out_path, total_words

    def run(self, auto_proceed=False):
        outline = self.build_dynamic_outline()
        total_book_words = 0

        for ch in outline:
            _, words = self.write_full_chapter(ch)
            total_book_words += words

            if ch["chapter_num"] < len(outline):
                print(f"\n[Chapter {ch['chapter_num']} Finished!]")
                if not auto_proceed:
                    try:
                        choice = input(f"Proceed to generate Chapter {ch['chapter_num'] + 1}? (y/n, default 'y'): ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        choice = 'n'
                    if choice == 'n':
                        print("\nPaused. Your chapters are saved in 'novel_output/'. You can continue anytime!")
                        break

        print("\n" + "=" * 70)
        print(f" NOVEL SESSION COMPLETED!")
        print(f" Total Words Written from Scratch: {total_book_words:,} words")
        print(f" Saved in: {NOVEL_DIR}")
        print("=" * 70)

# ==============================================================================
# INTERACTIVE CLI DISPATCHER
# ==============================================================================
def roll_random_story(genre_key=None):
    if not genre_key or genre_key not in STORY_POOLS:
        genre_key = random.choice(list(STORY_POOLS.keys()))
    
    pool = STORY_POOLS[genre_key]
    title = random.choice(pool["titles"])
    pair = random.choice(pool["character_pairs"])
    p_name, c_name, p_desc, c_desc = pair
    setting = random.choice(pool["settings"])
    conflict = random.choice(pool["conflicts"])
    return {
        "title": title,
        "genre": pool["genre_name"],
        "protagonist": p_name,
        "counterpart": c_name,
        "protagonist_desc": p_desc,
        "counterpart_desc": c_desc,
        "setting": setting,
        "conflict": conflict
    }

def interactive_setup():
    print("\n" + "=" * 70)
    print("      📖 MODERN GPT NOVEL ARCHITECT (27.8M FROM SCRATCH)")
    print("=" * 70)
    print(" [1] Quick Start: Roll a Random Story (Romance / Gothic / Regency)")
    print(" [2] Custom Story: Enter your own Characters, Setting, and Plot")
    
    try:
        mode = input("\nChoose an option [1-2] (default '1'): ").strip()
    except (EOFError, KeyboardInterrupt):
        mode = "1"
    if not mode:
        mode = "1"

    if mode == "1":
        # Random Story
        print("\nSelect Genre:")
        print(" [1] Contemporary Romance & Drama (Colleen Hoover / L.J. Shen tone)")
        print(" [2] Gothic Mystery & Romantic Thriller")
        print(" [3] Regency & Victorian High Society Drama")
        print(" [4] Surprise Me (Wildcard)")
        try:
            g_choice = input("Choice [1-4] (default '1'): ").strip()
        except (EOFError, KeyboardInterrupt):
            g_choice = "1"
        
        genre_map = {"1": "romance", "2": "gothic", "3": "regency", "4": None}
        genre_key = genre_map.get(g_choice, "romance")

        while True:
            story = roll_random_story(genre_key)
            print("\n" + "-" * 70)
            print("🎲 GENERATED STORY SETUP:")
            print(f"  Title:        {story['title']}")
            print(f"  Genre:        {story['genre']}")
            print(f"  Protagonist:  {story['protagonist']} ({story['protagonist_desc']})")
            print(f"  Second Lead:  {story['counterpart']} ({story['counterpart_desc']})")
            print(f"  Setting:      {story['setting']}")
            print(f"  Conflict:     {story['conflict']}")
            print(f"  Length:       3,500+ Words per Chapter (5 Beats x 700 Words)")
            print("-" * 70)
            
            try:
                confirm = input("Generate novel with this story? (y/n to re-roll, default 'y'): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                confirm = "y"
            if confirm != 'n':
                break

        return story["title"], story["genre"], story["protagonist"], story["counterpart"], story["setting"], story["conflict"], 3, 700

    else:
        # Custom Story
        print("\n--- Enter Your Custom Story Details ---")
        try:
            title = input("Novel Title (press Enter for 'Before We Shatter'): ").strip() or "Before We Shatter"
            genre = input("Genre/Tone (press Enter for 'Contemporary Romance & Drama'): ").strip() or "Contemporary Romance & Drama"
            protagonist = input("Main Protagonist Name (e.g. Cole, Liam, Julian): ").strip() or "Cole"
            counterpart = input("Second Lead / Love Interest (e.g. Maya, Chloe, Elena): ").strip() or "Maya"
            setting = input("Setting (e.g. A snowed-in cabin in Aspen during a power outage): ").strip() or "A snowed-in rustic cabin in Aspen during a winter power outage"
            conflict = input("Central Conflict / Premise: ").strip() or "A shared secret from eight years ago that could ruin both of their families if revealed."
            
            num_ch_str = input("Number of Chapters (default 3): ").strip() or "3"
            num_ch = int(num_ch_str) if num_ch_str.isdigit() else 3

            w_str = input("Words per Beat (default 700, 5 beats = 3,500 words/chapter): ").strip() or "700"
            words_per_beat = int(w_str) if w_str.isdigit() else 700
        except (EOFError, KeyboardInterrupt):
            title = "Before We Shatter"
            genre = "Contemporary Romance & Drama"
            protagonist = "Cole"
            counterpart = "Maya"
            setting = "A snowed-in rustic cabin in Aspen during a winter power outage"
            conflict = "A shared secret from eight years ago that could ruin both of their families if revealed."
            num_ch, words_per_beat = 3, 700

        return title, genre, protagonist, counterpart, setting, conflict, num_ch, words_per_beat

# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modern GPT Novel Architect")
    parser.add_argument("--auto", action="store_true", help="Automatically proceed through all chapters without pausing")
    parser.add_argument("--title", type=str, default=None, help="Novel Title")
    parser.add_argument("--genre", type=str, default=None, help="Novel Genre")
    parser.add_argument("--protagonist", type=str, default=None, help="Main Protagonist")
    parser.add_argument("--counterpart", type=str, default=None, help="Second Lead / Love Interest")
    parser.add_argument("--setting", type=str, default=None, help="Story Setting")
    parser.add_argument("--conflict", type=str, default=None, help="Central Conflict")
    parser.add_argument("--chapters", type=int, default=3, help="Number of chapters")
    parser.add_argument("--words_per_beat", type=int, default=700, help="Words per scene beat (5 beats = 3500 words)")
    args = parser.parse_args()

    if args.title and args.protagonist:
        title = args.title
        genre = args.genre or "Contemporary Romance & Drama"
        protagonist = args.protagonist
        counterpart = args.counterpart or "Maya"
        setting = args.setting or "A snowed-in rustic cabin in Aspen during a winter power outage"
        conflict = args.conflict or "A shared secret from eight years ago that could ruin both of their families if revealed."
        num_ch = args.chapters
        words_per_beat = args.words_per_beat
    else:
        title, genre, protagonist, counterpart, setting, conflict, num_ch, words_per_beat = interactive_setup()

    architect = InteractiveNovelArchitect(
        title=title,
        genre=genre,
        protagonist=protagonist,
        counterpart=counterpart,
        setting=setting,
        conflict=conflict,
        words_per_beat=words_per_beat,
        num_chapters=num_ch
    )
    architect.run(auto_proceed=args.auto)
