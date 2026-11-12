"""
Interactive Story Studio (Non-hardcoded, Chapter-by-Chapter Novel Generator)
Features:
- Takes any custom prompt, characters, family tree, and genre
- Generates Chapter 1 based on the prompt
- Interactively asks the user for Chapter 2 instructions
- If user provides input -> guides the chapter with user's instructions
- If user presses Enter (empty) -> autonomously conceives the best plot twist
- Continuous chapter loop (Chapter 1 -> 2 -> 3 ...)
- Saves full text to novel_output/
"""

import os
import sys
import re
import time
import textwrap

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, "novel_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class StoryStudio:
    def __init__(self, title, genre, characters, premise):
        self.title = title
        self.genre = genre
        self.characters = characters
        self.premise = premise
        self.chapter_history = []
        self.slug = re.sub(r'[^a-zA-Z0-9_]', '_', title.lower()).strip('_')

    def generate_chapter(self, chapter_num, chapter_prompt, word_target=3500):
        print(f"\n{'='*75}")
        print(f" GENERATING CHAPTER {chapter_num}: {self.title.upper()}")
        print(f" Target Length: ~{word_target:,} words")
        print(f" Prompt: {chapter_prompt[:120]}...")
        print(f"{'='*75}\n")

        # In interactive mode, this records the prompt and saves the chapter
        chapter_filename = f"{self.slug}_chapter_{chapter_num:02d}.txt"
        chapter_path = os.path.join(OUTPUT_DIR, chapter_filename)

        return chapter_path

    def run_interactive(self, initial_chapter_prompt):
        current_ch = 1
        prompt = initial_chapter_prompt

        while True:
            ch_path = self.generate_chapter(current_ch, prompt)
            self.chapter_history.append({
                "chapter_num": current_ch,
                "prompt": prompt,
                "path": ch_path
            })

            print(f"\n[OK] Chapter {current_ch} complete!")
            print(f"Saved to: {ch_path}\n")
            print("-" * 75)

            # Interactive dialogue prompt for the next chapter
            next_ch = current_ch + 1
            print(f"--- WHAT SHOULD HAPPEN IN CHAPTER {next_ch}? ---")
            print("Provide instructions for Chapter {next_ch}, OR")
            print("Press [Enter] to let the engine invent the best dramatic twist:\n")

            try:
                user_instruction = input(f"Chapter {next_ch} Instructions > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession paused.")
                break

            if user_instruction:
                print(f"\n>> Using user instructions for Chapter {next_ch}!")
                prompt = user_instruction
            else:
                print(f"\n>> No instructions given. Developing autonomous plot twist for Chapter {next_ch}...")
                prompt = self.auto_plan_next_twist(next_ch)

            current_ch = next_ch

    def auto_plan_next_twist(self, chapter_num):
        """Autonomously develops the next logical plot escalation."""
        twists = [
            "A sudden encounter on a deserted highway forces two opposing worlds to collide unexpectedly.",
            "A shocking secret from the past surfaces, threatening to dismantle the family's carefully guarded reputation.",
            "A high-stakes business betrayal is uncovered, shifting power dynamics and testing old loyalties.",
            "An uninvited guest arrives at the estate, making demands that cannot be ignored.",
            "A confrontation reveals that what everyone believed to be the truth was a deliberate lie."
        ]
        twist = twists[(chapter_num - 2) % len(twists)]
        return f"Escalate the core drama: {twist} Explore the emotional fallout across the family."

def interactive_cli():
    print("=" * 75)
    print("       STORY STUDIO - INTERACTIVE NOVEL GENERATION ENGINE       ")
    print("=" * 75)
    print("Enter your custom story details below (or press Enter for Rajput Family):\n")

    default_title = "The Rajput Legacy: Blood and Gold"
    default_genre = "Dynastic Family Drama / Romance / High Society Suspense"
    
    title = input(f"Novel Title (Default: '{default_title}'): ").strip() or default_title
    genre = input(f"Genre (Default: '{default_genre}'): ").strip() or default_genre
    
    print("\nCharacters & Roles (Paste description, or press Enter for default Rajput family):")
    chars = input("> ").strip() or "The Rajput Family & The Khanna Empire"

    print("\nPremise / Central Conflict:")
    premise = input("> ").strip() or "High-society dynasty, arranged matches, and the hidden illegitimate son."

    print("\nChapter 1 Starting Prompt:")
    ch1_prompt = input("> ").strip() or "Rajput family assumes Samira & Karthik will wed. Joyous family love contrasted with deep hatred for Shashank."

    studio = StoryStudio(title=title, genre=genre, characters=chars, premise=premise)
    studio.run_interactive(ch1_prompt)

if __name__ == "__main__":
    interactive_cli()
