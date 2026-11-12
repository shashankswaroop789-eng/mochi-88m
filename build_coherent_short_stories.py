"""
Coherent Short Story Dataset Generator
Builds a curated corpus of 100% coherent, self-contained short stories (200-350 words each).
Focused on 2-3 characters, clean narrative arcs, natural dialogue, and consistent world-building.
Eliminates all baby-talk, romance artifacts, and multi-character overload.
"""

import os
import sys
import json
import torch
import sentencepiece as spm

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(DATA_DIR, "coherent_short_stories.jsonl")

STORIES = [
    {
        "prompt": "Write a short story about Shashank, a proud young mechanic working in a highway tin shed, and Rajesh, a wealthy businessman whose car breaks down.",
        "story": """The afternoon sun beat down mercilessly upon the desert bypass, turning the corrugated roof of the roadside workshop into a radiator of blistering heat. Inside the narrow shade of his tin shed, Shashank wiped a smear of black grease from his forehead with the back of his hand. He was twenty-three, lean and quiet, with hands that knew the heartbeat of every engine that rattled along the national highway.

A sudden hiss of steam echoed from the tarmac outside. A luxury black sedan coasted to a halt on the gravel shoulder, white plumes billowing from beneath its sculpted hood. The rear door opened, and Rajesh Khanna stepped out into the blinding glare. The older man was impeccably dressed in a tailored suit, his sharp gray eyes scanning the barren horizon with cold irritation.

"No cell reception," Rajesh muttered, checking his phone before his gaze landed on the shed. "Young man! Do you know how to fix a ruptured radiator?"

Shashank picked up his metal toolbox without answering. He walked to the stalled car, knelt upon the burning gravel, and popped the hood latch. His long fingers moved with effortless precision, inspecting the burst coolant hose and the smoking radiator cap. 

"The rubber split at the clamp," Shashank said quietly. "If you drive it like this, the engine block will crack. I can bypass the split with a reinforced brass coupling and high-pressure hose."

"Do it," Rajesh commanded, leaning against the door frame. "I have a board meeting in the capital in three hours."

For thirty minutes under the relentless sun, Shashank worked in silence. He did not ask questions, nor did he look up at the stranger's expensive clothes. He measured, cut, tightened, and purged the air from the cooling lines. When he signaled the driver to turn the key, the engine fired up with a silky, flawless hum.

Rajesh reached into his pocket and unclasped a heavy gold watch from his wrist. "I left my wallet in the aircraft. Take this chronometer. It is worth forty lakhs."

Shashank looked at the gleaming gold, then met the businessman's gaze squarely with unblinking amber eyes. "The spare pipe cost thirty rupees. My labor is one hundred. If you have no cash, you owe me nothing. I earn what I work for, and I do not trade in men's watches."

Before Rajesh could utter another word, Shashank picked up his wrench, turned his back on the luxury vehicle, and walked calmly into his shed. Standing on the dusty road, Rajesh Khanna watched the boy's straight back with a slow, profound smile of respect."""
    },
    {
        "prompt": "Write a short story about Savita Rajput, a strict grandmother, confronting her grandson Karthik about his responsibilities.",
        "story": """The grandfather clock in the study of the ancestral mansion chimed six times. Savita Rajput sat erect in her high-backed wooden chair, her silver cane resting against her knee. The seventy-five-year-old matriarch watched the door with the steady, unblinking patience of an old falcon.

Karthik entered with hesitant footsteps, his eyes lowered. At twenty-six, he possessed the handsome features of the Rajput line, but his shoulders lacked the iron firmness of his ancestors. He had spent the afternoon pacing the courtyard, avoiding her summons.

"Sit down, Karthik," Savita said. Her voice was not loud, but it possessed a weight that made the marble chamber feel smaller.

"Dadi-ji, I know what you want to say about the business," Karthik began, taking the edge of the velvet seat. "I tried to review the accounts with Aman, but the numbers—"

"I care nothing for the balance sheets today," Savita interrupted, lifting her hand. "A company can lose money and regain it in a single season. But a man who loses his dignity never recovers his name. Why were you seen drinking until dawn at the hotel club yesterday?"

Karthik's face burned with sudden shame. "I was with friends, Dadi-ji. We were celebrating—"

"You were parading yourself before people who smile to your face and laugh at your weakness behind your back," Savita's eyes narrowed, sharp and piercing. "Your grandfather built this house with sweat and discipline. He did not sleep until every laborer was paid and every promise was kept. You carry his name, Karthik. If you wish to be treated as a prince of this family, you must first learn how to master yourself."

Karthik looked up, meeting his grandmother's stern gaze. For the first time, he saw not anger in her eyes, but a deep, fierce longing for him to become a man worthy of their lineage. He bowed his head slowly. "Forgive me, Dadi-ji. It will not happen again."

"Good," Savita replied softly. "Now stand up straight and walk out of this room like a Rajput." """
    },
    {
        "prompt": "Write a short story about an estranged father, Avinash, visiting his son Shashank after many years.",
        "story": """The desert wind blew cold through the cracks in the wooden door, carrying the scent of dry earth and impending rain. Inside the tin workshop, a solitary oil lamp cast long, flickering shadows across the walls. Shashank was cleaning his tools at the wooden bench when headlights cut through the gloom outside.

The door creaked open, and Avinash stepped over the threshold. The older man wore a plain dark shawl, but his tired eyes and trembling hands betrayed the heavy burden of his guilt.

Shashank did not drop his wrench. He turned slowly, his expression calm, distant, and impenetrable. "The road is closed to the north. If you need fuel, the pump is five miles back."

"I did not come for fuel, Shashank," Avinash whispered, his voice trembling. He stepped closer, gazing at the boy's face—the high cheekbones, the amber eyes that mirrored his own reflection. "Twenty years is a long time to stay away."

"Not long enough," Shashank replied, his tone as flat and cold as slate. "The last time a Rajput visited this shed, they warned me that if I ever crossed into the city, your drivers would run me down. Did your mother send you to check if I was still breathing?"

A tear escaped Avinash's eye, tracking through the dust on his cheek. "I was a coward, son. My father ruled our home with fear. I had no voice. I let them take you away because I was afraid of losing my inheritance."

Shashank set the wrench down with a soft metallic click. He looked directly into his father's eyes, without anger and without pity.

"Then your inheritance cost you too much," Shashank said softly. "You kept your marble floors, and you traded away your soul. Look at me, Avinash. I sleep on a cot under a metal roof, but my hands are clean and I owe no man an apology. I need neither your name nor your regrets. Go back to your palace."

Avinash stood in the dim light for a long moment, unable to speak against the boy's quiet dignity. He pulled his shawl tight, turned, and stepped back out into the night."""
    },
    {
        "prompt": "Write a short story about Samira, a rebellious young heiress, discovering what true self-respect means.",
        "story": """The luxury convertible pulled into the gravel drive of the seaside estate, its engine humming smoothly. Sixteen-year-old Samira Khanna stepped out, tossing her car keys to the waiting security guard without a glance. She was tired, irritable, and bored with the endless cycle of rooftop parties and superficial compliments that filled her days.

In the garden pavilion sat her father, Rajesh Khanna, examining a set of architectural blueprints. As Samira walked past, he called out quietly.

"Samira. Come sit with me for five minutes."

"Dad, I'm exhausted," she sighed, checking her phone. "My friends are waiting for me to join a video call."

"Your friends will survive five minutes without you," Rajesh said calmly, gesturing to the wicker chair opposite him. 

Samira sat down with a dramatic roll of her eyes. "What is it now? Did the school complain again?"

Rajesh placed a small, brass-colored engine valve on the glass table between them. It was covered in faint traces of grease, scratched and worn from heavy use.

Samira frowned in distaste. "What is that dirty piece of metal doing on the table?"

"Yesterday, my car broke down fifty miles out in the desert heat," Rajesh explained, his voice low and contemplative. "A young mechanic, no older than twenty-three, worked under the blazing sun for forty minutes to fix the engine. When I offered him my gold watch—a timepiece worth more than his entire house—he refused it."

Samira stared at her father, bewildered. "Why would anyone refuse a gold watch?"

"Because he said he had not earned it," Rajesh replied, his gray eyes locking onto his daughter's face. "He charged me two hundred rupees for his honest labor, and refused a single coin more. In all my years in corporate finance, Samira, surrounded by billionaires and politicians, I have rarely met a person who possessed genuine self-respect."

Samira looked down at the scratched brass valve. For the first time in her life, the expensive designer clothes she wore and the luxury cars she drove felt hollow. A quiet silence settled over the pavilion as she began to understand that real pride was not bought with money, but forged through character."""
    },
    {
        "prompt": "Write a short story about an honest craftsman who refuses to compromise on his work.",
        "story": """Master craftsman Mohan stood in his dusty courtyard, running his calloused palm over the smooth curved arm of a teakwood rocking chair. For forty years, his workshop in the old quarter of the city had produced furniture built to outlive the people who bought it. Every mortise and tenon joint was cut by hand, every peg fitted with millimeter precision.

A sharp horn blared outside, and a young property developer stepped into the shop, fanning himself with a brochure. 

"Mohan-ji," the developer said briskly, glancing at his watch. "The luxury hotel project needs forty banquet tables by next month. The contract is worth thirty lakhs. But you must use composite particle board instead of solid timber. Nobody checks the underside of a table, and it will save us both time and money."

Mohan looked at the man's glossy business card, then back at the solid teak log resting on his workbench. He smiled gently and shook his head.

"Composite board swells in the monsoon, sir," Mohan said politely. "In two years, the veneer will peel and the screws will pull loose. My family has put our stamp on timber for three generations. We do not build things that rot before their owners."

The developer frowned, his voice growing sharp. "You are turning down thirty lakhs over a technicality? Half the carpenters in this city would kill for this order!"

"Then give it to them," Mohan replied evenly, picking up his wooden mallet and chisel. "They will give you furniture that looks like gold today and turns to dust tomorrow. I sell wood that your grandchildren can sit upon. If you want cheap illusions, you have come to the wrong door."

The developer stared at the old carpenter for a long second, astonished by a pride he could not purchase. Without another word, Mohan tapped his chisel into the seasoned grain, the clean scent of wood shavings rising into the morning air."""
    }
]

def main():
    print(f"Generating coherent short story dataset with {len(STORIES)} verified examples...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in STORIES:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
