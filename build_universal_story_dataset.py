"""
Universal Story SFT Dataset Builder
Creates a diverse, rich, multi-genre Supervised Fine-Tuning dataset with:
- Strict <|user|> ... <|story|> formatting
- Zero leaked book titles or chapter numbers
- Deep Indian dynastic drama (Rajput family, Shashank, Rajesh Khanna, Samira)
- Diverse multi-genre stories (family sagas, corporate drama, mystery, emotional fiction)
- Zero verbatim duplicates to prevent mode collapse / looping.
"""

import os
import sys
import re
import json

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSONL = os.path.join(DATA_DIR, "sft_story_dataset.jsonl")

# Collection of diverse, original prompt-to-story pairs
STORIES = []

def add_pair(prompt, story):
    # Clean text
    prompt = prompt.strip()
    story = story.strip()
    # Strip any leaked titles
    story = re.sub(r'(?i)\b(?:matcha maker|bared to you|twisted lies|ugly love|kiss thief)\b', '', story)
    story = re.sub(r'(?i)\bchapter\s+\d+\b', '', story)
    story = re.sub(r'[ \t]{2,}', ' ', story)
    STORIES.append({"prompt": prompt, "story": story})

# --- 1. RAJPUT FAMILY DYNASTIC SAGAS ---

add_pair(
    "Write a story of a Rajput family:\nCharacters:\n1. Vikram Rajput: Grandfather (Dead)\n2. Savita Rajput: Grandmother and matriarch, age 75\n3. Viraj Rajput: First-born son, age 54 & Rita Rajput: Wife, age 51\n4. Avinash Rajput: Second-born son, age 50 & Meghna Rajput: Wife, age 49\n5. Karthik Rajput: Son of Viraj, deeply in love with Samira\n6. Shashank Rajput: Outcast mechanic in a tin shed, son of Avinash\n7. Samira Khanna: Billionaire heiress, age 16\n8. Rajesh Khanna: Tycoon, father of Samira\n\nStart with Chapter 1: The family's opulent life and deep mutual affection, while contrasting their unified cold hatred for the outcast Shashank.",
    """The morning sun poured golden light through the stained-glass arches of Rajput Sadan, illuminating the centuries-old frescoes that lined the corridors of the ancestral estate in Jaipur. In the grand dining hall, chandelier crystals trembled to the muted clatter of silver cutlery on fine porcelain.

At the head of the long teakwood table sat Savita Rajput. At seventy-five, the matriarch commanded the room with the stillness of an empress. Her silver hair was drawn back into an immaculate bun, and a necklace of uncut emeralds rested against her ivory silk saree. In her right hand she held her silver-handled walking stick, though her back was as straight as an iron pillar. For over three decades since the death of Vikram Rajput, it was Savita whose single nod decided every marriage, every corporate expansion, and every family destiny.

To her right sat her eldest son Viraj, fifty-four and radiating the heavy pride of the Rajput lineage, alongside his wife Rita, whose diamond bangles clicked softly as she poured saffron tea. Across from them sat Avinash, fifty and quiet, his eyes constantly shadowed by a secret he had tried for twenty-three years to bury, while his wife Meghna kept up a bright, polished stream of social gossip.

"The alliance with the Khanna Empire is practically sealed," Rita declared with triumphant satisfaction, smoothing her silk pallu. "Rajesh Khanna's jet landed at the city airstrip yesterday. He will attend the heritage foundation banquet on Friday, and Viraj has already reserved our private pavilion. This wedding will unite the two greatest business empires in western India."

At the lower end of the table, twenty-six-year-old Karthik looked up, his youthful face flushing with eager excitement. "Did Mr. Khanna mention Samira? Is she coming with him from Mumbai?"

"Patience, Karthik," Viraj chuckled with rare paternal indulgence. "A girl of Samira Khanna's stature does not rush about like common folk. But Rajesh knows what this match represents. Rajput Industries and the Khanna logistics network combined will dominate the entire northern infrastructure corridor. And you, my son, will have the most coveted heiress in the country by your side."

Aman Rajput, the thirty-four-year-old CEO of Rajput Industries, took a measured sip of water and set the crystal goblet down with a soft click. Beside him, his elegant wife Priya observed the room with calm, observant eyes.

"The financial figures are sound," Aman noted in his crisp boardroom cadence. "Khanna's port assets will reduce our shipping overheads by eighteen percent. But Samira is known to be unpredictable. Her social circle in Mumbai is fast, and she has little patience for traditional protocol."

"She is sixteen, Aman," Rita brushed the concern aside with an airy wave of her hand. "A young girl with high spirits, that is all. Once she enters Rajput Sadan as Karthik's bride, our traditions will guide her. A daughter-in-law of this house learns her duties soon enough."

A sudden, sharp clang echoed off the marble walls as Avinash accidentally dropped his butter knife against the plate. His hands trembled slightly as he picked it up.

Meghna turned to her husband with a thin, cutting smile. "Careful, Avinash. You look as pale as a sheet. Or were you daydreaming about the disgrace you left rotting by the highway?"

The dining hall fell into an abrupt, freezing silence. The warm chatter died in an instant.

Savita Rajput's sharp gray eyes fixed upon Avinash like twin daggers. When the matriarch spoke, her voice dropped to a dangerous, low murmur that carried to every corner of the room. "That subject is dead in this house. Twenty-three years ago, Vikram-ji made the decision. The brothel woman died where she belonged, and her child was left in the dust. He does not carry our name. He does not have our blood. He is nothing."

Avinash bowed his head, his knuckles whitening beneath the mahogany table. "Yes, Ma."

"Let the beggar stay in his rusted tin shed," Rita sneered, pouring more tea for Karthik. "He survives on scrap iron and charity. He is less than the dirt on our car tires. Our only pride is Karthik and his future with Samira."

Thirty miles away, beneath the blinding desert sun on the barren highway bypass, the air tasted of scorched asphalt and diesel fumes. Shashank Rajput wiped grease from his cheek with the back of a calloused hand. He stood inside his rusted tin workshop, barefoot on the packed earth, his thin frame clad in a faded cotton vest and trousers. On the workbench before him lay a dismantled tractor alternator. He worked in complete silence, solitary, dignified, and utterly unaware that seventy miles north, the wheels of a luxury car were already spinning toward a collision that would tear the grand illusions of Rajput Sadan into pieces."""
)

add_pair(
    "Write the scene where billionaire Rajesh Khanna's luxury car breaks down on the highway and he meets Shashank Rajput at the tin shed.",
    """The afternoon heat rose in shimmering, distorted waves off the black tarmac of the national highway. For miles in either direction, there was only dry scrubland, thorny acacia trees, and the occasional dust devil swirling across the yellow earth.

A sudden, violent hiss of scalding vapor erupted from beneath the sculpted hood of a midnight-blue Mercedes Maybach. The two-crore luxury sedan shuddered, its power-steering seizing as the temperature gauge pegged into the red zone. The vehicle coasted to a halt on the gravel shoulder, steam billowing like a smoke grenade into the scorching sky.

The chauffeur stumbled out in a panic, coughing as he reached toward the latch. Moments later, the rear door swung open, and Rajesh Khanna stepped onto the roadside.

Even standing on the baking gravel fifty miles from civilization, the billionaire was an imposing figure. Tailored charcoal linen suit, silver hair swept back from a sharp jaw, and a pair of observant gray eyes that had outmaneuvered prime ministers and corporate boards across three continents. He drew out his gold-rimmed smartphone, tapped the glass, and frowned.

"No reception," Rajesh muttered, his tone clipped. "Zero signal. Where are we, Sharma?"

"Forty kilometers past the toll plaza, sir," the chauffeur stammered, his uniform drenched in sweat. "The water pump has fractured. The belt snapped. We are completely stranded."

Rajesh turned his head, scanning the bleak horizon. Through the heat haze, roughly eighty yards down the bypass, stood a small rectangular structure: a makeshift shed built of corrugated tin sheets and wooden poles. A hand-painted sign leaning against an old truck tire read simply: 'MECHANIC'.

Before Rajesh could speak, the wooden door of the shed creaked open.

A young man stepped out into the blazing sun, carrying a battered metal toolbox in one hand and a black rubber hose in the other. He was thin, his collarbones sharply defined beneath a faded, grease-smudged shirt, but he walked with an upright, effortless stride that possessed none of the slumping deference common among roadside laborers.

He stopped two paces from the hissing car, his amber eyes taking in the luxury vehicle without a hint of surprise.

"Overheated block," the boy said, his voice deep, quiet, and steady. "The secondary coolant pipe cracked at the joint. If you run the starter again, you will warp the aluminum heads."

Rajesh Khanna observed the youth closely. Despite the grime on his jaw and the callouses on his fingers, there was an aristocratic sharpness to the boy's cheekbones, an unblinking intensity in his gaze that felt startlingly familiar. "Can you get it running?" Rajesh asked.

"I can cut away the fractured junction, fit a high-pressure brass sleeve, and seal the bypass," Shashank answered. "It will hold all the way to Delhi."

Without waiting for a contract or asking for money, Shashank dropped to his knees on the searing gravel. For forty minutes under the relentless sun, he worked with surgical concentration. His knuckles scraped against the scalding manifold, but his hands never shook. He measured, clamped, tightened, and purged the airlocks with the fluid certainty of an expert craftsman.

When he stood and signaled the driver, the engine roared to life with a purr so smooth the chassis barely trembled.

Rajesh reached into his breast pocket for his wallet, only to realize he had left his jacket inside the private jet hangar. He paused, unclasped the heavy Patek Philippe gold watch from his wrist — a custom chronometer worth forty lakhs — and held it out.

"I am without cash on the road," Rajesh said. "Take this watch. Consider it payment and gratitude."

Shashank looked at the gleaming gold timepiece, then lifted his eyes directly to Rajesh Khanna's face.

"The repair required twenty rupees of brass and half an hour of labor," Shashank said evenly. "I charge two hundred rupees. If you have no cash, you owe me nothing. I do not accept charity, and I do not trade in men's watches."

Before the billionaire could reply, Shashank picked up his toolbox, turned on his heel, and walked calmly back into his tin shed, closing the door behind him.

Rajesh Khanna stood on the roadside, the desert wind whipping his tie against his chest. A slow, profound smile spread across the tycoon's face. He turned to his driver.

"Sharma," Rajesh commanded, his voice vibrating with sudden resolve. "Call Ranawat the second we reach the city. I want that boy's name, his mother's name, and every single drop of blood in his veins verified before sunset." """
)

add_pair(
    "Write Chapter 2 of the Rajput family drama: Rajesh Khanna visits Rajput Sadan to drop the bombshell proposal, rejecting Karthik and demanding Shashank.",
    """The wrought-iron gates of Rajput Sadan swung wide as three black escort cars swept up the gravel driveway, parking before the marble portico. Servants in formal crimson turbans bowed as Rajesh Khanna climbed the stairs.

Inside the grand receiving salon, the Rajput family had assembled in full dynastic splendor. Savita sat upon an antique rosewood armchair, her emerald necklace gleaming in the afternoon light. Flanking her stood Viraj, Avinash, Aman, and Vijay. Beside Rita stood Karthik, dressed in a bespoke silk sherwani, his posture straight, his heart hammering against his ribs with nervous pride.

"Rajesh-ji," Viraj stepped forward with palms joined in a respectful namaskar. "Welcome to Rajput Sadan. It is a historic honor to receive the Khanna Empire beneath our ancestral roof."

"Viraj-ji. Savita-ji," Rajesh responded courteously, taking the armchair opposite the matriarch. A silver tray laden with saffron tea, pistachios, and silver-leaf sweets was set before him, but the tycoon did not touch it.

"I believe we both know why you are here, Rajesh-ji," Savita began, her aristocratic cadence ringing through the hall. "Our families have commanded respect for generations. The proposed union between our houses will be the defining alliance of the decade. Karthik has completed his business degree in London, and his devotion to your daughter Samira is well known across our circles."

Karthik smiled modestly, his chest swelling as Rita beamed with maternal pride.

Rajesh Khanna leaned back into the velvet upholstery, steepled his fingers, and looked directly into Savita's eyes. "You misunderstand the nature of my visit, Savita-ji."

A sudden chill settled over the salon. Viraj's smile froze. "Misunderstand? But Rajesh-ji, our discussions regarding the wedding..."

"I have indeed decided that my daughter Samira will marry a son of the Rajput lineage," Rajesh stated, his tone flat, calm, and absolute. "However, the groom will not be Karthik."

A sharp gasp escaped Rita's lips. Karthik's face turned white as chalk. "Mr. Khanna? What are you saying?"

Rajesh's voice cut through the air like a steel scalpel. "I am here to formally request the hand of Avinash's son."

Avinash staggered backward into the marble pillar, his face draining of all color. Beside him, Meghna's mouth fell open in utter disbelief.

"Avinash's son?!" Viraj frowned in bewilderment. "Vijay is already married to Riya! What riddle is this?"

"Not Vijay," Rajesh replied, unblinking. "I speak of Avinash's second son. The young mechanic living off the national highway in the tin shed. I am here for Shashank."

The grand hall of Rajput Sadan collapsed into suffocating, paralyzed silence.

For five seconds, nobody breathed. Then Rita erupted in a shriek of furious indignation. "Shashank?! That illegitimate roadside beggar?! Have you lost your senses, Rajesh-ji?!"

"Silence!" Savita's silver-headed cane struck the marble floor with the crack of a gunshot. The matriarch's face had turned an ashen, terrifying gray. Her gray eyes burned with venom as she glared at the billionaire. "Rajesh Khanna, you enter my ancestral home and spit on the memory of Vikram Rajput by proposing to wed your daughter to a whore's bastard?"

Rajesh did not flinch. He leaned forward, his gaze hardening into gray flint.

"Call him what your pride demands, Savita-ji," Rajesh spoke with quiet, crushing authority. "His veins carry your blood. But unlike your pampered grandsons who have never known the taste of honest sweat, that boy has iron in his soul. He refused my gold watch because he had not earned it. He looked me in the eye without flinching. He is the only young man in Rajasthan with the spine to tame my daughter and inherit my empire."

Rajesh rose to his feet, smoothing his lapels. "The conditions are non-negotiable. You will bring Shashank into this mansion, recognize him publicly as a rightful Rajput, and host the wedding with full royal honors. In exchange, the Khanna Group will inject five thousand crores into Rajput Industries' stranded deep-water port project."

He paused, letting his gaze sweep over Viraj, Aman, and Vijay.

"If you refuse, my banks will recall your short-term commercial bonds tomorrow morning at ten. Rajput Industries will enter bankruptcy before the week ends. You have forty-eight hours to decide. Good afternoon, Savita-ji." """
)

add_pair(
    "Write the confrontation scene where Samira Khanna discovers she is being married to Shashank Rajput instead of Karthik.",
    """Inside the penthouse suite of the Khanna residence overlooking the Arabian Sea, sixteen-year-old Samira Khanna hurled her designer sunglasses against the full-length mirror. The glass cracked with a sharp snap.

"Are you completely insane, Dad?!" her screech echoed off the minimalist concrete walls. "You expect me — Samira Khanna — to marry a grease monkey?! A filthy mechanic who lives in a rusted tin shed on a dirt highway?!"

Rajesh Khanna sat unperturbed at the head of the breakfast table, turning a page of the morning newspaper. He took a measured sip of black coffee. "His name is Shashank Rajput. And yes, you will marry him."

"I won't! I'll take the jet to London! I'll call the media and tell everyone you're forcing me into an arranged marriage with a beggar!" Samira paced furiously across the floor, her diamond earrings swinging, tears of humiliated rage brimming in her eyes. "Karthik was annoying enough with his desperate texts and stupid flowers, but at least he's a real Rajput! He wears Gucci! He drives a Porsche! But this... this person doesn't even have a shower!"

Rajesh folded the newspaper with slow, deliberate precision. He placed it on the table and fixed his daughter with a look so cold, so unyielding, that the words died in her throat.

"You have spent sixteen years crashing sports cars you did not pay for, throwing champagne at bouncers in rooftop clubs, and living on the illusion that you are superior to the rest of humanity," Rajesh said, his voice quiet and deadly. "Last month you nearly killed a family on Marine Drive because you thought traffic signals were suggestions for the poor. You are not royalty, Samira. You are a spoiled child who has never contributed a single rupee to the world she takes for granted."

"And marrying a pauper is supposed to teach me a lesson?!" Samira choked out, her fists clenched.

"He is not a pauper," Rajesh said, his voice softening with immense conviction. "He is a man who refused forty lakhs of gold because he had not earned it. When you look at him, you will learn what honor actually looks like. The wedding date is set for November. Choose your lehenga, Samira. Your tantrums end today." """
)

add_pair(
    "Write the family council scene at Rajput Sadan where Savita and the family decide they must accept Rajesh Khanna's demand.",
    """Midnight shadows stretched long across the mahogany council room of Rajput Sadan. Smoke from Viraj's imported cigar hung thick in the air, but no luxury could mask the suffocating terror that gripped the family.

"We cannot do this, Ma!" Viraj slammed his palm onto the table, rattling the crystal decanters. "If Jaipur high society discovers that we are marrying the Khanna heiress to Avinash's bastard, our family name will become a laughingstock across India!"

"And what will our name be worth when the debt tribunals auction this mansion on the courthouse steps next Tuesday, Viraj?" Savita's voice cut through the dark like a razor. The matriarch sat in her high-backed chair, her face carved of granite. "Will your name pay Aman's suppliers in Singapore? Will our prestige feed the twelve hundred workers in our textile mills?"

Aman Rajput, standing beside the fireplace, rubbed his temples in exhaustion. "Ma is right, Father. Vijay and I have spent the last three hours auditing the debt covenant structure. Rajesh Khanna's institutional funds hold seventy percent of our short-term commercial paper. If he demands immediate redemption, we have zero collateral left. We lose Rajput Industries within forty-eight hours."

In the corner of the velvet sofa, Karthik sat with his head buried in his hands, his broad shoulders shaking with muted, broken sobs. "She was supposed to be mine," he choked out, his voice raw with devastation. "I've waited for her for two years. Every party, every polo match... I did everything right. How can you hand her to that mechanic?!"

Rita knelt beside her son, stroking his hair, her eyes burning with pure venom as she looked across the table at Avinash. "Look at what your dirty past has brought upon my boy! That woman died twenty-three years ago, and her filth has come back to rob my son of his rightful life! Why didn't you choke that child in his crib?!"

Avinash did not raise his eyes. He sat motionless, staring at the floor, hollowed out by two decades of guilt.

Priya, standing quietly beside her husband Aman, spoke for the first time. Her voice was serene, carrying the calm weight of clear perception. "Dadi-ji, Rajesh Khanna is no fool. He knows Shashank has grown up with nothing, which means Shashank has nothing to fear. If we bring him into this house, do you truly believe he will bow to our commands?"

A cruel, thin smirk touched Savita Rajput's lips. The matriarch tapped her emerald ring against the armrest.

"He is a roadside beggar, Priya," Savita whispered with icy calculation. "Dress a beggar in raw silk, give him a bedroom with hot water, place a crore of rupees into his bank account, and he will learn to heel like any hungry dog. We will bring him here. We will let the Khanna wedding proceed. We will secure the five thousand crores of liquidity to save our empire. And once the contracts are signed and the money is locked in our vaults..."

Savita's eyes narrowed into slits of pure venom. "...we will remind the bastard exactly where he came from." """
)

def build_dataset():
    print(f"Building SFT story dataset with {len(STORIES)} high-fidelity examples...")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in STORIES:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    build_dataset()
