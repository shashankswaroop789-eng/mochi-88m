"""
🍡 Build Multi-Scene Continuity SFT Dataset (Scenes 1 through 6)
Teaches Mochi-Novel-300M both standalone chapter generation AND multi-chapter continuity:
- Scene 1: Desert Breakdown at the Tin Shed (Shashank & Rajesh / Shashank & Samira)
- Scene 2: The Investigation at Rambagh Palace (Ranawat & Rajesh discovering Shashank's lineage)
- Scene 3: The Bombshell at Rajput Sadan (Rajesh demanding Shashank instead of Karthik)
- Scene 4: Samira's Penthouse Confrontation with Rajesh
- Scene 5: Avinash's Midnight Visit to the Tin Shed
- Scene 6: Savita Confronting Karthik at Rajput Sadan

Trained on:
1. Direct Prompts: ### Prompt:\n{premise}\n\n### Story:\n
2. Instruction Continuations: ### Prompt:\nPrevious scene ending:\n{tail}\n\nNext scene:\n{instruction}\n\n### Story:\n
3. Natural Continuations: ### Prompt:\nContinue the story naturally from this scene:\n{tail}\n\n### Story:\n
Supports multiple tail slice variations (clean paragraph, 350-char slice, 250-char slice).
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
OUTPUT_PT = os.path.join(DATA_DIR, "multiscene_novel_sft_tokens.pt")
OUTPUT_JSONL = os.path.join(DATA_DIR, "multiscene_novel_sft_samples.jsonl")

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)
BOS = 1
EOS = 2
PAD = 0
SEQ_LEN = 1536

print("=" * 80)
print(" 🍡 BUILDING MULTI-SCENE NOVEL CONTINUITY DATASET")
print("=" * 80)

# ==============================================================================
# CANONICAL RAJPUT DYNASTY SCENES (Scenes 1 to 6)
# ==============================================================================

SCENE_1_SHASHANK_RAJESH = """The afternoon heat rose in shimmering waves off the cracked black asphalt of the national highway. Miles from the nearest town, where the scrub thorny bushes gave way to empty sand and rusted hoardings, stood a lone tin shed. Its corrugated metal roof groaned beneath the relentless sun, rattling whenever a heavy transport lorry thundered past in a cloud of diesel exhaust.

Inside the stifling shade of the workshop, Shashank wiped a streak of engine oil from his brow with the back of a calloused hand. He was twenty-three, thin-framed and lean as a desert wolf, his collarbones visible beneath a faded cotton shirt that had seen too many monsoon cycles. On the workbench lay an old motorcycle carburetor, dismantled into dozens of tiny brass jets and springs. Shashank worked with the methodical, patient focus of a surgeon, his long fingers moving with effortless familiarity.

A sudden hiss of steam and the violent shudder of a dying engine broke the highway's silence.

Fifty yards down the asphalt, a midnight-blue luxury sedan — a Maybach worth more than the entire village behind the ridge — coasted to a halt on the gravel shoulder, its radiator spewing white plumes into the blistering sky. The driver's door swung open, and a chauffeur in crisp white livery stepped out in panic, waving frantic hands at the hissing grille. Moments later, the rear door clicked open.

Rajesh Khanna stepped out into the blinding glare. Even stranded in the desert heat, the tycoon looked formidable: tailored charcoal linen suit, silver hair immaculately combed, and an aura of supreme authority that commanded boardrooms across Mumbai and London. He pulled out a gold-plated smartphone, scowled at the screen, and muttered in frustration. "No signal. Not even a single bar."

"Sir, the water pump has seized," the chauffeur stammered, wiping sweat from his forehead. "There is no service center for fifty kilometers."

Rajesh Khanna turned his gaze toward the lone tin shed. Through the heat haze, he saw the silhouette of a young mechanic stepping out of the shadows, carrying a battered metal toolbox in one hand and a gallon jug of water in the other.

Shashank approached without haste and without deference. His dark eyes, strikingly amber beneath the grime, took in the stalled luxury vehicle without a flicker of awe.

"Overheated radiator," Shashank said, his voice quiet, low, and steady. "The secondary coolant hose ruptured near the manifold."

Rajesh Khanna studied the young man. Despite his gaunt frame and faded clothes, there was something unsettling about the boy's bearing — a straightness in the spine, an unblinking gaze that refused to bow. "Can you fix it?" Rajesh Khanna asked crisply.

"I can bypass the ruptured line with a reinforced rubber sleeve and clamp it," Shashank answered. "It will hold until you reach the city."

Without waiting for permission or negotiating a price, Shashank knelt onto the scorching gravel. For forty minutes, under the blistering sun, the boy worked with quiet, surgical precision. His knuckles scraped against the searing engine block, but he did not wince. He replaced the cracked hose, tightened the clamps with steady turns of a wrench, purged the airlocks, and topped off the coolant reservoir with cool water.

When he stood up and signaled the chauffeur to turn the ignition, the twin-turbo engine roared to life with a silky, flawless purr.

Rajesh Khanna stepped forward, reaching into his breast pocket for his wallet. Finding it empty, he frowned, suddenly remembering he had left his briefcase in the private jet. He unclasped the heavy Patek Philippe timepiece from his wrist — an heirloom valued at over fifty lakhs — and held it out toward the boy.

"I left my cash at the hangar," Rajesh Khanna said. "Take this watch. Consider it a deposit on your work."

Shashank looked at the gleaming gold timepiece, then met the billionaire's gaze squarely. "The repair took fifty rupees of rubber and thirty minutes of labor. I charge two hundred rupees. If you do not have cash, you owe me nothing. I do not take charity, and I do not trade in gold."

Before Rajesh Khanna could utter another word, Shashank picked up his toolbox, turned his back on the luxury car, and walked calmly back into his tin shed.

Rajesh Khanna stood rooted to the gravel, staring at the boy's retreating back. A slow, calculating smile touched the tycoon's lips. He turned to his driver. "Call my personal investigator the moment we reach signal range. I want that boy's name, his parentage, and every breath he has taken since the day he was born.\""""

SCENE_1_SHASHANK_SAMIRA = """The afternoon heat rose in shimmering, distorted waves off the black tarmac of the national highway. For miles in either direction across the desert expanse, there was only dry scrubland, thorny acacia trees, and the occasional dust devil swirling across the yellow earth.

A sudden, violent hiss of scalding vapor erupted from beneath the sculpted hood of a midnight-red luxury sports coupe. The two-crore German engine shuddered violently, its power-steering seizing as warning alarms chimed frantically across the digital dashboard. The vehicle coasted to a halt on the loose gravel shoulder, dense plumes of white radiator steam billowing into the blistering sky.

The driver's door swung open with a sharp metallic latch, and Samira stepped onto the baking roadside. She was sixteen, dressed in designer sunglasses and an embroidered linen tunic, her expression a mixture of profound shock and mounting frustration. She stared at the smoking grille in disbelief, tapping her smartphone frantically against her palm.

"Zero signal," Samira muttered, her voice trembling with irritation. "Not a single bar. In the middle of nowhere."

She shielded her eyes, scanning the bleak horizon. Roughly eighty yards past the highway shoulder stood a solitary, weathered structure: a workshop built of rusted corrugated tin sheets and rough timber poles. A hand-painted wooden board leaning against an old truck tire read simply: 'MECHANIC'.

Before she could walk toward it, the tin door creaked open.

A young man stepped out into the blazing sun, carrying a battered metal toolbox in one hand and a black rubber hose in the other. He was twenty-three, lean and broad-shouldered beneath a faded cotton shirt smudged with grease and road dust. He walked with an upright, effortless stride that possessed none of the slumping deference common among roadside laborers.

He stopped two paces from the steaming sports car, his dark amber eyes taking in the stalled vehicle with calm, observant detachment.

"Overheated block," Shashank said, his voice deep, quiet, and steady. "The high-pressure coolant junction fractured at the manifold. If you had tried to restart it, you would have cracked the aluminum cylinder head."

Samira looked at him, startled by his voice and the unwavering composure in his eyes. Despite the grime on his forearms, there was a quiet nobility in his features, an aristocratic sharpness in his jawline that seemed completely out of place in a desert tin shed.

"Can you fix it?" Samira asked, crossing her arms and trying to sound authoritative. "I am on my way to Jaipur. My family has critical business in the city, and I have to be at the heritage hotel before evening. Name your price."

Shashank did not blink at her demanding tone. His expression remained utterly impassive. "I charge two hundred rupees for roadside labor, plus the cost of materials. If you want the car to run, step back from the steam."

Without waiting for her reply, Shashank dropped to his knees on the searing gravel. For forty minutes under the relentless desert sun, he worked with surgical concentration. His knuckles scraped against the scalding engine block, but his hands never shook. He measured, clamped, replaced the fractured junction with a reinforced brass sleeve, and purged the coolant airlocks with fluid, practiced precision.

Samira stood in the thin shade of an acacia tree, watching him in silence. Accustomed to fawning attendants and corporate yes-men who jumped at her family's shadow, she had never seen someone work with such quiet, unyielding self-respect.

When Shashank stood and signaled her to turn the ignition, the high-performance engine roared to life with a silky, flawless purr.

Samira reached into her leather handbag, pulled out a thick wad of two-thousand-rupee notes, and held them out. "Take this. All of it. You saved me hours of waiting in this heat."

Shashank looked at the bundle of cash, then met her gaze squarely. "The replacement brass cost fifty rupees. My labor is two hundred. Give me two hundred and fifty rupees, or take your car and leave. I earn my bread through work, not charity."

Samira froze, her hand hovering in mid-air. For the first time in her life, someone had refused her money. She slowly slipped the notes back into her purse, retrieved three one-hundred-rupee bills, and handed them over. "Keep the change as a tip, then."

"I do not take tips," Shashank replied evenly, handing her fifty rupees from his pocket.

He picked up his toolbox, turned on his heel, and walked calmly back into his tin shed, closing the door behind him.

Samira stood by the open car door, the hot desert wind tugging at her hair. A slow, bemused smile broke across her face as she looked toward the rusted shed. "Who on earth are you?" she whispered to herself, before sliding behind the wheel and driving toward Jaipur."""

SCENE_2_INVESTIGATION = """In the presidential suite of the Rambagh Palace Hotel in Jaipur, Rajesh Khanna stood before the floor-to-ceiling glass windows, holding a crystal glass of single malt scotch as evening rain began to drum against the glass. Behind him, standing respectfully near an ornate rosewood writing desk, was Inspector Ranawat, a retired senior superintendent of police who had headed Khanna's corporate intelligence network for over a decade.

On the polished desktop lay a thick manila folder stamped with a crimson seal: 'CONFIDENTIAL — EYES ONLY'.

"Report," Rajesh commanded without turning around, his reflection sharp in the dark glass.

Ranawat adjusted his spectacles and opened the file. "The young mechanic's name is Shashank. He is twenty-three years old. For the past four years, he has lived entirely alone in that corrugated tin shed off the national highway bypass, surviving on whatever roadside repairs pass by. Prior to that, he grew up in an unaccredited charity orphanage outside Ajmer."

"And his bloodline?" Rajesh asked, turning slowly, his gray eyes narrowing into razor slits. "A boy does not hold his spine like an iron pillar without lineage. Who were his parents?"

Ranawat drew a steadying breath before answering. "His biological mother was a woman named Mina — an unrecorded courtesan in the old quarter of Jaipur. She died of complications within three hours of childbirth in a municipal ward in December of 2003."

"And the father?"

"Avinash Rajput," Ranawat replied in a low, measured voice. "The second-born son of the late Vikram Rajput and Savita Rajput. The boy in that tin shed is a true-born grandson of Rajput Sadan."

Rajesh Khanna took a slow, contemplative sip of his scotch. A cold, appreciative smile touched the tycoon's lips. "A grandson of the Rajput dynasty. Living in a rusted tin shed on five hundred rupees a week."

"Authentic blood, sir, but violently repudiated," Ranawat explained, turning a page in the dossier. "When Vikram Rajput discovered the birth twenty-three years ago, he paid off the midwives, silenced the clinic, and ordered the infant abandoned in the desert scrub. Savita Rajput considers the boy an unspeakable stain upon the family honor. They have never acknowledged his existence. In fact, seven years ago, Viraj Rajput used the local police to threaten the boy, promising that if he ever crossed the city bypass into Jaipur, he would be crushed under a lorry."

Ranawat produced a faded, yellowed photograph from the folder and placed it under the desk lamp. "This was taken twenty-three years ago outside the old city hospital. That is Avinash Rajput, and in his arms is the newborn boy."

Rajesh walked over to the desk and picked up the photograph. He studied the infant's face, then recalled the young man who had knelt on the searing asphalt to fix his Maybach. The jawline, the broad shoulders, and above all, the unblinking amber eyes — they were unmistakable.

"And yet," Rajesh murmured softly, setting the photograph down onto the silver tray with a deliberate click, "the boy works with his hands. He refused my fifty-lakh gold watch because he had not earned it. He looked me in the eye without a flicker of fear and without a hint of greed. Compare that to the pampered whelps dining on silver plates inside Rajput Sadan — soft young men drowning in family money, wearing pride they never bled for."

"Sir," Ranawat asked cautiously, closing the folder, "what do you intend to do? Savita Rajput and Viraj have already booked the banquet pavilion. They are expecting you to finalize the golden alliance between Samira and their favored grandson, Karthik. Rajput Industries owes over four hundred crores to merchant banks, and they are counting on your capital to survive."

Rajesh Khanna let out a low, rumbling chuckle that carried the crushing authority of a man who commanded billions across three continents.

"Karthik is a pampered lapdog," Rajesh stated flatly. "He would spend his life writing poetry and flattering Samira's tantrums while she drove him into bankruptcy and an early grave. My daughter needs an anchor, Ranawat, not a court jester. She needs a man who knows what hunger tastes like, whose bones were tempered in fire and iron."

Rajesh smoothed his lapels and buttoned his charcoal jacket. "Have the Maybach brought around to the portico tomorrow morning at eleven sharp. We are paying an official visit to Rajput Sadan. Savita Rajput believes she is about to sell me a prince. I am going to buy her outcast.\""""

SCENE_3_RAJPUT_SADAN_BOMBSHELL = """The heavy wrought-iron gates of Rajput Sadan swung wide as three black Mercedes Maybachs glided up the curving gravel driveway. Servants in royal crimson turbans bowed deeply as Rajesh Khanna ascended the marble steps of the ancestral mansion.

Inside the grand receiving salon, the Rajput family had assembled in full dynastic splendor beneath shimmering crystal chandeliers. Savita Rajput sat upon a high-backed carved rosewood chair, her emerald necklace gleaming against her ivory silk saree. Flanking her stood Viraj, Avinash, Aman, and Vijay. Beside Rita stood Karthik, dressed in a bespoke raw-silk sherwani, his youthful face flushed with eager excitement.

"Rajesh-ji," Viraj stepped forward with palms joined in a respectful namaskar. "Welcome to Rajput Sadan. It is a historic honor to receive the Khanna Empire beneath our ancestral roof."

"Viraj-ji. Savita-ji," Rajesh greeted them with measured courtesy, taking the armchair opposite the matriarch. A silver tray laden with saffron tea, pistachios, and silver-leaf sweets was set before him, but the tycoon did not touch it.

"I believe we both know the joyful reason for your visit, Rajesh-ji," Savita began, her aristocratic cadence ringing through the marble hall. "Our families have commanded respect for generations. The proposed union between our houses will be the defining alliance of the decade. Karthik has completed his business degree in London, and his devotion to your daughter Samira is well known across our circles."

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

"If you refuse, my banks will recall your short-term commercial bonds tomorrow morning at ten. Rajput Industries will enter bankruptcy before the week ends. You have forty-eight hours to decide. Good afternoon, Savita-ji.\""""

SCENE_4_SAMIRA_CONFRONTATION = """Inside the penthouse suite of the Khanna residence overlooking the Arabian Sea, sixteen-year-old Samira Khanna hurled her designer sunglasses against the full-length mirror. The glass cracked with a sharp, explosive snap.

"Are you completely insane, Dad?!" her screech echoed off the minimalist concrete walls and vaulted ceilings. "You expect me — Samira Khanna — to marry a grease monkey?! A filthy mechanic who lives in a rusted tin shed on a dirt highway?!"

Rajesh Khanna sat unperturbed at the head of the long teak dining table, turning a page of the morning financial journal. Outside, monsoon squalls whipped against the reinforced panoramic windows, but inside, the air was as still as a tomb. He took a measured sip of black coffee before looking up.

"His name is Shashank Rajput," Rajesh said, his voice quiet, calm, and level. "And yes, you will marry him."

"I won't! I'll take the jet to London! I'll call the media and tell every news outlet in Mumbai that you're forcing me into an arranged marriage with a roadside pauper!" Samira paced furiously across the polished Italian marble, her diamond earrings swinging, tears of humiliated rage brimming in her eyes. "Karthik was annoying enough with his desperate texts and stupid flowers, but at least he's a real Rajput! He wears bespoke suits! He went to Harrow and Oxford! He drives a Porsche! But this boy... this mechanic doesn't even have running water!"

Rajesh folded the newspaper with slow, deliberate precision. He placed it squarely on the table, set his cup down, and fixed his daughter with a look so cold, so penetrating, that the fury caught in her throat.

"You have spent sixteen years crashing sports cars you did not pay for, throwing champagne at bouncers in rooftop clubs, and living on the delusion that your surname makes you superior to the rest of humanity," Rajesh spoke, every word cutting through the room like a razor. "Last month you nearly killed a family on Marine Drive because you thought traffic laws were suggestions for the poor. You are not royalty, Samira. You are a spoiled child who has never worked an honest hour in her life."

"And marrying a pauper in a desert is supposed to fix me?!" Samira choked out, her hands trembling in frustration.

"He is not a pauper," Rajesh said, his voice deepening with profound conviction. "He is a man who refused fifty lakhs of gold because he had not earned it. When your car broke down in the blazing sun, while you stood crying on the shoulder, he fixed your engine with his bare hands. When you look at him, Samira, you will learn what honor and dignity actually look like. The wedding will take place before the month ends. Choose your lehenga. Your tantrums end today."

Samira stood frozen beneath the chandelier, her breath coming in ragged gasps as her father rose and walked toward his private study, leaving her alone with the shattering reality of his decision."""

SCENE_5_AVINASH_VISIT = """The twilight sky over the Thar desert bled from amber into deep violet. A cool desert wind began to stir the sand, whistling through the gaps in the corrugated tin sheets of the roadside workshop. A solitary kerosene lantern hung from a wooden post, casting long, swaying shadows across greasy wrenches and dismantled motorcycle parts.

Shashank sat on an inverted oil drum, cleaning the grease from his knuckles with a rag soaked in kerosene. The highway had grown quiet, the distant drone of transport trucks fading into the vast desert night.

Footsteps sounded on the loose gravel outside.

Shashank did not look up immediately. He recognized the tentative, heavy cadence of someone who was not accustomed to walking on unpaved ground. When he finally lifted his eyes, a tall man stood in the open doorway of the shed.

Avinash Rajput wore an expensive woolen shawl over a crisp white kurta, but his shoulders were slumped, and his graying hair was disheveled by the night wind. At fifty, his face was lined with deep furrows of guilt and exhaustion.

For nearly a minute, neither man spoke. Only the hiss of the kerosene flame broke the stillness.

"You look like her," Avinash said quietly, his voice trembling slightly. "Your mother had that same unyielding look in her eyes whenever she stood her ground."

Shashank tossed the rag onto the workbench. He stood up slowly, crossing his arms over his chest. His expression was neither angry nor welcoming; it was as smooth and unreadable as desert stone.

"You took the bypass road," Shashank said, his tone flat and even. "The ancestral mansion is thirty miles north. Did you lose your way, Mr. Rajput?"

The formal address struck Avinash like a physical blow. He stepped forward into the lantern light, extending a hand before letting it fall helplessly to his side.

"Shashank... I know I have no right to come here," Avinash stammered, his eyes glistening with unshed tears. "For twenty-three years, Savita Ma forbade anyone from mentioning your existence. She threatened to disinherit Vijay and cast us all out. I was weak. I was a coward. But every single day, I knew you were here... working in this heat..."

"And yet you never came," Shashank interrupted quietly. There was no bitterness in his voice, only cold, unvarnished truth. "When the monsoon flooded this shed three years ago and washed away my tools, you did not come. When I was fifteen and went three days without food because no trucks broke down, you did not come. Why are you here tonight?"

Avinash bowed his head, his hands trembling. "Rajesh Khanna came to Rajput Sadan yesterday. He knows who you are. He told the entire family that he will only agree to the business alliance if Samira marries you, not Karthik. Savita Ma is planning to bring you back... to use you to save the family fortune from a five-thousand-crore default."

A faint, mocking smile touched Shashank's lips. He looked around his humble tin shed, at the rusted iron and the tools he had earned with his own sweat.

"Tell Savita Rajput that her mansion has no walls high enough to buy my soul," Shashank said softly. "I owe nothing to the Rajput name, and I want nothing from it. Go home, Avinash. Before someone sees your luxury car parked by a beggar's shed."

Avinash lingered in the doorway, but Shashank turned his back and picked up his wrench, the steady clinking of metal against metal signaling that the conversation was over."""

SCENE_6_SAVITA_KARTHIK = """Inside the grand high-ceilinged drawing room of Rajput Sadan, afternoon shadows lengthened across antique Persian carpets. Heavy mahogany pillars supported arches carved from pink Rajasthani sandstone, their capitals adorned with family crests that had stood for three centuries.

At the center of the chamber sat Savita Rajput. At seventy-five, the matriarch commanded the room with the unyielding posture of an empress. Her silver hair was drawn back into an immaculate bun, secured by a gold pin, and a necklace of uncut emeralds rested against her ivory silk saree. In her right hand she held a silver-headed cane, though her spine remained as straight as an iron pillar.

Standing before her, looking pale and disheveled, was twenty-six-year-old Karthik. His bespoke Italian blazer was unbuttoned, his tie loosened, and his eyes carried the nervous exhaustion of a young man caught between his desires and the crushing weight of his lineage.

"You were seen leaving the Khanna residence in Mumbai at three in the morning, Karthik," Savita began, her voice dropping into a dangerous, low murmur that carried to every corner of the room. "And this morning, the financial dailies are reporting that the Khanna Empire has suspended negotiations on our joint port terminal."

Karthik swallowed hard, his hands clenching at his sides. "Dadi-ji, you don't understand. She won't even speak to me. She laughed in my face when I showed up at the gala. She called me a puppet dancing on family strings."

"And are you?" Savita's eyes narrowed into twin points of cold steel. "Did I spend twenty-six years educating you in London and grooming you to lead Rajput Industries so that a sixteen-year-old girl could strip away your spine? You are a Rajput, Karthik. We do not beg for alliances. We command them."

"Then why are we on the brink of bankruptcy?!" Karthik's voice cracked with raw desperation. "Viraj Uncle says we owe four hundred crores to the merchant banks. If Rajesh Khanna pulls the liquidity, we lose everything! Even this house!"

The room fell into an icy, suffocating silence.

Savita slowly rose from her chair, leaning on her cane with measured, deliberate dignity. She walked toward Karthik, stopping only inches from him. Her gaze did not waver.

"Rajput Sadan was built on blood and sacrifice, not bank loans," she said, her tone as cold as winter frost. "Generations of men have died to keep this crest untouched. If you believe your comfortable life was given to you for idle amusement, you are mistaken. You will dress yourself. You will apologize to Rajesh Khanna with the grace of a prince. And you will secure the alliance before this month ends. If you fail..."

She paused, letting the silence press heavily into Karthik's chest.

"If you fail, Karthik, I will find a son of this bloodline who possesses the courage you seem to lack. Do not test my resolve."

With a sharp tap of her cane against the marble floor, Savita turned and walked toward her private chambers, leaving Karthik trembling in the vast, echoing silence of the ancestral hall."""

# Master list of scenes
ALL_SCENES = [
    (1, "Scene 1: Breakdown (Shashank & Rajesh)", SCENE_1_SHASHANK_RAJESH),
    (1, "Scene 1: Breakdown (Shashank & Samira)", SCENE_1_SHASHANK_SAMIRA),
    (2, "Scene 2: Investigation & Revelation", SCENE_2_INVESTIGATION),
    (3, "Scene 3: Bombshell at Rajput Sadan", SCENE_3_RAJPUT_SADAN_BOMBSHELL),
    (4, "Scene 4: Samira Confrontation", SCENE_4_SAMIRA_CONFRONTATION),
    (5, "Scene 5: Avinash Midnight Visit", SCENE_5_AVINASH_VISIT),
    (6, "Scene 6: Savita & Karthik Confrontation", SCENE_6_SAVITA_KARTHIK)
]

dataset_pairs = []

# --- 1. DIRECT PROMPTS ---
PROMPTS_DIRECT = {
    1: [
        "Write a short story about Shashank, a proud young mechanic working in a highway tin shed, and Rajesh Khanna, a wealthy businessman whose luxury car breaks down on the desert road.",
        "Write a short story about Shashank, a proud young mechanic working in a highway tin shed, and Samira Khanna a daughter of wealthy businessman whose luxury car breaks down on the desert road.",
        "Write Scene 1: The desert breakdown where billionaire Rajesh Khanna meets the unyielding mechanic Shashank."
    ],
    2: [
        "Write Scene 2: Rajesh Khanna's personal investigator uncovers that Shashank is the outcast grandson of Savita Rajput, and Rajesh decides to visit the Rajput mansion.",
        "Write the scene where Rajesh Khanna's detective Ranawat presents the confidential dossier proving Shashank is the true-born son of Avinash Rajput and grandson of Savita Rajput.",
        "In the presidential suite of Rambagh Palace, Rajesh Khanna receives the investigative dossier revealing Shashank's royal Rajput bloodline."
    ],
    3: [
        "Write Scene 3: Rajesh Khanna visits Rajput Sadan to drop the bombshell proposal, rejecting Karthik and demanding Shashank Rajput.",
        "Write the scene at Rajput Sadan where billionaire Rajesh Khanna confronts Savita Rajput and demands Shashank as groom for Samira.",
        "Rajesh Khanna shocks the assembled Rajput family by demanding Shashank, the roadside mechanic, marry his daughter Samira."
    ],
    4: [
        "Write the confrontation scene where Samira Khanna discovers she is being married to Shashank Rajput instead of Karthik.",
        "Write Scene 4: Samira Khanna furiously confronts her father Rajesh Khanna about marrying a roadside mechanic.",
        "In their Mumbai penthouse, Samira argues with Rajesh Khanna over the arranged marriage to the mechanic Shashank."
    ],
    5: [
        "Write a short story about Avinash Rajput quietly visiting his estranged son Shashank at his dusty roadside workshop after years of bitter silence.",
        "Write Scene 5: Avinash secretly visits Shashank at the tin shed at night.",
        "Avinash Rajput confronts his long-abandoned son Shashank in the desert workshop."
    ],
    6: [
        "Write a short story about Savita Rajput, the proud family matriarch, confronting her grandson Karthik inside the ancestral mansion about his loyalty to the family name.",
        "Write Scene 6: Savita Rajput confronts Karthik in the study about family honor and the Khanna alliance.",
        "Savita Rajput strikes her cane and delivers an ultimatum to Karthik about saving the family name."
    ]
}

for ch_num, title, text in ALL_SCENES:
    prompts = PROMPTS_DIRECT.get(ch_num, [])
    for p in prompts:
        dataset_pairs.append({
            "prompt": f"### Prompt:\n{p}\n\n### Story:\n",
            "story": text.strip()
        })

# --- 2. MULTI-SCENE CONTINUATION PROMPTS ---
def get_tail_variants(scene_text):
    """Return multiple tail variations: clean last paragraph, 350-char slice, 250-char slice."""
    paras = [p.strip() for p in scene_text.strip().split("\n\n") if p.strip()]
    last_para = paras[-1]
    
    tails = [last_para]
    if len(scene_text) >= 350:
        tails.append(scene_text[-350:].strip())
    if len(scene_text) >= 250:
        tails.append(scene_text[-250:].strip())
    return tails

# Transition 1 -> 2:
s1_tails = get_tail_variants(SCENE_1_SHASHANK_RAJESH) + get_tail_variants(SCENE_1_SHASHANK_SAMIRA)
s2_instructions = [
    "Rajesh Khanna's personal investigator uncovers that Shashank is the outcast grandson of Savita Rajput, and Rajesh decides to visit the Rajput mansion.",
    "Write Scene 2: The investigation at Rambagh Palace where Ranawat reveals Shashank's Rajput lineage to Rajesh Khanna.",
    "Inspector Ranawat presents the confidential dossier on Shashank to Rajesh Khanna in the presidential suite.",
    "Show Rajesh Khanna discovering the truth about the mechanic Shashank from his detective Ranawat."
]
for tail in s1_tails:
    for inst in s2_instructions:
        dataset_pairs.append({
            "prompt": f"### Prompt:\nPrevious scene ending:\n{tail}\n\nNext scene:\n{inst}\n\n### Story:\n",
            "story": SCENE_2_INVESTIGATION.strip()
        })
    dataset_pairs.append({
        "prompt": f"### Prompt:\nContinue the story naturally from this scene:\n{tail}\n\n### Story:\n",
        "story": SCENE_2_INVESTIGATION.strip()
    })

# Transition 2 -> 3:
s2_tails = get_tail_variants(SCENE_2_INVESTIGATION)
s3_instructions = [
    "Rajesh Khanna visits Rajput Sadan to drop the bombshell proposal, rejecting Karthik and demanding Shashank Rajput.",
    "Write Scene 3: Rajesh Khanna arrives at Rajput Sadan and confronts Savita Rajput with the 5000-crore ultimatum.",
    "Rajesh Khanna shocks the assembled Rajput family by refusing Karthik and demanding Shashank as Samira's husband."
]
for tail in s2_tails:
    for inst in s3_instructions:
        dataset_pairs.append({
            "prompt": f"### Prompt:\nPrevious scene ending:\n{tail}\n\nNext scene:\n{inst}\n\n### Story:\n",
            "story": SCENE_3_RAJPUT_SADAN_BOMBSHELL.strip()
        })
    dataset_pairs.append({
        "prompt": f"### Prompt:\nContinue the story naturally from this scene:\n{tail}\n\n### Story:\n",
        "story": SCENE_3_RAJPUT_SADAN_BOMBSHELL.strip()
    })

# Transition 3 -> 4:
s3_tails = get_tail_variants(SCENE_3_RAJPUT_SADAN_BOMBSHELL)
s4_instructions = [
    "Samira Khanna furiously confronts her father Rajesh Khanna in their penthouse after learning she is being married to Shashank.",
    "Samira argues with her father about the arranged marriage to the mechanic."
]
for tail in s3_tails:
    for inst in s4_instructions:
        dataset_pairs.append({
            "prompt": f"### Prompt:\nPrevious scene ending:\n{tail}\n\nNext scene:\n{inst}\n\n### Story:\n",
            "story": SCENE_4_SAMIRA_CONFRONTATION.strip()
        })
    dataset_pairs.append({
        "prompt": f"### Prompt:\nContinue the story naturally from this scene:\n{tail}\n\n### Story:\n",
        "story": SCENE_4_SAMIRA_CONFRONTATION.strip()
    })

# Transition 4 -> 5:
s4_tails = get_tail_variants(SCENE_4_SAMIRA_CONFRONTATION)
s5_instructions = [
    "Avinash Rajput secretly visits Shashank at his roadside tin shed in the middle of the night.",
    "Avinash meets his estranged son Shashank at the tin shed after Rajesh Khanna's bombshell."
]
for tail in s4_tails:
    for inst in s5_instructions:
        dataset_pairs.append({
            "prompt": f"### Prompt:\nPrevious scene ending:\n{tail}\n\nNext scene:\n{inst}\n\n### Story:\n",
            "story": SCENE_5_AVINASH_VISIT.strip()
        })
    dataset_pairs.append({
        "prompt": f"### Prompt:\nContinue the story naturally from this scene:\n{tail}\n\n### Story:\n",
        "story": SCENE_5_AVINASH_VISIT.strip()
    })

# Transition 5 -> 6:
s5_tails = get_tail_variants(SCENE_5_AVINASH_VISIT)
s6_instructions = [
    "Savita Rajput severely reprimands Karthik in the ancestral drawing room about family honor and the Khanna alliance.",
    "Savita confronts Karthik about his failure to secure the Khanna alliance."
]
for tail in s5_tails:
    for inst in s6_instructions:
        dataset_pairs.append({
            "prompt": f"### Prompt:\nPrevious scene ending:\n{tail}\n\nNext scene:\n{inst}\n\n### Story:\n",
            "story": SCENE_6_SAVITA_KARTHIK.strip()
        })
    dataset_pairs.append({
        "prompt": f"### Prompt:\nContinue the story naturally from this scene:\n{tail}\n\n### Story:\n",
        "story": SCENE_6_SAVITA_KARTHIK.strip()
    })

def build_dataset():
    print(f"Total base narrative pairs: {len(dataset_pairs)}")

    # Multiply canon pairs 6x to ensure rock-solid alignment while keeping diverse distribution
    expanded_pairs = []
    for _ in range(6):
        expanded_pairs.extend(dataset_pairs)

    print(f"Total training pairs after expansion: {len(expanded_pairs)}")

    random.seed(42)
    random.shuffle(expanded_pairs)

    # Save preview jsonl
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in expanded_pairs[:10]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Tokenize and Pack
    packed_tensors = []
    for item in expanded_pairs:
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

    print(f"SUCCESS: Saved {len(packed_tensors)} multi-scene packed sequences to {OUTPUT_PT}!")
    print("=" * 80)

if __name__ == "__main__":
    build_dataset()
