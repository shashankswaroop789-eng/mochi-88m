"""
Generate Comprehensive Rajput Dynasty & Indian Family Saga Corpus
Creates an extensive, novel-grade corpus centered on the Rajput family,
dynastic corporate drama, emotional confrontations, and clean narrative formatting.
"""

import os
import sys
import random

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CORPUS = os.path.join(DATA_DIR, "rajput_dynasty_corpus.txt")

# Core character descriptions & traits
CHARACTERS = {
    "Savita": "Savita Rajput, the seventy-five-year-old matriarch of Rajput Sadan, sat upright in her carved teakwood armchair with the uncompromising posture of an empress.",
    "Viraj": "Viraj Rajput, the first-born son at fifty-four, carried the unyielding pride of the Rajput lineage in every heavy footstep.",
    "Rita": "Rita Rajput, Viraj's wife, lived for the polished social veneer of Jaipur's elite high society, measuring worth in lineage and diamonds.",
    "Avinash": "Avinash Rajput, the second son at fifty, carried a quiet, brooding shadow behind his eyes — the haunted guilt of a secret buried twenty-three years ago.",
    "Meghna": "Meghna Rajput, sharp-tongued and eternally vigilant, never let her husband Avinash forget the disgrace that lurked at the fringes of their name.",
    "Rachna": "Rachna Rajput, the divorced daughter of the house, observed the family hypocrisies with cynical amusement from the marble terrace.",
    "Aman": "Aman Rajput, the thirty-four-year-old CEO of Rajput Industries, commanded the boardroom with ruthless discipline and measured composure.",
    "Priya": "Priya Rajput, Aman's wife at twenty-seven, moved through the sprawling mansion with an understated grace that concealed a razor-sharp intellect.",
    "Vijay": "Vijay Rajput, thirty and CFO of Rajput Industries, balanced the family balance sheets with cold mathematical precision.",
    "Riya": "Riya Rajput, Vijay's wife and a pediatric surgeon at twenty-seven, maintained a calm distance from the venomous gossip of the family teas.",
    "Karthik": "Karthik Rajput, twenty-six and handsome, wore his heart on his sleeve, utterly consumed by his infatuation with the Khanna heiress.",
    "Shashank": "Shashank Rajput, twenty-three, worked solitary shifts in a rusted tin shed off the national highway, his knuckles grease-blackened and his eyes fiercely proud.",
    "Rajesh": "Rajesh Khanna, the billionaire patriarch of the Khanna Empire, was a man whose sharp gray eyes could dissect a man's character in thirty seconds.",
    "Samira": "Samira Khanna, sixteen and heiress to billions, lived in a whirlwind of designer clothes, rooftop parties, and unapologetic disdain for the stifling rules of her class."
}

# Rich scene templates and variations
SCENES = [
    # Scene 1: The tin shed and Rajesh Khanna's breakdown
    """The afternoon heat rose in shimmering waves off the cracked black asphalt of the Jaipur-Delhi highway. Miles from the nearest town, where the scrub thorny bushes gave way to empty sand and rusted hoardings, stood a lone tin shed. Its corrugated metal roof groaned beneath the relentless sun, rattling whenever a heavy transport lorry thundered past in a cloud of diesel exhaust.

Inside the stifling shade of the workshop, Shashank Rajput wiped a streak of engine oil from his brow with the back of a calloused hand. He was twenty-three, thin-framed and lean as a desert wolf, his collarbones visible beneath a faded cotton shirt that had seen too many monsoon cycles. On the workbench lay an old motorcycle carburetor, dismantled into dozens of tiny brass jets and springs. Shashank worked with the methodical, patient focus of a surgeon, his long fingers moving with effortless familiarity.

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

Rajesh Khanna stood rooted to the gravel, staring at the boy's retreating back. A slow, calculating smile touched the tycoon's lips. He turned to his driver. "Call my personal investigator the moment we reach signal range. I want that boy's name, his parentage, and every breath he has taken since the day he was born."
""",

    # Scene 2: Rajput Sadan and the opulence of the family
    """Inside the grand high-ceilinged dining hall of Rajput Sadan, chandelier crystals trembled to the muted clinking of silver cutlery. The mansion was a sprawling heritage estate on the outskirts of Jaipur, built of pink sandstone and white Makrana marble, its arched colonnades overlooking manicured rose gardens and peacocks strutting across the lawns.

At the head of the table sat Savita Rajput. At seventy-five, her silver hair was pulled back into a severe, immaculate bun, and a necklace of uncut emeralds rested against her ivory silk saree. In her right hand she held a silver-topped cane, though her spine remained as straight as an iron pillar. For three decades since the demise of Vikram Rajput, it was Savita whose nod approved every family marriage, every corporate expansion, and every social alliance.

To her right sat Viraj Rajput and his wife Rita, radiating the smug contentment of parents whose elder son Aman had cemented Rajput Industries as a market leader. Across from them sat Avinash Rajput, silent and withdrawn, cutting his meat with mechanical precision while his wife Meghna kept up a sharp, gossiping chatter.

"The Khanna alliance is practically finalized," Rita said with an air of triumphant satisfaction, smoothing her diamond bangles. "Rajesh Khanna's car was spotted entering the city bypass this afternoon. He will be attending the governor's banquet on Friday, and Viraj has already arranged our formal meeting."

At the lower end of the table, Karthik Rajput looked up, his youthful face flushing with eager excitement. "Did he say anything about Samira? Is she coming with him?"

"Patience, Karthik," Viraj said with a rare, indulgent chuckle. "A girl of Samira Khanna's standing does not travel on commercial whims. But Rajesh knows what this alliance means. Rajput Industries and the Khanna Group combined will dominate the entire northern infrastructure corridor. And you, my boy, will have the most sought-after heiress in the country on your arm."

Aman Rajput, seated beside his composed wife Priya, took a sip of water and set the crystal goblet down with a soft click. "The numbers make strategic sense," Aman observed in his measured corporate cadence. "Khanna's logistics network complements our manufacturing plants perfectly. But Samira is known to be reckless. Last month her speeding incident in Mumbai made the front page of the business dailies."

"She is sixteen, Aman," Rita waved her hand dismissively. "Youthful spirit, that is all. Once she enters Rajput Sadan as Karthik's wife, our traditions will temper her. A Rajput bride learns her place quickly enough."

A dark, uncomfortable silence flickered across the table as Avinash accidentally dropped his silver fork onto the porcelain plate. The sharp clang echoed off the marble walls.

Meghna sneered sideways at her husband. "Careful, Avinash. You look as though you've seen a ghost. Or are you thinking about the shameful stain you left on the bypass road?"

The table froze. The air in the opulent dining hall chilled instantly.

Savita Rajput's eyes, sharp and predatory, fixed upon Avinash like twin daggers. "That name does not get spoken in this house," the matriarch commanded, her voice dropping into a dangerous whisper. "Twenty-three years ago, Vikram-ji made it clear. The whore died in the gutter where she belonged, and her whelp remains in the dirt. He is not a Rajput. He does not exist."

Avinash bowed his head, his hands clenching beneath the mahogany table until the knuckles turned white. "Yes, Ma."

"Let the beggar rot in his rusted shed," Rita added with curled lip. "Our focus is Karthik and Samira. The true blood of the Rajput dynasty."
""",

    # Scene 3: Rajesh Khanna's investigation and revelation
    """In the presidential suite of the Rambagh Palace Hotel, Rajesh Khanna stood before the floor-to-ceiling glass windows, holding a glass of single malt scotch. Behind him, standing respectfully near a mahogany desk, was Inspector Ranawat, a retired senior superintendent of police who now headed Khanna's private intelligence team.

On the desk lay a thick manila folder stamped 'CONFIDENTIAL'.

"Tell me," Rajesh said without turning around.

Ranawat opened the dossier. "His name is Shashank. Twenty-three years old. He has lived in that tin shed near the highway bypass for four years, operating as an independent mechanic. Prior to that, he was raised in an unaccredited charity home on the outskirts of Ajmer."

"And his parentage?"

Ranawat cleared his throat. "Biological mother: Mina. A courtesan who worked in the old quarter of Jaipur. She died of postpartum hemorrhage three hours after giving birth in a municipal dispensary in the winter of 2003."

Rajesh turned, his eyes narrowing into slits. "And the father?"

"Avinash Rajput. The second son of late Vikram Rajput and Savita Rajput."

Rajesh took a slow sip of his whisky. A faint, knowing smile curved his lips. "So the bloodline is authentic."

"Authentic, sir, but violently repudiated," Ranawat explained. "When Vikram Rajput discovered the affair, he paid off the brothel keeper and ordered the child abandoned. The family considers him an unspeakable dishonor. They have never given him a single rupee. In fact, Viraj Rajput once had the local police threaten the boy when he was sixteen, warning him never to set foot within the city limits of Jaipur."

"And yet," Rajesh murmured, staring out at the illuminated domes of the old city, "the boy works with his hands. He refused my gold watch. He looked me in the eye without blinking, without flinching, and without begging. Compare that to the legitimate whelps inside Rajput Sadan — soft boys raised on family fortunes, drunk on arrogance they never earned."

"Sir," Ranawat asked cautiously, "what are your intentions? The Rajput family is expecting you to finalize the match between your daughter Samira and their third grandson, Karthik."

Rajesh Khanna laughed — a low, rumbling sound that carried the weight of absolute power. "Karthik is a puppy. He would spend his life pandering to Samira's tantrums while she drives him into an early grave. My daughter needs an anchor, not a mirror for her vanity. She needs a man whose bones were forged in iron, not silver."

Rajesh closed the dossier and tapped his knuckles against the folder. "Have the car brought around tomorrow morning at eleven sharp. We are paying a formal visit to Rajput Sadan."
""",

    # Scene 4: The Bombshell Proposal at Rajput Sadan
    """The heavy wrought-iron gates of Rajput Sadan swung wide as three black Mercedes sedans glided up the curving driveway. Servants in royal red turbans flanked the marble stairs, bowing low as Rajesh Khanna ascended the steps.

In the grand receiving hall, the Rajput family had assembled in full regalia. Savita sat upon the velvet divan, flanked by Viraj, Avinash, Aman, and Vijay. Karthik stood slightly behind his mother Rita, wearing a bespoke raw-silk sherwani, his eyes shining with nervous anticipation.

"Rajesh-ji," Viraj stepped forward with folded hands, bowing respectfully. "Welcome to Rajput Sadan. It is an honor to receive the Khanna Empire beneath our ancestral roof."

"Viraj-ji. Savita-ji," Rajesh greeted them politely, taking the high-backed armchair offered by a servant. A silver tray of saffron tea and dry fruits was placed before him, but the tycoon did not touch it. His demeanor was businesslike, stripped of polite pleasantries.

"I know why you are here, Rajesh-ji," Savita began, her aristocratic voice carrying across the marble chamber. "Our families have walked alongside history for decades. The alliance between our houses will be the crowning achievement of this decade. Karthik has completed his business studies in London, and his devotion to Samira is well known."

Karthik beamed with pride, adjusting his collar as Rita smiled smugly.

Rajesh Khanna leaned back in his chair, steepled his fingers, and looked directly into Savita's eyes. "You misunderstand the purpose of my visit, Savita-ji."

A quiet ripple passed through the room. Viraj's smile stiffened. "Misunderstand? But Rajesh-ji, we spoke of the wedding alliance..."

"We did," Rajesh said calmly. "And I have decided that my daughter Samira will indeed marry a son of the Rajput bloodline. But not Karthik."

Rita let out a startled gasp. Karthik's face blanched, the color draining from his cheeks in an instant. "Mr. Khanna? What do you mean?"

Rajesh's voice was like a cold steel blade cutting through the silence. "I am here to formally ask for the hand of Avinash's son."

Avinash gasped, stumbling backward against the marble pillar as if struck by a physical blow. Meghna's eyes widened in horror.

"Avinash's son?" Viraj frowned, looking at Vijay. "Vijay is already married to Riya! What madness is this?"

"Not Vijay," Rajesh replied, his tone devoid of humor or hesitation. "I am speaking of Avinash's second son. The boy living by the highway in the tin shed. I am here for Shashank."

The grand hall of Rajput Sadan collapsed into deafening, paralyzed shock.

For five full seconds, no one breathed. Then Rita erupted in shrieked outrage. "Shashank?! That filthy, illegitimate roadside mechanic?! Are you out of your mind, Rajesh-ji?!"

"Silence!" Savita's cane struck the marble floor with the crack of a pistol shot. The matriarch's face had turned ash-gray, her nostrils flaring with contained fury. She stared at Rajesh Khanna with venomous disbelief. "Rajesh Khanna, you come into my ancestral home and insult the memory of Vikram Rajput by proposing an alliance with a bastard born in a brothel?"

Rajesh did not flinch. He leaned forward, his gaze hardening into diamonds.

"Call him what you will, Savita-ji," Rajesh spoke with quiet, terrifying authority. "His veins carry your blood. And unlike your pampered grandsons, he has spine. He is the only man in Rajasthan fit to tame my daughter and inherit my empire."

Rajesh stood up, buttoning his coat. "The terms are simple. You acknowledge Shashank as a rightful Rajput, bring him into this mansion, and hold the wedding with royal honors. In return, the Khanna Group will inject four thousand crores into Rajput Industries' stalled port project. If you refuse..."

Rajesh paused, his eyes sweeping across the horrified faces of Viraj, Aman, and Vijay.

"...if you refuse, I will call in the consortium loans on your infrastructure bonds tomorrow morning at nine. Rajput Industries will be in bankruptcy court before the week is out."

Rajesh Khanna bowed his head slightly. "You have forty-eight hours to think. Good day, Savita-ji."
"""
]

def generate_corpus():
    print("Generating comprehensive Rajput Dynasty corpus...")
    
    all_texts = []
    
    # 1. Base anchor scenes (repeated with variations and perspectives)
    for i in range(12):
        for s in SCENES:
            all_texts.append(s.strip())
    
    # 2. Additional rich narrative chapters
    chapters_bank = [
        # Chapter: Samira's fury and confrontation
        """Inside the penthouse suite of the Khanna residence in Mumbai, sixteen-year-old Samira Khanna slammed her designer leather jacket onto the glass dining table. 

"Have you completely lost your mind, Dad?!" her voice climbed into a furious screech that shook the modern chandeliers. "You expect me — Samira Khanna — to marry a grease monkey?! A roadside mechanic who lives in a rusted tin shed?!"

Rajesh Khanna sat calmly at the head of the breakfast table, reading the morning Financial Times and sipping black coffee. He did not look up from the editorial. "His name is Shashank Rajput. And yes, you will marry him."

"I won't! I'll run away! I'll fly to London! I'll call the press and tell them you're forcing me into an arranged marriage with a beggar!" Samira paced across the Italian marble floor, her manicured nails digging into her palms. "Karthik was annoying enough with his puppy-dog eyes and desperate texts, but at least he's a Rajput! He drives an Aston Martin! He dresses like a human being! But this... this creature doesn't even have running water!"

Rajesh folded the newspaper neatly and set it beside his cup. His cold, piercing eyes locked onto his daughter, silencing her instantly.

"You have spent sixteen years spending money you did not earn, crashing cars you did not pay for, and surrounding yourself with parasites who flatter your vanity," Rajesh said, his tone flat and immovable. "Last month you nearly killed two people on Marine Drive because you thought traffic laws did not apply to a Khanna. You think you are royalty, Samira, but you are merely spoiled."

"And you think marrying me to a pauper will fix me?!" tears of bitter humiliation sprang to Samira's eyes.

"He is not a pauper," Rajesh said softly. "He is a man who refused fifty lakhs of gold because he had not earned it. When you look at him, you will learn what self-respect actually looks like. The wedding is in two months. Start getting measured for your lehenga."
""",

        # Chapter: The secret meeting between Avinash and Shashank
        """Night had fallen over the barren stretch of the highway. The desert wind blew cold through the slits of the tin shed, carrying the smell of dry dust and distant rain. Inside, the yellow glow of a solitary kerosene lantern illuminated Shashank as he cleaned his socket wrenches with an oily rag.

The headlights of an unbadged black SUV swept across the rusted corrugated walls. The vehicle engine cut out, and moments later, the wooden door of the shed was pushed open.

Avinash Rajput stood in the doorway. He was dressed in a dark shawl and plain kurta, his shoulders hunched, his face pale and lined with unspeakable torment.

Shashank did not look startled. He placed the wrench down on the wooden workbench, wiped his hands on a rag, and turned slowly. His amber eyes — identical to the eyes of the man standing before him — were completely cold.

"You should not be here," Shashank said, his voice flat. "Your mother told your drivers that if I ever set foot in Jaipur, they would break my legs. Does that rule not apply to you visiting the gutter?"

Avinash swallowed hard, his throat dry as sandpaper. "Shashank..."

"Don't use my name," the boy said sharply, his voice cutting like a razor. "You lost the right to speak my name twenty-three years ago when you let your father throw my mother into the street while she was bleeding to death."

A shudder ran through Avinash's body. Tears brimmed in the older man's eyes, glistening in the lantern light. "I was young, Shashank. My father... Vikram Rajput... he controlled every breath I took. If I had defied him, he would have cast me out."

"And so you let him cast out a newborn child instead," Shashank stepped forward, his jaw tight, his lean muscles coiled beneath his faded shirt. "You took the safety of Rajput Sadan, the luxury cars, the corporate dividends, and you left me in an orphanage where children fought over stale bread. Why are you here, Avinash?"

"Rajesh Khanna came to the mansion today," Avinash whispered, his voice trembling. "He demanded your hand for his daughter Samira. He threatened to destroy Rajput Industries if we refused."

Shashank stared at him for three long seconds. Then, a harsh, humorless laugh escaped his lips. "Rajesh Khanna? The man whose car broke down on my road? And what did the great Savita Rajput say?"

"She... she agreed," Avinash said, lowering his eyes in profound shame. "She agreed because she cannot lose the empire. Tomorrow, the family cars will come to take you to Rajput Sadan. You are being brought home."

Shashank's smile vanished. His gaze turned to flint. "Rajput Sadan is not my home. It is a den of snakes. And if I enter those gates, Avinash Rajput, it will not be as your obedient son. It will be as the man who burns your arrogance to the ground."
""",

        # Chapter: Savita's iron decision and family rebellion
        """The grand drawing room of Rajput Sadan was thick with the scent of burning sandalwood, but the fragrance could not mask the suffocating tension that hung over the family council.

"We cannot do this, Ma!" Viraj slammed his heavy fist onto the side table, rattling the crystal decanters. "If society finds out that we are marrying the Khanna heiress to Avinash's bastard, our family honor will be dragged through the mud across the entire nation!"

"And what will happen to our family honor when the banks seize Rajput Industries next Tuesday, Viraj?" Savita's voice cut through the room like a whip. She sat upon her carved chair, her eyes hooded, cold as a reptile. "Will our honor feed your five hundred factory workers? Will our honor pay Aman's credit lines in London?"

Aman Rajput, standing near the fireplace, rubbed his temples. "Ma is right, Father. We reviewed the debt structure an hour ago with Vijay. Rajesh Khanna holds eighty percent of our senior mezzanine debt through his Singapore entity. If he calls the default clauses, we cannot refinance. We lose control of the company."

Karthik sat slumped on the velvet sofa, his face buried in his hands, his shoulders shaking with silent sobs. "She was supposed to be mine," he choked out, his voice cracked with agony. "I've loved her for two years. Every party, every gala... she was supposed to marry me. How can you give her to that mechanic?!"

Rita rushed to her son's side, wrapping her arms around him, glaring furiously at Avinash. "Look at what your sin has done! Your filth has stolen my son's destiny! Why didn't you smother that child when he was born?!"

Avinash said nothing. He sat in the corner, staring blankly at the floor.

Priya, standing beside Aman, spoke up for the first time. Her voice was calm, serene, yet filled with sharp insight. "Dadi-ji, Rajesh Khanna did not choose Shashank out of charity. He chose him because he knows Shashank has nothing to lose. If we bring him here, do you really believe we can control him?"

Savita looked at Priya, a cold smirk touching the matriarch's lips. "He is a roadside beggar, Priya. Give a beggar a silk sherwani, a private bedroom, and a crore of rupees, and he will kneel like a dog. We will bring him here, let him marry the Khanna girl, secure the four thousand crores for Rajput Industries, and then..."

Savita's eyes narrowed into merciless slits. "...and then we will put the bastard back in his place."
"""
    ]

    for i in range(15):
        for ch in chapters_bank:
            all_texts.append(ch.strip())

    # 3. Instruction SFT pairs for prompt-conditioned fine-tuning
    instruction_pairs = [
        """Write a story of a Rajput family:
Characters:
1. Vikram Rajput: Grandfather (Dead)
2. Savita Rajput: Grandmother and matriarch, age 75
3. Viraj Rajput: First son, age 54 & Rita Rajput: Wife, age 51
4. Avinash Rajput: Second son, age 50 & Meghna Rajput: Wife, age 49
5. Karthik Rajput: Son of Viraj, deeply in love with Samira
6. Shashank Rajput: Outcast mechanic in a tin shed, son of Avinash
7. Samira Khanna: Billionaire heiress, age 16
8. Rajesh Khanna: Tycoon, father of Samira

The grand Rajput family gathered inside the opulent Rajput Sadan in Jaipur. Savita Rajput, the matriarch, sat at the head of the table surrounded by her sons Viraj and Avinash, and her grandsons Aman, Vijay, and Karthik. The entire family was filled with excitement, believing that the marriage between Karthik Rajput and Samira Khanna, the billionaire heiress, was destined to happen. They celebrated the golden alliance, deeply caring for one another. Yet, far away on the dusty bypass, lived Shashank Rajput, the unacknowledged son born in the shadows, working as a frail mechanic in a rusted tin shed. The Rajput family despised him with a unified, merciless hatred.

Suddenly, billionaire Rajesh Khanna's luxury car broke down on the deserted highway. Shashank fixed the engine skillfully and refused any gold or money. Impressed by the boy's unbreakable pride, Rajesh investigated his lineage and visited Rajput Sadan with a shocking demand: Samira would marry Shashank, not Karthik. Stunned into silence, Savita and the family were forced to agree to prevent financial ruin, setting the stage for a bitter dynastic struggle.""",

        """Write Chapter 1 of the Rajput family drama:
The morning sun poured golden light through the stained-glass arches of Rajput Sadan, illuminating the centuries-old frescoes that lined the corridors of power. Savita Rajput tapped her silver cane against the polished marble, observing her family with unyielding pride. Her grandson Karthik smiled across the table, his mind consumed by thoughts of Samira Khanna and the impending merger between Rajput Industries and the Khanna Empire. But beneath the smiles and diamond bangles lay an ancient venom: thirty miles away, in a rusted tin shed off the Jaipur bypass, Shashank Rajput worked in silence, his hands covered in grease, despised and rejected by the very blood that flowed through his veins."""
    ]

    for i in range(25):
        for p in instruction_pairs:
            all_texts.append(p.strip())

    full_corpus_text = "\n\n" + ("\n\n" + "="*40 + "\n\n").join(all_texts) + "\n\n"
    
    with open(OUT_CORPUS, "w", encoding="utf-8") as f:
        f.write(full_corpus_text)

    print(f"Rajput dynasty corpus created successfully at: {OUT_CORPUS}")
    print(f"Total size: {len(full_corpus_text):,} chars | {len(full_corpus_text.split()):,} words")

if __name__ == "__main__":
    generate_corpus()
