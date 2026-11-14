import json
import re
from collections import Counter

common_words = {
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
    'Sister', 'Uncle', 'Aunt', 'Grandmother', 'Grandfather', 'God', 'Earth'
}

with open(r'C:\Users\user\.gemini\antigravity\scratch\gpt_from_scratch\data\opus-writing-prompts-1-raw.jsonl', 'r', encoding='utf-8') as f:
    for i in range(5):
        d = json.loads(f.readline())
        story = d['story']
        names = re.findall(r'\b([A-Z][a-z]{2,15})\s+(?:said|replied|whispered|asked|muttered|sighed|shouted|nodded|stepped|turned|looked|glanced|smiled|frowned|walked|stood)\b', story)
        names += re.findall(r'\b(?:said|replied|whispered|asked|muttered|sighed|shouted)\s+([A-Z][a-z]{2,15})\b', story)
        counts = Counter([n for n in names if n not in common_words])
        print(f"Story {i+1} '{d.get('title')}': Found Characters -> {counts.most_common(3)}")
