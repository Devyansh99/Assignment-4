import sys; sys.path.insert(0,'.')
from wiki_tool import get_links

# Does India link directly to Nuclear physics?
india = get_links('https://en.wikipedia.org/wiki/India')
nuke = [l for l in india if 'nuclear' in l['text'].lower()]
print('India nuclear links:')
for l in nuke[:10]: print(f"  {l['text']} -> {l['url'].split('/wiki/')[-1]}")

# Check CTBT -> Nuclear physics
ctbt = get_links('https://en.wikipedia.org/wiki/Comprehensive_Nuclear-Test-Ban_Treaty')
np_link = [l for l in ctbt if 'nuclear physics' in l['text'].lower() or 'Nuclear_physics' in l['url']]
print(f"\nCTBT -> Nuclear physics? {np_link[:3]}")

# Check US -> Mathematics
us = get_links('https://en.wikipedia.org/wiki/United_States')
math = [l for l in us if l['text'].lower() == 'mathematics' or 'Mathematics' == l['url'].split('/wiki/')[-1]]
print(f"\nUS -> Mathematics direct? {math[:3]}")

# Check Japan -> Plate tectonics
japan = get_links('https://en.wikipedia.org/wiki/Japan')
geo = [l for l in japan if 'plate tectonic' in l['text'].lower() or 'Plate_tectonics' in l['url']]
print(f"\nJapan -> Plate tectonics direct? {geo[:3]}")
asia = [l for l in japan if l['text'] == 'Asia']
print(f"Japan -> Asia direct? {asia[:3]}")

# Check Asia -> Plate tectonics
asia_pg = get_links('https://en.wikipedia.org/wiki/Asia')
geo2 = [l for l in asia_pg if 'plate tectonic' in l['text'].lower() or 'Plate_tectonics' in l['url']]
print(f"\nAsia -> Plate tectonics direct? {geo2[:3]}")
