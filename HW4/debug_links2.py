import sys; sys.path.insert(0,'.')
from wiki_tool import get_links

# Check Nuclear triad -> Nuclear physics
nt = get_links('https://en.wikipedia.org/wiki/Nuclear_triad')
np_link = [l for l in nt if 'Nuclear_physics' in l['url'] or 'nuclear physics' in l['text'].lower()]
print(f"Nuclear triad -> Nuclear physics: {np_link}")

# Check Nuclear power in India -> Nuclear physics
npi = get_links('https://en.wikipedia.org/wiki/Nuclear_power_in_India')
np2 = [l for l in npi if 'Nuclear_physics' in l['url'] or 'nuclear physics' in l['text'].lower()]
print(f"Nuclear power in India -> Nuclear physics: {np2[:3]}")

# Check India -> Nuclear physics directly somehow
india = get_links('https://en.wikipedia.org/wiki/India')
np3 = [l for l in india if 'Nuclear_physics' in l['url'] or 'nuclear physics' in l['text'].lower()]
print(f"India -> Nuclear physics direct: {np3}")

# Best path for Jazz -> Mathematics
# Try China -> Mathematics
china = get_links('https://en.wikipedia.org/wiki/China')
math = [l for l in china if 'Mathematics' == l['url'].split('/wiki/')[-1] or l['text'].lower() == 'mathematics']
print(f"\nChina -> Mathematics direct: {math[:3]}")

# Try Science -> Mathematics
sci = get_links('https://en.wikipedia.org/wiki/Science')
math2 = [l for l in sci if 'Mathematics' == l['url'].split('/wiki/')[-1] or l['text'].lower() == 'mathematics']
print(f"Science -> Mathematics direct: {math2[:3]}")

# Check Jazz -> Science (or Mathematics) via US
us = get_links('https://en.wikipedia.org/wiki/United_States')
sci2 = [l for l in us if l['text'].lower() in ('science', 'science and technology in the united states')]
print(f"\nUS -> Science: {sci2[:5]}")
