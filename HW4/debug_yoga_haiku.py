import sys
sys.path.insert(0, '.')
from wiki_tool import get_links

# Check Yoga page for nuclear/physics/science links
yoga_links = get_links('https://en.wikipedia.org/wiki/Yoga')
print(f"Total Yoga links: {len(yoga_links)}")

# Find links that could navigate toward Nuclear physics
good = [l for l in yoga_links if any(w in l['text'].lower() for w in 
    ['science', 'physics', 'nuclear', 'india', 'atom', 'energy', 'nature', 
     'university', 'research', 'ancient', 'history', 'europe', 'greek'])]
print("Potentially useful links from Yoga:")
for l in good[:20]:
    print(f"  {l['text']}: {l['url'].split('/wiki/')[-1]}")

print()
# Check Haiku page for plate tectonics paths
haiku_links = get_links('https://en.wikipedia.org/wiki/Haiku')
print(f"Total Haiku links: {len(haiku_links)}")
good2 = [l for l in haiku_links if any(w in l['text'].lower() for w in
    ['plate', 'tectonic', 'earth', 'geology', 'science', 'nature', 'japan',
     'geography', 'island', 'mountain', 'asia', 'rock', 'history'])]
print("Potentially useful links from Haiku:")
for l in good2[:20]:
    print(f"  {l['text']}: {l['url'].split('/wiki/')[-1]}")
