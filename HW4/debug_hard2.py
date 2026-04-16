import sys
sys.path.insert(0, '.')
from wiki_tool import get_links

# Check what intermediate articles US/India/Japan have that
# directly link to Mathematics, Nuclear physics, Plate tectonics

checks = [
    ('United_States', 'https://en.wikipedia.org/wiki/United_States', 'Mathematics'),
    ('India', 'https://en.wikipedia.org/wiki/India', 'Nuclear physics'),
    ('Japan', 'https://en.wikipedia.org/wiki/Japan', 'Plate tectonics'),
]

for hub_name, hub_url, target in checks:
    links = get_links(hub_url)
    # Find exact target
    exact = [l for l in links if l['text'].lower() == target.lower()]
    print(f"\n{hub_name} -> '{target}':")
    if exact:
        print(f"  DIRECT LINK: {exact}")
    else:
        # Find partial
        t_words = set(target.lower().split()) - {'the', 'a', 'of'}
        partial = [l for l in links if all(w in l['text'].lower() for w in t_words)]
        print(f"  Partial matches: {partial[:5]}")
        
        # What known hubs are on these pages?
        known = {"science", "physics", "chemistry", "mathematics", "technology",
                 "geology", "nuclear", "plate", "tectonic", "earth science",
                 "natural science", "history of science"}
        hub_links = [l for l in links if any(k in l['text'].lower() for k in known)]
        print(f"  Science-related links: {hub_links[:10]}")
