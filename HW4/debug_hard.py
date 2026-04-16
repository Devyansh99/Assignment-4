import sys
sys.path.insert(0, '.')
from wiki_tool import get_links

# Check what KNOWN_HUBS are directly on the hard start pages
# and what hubs those hubs contain that could reach the targets

hard_pairs = [
    ('Jazz', 'https://en.wikipedia.org/wiki/Jazz', 'Mathematics'),
    ('Yoga', 'https://en.wikipedia.org/wiki/Yoga', 'Nuclear physics'),
    ('Haiku', 'https://en.wikipedia.org/wiki/Haiku', 'Plate tectonics'),
    ('Chocolate', 'https://en.wikipedia.org/wiki/Chocolate', 'Roman law'),
    ('Ballet', 'https://en.wikipedia.org/wiki/Ballet', 'Computer science'),
]

known_hubs = {
    "europe", "asia", "india", "china", "japan", "france", "germany",
    "science", "history", "mathematics", "physics", "chemistry",
    "biology", "technology", "philosophy", "religion", "culture",
    "art", "music", "united states", "america", "england", "war",
    "ancient", "medieval", "nature", "earth", "atom", "energy",
    "nuclear", "geology", "engineering", "agriculture", "revolution",
    "astronomy",
}

for name, url, target in hard_pairs:
    links = get_links(url)
    hubs_found = [l for l in links if l['text'].lower().strip() in known_hubs]
    print(f"\n{name} -> {target}")
    print(f"  Hub links on page: {[l['text'] for l in hubs_found]}")
    
    # Check if any hub directly contains the target  
    import time
    for hub in hubs_found[:5]:
        try:
            hub_links = get_links(hub['url'])
            target_there = any(target.lower() in l['text'].lower() for l in hub_links)
            if target_there:
                matching = [l for l in hub_links if target.lower() in l['text'].lower()]
                print(f"  ** {hub['text']} HAS '{target}': {matching[:3]}")
            else:
                # Check partial
                t_words = set(target.lower().split()) - {'the', 'a', 'of', 'in'}
                partial = [l for l in hub_links if all(w in l['text'].lower() for w in t_words)]
                if partial:
                    print(f"  * {hub['text']} has partial: {partial[:2]}")
        except:
            pass
