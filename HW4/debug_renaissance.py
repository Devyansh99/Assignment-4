import sys
sys.path.insert(0, '.')
from wiki_tool import get_links, _url_to_title

# Check what The_Renaissance page looks like
try:
    links = get_links('https://en.wikipedia.org/wiki/The_Renaissance')
    ren_links = [l for l in links if 'renaissance' in l['text'].lower() or 'renaissance' in l['url'].lower()]
    print("The_Renaissance page links mentioning renaissance:", ren_links[:5])
    print("Total links on The_Renaissance:", len(links))
    print("First 10 links:", links[:10])
except Exception as e:
    print(f"Error: {e}")

# Also check: what does wiki_tool return for Cake page (pair 9 path)?
try:
    cake_links = get_links('https://en.wikipedia.org/wiki/Cake')
    ren_links = [l for l in cake_links if 'renaissance' in l['text'].lower() or 'renaissance' in l['url'].lower()]
    print("\nCake page renaissance links:", ren_links)
    # Also 'the renaissance' specifically
    the_ren = [l for l in cake_links if 'the renaissance' in l['text'].lower()]
    print("Cake 'The Renaissance' links:", the_ren[:5])
except Exception as e:
    print(f"Error getting Cake links: {e}")
