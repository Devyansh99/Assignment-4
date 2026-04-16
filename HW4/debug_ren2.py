import sys
sys.path.insert(0, '.')
from wiki_tool import get_links, _url_to_title

# Check if Renaissance (exact) is on The_Renaissance page
links = get_links('https://en.wikipedia.org/wiki/The_Renaissance')
exact_ren = [l for l in links if l['url'].rstrip('/').lower() == 'https://en.wikipedia.org/wiki/renaissance']
print("Exact Renaissance URL on The_Renaissance:", exact_ren)

exact_text = [l for l in links if l['text'].lower() == 'renaissance']
print("Exact 'Renaissance' text on The_Renaissance:", exact_text)

# Check: does the validator in score.py compare titles or URLs?
# score.py line 83: _url_to_title(path[-1]).lower() != _url_to_title(target_url).lower()
# So if path[-1] = The_Renaissance -> "the renaissance" != "renaissance" -> FAIL

# What is 'The_Renaissance' actually? Let's see the API response
import json, urllib.request, urllib.parse
API_URL = "https://en.wikipedia.org/w/api.php"
params = {"action": "parse", "page": "The Renaissance", "prop": "links", "redirects": "1", "pllimit": "500", "format": "json"}
url = API_URL + "?" + urllib.parse.urlencode(params)
req = urllib.request.Request(url, headers={"User-Agent": "test"})
with urllib.request.urlopen(req, timeout=10) as r:
    data = json.loads(r.read())
print("Canonical title of 'The Renaissance':", data['parse']['title'])
print("Redirects:", data['parse'].get('redirects', []))
