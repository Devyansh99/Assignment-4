import sys
sys.path.insert(0, '.')
from wiki_tool import get_links, _url_to_title
import time, json, importlib.util

spec = importlib.util.spec_from_file_location('agent', 'avishkar_devyansh.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

pairs_to_test = [
    {'pair_id': '4', 'start': 'https://en.wikipedia.org/wiki/Photosynthesis', 'target': 'https://en.wikipedia.org/wiki/World_War_II', 'difficulty': 'medium'},
    {'pair_id': '5', 'start': 'https://en.wikipedia.org/wiki/Chocolate', 'target': 'https://en.wikipedia.org/wiki/Roman_law', 'difficulty': 'medium'},
    {'pair_id': '7', 'start': 'https://en.wikipedia.org/wiki/Jazz', 'target': 'https://en.wikipedia.org/wiki/Mathematics', 'difficulty': 'hard'},
    {'pair_id': '8', 'start': 'https://en.wikipedia.org/wiki/Yoga', 'target': 'https://en.wikipedia.org/wiki/Nuclear_physics', 'difficulty': 'hard'},
    {'pair_id': '9', 'start': 'https://en.wikipedia.org/wiki/Telescope', 'target': 'https://en.wikipedia.org/wiki/Renaissance', 'difficulty': 'medium'},
    {'pair_id': '10', 'start': 'https://en.wikipedia.org/wiki/Haiku', 'target': 'https://en.wikipedia.org/wiki/Plate_tectonics', 'difficulty': 'hard'},
]

for pair in pairs_to_test:
    deadline = time.time() + 40
    global_budget = {'remaining': 6}
    result = mod._solve_one(pair, deadline, 5, global_budget)
    path_end = result['path'][-1].split('/wiki/')[-1] if result['path'] else 'EMPTY'
    path_titles = [p.split('/wiki/')[-1] for p in result['path']]
    print(f"Pair {pair['pair_id']}: success={result['success']}, steps={result['steps']}, ended_at={path_end}")
    print(f"  Path: {path_titles}")
    print()
