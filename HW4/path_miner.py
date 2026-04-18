"""
path_miner.py
------------
Mine reusable Wikipedia bridge priors using only wiki_tool.get_links().

Example:
    python path_miner.py --pairs practice_pairs.csv --output bridge_priors.json
"""

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict

from wiki_tool import get_links


STOP_WORDS = {
    "the", "a", "an", "of", "in", "and", "or", "to", "for", "is",
    "are", "was", "were", "by", "at", "with", "from", "its", "it",
    "as", "that", "this", "on", "be", "has", "had", "have",
    "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii",
}

SEED_HUBS = {
    "science", "history", "technology", "physics", "chemistry", "biology",
    "mathematics", "engineering", "energy", "astronomy", "geology", "law",
    "united states", "india", "japan", "china", "europe", "ancient greece",
}


def _title(url: str) -> str:
    return url.split("/wiki/")[-1].replace("_", " ")


def _norm(text: str) -> str:
    t = text.strip().lower()
    if t.startswith("the "):
        t = t[4:]
    return " ".join(t.split())


def _kwords(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower())) - STOP_WORDS


def _is_symbol_title(text: str) -> bool:
    t = text.strip()
    if re.fullmatch(r"[^a-zA-Z]+|[A-Z]{1,2}", t):
        return True
    if len(t) <= 8 and re.fullmatch(r"[A-Z][A-Z0-9.\-]+", t):
        return True
    return False


def _find_direct(links: list, target_url: str, target_title: str) -> dict | None:
    tu = target_url.rstrip("/").lower()
    nt = _norm(target_title)
    for link in links:
        lu = link["url"].rstrip("/").lower()
        if lu == tu:
            return link
        if _norm(_title(link["url"])) == nt:
            return link
    return None


def _rank_links(links: list, target_words: set, limit: int) -> list:
    scored = []
    for i, link in enumerate(links):
        text = link.get("text", "")
        words = _kwords(text + " " + _title(link["url"]))
        overlap = len(words & target_words)

        hub_bonus = 1 if _norm(text) in SEED_HUBS else 0
        short_penalty = 100 if (_is_symbol_title(text) or len(text) < 4) else 0
        length_penalty = len(text) if len(text) <= 30 else 60

        scored.append((-(overlap + hub_bonus), short_penalty, length_penalty, i, link))

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return [x[4] for x in scored[:limit]]


def _mine_pair(pair: dict, max_pivots: int, max_mids: int, per_pair_s: float):
    start = pair["start"]
    target = pair["target"]
    target_title = _title(target)
    target_words = _kwords(target_title)
    deadline = time.time() + per_pair_s

    used_hubs = []
    used_words = []

    try:
        start_links = get_links(start)
    except Exception:
        return None, used_hubs, used_words

    direct = _find_direct(start_links, target, target_title)
    if direct:
        return [start, direct["url"]], used_hubs, used_words

    pivots = _rank_links(start_links, target_words, max_pivots)

    for pivot in pivots:
        if time.time() > deadline:
            break

        purl = pivot["url"]
        ptitle = _title(purl)
        pnorm = _norm(ptitle)

        try:
            plinks = get_links(purl)
        except Exception:
            continue

        used_hubs.append(pnorm)
        used_words.extend(_kwords(ptitle))

        d2 = _find_direct(plinks, target, target_title)
        if d2:
            return [start, purl, d2["url"]], used_hubs, used_words

        mids = _rank_links(plinks, target_words, max_mids)
        for mid in mids:
            if time.time() > deadline:
                break

            murl = mid["url"]
            mtitle = _title(murl)
            mnorm = _norm(mtitle)

            try:
                mlinks = get_links(murl)
            except Exception:
                continue

            used_hubs.append(mnorm)
            used_words.extend(_kwords(mtitle))

            d3 = _find_direct(mlinks, target, target_title)
            if d3:
                return [start, purl, murl, d3["url"]], used_hubs, used_words

    return None, used_hubs, used_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="CSV with pair_id,start,target,difficulty")
    ap.add_argument("--output", default="bridge_priors.json")
    ap.add_argument("--max-pivots", type=int, default=18)
    ap.add_argument("--max-mids", type=int, default=8)
    ap.add_argument("--max-seconds-per-pair", type=float, default=7.5)
    ap.add_argument("--pair-limit", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    with open(args.pairs, newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    if args.pair_limit > 0:
        pairs = pairs[:args.pair_limit]

    global_hubs = Counter()
    target_words = defaultdict(Counter)
    sample_paths = {}
    solved = 0

    t0 = time.time()
    for pair in pairs:
        target_key = _norm(_title(pair["target"]))
        path, hubs, words = _mine_pair(
            pair,
            max_pivots=args.max_pivots,
            max_mids=args.max_mids,
            per_pair_s=args.max_seconds_per_pair,
        )

        for h in hubs:
            if h and len(h) >= 4:
                global_hubs[h] += 1
        for w in words:
            if len(w) >= 4:
                target_words[target_key][w] += 1

        if path:
            solved += 1
            sample_paths[pair["pair_id"]] = [_title(u) for u in path]
            # Favor actual successful path nodes a bit more.
            for node in path[1:-1]:
                global_hubs[_norm(_title(node))] += 3
                for w in _kwords(_title(node)):
                    if len(w) >= 4:
                        target_words[target_key][w] += 2

        print(
            f"pair {pair['pair_id']}: "
            f"{'found' if path else 'none'}"
        )

    output = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 2),
        "pairs_processed": len(pairs),
        "pairs_solved": solved,
        "global_hubs": [k for k, _ in global_hubs.most_common(40)],
        "target_boost_words": {
            tgt: [w for w, _ in ctr.most_common(10)]
            for tgt, ctr in target_words.items()
            if ctr
        },
        "sample_paths": sample_paths,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {args.output}")
    print(f"Solved: {solved}/{len(pairs)}")


if __name__ == "__main__":
    main()
