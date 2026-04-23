"""
agent.py — Wikipedia Navigation Agent v9
==========================================
Base: v8 (127 pts / 10/10 reached, avg ~6.5 steps).

Changes from v8 — all targeted at generalization, nothing hardcoded:

  REMOVED: _BRIDGE_HINTS
    Was a dict of all 10 practice target names with hand-picked keywords.
    Would add zero benefit on hidden pairs and risk false matches.
    Replaced by: _DOMAIN_EXPANSIONS — a general keyword map keyed on
    domain words found IN the target title (e.g. any target containing
    "physics" also searches for {"science","energy"}). Works for any pair.

  REMOVED: BFS math special-case
    `if difficulty=="hard" and target_key=="mathematics"` was pure
    overfitting to pair 7. Removed. BFS parameters come only from the
    difficulty-keyed dicts (_LOOKAHEAD_PIVOTS, _LOOKAHEAD_MIDS) which
    are general.

  REMOVED: bridge_priors.json / _load_bridge_priors()
    External file that won't exist in the test environment. The code
    degraded gracefully but added dead complexity for zero gain.

  KEPT (unchanged): _KNOWN_HUBS, BFS lookahead, path compression,
    _two_hop_upgrade, roman numeral stop words, hub penalty, dynamic
    confidence threshold, hub-mode LLM prompt, link caching.

  IMPROVED: _DOMAIN_EXPANSIONS
    Covers 20 common academic domains. Any target whose title contains
    a domain word gets its search vocabulary expanded with related terms.
    This generalises the semantic search without knowing the pair in advance.

  IMPROVED: _KNOWN_HUBS
    Expanded from ~50 to ~80 entries covering more Wikipedia high-degree
    nodes (major universities, institutions, organisations that appear
    frequently as transit articles).
"""

import json, os, re, time, urllib.request
from wiki_tool import get_links

# ── API setup ─────────────────────────────────────────────────────────────────
_API_KEY = "AIzaSyByVuCEh7DCmkqmwBVyZ3bVgBC7OY7M7-s"

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent?key=" + _API_KEY
)

# ── Tuning constants ──────────────────────────────────────────────────────────
_MAX_STEPS        = 14
_STOP_BUFFER      = 1.5
_MIN_PAIR_TIME    = 4.5
_LINKS_TO_LLM     = 50
_GLOBAL_LLM_CAP   = 50
_MIN_HUB_LEN      = 5
_MAX_HUB_LEN      = 28
_DIFF_LLM_BUDGET  = {"easy": 3, "medium": 5, "hard": 7}
_DIFF_ORDER       = {"easy": 0, "medium": 1, "hard": 2}
_LOOKAHEAD_PIVOTS = {"easy": 4, "medium": 5, "hard": 6}
_LOOKAHEAD_MIDS   = {"easy": 0, "medium": 2, "hard": 4}
_IMPROVE_MIN_REMAIN = 38.0
_IMPROVE_SWEEP    = {"medium": 24, "hard": 12}

# Roman numerals added: prevent "Photosystem II" matching "World War II"
_STOP_WORDS = {
    "the", "a", "an", "of", "in", "and", "or", "to", "for", "is",
    "are", "was", "were", "by", "at", "with", "from", "its", "it",
    "as", "that", "this", "on", "be", "has", "had", "have",
    "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii",
}

# ── High-connectivity Wikipedia hub articles ──────────────────────────────────
# These appear as transit nodes in a huge fraction of Wikipedia paths.
# Penalising everything else in _hub_penalty() makes these rise to the top
# when there is no keyword overlap — fully general, not pair-specific.
_KNOWN_HUBS = {
    # Geography — regions / continents
    "europe", "asia", "africa", "north america", "south america",
    "americas", "oceania", "middle east", "south asia", "east asia",
    "southeast asia", "central asia",
    # Geography — major countries
    "india", "china", "japan", "germany", "france", "england",
    "united kingdom", "united states", "russia", "italy", "spain",
    "greece", "egypt", "brazil", "australia", "canada", "iran",
    "turkey", "mexico", "pakistan", "indonesia", "nigeria",
    # Civilisations / historical empires
    "rome", "ancient rome", "ancient greece", "byzantine empire",
    "ottoman empire", "roman empire", "british empire",
    "mongol empire", "persian empire",
    # Broad academic disciplines (highest out-degree on Wikipedia)
    "science", "history", "mathematics", "physics", "chemistry",
    "biology", "technology", "philosophy", "religion", "culture",
    "art", "music", "literature", "language", "geography",
    "economics", "politics", "law", "medicine", "education",
    "astronomy", "geology", "ecology", "engineering",
    # Historical eras
    "ancient history", "medieval", "renaissance", "modern history",
    # Generic high-degree concepts
    "war", "military", "government", "society", "industry",
    "trade", "revolution", "universe", "nature", "earth",
    "energy", "force", "matter", "atom", "democracy",
    # Major universities / institutions (very high link count)
    "oxford university", "cambridge university", "harvard university",
    "mit", "stanford university",
}

# ── General domain expansion ──────────────────────────────────────────────────
# Keyed on words that may appear in ANY target title.
# When a target contains one of these words, its search vocabulary is
# expanded with the associated set — helping the scorer find relevant
# intermediate articles even before keywords directly match.
# This replaces the old _BRIDGE_HINTS which was hardcoded to the 10 practice
# pairs. This version works for any unseen target.
_DOMAIN_EXPANSIONS = {
    "physics":     {"science", "energy", "quantum", "particle", "force"},
    "chemistry":   {"science", "element", "molecule", "reaction", "compound"},
    "biology":     {"science", "life", "organism", "evolution", "cell"},
    "mathematics": {"science", "number", "geometry", "algebra", "logic", "calculation"},
    "astronomy":   {"space", "star", "planet", "universe", "cosmos", "telescope"},
    "geology":     {"earth", "rock", "mineral", "plate", "seismic", "terrain"},
    "ecology":     {"nature", "environment", "species", "habitat", "climate"},
    "medicine":    {"health", "disease", "treatment", "surgery", "biology", "body"},
    "psychology":  {"mind", "behavior", "mental", "brain", "cognition"},
    "history":     {"ancient", "medieval", "civilization", "empire", "war", "era"},
    "law":         {"legal", "court", "justice", "rights", "government", "constitution"},
    "war":         {"military", "battle", "conflict", "army", "weapon", "soldier"},
    "politics":    {"government", "democracy", "state", "power", "election", "nation"},
    "economics":   {"trade", "market", "finance", "commerce", "industry", "money"},
    "philosophy":  {"thought", "ethics", "logic", "metaphysics", "reason"},
    "religion":    {"spiritual", "belief", "church", "god", "faith", "sacred"},
    "art":         {"culture", "painting", "sculpture", "aesthetic", "creative"},
    "music":       {"instrument", "rhythm", "composition", "song", "sound"},
    "literature":  {"writing", "poetry", "novel", "author", "narrative", "text"},
    "technology":  {"engineering", "computer", "invention", "industry", "machine"},
    "engineering": {"design", "machine", "construction", "system", "technical"},
    "agriculture": {"farming", "crop", "soil", "plant", "food", "rural"},
    "nuclear":     {"radiation", "fission", "energy", "power", "atom", "physics"},
    "ancient":     {"history", "civilization", "classical", "era", "empire"},
    "revolution":  {"history", "political", "change", "industry", "movement"},
    "science":     {"research", "experiment", "discovery", "knowledge", "academic"},
    "industrial":  {"industry", "manufacturing", "production", "factory", "economic"},
    "computer":    {"technology", "software", "hardware", "digital", "programming"},
    "space":       {"universe", "astronomy", "planet", "rocket", "satellite"},
}


def _norm(title: str) -> str:
    t = title.strip().lower()
    if t.startswith("the "):
        t = t[4:]
    return " ".join(t.split())


def _expand_target_words(target: str, base_words: set) -> set:
    """
    Augment the target keyword set with related domain vocabulary.
    Uses _DOMAIN_EXPANSIONS: if the target title contains any domain key,
    add the corresponding expansion words to the search vocabulary.
    Fully general — no pair-specific knowledge required.
    """
    words = set(base_words)
    target_lower = target.lower()
    for domain_key, extras in _DOMAIN_EXPANSIONS.items():
        if domain_key in target_lower:
            words.update(extras)
    return words


# ── Gemini wrapper ────────────────────────────────────────────────────────────

def _ask_gemini(prompt: str, max_tokens: int = 6, timeout: int = 18) -> str:
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_tokens},
    }).encode()
    req = urllib.request.Request(
        _GEMINI_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ── URL / title helpers ───────────────────────────────────────────────────────

def _title(url: str) -> str:
    return url.split("/wiki/")[-1].replace("_", " ")


def _kwords(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower())) - _STOP_WORDS


def _at_target_url(cur: str, tgt: str) -> bool:
    return cur.rstrip("/").lower() == tgt.rstrip("/").lower()


def _at_target_title(cur_url: str, target: str) -> bool:
    return _norm(_title(cur_url)) == _norm(target)


def _at_target(cur_url: str, target_url: str, target: str) -> bool:
    return _at_target_url(cur_url, target_url) or _at_target_title(cur_url, target)


# ── Link helpers ──────────────────────────────────────────────────────────────

def _find_direct(links: list, target: str, target_url: str) -> dict | None:
    tu  = target_url.rstrip("/").lower()
    tnt = _norm(target)
    tkw = _kwords(target)
    for link in links:
        if link["url"].rstrip("/").lower() == tu:
            return link
        if _norm(_title(link["url"])) == tnt:
            return link
        if tkw and _kwords(link["text"]) == tkw:
            return link
    return None


def _is_symbol_title(text: str) -> bool:
    """True for single-letter articles, pure-symbol titles, and short acronyms."""
    t = text.strip()
    if re.fullmatch(r"[^a-zA-Z]+|[A-Z]{1,2}", t):
        return True
    if len(t) <= 8 and re.fullmatch(r"[A-Z][A-Z0-9.\-]+", t):
        return True
    return False


def _hub_penalty(raw_len: int, text: str) -> int:
    """
    Tiebreak score for hub selection when keyword scores are equal.
    Lower = preferred. _KNOWN_HUBS get strongly negative scores so they
    always appear first. Symbols/acronyms get +3000 so they appear last.
    Normal-length titles get their raw length (shorter = broader = better).
    """
    text_norm = _norm(text)
    if text_norm in _KNOWN_HUBS:
        return -100          # strongly prefer known high-connectivity hubs
    if _is_symbol_title(text) or raw_len < _MIN_HUB_LEN:
        return 3000 + raw_len  # push single letters, symbols to back
    if raw_len > _MAX_HUB_LEN:
        return 1000 + raw_len  # push very long obscure titles back
    return raw_len             # shorter real title = better hub candidate


def _ranked_links(links: list, target_words: set,
                  visited: set, n: int) -> tuple:
    """
    Score every unvisited link by keyword overlap with the (expanded) target
    vocabulary. Tie-break by hub quality. Returns (top_n_list, best_score).
    """
    scored = []
    for i, link in enumerate(links):
        if link["url"] in visited:
            continue
        lw    = _kwords(link["text"] + " " + _title(link["url"]))
        score = len(lw & target_words)
        tie   = _hub_penalty(len(link["text"]), link["text"])
        scored.append((score, tie, i, link))

    if not scored:
        scored = [(0, 3000, i, l) for i, l in enumerate(links)]

    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    top        = [x[3] for x in scored[:n]]
    best_score = scored[0][0] if scored else 0
    return top, best_score


# ── LLM selection ─────────────────────────────────────────────────────────────

def _llm_pick(top_links: list, current: str, target: str,
              hub_mode: bool = False) -> tuple:
    """
    Pick from REAL on-page links — hallucination impossible.

    hub_mode=True when keyword score=0: instructs the LLM to escape the
    current domain by jumping to a broad hub article, rather than staying
    in the same topic area.
    """
    numbered = "\n".join(f"{i+1}. {l['text']}"
                         for i, l in enumerate(top_links))

    if hub_mode:
        prompt = (
            f"Wikipedia navigation: '{current}' -> '{target}'.\n"
            f"These topics are unrelated — pick the BROADEST hub article "
            f"(major country, continent, historical era, or widely-linked concept) "
            f"that could eventually connect to '{target}'. "
            f"Avoid articles closely related to '{current}'.\n"
            f"{numbered}\n"
            f"NUMBER ONLY."
        )
    else:
        prompt = (
            f"Wikipedia: '{current}' -> '{target}'.\n"
            f"{numbered}\n"
            f"Best link? NUMBER ONLY."
        )

    try:
        resp = _ask_gemini(prompt, max_tokens=6, timeout=18)
        m    = re.search(r"\d+", resp)
        if m:
            idx = int(m.group()) - 1
            if 0 <= idx < len(top_links):
                return top_links[idx], 1
    except Exception:
        pass

    return top_links[0], 1


# ── BFS lookahead ─────────────────────────────────────────────────────────────

def _bfs_lookahead(start_url: str, target_url: str, target: str,
                   target_words: set, deadline: float,
                   difficulty: str) -> list | None:
    """
    Bounded BFS from start page using cached wiki calls.

    Depth 1: start → target (direct)
    Depth 2: start → pivot → target
    Depth 3: start → pivot → mid → target (medium/hard only)

    Parameters come from difficulty-keyed dicts only — no pair-specific logic.
    All calls hit the wiki_tool cache, so they cost ~0.6s each uncached.
    """
    remaining = deadline - time.time()
    if remaining < 8:
        return None

    pivots    = _LOOKAHEAD_PIVOTS.get(difficulty, 5)
    mids      = _LOOKAHEAD_MIDS.get(difficulty, 2)
    max_calls = 5 if difficulty == "easy" else (9 if difficulty == "medium" else 12)

    if remaining < 18:
        mids      = min(mids, 1)
        max_calls = min(max_calls, 6)

    calls_used      = 0
    lookahead_words = _expand_target_words(target, target_words)

    try:
        start_links = get_links(start_url)
        calls_used += 1
    except Exception:
        return None

    # Cheap win: if target is already on the start page, avoid any extra BFS probes.
    direct = _find_direct(start_links, target, target_url)
    if direct:
        return [start_url, direct["url"]]

    top_pivots, _ = _ranked_links(start_links, lookahead_words, {start_url}, pivots)

    best_depth3 = None

    for pivot_link in top_pivots:
        if calls_used > max_calls or time.time() > deadline - _STOP_BUFFER - 1.2:
            break
        pivot_url = pivot_link["url"]

        if _at_target(pivot_url, target_url, target):
            return [start_url, pivot_url]         # 1-hop found

        try:
            pivot_links = get_links(pivot_url)
            calls_used += 1
        except Exception:
            continue

        direct = _find_direct(pivot_links, target, target_url)
        if direct:
            return [start_url, pivot_url, direct["url"]]  # 2-hop found

        if best_depth3 is not None or mids <= 0:
            continue   # already have a 3-hop; keep looking for cheaper 2-hop

        top_mids, _ = _ranked_links(
            pivot_links, lookahead_words, {start_url, pivot_url}, mids,
        )

        for mid_link in top_mids:
            if calls_used > max_calls or time.time() > deadline - _STOP_BUFFER - 1.2:
                break
            mid_url = mid_link["url"]

            if _at_target(mid_url, target_url, target):
                return [start_url, pivot_url, mid_url]

            try:
                mid_links = get_links(mid_url)
                calls_used += 1
            except Exception:
                continue

            direct_mid = _find_direct(mid_links, target, target_url)
            if direct_mid and best_depth3 is None:
                best_depth3 = [start_url, pivot_url, mid_url, direct_mid["url"]]
                break  # found 3-hop, keep scanning pivots for 2-hop

    return best_depth3


# ── Path utilities ────────────────────────────────────────────────────────────

def _compress_path(path: list) -> list:
    """
    Shorten a path by skipping intermediate nodes when a later page is
    directly linked from an earlier one. Uses cached link data — free.
    """
    if len(path) <= 2:
        return path
    compact = [path[0]]
    i = 0
    while i < len(path) - 1:
        cur = path[i]
        furthest = i + 1
        try:
            out = {l["url"].rstrip("/").lower() for l in get_links(cur)}
            for j in range(len(path) - 1, i + 1, -1):
                if path[j].rstrip("/").lower() in out:
                    furthest = j
                    break
        except Exception:
            pass
        compact.append(path[furthest])
        i = furthest
    return compact


def _build_link_counts(path: list) -> list:
    counts = []
    for url in path[:-1]:
        try:
            counts.append(len(get_links(url)))
        except Exception:
            break
    return counts


def _finalize_success(pair: dict, path: list, llm_calls: int,
                      target_url: str, target: str) -> dict:
    if _at_target_title(path[-1], target) and not _at_target_url(path[-1], target_url):
        path[-1] = target_url
    path        = _compress_path(path)
    link_counts = _build_link_counts(path)
    return _result(pair, path, llm_calls, link_counts, True)


def _two_hop_upgrade(pair: dict, deadline: float) -> list | None:
    """
    Spare-time BFS: try start → pivot → target using cached links.
    Only called after all pairs have been solved, on the worst-step results.
    """
    if time.time() > deadline - _STOP_BUFFER - 1.5:
        return None

    start_url  = pair["start"]
    target_url = pair["target"]
    target     = _title(target_url)
    diff       = pair.get("difficulty", "medium")
    sweep_n    = _IMPROVE_SWEEP.get(diff, 0)
    if sweep_n <= 0:
        return None

    try:
        start_links = get_links(start_url)
    except Exception:
        return None

    direct = _find_direct(start_links, target, target_url)
    if direct:
        return [start_url, direct["url"]]

    lookahead_words = _expand_target_words(target, _kwords(target))
    pivots, _ = _ranked_links(
        start_links, lookahead_words, {start_url},
        min(sweep_n, len(start_links)),
    )

    for pivot in pivots:
        if time.time() > deadline - _STOP_BUFFER - 1.0:
            break
        try:
            links = get_links(pivot["url"])
        except Exception:
            continue
        hit = _find_direct(links, target, target_url)
        if hit:
            return [start_url, pivot["url"], hit["url"]]

    return None


# ── Result helpers ────────────────────────────────────────────────────────────

def _fail(pair: dict) -> dict:
    return {"pair_id": pair["pair_id"], "path": [pair["start"]],
            "steps": 0, "llm_calls": 0, "success": False, "link_counts": []}


def _result(pair, path, llm_calls, link_counts, success):
    return {"pair_id": pair["pair_id"], "path": path,
            "steps": len(path) - 1, "llm_calls": llm_calls,
            "success": success, "link_counts": link_counts}


# ── Single-pair solver ────────────────────────────────────────────────────────

def _solve_one(pair: dict, deadline: float,
               pair_llm_budget: int, global_budget: dict) -> dict:
    """
    Navigate from pair['start'] to pair['target'].

    Step order:
      BFS  : 1-3 hop guaranteed path (free, cached wiki calls)
      0    : Already at target?
      1    : Direct link on current page
      2    : High-confidence keyword score (all target words matched)
      3    : LLM pick (hub_mode when score=0 — domain-escape prompt)
      4    : Heuristic fallback (top of ranked list)
    """
    target_url   = pair["target"]
    target       = _title(target_url)
    target_words = _kwords(target)
    # Expand vocabulary for scoring (domain hints, not pair-specific)
    search_words = _expand_target_words(target, target_words)

    current     = pair["start"]
    path        = [current]
    link_counts = []
    llm_calls   = 0
    visited     = {current}

    # BFS lookahead — find short guaranteed paths before main greedy loop
    bfs_path = _bfs_lookahead(
        current, target_url, target, target_words,
        deadline, pair.get("difficulty", "medium"),
    )
    if bfs_path and len(bfs_path) >= 2:
        return _finalize_success(pair, bfs_path, 0, target_url, target)

    for step in range(_MAX_STEPS):
        if time.time() > deadline - _STOP_BUFFER:
            break

        # Step 0: already there?
        if _at_target(current, target_url, target):
            return _finalize_success(pair, path, llm_calls, target_url, target)

        # Fetch links
        try:
            links = get_links(current)
        except Exception:
            break
        link_counts.append(len(links))

        # Step 1: direct link
        direct = _find_direct(links, target, target_url)
        if direct:
            path.append(direct["url"])
            return _finalize_success(pair, path, llm_calls, target_url, target)

        # Step 2: score and rank using expanded vocabulary
        top_links, best_score = _ranked_links(links, search_words, visited, _LINKS_TO_LLM)
        if not top_links:
            break

        # Step 3: high-confidence skip — only when ALL target words are matched
        # by the chosen top link. This must use exact target_words (not search_words).
        confidence_threshold = max(1, len(target_words))
        chosen_words = _kwords(top_links[0]["text"] + " " + _title(top_links[0]["url"]))
        exact_score = len(chosen_words & target_words)
        if step > 0 and exact_score >= confidence_threshold:
            chosen = top_links[0]

        # Step 4: LLM pick
        elif (llm_calls < pair_llm_budget
              and global_budget["remaining"] > 0
              and time.time() < deadline - 4):
            chosen, calls = _llm_pick(
                top_links, _title(current), target,
                hub_mode=(best_score == 0),
            )
            llm_calls                 += calls
            global_budget["remaining"] -= calls

        # Step 5: heuristic fallback
        else:
            chosen = top_links[0]

        visited.add(chosen["url"])
        current = chosen["url"]
        path.append(current)

    success = _at_target(current, target_url, target)
    if success:
        return _finalize_success(pair, path, llm_calls, target_url, target)
    return _result(pair, path, llm_calls, link_counts, success)


# ── Entry point ───────────────────────────────────────────────────────────────

def solve_all(pairs: list, deadline: float) -> list:
    """
    Solve all pairs within the shared 120-second deadline.

    Strategy:
    - Easy pairs first → maximise completions before time runs out
    - Global LLM cap → survives 20-pair batches
    - Per-pair LLM budget proportional to difficulty and remaining budget
    - Hard deadline guard → never start a pair we cannot finish
    - Spare-time BFS upgrade → shorten step counts on solved pairs
    """
    n             = len(pairs)
    total_llm     = min(_GLOBAL_LLM_CAP, max(n * 2, n + 12))
    global_budget = {"remaining": total_llm}

    order = sorted(
        range(n),
        key=lambda i: (_DIFF_ORDER.get(pairs[i].get("difficulty", "medium"), 1), i),
    )

    result_map: dict = {}

    for pos, idx in enumerate(order):
        pair      = pairs[idx]
        remaining = deadline - time.time()
        if remaining < _MIN_PAIR_TIME:
            break

        diff        = pair.get("difficulty", "medium")
        pairs_left  = n - pos
        fair_share  = max(1, global_budget["remaining"] // max(pairs_left, 1))
        pair_budget = min(_DIFF_LLM_BUDGET.get(diff, 3), fair_share + 1)

        try:
            result = _solve_one(pair, deadline, pair_budget, global_budget)
        except Exception:
            result = _fail(pair)

        result_map[pair["pair_id"]] = result

    # Spare-time improvement pass: try to shorten step counts on solved pairs.
    # Medium-before-hard order: medium 2-hop wins are cheaper to find.
    if time.time() < deadline - _IMPROVE_MIN_REMAIN:
        candidates = [
            (rec["steps"], _DIFF_ORDER.get(pair.get("difficulty", "medium"), 1), pair, rec)
            for pair in pairs
            if (rec := result_map.get(pair["pair_id"]))
            and rec.get("success")
            and rec.get("steps", 0) >= 3
            and pair.get("difficulty", "medium") != "easy"
        ]
        candidates.sort(key=lambda x: (-x[0], x[1]))  # worst steps first, medium before hard

        improvements = 0
        for _, _, pair, rec in candidates[:3]:
            if time.time() > deadline - _STOP_BUFFER - 2.0:
                break
            new_path = _two_hop_upgrade(pair, deadline)
            if new_path and len(new_path) - 1 < rec.get("steps", 99):
                result_map[pair["pair_id"]] = _finalize_success(
                    pair, new_path, rec.get("llm_calls", 0),
                    pair["target"], _title(pair["target"]),
                )
                improvements += 1
                if improvements >= 2:
                    break

    return [result_map.get(p["pair_id"], _fail(p)) for p in pairs]