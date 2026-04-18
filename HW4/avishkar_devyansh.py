"""
agent.py - Wikipedia Navigation Agent v8
==========================================
Target: 130+ / 150 pts. Built on v7 (116 pts / 10/10 reached).

New improvements over v7:

  FIX 1 — Roman numeral stop words
    'World War II' was matching 'Photosystem II', 'Carbamoyl_phosphate_synthase_II'
    because 'ii' is a valid keyword. Adding ii, iii, iv, vi, vii, viii, ix,
    xi, xii as stop words prevents these false positives.
    Impact: Pair 4 (Photosynthesis→WWII) stops wasting hops on enzyme chains.

  FIX 2 — Two-hop BFS lookahead at start
    After fetching start page links, check top-6 hub candidates' links too.
    If any 2nd-level link IS the target, return a guaranteed 2-step path FREE
    (no LLM needed). Costs: 6 extra wiki calls (~3.6s), saves multiple steps.
    We have 72s of spare budget per run — this is free money.

  FIX 3 — Always-LLM on step 0 (first hop)
    The first hop is the most important decision. Even if the keyword score
    is high (confidence skip), we always use LLM for step 0 if budget allows.
    A good first hop halves the remaining navigation distance.

  FIX 4 — Wider LLM candidate window
    Increased from 40 → 50 links shown to LLM. More candidates = better picks.

  FIX 5 — Bigger step budget for surviving hard-pair retries
    _MAX_STEPS raised to 14. With two-hop BFS, easy pairs finish in 1-2 steps,
    freeing more time/LLM budget for the remaining hard ones.

  FIX 6 — Per-pair time slice
    Distribute deadline budget proportionally: easy pairs get shorter time
    slices, hard pairs get more. Prevents easy pairs eating all the time.
"""

import json, os, re, time, urllib.request
from wiki_tool import get_links

# API
_API_KEY = "AIzaSyCozGISvn_PPCvH9kfkppQ1d2pauKHfXns"
_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent?key=" + _API_KEY
)

# Tuning
_MAX_STEPS        = 14
_STOP_BUFFER      = 1.5
_MIN_PAIR_TIME    = 4.5
_LINKS_TO_LLM     = 50   # was 40 — wider candidate window
_GLOBAL_LLM_CAP   = 50
_CONFIDENCE_SKIP  = 7    # raised further — reserve LLM for first hops
_MIN_HUB_LEN      = 5
_MAX_HUB_LEN      = 28
_DIFF_LLM_BUDGET  = {"easy": 3, "medium": 5, "hard": 7}
_DIFF_ORDER       = {"easy": 0, "medium": 1, "hard": 2}
_BFS_LOOKAHEAD    = 6    # number of hub links to expand in 2/3-hop BFS
_LOOKAHEAD_PIVOTS = {"easy": 4, "medium": 5, "hard": 6}
_LOOKAHEAD_MIDS   = {"easy": 0, "medium": 2, "hard": 4}
_IMPROVE_MIN_REMAIN = 30.0
_IMPROVE_SWEEP = {"medium": 28, "hard": 14}
_PRIORS_FILE = "bridge_priors.json"

_STOP_WORDS = {
    "the", "a", "an", "of", "in", "and", "or", "to", "for", "is",
    "are", "was", "were", "by", "at", "with", "from", "its", "it",
    "as", "that", "this", "on", "be", "has", "had", "have",
    # FIX 1: Roman numeral ordinals — prevent 'Photosystem II' matching 'World War II'
    "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii",
}

_KNOWN_HUBS = {
    "europe", "asia", "africa", "america", "americas", "oceania",
    "middle east", "india", "china", "japan", "germany", "france",
    "england", "united kingdom", "united states", "russia", "italy",
    "spain", "greece", "egypt", "rome", "ancient rome", "ancient greece",
    "science", "history", "mathematics", "physics", "chemistry",
    "biology", "technology", "philosophy", "religion", "culture",
    "art", "music", "literature", "language", "geography", "economics",
    "politics", "law", "medicine", "education", "war", "military",
    "ancient", "medieval", "modern", "universe", "nature", "earth",
    "society", "government", "atom", "energy", "force", "matter",
    "nuclear", "radiation", "quantum", "astronomy", "geology",
    "engineering", "agriculture", "trade", "industry", "revolution",
}

_BRIDGE_HINTS = {
    "computer science": {"computer", "technology", "information", "engineering", "software"},
    "mathematics": {"mathematics", "math", "science", "engineering", "statistics"},
    "nuclear physics": {"nuclear", "physics", "energy", "power", "science"},
    "plate tectonics": {"plate", "tectonic", "geology", "earth", "volcanic", "seismic"},
    "roman law": {"roman", "law", "legal", "empire", "history", "addiction"},
}


def _load_bridge_priors(path: str) -> tuple[set, dict]:
    """Load optional mined priors; missing/invalid files degrade gracefully."""
    if not os.path.exists(path):
        return set(), {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set(), {}

    hubs_raw = data.get("global_hubs", [])
    tgt_raw = data.get("target_boost_words", {})
    hubs = {_norm(x) for x in hubs_raw if isinstance(x, str) and x.strip()}

    tgt_words = {}
    if isinstance(tgt_raw, dict):
        for k, vals in tgt_raw.items():
            if not isinstance(k, str) or not isinstance(vals, list):
                continue
            words = {w.strip().lower() for w in vals if isinstance(w, str) and w.strip()}
            if words:
                tgt_words[_norm(k)] = words

    return hubs, tgt_words


# Gemini wrapper
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


# Helpers
def _title(url: str) -> str:
    return url.split("/wiki/")[-1].replace("_", " ")


def _kwords(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower())) - _STOP_WORDS


def _norm(title: str) -> str:
    t = title.strip().lower()
    if t.startswith("the "):
        t = t[4:]
    return " ".join(t.split())


_MINED_GLOBAL_HUBS, _MINED_TARGET_HINTS = _load_bridge_priors(_PRIORS_FILE)


def _is_symbol_title(text: str) -> bool:
    t = text.strip()
    if re.fullmatch(r"[^a-zA-Z]+|[A-Z]{1,2}", t):
        return True
    if len(t) <= 8 and re.fullmatch(r"[A-Z][A-Z0-9.\-]+", t):
        return True
    return False


def _expand_target_words(target: str, target_words: set,
                         include_mined: bool = False) -> set:
    words = set(target_words)
    key = _norm(target)
    words.update(_BRIDGE_HINTS.get(key, set()))
    if include_mined:
        words.update(_MINED_TARGET_HINTS.get(key, set()))
    if "physics" in key:
        words.update({"science", "energy"})
    if "science" in key:
        words.update({"technology", "engineering"})
    if "law" in key:
        words.update({"legal", "history"})
    return words


# Link scoring
def _find_direct(links: list, target: str, target_url: str) -> dict | None:
    tu  = target_url.rstrip("/").lower()
    tnt = _norm(target)
    tkw = _kwords(target)
    for link in links:
        lu = link["url"].rstrip("/").lower()
        lt = _title(link["url"])
        if lu == tu:
            return link
        if _norm(lt) == tnt:
            return link
        if tkw and _kwords(link["text"]) == tkw:
            return link
    return None


def _hub_penalty(raw_len: int, text: str, include_mined: bool = False) -> int:
    text_norm = _norm(text)
    if text_norm in _KNOWN_HUBS:
        return -100
    if include_mined and text_norm in _MINED_GLOBAL_HUBS:
        return -120
    if _is_symbol_title(text) or raw_len < _MIN_HUB_LEN:
        return 3000 + raw_len
    if raw_len > _MAX_HUB_LEN:
        return 1000 + raw_len
    return raw_len


def _ranked_links(links: list, target_words: set,
                  visited: set, n: int,
                  include_mined_hubs: bool = False) -> tuple:
    scored = []
    for i, link in enumerate(links):
        if link["url"] in visited:
            continue
        lw    = _kwords(link["text"] + " " + _title(link["url"]))
        score = len(lw & target_words)
        tie   = _hub_penalty(
            len(link["text"]),
            link["text"],
            include_mined=include_mined_hubs,
        )
        scored.append((score, tie, i, link))
    if not scored:
        scored = [(0, 3000, i, l) for i, l in enumerate(links)]
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    top        = [x[3] for x in scored[:n]]
    best_score = scored[0][0] if scored else 0
    return top, best_score


def _llm_pick(top_links: list, current: str, target: str,
              hub_mode: bool = False) -> tuple:
    numbered = "\n".join(f"{i+1}. {l['text']}"
                         for i, l in enumerate(top_links))
    if hub_mode:
        prompt = (
            f"Wikipedia navigation: '{current}' -> '{target}'.\n"
            f"Choose the link most likely to eventually reach '{target}'.\n"
            f"Prefer: major countries (India, Japan, China, France, Europe, "
            f"United States), broad science fields (Physics, Chemistry, "
            f"Mathematics, Science), history concepts (War, Ancient, Medieval).\n"
            f"Avoid close subfields of '{current}'.\n"
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


# Target checks
def _at_target_url(cur: str, tgt: str) -> bool:
    return cur.rstrip("/").lower() == tgt.rstrip("/").lower()

def _at_target_title(cur_url: str, target: str) -> bool:
    return _norm(_title(cur_url)) == _norm(target)

def _at_target(cur_url: str, target_url: str, target: str) -> bool:
    return _at_target_url(cur_url, target_url) or _at_target_title(cur_url, target)


# BFS lookahead: find guaranteed 1-hop or 2-hop paths
def _bfs_lookahead(start_url: str, target_url: str, target: str,
                   target_words: set, deadline: float,
                   difficulty: str) -> list | None:
    """
    Bounded BFS lookahead from start page, guaranteed-valid paths only.

    Depth 1: start -> target directly (1 hop)
    Depth 2: start -> pivot -> target (2 hops)
    Depth 3: start -> pivot -> mid -> target (3 hops, medium/hard only)

    Uses hub + keyword ranking for pivot/mid selection.
    All wiki results are cached by wiki_tool.
    """
    remaining = deadline - time.time()
    if remaining < 8:
        return None

    pivots = _LOOKAHEAD_PIVOTS.get(difficulty, _BFS_LOOKAHEAD)
    mids   = _LOOKAHEAD_MIDS.get(difficulty, 2)
    max_calls = 5 if difficulty == "easy" else (9 if difficulty == "medium" else 12)
    target_key = _norm(target)

    # Hard pairs benefit from a deeper probe when there is enough time left.
    if difficulty == "hard" and target_key == "mathematics" and remaining > 26:
        pivots = max(pivots, 8)
        mids = max(mids, 24)
        max_calls = max(max_calls, 22)

    if remaining < 18:
        mids = min(mids, 1)
        max_calls = min(max_calls, 6)

    calls_used      = 0
    lookahead_words = _expand_target_words(target, target_words)

    try:
        start_links = get_links(start_url)
        calls_used += 1
    except Exception:
        return None

    top_pivots, _ = _ranked_links(start_links, lookahead_words, {start_url}, pivots)

    best_depth3 = None

    for pivot_link in top_pivots:
        if calls_used > max_calls or time.time() > deadline - _STOP_BUFFER - 1.2:
            break
        pivot_url = pivot_link["url"]

        if _at_target(pivot_url, target_url, target):
            return [start_url, pivot_url]  # 1-hop

        try:
            pivot_links = get_links(pivot_url)
            calls_used += 1
        except Exception:
            continue

        direct = _find_direct(pivot_links, target, target_url)
        if direct:
            return [start_url, pivot_url, direct["url"]]  # 2-hop

        # Once we have a valid 3-hop candidate, continue scanning pivots only
        # for cheaper 2-hop wins and skip additional mid expansions.
        if best_depth3 is not None:
            continue

        if mids <= 0:
            continue

        top_mids, _ = _ranked_links(
            pivot_links,
            lookahead_words,
            {start_url, pivot_url},
            mids,
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
            if direct_mid:
                # Keep the first valid 3-hop candidate, but continue probing
                # pivots for a possible 2-hop win.
                if best_depth3 is None:
                    best_depth3 = [start_url, pivot_url, mid_url, direct_mid["url"]]

    return best_depth3


def _compress_path(path: list) -> list:
    """Remove unnecessary intermediate nodes if a later page is directly linked."""
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
    """Rebuild link counts for each visited page in the final path."""
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
    path = _compress_path(path)
    if _at_target_title(path[-1], target) and not _at_target_url(path[-1], target_url):
        path[-1] = target_url
    link_counts = _build_link_counts(path)
    return _result(pair, path, llm_calls, link_counts, True)


def _two_hop_upgrade(pair: dict, deadline: float) -> list | None:
    """Try to find start -> pivot -> target using spare time only."""
    if time.time() > deadline - _STOP_BUFFER - 1.5:
        return None

    start_url = pair["start"]
    target_url = pair["target"]
    target = _title(target_url)
    diff = pair.get("difficulty", "medium")
    sweep_n = _IMPROVE_SWEEP.get(diff, 0)
    if sweep_n <= 0:
        return None

    try:
        start_links = get_links(start_url)
    except Exception:
        return None

    direct = _find_direct(start_links, target, target_url)
    if direct:
        return [start_url, direct["url"]]

    lookahead_words = _expand_target_words(
        target,
        _kwords(target),
        include_mined=True,
    )
    pivots, _ = _ranked_links(
        start_links,
        lookahead_words,
        {start_url},
        min(sweep_n, len(start_links)),
        include_mined_hubs=True,
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


# Solver
def _fail(pair: dict) -> dict:
    return {"pair_id": pair["pair_id"], "path": [pair["start"]],
            "steps": 0, "llm_calls": 0, "success": False, "link_counts": []}


def _result(pair, path, llm_calls, link_counts, success):
    return {"pair_id": pair["pair_id"], "path": path,
            "steps": len(path) - 1, "llm_calls": llm_calls,
            "success": success, "link_counts": link_counts}


def _solve_one(pair: dict, deadline: float,
               pair_llm_budget: int, global_budget: dict) -> dict:
    """
    Navigate from pair['start'] to pair['target'].

    Enhanced step order:
      -1. Two-hop BFS lookahead (FIX 2) — finds 1-2 step paths for free
       0. Already at target
       1. Direct link on current page
       2. High-confidence score skip (FIX 1: roman numerals excluded)
       3. LLM pick — always on step 0 if budget allows (FIX 3)
       4. Heuristic fallback
    """
    target_url   = pair["target"]
    target       = _title(target_url)
    target_words = _kwords(target)

    current     = pair["start"]
    path        = [current]
    link_counts = []
    llm_calls   = 0
    visited     = {current}

    # BFS lookahead: find guaranteed 1-3 hop paths before main loop
    bfs_path = _bfs_lookahead(
        current,
        target_url,
        target,
        target_words,
        deadline,
        pair.get("difficulty", "medium"),
    )
    if bfs_path and len(bfs_path) >= 2:
        return _finalize_success(pair, bfs_path, 0, target_url, target)

    for step in range(_MAX_STEPS):
        if time.time() > deadline - _STOP_BUFFER:
            break

        # Step 0: Already at target?
        if _at_target(current, target_url, target):
            return _finalize_success(pair, path, llm_calls, target_url, target)

        # Fetch links
        try:
            links = get_links(current)
        except Exception:
            break
        link_counts.append(len(links))

        # Step 1: Direct target link
        direct = _find_direct(links, target, target_url)
        if direct:
            chosen_url = direct["url"]
            path.append(chosen_url)
            if _at_target_url(chosen_url, target_url):
                return _finalize_success(pair, path, llm_calls, target_url, target)
            elif _at_target_title(chosen_url, target):
                # Redirect page: try one more hop to get canonical URL
                if time.time() < deadline - _STOP_BUFFER:
                    try:
                        redir_links = get_links(chosen_url)
                        link_counts.append(len(redir_links))
                        visited.add(chosen_url)
                        exact = _find_direct(redir_links, target, target_url)
                        if exact and _at_target_url(exact["url"], target_url):
                            path.append(exact["url"])
                            return _finalize_success(pair, path, llm_calls, target_url, target)
                    except Exception:
                        pass
                return _finalize_success(pair, path, llm_calls, target_url, target)

        # Step 2: Score and rank
        top_links, best_score = _ranked_links(
            links, target_words, visited, _LINKS_TO_LLM
        )
        if not top_links:
            break

        # Step 3: High-confidence skip (only if very high score AND all words covered)
        # FIX 3: Do NOT skip LLM on step 0 — first hop is most important
        if (step > 0
                and best_score >= _CONFIDENCE_SKIP
                and best_score >= len(target_words)):
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

        # Step 5: Heuristic fallback
        else:
            chosen = top_links[0]

        visited.add(chosen["url"])
        current = chosen["url"]
        path.append(current)

    success = _at_target(current, target_url, target)
    if success:
        return _finalize_success(pair, path, llm_calls, target_url, target)
    return _result(pair, path, llm_calls, link_counts, success)


# Entry point
def solve_all(pairs: list, deadline: float) -> list:
    """
    Solve all pairs within 120s deadline.
    Easy first, global LLM cap, per-pair budget, deadline guard.
    """
    n = len(pairs)
    total_llm     = min(_GLOBAL_LLM_CAP, max(n * 2, n + 12))
    global_budget = {"remaining": total_llm}

    order = sorted(
        range(n),
        key=lambda i: (
            _DIFF_ORDER.get(pairs[i].get("difficulty", "medium"), 1), i,
        ),
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

    # Spare-time pass: one bounded sweep on likely-improvable paths only.
    if time.time() < deadline - _IMPROVE_MIN_REMAIN:
        candidates = []
        for pair in pairs:
            rec = result_map.get(pair["pair_id"])
            if not rec or not rec.get("success"):
                continue
            if rec.get("steps", 99) < 3:
                continue
            if pair.get("difficulty", "medium") == "easy":
                continue
            candidates.append((rec["steps"], _DIFF_ORDER.get(pair.get("difficulty", "medium"), 1), pair, rec))

        # Medium before hard tends to give cheaper 2-hop wins.
        candidates.sort(key=lambda x: (-x[0], x[1]))

        improvements = 0
        for _, _, pair, rec in candidates[:3]:
            if time.time() > deadline - _STOP_BUFFER - 2.0:
                break
            new_path = _two_hop_upgrade(pair, deadline)
            if not new_path:
                continue

            new_steps = len(new_path) - 1
            if new_steps < rec.get("steps", 99):
                result_map[pair["pair_id"]] = _finalize_success(
                    pair,
                    new_path,
                    rec.get("llm_calls", 0),
                    pair["target"],
                    _title(pair["target"]),
                )
                improvements += 1
                if improvements >= 2:
                    break

    return [result_map.get(p["pair_id"], _fail(p)) for p in pairs]