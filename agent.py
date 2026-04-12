"""
agent.py — Wikipedia Navigation Agent
======================================
Strategy overview:
  1. Direct target detection  — free O(n) string check, no LLM needed
  2. One-shot route planning  — 1 LLM call that maps intermediate hops
  3. Greedy plan following    — pure string matching, zero LLM cost
  4. LLM fallback selection   — only when the plan fails at a step
  5. Time-aware scheduling    — easy pairs first; bail early if budget exhausted

This keeps LLM calls to ~1-2 per pair while still reaching the target
intelligently, which is critical when 20 pairs share a 120-second budget.
"""

import json, os, random, re, time, urllib.request
from wiki_tool import get_links

# ── API setup ─────────────────────────────────────────────────────────────────
with open(os.path.expanduser("~/gemini_api_key.txt")) as f:
    _API_KEY = f.read().strip()

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent?key=" + _API_KEY
)

_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
_MAX_STEPS = 14          # 15+ hops = 0 pts, so never waste time past 14
_STOP_BUFFER = 2.0       # seconds to keep in reserve before deadline
_MIN_TIME_PER_PAIR = 5.0 # don't start a pair with less than this remaining
_MAX_GLOBAL_LLM_CALLS = 45
_PAIR_LLM_BUDGET = {"easy": 1, "medium": 2, "hard": 3}

_LINKS_CACHE: dict[str, list] = {}


# ── Gemini wrapper ────────────────────────────────────────────────────────────

def _ask_gemini(prompt: str, max_tokens: int = 200,
                temperature: float = 0.1, timeout: int = 18) -> str:
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        _GEMINI_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ── URL / title helpers ───────────────────────────────────────────────────────

def _title(url: str) -> str:
    return url.split("/wiki/")[-1].replace("_", " ")


def _norm_title(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(tokens)


def _get_links_cached(url: str) -> list:
    cached = _LINKS_CACHE.get(url)
    if cached is not None:
        return cached
    links = get_links(url)
    _LINKS_CACHE[url] = links
    return links


def _can_spend_llm(global_budget: dict | None) -> bool:
    return global_budget is None or global_budget.get("remaining", 0) > 0


def _spend_llm(global_budget: dict | None, calls: int = 1) -> None:
    if global_budget is not None:
        global_budget["remaining"] = max(0, global_budget.get("remaining", 0) - calls)


# ── Link helpers ──────────────────────────────────────────────────────────────

_STOP_WORDS = {"the", "a", "an", "of", "in", "and", "or", "to", "for",
               "is", "are", "was", "were", "by", "at", "with", "from"}


def _find_exact(links: list, target: str) -> dict | None:
    """Case-insensitive exact match — free, no LLM required."""
    tl = _norm_title(target)
    for link in links:
        if (
            _norm_title(link["text"]) == tl
            or _norm_title(_title(link["url"])) == tl
        ):
            return link
    return None


def _topic_words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOP_WORDS


def _find_fuzzy(links: list, topic: str, visited: set, top_n: int = 5) -> list:
    """Fuzzy word-overlap match against a planned topic, excluding visited."""
    tw = _topic_words(topic)
    if not tw:
        return []
    scored = []
    for link in links:
        if link["url"] in visited:
            continue
        link_words = _topic_words(link["text"] + " " + _title(link["url"]))
        overlap = len(link_words & tw)
        if overlap:
            scored.append((overlap, link))
    scored.sort(key=lambda x: -x[0])
    return [x[1] for x in scored[:top_n]]


def _smart_filter(links: list, target: str, intermediates: list,
                  visited: set, n: int = 50) -> list:
    """
    Score every link by relevance to target + upcoming plan steps,
    return top-n preserving original order (for stable LLM numbering).
    Falls back to all unvisited links if scoring yields nothing.
    """
    tw = _topic_words(target)
    iw: set = set()
    for t in intermediates:
        iw |= _topic_words(t)

    scored = []
    for i, link in enumerate(links):
        if link["url"] in visited:
            continue
        words = _topic_words(link["text"] + " " + _title(link["url"]))
        score = len(words & tw) * 3 + len(words & iw)
        scored.append((score, i, link))

    # If everything was visited, allow revisits (prefer high-score)
    if not scored:
        scored = [(0, i, l) for i, l in enumerate(links)]

    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[:n]
    top.sort(key=lambda x: x[1])     # restore original order
    return [x[2] for x in top]


# ── Route planning (1 LLM call) ───────────────────────────────────────────────

def _plan_route(start: str, target: str, start_links: list) -> list:
    """
    Ask Gemini to plan a chain of intermediate Wikipedia article titles
    bridging `start` → `target`.  Returns a list that ends with `target`.
    If the LLM fails, returns [target] so we still try a direct jump.
    """
    sample = ", ".join(l["text"] for l in start_links[:70])
    prompt = f"""You are playing the Wikipedia navigation game.
Goal: navigate from the article "{start}" to the article "{target}" by following links.

Links currently visible on "{start}" (sample): {sample}

Plan the SHORTEST path. List ONLY the intermediate Wikipedia article titles you
would visit, ending with "{target}". Do NOT include "{start}".
Prefer well-known, highly-connected articles (countries, major events, broad
concepts) as intermediate hops — they have the most outgoing links.
Use real Wikipedia article titles (exact capitalisation isn't critical).

Reply with ONLY a JSON array of strings (max 5 elements), e.g.:
["Hub Article", "Closer Article", "{target}"]"""

    try:
        resp = _ask_gemini(prompt, max_tokens=160, temperature=0.1)
        # Strip markdown fences if present
        clean = resp.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        plan = json.loads(clean)
        if isinstance(plan, list) and len(plan) > 0:
            # Ensure target is in the plan (LLM sometimes omits it)
            result = [str(p).strip() for p in plan if p]
            if not result or _title(target if "/" in target else f"/wiki/{target}").lower() \
                    not in [x.lower() for x in result]:
                result.append(target)
            return result
    except Exception:
        pass
    return [target]


# ── Greedy LLM link selection (fallback) ─────────────────────────────────────

def _llm_pick(links: list, current: str, target: str,
              visited: set, remaining_plan: list) -> tuple:
    """
    Use Gemini to select the single best next link.
    Returns (chosen_link_dict, llm_calls_count).
    """
    plan_hint = ""
    if remaining_plan:
        plan_hint = f"Intermediate goals still ahead: {' → '.join(remaining_plan[:3])}\n"

    filtered = _smart_filter(links, target, remaining_plan[:3], visited, n=50)
    if not filtered:
        filtered = [l for l in links if l["url"] not in visited][:50] or links[:50]

    numbered = "\n".join(f"{i+1}. {l['text']}" for i, l in enumerate(filtered))

    prompt = f"""Wikipedia navigation.
Current article: {current}
Target article:  {target}
{plan_hint}
Available links:
{numbered}

Which link leads most directly toward "{target}"?
Think step by step about what connects "{current}" to "{target}".
Reply with ONLY the number of the best link (nothing else)."""

    try:
        resp = _ask_gemini(prompt, max_tokens=8, temperature=0.1)
        m = re.search(r"\d+", resp)
        choice = int(m.group()) if m else -1
        if 1 <= choice <= len(filtered):
            return filtered[choice - 1], 1
    except Exception:
        pass

    return (filtered[0] if filtered else links[0]), 1


# ── Single-pair solver ────────────────────────────────────────────────────────

def _make_failure(pair: dict) -> dict:
    return {"pair_id": pair["pair_id"], "path": [pair["start"]],
            "steps": 0, "llm_calls": 0, "success": False, "link_counts": []}


def _solve_one(pair: dict, deadline: float, max_llm_calls: int,
               global_llm_budget: dict | None = None) -> dict:
    start_url  = pair["start"]
    target_url = pair["target"]
    target     = _title(target_url)

    path        = [start_url]
    link_counts = []
    llm_calls   = 0
    visited     = {start_url}
    current     = start_url
    plan        = []
    plan_idx    = 0
    first_links = None   # cache links fetched at step 0

    for step in range(_MAX_STEPS):
        if time.time() > deadline - _STOP_BUFFER:
            break

        cur_title = _title(current)

        # ── Success check ──────────────────────────────────────────────────
        if cur_title.lower() == target.lower():
            return {"pair_id": pair["pair_id"], "path": path,
                    "steps": len(path) - 1, "llm_calls": llm_calls,
                    "success": True, "link_counts": link_counts}

        # ── Fetch links ────────────────────────────────────────────────────
        try:
            links = _get_links_cached(current)
        except Exception:
            break
        link_counts.append(len(links))
        if step == 0:
            first_links = links

        # ── 1. Direct target check (free) ──────────────────────────────────
        direct = _find_exact(links, target)
        if direct:
            path.append(direct["url"])
            return {"pair_id": pair["pair_id"], "path": path,
                    "steps": len(path) - 1, "llm_calls": llm_calls,
                    "success": True, "link_counts": link_counts}

        # ── 2. Plan the route once (step 0) ───────────────────────────────
        if step == 0:
            if (
                time.time() < deadline - 8
                and llm_calls < max_llm_calls
                and _can_spend_llm(global_llm_budget)
            ):   # only plan if we have breathing room and LLM budget
                plan = _plan_route(cur_title, target, first_links)
                llm_calls += 1
                _spend_llm(global_llm_budget, 1)
                plan_idx = 0

        # ── 3. Follow the plan greedily (zero LLM cost) ───────────────────
        next_link = None
        while plan_idx < len(plan):
            planned = plan[plan_idx]

            # Exact match first
            cand = _find_exact(links, planned)
            if cand and cand["url"] not in visited:
                next_link = cand
                plan_idx += 1
                break

            # Fuzzy word-overlap match
            fuzzy = _find_fuzzy(links, planned, visited, top_n=3)
            if fuzzy:
                next_link = fuzzy[0]
                plan_idx += 1
                break

            # This plan step is unreachable from here; skip to next
            plan_idx += 1

        # ── 4. LLM fallback if plan is exhausted / stuck ───────────────────
        if next_link is None:
            if time.time() > deadline - 5:
                # No time for an LLM call — pick the highest-relevance unvisited link
                fallback = _smart_filter(links, target, [], visited, n=5)
                next_link = fallback[0] if fallback else (
                    [l for l in links if l["url"] not in visited] or links
                )[0]
            else:
                if llm_calls < max_llm_calls and _can_spend_llm(global_llm_budget):
                    remaining_plan = plan[plan_idx:]
                    next_link, calls = _llm_pick(
                        links, cur_title, target, visited, remaining_plan
                    )
                    llm_calls += calls
                    _spend_llm(global_llm_budget, calls)
                else:
                    fallback = _smart_filter(links, target, plan[plan_idx:], visited, n=5)
                    next_link = fallback[0] if fallback else (
                        [l for l in links if l["url"] not in visited] or links
                    )[0]

        if next_link is None:
            break

        visited.add(next_link["url"])
        current = next_link["url"]
        path.append(current)

    # ── Final success check ────────────────────────────────────────────────
    success = _title(current).lower() == target.lower()
    return {"pair_id": pair["pair_id"], "path": path,
            "steps": len(path) - 1, "llm_calls": llm_calls,
            "success": success, "link_counts": link_counts}


# ── Entry point ───────────────────────────────────────────────────────────────

def solve_all(pairs: list, deadline: float) -> list:
    """
    Solve all pairs within the shared 2-minute deadline.

    Key choices:
    - Easy pairs first → more completions, higher total score
    - Hard deadline check before each pair → never go over time
    - At most ~2 LLM calls per pair (1 plan + 1 fallback) → fits 20-pair budget
    - Cached get_links() → reuse across pairs sharing hub articles
    """
    # Sort: easy → medium → hard; stable within same difficulty
    sorted_pairs = sorted(
        range(len(pairs)),
        key=lambda i: (
            _DIFFICULTY_ORDER.get(pairs[i].get("difficulty", "medium"), 1),
            i,
        ),
    )

    result_map: dict = {}
    global_llm_budget = {
        "remaining": min(_MAX_GLOBAL_LLM_CALLS, max(len(pairs), len(pairs) * 2))
    }

    for position, idx in enumerate(sorted_pairs):
        pair     = pairs[idx]
        remaining = deadline - time.time()

        if remaining < _MIN_TIME_PER_PAIR:
            break   # not enough budget — leave the rest as failures

        remaining_pairs = max(1, len(sorted_pairs) - position)
        avg_left = max(1, global_llm_budget["remaining"] // remaining_pairs)
        diff = pair.get("difficulty", "medium")
        pair_llm_budget = min(_PAIR_LLM_BUDGET.get(diff, 2), avg_left + 1)

        try:
            result = _solve_one(pair, deadline, pair_llm_budget, global_llm_budget)
        except Exception:
            result = _make_failure(pair)

        result_map[pair["pair_id"]] = result

    # Return in original input order; fill failures for unattempted pairs
    return [
        result_map.get(pair["pair_id"], _make_failure(pair))
        for pair in pairs
    ]
