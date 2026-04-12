"""
agent.py — Wikipedia Navigation Agent v3
=========================================
Root-cause fixes from v2:

  PROBLEM 1 — Wandering (blind link[0] fallback)
  FIX: Smart scoring always ranks links. When budget is gone, pick by
       keyword overlap or shortest-title hub heuristic. Never random.

  PROBLEM 2 — Hallucinated plans (LLM invents articles not on the page)
  FIX: Removed multi-step planning entirely. LLM now picks from the
       ACTUAL links visible on the current page — hallucination impossible.

  PROBLEM 3 — Too slow for 20-pair test sets
  FIX: Short one-line prompts (~2s per LLM call), global LLM cap (35),
       and confidence gating (skip LLM when keyword score is obvious).

  PROBLEM 4 — 70-link ceiling for planning
  FIX: Planning is gone. Top-40 scoring is used per step, not globally.

Timing target: ~5-6s per pair on average → survives 20 pairs in 120s.
LLM calls: 1-2 per pair average (capped globally at 35 across all pairs).
"""

import json, os, re, time, urllib.request
from wiki_tool import get_links

# ── API setup ─────────────────────────────────────────────────────────────────
_API_KEY = "AIzaSyCozGISvn_PPCvH9kfkppQ1d2pauKHfXns"

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent?key=" + _API_KEY
)

# ── Tuning constants ──────────────────────────────────────────────────────────
_MAX_STEPS        = 12    # 13+ hops score <= 2 pts; not worth the time
_STOP_BUFFER      = 1.5   # seconds to reserve before hard deadline
_MIN_PAIR_TIME    = 4.0   # skip pair if less than this remains
_CONFIDENCE_SKIP  = 4     # skip LLM when keyword score >= this
_LINKS_TO_LLM     = 40    # candidate links shown per LLM call
_GLOBAL_LLM_CAP   = 35    # max LLM calls across ALL pairs in one run
_DIFF_LLM_BUDGET  = {"easy": 2, "medium": 3, "hard": 4}
_DIFF_ORDER       = {"easy": 0, "medium": 1, "hard": 2}

_STOP_WORDS = {
    "the", "a", "an", "of", "in", "and", "or", "to", "for", "is",
    "are", "was", "were", "by", "at", "with", "from", "its", "it",
    "as", "that", "this", "on", "be", "has", "had", "have",
}


# ── Gemini wrapper ────────────────────────────────────────────────────────────

def _ask_gemini(prompt: str, max_tokens: int = 6, timeout: int = 16) -> str:
    """Call Gemini 2.5 Flash. Short prompts -> faster responses."""
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,        # deterministic = faster & consistent
            "maxOutputTokens": max_tokens,
        },
    }).encode()
    req = urllib.request.Request(
        _GEMINI_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ── URL / text helpers ────────────────────────────────────────────────────────

def _title(url: str) -> str:
    return url.split("/wiki/")[-1].replace("_", " ")


def _kwords(text: str) -> set:
    """Lower-case content words, stop-words removed."""
    return set(re.findall(r"[a-z]+", text.lower())) - _STOP_WORDS


# ── Link scoring & selection ──────────────────────────────────────────────────

def _find_direct(links: list, target: str, target_url: str) -> dict | None:
    """
    O(n) check: is the target page already linked on this page?
    Checks URL equality AND keyword-set equality (handles title variants).
    Free — no LLM needed.
    """
    tu  = target_url.rstrip("/").lower()
    tkw = _kwords(target)

    for link in links:
        # URL match (most reliable — handles redirects)
        if link["url"].rstrip("/").lower() == tu:
            return link
        # Keyword-set match on title
        if tkw and _kwords(link["text"]) == tkw:
            return link
    return None


def _ranked_links(links: list, target_words: set,
                  visited: set, n: int) -> tuple:
    """
    Score every unvisited link by keyword overlap with the target.
    Returns (top_n_list, best_score).

    Tie-breaking: shorter title = broader hub article = better transit node.
    This means when no keywords match (score = 0), we prefer hub articles
    like "United States", "History", "Europe" over obscure articles — these
    have the most outgoing links and are the best waypoints.
    """
    scored = []
    for i, link in enumerate(links):
        if link["url"] in visited:
            continue
        lw    = _kwords(link["text"] + " " + _title(link["url"]))
        score = len(lw & target_words)
        tie   = len(link["text"])          # shorter title = lower tie value = higher rank
        scored.append((score, tie, i, link))

    if not scored:
        # All URLs already visited — allow revisit rather than crashing
        scored = [(0, len(l["text"]), i, l) for i, l in enumerate(links)]

    # Primary sort: highest score. Tiebreak: shortest title, then original order.
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    top        = [x[3] for x in scored[:n]]
    best_score = scored[0][0] if scored else 0
    return top, best_score


def _llm_pick(top_links: list, current: str, target: str) -> tuple:
    """
    Show the LLM a numbered list of REAL on-page links and ask for the
    best one. Hallucination is structurally impossible because we only
    accept a number, then index into the actual list.

    Returns (chosen_link, calls_used).
    Prompt is kept minimal (<100 chars + list) for ~1-2s response time.
    """
    numbered = "\n".join(f"{i+1}. {l['text']}"
                         for i, l in enumerate(top_links))
    prompt = (
        f"Wikipedia: '{current}' -> '{target}'.\n"
        f"{numbered}\n"
        f"Best link number? NUMBER ONLY."
    )
    try:
        resp = _ask_gemini(prompt, max_tokens=6, timeout=16)
        m    = re.search(r"\d+", resp)
        if m:
            idx = int(m.group()) - 1
            if 0 <= idx < len(top_links):
                return top_links[idx], 1
    except Exception:
        pass

    # Graceful fallback to highest-scored (never blind index 0)
    return top_links[0], 1


# ── Single-pair solver ────────────────────────────────────────────────────────

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

    Decision at each step:
      1. Direct match (URL / keyword equality) → FREE, jump immediately
      2. Keyword score >= 4                    → FREE, high-confidence pick
      3. LLM from REAL links                   → ~2s, 1 call
      4. Budget gone, score > 0               → best keyword pick (no LLM)
      5. Budget gone, score = 0               → shortest-title hub heuristic
    """
    target_url   = pair["target"]
    target       = _title(target_url)
    target_words = _kwords(target)

    current     = pair["start"]
    path        = [current]
    link_counts = []
    llm_calls   = 0
    visited     = {current}

    for step in range(_MAX_STEPS):
        if time.time() > deadline - _STOP_BUFFER:
            break

        # ── Already at target? ───────────────────────────────────────────────
        if (current.rstrip("/").lower() == target_url.rstrip("/").lower()
                or _title(current).lower() == target.lower()):
            return _result(pair, path, llm_calls, link_counts, True)

        # ── Fetch links ──────────────────────────────────────────────────────
        try:
            links = get_links(current)
        except Exception:
            break
        link_counts.append(len(links))

        # ── Step 1: Direct target on page (FREE) ─────────────────────────────
        direct = _find_direct(links, target, target_url)
        if direct:
            path.append(direct["url"])
            return _result(pair, path, llm_calls, link_counts, True)

        # ── Step 2: Score and rank candidates ────────────────────────────────
        top_links, best_score = _ranked_links(
            links, target_words, visited, _LINKS_TO_LLM
        )
        if not top_links:
            break

        # ── Step 3: High-confidence keyword pick — skip LLM ──────────────────
        if best_score >= _CONFIDENCE_SKIP:
            chosen = top_links[0]

        # ── Step 4: LLM pick from REAL links (no hallucination) ───────────────
        elif (llm_calls < pair_llm_budget
              and global_budget["remaining"] > 0
              and time.time() < deadline - 4):
            chosen, calls = _llm_pick(top_links, _title(current), target)
            llm_calls                 += calls
            global_budget["remaining"] -= calls

        # ── Step 5: Budget / time exhausted — use best available ──────────────
        else:
            # top_links is already sorted: highest keyword score first,
            # then shortest title (hub proxy). So top_links[0] is always
            # the best non-random choice, even when score = 0.
            chosen = top_links[0]

        visited.add(chosen["url"])
        current = chosen["url"]
        path.append(current)

    success = _title(current).lower() == target.lower()
    return _result(pair, path, llm_calls, link_counts, success)


# ── Entry point ───────────────────────────────────────────────────────────────

def solve_all(pairs: list, deadline: float) -> list:
    """
    Solve all pairs within the shared 120-second deadline.

    Scheduling strategy
    -------------------
    * Easy first  — maximise pairs reached before time runs out
    * Global LLM cap (35) — prevents 20-pair sets from blowing the budget
    * Per-pair LLM budget — scales with difficulty, clamped by fair share
    * Hard deadline guard — never start a pair we cannot meaningfully finish
    """
    n = len(pairs)

    # Scale global LLM budget with batch size, capped at _GLOBAL_LLM_CAP
    total_llm     = min(_GLOBAL_LLM_CAP, max(n * 2, n + 10))
    global_budget = {"remaining": total_llm}

    # Sort: easy -> medium -> hard; stable within same difficulty
    order = sorted(
        range(n),
        key=lambda i: (
            _DIFF_ORDER.get(pairs[i].get("difficulty", "medium"), 1),
            i,
        ),
    )

    result_map: dict = {}

    for pos, idx in enumerate(order):
        pair      = pairs[idx]
        remaining = deadline - time.time()

        if remaining < _MIN_PAIR_TIME:
            break                    # rest score as 0 — better than timeout

        diff        = pair.get("difficulty", "medium")
        pairs_left  = n - pos
        # Fair share of remaining LLM calls, +1 so first pair isn't starved
        fair_share  = max(1, global_budget["remaining"] // max(pairs_left, 1))
        pair_budget = min(_DIFF_LLM_BUDGET.get(diff, 3), fair_share + 1)

        try:
            result = _solve_one(pair, deadline, pair_budget, global_budget)
        except Exception:
            result = _fail(pair)

        result_map[pair["pair_id"]] = result

    # Return in original input order; unattempted pairs become failures
    return [result_map.get(p["pair_id"], _fail(p)) for p in pairs]
