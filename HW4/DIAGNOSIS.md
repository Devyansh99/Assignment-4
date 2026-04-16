# Wikipedia Navigation Agent - Performance Diagnosis

## Summary
**Current Score: 65/150 (43.3%) | Reached: 8/10 pairs**

The agent successfully reaches 80% of targets but loses significant points due to inefficient paths and two critical algorithmic failures. The code shows evidence of v4 design goals in comments but implements v3 logic, missing critical bug fixes.

---

## 🔴 Critical Failures (2 pairs - 30 points lost)

### Failure 1: Pair 4 - "Photosynthesis → World War II"
**Status:** Target NOT reached (ended at: DNA polymerase II)  
**Root Cause:** Hardcoded confidence threshold too strict for short targets

```
Target keywords: {world, war} (only 2 keywords)
_CONFIDENCE_SKIP = 4 (too high!)
→ Algorithm never reaches confidence threshold
→ Keeps searching with LLM until max_steps = 12
→ Fails to find the target
```

**Why it fails:**
- "World War II" has only 2 content words
- The algorithm requires 4 keyword matches to skip LLM (hardcoded)
- But it can never reach 4 matches because the target only HAS 2 keywords!
- The LLM budget exhausts before finding the target

**Code location:** Line ~242
```python
if best_score >= _CONFIDENCE_SKIP:  # _CONFIDENCE_SKIP = 4
    chosen = top_links[0]
```

---

### Failure 2: Pair 9 - "Telescope → Renaissance"  
**Status:** Target NOT reached (ended at: Ǝ)  
**Root Cause:** Hub heuristic allows garbage articles to become traps

```
Target keywords: {telescope, renaissance}
When keyword score = 0 (no matches):
→ Algorithm picks shortest title to find "hub" articles
→ "Ǝ" (1 character) ranks higher than legitimate articles
→ Gets stuck in dead-end single-character Wikipedia article
```

**Why it fails:**
- Wikipedia has many single-character, special-character articles (Ǝ, €, E, II)
- These are mathematical/symbol notation pages, not navigation hubs
- Current tie-breaker: `tie = len(link["text"])` treats "Ǝ" (len=1) as excellent hub
- Algorithm cannot escape - it's a dead end

**Code location:** Line ~195
```python
tie = len(link["text"])  # No penalty for short titles!
scored.sort(key=lambda x: (-x[0], x[1], x[2]))  # Ǝ ranks first when score=0
```

---

## 🟡 Inefficiency Issues (Solutions found but paths too long)

### Problem: Overly Long Paths Waste Steps

| Pair | Start | Target | Steps | Optimal? | Example Path |
|------|-------|--------|-------|----------|--|
| 1 | Cricket | Industrial Revolution | 9 | ❌ | Cricket → Golf → BBC → IT → IBM → Dow Jones → Iranian Revolution → On Revolution → Revolution → Industrial Revolution |
| 5 | Chocolate | Roman law | 11 | ❌ | Chocolate → Fat → Oil → Wax → PH → Lye → Tin → Air → Game law → Land law → Law → Roman law |
| 7 | Jazz | Mathematics | 12 | ❌ (MAX!) | Jazz → Ska → Jwé → Kaiso → Jig → Feis → Kilt → Dye → EU → € → E → Latin letters in math → Mathematics |

**Why this happens:**
1. **Weak initial hops**: Cricket → Golf → BBC is already bad (no semantic connection)
2. **Keyword matching breaks down**: Once you're far from target, keyword overlap becomes random
3. **Hub heuristic is too dumb**: Picking "shortest title" when no keywords match leads to noise articles (€, E, II, etc.)
4. **LLM is underused**: High `_CONFIDENCE_SKIP = 4` prevents LLM calls on long-keyword targets

---

## 🐛 Root Causes: 3 Design Issues

### Issue 1: Hardcoded Confidence Threshold (Line 242)
```python
_CONFIDENCE_SKIP = 4  # ❌ WRONG FOR SHORT TARGETS

if best_score >= _CONFIDENCE_SKIP:
    chosen = top_links[0]
```

**Problem:** 
- Single-keyword targets (e.g., "E", "Om") only need score >= 1
- Two-keyword targets (e.g., "World War II") only need score >= 2
- But code demands score >= 4 for ALL targets
- Targets with ≤2 keywords NEVER trigger high-confidence skip
- Causes unnecessary LLM budget waste and max-step exhaustion

**The Fix (mentioned in your v4 comments but not implemented):**
```python
confidence_threshold = max(1, len(target_words))
if best_score >= confidence_threshold:
    chosen = top_links[0]
```

---

### Issue 2: No Penalty for Garbage Articles (Line 195)
```python
tie = len(link["text"])  # ❌ Treats "Ǝ" same as hub articles

# When best_score = 0, "Ǝ" (len=1) ranks FIRST
# Real hubs like "Europe" (len=6) rank AFTER garbage
scored.sort(key=lambda x: (-x[0], x[1], x[2]))
```

**Problem:**
- Wikipedia has ~200+ single-character articles (mathematical symbols, abbreviations)
- These are dead-ends: no semantic content, random links
- Current algorithm ranks them as "best hubs" when no keywords match
- Agent gets trapped and can't escape

**The Fix (mentioned in your v4 comments but not implemented):**
```python
# Penalize titles < 4 chars by pushing them to back
tie = len(link["text"]) if len(link["text"]) >= 4 else 1000 + len(link["text"])

# Now "Ǝ" gets tie=1001, "E" gets tie=1001, but "Europe" gets tie=6
scored.sort(key=lambda x: (-x[0], x[1], x[2]))
```

---

### Issue 3: Mismatch Between Design Goals and Implementation

Your v4 comments describe these exact fixes:
```python
"""
PROBLEM A — Single-letter rabbit holes (Ǝ, €, E)
FIX: _ranked_links penalises titles shorter than 4 chars...
PROBLEM B — Confidence skip broken for short targets  
FIX: Removed fixed _CONFIDENCE_SKIP = 4. Threshold is now dynamic:
     max(1, len(target_words))
"""
```

But the actual code is **v3**, which still has both bugs!

---

## 📊 Expected Improvement from Fixes

| Fix | Impact | Pairs Fixed | Points Gained |
|-----|--------|-------------|---|
| **Dynamic confidence threshold** | Allows short-keyword targets to use high-confidence skip; fewer wasted LLM calls | Pair 4 | +15 |
| **Penalize short titles** | Prevents garbage articles from being selected; escapes dead-ends | Pair 9 | +15 |
| **More LLM budget for hard pairs** | Hard pairs (7, 8, 10) currently use only 2-4 LLM calls; could find better paths | Pairs 7, 8 | +10-15 |
| **All three combined** | Better path quality overall | All pairs | **+40-45 → 105-110/150 (70-73%)** |

---

## 🔧 Recommended Priority Fixes

### Priority 1: CRITICAL (Fixes failures)
1. **Implement dynamic confidence threshold**
   ```python
   confidence_threshold = max(1, len(target_words))
   if best_score >= confidence_threshold:
       chosen = top_links[0]
   ```
   - Fixes Pair 4 (Photosynthesis → World War II)
   - ~2 minute implementation

2. **Penalize garbage articles in ranking**
   ```python
   tie = len(link["text"]) if len(link["text"]) >= 4 else 1000 + len(link["text"])
   ```
   - Fixes Pair 9 (Telescope → Renaissance)
   - ~1 minute implementation

### Priority 2: HIGH (Improves efficiency)
3. **Increase LLM budget for hard pairs**
   ```python
   _DIFF_LLM_BUDGET = {"easy": 2, "medium": 3, "hard": 5}  # was 4, now 5
   ```
   - Helps Pairs 7, 8 find shorter paths
   - ~30 seconds implementation

4. **Require minimum keyword match before hub jump**
   ```python
   # Don't accept score=0 hub hops; require at least 1 keyword match
   if best_score >= 1 or (len(visited) < 2):
       chosen = top_links[0]
   else:
       # Fall back to LLM if no match
   ```
   - Prevents Cricket → Golf → BBC bad hops
   - ~3 minutes implementation

### Priority 3: OPTIMIZATION (Polish)
5. **Reduce _MAX_STEPS from 12 to 10 or 8**
   - Encourages early LLM use instead of wandering
   - Trade: Might fail on very hard targets, but could improve avg steps

6. **Smarter LLM prompt**
   - Current: "Best link number?"
   - Improved: "Which link gets closest to: World War II? (historical, war, military, conflict)"
   - Helps LLM understand semantic goal, not just pick any link

---

## 📈 Benchmark Comparison

**Current (v3):**
- Score: 65/150 (43.3%)
- Reached: 8/10
- Avg steps: 6.9
- LLM calls: ~15 total

**With Fixes 1+2:**
- Score: ~95/150 (63%)  ← +30 pts
- Reached: 10/10
- Avg steps: ~7.5
- LLM calls: ~18 total

**With Fixes 1+2+3+4:**
- Score: ~105-110/150 (70-73%)  ← +40-45 pts
- Reached: 10/10
- Avg steps: ~5.5-6.0  ← Much more efficient
- LLM calls: ~20-22 total

---

## Conclusion

The code is **architecturally sound** but has **two unimplemented bug fixes** that prevent better performance:

1. The dynamic confidence threshold fix prevents failures on short-keyword targets
2. The garbage article penalty prevents dead-end traps

These fixes are mentioned in your v4 design comments but not actually in the v3 code. Implementing them should be straightforward and yield +30 points immediately with minimal effort.
