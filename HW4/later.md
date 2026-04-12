# Project Progress and Analysis (Internal Notes)

## 1. Current Agent Strategy (What has been done)
### Initial Approach (v1/v2)
The agent implemented a "Plan-then-Follow" architecture:
- **One-Shot Route Planning**: Sent a sample of links to Gemini at step 0 to generate a 5-step ideal path.
- **Greedy Plan Following**: Used string matching to follow the plan without further LLM calls.
- **LLM Fallback**: One LLM call to pick the best next link when the plan failed.

### Revised Approach (v3 - Current)
The agent transitioned to a "Dynamic Selection" architecture to eliminate hallucinations and speed up execution:
- **No Planning**: Removed multi-step planning. The LLM now picks from the ACTUAL links visible on the current page.
- **Shortest Title Heuristic**: When LLM budget is exhausted, it picks links with the shortest titles (acting as proxies for broad "hub" articles like "History" or "Europe") rather than wandering randomly.
- **Confidence Gating**: Skips the LLM entirely if a link has high keyword overlap with the target.
- **Global Budget Management & Fast Prompts**: Capped LLM calls to 35 globally and reduced prompt size to achieve ~2s response times, ensuring survival on 20-pair test sets.

## 2. Where the Agent Breaks (Observed Failures)
### Initial Failures (v1/v2 - Score: 28/150)
- **The "Rabbit Hole" Effect**: Blind fallbacks led to obscure pages (*"Akihiro Kanamori"*).
- **Plan Hallucination**: The LLM invented intermediate articles that didn't exist on the page.

### Current Failures (v3 - Score: 65/150)
While significantly better (8/10 reached), the agent still failed on 2 pairs due to two specific edge cases:
1. **The "Single Letter" Rabbit Hole (Failed: Telescope -> Renaissance)**
   - *The Issue*: The "Shortest Title" heuristic for picking hubs backfired. When the budget ran out, the code picked the link with the absolute shortest text (`len(link["text"])`).
   - *The Result*: It picked 1-character articles like numbers (`0`, `6`) or obscure letters (`Ǝ`, `E`), getting permanently stuck on disambiguation pages or dead ends.
   - *The Fix*: Penalize titles shorter than 4 characters so it picks actual words (e.g., "Art") as hubs.
2. **The Confidence Skip is Broken (Failed: Photosynthesis -> World War II)**
   - *The Issue*: `_CONFIDENCE_SKIP` was hardcoded to `4`. For 1-word targets (e.g., "Renaissance"), it could never trigger. For multi-word targets, a partial match wasn't handled dynamically.
   - *The Result*: Wasted LLM calls on single-word targets, and incorrect blind jumps when out of budget.
   - *The Fix*: Make the confidence skip dynamic (`skip if best_score == len(target_words)`).

## 3. Reflection Insights (For 30% Grade)

### Q1: Where did the agent break down, and what did it reveal about LLM reasoning?
- **The Breakdown**: Initially, the agent failed because it assumed plans made by the LLM were executable. In v3, it broke down on edge cases where our programmatic heuristics (shortest title) clashed with Wikipedia's actual structure (single-character articles).
- **LLM Insight**: LLMs are excellent at **Abstract Logic** (knowing how concepts relate) but poor at **Spatial/Contextual Constraints** (knowing if a link actually exists on a screen). By forcing the LLM to choose from a numbered list of *real* links (v3), we bypassed its tendency to "hallucinate connectivity" based on its internal knowledge graph.

### Q2: Principle for next time?
- **"Dynamic Correction over Static Planning"**: In a changing environment (like a web graph), a static plan made at the start is a liability.
- **Application**: Ground the LLM in reality. Instead of asking "What is the best path?", ask "Which of these 40 actual links is the best next step?". Constraining the LLM's output to a strict choice from provided context is far more reliable than open-ended planning.

## 4. Current Status Summary
- **Successes**: 8/10 (Passed all Easy and Hard pairs).
- **Time**: 87s / 120s (Very safe buffer for larger sets).
- **Score**: 65 points (43.3% - massive improvement from 18.7%).