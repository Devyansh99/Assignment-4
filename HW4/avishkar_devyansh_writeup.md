# HW4 Writeup — Wikipedia Navigation Agent

## Q1. Where did your agent break down, and what did that reveal about how LLMs reason?

The agent failed most visibly on paths that require **semantic detour** — cases where the optimal route passes through an article that looks *further* from the target by surface meaning, not closer. For example, navigating from "Banjo" to "Treaty of Versailles" ideally goes through a hub like "United States" or "World War I," but Gemini's planning call tended to generate a chain of plausible-sounding intermediate topics (e.g., "Folk Music → American Culture → World War I → Treaty of Versailles") where several of those intermediate articles didn't actually contain links to the next one. The plan *felt* right thematically but failed structurally because Wikipedia's link graph doesn't mirror semantic associations.

A second failure mode appeared with **long link lists**. When a page had 400–500 links and the target required navigating toward a specific era or geography, Gemini would often fixate on the first plausible-sounding keyword match rather than reasoning about graph distance. In one test, from "Roman Empire" trying to reach "Jazz music," the model kept picking links about ancient history when the correct move was to jump to "United States" — a counter-intuitive first step that required thinking two hops ahead.

What this revealed is that LLMs reason by **associative similarity**, not by graph topology. They're very good at knowing what *topics* are related but poor at modeling whether a link *actually exists* on a given Wikipedia page or whether a particular hub article is a useful transit node. The model's world model is a dense semantic web; Wikipedia's link structure is a sparse directed graph — and those two structures diverge constantly.

---

## Q2. What is one principle you take away from building this?

**Pre-filter the search space and give the LLM a planning task, not a selection task.**

The naive approach — show the LLM 500 links and ask "which one?" — puts the model in a context where attention is diluted across irrelevant noise. It also forces a *local* decision when the optimal move requires *global* reasoning about the path.

The better approach, which we implemented, is to first compress the problem: one LLM call produces a route plan as structured data (a JSON list of intermediate articles), and subsequent steps become deterministic string-matching against that plan. This separates two concerns: *where do I want to go?* (LLM's strength — world knowledge and semantic reasoning) from *which exact link gets me there?* (pure lookup — the LLM's weakness due to link existence uncertainty).

The general principle is: **use LLMs for knowledge retrieval and planning, use deterministic code for execution**. An LLM is expensive, slow, and occasionally wrong about details. Minimize the number of calls by front-loading the reasoning into one well-structured planning prompt, then follow that plan with cheap, reliable code. Only re-invoke the LLM when the plan breaks down — not at every single decision point. This pattern (plan once, execute deterministically, re-plan on failure) generalizes far beyond Wikipedia navigation to any agentic task with a large branching factor.
