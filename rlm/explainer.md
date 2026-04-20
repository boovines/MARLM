# RLM Project Explainer

## What is RLM?

**Recursive Language Models** (RLM) is an inference-time paradigm from MIT OASYS Lab (Alex Zhang, Tim Kraska, Omar Khattab — arXiv:2512.24601). Instead of feeding an entire long prompt into an LLM's context window, RLM externalizes the prompt as a variable inside a Python REPL and lets the LLM programmatically examine, decompose, and recursively call itself over snippets.

**Key insight**: The LLM writes code that chunks the context, delegates sub-queries to other LLM API calls (`llm_query`), aggregates results in code, and iterates — all driven by the model's own generated code.

**Results**: RLM processes inputs up to 100x beyond model context windows. RLM-Qwen3-8B beats baseline Qwen3-8B by ~28% average on long-context tasks, approaching vanilla GPT-5 quality at comparable cost.

---

## How RLM Works (Complete Flow)

### The Core Loop

1. Your context goes into a `context` variable inside an `exec()` sandbox (the "REPL")
2. The framework calls the LLM API (the "parent") in a loop — each loop is one **iteration**
3. Parent generates text + code → framework runs the code via `exec()` → stdout captured → appended to message history → parent called again with full history
4. `llm_query("some text")` inside the code is just a plain LLM API call — sends a string, gets a string back. No REPL, no memory, no history.
5. Parent says `FINAL(answer)` when done

### Key Concepts

- **Parent**: The LLM called by the framework each iteration. It generates code. It's the only one that sees the message history.
- **`llm_query()`**: A function injected into the REPL. It's just an API call — sends text to an LLM, gets text back. The receiving LLM has NO context about the task, NO history, NO REPL. It just gets whatever string the parent's code passed.
- **Iteration**: One cycle of: call parent → get code → run code → append results to history. An iteration can spawn 0 or 100 `llm_query` calls depending on what the parent generates.
- **Message history**: A growing list of `[assistant]`/`[user]` message dicts. The parent re-reads ALL of it every iteration.
- **REPL variables**: Data stored in the `exec()` namespace. Persists across iterations. The parent can't "see" variables directly — it has to write code that accesses them.
- **`[assistant]` message**: The parent LLM's raw text response (reasoning + code blocks).
- **`[user]` message**: What the framework constructs after running code — contains the code that was run + whatever `print()` produced. Only printed output goes into history; REPL variable contents do NOT.

### What Goes Into History vs What Stays in REPL Variables

This is critical:
- **In history**: Only what `print()` outputs (truncated at 20K chars by `parsing.py:94`)
- **In REPL variables**: The actual data — full classification results, lists, dicts, etc.
- The parent sees the history but must write code to access variables
- If history is compacted, the parent may forget what variables exist (mitigated by `SHOW_VARS()`)

### Depth System

- `max_depth=0`: No sub-calls. `rlm_query` falls back to `llm_query`.
- `max_depth=1` (paper's setting): Parent has REPL + `llm_query`. No recursive children.
- `max_depth=2`: Parent can call `rlm_query()` which spawns a child RLM with its own REPL + iteration loop. That child's `rlm_query` falls back to `llm_query`.

### The Three Files That Matter

1. **`rlm/core/rlm.py`** — The loop. Calls parent, extracts code, runs it, checks for FINAL, appends to history, repeats.
2. **`rlm/utils/prompts.py`** — The system prompt that teaches the parent how to use the REPL.
3. **`rlm/environments/local_repl.py`** — The `exec()` sandbox. Runs code, captures stdout, provides `llm_query`/`rlm_query`.

### Why the TCP Server Exists

The `LMHandler` is a TCP server (`lm_handler.py`) that wraps the LLM API client. It exists so cloud environments (Modal, Docker) can reach the API. For local use it's unnecessary overhead but keeps the architecture uniform. The parent's own calls to the LLM bypass TCP entirely (`lm_handler.py:207-209`).

---

## Paper Challenges & Negative Results (Appendix B)

1. **FINAL() is brittle**: Models mix up FINAL(answer) with FINAL_VAR(variable), or output their plan wrapped in FINAL(). 16% of fine-tuning turns had incorrect FINAL usage.
2. **Same prompt doesn't work for all models**: Qwen3-Coder made hundreds of sub-calls per task. Had to add a warning line to its prompt.
3. **Weak coders fail as parents**: Qwen3-8B struggled to follow REPL instructions.
4. **Thinking models run out of output tokens**: Qwen3-235B used thinking tokens, exceeded output limits mid-trajectory.
5. **All sub-calls are synchronous**: Sequential `llm_query` calls are slow. (Codebase now has `llm_query_batched` with async semaphore.)
6. **RLM worse on short contexts**: Base LLM beats RLM when context fits in the window — overhead isn't worth it.
7. **Cost variance is high**: Median RLM run is cheap but tail is expensive (Figure 3).
8. **Models discard correct answers**: Sometimes found the answer then kept iterating.
9. **Redundant verification**: Models waste calls re-checking answers they already found.

---

## Benchmarks

### Paper's Benchmarks

| Benchmark | Tasks | Context | Complexity | Scoring | Key result |
|-----------|-------|---------|-----------|---------|------------|
| S-NIAH | 50 | 8K→256K | O(1) constant | Accuracy | 100% — solved trivially with regex |
| OOLONG (trec_coarse) | 50 | 131K | O(N) linear | 0.75^|error| for numeric, EM for labels | 44%→56.5% (GPT-5) |
| OOLONG-Pairs | 20 | 32K | O(N²) quadratic | F1 | 0.04%→58% (GPT-5) |
| BrowseComp+ (1K docs) | 150 | 6M-11M | Multi-hop | Accuracy | 0%→91.3% |
| CodeQA (LongBench-v2) | varies | 23K-4.2M | Linear | Accuracy | 24%→62% |

### Critique of Paper's Benchmarks

- **S-NIAH**: Solved with `ctrl+F` in the REPL. Useless for demonstrating RLM value.
- **OOLONG**: Scoring collapses at scale (0.75^50 ≈ 0). Cherry-picked for RLM's strengths — decomposable aggregation.
- **OOLONG-Pairs**: Better — F1 is more forgiving, pairs genuinely benefit from code.
- **Missing**: No holistic comprehension tasks (novel themes, mood, character arcs). No tasks requiring cross-chunk information sharing. RLM is specifically good at "split, classify independently, count in code" — the paper didn't test where this paradigm fails.
- **MuSiQue confirms cherry-picking**: On 5 MuSiQue 2-hop tasks (multi-hop QA requiring chaining facts across paragraphs), vanilla GPT-5 scored 0.80 EM vs RLM's 0.60 EM. RLM was also 14x slower (196s vs 14s avg). The gap from OOLONG (where RLM > vanilla) shrinks and reverses on benchmarks that don't decompose cleanly.

### MuSiQue Results (2026-03-27, n=5)

| Method | EM | Avg F1 | Avg Time |
|--------|-----|--------|----------|
| Vanilla GPT-5 | **0.800** (4/5) | **0.933** | 13.85s |
| RLM (GPT-5 + GPT-5-mini) | 0.600 (3/5) | 0.600 | 195.87s |

All 5 tasks were 2-hop. RLM failed on Task 1 (couldn't chain the hops) where vanilla succeeded easily. Supports the hypothesis that OOLONG was cherry-picked for RLM's decompose-and-aggregate strengths. Script: `benchmark_musique.py`, results: `results/musique_benchmark.jsonl`.

### Better Benchmarks for CLMM Project

| Benchmark | Context | Scoring | Why it fits | On HuggingFace? |
|-----------|---------|---------|-------------|-----------------|
| **InfiniteBench En.MC** | 184K (full novels) | MC accuracy (automated) | Holistic novel comprehension, requires cross-chapter reasoning | Yes |
| **InfiniteBench En.QA** | 192K (full novels) | ROUGE (automated) | Same novels, open-ended answers | Yes |
| **MuSiQue** | 5K-15K | EM/F1 (automated) | Multi-hop — answer requires combining 2-4 facts from different locations | Yes |
| **QuALITY (hard)** | ~5K | MC accuracy (automated) | Anti-skimming questions requiring full-text reading | Yes |
| **LongBench v2** | 8K-2M | MC accuracy (automated) | 503 hard questions, wide context range | Yes |

---

## Cost Analysis

### Model Pricing

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|----------------------|
| Claude Sonnet | $3.00 | $15.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| GPT-5-nano | $0.20 | $1.25 |
| Claude Haiku | $0.80 | $4.00 |
| Gemini 2.5 Flash (free tier) | $0 (250 req/day, 10/min) | $0 |
| Local (Ollama/vLLM) | $0 | $0 |

### Cost per OOLONG task (~131K tokens)

| Setup | Est. cost per task |
|-------|-------------------|
| Sonnet parent + Sonnet children | ~$1.00 |
| Sonnet parent + Haiku children | ~$0.50 |
| Sonnet parent + GPT-4o-mini children | ~$0.30 |
| GPT-4o-mini for everything | ~$0.05-0.10 |
| Sonnet parent + local children (Ollama) | ~$0.08 (parent only) |

### Cost: parent vs children

Children dominate cost because they process the actual context chunks. Parent cost grows slowly (just history). At 131K tokens: children use ~6x more tokens than parent. At 1M tokens: ~33x. At 10M: ~250x.

### Local model performance (Phillip's hardware)

- **MacBook M1 16GB**: Qwen3-8B at ~12-18 tok/s
- **Mac Mini M4**: Qwen3-8B at ~25-35 tok/s
- 131K token OOLONG task: ~10-20 min per task on M4
- 1B tokens: not practical (~10+ days continuous inference on M4)

---

## CLMM Project: Memory-Augmented Recursive Language Models

**Team**: Maksym Bondarenko, Tevin Kim, Justin Hou, Phillip Yan (February 2026)

**Core thesis**: RLMs are stateless across recursive calls — each `llm_query` call has no memory of what other calls discovered. The project investigates whether augmenting RLMs with persistent memory improves performance on tasks requiring reasoning across distributed context.

### Phase 1 (Intermediary Report) — Current Phase
1. **Replicate** vanilla RLM on OOLONG-synth (data on HuggingFace, eval scripts in `oolong/` GitHub repo)
2. **Characterize failure modes** — where does statelessness hurt?

### Phase 2 (Final Report)
1. **Implement memory layer** — graph (Graphiti), flat key-value, and/or vector store (RAG)
2. **Benchmark** memory-augmented RLM vs vanilla RLM
3. **Ablations** over memory types to isolate what actually helps

### Key Insights from Research

**When memory helps:**
- Information from one chunk needed to understand another (cross-references, callbacks)
- Consistency matters (same entity should be classified same way everywhere)
- Task requires comparing/connecting facts across chunks (contradictions, timelines, relationships)

**When memory does NOT help:**
- Each chunk is independently processable (OOLONG counting)
- Task only needs final aggregate (simple summarization)
- Single model can fit the entire context (Gemini 1M window)

**Graph vs other memory types:**
- **Flat key-value**: Simple, works for direct lookups. Bad for "how is X connected to Y?"
- **Vector store (RAG)**: Good for finding similar statements. Bad for structural relationships.
- **Graph**: Good for traversing relationship chains (A→B→C). Overkill for independent facts.
- Ablation across all three is the right approach.

### Research Ideas

1. **Double-pass children** (self-consistency, Wang et al. 2023): Run `llm_query` N times, majority vote. Could improve classification accuracy on OOLONG. Implementable by modifying `_llm_query` in `local_repl.py`.
2. **Prompt engineering**: Few-shot examples, output format, batch size per call.
3. **RLM vs long-context Gemini**: Direct comparison on holistic benchmarks (InfiniteBench). If RLM is worse, that characterizes where the paradigm fails.
4. **Custom benchmark**: Design tasks specifically requiring cross-call memory (contradiction detection, timeline reconstruction, entity disambiguation across documents).
5. **Persistent variable registry**: Auto-include variable names in every prompt to prevent information loss after compaction.

### Where Memory Connects to the Codebase

- **`custom_tools`** (`base_env.py`, `local_repl.py`): Inject `graph_query()` and `graph_update()` as REPL functions
- **`_subcall()` in `rlm.py:645`**: Where child RLMs are spawned — propagate shared memory reference here
- **`tools_subtools` branch**: Deals with passing tools to child RLMs
- **`compaction` branch**: Closest existing "memory" — flat text summarization

---

## Setup Status

- [x] venv created (Python 3.11)
- [x] `rlm` installed in editable mode
- [x] `.env` with `ANTHROPIC_API_KEY`
- [x] `datasets` library installed
- [x] OOLONG-synth data cached from HuggingFace (5200 samples)
- [x] OoLong GitHub repo cloned at `rlm/oolong/`
- [x] `test_run.py` created (basic end-to-end test with Anthropic)
- [ ] Run `test_run.py` to verify framework works
- [ ] Run 2-3 OOLONG tasks through RLM
- [ ] Write benchmark script using OoLong eval scoring

### Immediate Next Steps
1. Run `python test_run.py` — verify plumbing works
2. Run 2-3 short OOLONG-synth tasks (1024-4096 token context) through RLM
3. Observe trajectories — what does the parent actually generate? How does it chunk? How many `llm_query` calls?
4. Compare against vanilla baseline (direct LLM call, no RLM)
5. Decide on final benchmark strategy based on observations

### Key Data Locations

- **OOLONG-synth (HuggingFace)**: `load_dataset("oolongbench/oolong-synth", split="test")` — 5200 samples, 8 datasets (agnews, imdb, yahoo, etc.), contexts from 1024 to 4.2M tokens. **Does NOT include trec_coarse.**
- **trec_coarse raw data**: `oolong/src/data_gen/oolong-synth/validated_data/trec_coarse_validated.jsonl` — needs data gen scripts to build benchmark contexts.
- **OOLONG-Pairs queries**: RLM paper Appendix D.1 (page 20 of `rlm.pdf`) — 20 custom queries over trec_coarse data.
- **OoLong eval scripts**: `oolong/src/eval/eval_script_batched.py` — uses LiteLLM, scores via `synth_process_response()` in `eval_helpers.py`.
- **RLM paper**: `rlm.pdf` in repo root.
- **CLMM proposal**: `CLMM_Project_Proposal (1).pdf` in repo root.

---

## Paper Reference

- **Title**: Recursive Language Models
- **Authors**: Alex L. Zhang, Tim Kraska, Omar Khattab (MIT OASYS Lab)
- **arXiv**: 2512.24601 (Dec 2025, updated Jan 2026)
- **Key claim**: Task-agnostic inference paradigm that handles inputs 100x beyond context windows
- **Config**: max_depth=1, GPT-5 parent, GPT-5-mini for llm_query calls, fixed system prompt across all tasks
