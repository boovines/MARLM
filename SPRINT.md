# MARLM 2-Day Sprint — Presentation-Ready MVP

**Deadline:** 2026-04-22 (presentation) · **Paper:** 2026-05-11 (3 weeks)
**Goal:** demoable memory-augmented RLM with flat-KV + Graphiti-KG backends, OOLONG + MuSiQue results vs vanilla / stateless-RLM baselines.

## Parallelization map

```
 HOUR 0-1   M0 (solo, interface freeze — blocker)
            │
 HOUR 1-6   ├── Agent A: M1 (flat KV)           ─┐
            ├── Agent B: M3 (Graphiti+Kuzu)      ├── all use M0 stub interface
            └── Agent C: B1 (bench adapters)    ─┘
            │
 HOUR 6-8   D1 (smoke sweep, human-driven)
 HOUR 8-10  D2 (slides)
```

**Interface freeze at hour 1 is non-negotiable.** Once M0 merges, A/B/C cannot need to renegotiate the `MemoryBackend` signature. Any change after hour 1 requires syncing all three lanes.

---

## PRD M0 — Memory interface + tool wiring

**Owner:** 1 human, ~1 hr · **Blocks:** M1, M3, B1

### Contract to freeze (copy this verbatim into `rlm/memory/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class MemoryHit:
    key: str
    value: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

class MemoryBackend(ABC):
    @abstractmethod
    def write(self, key: str, value: str) -> None: ...
    @abstractmethod
    def read(self, query: str, top_k: int = 5) -> list[MemoryHit]: ...
    @abstractmethod
    def reset(self) -> None: ...
```

### Deliverables

1. `rlm/rlm/memory/__init__.py` — exports `MemoryBackend`, `MemoryHit`, `make_memory_tools`, `DictBackend`.
2. `rlm/rlm/memory/base.py` — the contract above, plus a `DictBackend(MemoryBackend)` stub (just a `dict[str,str]`) so A/B/C have something to develop against before M1/M3 land.
3. `rlm/rlm/memory/tools.py` — `make_memory_tools(backend: MemoryBackend) -> dict` returning:
    ```python
    {
      "memory_write": {
        "tool": lambda key, value: backend.write(key, value),
        "description": "Store a string value under a string key in persistent "
                       "memory that survives across llm_query calls and across "
                       "iterations. Use to stash intermediate findings (e.g. "
                       "'hop1_performer' -> 'Steve Hillage') so later sub-calls "
                       "can retrieve them.",
      },
      "memory_read": {
        "tool": lambda query, top_k=5: [
            {"key": h.key, "value": h.value, "score": h.score}
            for h in backend.read(query, top_k=top_k)
        ],
        "description": "Retrieve previously stored memory entries most relevant "
                       "to `query`. Returns a list of {key, value, score} dicts. "
                       "Call BEFORE writing to check if relevant context exists.",
      },
    }
    ```
    *The description wording above is load-bearing — it's what tells the parent LLM when to call these. Do not shorten it.*
4. `rlm/tests/memory/test_tools_injection.py` — minimal test: build `DictBackend`, wrap with `make_memory_tools`, construct an `RLM(custom_tools=..., custom_sub_tools=...)`, call `completion("write 'x' under key 'k', then read key 'k'")` with a cheap model, assert the write-then-read round-trip.

### Files to touch

- `rlm/rlm/memory/__init__.py` (NEW)
- `rlm/rlm/memory/base.py` (NEW)
- `rlm/rlm/memory/tools.py` (NEW)
- `rlm/tests/memory/test_tools_injection.py` (NEW)
- **Do not touch** `rlm/core/rlm.py` or `rlm/environments/local_repl.py`.

### Done criteria

- `uv run pytest rlm/tests/memory/ -v` green.
- A 5-line Python snippet in `examples/marlm_demo.py` builds an RLM with `DictBackend`, runs a trivial write/read script, prints the hit.

### What NOT to do

- No `reset()` scoping per task yet — just a single backend instance per benchmark run.
- No metadata/tags parameter — keep the signature as above.
- No embedding, no retrieval quality tuning.

---

## PRD M1 — Flat KV backend

**Owner:** Agent A, ~1 hr · **Starts at:** hour 1 (after M0 freeze)

### Deliverables

1. `rlm/rlm/memory/flat_kv.py`:
    ```python
    class FlatKVBackend(MemoryBackend):
        def __init__(self):
            self._store: dict[str, str] = {}
        def write(self, key, value):
            self._store[key] = value
        def read(self, query, top_k=5):
            # exact hit first
            if query in self._store:
                return [MemoryHit(key=query, value=self._store[query], score=1.0)]
            # substring fallback
            hits = [
                MemoryHit(key=k, value=v, score=0.5)
                for k, v in self._store.items()
                if query.lower() in k.lower() or query.lower() in v.lower()
            ]
            return hits[:top_k]
        def reset(self):
            self._store.clear()
    ```
2. `rlm/tests/memory/test_flat_kv.py` — write/read/miss/substring/reset.

### Done criteria

- Tests pass.
- Swap `DictBackend` for `FlatKVBackend` in `examples/marlm_demo.py`; behavior identical.

### What NOT to do

- No fuzzy matching beyond substring. No LLM calls. No sorting by recency.

---

## PRD M3 — Graphiti + Kuzu backend

**Owner:** Agent B, ~3-4 hrs · **Starts at:** hour 1 (after M0 freeze) · **Long pole**

### Setup

1. Add to `rlm/pyproject.toml` dependencies: `graphiti-core[kuzu]>=0.x` (check the vendored `graphiti/pyproject.toml` for actual version — prefer vendored).
2. Confirm `OPENAI_API_KEY` is in `.env` (Graphiti uses it directly — **no custom LLMClient shim in MVP**, that's a paper-phase task).

### Deliverables

1. `rlm/rlm/memory/graphiti_kg.py`:

    ```python
    import asyncio, tempfile, os, shutil
    from datetime import datetime, timezone
    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver  # adjust if API differs
    from .base import MemoryBackend, MemoryHit

    class GraphitiBackend(MemoryBackend):
        def __init__(self):
            self._tmpdir = tempfile.mkdtemp(prefix="marlm_kuzu_")
            self._loop = asyncio.new_event_loop()
            driver = KuzuDriver(db=os.path.join(self._tmpdir, "graph.kz"))
            self._g = Graphiti(graph_driver=driver)
            self._loop.run_until_complete(self._g.build_indices_and_constraints())

        def write(self, key, value):
            self._loop.run_until_complete(self._g.add_episode(
                name=key,
                episode_body=value,
                source_description="marlm",
                reference_time=datetime.now(tz=timezone.utc),
            ))

        def read(self, query, top_k=5):
            edges = self._loop.run_until_complete(
                self._g.search(query, num_results=top_k)
            )
            return [
                MemoryHit(
                    key=e.fact[:80],
                    value=e.fact,
                    score=1.0,  # Graphiti doesn't expose normalized score here
                    metadata={"src": e.source_node_uuid, "tgt": e.target_node_uuid},
                )
                for e in edges
            ]

        def reset(self):
            self._loop.run_until_complete(self._g.close())
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self.__init__()
    ```

2. `rlm/tests/memory/test_graphiti_kg.py` — the exact MuSiQue Task 1 failure-mode test:
    - Write: `write("fact_1", "Steve Hillage performed the song Green")`, `write("fact_2", "Steve Hillage married Miquette Giraudy")`.
    - Read: `read("spouse of the Green performer")`.
    - Assert: "Miquette Giraudy" appears in at least one hit's value or metadata. **If this test passes, we have evidence the KG bridges the hop that RLM failed in the report.**

### Critical knobs (tune once, don't iterate)

- **Episode batching** for OOLONG: *don't* call `write()` per article in the benchmark — the parent LLM decides when to write. If it calls per-article, that's ~1500 × 30s = catastrophe. Cap total writes per task to ~50 by writing in the description that the model should batch. **Put this in the `memory_write` description in M0** before kicking off runs. (Retroactively edit M0 if needed — this is the one interface-touching change we pre-authorize.)

### Done criteria

- Graphiti test passes.
- 1-task OOLONG-16K smoke with `GraphitiBackend` completes without crashing (cost/score don't matter yet, just plumbing).

### Fallback if blocked

If Graphiti+Kuzu integration takes >4 hrs: swap to **FalkorDB via docker** (`docker run -p 6379:6379 falkordb/falkordb:latest`). Same `Graphiti()` API, different driver. Do not burn more than 4 hrs on Kuzu specifically — cut losses at hour 5.

### What NOT to do

- No custom LLMClient shim. Graphiti calls OpenAI directly; we eat the extra API calls.
- No per-task `group_id` scoping. Fresh tmpdir + fresh instance per task is simpler and safer.
- No temporal query features. `add_episode` + `search` only.

---

## PRD B1 — Benchmark adapters with `--memory` flag

**Owner:** Agent C, ~2-3 hrs · **Starts at:** hour 1 (after M0 freeze)

### Deliverables

1. Refactor `rlm/benchmark_oolong.py` and `rlm/benchmark_musique.py` to accept CLI args:
    ```
    --memory {none,flat,graph}       default: none
    --num-tasks N                     default: 3
    --context-len N                   default: 16384 (OOLONG only)
    --output FILE                     default: results/{bench}_{memory}_{ts}.jsonl
    ```
2. Shared helper `rlm/rlm/bench/memory_factory.py`:
    ```python
    def build_memory(name: str) -> MemoryBackend | None:
        if name == "none": return None
        if name == "flat": return FlatKVBackend()
        if name == "graph": return GraphitiBackend()
        raise ValueError(name)
    ```
3. In each benchmark's RLM construction:
    ```python
    backend = build_memory(args.memory)
    custom_tools = make_memory_tools(backend) if backend else None
    rlm = RLM(
        ...,
        custom_tools=custom_tools,
        custom_sub_tools=custom_tools,  # same closure → shared store
    )
    # per-task: backend.reset() before rlm.completion(...) to avoid cross-task bleed
    ```
4. **JSONL schema additions** (needed for D1 plotting):
    - `memory_backend` (str)
    - `memory_calls_write` (int) — count by wrapping the closure with a counter
    - `memory_calls_read` (int)

### Done criteria

- `python benchmark_oolong.py --memory none --num-tasks 1 --context-len 16384` reproduces the interim report's 16K number within stochastic variance. (Regression guard — if this drifts, the memory paths are untrustworthy.)
- `python benchmark_oolong.py --memory flat --num-tasks 1 --context-len 16384` writes a JSONL with non-zero `memory_calls_*` fields.
- Same for MuSiQue.

### What NOT to do

- No `sweep.py` driver. D1 is hand-run.
- No LaTeX export. D2 handles presentation formatting separately.
- No changes to scoring helpers — reuse the existing `synth_process_response` for OOLONG and whatever MuSiQue uses.

---

## PRD D1 — Smoke sweep run

**Owner:** you (human), ~2 hrs · **Starts at:** hour 6 (after M1, M3, B1 all green)

### Grid (24 runs)

| Benchmark | none | flat | graph |
|---|---|---|---|
| OOLONG-16K (n=3) | ✓ | ✓ | ✓ |
| OOLONG-32K (n=3) | ✓ | ✓ | ✓ |
| MuSiQue-2hop (n=5) | ✓ | ✓ | ✓ |

+ vanilla-GPT-5 rows for OOLONG-16K/32K and MuSiQue-2hop from existing `benchmark_oolong.py` / `benchmark_musique.py` runs (reuse the interim report's numbers if already run — don't re-burn budget).

### Budget ceiling

Estimate before running: OOLONG-32K × RLM × graph is the most expensive cell — expect ~$2-3/task × 3 tasks = ~$10. Total sweep ~$60-80. **If a single cell exceeds $15, kill it and lower to n=1.**

### Command pattern

```bash
for MEM in none flat graph; do
  python benchmark_oolong.py   --memory $MEM --num-tasks 3 --context-len 16384 &
  python benchmark_oolong.py   --memory $MEM --num-tasks 3 --context-len 32768 &
  python benchmark_musique.py  --memory $MEM --num-tasks 5 &
  wait
done
```

Run the three `$MEM` values sequentially (to keep API rate-limits sane), but benchmarks within a `$MEM` in parallel.

### Done criteria

- 9 JSONL result files in `results/`.
- `results/summary.md` table (hand-compile — don't build infra): rows = cells, columns = avg score, avg time, avg cost, memory_writes, memory_reads.

---

## PRD D2 — Presentation deck

**Owner:** you (human), ~1-2 hrs · **Starts at:** hour 8

### Required slides

1. **Motivation** (1 slide): reuse interim report §5.3 — MuSiQue Task 1 failure, "stateless sub-calls can't chain hops."
2. **Approach** (1 slide): diagram — parent REPL with `memory_write` / `memory_read`; three backends; `custom_tools` injection, no engine fork.
3. **Results: OOLONG** (1 slide): bar chart, 3 memory conditions × 2 context lengths, vs vanilla baseline.
4. **Results: MuSiQue** (1 slide): bar chart, 3 memory conditions, per-task EM breakdown emphasizing Task 1.
5. **Cost vs benefit** (1 slide): score Δ vs $ Δ vs baseline. Matches the report's cost-forward framing.
6. **What's next** (1 slide): vector RAG, InfiniteBench, full sweep, depth=2 ablation — i.e. the deferred items from this sprint. This is the honest 3-week-paper plan.

### Do NOT include

- Long architecture diagrams. One mechanism slide is enough.
- Ablation "matter of memory type" claims — you have flat vs graph, not vector, so don't over-claim on the retrieval-vs-structure axis.
- Anything about self-consistency / majority-vote unless you re-ran it with memory — that's out of MVP scope.

---

## Hour-by-hour plan (single human + parallel agents)

| T+ | Action |
|---|---|
| 0:00 | You: write M0. Freeze the `MemoryBackend` contract and `make_memory_tools` descriptions. |
| 1:00 | M0 merged. Kick off **three parallel agents** with PRDs M1, M3, B1 copy-pasted as their prompts. |
| 1:00–3:00 | Monitor agents. M1 likely done ~2:00. |
| 3:00–5:00 | B1 likely done ~4:00. Start running M1-based smoke tasks from D1 in parallel — `none` and `flat` cells can start before M3 lands. |
| 5:00–7:00 | M3 lands ~5-6:00 (or fallback to FalkorDB). Smoke-test Graphiti on one task before the full graph sweep. |
| 7:00–9:00 | D1: run remaining `graph` cells. Compile `results/summary.md`. |
| 9:00–11:00 | D2: slides. |

Slack: 2 hrs before day 2 presentation for re-runs on any failed cell.

---

## Risk register (keep this visible)

1. **Graphiti+Kuzu integration fails** → fallback to FalkorDB-in-docker. Cut at hour 5.
2. **Parent LLM never calls `memory_write`** → invisible failure. Sanity-check one OOLONG-flat trace log after B1 lands, before the full sweep. If zero writes, tune the description and re-test.
3. **Graphiti episode cost explodes** → the description already tells the parent to batch. If it still writes per-article, add an explicit line: "writes are expensive (~10s each); batch many items into one value."
4. **Budget blow-out** → $15/cell kill-switch, stated in D1.
5. **Graphiti `search()` returns empty on short graphs** → falls back to no-hit, parent just proceeds without memory context. Not fatal; `memory_calls_read > 0 AND zero hits` is a diagnostic to surface in summary.md.
