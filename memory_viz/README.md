# memory_viz

Tiny viewer for the two memory backends used by the memory-augmented RLM:

- **Knowledge graph** (Graphiti + Kuzu) — force-directed node/edge view with
  node-summary inspection.
- **Flat KV memory** — straight key/value table of `FlatKVBackend._store`.

## Run

```bash
cd memory_viz
npm install
npm run dev        # Vite at http://localhost:5173
```

The app reads `public/data/graph.json` and `public/data/flat.json`. The
repo ships with seed data matching MuSiQue Task 1
("Who is the spouse of the Green performer?").

## Refresh the data from real backend state

From the repo root:

```bash
# Simulated — no OpenAI calls, instant, deterministic demo
python rlm/scripts/dump_memory_viz.py

# Real — runs one MuSiQue task through the real RLM + memory pipeline
# (~3 min, ~$0.50 in API costs). Writes whatever the parent actually wrote.
python rlm/scripts/dump_memory_viz.py --mode real
```

The script snapshots the live Graphiti Kuzu graph (nodes + edges) and the
FlatKVBackend dict, writing both to `memory_viz/public/data/`. Reload the
browser to see the new state.

## Expected file shapes

`graph.json`:
```json
{
  "meta": {"task_id": "...", "question": "...", "gold_answer": "...", "source": "...", "generated_at": "..."},
  "nodes": [{"id": "...", "name": "...", "summary": "...", "kind": "Entity"}],
  "edges": [{"source": "...", "target": "...", "fact": "...", "name": "..."}]
}
```

`flat.json`:
```json
{
  "meta": {"task_id": "...", "question": "...", "gold_answer": "...", "source": "...", "generated_at": "..."},
  "store": {"key": "value", ...}
}
```
