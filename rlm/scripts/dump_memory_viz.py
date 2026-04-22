"""Populate a FlatKVBackend and a GraphitiBackend with a small demo, then
dump their state to memory_viz/public/data/{flat,graph}.json so the
memory_viz dev app can render them.

Modes:
  simulated (default)  — seed both backends with 4 known MuSiQue-T1 facts.
                         No OpenAI API calls. Instant. ~4 nodes, ~4 edges.
  real                 — run one MuSiQue task through the real RLM + memory
                         pipeline. Dumps what the parent actually wrote.
                         Makes API calls (~$1, ~3-5 min). Density depends on
                         how much the parent chose to write.
  dense                — bypass the parent. Write every paragraph of a chosen
                         MuSiQue task directly to Graphiti as its own episode.
                         Shows what a rich KG looks like when given real text
                         content. Makes API calls (~$1-2, ~3-10 min depending
                         on paragraph count). Typically yields 30-100+ nodes.

Task selection flags (apply to real / dense):
  --task-index N   pick the N-th task (default 0)
  --min-hops H     skip tasks whose decomposition has fewer than H hops

Usage:
    python rlm/scripts/dump_memory_viz.py                          # simulated
    python rlm/scripts/dump_memory_viz.py --mode real --min-hops 3 # 3-hop real
    python rlm/scripts/dump_memory_viz.py --mode dense --min-hops 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
VIZ_DATA = REPO_ROOT / "memory_viz" / "public" / "data"
VIZ_DATA.mkdir(parents=True, exist_ok=True)


# ── Simulated seed data ────────────────────────────────────────────────

SIMULATED_FACTS = [
    ("fact_1", "Steve Hillage performed the song Green."),
    ("fact_2", "Steve Hillage is a British progressive-rock guitarist."),
    ("fact_3", "Steve Hillage married Miquette Giraudy."),
    ("fact_4", "Miquette Giraudy is a synthesist and long-time musical collaborator of Steve Hillage."),
]

SIMULATED_FLAT = {
    "hop1_question": "Who is the performer of the song 'Green'?",
    "hop1_performer": "Steve Hillage",
    "hop1_source": "Section 'Green (Steve Hillage album)' — Steve Hillage performed this album.",
    "hop2_question": "Who is Steve Hillage's spouse?",
    "hop2_spouse": "Miquette Giraudy",
    "hop2_source": "Biographical paragraph: 'Steve Hillage married Miquette Giraudy.'",
    "final_answer": "Miquette Giraudy",
}

TASK_META = {
    "task_id": "2hop__460946_294723",
    "question": "Who is the spouse of the Green performer?",
    "gold_answer": "Miquette Giraudy",
}


# ── Dump helpers ───────────────────────────────────────────────────────

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {path} ({path.stat().st_size:,} bytes)")


def dump_flat(flat_store: dict[str, str], source: str) -> None:
    payload = {
        "meta": {**TASK_META, "source": source, "generated_at": datetime.now(timezone.utc).isoformat()},
        "store": flat_store,
    }
    write_json(VIZ_DATA / "flat.json", payload)


def dump_graph(nodes: list[dict], edges: list[dict], source: str) -> None:
    payload = {
        "meta": {**TASK_META, "source": source, "generated_at": datetime.now(timezone.utc).isoformat()},
        "nodes": nodes,
        "edges": edges,
    }
    write_json(VIZ_DATA / "graph.json", payload)


# ── Graphiti snapshotting ──────────────────────────────────────────────

async def snapshot_graphiti(graphiti) -> tuple[list[dict], list[dict]]:
    """Read every Entity node + every edge fact from the live Kuzu DB behind
    a Graphiti instance. Works on graphiti 0.28.x."""
    driver = graphiti.driver

    # Entity nodes.
    n_records, _, _ = await driver.execute_query(
        "MATCH (n:Entity) RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary"
    )
    nodes = [
        {
            "id": r["uuid"],
            "name": r["name"] or r["uuid"],
            "summary": r["summary"] or "",
            "kind": "Entity",
        }
        for r in n_records
    ]

    # Edges: RELATES_TO goes Entity -> RelatesToNode_ -> Entity.
    e_records, _, _ = await driver.execute_query(
        "MATCH (a:Entity)-[:RELATES_TO]->(r:RelatesToNode_)-[:RELATES_TO]->(b:Entity) "
        "RETURN a.uuid AS src, b.uuid AS tgt, r.fact AS fact, r.name AS name"
    )
    edges = [
        {
            "source": r["src"],
            "target": r["tgt"],
            "fact": r["fact"] or "",
            "name": r["name"] or "",
        }
        for r in e_records
    ]
    return nodes, edges


# ── Modes ──────────────────────────────────────────────────────────────

def run_simulated() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    print("Simulated mode — no RLM calls.")

    # Flat: just write the seed dict.
    from rlm.memory import FlatKVBackend
    flat = FlatKVBackend()
    for k, v in SIMULATED_FLAT.items():
        flat.write(k, v)
    dump_flat(dict(flat._store), source="simulated")

    # Graph: seed Graphiti with fact episodes, then snapshot.
    if not os.getenv("OPENAI_API_KEY"):
        print("  OPENAI_API_KEY not set — skipping Graphiti seed; dumping empty graph.")
        dump_graph([], [], source="simulated (no OPENAI_API_KEY; graph skipped)")
        return

    from rlm.memory import GraphitiBackend
    g = GraphitiBackend()
    try:
        for k, v in SIMULATED_FACTS:
            g.write(k, v)
        nodes, edges = g._loop.run_until_complete(snapshot_graphiti(g._graphiti))
        dump_graph(nodes, edges, source="simulated")
    finally:
        g.reset()


def _pick_task(task_index: int, min_hops: int):
    from datasets import load_dataset
    ds = load_dataset("bdsaglam/musique", "answerable", split="validation")
    if min_hops > 2:
        ds = ds.filter(lambda x: len(x["question_decomposition"]) >= min_hops)
    if task_index >= len(ds):
        raise SystemExit(f"task_index {task_index} >= {len(ds)} matching tasks")
    return ds[task_index]


def run_real(task_index: int = 0, min_hops: int = 0) -> None:
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("--mode real requires OPENAI_API_KEY in env / .env")

    print(f"Real mode — running one MuSiQue task (index={task_index}, min-hops={min_hops}). ~3-5 min.")

    from rlm import RLM
    from rlm.memory import FlatKVBackend, GraphitiBackend, make_memory_tools
    from rlm.bench.harness import wrap_tools_with_counters

    task = _pick_task(task_index, min_hops)
    question = task["question"]
    paragraphs = "\n\n".join(
        f"[{p.get('title', f'Paragraph {i+1}')}]\n{p['paragraph_text']}"
        for i, p in enumerate(task["paragraphs"])
    )
    meta = {"task_id": task["id"], "question": question, "gold_answer": task["answer"]}
    globals()["TASK_META"] = meta  # so the dump helpers pick it up

    hint = (
        " This question requires chaining facts across paragraphs (multi-hop). "
        "Use memory_write to stash intermediate findings (e.g. "
        "memory_write('hop1_performer', 'Steve Hillage')), and memory_read to "
        "retrieve them in later sub-calls. The memory persists across llm_query calls."
    )

    for backend_name, BackendCls in (("flat", FlatKVBackend), ("graph", GraphitiBackend)):
        print(f"\n[{backend_name}] running …")
        backend = BackendCls()
        tools = make_memory_tools(backend)
        tools, counters = wrap_tools_with_counters(tools)
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "gpt-5"},
            other_backends=["openai"],
            other_backend_kwargs=[{"model_name": "gpt-5-mini"}],
            environment="local",
            max_depth=1,
            max_iterations=15,
            verbose=False,
            custom_tools=tools,
            custom_sub_tools=tools,
        )
        try:
            rlm.completion(prompt=paragraphs, root_prompt=question + hint)
            print(f"  writes={counters['write']} reads={counters['read']}")
            if backend_name == "flat":
                dump_flat(dict(backend._store), source=f"real / MuSiQue task {meta['task_id']}")
            else:
                nodes, edges = backend._loop.run_until_complete(snapshot_graphiti(backend._graphiti))
                dump_graph(nodes, edges, source=f"real / MuSiQue task {meta['task_id']}")
        finally:
            try:
                rlm.close()
            except Exception:
                pass
            try:
                backend.reset()
            except Exception:
                pass


def run_dense(task_index: int = 0, min_hops: int = 0) -> None:
    """Write every paragraph of a MuSiQue task directly to Graphiti as a
    separate episode. Produces the richest possible graph. No RLM; the
    parent does not participate. Also populates a flat KV with a small
    per-paragraph index so the Flat tab isn't empty."""
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("--mode dense requires OPENAI_API_KEY in env / .env")

    from rlm.memory import FlatKVBackend, GraphitiBackend

    task = _pick_task(task_index, min_hops)
    paragraphs = task["paragraphs"]
    n_hops = len(task["question_decomposition"])

    print(
        f"Dense mode — seeding Graphiti with {len(paragraphs)} paragraphs from "
        f"MuSiQue task id={task['id']} ({n_hops}-hop).\n"
        f"Each write takes ~10-30s of Graphiti entity extraction. "
        f"Estimated total: ~{len(paragraphs) * 15 / 60:.0f} min."
    )

    meta = {
        "task_id": task["id"],
        "question": task["question"],
        "gold_answer": task["answer"],
    }
    globals()["TASK_META"] = meta

    # Flat: store each paragraph as its own key for quick lookup.
    flat = FlatKVBackend()
    for i, p in enumerate(paragraphs):
        title = p.get("title", f"Paragraph {i+1}")
        flat.write(f"paragraph:{title}", p["paragraph_text"])
    flat.write("_question", task["question"])
    flat.write("_gold_answer", task["answer"])
    dump_flat(dict(flat._store), source=f"dense / MuSiQue task {task['id']} ({n_hops}-hop)")

    # Graph: each paragraph becomes its own episode.
    g = GraphitiBackend()
    try:
        for i, p in enumerate(paragraphs):
            title = p.get("title", f"Paragraph {i+1}")
            text = p["paragraph_text"]
            print(f"  [{i+1}/{len(paragraphs)}] writing '{title[:60]}' ({len(text)} chars)…")
            g.write(f"p_{i}_{title[:40]}", f"[{title}] {text}")
        nodes, edges = g._loop.run_until_complete(snapshot_graphiti(g._graphiti))
        print(f"\nGraph snapshot: {len(nodes)} nodes, {len(edges)} edges")
        dump_graph(nodes, edges, source=f"dense / MuSiQue task {task['id']} ({n_hops}-hop)")
    finally:
        g.reset()


# ── NIAH mode: the sparse end of the spectrum ─────────────────────────

# A synthetic NIAH haystack: 19 deliberately low-entity filler sentences and
# one information-rich needle. Graphiti should extract almost nothing from the
# filler (no proper nouns, no relations) and a few entities from the needle —
# the opposite extreme of the dense-MuSiQue 196-node graph.

NIAH_FILLER = [
    "The sky was cloudy and overcast throughout the day.",
    "It rained lightly during the afternoon hours.",
    "A gentle breeze moved through the empty streets.",
    "The grass remained wet from the morning dew.",
    "Small birds chirped intermittently in the early light.",
    "The coffee in the cup had gone cold by noon.",
    "Leaves rustled softly whenever the wind picked up.",
    "Somewhere in the distance a door creaked shut.",
    "The street lamps flickered once and then steadied.",
    "Paper bags drifted along the curb without purpose.",
    "A window had been left open and the curtains swayed.",
    "The floor tiles were chipped and uneven underfoot.",
    "Water dripped from a leaky pipe into an empty basin.",
    "The mail arrived later than usual that day.",
    "Somebody had forgotten to turn the porch light off.",
    "The clock on the wall had stopped at an odd hour.",
    "Dust settled slowly onto the unused shelves.",
    "The lobby was quiet except for a faint humming sound.",
    "A forgotten cup sat on the edge of the countertop.",
]

NIAH_NEEDLE = (
    "Dr. Alice Chen, the lead researcher at the Stanford Oncology Lab, "
    "keeps her access code 7-ECHO-4 stored in a locked drawer."
)

NIAH_QUESTION = "What is Dr. Alice Chen's access code?"
NIAH_GOLD = "7-ECHO-4"


def run_niah(needle_position: int = 12) -> None:
    """Seed Graphiti with a NIAH-style haystack: many low-entity filler
    sentences + one fact-rich needle. Illustrates the sparse end of what
    Graphiti produces when the source text has almost no relational structure."""
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("--mode niah requires OPENAI_API_KEY in env / .env")

    from rlm.memory import FlatKVBackend, GraphitiBackend

    pos = max(0, min(needle_position, len(NIAH_FILLER)))
    paragraphs = list(NIAH_FILLER)
    paragraphs.insert(pos, NIAH_NEEDLE)

    print(
        f"NIAH mode — seeding Graphiti with 1 needle + {len(NIAH_FILLER)} "
        f"entity-sparse filler sentences (needle at index {pos}).\n"
        f"Expect ~3–8 nodes total — the opposite extreme of dense-MuSiQue."
    )

    meta = {
        "task_id": "synthetic_niah",
        "question": NIAH_QUESTION,
        "gold_answer": NIAH_GOLD,
    }
    globals()["TASK_META"] = meta

    # Flat: write each line (filler + needle) so the KV table shows the haystack.
    flat = FlatKVBackend()
    for i, line in enumerate(paragraphs):
        key = f"needle:line_{i:02d}" if i == pos else f"line_{i:02d}"
        flat.write(key, line)
    flat.write("_question", NIAH_QUESTION)
    flat.write("_gold_answer", NIAH_GOLD)
    dump_flat(dict(flat._store), source="niah (synthetic haystack)")

    # Graph: each sentence is its own episode.
    g = GraphitiBackend()
    try:
        for i, line in enumerate(paragraphs):
            is_needle = i == pos
            label = "NEEDLE" if is_needle else f"filler_{i:02d}"
            print(f"  [{i+1}/{len(paragraphs)}] writing '{label}'…")
            g.write(label, line)
        nodes, edges = g._loop.run_until_complete(snapshot_graphiti(g._graphiti))
        print(f"\nGraph snapshot: {len(nodes)} nodes, {len(edges)} edges (vs 196/164 on dense-MuSiQue)")
        dump_graph(nodes, edges, source="niah (synthetic haystack, 1 needle + 19 filler)")
    finally:
        g.reset()


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["simulated", "real", "dense", "niah"], default="simulated")
    p.add_argument("--task-index", type=int, default=0, help="which matching task to pick (real/dense only)")
    p.add_argument("--min-hops", type=int, default=0, help="filter MuSiQue tasks to at least H hops (real/dense only)")
    p.add_argument("--needle-pos", type=int, default=12, help="index of needle within the haystack (niah only)")
    args = p.parse_args()
    print(f"Writing viz data to {VIZ_DATA}")
    if args.mode == "simulated":
        run_simulated()
    elif args.mode == "real":
        run_real(task_index=args.task_index, min_hops=args.min_hops)
    elif args.mode == "dense":
        run_dense(task_index=args.task_index, min_hops=args.min_hops)
    else:
        run_niah(needle_position=args.needle_pos)
    print("\nDone. Start the viewer with:")
    print("  cd memory_viz && npm install && npm run dev")


if __name__ == "__main__":
    main()
