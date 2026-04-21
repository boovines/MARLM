"""Populate a FlatKVBackend and a GraphitiBackend with a small demo, then
dump their state to memory_viz/public/data/{flat,graph}.json so the
memory_viz dev app can render them.

Modes:
  simulated (default)  — seed both backends with known MuSiQue-T1 facts.
                         No OpenAI API calls. Instant. Good for a clean demo.
  real                 — run one MuSiQue task (2-hop, first example) through
                         the real RLM + memory pipeline and dump whatever the
                         parent actually wrote. Makes API calls (~$0.50, ~3 min).

Usage:
    python rlm/scripts/dump_memory_viz.py                # simulated
    python rlm/scripts/dump_memory_viz.py --mode real    # live run
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


def run_real() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("--mode real requires OPENAI_API_KEY in env / .env")

    print("Real mode — running one MuSiQue task end-to-end. This takes ~3 min.")

    from datasets import load_dataset
    from rlm import RLM
    from rlm.memory import FlatKVBackend, GraphitiBackend, make_memory_tools
    from rlm.bench.harness import wrap_tools_with_counters

    ds = load_dataset("bdsaglam/musique", "answerable", split="validation")
    task = ds[0]
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


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["simulated", "real"], default="simulated")
    args = p.parse_args()
    print(f"Writing viz data to {VIZ_DATA}")
    if args.mode == "simulated":
        run_simulated()
    else:
        run_real()
    print("\nDone. Start the viewer with:")
    print("  cd memory_viz && npm install && npm run dev")


if __name__ == "__main__":
    main()
