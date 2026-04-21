"""MuSiQue benchmark — all 5 methods against 2-hop QA with distractor paragraphs.

Methods: vanilla, rlm, rlm_majority, rlm_memory_flat, rlm_memory_graph.

Scoring: SQuAD-style EM + token-level F1, max over gold + aliases.

Usage:
    python benchmark_musique.py \\
        --methods vanilla,rlm,rlm_memory_flat \\
        --num-tasks 5 --max-wall-s 600
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from collections import Counter
from datetime import datetime as dt_datetime

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

from rlm import RLM
from rlm.bench.harness import (
    METHODS,
    TaskTimeout,
    parse_method,
    wall_clock_limit,
    wrap_tools_with_counters,
)
from rlm.bench.majority import install_majority_vote
from rlm.bench.memory_factory import build_memory
from rlm.logger import RLMLogger
from rlm.memory import make_memory_tools

load_dotenv()

PARENT_MODEL = "gpt-5"
CHILD_MODEL = "gpt-5-mini"
NUM_HOPS = None
MAX_ITERATIONS = 15
MAX_DEPTH = 1

MEMORY_HINT = (
    " This question requires chaining facts across paragraphs (multi-hop). "
    "Use memory_write to stash intermediate findings between hops (e.g. "
    "memory_write('hop1_performer', 'Steve Hillage')), and memory_read to "
    "retrieve them in later sub-calls. The memory persists across llm_query "
    "calls, so sub-calls can share discoveries."
)


# ── Scoring ────────────────────────────────────────────────────────────

def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def exact_match(pred, golds):
    np = normalize_answer(pred)
    return max(int(normalize_answer(g) == np) for g in golds)


def f1_score(pred, golds):
    np_ = normalize_answer(pred)
    pt = np_.split()
    best = 0.0
    for g in golds:
        gt = normalize_answer(g).split()
        common = Counter(pt) & Counter(gt)
        nc = sum(common.values())
        if nc == 0:
            continue
        p = nc / len(pt) if pt else 0
        r = nc / len(gt) if gt else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        best = max(best, f1)
    return best


def score_prediction(pred, gold, aliases):
    golds = [gold] + (aliases or [])
    return {"em": exact_match(pred, golds), "f1": f1_score(pred, golds)}


# ── Tasks ──────────────────────────────────────────────────────────────

def format_context(paragraphs):
    return "\n\n".join(f"[{p.get('title', f'Paragraph {i+1}')}]\n{p['paragraph_text']}" for i, p in enumerate(paragraphs))


def load_tasks(num_tasks):
    print("Loading MuSiQue from HuggingFace...")
    ds = load_dataset("bdsaglam/musique", "answerable", split="validation")
    print(f"  total: {len(ds)}")
    if NUM_HOPS is not None:
        ds = ds.filter(lambda x: len(x["question_decomposition"]) == NUM_HOPS)
    n = min(num_tasks, len(ds))
    tasks = [ds[i] for i in range(n)]
    print(f"  using {n} tasks; hop distribution: {dict(Counter(len(t['question_decomposition']) for t in tasks))}")
    return tasks


# ── Args ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", type=str, default="vanilla,rlm,rlm_memory_flat",
                   help=f"comma-sep subset of {METHODS}")
    p.add_argument("--num-tasks", type=int, default=5)
    p.add_argument("--max-wall-s", type=int, default=600)
    p.add_argument("--majority-n", type=int, default=3)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    ms = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in ms if m not in METHODS]
    if bad:
        raise SystemExit(f"unknown method(s): {bad}; allowed: {METHODS}")
    args.methods = ms
    return args


# ── Runners ────────────────────────────────────────────────────────────

def run_vanilla(question, context, client, max_wall):
    prompt = (
        f"Answer the following question based on the provided context. "
        f"Give only the answer, no explanation.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    start = time.time()
    with wall_clock_limit(max_wall):
        r = client.chat.completions.create(
            model=PARENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    elapsed = time.time() - start
    return r.choices[0].message.content.strip(), elapsed, {
        "input_tokens": r.usage.prompt_tokens,
        "output_tokens": r.usage.completion_tokens,
    }


def build_rlm(spec, logger):
    if spec.majority_n > 1:
        install_majority_vote(spec.majority_n)
    backend = build_memory(spec.memory) if spec.memory else None
    tools = None
    counters = {"write": 0, "read": 0}
    if backend is not None:
        tools = make_memory_tools(backend)
        tools, counters = wrap_tools_with_counters(tools)
    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": PARENT_MODEL},
        other_backends=["openai"],
        other_backend_kwargs=[{"model_name": CHILD_MODEL}],
        environment="local",
        max_depth=MAX_DEPTH,
        max_iterations=MAX_ITERATIONS,
        verbose=True,
        logger=logger,
        custom_tools=tools,
        custom_sub_tools=tools,
    )
    return rlm, backend, counters


def run_rlm(question, context, rlm_instance, max_wall, memory_enabled):
    root_prompt = question + (MEMORY_HINT if memory_enabled else "")
    start = time.time()
    with wall_clock_limit(max_wall):
        res = rlm_instance.completion(prompt=context, root_prompt=root_prompt)
    elapsed = time.time() - start
    usage = res.usage_summary.to_dict() if res.usage_summary else {}
    return res.response, elapsed, usage


# ── Main ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    if args.output is None:
        ts = dt_datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/musique_sweep_{ts}.jsonl"

    tasks = load_tasks(args.num_tasks)
    if not tasks:
        print("No tasks found!")
        return

    print(f"\n=== MuSiQue sweep ===")
    print(f"Methods: {args.methods}")
    print(f"Tasks:   {len(tasks)}")
    print(f"Wall-clock cap: {args.max_wall_s}s per task")
    print(f"Output:  {args.output}\n")

    vanilla_client = OpenAI()
    logger = RLMLogger(log_dir="./logs")

    for ti, task in enumerate(tasks):
        question = task["question"]
        context = format_context(task["paragraphs"])
        gold = task["answer"]
        aliases = task.get("answer_aliases", []) or []
        n_hops = len(task["question_decomposition"])

        print(f"\n{'='*70}\nTASK {ti+1}/{len(tasks)} ({n_hops}-hop)  {question}\n  gold={gold!r}\n{'='*70}")

        for method_name in args.methods:
            spec = parse_method(method_name, majority_n=args.majority_n)
            print(f"\n--- method={method_name} ---")

            row = {
                "task_id": task["id"],
                "question": question,
                "gold_answer": gold,
                "answer_aliases": aliases,
                "n_hops": n_hops,
                "n_paragraphs": len(task["paragraphs"]),
                "method": method_name,
                "majority_n": spec.majority_n,
                "memory_backend": spec.memory,
            }

            try:
                if not spec.use_rlm:
                    ans, t, usage = run_vanilla(question, context, vanilla_client, args.max_wall_s)
                    counters = {"write": 0, "read": 0}
                else:
                    rlm, backend, counters = build_rlm(spec, logger)
                    try:
                        ans, t, usage = run_rlm(question, context, rlm, args.max_wall_s, memory_enabled=backend is not None)
                    finally:
                        try:
                            rlm.close()
                        except Exception:
                            pass
                        if backend is not None:
                            try:
                                backend.reset()
                            except Exception:
                                pass

                scores = score_prediction(ans, gold, aliases)
                print(f"  answer={ans[:100]!r}  EM={scores['em']} F1={scores['f1']:.3f} t={t:.1f}s")
                row.update({
                    "answer": ans,
                    "em": scores["em"],
                    "f1": scores["f1"],
                    "time_s": t,
                    "usage": usage,
                    "memory_calls_write": counters["write"],
                    "memory_calls_read": counters["read"],
                    "error": None,
                })
            except TaskTimeout as e:
                print(f"  TIMEOUT: {e}")
                row.update({"answer": str(e), "em": 0, "f1": 0.0, "time_s": args.max_wall_s, "usage": {}, "memory_calls_write": 0, "memory_calls_read": 0, "error": "timeout"})
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                row.update({"answer": str(e), "em": 0, "f1": 0.0, "time_s": 0, "usage": {}, "memory_calls_write": 0, "memory_calls_read": 0, "error": f"{type(e).__name__}: {e}"})

            row["timestamp"] = dt_datetime.now().isoformat()
            with open(args.output, "a") as f:
                f.write(json.dumps(row) + "\n")

    print(f"\nAll done. Results → {args.output}")


if __name__ == "__main__":
    main()
