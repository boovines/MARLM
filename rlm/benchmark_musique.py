"""
MuSiQue benchmark: multi-hop QA that resists RLM's decomposition strategy.

Tests the hypothesis that OOLONG is cherry-picked for RLM's strengths.
MuSiQue requires chaining 2-4 facts across paragraphs — each hop's answer
is the input to the next hop. Independent chunking should hurt here.

Runs:
  1. Vanilla GPT-5 (direct call, full context)
  2. RLM (GPT-5 parent + GPT-5-mini children)

Scoring: SQuAD-style EM and F1 (max over answer aliases).

Usage:
    python benchmark_musique.py
"""

import os
import re
import json
import time
import string
from collections import Counter
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

from rlm import RLM
from rlm.logger import RLMLogger
from datetime import datetime as dt_datetime

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────
PARENT_MODEL = "gpt-5"           # root RLM model
CHILD_MODEL = "gpt-5-mini"      # llm_query sub-call model
NUM_TASKS = 5                    # how many examples to evaluate
NUM_HOPS = None                  # None = all, or 2/3/4 to filter
MAX_ITERATIONS = 15              # RLM iteration cap
MAX_DEPTH = 1                    # paper setting
OUTPUT_FILE = "results/musique_benchmark.jsonl"
# ────────────────────────────────────────────────────────────────────────


# ── SQuAD-style scoring ────────────────────────────────────────────────

def normalize_answer(s):
    """Lowercase, strip articles/punctuation/whitespace."""
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    s = ' '.join(s.split())
    return s


def exact_match(prediction, gold_answers):
    """EM: 1 if normalized prediction matches any gold answer."""
    norm_pred = normalize_answer(prediction)
    return max(int(normalize_answer(g) == norm_pred) for g in gold_answers)


def f1_score(prediction, gold_answers):
    """Token-level F1, max over gold answers."""
    norm_pred = normalize_answer(prediction)
    pred_tokens = norm_pred.split()

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens) if pred_tokens else 0
        recall = num_common / len(gold_tokens) if gold_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        best_f1 = max(best_f1, f1)

    return best_f1


def score_prediction(prediction, gold_answer, gold_aliases):
    """Score a prediction against gold answer + aliases."""
    all_golds = [gold_answer] + (gold_aliases or [])
    return {
        "em": exact_match(prediction, all_golds),
        "f1": f1_score(prediction, all_golds),
    }


# ── Data loading ───────────────────────────────────────────────────────

def format_context(paragraphs):
    """Concatenate paragraphs into a single context string."""
    parts = []
    for i, p in enumerate(paragraphs):
        title = p.get("title", f"Paragraph {i+1}")
        text = p["paragraph_text"]
        parts.append(f"[{title}]\n{text}")
    return "\n\n".join(parts)


def load_tasks():
    """Load MuSiQue answerable tasks from HuggingFace."""
    print("Loading MuSiQue from HuggingFace...")
    ds = load_dataset("bdsaglam/musique", "answerable", split="validation")
    print(f"  Total validation examples: {len(ds)}")

    # Optional: filter by number of hops
    if NUM_HOPS is not None:
        ds = ds.filter(lambda x: len(x["question_decomposition"]) == NUM_HOPS)
        print(f"  After filtering to {NUM_HOPS}-hop: {len(ds)}")

    # Take first NUM_TASKS
    n = min(NUM_TASKS, len(ds))
    tasks = [ds[i] for i in range(n)]
    print(f"  Using {n} tasks")

    # Print hop distribution
    hop_counts = Counter(len(t["question_decomposition"]) for t in tasks)
    for h, c in sorted(hop_counts.items()):
        print(f"    {h}-hop: {c}")

    return tasks


# ── Runners ────────────────────────────────────────────────────────────

def run_vanilla_baseline(question, context, client):
    """Run vanilla GPT-5 on the full context."""
    prompt = f"Answer the following question based on the provided context. Give only the answer, no explanation.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    start = time.time()
    response = client.chat.completions.create(
        model=PARENT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - start

    answer = response.choices[0].message.content.strip()
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return answer, elapsed, usage


def run_rlm(question, context, rlm_instance):
    """Run RLM on the task."""
    start = time.time()
    result = rlm_instance.completion(
        prompt=context,
        root_prompt=question,
    )
    elapsed = time.time() - start

    usage = result.usage_summary.to_dict() if result.usage_summary else {}
    return result.response, elapsed, usage


# ── Main ───────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    tasks = load_tasks()
    if not tasks:
        print("No tasks found!")
        return

    # Preview
    for i, t in enumerate(tasks[:3]):
        n_hops = len(t["question_decomposition"])
        print(f"  Example {i+1} ({n_hops}-hop): {t['question']}")
        print(f"    Answer: {t['answer']}")
        print(f"    Paragraphs: {len(t['paragraphs'])}")
        print()

    # ── Setup RLM ───────────────────────────────────────────────────────
    logger = RLMLogger(log_dir="./logs")
    rlm_instance = RLM(
        backend="openai",
        backend_kwargs={"model_name": PARENT_MODEL},
        other_backends=["openai"],
        other_backend_kwargs=[{"model_name": CHILD_MODEL}],
        environment="local",
        max_depth=MAX_DEPTH,
        max_iterations=MAX_ITERATIONS,
        verbose=True,
        logger=logger,
    )

    vanilla_client = OpenAI()

    # ── Run benchmarks ──────────────────────────────────────────────────
    results = []

    for i, task in enumerate(tasks):
        question = task["question"]
        context = format_context(task["paragraphs"])
        gold = task["answer"]
        aliases = task.get("answer_aliases", [])
        n_hops = len(task["question_decomposition"])

        print(f"\n{'='*60}")
        print(f"TASK {i+1}/{len(tasks)} ({n_hops}-hop)")
        print(f"Question: {question}")
        print(f"Gold: {gold}")
        if aliases:
            print(f"Aliases: {aliases}")
        print(f"{'='*60}")

        result = {
            "task_id": task["id"],
            "question": question,
            "gold_answer": gold,
            "answer_aliases": aliases,
            "n_hops": n_hops,
            "n_paragraphs": len(task["paragraphs"]),
            "decomposition": [
                {"question": d["question"], "answer": d["answer"]}
                for d in task["question_decomposition"]
            ],
        }

        # ── Vanilla baseline ────────────────────────────────────────────
        print(f"\n--- Vanilla {PARENT_MODEL} ---")
        try:
            ans, t_elapsed, usage = run_vanilla_baseline(question, context, vanilla_client)
            scores = score_prediction(ans, gold, aliases)
            print(f"  Answer: {ans}")
            print(f"  EM: {scores['em']}, F1: {scores['f1']:.3f}")
            print(f"  Time: {t_elapsed:.2f}s")
            result["vanilla"] = {
                "answer": ans, **scores,
                "time_s": t_elapsed, "usage": usage,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            result["vanilla"] = {"answer": str(e), "em": 0, "f1": 0, "time_s": 0}

        # ── RLM ─────────────────────────────────────────────────────────
        print(f"\n--- RLM ({PARENT_MODEL} + {CHILD_MODEL}) ---")
        try:
            ans, t_elapsed, usage = run_rlm(question, context, rlm_instance)
            scores = score_prediction(ans, gold, aliases)
            print(f"  Answer: {ans}")
            print(f"  EM: {scores['em']}, F1: {scores['f1']:.3f}")
            print(f"  Time: {t_elapsed:.2f}s")
            result["rlm"] = {
                "answer": ans, **scores,
                "time_s": t_elapsed, "usage": usage,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            result["rlm"] = {"answer": str(e), "em": 0, "f1": 0, "time_s": 0}

        result["timestamp"] = dt_datetime.now().isoformat()
        results.append(result)

        # Write incrementally
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Tasks: {len(results)}")

    for method in ["vanilla", "rlm"]:
        label = f"Vanilla {PARENT_MODEL}" if method == "vanilla" else f"RLM ({PARENT_MODEL} + {CHILD_MODEL})"
        ems = [r[method]["em"] for r in results if method in r]
        f1s = [r[method]["f1"] for r in results if method in r]
        times = [r[method]["time_s"] for r in results if method in r]

        if ems:
            print(f"\n  {label}:")
            print(f"    EM:       {sum(ems)/len(ems):.3f} ({sum(ems)}/{len(ems)})")
            print(f"    Avg F1:   {sum(f1s)/len(f1s):.3f}")
            print(f"    Avg time: {sum(times)/len(times):.2f}s")

    # Per-hop breakdown
    hop_values = sorted(set(r["n_hops"] for r in results))
    if len(hop_values) > 1:
        print(f"\n  Per-hop breakdown:")
        for n_hop in hop_values:
            subset = [r for r in results if r["n_hops"] == n_hop]
            for method in ["vanilla", "rlm"]:
                ems = [r[method]["em"] for r in subset if method in r]
                f1s = [r[method]["f1"] for r in subset if method in r]
                label = "Vanilla" if method == "vanilla" else "RLM"
                if ems:
                    print(f"    {n_hop}-hop {label}: EM={sum(ems)/len(ems):.3f}, F1={sum(f1s)/len(f1s):.3f} (n={len(ems)})")

    print(f"\nResults saved to {OUTPUT_FILE}")
    rlm_instance.close()


if __name__ == "__main__":
    main()
