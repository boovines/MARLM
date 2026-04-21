"""OOLONG benchmark — runs multiple methods (vanilla / RLM / majority-vote /
memory-augmented RLM) over the same OOLONG-agnews tasks.

Methods (pick via --methods a,b,c):
    vanilla             gpt-5, direct call, no RLM
    rlm                 RLM (gpt-5 parent + gpt-5-mini children)
    rlm_majority        RLM + N-way majority vote on every sub-call
    rlm_memory_flat     RLM + flat-KV persistent memory (write/read tools)
    rlm_memory_graph    RLM + Graphiti-Kuzu knowledge graph memory

Usage:
    python benchmark_oolong.py \\
        --methods vanilla,rlm,rlm_memory_flat \\
        --num-tasks 3 --context-len 16384 --max-wall-s 600
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import time
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
DATASET_NAME = "agnews"
MAX_ITERATIONS = 15
MAX_DEPTH = 1


# ── Scoring (from oolong eval_helpers.py) ──────────────────────────────

def synth_attempt_answer_parse(answer):
    parse_confidence = "low"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, parse_confidence
        else:
            return answer.split()[-1], parse_confidence
    candidate_answer = answer.split(":")[-1].strip()
    candidate_answer = candidate_answer.replace("*", "").replace("[", "").replace("]", "")
    parse_confidence = "med"
    if "User:" in answer or "Answer:" in answer or "Date:" in answer or "Label" in answer:
        parse_confidence = "high"
    if len(candidate_answer) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate_answer:
        candidate_answer = "more common"
    elif "less common" in candidate_answer:
        candidate_answer = "less common"
    elif "same frequency" in candidate_answer:
        candidate_answer = "same frequency"
    return candidate_answer, parse_confidence


def synth_process_response(datapoint, output, model):
    import dateutil.parser
    score = 0
    gold = (
        ast.literal_eval(datapoint["answer"])[0]
        if "datetime" not in datapoint["answer"]
        else dt_datetime.strptime(datapoint["answer"], "[datetime.date(%Y, %m, %d)]")
    )
    trimmed_output, parse_confidence = synth_attempt_answer_parse(output)
    if str(trimmed_output) == str(gold):
        score = 1
    elif str(trimmed_output) in ["more common", "less common", "same frequency"]:
        if str(trimmed_output) in str(gold):
            score = 1
    elif datapoint["answer_type"] == "ANSWER_TYPE.NUMERIC":
        try:
            score = 0.75 ** (abs(int(gold) - int(trimmed_output)))
            trimmed_output = int(trimmed_output)
            gold = int(gold)
        except Exception:
            parse_confidence = "low"
    elif datapoint["answer_type"] == "ANSWER_TYPE.DATE":
        try:
            trimmed_output = dateutil.parser.parse(trimmed_output)
            score = trimmed_output == gold
        except Exception:
            parse_confidence = "low"
    return {
        "attempted_parse": str(trimmed_output),
        "parse_confidence": parse_confidence,
        "full_answer": output,
        "score": float(score),
        "gold": str(gold),
    }


# ── Args ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--methods",
        type=str,
        default="vanilla,rlm,rlm_memory_flat",
        help=f"comma-separated subset of {METHODS}",
    )
    p.add_argument("--num-tasks", type=int, default=3)
    p.add_argument("--context-len", type=int, default=16384)
    p.add_argument("--max-wall-s", type=int, default=600, help="per-task wall-clock cap (seconds); 0 disables")
    p.add_argument("--majority-n", type=int, default=3)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise SystemExit(f"unknown method(s): {bad}; allowed: {METHODS}")
    args.methods = methods
    return args


# ── Tasks ───────────────────────────────────────────────────────────────

def load_tasks(num_tasks, context_len):
    print(f"Loading OOLONG-synth from HuggingFace...")
    ds = load_dataset("oolongbench/oolong-synth", split="test")
    filtered = ds.filter(
        lambda x: (
            x["dataset"] == DATASET_NAME
            and x["context_len"] == context_len
            and x["answer_type"] == "ANSWER_TYPE.NUMERIC"
        )
    )
    print(f"Found {len(filtered)} numeric tasks at {context_len}; using {min(num_tasks, len(filtered))}")
    return [filtered[i] for i in range(min(num_tasks, len(filtered)))]


# ── Runners ─────────────────────────────────────────────────────────────

def run_vanilla(task, client, max_wall):
    prompt = f"{task['context_window_text']}\n\n{task['question']}"
    start = time.time()
    with wall_clock_limit(max_wall):
        r = client.chat.completions.create(
            model=PARENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    elapsed = time.time() - start
    return r.choices[0].message.content, elapsed, {
        "input_tokens": r.usage.prompt_tokens,
        "output_tokens": r.usage.completion_tokens,
    }


def build_rlm(spec, logger):
    """Build an RLM instance for the given MethodSpec.

    For rlm_majority, install the majority-vote environment monkey-patch first.
    For memory variants, build the backend + tools.
    Returns (rlm_instance, memory_backend_or_None, counters_dict, custom_tools_or_None)."""
    if spec.majority_n > 1:
        install_majority_vote(spec.majority_n)

    backend = build_memory(spec.memory) if spec.memory else None
    custom_tools = None
    counters = {"write": 0, "read": 0}
    if backend is not None:
        custom_tools = make_memory_tools(backend)
        custom_tools, counters = wrap_tools_with_counters(custom_tools)

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
        custom_tools=custom_tools,
        custom_sub_tools=custom_tools,
    )
    return rlm, backend, counters


def run_rlm(task, rlm_instance, max_wall):
    start = time.time()
    with wall_clock_limit(max_wall):
        res = rlm_instance.completion(prompt=task["context_window_text"], root_prompt=task["question"])
    elapsed = time.time() - start
    usage = res.usage_summary.to_dict() if res.usage_summary else {}
    return res.response, elapsed, usage


# ── Main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if args.output is None:
        ts = dt_datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/oolong_sweep_{ts}.jsonl"

    tasks = load_tasks(args.num_tasks, args.context_len)
    if not tasks:
        print("No tasks found!")
        return

    print(f"\n=== OOLONG sweep ===")
    print(f"Methods: {args.methods}")
    print(f"Tasks:   {len(tasks)} at {args.context_len} tokens")
    print(f"Wall-clock cap: {args.max_wall_s}s per task")
    print(f"Output:  {args.output}\n")

    vanilla_client = OpenAI()
    logger = RLMLogger(log_dir="./logs")

    for ti, task in enumerate(tasks):
        print(f"\n{'='*70}\nTASK {ti+1}/{len(tasks)}  Q={task['question'][:70]}\n  gold={task['answer']}\n{'='*70}")

        for method_name in args.methods:
            spec = parse_method(method_name, majority_n=args.majority_n)
            print(f"\n--- method={method_name} ---")

            row = {
                "task_id": task["id"],
                "context_window_id": task["context_window_id"],
                "dataset": task["dataset"],
                "context_len": task["context_len"],
                "question": task["question"],
                "gold_answer": task["answer"],
                "method": method_name,
                "majority_n": spec.majority_n,
                "memory_backend": spec.memory,
            }

            try:
                if not spec.use_rlm:
                    ans, t, usage = run_vanilla(task, vanilla_client, args.max_wall_s)
                    counters = {"write": 0, "read": 0}
                else:
                    rlm, backend, counters = build_rlm(spec, logger)
                    try:
                        ans, t, usage = run_rlm(task, rlm, args.max_wall_s)
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

                score = synth_process_response(task, ans, method_name)
                print(f"  parsed={score['attempted_parse']!r} score={score['score']:.3f} t={t:.1f}s")
                row.update({
                    "answer": ans,
                    "parsed": score["attempted_parse"],
                    "parse_confidence": score["parse_confidence"],
                    "score": score["score"],
                    "time_s": t,
                    "usage": usage,
                    "memory_calls_write": counters["write"],
                    "memory_calls_read": counters["read"],
                    "error": None,
                })
            except TaskTimeout as e:
                print(f"  TIMEOUT: {e}")
                row.update({"answer": str(e), "parsed": None, "score": 0.0, "time_s": args.max_wall_s, "usage": {}, "memory_calls_write": 0, "memory_calls_read": 0, "error": "timeout"})
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                row.update({"answer": str(e), "parsed": None, "score": 0.0, "time_s": 0, "usage": {}, "memory_calls_write": 0, "memory_calls_read": 0, "error": f"{type(e).__name__}: {e}"})

            row["timestamp"] = dt_datetime.now().isoformat()
            with open(args.output, "a") as f:
                f.write(json.dumps(row) + "\n")

    print(f"\nAll done. Results → {args.output}")


if __name__ == "__main__":
    main()
