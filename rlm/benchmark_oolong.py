"""
Phase 1: OOLONG benchmark — small test with agnews at 1024 tokens.
Runs RLM (GPT-5 parent + GPT-5-mini children) and vanilla baseline,
scores with OoLong eval helpers.

Usage:
    python benchmark_oolong.py [--memory {none,flat,graph}] [--num-tasks N]
                               [--context-len N] [--output PATH]
"""

import os
import json
import time
import argparse
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

from rlm import RLM
from rlm.logger import RLMLogger
from rlm.bench.memory_factory import build_memory

import ast
from datetime import datetime as dt_datetime


def synth_attempt_answer_parse(answer):
    """Parse answer from model output. Copied from oolong eval_helpers.py to avoid litellm dependency."""
    parse_confidence = "low"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, parse_confidence
        else:
            return answer.split()[-1], parse_confidence
    candidate_answer = answer.split(":")[-1].strip()
    candidate_answer = candidate_answer.replace("*", "")
    candidate_answer = candidate_answer.replace("[", "")
    candidate_answer = candidate_answer.replace("]", "")
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
    """Score model output against gold answer. Copied from oolong eval_helpers.py."""
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
            trimmed_output = int(trimmed_output)
            gold = int(gold)
            score = 0.75 ** (abs(gold - trimmed_output))
        except Exception:
            parse_confidence = "low"
    elif datapoint["answer_type"] == "ANSWER_TYPE.DATE":
        try:
            trimmed_output = dateutil.parser.parse(trimmed_output)
            score = trimmed_output == gold
        except Exception:
            parse_confidence = "low"

    return {
        "id": datapoint["id"],
        "context_window_id": datapoint["context_window_id"],
        "dataset": datapoint["dataset"],
        "model": model,
        "attempted_parse": str(trimmed_output),
        "parse_confidence": parse_confidence,
        "full_answer": output,
        "score": score,
        "answer": str(gold),
    }

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────
PARENT_MODEL = "gpt-5"           # root RLM model
CHILD_MODEL = "gpt-5-mini"      # llm_query sub-call model
DATASET_NAME = "agnews"          # which OOLONG dataset
MAX_ITERATIONS = 15              # RLM iteration cap
MAX_DEPTH = 1                    # paper setting
# ────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--memory", choices=["none", "flat", "graph"], default="none")
    p.add_argument("--num-tasks", type=int, default=3)
    p.add_argument("--context-len", type=int, default=16384)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def load_tasks(num_tasks, context_len):
    """Load OOLONG-synth numeric tasks from HuggingFace (partial credit via 0.75^|error|)."""
    print(f"Loading OOLONG-synth from HuggingFace...")
    ds = load_dataset("oolongbench/oolong-synth", split="test")

    filtered = ds.filter(
        lambda x: (
            x["dataset"] == DATASET_NAME
            and x["context_len"] == context_len
            and x["answer_type"] == "ANSWER_TYPE.NUMERIC"
        )
    )
    print(f"Found {len(filtered)} numeric tasks for {DATASET_NAME} at {context_len} tokens")

    tasks = [filtered[i] for i in range(min(num_tasks, len(filtered)))]
    return tasks


def wrap_tools_with_counters(tools):
    """Wrap memory_write/memory_read tool callables with a counter.
    Returns (tools_dict, counters_dict). Mutates input dict."""
    counters = {"write": 0, "read": 0}
    if not tools:
        return tools, counters

    def make_wrapper(fn, key):
        def wrapper(*args, **kwargs):
            counters[key] += 1
            return fn(*args, **kwargs)
        return wrapper

    for name, entry in list(tools.items()):
        if "write" in name:
            counter_key = "write"
        elif "read" in name:
            counter_key = "read"
        else:
            continue

        if isinstance(entry, dict) and "tool" in entry:
            entry["tool"] = make_wrapper(entry["tool"], counter_key)
        elif callable(entry):
            tools[name] = make_wrapper(entry, counter_key)

    return tools, counters


def run_vanilla_baseline(task, client):
    """Run vanilla GPT-5 on the full context (no RLM)."""
    context = task["context_window_text"]
    question = task["question"]
    prompt = f"{context}\n\n{question}"

    start = time.time()
    response = client.chat.completions.create(
        model=PARENT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - start

    answer = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return answer, elapsed, usage


def run_rlm(task, rlm_instance):
    """Run RLM on the task."""
    context = task["context_window_text"]
    question = task["question"]

    start = time.time()
    result = rlm_instance.completion(
        prompt=context,
        root_prompt=question,
    )
    elapsed = time.time() - start

    usage = result.usage_summary.to_dict() if result.usage_summary else {}
    return result.response, elapsed, usage


def main():
    args = parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if args.output is None:
        ts = dt_datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/oolong_{args.memory}_{ts}.jsonl"
    else:
        output_file = args.output

    # Load tasks
    tasks = load_tasks(args.num_tasks, args.context_len)
    if not tasks:
        print("No tasks found! Check dataset name and context length.")
        return

    # Print task info
    for i, t in enumerate(tasks):
        print(f"  Task {i+1}: {t['question'][:80]}...")
        print(f"    Gold answer: {t['answer']}")
        print(f"    Task type: {t['task']}, Answer type: {t['answer_type']}")
        print()

    # ── Setup memory backend + tools ────────────────────────────────────
    backend = build_memory(args.memory)
    custom_tools = None
    counters = {"write": 0, "read": 0}
    if backend is not None:
        from rlm.memory.tools import make_memory_tools
        custom_tools = make_memory_tools(backend)
        custom_tools, counters = wrap_tools_with_counters(custom_tools)

    # ── Setup RLM ───────────────────────────────────────────────────────
    logger = RLMLogger(log_dir="./logs")

    rlm_instance = RLM(
        backend="openai",
        backend_kwargs={"model_name": PARENT_MODEL},
        # Use GPT-5-mini for sub-calls
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

    # ── Setup vanilla baseline client ───────────────────────────────────
    vanilla_client = OpenAI()

    # ── Run benchmarks ──────────────────────────────────────────────────
    results = []

    for i, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"TASK {i+1}/{len(tasks)}")
        print(f"Question: {task['question']}")
        print(f"Gold: {task['answer']}")
        print(f"{'='*60}")

        # Reset memory between tasks to prevent cross-task bleed
        if backend is not None:
            backend.reset()
        counters["write"] = 0
        counters["read"] = 0

        # Vanilla baseline
        print(f"\n--- Vanilla {PARENT_MODEL} ---")
        try:
            vanilla_answer, vanilla_time, vanilla_usage = run_vanilla_baseline(task, vanilla_client)
            vanilla_score = synth_process_response(task, vanilla_answer, PARENT_MODEL)
            print(f"  Answer: {vanilla_score['attempted_parse']}")
            print(f"  Score: {vanilla_score['score']}")
            print(f"  Time: {vanilla_time:.2f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            vanilla_answer, vanilla_time, vanilla_usage = str(e), 0, {}
            vanilla_score = {"score": 0, "attempted_parse": str(e), "parse_confidence": "error"}

        # RLM
        print(f"\n--- RLM ({PARENT_MODEL} + {CHILD_MODEL}) ---")
        try:
            rlm_answer, rlm_time, rlm_usage = run_rlm(task, rlm_instance)
            rlm_score = synth_process_response(task, rlm_answer, f"RLM({PARENT_MODEL})")
            print(f"  Answer: {rlm_score['attempted_parse']}")
            print(f"  Score: {rlm_score['score']}")
            print(f"  Time: {rlm_time:.2f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            rlm_answer, rlm_time, rlm_usage = str(e), 0, {}
            rlm_score = {"score": 0, "attempted_parse": str(e), "parse_confidence": "error"}

        # Record result
        result = {
            "task_id": task["id"],
            "dataset": task["dataset"],
            "context_len": task["context_len"],
            "question": task["question"],
            "gold_answer": task["answer"],
            "task_type": task["task"],
            "answer_type": task["answer_type"],
            "memory_backend": args.memory,
            "memory_calls_write": counters["write"],
            "memory_calls_read": counters["read"],
            "vanilla": {
                "answer": vanilla_answer,
                "score": vanilla_score["score"],
                "parsed": vanilla_score["attempted_parse"],
                "confidence": vanilla_score["parse_confidence"],
                "time_s": vanilla_time,
                "usage": vanilla_usage,
            },
            "rlm": {
                "answer": rlm_answer,
                "score": rlm_score["score"],
                "parsed": rlm_score["attempted_parse"],
                "confidence": rlm_score["parse_confidence"],
                "time_s": rlm_time,
                "usage": rlm_usage,
            },
            "timestamp": dt_datetime.now().isoformat(),
        }
        results.append(result)

        # Write incrementally
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    vanilla_scores = [r["vanilla"]["score"] for r in results]
    rlm_scores = [r["rlm"]["score"] for r in results]
    vanilla_times = [r["vanilla"]["time_s"] for r in results]
    rlm_times = [r["rlm"]["time_s"] for r in results]

    print(f"Dataset: {DATASET_NAME}, Context: {args.context_len} tokens, Tasks: {len(results)}")
    print(f"Memory backend: {args.memory}")
    print(f"")
    print(f"  Vanilla {PARENT_MODEL}:")
    print(f"    Avg score: {sum(vanilla_scores)/len(vanilla_scores):.3f}")
    print(f"    Avg time:  {sum(vanilla_times)/len(vanilla_times):.2f}s")
    print(f"")
    print(f"  RLM ({PARENT_MODEL} + {CHILD_MODEL}):")
    print(f"    Avg score: {sum(rlm_scores)/len(rlm_scores):.3f}")
    print(f"    Avg time:  {sum(rlm_times)/len(rlm_times):.2f}s")
    print(f"")
    print(f"Results saved to {output_file}")

    # Cleanup
    rlm_instance.close()


if __name__ == "__main__":
    main()
