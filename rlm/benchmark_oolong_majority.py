"""
OOLONG benchmark: majority-vote RLM variant.

Each llm_query sub-call is run N_VOTES times via llm_query_vote(),
and the most common response is returned.

Runs RLM majority only (use benchmark_oolong.py for vanilla + standard RLM).

Usage:
    python benchmark_oolong_majority.py
"""

import os
import json
import time
import ast
from collections import Counter
from datetime import datetime as dt_datetime
from datasets import load_dataset
from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────
PARENT_MODEL = "gpt-5"
CHILD_MODEL = "gpt-5-mini"
DATASET_NAME = "agnews"
CONTEXT_LEN = 16384              # 16K — vanilla GPT-5 starts degrading
NUM_TASKS = 2
MAX_ITERATIONS = 15
MAX_DEPTH = 1
N_VOTES = 3
OUTPUT_FILE = "results/phase1_oolong_majority.jsonl"
# ────────────────────────────────────────────────────────────────────────


# ── Scoring (from oolong eval_helpers.py, no litellm dependency) ────────
def synth_attempt_answer_parse(answer):
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


# ── Majority vote tool ──────────────────────────────────────────────────
from rlm.environments.local_repl import LocalREPL
from rlm.environments import get_environment
from rlm.environments.local_repl import send_lm_request_batched


class MajorityVoteREPL(LocalREPL):
    """
    LocalREPL subclass where llm_query and llm_query_batched transparently
    run each prompt N_VOTES times and return the majority-voted answer.

    The parent model's code is UNCHANGED — it still calls llm_query() and
    llm_query_batched() as normal. The voting happens under the hood.
    """

    def __init__(self, *args, n_votes=3, **kwargs):
        self.n_votes = n_votes
        super().__init__(*args, **kwargs)

    def _llm_query(self, prompt, model=None):
        """Single prompt → N calls → majority vote."""
        responses = self._raw_batched([prompt] * self.n_votes, model)
        return Counter(responses).most_common(1)[0][0]

    def _llm_query_batched(self, prompts, model=None):
        """List of prompts → each run N times → majority vote per prompt."""
        # Expand all prompts × N_VOTES into one big concurrent batch
        expanded = []
        for p in prompts:
            expanded.extend([p] * self.n_votes)

        all_responses = self._raw_batched(expanded, model)

        # Majority vote per original prompt
        results = []
        for i in range(len(prompts)):
            votes = all_responses[i * self.n_votes : (i + 1) * self.n_votes]
            winner = Counter(votes).most_common(1)[0][0]
            results.append(winner)
        return results

    def _raw_batched(self, prompts, model=None):
        """The original batched call without voting (calls parent's implementation)."""
        if not self.lm_handler_address:
            return ["Error: No LM handler configured"] * len(prompts)
        try:
            responses = send_lm_request_batched(
                self.lm_handler_address, prompts, model=model, depth=self.depth
            )
            results = []
            for response in responses:
                if not response.success:
                    results.append(f"Error: {response.error}")
                else:
                    self._pending_llm_calls.append(response.chat_completion)
                    results.append(response.chat_completion.response)
            return results
        except Exception as e:
            return [f"Error: LM query failed - {e}"] * len(prompts)


# ── Main ────────────────────────────────────────────────────────────────
def load_tasks():
    print(f"Loading OOLONG-synth from HuggingFace...")
    ds = load_dataset("oolongbench/oolong-synth", split="test")
    filtered = ds.filter(
        lambda x: (
            x["dataset"] == DATASET_NAME
            and x["context_len"] == CONTEXT_LEN
            and x["answer_type"] == "ANSWER_TYPE.NUMERIC"
        )
    )
    print(f"Found {len(filtered)} numeric tasks for {DATASET_NAME} at {CONTEXT_LEN} tokens, taking {min(NUM_TASKS, len(filtered))}")
    return [filtered[i] for i in range(min(NUM_TASKS, len(filtered)))]


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    tasks = load_tasks()
    if not tasks:
        print("No tasks found!")
        return

    for i, t in enumerate(tasks):
        print(f"  Task {i+1}: {t['question'][:80]}...")
        print(f"    Gold: {t['answer']}")
        print()

    # ── Setup RLM with majority-vote REPL ──────────────────────────────
    logger = RLMLogger(log_dir="./logs")

    # Monkey-patch the environment factory so "local" returns our subclass
    import rlm.environments as env_module
    _original_get_env = env_module.get_environment

    def _majority_get_environment(env_type, env_kwargs):
        if env_type == "local":
            return MajorityVoteREPL(n_votes=N_VOTES, **env_kwargs)
        return _original_get_env(env_type, env_kwargs)

    env_module.get_environment = _majority_get_environment

    # Standard RLM — no prompt changes needed. The parent uses llm_query
    # and llm_query_batched as normal. Voting is transparent.
    rlm_majority = RLM(
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

    # ── Run ─────────────────────────────────────────────────────────────
    results = []

    for i, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"TASK {i+1}/{len(tasks)} (context: {task['context_len']} tokens)")
        print(f"Question: {task['question']}")
        print(f"Gold: {task['answer']}")
        print(f"{'='*60}")

        print(f"\n--- RLM majority-vote x{N_VOTES} ({PARENT_MODEL} + {CHILD_MODEL}) ---")
        try:
            start = time.time()
            res = rlm_majority.completion(
                prompt=task["context_window_text"],
                root_prompt=task["question"],
            )
            elapsed = time.time() - start
            usage = res.usage_summary.to_dict() if res.usage_summary else {}

            score = synth_process_response(task, res.response, f"RLM-majority({PARENT_MODEL})")
            print(f"  Answer: {score['attempted_parse']}")
            print(f"  Score: {score['score']}")
            print(f"  Time: {elapsed:.2f}s")

            result = {
                "task_id": task["id"],
                "dataset": task["dataset"],
                "context_len": task["context_len"],
                "question": task["question"],
                "gold_answer": task["answer"],
                "task_type": task["task"],
                "answer_type": task["answer_type"],
                "method": "rlm_majority",
                "n_votes": N_VOTES,
                "answer": res.response,
                "score": score["score"],
                "parsed": score["attempted_parse"],
                "confidence": score["parse_confidence"],
                "time_s": elapsed,
                "usage": usage,
                "timestamp": dt_datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                "task_id": task["id"],
                "dataset": task["dataset"],
                "context_len": task["context_len"],
                "question": task["question"],
                "gold_answer": task["answer"],
                "method": "rlm_majority",
                "error": str(e),
                "score": 0,
                "time_s": 0,
                "timestamp": dt_datetime.now().isoformat(),
            }

        results.append(result)
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    scores = [r["score"] for r in results]
    times = [r["time_s"] for r in results]
    print(f"Dataset: {DATASET_NAME}, Context: {CONTEXT_LEN} tokens, Tasks: {len(results)}")
    print(f"Method: RLM majority x{N_VOTES} ({PARENT_MODEL} + {CHILD_MODEL})")
    print(f"  Avg score: {sum(scores)/len(scores):.3f}")
    print(f"  Avg time:  {sum(times)/len(times):.2f}s")
    print(f"\nResults saved to {OUTPUT_FILE}")

    rlm_majority.close()


if __name__ == "__main__":
    main()
