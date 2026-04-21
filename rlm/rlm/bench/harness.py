"""Shared benchmark harness: method selection, wall-clock guardrail, counters.

METHODS = {vanilla, rlm, rlm_majority, rlm_memory_flat, rlm_memory_graph}

Guardrail: per-task wall-clock timeout. Trajectories exceeding the limit are
killed and recorded as `timeout` with score 0 (caller decides how to score),
so a single runaway trajectory cannot sink the whole sweep.
"""
from __future__ import annotations

import os
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

METHODS = (
    "vanilla",
    "rlm",
    "rlm_majority",
    "rlm_memory_flat",
    "rlm_memory_graph",
)


class TaskTimeout(Exception):
    """Raised when a per-task wall-clock guardrail fires."""


@contextmanager
def wall_clock_limit(seconds: int):
    """SIGALRM-based wall-clock cap. Unix only. seconds<=0 disables."""
    if seconds is None or seconds <= 0:
        yield
        return

    def _handle(signum, frame):
        raise TaskTimeout(f"task exceeded {seconds}s wall-clock limit")

    prev = signal.signal(signal.SIGALRM, _handle)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def wrap_tools_with_counters(tools: Optional[dict]) -> tuple[Optional[dict], dict]:
    """Wrap memory_write/memory_read callables with a call-counter dict.
    Mutates `tools` in place; returns (tools, counters). Safe to call with None."""
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
            ck = "write"
        elif "read" in name:
            ck = "read"
        else:
            continue
        if isinstance(entry, dict) and "tool" in entry:
            entry["tool"] = make_wrapper(entry["tool"], ck)
        elif callable(entry):
            tools[name] = make_wrapper(entry, ck)
    return tools, counters


@dataclass
class MethodSpec:
    """Parsed description of how to run one task under one method."""
    name: str                    # vanilla | rlm | rlm_majority | rlm_memory_{flat,graph}
    memory: str | None           # None | "flat" | "graph"
    majority_n: int              # 1 = no voting
    use_rlm: bool                # True for all rlm_* methods; False for vanilla


def parse_method(method: str, majority_n: int = 3) -> MethodSpec:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if method == "vanilla":
        return MethodSpec(name=method, memory=None, majority_n=1, use_rlm=False)
    if method == "rlm":
        return MethodSpec(name=method, memory=None, majority_n=1, use_rlm=True)
    if method == "rlm_majority":
        return MethodSpec(name=method, memory=None, majority_n=majority_n, use_rlm=True)
    if method == "rlm_memory_flat":
        return MethodSpec(name=method, memory="flat", majority_n=1, use_rlm=True)
    if method == "rlm_memory_graph":
        return MethodSpec(name=method, memory="graph", majority_n=1, use_rlm=True)
    raise AssertionError("unreachable")
