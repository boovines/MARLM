"""Memory tool factory — wraps a MemoryBackend as RLM custom_tools.

Per SPRINT.md M0. The tool descriptions here are load-bearing: they are
what the parent LLM reads to decide when to call memory_write / memory_read.
"""
from __future__ import annotations

from typing import Any

from rlm.memory.base import MemoryBackend


def make_memory_tools(backend: MemoryBackend) -> dict[str, Any]:
    """Return a custom_tools dict exposing memory_write/memory_read over `backend`.

    Shape matches RLM's custom_tools contract (see rlm/environments/base_env.py::parse_tool_entry):
        {"name": {"tool": callable, "description": str}}
    """

    def _memory_write(key: str, value: str) -> None:
        backend.write(key, value)

    def _memory_read(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return [
            {"key": h.key, "value": h.value, "score": h.score}
            for h in backend.read(query, top_k=top_k)
        ]

    return {
        "memory_write": {
            "tool": _memory_write,
            "description": (
                "Store a string value under a string key in persistent memory "
                "that survives across llm_query calls and across iterations. "
                "Use to stash intermediate findings (e.g. 'hop1_performer' -> "
                "'Steve Hillage') so later sub-calls can retrieve them. "
                "Writes are expensive for the graph backend (~10s each); "
                "batch many items into one value when possible."
            ),
        },
        "memory_read": {
            "tool": _memory_read,
            "description": (
                "Retrieve previously stored memory entries most relevant to "
                "`query`. Returns a list of {key, value, score} dicts. "
                "Call BEFORE writing to check if relevant context exists."
            ),
        },
    }
