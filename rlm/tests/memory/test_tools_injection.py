"""End-to-end injection test: RLM -> memory tools -> FlatKVBackend.

Per SPRINT.md M0 done-criteria. Proves the whole path:
  make_memory_tools(backend)
    -> RLM(custom_tools=..., custom_sub_tools=...)
    -> .completion(prompt)
    -> parent LLM invokes memory_write/memory_read in the REPL
    -> backend store reflects the write
    -> read round-trip surfaces the value

Uses gpt-5-mini as the parent (cheap) with max_depth=1 (REPL enabled,
one level of child recursion allowed).

If this test fails with `len(backend._store) == 0`, the parent never invoked
memory_write — that is the signal that tool descriptions aren't reaching the
system prompt, independent of whether the benchmark task is hard enough to
need memory.

Note: max_depth=0 would *disable the REPL entirely* (single-shot LLM call with
no code execution), which also zeroes out the store — that's a harness
limitation, not a memory-tool bug. Always use max_depth>=1 for this test.
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from rlm import RLM
from rlm.memory import FlatKVBackend, make_memory_tools

load_dotenv()


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for RLM .completion() round-trip",
)
def test_memory_tools_round_trip() -> None:
    backend = FlatKVBackend()
    tools = make_memory_tools(backend)

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": "gpt-5-mini"},
        other_backends=["openai"],
        other_backend_kwargs=[{"model_name": "gpt-5-mini"}],
        environment="local",
        max_depth=1,
        max_iterations=5,
        verbose=False,
        custom_tools=tools,
        custom_sub_tools=tools,
    )

    try:
        prompt = (
            "Use memory_write('greeting', 'hello world') inside a repl block. "
            "Then memory_read('greeting') and print it. "
            "Then FINAL_VAR the result."
        )
        rlm.completion(prompt=prompt, root_prompt=prompt)
    finally:
        rlm.close()

    assert "greeting" in backend._store, (
        f"parent LLM did not invoke memory_write. "
        f"backend._store={backend._store}. "
        f"This means tool descriptions from make_memory_tools are not "
        f"reaching the parent's system prompt, or the parent chose to ignore them."
    )
    assert backend._store["greeting"] == "hello world"
