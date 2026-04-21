"""Shared majority-vote REPL wrapper for benchmarks.

Extracted from benchmark_oolong_majority.py so both OOLONG and MuSiQue
benchmarks can opt-in via `--majority N`. Each llm_query/llm_query_batched
sub-call gets run N times; the most common response is returned. The
parent model's code is unchanged.
"""
from __future__ import annotations

from collections import Counter

from rlm.environments.local_repl import LocalREPL, send_lm_request_batched


class MajorityVoteREPL(LocalREPL):
    def __init__(self, *args, n_votes: int = 3, **kwargs):
        self.n_votes = n_votes
        super().__init__(*args, **kwargs)

    def _llm_query(self, prompt, model=None):
        responses = self._raw_batched([prompt] * self.n_votes, model)
        return Counter(responses).most_common(1)[0][0]

    def _llm_query_batched(self, prompts, model=None):
        expanded: list = []
        for p in prompts:
            expanded.extend([p] * self.n_votes)
        all_responses = self._raw_batched(expanded, model)
        results = []
        for i in range(len(prompts)):
            votes = all_responses[i * self.n_votes : (i + 1) * self.n_votes]
            results.append(Counter(votes).most_common(1)[0][0])
        return results

    def _raw_batched(self, prompts, model=None):
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


def install_majority_vote(n_votes: int) -> None:
    """Monkey-patch the rlm.environments factory so 'local' env returns a
    MajorityVoteREPL with the given n_votes. Call ONCE before constructing RLM."""
    import rlm.environments as env_module
    original = env_module.get_environment

    def _patched(env_type, env_kwargs):
        if env_type == "local":
            return MajorityVoteREPL(n_votes=n_votes, **env_kwargs)
        return original(env_type, env_kwargs)

    env_module.get_environment = _patched
