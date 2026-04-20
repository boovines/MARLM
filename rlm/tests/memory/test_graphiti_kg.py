import os

import pytest

from rlm.memory import GraphitiBackend


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for Graphiti entity/edge extraction",
)
def test_musique_spouse_of_green_performer() -> None:
    """Thesis claim: KG memory enables multi-hop reasoning RLMs fail at.

    Hop 1: Steve Hillage performed the song Green.
    Hop 2: Steve Hillage married Miquette Giraudy.
    Query: "spouse of the Green performer" -> must surface Miquette Giraudy.
    """
    backend = GraphitiBackend()
    try:
        backend.write("fact_1", "Steve Hillage performed the song Green")
        backend.write("fact_2", "Steve Hillage married Miquette Giraudy")

        hits = backend.read("spouse of the Green performer")

        assert any("Miquette Giraudy" in h.value for h in hits), (
            "THESIS FAILURE: multi-hop retrieval did not surface 'Miquette Giraudy'. "
            f"Hits: {[h.value for h in hits]}"
        )
    finally:
        backend.reset()
