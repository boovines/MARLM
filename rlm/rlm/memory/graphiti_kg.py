from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EpisodeReranker,
    EpisodeSearchConfig,
    EpisodeSearchMethod,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
)

from rlm.memory.base import MemoryBackend, MemoryHit


# Hybrid search config: BM25 keyword + cosine embeddings over edges AND nodes,
# reranked by reciprocal-rank fusion (no LLM cross-encoder → cheap). Returning
# nodes is critical: Graphiti merges multi-episode facts about the same entity
# into the node's `summary`, which is what bridges multi-hop queries like
# "spouse of the Green performer" (Steve Hillage's node summary contains BOTH
# "performed the song Green" AND "married Miquette Giraudy").
_HYBRID_SEARCH_CONFIG = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.rrf,
    ),
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
        reranker=NodeReranker.rrf,
    ),
    episode_config=EpisodeSearchConfig(
        search_methods=[EpisodeSearchMethod.bm25],
        reranker=EpisodeReranker.rrf,
    ),
    limit=10,
)


class GraphitiBackend(MemoryBackend):
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._tempdir: str | None = None
        self._graphiti: Graphiti | None = None
        self._init_graph()

    def _init_graph(self) -> None:
        self._tempdir = tempfile.mkdtemp(prefix="marlm_graphiti_")
        db_path = os.path.join(self._tempdir, "kuzu.db")
        driver = KuzuDriver(db=db_path)
        self._graphiti = Graphiti(graph_driver=driver)
        self._loop.run_until_complete(self._graphiti.build_indices_and_constraints())
        # graphiti 0.28.2's KuzuDriver.build_indices_and_constraints is a no-op,
        # but Kuzu FTS indexes are required for graphiti.search(). Create them here.
        # (Extension load + CREATE_FTS_INDEX must run against the live AsyncConnection.)
        self._loop.run_until_complete(self._ensure_fts_indexes())

    async def _ensure_fts_indexes(self) -> None:
        assert self._graphiti is not None
        driver = self._graphiti.driver
        await driver.execute_query("INSTALL fts;")
        await driver.execute_query("LOAD EXTENSION fts;")
        for q in (
            "CALL CREATE_FTS_INDEX('Episodic', 'episode_content', "
            "['content', 'source', 'source_description']);",
            "CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary']);",
            "CALL CREATE_FTS_INDEX('Community', 'community_name', ['name']);",
            "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);",
        ):
            await driver.execute_query(q)

    def write(self, key: str, value: str) -> None:
        assert self._graphiti is not None
        self._loop.run_until_complete(
            self._graphiti.add_episode(
                name=key,
                episode_body=value,
                source_description="marlm",
                reference_time=datetime.now(tz=timezone.utc),
            )
        )

    def read(self, query: str, top_k: int = 5) -> list[MemoryHit]:
        assert self._graphiti is not None
        # Use the richer .search_() with hybrid BM25+cosine over edges AND nodes.
        # Node summaries carry merged multi-episode facts about each entity,
        # which is what enables multi-hop bridging through the node's summary.
        config = _HYBRID_SEARCH_CONFIG.model_copy(update={"limit": max(top_k * 2, 10)})
        results = self._loop.run_until_complete(
            self._graphiti.search_(query=query, config=config)
        )

        hits: list[MemoryHit] = []

        # Nodes first — their summaries bridge multi-hop queries.
        for node in results.nodes:
            summary = (node.summary or "").strip()
            if not summary:
                continue
            hits.append(
                MemoryHit(
                    key=f"node:{node.name}",
                    value=summary,
                    score=1.0,
                    metadata={"kind": "node", "uuid": node.uuid, "name": node.name},
                )
            )

        # Then edges — single-hop facts.
        for edge in results.edges:
            hits.append(
                MemoryHit(
                    key=f"edge:{edge.fact[:80]}",
                    value=edge.fact,
                    score=1.0,
                    metadata={
                        "kind": "edge",
                        "src": edge.source_node_uuid,
                        "tgt": edge.target_node_uuid,
                    },
                )
            )

        return hits[:top_k]

    def reset(self) -> None:
        if self._graphiti is not None:
            try:
                self._loop.run_until_complete(self._graphiti.close())
            except Exception:
                pass
        self._graphiti = None
        if self._tempdir is not None and os.path.isdir(self._tempdir):
            shutil.rmtree(self._tempdir, ignore_errors=True)
        self._tempdir = None
        self._init_graph()

    def __del__(self) -> None:
        try:
            if self._graphiti is not None and not self._loop.is_closed():
                try:
                    self._loop.run_until_complete(self._graphiti.close())
                except Exception:
                    pass
            if self._tempdir is not None and os.path.isdir(self._tempdir):
                shutil.rmtree(self._tempdir, ignore_errors=True)
            if not self._loop.is_closed():
                self._loop.close()
        except Exception:
            pass
