from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver

from rlm.memory.base import MemoryBackend, MemoryHit


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
        edges = self._loop.run_until_complete(
            self._graphiti.search(query, num_results=top_k)
        )
        return [
            MemoryHit(
                key=edge.fact[:80],
                value=edge.fact,
                score=1.0,
                metadata={
                    "src": edge.source_node_uuid,
                    "tgt": edge.target_node_uuid,
                },
            )
            for edge in edges
        ]

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
