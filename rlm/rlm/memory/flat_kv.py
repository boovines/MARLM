from .base import MemoryBackend, MemoryHit


class FlatKVBackend(MemoryBackend):
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def write(self, key: str, value: str) -> None:
        self._store[key] = value

    def read(self, query: str, top_k: int = 5) -> list[MemoryHit]:
        if query in self._store:
            return [MemoryHit(key=query, value=self._store[query], score=1.0)]

        needle = query.lower()
        hits: list[MemoryHit] = []
        for key, value in self._store.items():
            if needle in key.lower() or needle in value.lower():
                hits.append(MemoryHit(key=key, value=value, score=0.5))
                if len(hits) >= top_k:
                    break
        return hits

    def reset(self) -> None:
        self._store.clear()
