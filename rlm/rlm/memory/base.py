from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryHit:
    key: str
    value: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryBackend(ABC):
    @abstractmethod
    def write(self, key: str, value: str) -> None: ...

    @abstractmethod
    def read(self, query: str, top_k: int = 5) -> list[MemoryHit]: ...

    @abstractmethod
    def reset(self) -> None: ...
