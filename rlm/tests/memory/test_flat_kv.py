import pytest

from rlm.memory import FlatKVBackend, MemoryHit


@pytest.fixture
def backend() -> FlatKVBackend:
    return FlatKVBackend()


def test_exact_hit(backend: FlatKVBackend) -> None:
    backend.write("foo", "bar")
    hits = backend.read("foo")
    assert hits == [MemoryHit(key="foo", value="bar", score=1.0)]


def test_miss_returns_empty(backend: FlatKVBackend) -> None:
    backend.write("foo", "bar")
    assert backend.read("nothing") == []


def test_empty_backend_miss(backend: FlatKVBackend) -> None:
    assert backend.read("anything") == []


def test_substring_key_hit(backend: FlatKVBackend) -> None:
    backend.write("user_profile", "alice")
    backend.write("unrelated", "bob")
    hits = backend.read("PROFILE")
    assert len(hits) == 1
    assert hits[0].key == "user_profile"
    assert hits[0].value == "alice"
    assert hits[0].score == 0.5


def test_substring_value_hit(backend: FlatKVBackend) -> None:
    backend.write("k1", "The quick brown fox")
    backend.write("k2", "something else")
    hits = backend.read("QUICK")
    assert len(hits) == 1
    assert hits[0].key == "k1"
    assert hits[0].score == 0.5


def test_overwrite(backend: FlatKVBackend) -> None:
    backend.write("key", "v1")
    backend.write("key", "v2")
    hits = backend.read("key")
    assert hits == [MemoryHit(key="key", value="v2", score=1.0)]


def test_top_k_respected(backend: FlatKVBackend) -> None:
    for i in range(10):
        backend.write(f"match_{i}", f"value_{i}")
    hits = backend.read("match", top_k=3)
    assert len(hits) == 3
    assert all(h.score == 0.5 for h in hits)


def test_top_k_default_is_five(backend: FlatKVBackend) -> None:
    for i in range(10):
        backend.write(f"match_{i}", f"value_{i}")
    assert len(backend.read("match")) == 5


def test_exact_hit_ignores_top_k(backend: FlatKVBackend) -> None:
    backend.write("exact", "v")
    backend.write("exact_suffix", "other")
    hits = backend.read("exact", top_k=10)
    assert hits == [MemoryHit(key="exact", value="v", score=1.0)]


def test_reset_clears(backend: FlatKVBackend) -> None:
    backend.write("a", "1")
    backend.write("b", "2")
    backend.reset()
    assert backend.read("a") == []
    assert backend.read("b") == []
