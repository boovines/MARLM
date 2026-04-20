def build_memory(name: str):
    if name == "none":
        return None
    if name == "flat":
        from rlm.memory.flat_kv import FlatKVBackend
        return FlatKVBackend()
    if name == "graph":
        from rlm.memory.graphiti_kg import GraphitiBackend
        return GraphitiBackend()
    raise ValueError(f"unknown memory backend: {name}")
