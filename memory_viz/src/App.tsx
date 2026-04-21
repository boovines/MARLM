import { useEffect, useState } from "react";
import type { GraphData, FlatData } from "./types";
import GraphView from "./GraphView";
import FlatView from "./FlatView";

type Tab = "graph" | "flat";

export default function App() {
  const [tab, setTab] = useState<Tab>("graph");
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [flat, setFlat] = useState<FlatData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/graph.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then(setGraph)
      .catch((e) => setError(`graph.json: ${e}`));
    fetch("/data/flat.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then(setFlat)
      .catch((e) => setError((prev) => (prev ? `${prev}; ` : "") + `flat.json: ${e}`));
  }, []);

  const meta = (tab === "graph" ? graph?.meta : flat?.meta) ?? {};

  return (
    <div className="app">
      <header>
        <h1>MARLM memory viewer</h1>
        <span className="meta">
          {meta.task_id ? <>task {meta.task_id} · </> : null}
          {meta.gold_answer ? <>gold = <code>{meta.gold_answer}</code> · </> : null}
          {meta.source ?? "—"}
        </span>
      </header>

      <nav className="tabs">
        <button className={tab === "graph" ? "active" : ""} onClick={() => setTab("graph")}>
          Knowledge graph (Graphiti)
        </button>
        <button className={tab === "flat" ? "active" : ""} onClick={() => setTab("flat")}>
          Flat KV memory
        </button>
      </nav>

      <main>
        {error ? (
          <div className="empty">
            <p>Couldn't load data.</p>
            <p className="hint">{error}</p>
            <p className="hint">
              Run <code>python rlm/scripts/dump_memory_viz.py</code> to regenerate
              <code> memory_viz/public/data/*.json</code>.
            </p>
          </div>
        ) : tab === "graph" ? (
          <GraphView data={graph} />
        ) : (
          <FlatView data={flat} />
        )}
      </main>
    </div>
  );
}
