import type { FlatData } from "./types";

type Props = { data: FlatData | null };

export default function FlatView({ data }: Props) {
  if (!data) return <div className="empty">Loading flat memory…</div>;

  const entries = Object.entries(data.store ?? {});

  return (
    <div className="panel">
      <div className="summary">
        <span>
          <strong>{entries.length}</strong> key{entries.length === 1 ? "" : "s"} in store
        </span>
        {data.meta.question ? (
          <span>
            task: <em>{data.meta.question}</em>
          </span>
        ) : null}
      </div>

      {entries.length === 0 ? (
        <div className="kv-empty">
          <p>Flat KV store is empty.</p>
          <p className="hint">No memory_write calls were issued for this run.</p>
        </div>
      ) : (
        <table className="kv">
          <thead>
            <tr>
              <th>key</th>
              <th>value</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([k, v]) => (
              <tr key={k}>
                <td className="key">{k}</td>
                <td className="value">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
