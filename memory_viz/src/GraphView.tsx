import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { GraphData, GraphNode } from "./types";

type Props = { data: GraphData | null };

export default function GraphView({ data }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [selected, setSelected] = useState<GraphNode | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => setSize({ w: el.clientWidth, h: el.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const graphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };
    const degree = new Map<string, number>();
    for (const e of data.edges) {
      degree.set(e.source, (degree.get(e.source) ?? 0) + 1);
      degree.set(e.target, (degree.get(e.target) ?? 0) + 1);
    }
    return {
      nodes: data.nodes.map((n) => ({ ...n, degree: degree.get(n.id) ?? 0 })),
      links: data.edges.map((e) => ({ ...e })),
    };
  }, [data]);

  const isLarge = (data?.nodes.length ?? 0) > 30;

  if (!data) {
    return <div className="empty">Loading graph…</div>;
  }

  if (data.nodes.length === 0) {
    return (
      <div className="empty">
        <p>Graph is empty.</p>
        <p className="hint">No episodes were written to Graphiti for this run.</p>
      </div>
    );
  }

  return (
    <div className="graph-wrap" ref={containerRef}>
      <div className="graph-legend">
        <div>
          <strong>{data.nodes.length}</strong> nodes · <strong>{data.edges.length}</strong> edges
        </div>
        <div>click a node to inspect its summary</div>
        {isLarge && (
          <div>
            node size ∝ degree · zoom in for labels · hover edge for fact
          </div>
        )}
      </div>

      <ForceGraph2D
        graphData={graphData as any}
        width={size.w}
        height={size.h}
        backgroundColor="#0f1115"
        nodeLabel={(n: any) => n.name}
        nodeAutoColorBy="kind"
        nodeCanvasObject={(node: any, ctx, globalScale) => {
          const label = node.name as string;
          const deg = node.degree ?? 0;
          // Radius grows with degree so hubs visually pop.
          const radius = Math.max(4, Math.min(18, 4 + Math.sqrt(deg) * 2.2));
          ctx.fillStyle = node.color || "#5b8cff";
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
          ctx.fill();

          // Hide labels when zoomed out on large graphs; always show top hubs.
          const showLabel = !isLarge || globalScale > 1.8 || deg >= 8;
          if (!showLabel) return;
          const fontSize = Math.max(10 / globalScale, 3);
          ctx.font = `${fontSize}px ui-sans-serif, system-ui, sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillStyle = "#e6e9ef";
          ctx.fillText(label, node.x, node.y + radius + 2);
        }}
        linkLabel={(l: any) => l.fact}
        linkColor={() => "#3a4466"}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={1}
        linkCanvasObjectMode={() => "after"}
        linkCanvasObject={(link: any, ctx, globalScale) => {
          // Suppress edge labels when zoomed out or on large graphs —
          // hundreds of overlapping labels is worse than none.
          if (isLarge && globalScale < 2.2) return;
          const label = link.name || "";
          if (!label) return;
          const fontSize = 10 / globalScale;
          const midX = (link.source.x + link.target.x) / 2;
          const midY = (link.source.y + link.target.y) / 2;
          ctx.font = `${fontSize}px ui-sans-serif, system-ui, sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillStyle = "#9aa3b2";
          ctx.fillText(label, midX, midY);
        }}
        onNodeClick={(n: any) => setSelected(n)}
      />

      {selected && (
        <div className="node-details">
          <h4>{selected.name}</h4>
          {selected.kind && (
            <p>
              <span className="label">kind </span>
              {selected.kind}
            </p>
          )}
          {selected.summary && (
            <>
              <p className="label">summary</p>
              <p>{selected.summary}</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
