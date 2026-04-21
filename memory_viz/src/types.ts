export type GraphNode = {
  id: string;
  name: string;
  summary?: string;
  kind?: string;
};

export type GraphEdge = {
  source: string;
  target: string;
  fact: string;
  name?: string;
};

export type GraphData = {
  meta: {
    source?: string;
    task_id?: string;
    question?: string;
    gold_answer?: string;
    generated_at?: string;
  };
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type FlatData = {
  meta: {
    source?: string;
    task_id?: string;
    question?: string;
    gold_answer?: string;
    generated_at?: string;
  };
  store: Record<string, string>;
};
