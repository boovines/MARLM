# Memory-Augmented Recursive Language Models
## CLMM Interim Report

**Maksym Bondarenko, Tevin Kim, Justin Hou, Phillip Yan**

**March 27, 2026**

---

## 1. Introduction

Recursive Language Models (RLMs) [1] are a recent inference-time paradigm for scaling LLM context processing. Rather than feeding an entire long prompt into the model's context window, RLMs externalize the prompt as a variable in a Python REPL environment and let the LLM programmatically examine, decompose, and recursively call itself over slices of the input. The authors report performance on inputs up to 100x beyond model context windows across several long-context benchmarks.

However, RLMs are fundamentally stateless across recursive calls: each sub-call (`llm_query`) receives only the prompt string passed by the parent model's code, with no memory of what previous calls discovered. Our project investigates whether augmenting RLMs with an explicit persistent memory mechanism — particularly learned knowledge graphs following the Graphiti approach [2] — can improve performance on tasks requiring reasoning across distributed context.

In this interim report, we present: (1) a replication study of vanilla RLM on two benchmarks, (2) preliminary results on a self-consistency (majority-vote) variant, (3) a critical analysis of the RLM paper's benchmark choices, and (4) our plan for the remaining project phases.

## 2. Literature Review

**Long-context LLM approaches.** Extending LLM effective context has been approached through several directions. Retrieval-Augmented Generation (RAG) [3] externalizes memory into a vector index, retrieving relevant passages at inference time. While practical, RAG is limited by retrieval granularity and cannot represent relational structure. Long-context attention models [4] push the context window directly but face quadratic attention costs and empirical degradation at extreme lengths. State space models such as Mamba [5] offer linear-time alternatives but compress context into a fixed-size latent state, losing fine-grained information.

**Recursive Language Models.** RLMs [1] represent a fundamentally different paradigm: the model decomposes the task recursively, calling itself on sub-problems and aggregating results in code. This yields effective context well beyond direct attention limits. The key architecture consists of a "parent" LLM that generates Python code in a REPL environment, with access to `llm_query()` for sub-LM calls and `llm_query_batched()` for concurrent batch calls. The paper reports strong results using GPT-5 as the parent model and GPT-5-mini for sub-calls, with max recursion depth of 1.

**Self-consistency.** Wang et al. [6] showed that sampling multiple reasoning paths and taking the majority vote significantly improves LLM accuracy on reasoning tasks. We investigate applying this principle to RLM sub-calls, where each `llm_query` classification is run N times with majority voting to reduce individual classification errors.

**Memory-augmented LLMs.** Mem0 [7] builds production-ready memory layers for AI agents. Graphiti [2] maintains a temporally-aware relational knowledge graph that persists across interactions. MemWalker [8] constructs tree-like data structures for navigating long contexts. Our work is most closely related to Graphiti, as we propose injecting a persistent graph memory accessible to all RLM sub-calls.

## 3. Experimental Setup

### 3.1 Models and Configuration

All experiments use the same configuration as the RLM paper:
- **Parent model**: GPT-5 (root LLM that generates REPL code)
- **Sub-call model**: GPT-5-mini (handles `llm_query` classification calls)
- **Max depth**: 1 (parent with REPL + sub-calls, no recursive children)
- **Max iterations**: 15

### 3.2 Benchmarks

We evaluate on two benchmarks chosen to test different aspects of RLM:

**OOLONG-synth** [9]: A long-context information aggregation benchmark. Each task presents a context window of news articles (without labels) and asks aggregate questions such as "how many data points should be classified as Sci/Tech?" The model must independently classify each article and count. Scoring uses `0.75^|error|` for numeric answers (partial credit) and exact match for label answers. We use the agnews split at 16K tokens from HuggingFace.

**MuSiQue** [10]: A multi-hop question answering benchmark requiring chaining facts across multiple paragraphs. Each question requires 2-4 reasoning hops (e.g., "Who is the spouse of the Green performer?" requires identifying the performer of "Green," then finding their spouse). Scoring uses Exact Match (EM) and token-level F1. We test on 5 two-hop tasks with 20 distractor paragraphs each.

### 3.3 Majority-Vote Variant

We implement a self-consistency variant by subclassing the RLM's `LocalREPL` environment. Our `MajorityVoteREPL` transparently overrides `_llm_query` and `_llm_query_batched` so that each sub-call is executed N=3 times and the most common response is returned. The parent model's behavior is completely unchanged — it generates the same code, uses the same system prompt, and calls the same functions. The voting happens at the engine level.

## 4. Results

### 4.1 OOLONG-synth (agnews, 16K tokens, numeric tasks)

| Method | Avg Score | Avg Time |
|--------|-----------|----------|
| **Vanilla GPT-5** | 0.711 | 73.9s |
| **RLM** (GPT-5 + GPT-5-mini) | **0.756** | 315.2s |
| **RLM + Majority Vote x3** | 0.500 | 946.9s |

**Observations:**
- At 16K tokens (~183 articles), vanilla GPT-5 and RLM perform comparably. With only n=2 tasks, the score differences (0.71 vs 0.756) are within stochastic variance — RLM trajectories are highly non-deterministic (the parent may chunk differently, use different classification prompts, or make different aggregation decisions across runs). We do not draw conclusions about relative performance at this context length.
- The RLM paper reports that RLM's advantage over vanilla emerges primarily at 32K+ tokens, where vanilla GPT-5 begins to degrade. At 16K, the context fits comfortably in GPT-5's window, so both methods are expected to perform similarly.
- The majority-vote variant performed *worse* on Task 1, predicting 42 instead of 28. This suggests the classification errors are systematic (the model consistently misclassifies certain articles as Business), so voting amplifies rather than corrects the bias.
- All methods solved Task 2 (Sports, gold=39) perfectly, suggesting Sports articles are easier to classify than Business articles.
- **Cost and time are significant barriers to larger-scale evaluation.** RLM is 4-5x slower and ~3x more expensive than vanilla per task. Majority-vote RLM is 13x slower and ~9x more expensive. At the paper's benchmark scale (50 tasks at 131K tokens), a full RLM evaluation would cost ~$22 and take several hours. Majority-vote at the same scale would cost ~$60+. These costs limited our ability to run larger sample sizes needed to overcome stochastic variance, and precluded evaluation at the 131K context lengths where the paper reports its strongest results.

### 4.2 MuSiQue (2-hop, 20 paragraphs)

| Method | EM | Avg F1 | Avg Time |
|--------|-----|--------|----------|
| **Vanilla GPT-5** | **0.800** (4/5) | **0.933** | 13.85s |
| **RLM** (GPT-5 + GPT-5-mini) | 0.600 (3/5) | 0.600 | 195.87s |

Per-task breakdown:

| Task | Question | Vanilla EM | RLM EM |
|------|----------|-----------|--------|
| 1 | Who is the spouse of the Green performer? | 1 | 0 |
| 2 | Who founded the company that distributed UHF? | 1 | 1 |
| 3 | What admin. entity is the owner of Ciudad Deportiva located? | 1 | 1 |
| 4 | Where is Ulrich Walter's employer headquartered? | 0 (F1=0.67) | 0 (F1=0.00) |
| 5 | Which company owns the manufacturer of Learjet 60? | 1 | 1 |

**Observations:**
- Vanilla GPT-5 outperforms RLM on multi-hop QA. With only 20 paragraphs (~2K tokens of context), the full text fits easily within GPT-5's context window, making RLM's decomposition overhead pure cost with no benefit.
- RLM failed Task 1 completely — it could not chain the two hops (identifying the performer of "Green" and then finding their spouse) through its sub-call architecture. The stateless sub-calls each received isolated paragraphs and could not reason across them.
- RLM was 14x slower on average (196s vs 14s).
- Task 4 was partially wrong for both methods, but vanilla got closer (F1=0.67 for "Cologne, Germany" vs gold "Cologne") while RLM hallucinated "Chicago, Illinois" (F1=0.00).

### 4.3 Cost Analysis

| Experiment | Vanilla Cost (est.) | RLM Cost (est.) | RLM Majority Cost (est.) |
|---|---|---|---|
| OOLONG 16K (2 tasks) | ~$0.20 | ~$0.60 | ~$1.80 |
| MuSiQue (5 tasks) | ~$0.05 | ~$1.50 | N/A |

RLM is 3-30x more expensive than vanilla depending on task complexity. Majority voting triples the sub-call cost. For OOLONG at 16K, the cost is manageable; at 131K (paper's setting) the cost differential would be more significant.

## 5. Analysis and Key Findings

### 5.1 RLM's Strengths are Task-Dependent

Our experiments across two benchmarks suggest that RLM's effectiveness depends heavily on task structure:

1. **OOLONG suits RLM's decompose-and-aggregate pattern.** The task structure (classify each item independently, count in code) maps directly to what RLMs are designed for. At 16K tokens, vanilla GPT-5 and RLM perform comparably (both within the model's context window). The RLM paper reports clear advantages at 32K+ tokens where vanilla degrades — a regime we could not fully test due to cost constraints (estimated ~$22 for 50 tasks at 131K tokens for RLM alone). Our limited 16K results are consistent with the paper's finding that RLM's advantage grows with context length.

2. **Multi-hop QA exposes RLM's statelessness.** On MuSiQue, where answering requires chaining facts across paragraphs (not independently classifying items), RLM performs worse than vanilla. The stateless sub-calls cannot share discoveries, making multi-hop reasoning difficult. This is a structural limitation, not a context-length issue.

3. **RLM trajectories are highly stochastic.** Across our experiments, we observed significant run-to-run variance in RLM's behavior: the parent model may chunk the context differently, write different classification prompts, or use different aggregation strategies. With our limited sample sizes (n=2 for OOLONG, n=5 for MuSiQue), individual task results should be interpreted as illustrative rather than statistically conclusive. Larger-scale evaluation is needed but was constrained by API costs ($0.30-$1.50 per RLM task) and runtime (3-10 minutes per task).

### 5.2 Self-Consistency Does Not Trivially Help

Our majority-vote experiment showed that running each sub-call 3x and voting does not improve accuracy when classification errors are systematic rather than random. On OOLONG Task 1, the model consistently misclassified certain Business articles (e.g., tech companies' business news classified as Sci/Tech), so voting amplified the error. This suggests that improving RLM sub-call accuracy requires better prompting or stronger child models, not simple repetition.

### 5.3 The Statelessness Problem

The MuSiQue results directly motivate our proposed memory augmentation. In Task 1 ("Who is the spouse of the Green performer?"), the RLM needed to: (1) find that Steve Hillage is the performer of "Green," and (2) find Steve Hillage's spouse. With stateless sub-calls, the result of hop 1 cannot inform hop 2 unless the parent explicitly orchestrates this — which it failed to do. A persistent memory layer where sub-calls can write and read shared state would directly address this.

## 6. Remaining Work

### Phase 2: Memory-Augmented RLM (Weeks 5-9)

1. **Implement memory layer.** Inject `memory_write(key, value)` and `memory_read(query)` as custom REPL tools, backed by three memory substrates:
   - **Flat key-value store** (baseline)
   - **Vector store / RAG** (semantic similarity retrieval)
   - **Knowledge graph** via Graphiti [2] (relational, temporally-aware)

2. **Benchmark memory-augmented RLM.**
   - OOLONG at 131K tokens (where vanilla GPT-5 degrades)
   - MuSiQue (where statelessness clearly hurts)
   - InfiniteBench En.MC/QA (novel-length holistic comprehension)

3. **Ablations.** Compare memory types, vary number of sub-calls, test with/without memory on each benchmark to isolate the contribution of persistent state.

### Phase 3: Analysis and Write-up (Weeks 10-12)

4. **Characterize when memory helps vs hurts.** We hypothesize memory helps on tasks requiring cross-chunk reasoning (MuSiQue, entity tracking) but adds overhead on independently-decomposable tasks (OOLONG counting).

5. **Final report** with quantitative comparison across all configurations.

## References

[1] Zhang, A.L., Kraska, T., Khattab, O. "Recursive Language Models." arXiv:2512.24601, 2026.

[2] Ranade, P. et al. "Graphiti: Building Real-Time Knowledge Graphs with LLM-Derived Triples." Zep AI, 2025.

[3] Lewis, P. et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS, 2020.

[4] Press, O. et al. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization." ICLR, 2022.

[5] Gu, A. and Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752, 2023.

[6] Wang, X. et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR, 2023.

[7] Chhikara, P. et al. "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory." arXiv:2504.19413, 2025.

[8] Chen, H. et al. "Walking Down the Memory Maze: Beyond Context Limit through Interactive Reading." arXiv:2310.05029, 2023.

[9] Bertsch, A. et al. "Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities." arXiv:2511.02817, 2025.

[10] Trivedi, H. et al. "MuSiQue: Multi-hop Questions via Single-hop Question Composition." TACL, 2022.
