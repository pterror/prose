# Architecture Documentation

## Core Design Decisions

### 1. Abstract Syntax Graphs (ASG) vs. AST

**Decision**: Use graphs with three edge types instead of traditional tree-only ASTs.

**Rationale**:

- **Long-context problem**: In standard ASTs, a variable reference might be 50+ nodes away from its definition. The model has to learn to traverse this distance through attention/convolution.
- **Direct connections**: DataFlow edges create explicit links between `def` and `use` sites, making this relationship a single-hop traversal.
- **Execution semantics**: Sibling edges encode sequential evaluation order, which is implicit in tree structure but explicit in graph form.

**Edge Types**:

- **Child**: Standard parent-child syntactic hierarchy (e.g., function → arguments)
- **Sibling**: Sequential evaluation order (left-to-right in S-expressions)
- **DataFlow**: Variable definition → usage (computed via symbol table analysis)

### 2. Property-Based Synthetic Generation

**Decision**: Generate programs from semantic templates with stochastic variations, not pure grammar sampling.

**Rationale**:

- **Usefulness**: Random CFG sampling produces syntactically valid but semantically meaningless programs (e.g., `(+ (+ (+ 1 1) 1) 1)` repeated infinitely).
- **Diversity constraints**: Track operator distributions, AST shapes to prevent mode collapse.
- **Semantic properties**: Each program must compute something meaningful (recursion has base cases, functions have parameters, etc.).

**Template Categories**:

1. **Arithmetic**: Operator composition (`(+ (* a b) (- c d))`)
2. **Recursion**: Factorial, Fibonacci patterns with explicit base cases
3. **Higher-order**: Map/filter/reduce with lambdas
4. **Scoping**: Let-bindings to test variable shadowing and DataFlow edges

**Rejection Sampling**: Programs failing property checks (depth < 2, no operators, no variables) are discarded.

### 3. Graph U-Net Architecture

**Decision**: Use hierarchical encoder-decoder with TopK pooling instead of flat graph convolution.

**Rationale**:

- **Hierarchical abstraction**: Code has natural coarse-to-fine structure (function → statement → expression → token). Pooling captures this.
- **Efficient computation**: TopK pooling reduces graph size by 50% per level, making deeper networks tractable.
- **Skip connections**: Preserve fine-grained details during unpooling, similar to image segmentation U-Nets.

**Layer Choice (GAT vs. GCN)**:

- **GAT** (Graph Attention): Learns edge importance dynamically. Useful since Child/DataFlow edges have different semantic weights.
- **GCN**: Simpler, faster. Treats all edges equally.
- **Current choice**: GAT for Phase 1 (can A/B test later).

**Simplified Unpooling** (Phase 1):

- Full unpooling requires restoring exact graph topology from pooled indices (complex).
- Current implementation uses a single encoding/decoding layer at full resolution as a prototype.
- Phase 2 will implement proper unpooling with skip connections.

### 6. Attention Mechanisms: Implicit vs Explicit

**Question**: Do we need explicit Transformer-style attention on top of the Graph U-Net?

**Answer**: No. The ASG structure already provides "implicit attention" via message passing along graph edges, which is more appropriate for code than free-form all-to-all attention.

#### Why Transformer Attention Exists

```python
# Transformers: Every token attends to every other token
attention_weights = softmax(Q @ K^T / sqrt(d_k))  # [N, N] matrix
output = attention_weights @ V

# Pros: Learns dynamic, input-dependent relationships
# Cons: O(N²) memory and compute, no structural bias
```

**Transformers use attention because:**

- Natural language has fluid, context-dependent relationships
- Linear sequences have no inherent structure
- Need to learn "soft" alignments dynamically (e.g., pronoun resolution)

#### How ASG Provides Implicit Attention

Our graph edges **encode the relationships** that attention would need to learn:

**1. Syntactic Structure (CHILD Edges)**

```lisp
(define (foo x)
  (+ x 1))
```

CHILD edges: `define` → `foo`, `define` → `body`, `+` → `x`, `+` → `1`

- **Transformer approach**: Must learn that operators attend to their arguments
- **ASG approach**: Parent-child relationships are explicitly encoded

**2. Sequential Context (SIBLING Edges)**

```lisp
(let ((a 1) (b 2) (c 3)) ...)
```

SIBLING edges: `a → b → c` (evaluation order)

- **Transformer approach**: Uses positional encodings to track sequence
- **ASG approach**: Makes sequence explicit via directed edges

**3. Long-Range Dependencies (DATAFLOW Edges)**

**This is the key insight.** The original motivation for ASG over AST!

```lisp
(define (factorial n)
  (if (< n 1)
      1
      (* n (factorial (- n 1)))))
```

DATAFLOW edges directly connect:

- Variable definition (`n` parameter) → all usages of `n` (4 edges)
- Function name (`factorial`) → recursive call

**Implementation** (from `src/data/asg_builder.py`):

```python
def _add_dataflow_edges(self, root_id: int) -> None:
    """Add DataFlow edges from variable definitions to uses."""
    for node_id in self.graph.nodes():
        if node_data["node_type"] == NodeType.SYMBOL.value:
            symbol_name = node_data.get("value")

            if symbol_name in self.symbol_table:
                # Direct edge: definition → usage (O(1) lookup!)
                for def_id in self.symbol_table[symbol_name]:
                    self.graph.add_edge(
                        def_id, node_id,
                        edge_type=EdgeType.DATAFLOW.value
                    )
```

**Comparison:**

| Problem                   | Transformer Solution                | ASG Solution                |
| ------------------------- | ----------------------------------- | --------------------------- |
| Find variable definition  | Attend over entire sequence (O(N²)) | Follow DATAFLOW edge (O(1)) |
| Operator-argument binding | Learn attention patterns            | Encoded in CHILD edges      |
| Evaluation order          | Positional encoding                 | SIBLING edges               |

**Result**: ASG solves the "long-range dependency problem" that attention was designed for, but in a **structured, O(E) way** instead of O(N²).

#### When Explicit Attention Could Help

There are scenarios where adding attention _on top of_ ASG structure might be beneficial:

**1. Dynamic Edge Importance (Graph Attention Networks)**

```python
# GAT: Learn which edges matter more
alpha_ij = attention_score(node_i, node_j)  # Learnable weight
message = Σ alpha_ij * node_j.features
```

**Use case**: Maybe DATAFLOW edges should have higher weight than SIBLING edges for certain tasks.

**2. Cross-Subgraph Relationships**

If we need to reason about patterns _not_ captured by explicit edges:

- "These two variables have similar names" (semantic similarity)
- "These functions have similar behavior" (clone detection)

**3. Global Context Aggregation**

When every node needs awareness of the entire program:

- "Is this a recursive function?" (needs global call graph)
- "What's the maximum nesting depth?" (global property)

**Current solution**: U-Net pooling aggregates global context at the bottleneck layer.

**Alternative**: Add "virtual global node" (like `[CLS]` token in BERT) that connects to all nodes.

#### Multi-Hop Message Passing as Implicit Attention

**Key insight**: Multiple GNN layers already provide multi-hop reasoning:

```python
# Layer 1: Each node sees 1-hop neighbors
# Layer 2: Each node sees 2-hop neighbors
# Layer k: Each node sees k-hop neighbors
```

**Example**: Variable usage sees definition in 1 hop via DATAFLOW edge:

```
Layer 0: [symbol 'n']
Layer 1: [symbol 'n'] + message from [parameter 'n'] (via DATAFLOW)
Layer 2: [symbol 'n'] + context from entire function signature
```

This is equivalent to "attending" to all k-hop neighbors, but with structured inductive bias.

#### Connection to Scratchpad Nodes (Phase 1.5)

The **scratchpad architecture** proposed in [`phase1.5.md`](phase1.5.md) is actually a form of structured attention:

```python
# Scratchpad = learnable "attention anchors"
scratch_nodes = [s1, s2, ..., s10]  # Virtual reasoning workspace

# Connect scratch to all program nodes
for si in scratch_nodes:
    for pj in program_nodes:
        add_edge(si, pj)  # Bidirectional
```

**What scratch nodes do:**

- Aggregate global information (like attention pooling)
- Broadcast to all nodes (like cross-attention)
- Maintain state across iterations (like recurrent memory)

**Similar to:**

- `[CLS]` token in BERT (global aggregation)
- Memory networks (explicit reasoning workspace)

**Advantage over full Transformer attention:**

- Fixed number of scratch nodes (e.g., 10) → O(N × 10) instead of O(N²)
- Can specialize (e.g., "one scratch node tracks test constraints")

#### Design Decision Summary

| Mechanism                      | What It Does                       | Complexity | When to Use                            |
| ------------------------------ | ---------------------------------- | ---------- | -------------------------------------- |
| **ASG Edges** (current)        | Encodes syntax, sequence, dataflow | O(E)       | Default (proven effective in Phase 1)  |
| **GNN Message Passing**        | Multi-hop propagation              | O(E × L)   | Always (core architecture)             |
| **Graph Attention (GAT)**      | Learns edge importance             | O(E)       | If ablation shows edges need weighting |
| **Scratchpad Nodes**           | Global reasoning workspace         | O(N × K)   | For iterative refinement (Phase 1.5)   |
| **Full Transformer Attention** | All-to-all pairwise                | O(N²)      | **Avoid** (loses structural bias)      |

**Current decision**: Use ASG structure + GAT message passing. Explicit attention is **implicit** via edges, which is ideal for code's inherent structure.

**When to revisit**:

- If Phase 1.5 error analysis shows model isn't using DATAFLOW edges effectively
- If specific failure modes emerge (e.g., deep recursion, complex scoping)
- If we need dynamic, input-dependent edge weights for refinement

### 4. Denoising Auto-Encoder Task

**Decision**: Mask 20% of nodes, predict original node types (not pure diffusion yet).

**Rationale**:

- **Simpler baseline**: Validates that the model can learn graph structure before adding diffusion complexity.
- **BERT-style masking**: Proven effective for learning representations in NLP.
- **Phase 2 upgrade path**: This task is a special case of discrete diffusion (1-step denoising).

**Why not autoregressive?**

- Code graphs are not sequential; no natural "left-to-right" ordering.
- Diffusion allows parallel generation of multiple nodes.

### 5. Mini-Lisp as Target Language

**Decision**: Use S-expressions instead of Rust/Python for Phase 1.

**Rationale**:

- **Minimal syntax**: Entire grammar fits in ~10 node types, reducing vocabulary size.
- **Rich structure**: Recursion, higher-order functions, scoping all present.
- **Fast parsing**: No need for tree-sitter; can generate ASGs directly.
- **Clear evaluation**: Can run programs through a simple Lisp interpreter to verify correctness.

**Lingua franca approach**: The core techniques (graph convolution, DataFlow edges, diffusion) are language-agnostic. Starting simple lets us validate the approach before scaling to real-world languages.

## Key Implementation Details

### Node Features

- **Current**: Integer node type ID (0-8) embedded into 128-dim continuous vector.
- **Future**: Add position encodings (depth, sibling index), literal values (for numbers/strings).

### Edge Features

- **Current**: Integer edge type ID (0-2: Child/Sibling/DataFlow).
- **Future**: Edge embeddings could encode direction, distance, or AST path information.

### Memory Optimization for RTX 3060

- **Gradient accumulation**: Effective batch size 128 (32 × 4 accumulation steps).
- **Mixed precision (AMP)**: FP16 reduces memory usage by ~40%.
- **PyG batching**: Disjoint graph representation (single large graph) is memory-efficient for variable-size inputs.

### Checkpointing Strategy

- Save every 5 epochs (avoid disk I/O overhead).
- Store both optimizer and scheduler state for resumability.
- Keep separate "best_model.pt" based on validation loss.

## Open Questions / Future Work

1. **Unpooling**: How to restore exact graph topology? Options:

   - Store pooling indices and reverse the operation.
   - Learn a "graph expansion" network (GAN-style).
   - Use message passing to "fill in" missing nodes.

2. **Discrete Diffusion**: How to represent graph edits as diffusion steps?

   - Node insertion/deletion changes graph size.
   - Edge insertion/deletion changes connectivity.
   - Tree constraints must be maintained (no cycles for ASTs).

3. **Symbolic Constraints**: How to enforce syntax during generation?

   - **Logit masking**: Dynamically mask invalid next tokens based on grammar.
   - **SMT solver**: Post-process outputs to verify logical constraints.
   - **Constrained beam search**: Reject invalid partial graphs during generation.

4. **Evaluation Metrics**: What defines "correctness" for code synthesis?

   - **Exact match**: Too strict (many equivalent programs).
   - **Execution equivalence**: Run on test inputs, compare outputs.
   - **Edit distance**: Measure similarity to ground truth.
   - **Human eval**: Subjective but realistic.

5. **Scaling to Real Languages**: What changes for Python/Rust?
   - Larger node vocabulary (50-200 token types).
   - More complex DataFlow (aliasing, mutation, classes).
   - Type system integration (use type checker as constraint oracle).
   - Control flow graphs (if/while/exception handling).

## References

- **PyTorch Geometric**: Fey & Lenssen (2019), "Fast Graph Representation Learning with PyTorch Geometric"
- **Graph U-Nets**: Gao & Ji (2019), "Graph U-Nets"
- **Tree Diffusion**: Kapur et al. (2025), "Diffusion On Syntax Trees For Program Synthesis"
- **Property-based Testing**: Hughes (2007), "QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs"
