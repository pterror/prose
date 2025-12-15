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
