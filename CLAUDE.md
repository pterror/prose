# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**prose** is a neuro-symbolic program synthesis experiment using Graph U-Nets on Abstract Syntax Graphs (ASG). The project implements a specialized neural model (~5M parameters) for code synthesis and refactoring, designed to run on local hardware (NVIDIA RTX 3060).

**Key Insight:** Code is treated as an Abstract Syntax Graph (ASG), not text. ASGs extend ASTs with three edge types:
- **Child**: Parent-child syntactic hierarchy
- **Sibling**: Sequential evaluation order
- **DataFlow**: Direct links from variable definitions to usage sites (solves the "long context" problem)

**Current Status:** Phase 1.5 complete - iterative refinement with test-driven feedback. Model predicts both node types and token values, refines programs over multiple iterations using test execution feedback.

## Common Commands

### Testing
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_asg_builder.py

# Run specific test
python -m pytest tests/test_asg_builder.py::TestASGBuilder::test_simple_expression
```

### Data Generation
```bash
# Generate Phase 1 dataset (type-only, 10K samples)
python scripts/01_generate_data.py --num-samples 10000 --output data/processed/train --balanced

# Generate Phase 1.5 dataset (token-level with tests, requires vocabulary)
python scripts/generate_phase1_5_dataset.py

# Build vocabulary from templates (run before generating Phase 1.5 data)
python scripts/build_vocabulary.py

# Test data pipeline
python scripts/test_data_pipeline.py
```

### Training
```bash
# Quick smoke test (1 epoch)
python scripts/02_train_prototype.py --config configs/phase1_prototype.yaml --epochs 1 --batch-size 32

# Full Phase 1 training (50 epochs, ~6-10 hours on RTX 3060)
python scripts/02_train_prototype.py --config configs/phase1_prototype.yaml

# Monitor with TensorBoard
tensorboard --logdir runs/
```

### Evaluation
```bash
# Evaluate checkpoint on test set
python scripts/03_evaluate.py --checkpoint checkpoints/best_model.pt --test-dir data/processed/test

# Skip visualization (faster)
python scripts/03_evaluate.py --checkpoint checkpoints/best_model.pt --no-viz
```

### Demos
```bash
# Demo Phase 1.5 infrastructure and iterative refinement
python demo_phase1_5.py
```

## Git Workflow

**Commit Frequently:** Commit after each logical unit of work. Don't batch multiple unrelated changes into one commit.

Examples of units of work:
- Adding a new template
- Implementing a new feature (e.g., a loss function)
- Fixing a bug
- Adding tests for a module
- Documentation updates

When committing:
```bash
# Stage changes
git add <files>

# Commit with descriptive message
git commit -m "feat(module): description of what changed"

# Or for multiple related files
git add . && git commit -m "feat(module): description"
```

**Commit message format:**
- `feat(module):` - New feature
- `fix(module):` - Bug fix
- `docs:` - Documentation only
- `test(module):` - Adding tests
- `refactor(module):` - Code refactoring

## Architecture & Design

### Abstract Syntax Graphs (ASG)

**Why graphs instead of trees?**
- Traditional ASTs require O(N) hops to link variable definitions to usage sites
- DataFlow edges create O(1) direct connections, solving the "long context" problem
- Example: In `(define (factorial n) (* n (factorial (- n 1))))`, DataFlow edges directly connect the parameter `n` to all 4 usage sites

**Implementation:** `src/data/asg_builder.py` builds ASGs from S-expressions with symbol table analysis for DataFlow edges.

### Graph U-Net Model

**Hierarchical encoder-decoder architecture** (similar to image segmentation U-Nets):
1. **Encoder:** Graph Attention Networks (GAT) layers + TopK pooling (3 levels, 50% reduction per level)
2. **Bottleneck:** Deep processing at coarsest granularity
3. **Decoder:** Unpooling + skip connections (simplified in Phase 1)

**Two model variants:**
- `GraphUNet` (Phase 1): Type-only prediction (9 node types: SYMBOL, OPERATOR, NUMBER, etc.)
- `IterativeGraphUNet` (Phase 1.5): Token-level prediction (vocabulary of ~95-500 tokens) with iterative refinement

**Why GAT over GCN?**
- GAT learns dynamic edge importance (useful since Child/DataFlow/Sibling edges have different semantic weights)
- Can use GCN for faster but less expressive alternative

**Location:** `src/models/graph_unet.py`

### Phase 1.5: Iterative Refinement

**Key Innovation:** Model refines programs over multiple iterations using test execution feedback.

**Node Features (6-dimensional):**
1. `token_id`: Current token from vocabulary
2. `prev_token_id`: Prediction from previous iteration (provides refinement context)
3. `depth`: AST depth (positional encoding)
4. `sibling_index`: Position among siblings (positional encoding)
5. `iteration`: Current refinement iteration (0-4)
6. `test_signal`: 1.0 if node is on failing test execution path, 0.0 otherwise

**Output:**
- `logits`: [num_nodes, vocab_size] - token predictions
- `confidence`: [num_nodes, 1] - per-node confidence scores (enables early stopping)

**Training Loop:**
1. Corrupt program (curriculum: 20% → 100% over 50 epochs)
2. Generate refinement trajectory (model or random policy)
3. Execute tests, compute test signals via execution tracing
4. Train with multi-objective loss:
   - Reconstruction: predict correct tokens
   - Stability: don't change already-correct nodes (weight=0.1)
   - Correction: fix incorrect nodes (weight=0.5)
   - Confidence: calibrate confidence scores (weight=0.2)

**Inference:** Start with fully masked graph, refine iteratively until tests pass or high confidence achieved.

**Key Components:**
- Vocabulary: `src/data/vocabulary.py`
- Mini-Lisp Interpreter: `src/runtime/interpreter.py`
- Trajectory Generation: `src/training/trajectory.py`
- Iterative Loss: `src/training/denoising_task.py`
- Metrics: `src/training/denoising_metrics.py`
- Inference: `src/inference/inference.py`

### Mini-Lisp Language

**Why Lisp for Phase 1?**
- Minimal syntax (~10 node types) reduces vocabulary size
- Rich structure: recursion, higher-order functions, scoping
- Fast parsing: direct S-expression to ASG conversion
- Clear evaluation: simple interpreter for test execution

**Supported primitives:** `+`, `-`, `*`, `/`, `<`, `>`, `=`, `if`, `let`, `lambda`, `define`

**The approach is language-agnostic** - Graph U-Net architecture and DataFlow edges work for any language. Starting with Lisp validates the approach before scaling to Rust/Python.

### Attention: Implicit vs Explicit

**This architecture does NOT use Transformer-style attention.** Why?

ASG structure provides "implicit attention" via edges:
- **Child edges** encode operator-argument relationships (what attention would learn)
- **Sibling edges** encode sequence (what positional encodings provide)
- **DataFlow edges** solve long-range dependencies (what attention was designed for) in O(E) instead of O(N²)

Multi-hop message passing provides k-hop reasoning:
- Layer 1: Each node sees 1-hop neighbors
- Layer 2: Each node sees 2-hop neighbors
- Layer k: Each node sees k-hop neighbors

**When explicit attention might help:**
- Dynamic edge importance (already handled by GAT)
- Cross-subgraph patterns not captured by edges
- Global context aggregation (handled by U-Net pooling + future scratchpad nodes)

See `docs/architecture.md` §6 for detailed discussion.

## Code Organization

### Source Structure
```
src/
├── data/              # ASG construction and dataset generation
│   ├── asg_builder.py         # AST → ASG conversion with DataFlow edges
│   ├── synthetic_gen.py       # Template-based program generation (40 templates)
│   ├── dataset.py             # PyTorch datasets (Phase 1 & 1.5)
│   ├── vocabulary.py          # Token ↔ ID mapping (Phase 1.5)
│   └── test_generator.py      # Automatic test case generation
├── models/
│   └── graph_unet.py          # GraphUNet & IterativeGraphUNet
├── training/
│   ├── denoising_task.py      # Loss functions (single-step & iterative)
│   ├── denoising_metrics.py   # Evaluation metrics
│   └── trajectory.py          # Refinement trajectory generation (Phase 1.5)
├── runtime/
│   └── interpreter.py         # Mini-Lisp evaluator with execution tracing
├── inference/
│   └── inference.py           # Iterative refinement loop (Phase 1.5)
└── utils/
    └── visualize.py           # ASG visualization with NetworkX
```

### Scripts
```
scripts/
├── 01_generate_data.py        # Synthetic dataset generation
├── 02_train_prototype.py      # Training script with gradient accumulation
├── 03_evaluate.py             # Evaluation with visualization
├── build_vocabulary.py        # Extract vocabulary from templates
├── generate_phase1_5_dataset.py  # Phase 1.5 dataset with tests
└── verify_cuda.sh            # CUDA/GPU verification
```

### Tests
100 tests organized by module:
- `tests/test_asg_builder.py` - ASG construction, DataFlow edges
- `tests/test_dataset.py` - Dataset loading, corruption
- `tests/test_synthetic_gen.py` - Template generation
- `tests/test_vocabulary.py` - Tokenization, roundtrip
- `tests/test_interpreter.py` - Mini-Lisp evaluation (24 tests)
- `tests/test_iterative_model.py` - IterativeGraphUNet forward pass
- `tests/test_trajectory.py` - Trajectory generation, curriculum
- `tests/test_inference.py` - Iterative refinement, early stopping
- `tests/test_denoising_metrics.py` - Metrics calculation

## Development Workflow

### Adding a New Template

1. Edit `src/data/synthetic_gen.py`
2. Add class inheriting from `TemplateGenerator`
3. Implement `generate()` and optionally `generate_tests()`
4. Register in `ALL_GENERATORS` list
5. Rebuild vocabulary: `python scripts/build_vocabulary.py`
6. Regenerate dataset: `python scripts/generate_phase1_5_dataset.py`
7. Run tests: `python -m pytest tests/test_synthetic_gen.py`

### Modifying the Model

**For Phase 1 (type-only):**
- Edit `GraphUNet` in `src/models/graph_unet.py`
- Update config: `configs/phase1_prototype.yaml`
- Test: `python -m pytest tests/test_graph_unet.py` (if exists)

**For Phase 1.5 (iterative refinement):**
- Edit `IterativeGraphUNet` in `src/models/graph_unet.py`
- Update embedding dimensions, heads, or feature projections
- Test: `python -m pytest tests/test_iterative_model.py`

### Understanding Training Results

**Key metrics:**
- **Node Accuracy**: % of nodes with correct type/token (random baseline ~11% for Phase 1, ~1% for Phase 1.5)
- **Exact Match Rate**: % of graphs perfectly reconstructed
- **Edge F1**: Precision/recall for edge prediction
- **Syntax Validity**: % passing Mini-Lisp grammar validation
- **Phase 1.5 metrics:**
  - Improvement: final_accuracy - initial_accuracy
  - Convergence: whether iterations stopped changing
  - Confidence calibration: correct vs incorrect node confidence

**Phase 1 results (Dec 2025):**
- Best validation accuracy: 86.39% (type prediction)
- Model size: 0.63M parameters
- Key finding: Systematic OPERATOR ↔ SYMBOL confusion due to positional overfitting
- Root cause: Limited template diversity (~400-500 unique templates)

### Debugging Tips

**Common issues:**

1. **CUDA OOM:** Reduce batch size in config (try 16 or 8), or reduce hidden_channels
2. **Slow data loading:** Pre-generated `.pt` files should be fast; check `num_workers` (set to 0 for debugging)
3. **Poor accuracy:** Check template diversity with `python scripts/analyze_diversity.py`
4. **Interpreter errors:** Test with `python -m pytest tests/test_interpreter.py -v`
5. **Graph structure issues:** Visualize with `python scripts/show_examples.py`

## Important Constraints

### Graph Validity
- ASG must remain a valid directed acyclic graph (DAG) for tree structure
- DataFlow edges can create cycles (variable can reference itself recursively)
- TopK pooling preserves edge connectivity via score-based node selection

### Memory Optimization for RTX 3060
- Gradient accumulation: effective batch size 128 = 32 × 4 steps
- Mixed precision (AMP): FP16 reduces memory ~40%
- PyG batching: disjoint graph representation is memory-efficient

### Template-Based Generation
Property-based synthesis ensures semantic validity:
- Track operator distributions to prevent mode collapse
- Enforce depth constraints (depth >= 2)
- Require base cases for recursion
- Rejection sampling for malformed programs

**NOT pure grammar sampling** - that produces syntactically valid but semantically meaningless programs like `(+ (+ (+ 1 1) 1) 1)` repeated infinitely.

## Configuration

Edit `configs/phase1_prototype.yaml` to tune:

**Model:**
- `hidden_channels`: Hidden dimension (256 default, reduce to 128 if OOM)
- `depth`: Number of U-Net levels (3 default)
- `num_node_types`: 9 for Phase 1 (8 types + MASK)
- `layer_type`: GAT (default) or GCN

**Training:**
- `epochs`: 50 for full training
- `batch_size`: 32 (reduce if OOM)
- `lr`: 0.001 with warmup
- `gradient_accumulation_steps`: 4 (effective batch = 128)

**Data:**
- `corruption_rate`: 0.2 for Phase 1 (20% nodes masked)
- Phase 1.5 uses curriculum: 20% → 100% over epochs

## Hardware Requirements

**Minimum:**
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- RAM: 16GB
- Storage: ~10GB for 1M samples

**Typical resource usage:**
- Training: ~10GB VRAM
- Batch size: 32 with 4× gradient accumulation
- Time: 6-10 hours for 50 epochs (Phase 1)

## Project Phases

**Phase 1 (Complete):** Type-only denoising auto-encoder
- Task: Predict masked node types (SYMBOL, OPERATOR, NUMBER, etc.)
- Status: 86.39% validation accuracy
- See: `docs/phase1_results.md`, `docs/phase1_analysis.md`

**Phase 1.5 (Complete):** Test-driven iterative refinement
- Task: Token-level prediction with test execution feedback
- Status: Full pipeline implemented (vocabulary, interpreter, trajectory generation, metrics)
- See: `docs/phase1.5.md`, `demo_phase1_5.py`

**Phase 2 (Planned):** Discrete diffusion
- Task: Full graph generation from scratch
- Approach: Tree diffusion with constrained decoding
- See: `docs/plan.md`, `docs/NEXT_STEPS.md`

## References

Key implementation details in:
- `docs/architecture.md` - Design decisions, attention mechanisms, neuro-symbolic approach
- `docs/phase1.5.md` - Complete Phase 1.5 specification with implementation status
- `docs/plan.md` - Original project roadmap
- `README.md` - Quick start guide
