# TODO

## Phase 1: Prototype Completion

### Dataset & Training

- [ ] Generate full training dataset (100K-1M samples, balanced across templates)
- [ ] Generate test set (10K-20K samples)
- [ ] Run full 50-epoch training on RTX 3060
- [ ] Profile GPU memory usage and optimize batch size
- [ ] Add learning rate warmup + cosine schedule tuning

### Evaluation

- [ ] Implement evaluation metrics:
  - [ ] Exact match rate (perfect reconstruction)
  - [ ] Node accuracy (per-node correctness)
  - [ ] Edge F1 (precision/recall for edge prediction)
  - [ ] Syntax validity (% parseable as Mini-Lisp)
- [ ] Create evaluation script (`scripts/03_evaluate.py`)
- [ ] Visualize reconstructions (corrupted → prediction → ground truth)

### Baselines

- [ ] Implement Transformer baseline (5M params, same task)
- [ ] Compare Graph U-Net vs Transformer on:
  - [ ] Exact match rate
  - [ ] Training speed (samples/sec)
  - [ ] Memory usage

### Model Improvements

- [ ] Implement proper unpooling (restore full graph topology)
- [ ] Add position encodings (depth in tree, sibling index)
- [ ] Experiment with edge-aware attention (different weights for Child/Sibling/DataFlow)
- [ ] Try GCN vs GAT comparison

### Code Quality

- [ ] Add unit tests:
  - [ ] Test ASG builder with complex programs
  - [ ] Test corruption preserves graph validity
  - [ ] Test model forward pass shapes
- [ ] Add type hints coverage check
- [ ] Profile data loading bottlenecks

## Phase 2: Scaling & Diffusion

### Model Scaling

- [ ] Scale to 50M-100M parameters
- [ ] Implement gradient checkpointing for memory efficiency
- [ ] Add mixed precision training (AMP)
- [ ] Benchmark training time on full dataset

### Discrete Diffusion

- [ ] Research discrete diffusion on graphs (DiGress paper)
- [ ] Implement forward diffusion (curriculum of destruction):
  - [ ] Node masking
  - [ ] Node deletion
  - [ ] Edge deletion
  - [ ] Subtree deletion
- [ ] Implement reverse diffusion (iterative denoising)
- [ ] Add diffusion scheduler (timesteps, noise schedule)

### Symbolic Constraints

- [ ] Implement logit masking for grammar enforcement:
  - [ ] Build Mini-Lisp grammar constraint table
  - [ ] Dynamic masking based on partial graph state
- [ ] Add SMT solver integration (Z3) for value constraints
- [ ] Implement constrained beam search

### Language Expansion

- [ ] Design Python subset grammar
- [ ] Implement Python ASG builder (use tree-sitter)
- [ ] Add control flow graph edges (if/while/exception)
- [ ] Handle type system constraints

## Phase 3: Applications

### Code Repair

- [ ] Create synthetic bug dataset (remove statements, corrupt expressions)
- [ ] Fine-tune model on repair task
- [ ] Evaluate on real bugs (from GitHub)

### Code Generation

- [ ] Create specification → code dataset
- [ ] Implement conditional generation (given partial program)
- [ ] Add natural language → ASG translation

### Interpretability

- [ ] Visualize learned node embeddings (t-SNE/UMAP)
- [ ] Attention map visualization (which edges matter most?)
- [ ] Ablation studies (remove edge types, compare performance)

## Infrastructure

### Experiment Tracking

- [ ] Set up Weights & Biases integration
- [ ] Create experiment configs for hyperparameter sweeps
- [ ] Add automated result reporting

### Documentation

- [ ] Training hyperparameter tuning guide
- [ ] Model architecture diagram (visual)
- [ ] Dataset generation best practices
- [ ] Contribution guide

### Deployment

- [ ] Export model to ONNX for inference
- [ ] Build CLI tool for code repair
- [ ] Add web demo (gradio/streamlit)

## Research Questions

- [ ] How does graph structure impact learning vs sequence models?
- [ ] Does DataFlow edge improve accuracy significantly? (ablation study)
- [ ] What's the optimal corruption rate? (sweep 10%-50%)
- [ ] Can we learn useful representations without supervised labels?
- [ ] How to handle very large programs (1000+ nodes)?
