# TODO

## Project 1: Graph U-Net for Code Synthesis

### Phase 1: Prototype Completion

#### Dataset & Training

> **Note**: Using hybrid approach - started with 2K/500/500 split (4-6× duplication) based on template diversity analysis. Will expand templates before scaling dataset.

- [x] Generate initial training dataset (2K samples, balanced across templates)
- [x] Generate test set (500 val + 500 test samples)
- [ ] Run full 50-epoch training on RTX 3060
- [ ] Profile GPU memory usage and optimize batch size
- [ ] Add learning rate warmup + cosine schedule tuning

#### Evaluation

- [x] Implement evaluation metrics:
  - [x] Exact match rate (perfect reconstruction)
  - [x] Node accuracy (per-node correctness)
  - [x] Edge F1 (precision/recall for edge prediction)
  - [x] Syntax validity (% parseable as Mini-Lisp)
- [x] Create evaluation script (`scripts/03_evaluate.py`)
- [x] Visualize reconstructions (corrupted → prediction → ground truth)

#### Baselines

- [ ] Implement Transformer baseline (5M params, same task)
- [ ] Compare Graph U-Net vs Transformer on:
  - [ ] Exact match rate
  - [ ] Training speed (samples/sec)
  - [ ] Memory usage

#### Model Improvements

- [ ] Implement proper unpooling (restore full graph topology)
- [ ] Add position encodings (depth in tree, sibling index)
- [ ] Experiment with edge-aware attention (different weights for Child/Sibling/DataFlow)
- [ ] Try GCN vs GAT comparison

#### Code Quality

- [ ] Add unit tests:
  - [ ] Test ASG builder with complex programs
  - [ ] Test corruption preserves graph validity
  - [ ] Test model forward pass shapes
- [ ] Add type hints coverage check
- [ ] Profile data loading bottlenecks

### Phase 2: Scaling & Diffusion

#### Model Scaling

- [ ] Scale to 50M-100M parameters
- [ ] Implement gradient checkpointing for memory efficiency
- [ ] Add mixed precision training (AMP)
- [ ] Benchmark training time on full dataset

#### Discrete Diffusion

- [ ] Research discrete diffusion on graphs (DiGress paper)
- [ ] Implement forward diffusion (curriculum of destruction):
  - [ ] Node masking
  - [ ] Node deletion
  - [ ] Edge deletion
  - [ ] Subtree deletion
- [ ] Implement reverse diffusion (iterative denoising)
- [ ] Add diffusion scheduler (timesteps, noise schedule)

#### Symbolic Constraints

- [ ] Implement logit masking for grammar enforcement:
  - [ ] Build Mini-Lisp grammar constraint table
  - [ ] Dynamic masking based on partial graph state
- [ ] Add SMT solver integration (Z3) for value constraints
- [ ] Implement constrained beam search

#### Language Expansion

- [ ] Design Python subset grammar
- [ ] Implement Python ASG builder (use tree-sitter)
- [ ] Add control flow graph edges (if/while/exception)
- [ ] Handle type system constraints

### Phase 3: Applications

#### Code Repair

- [ ] Create synthetic bug dataset (remove statements, corrupt expressions)
- [ ] Fine-tune model on repair task
- [ ] Evaluate on real bugs (from GitHub)

#### Code Generation

- [ ] Create specification → code dataset
- [ ] Implement conditional generation (given partial program)
- [ ] Add natural language → ASG translation

#### Interpretability

- [ ] Visualize learned node embeddings (t-SNE/UMAP)
- [ ] Attention map visualization (which edges matter most?)
- [ ] Ablation studies (remove edge types, compare performance)

### Infrastructure

#### Experiment Tracking

- [ ] Set up Weights & Biases integration
- [ ] Create experiment configs for hyperparameter sweeps
- [ ] Add automated result reporting

#### Documentation

- [ ] Training hyperparameter tuning guide
- [ ] Model architecture diagram (visual)
- [ ] Dataset generation best practices
- [ ] Contribution guide

#### Deployment

- [ ] Export model to ONNX for inference
- [ ] Build CLI tool for code repair
- [ ] Add web demo (gradio/streamlit)

### Research Questions

- [ ] How does graph structure impact learning vs sequence models?
- [ ] Does DataFlow edge improve accuracy significantly? (ablation study)
- [ ] What's the optimal corruption rate? (sweep 10%-50%)
- [ ] Can we learn useful representations without supervised labels?
- [ ] How to handle very large programs (1000+ nodes)?

## Project 2: Expansion/Synergy

### Phase 1: The "Architect" (GFlowNet)

**Goal:** Generate diverse, valid _Abstract Syntax Graphs (ASGs)_ from a prompt.
**Why:** We need a "Drafting Engine" that explores multiple valid algorithmic approaches (e.g., recursive vs. iterative) rather than just predicting the next most likely token.

- **Reference Paper:** _Bengio et al. (2021) "GFlowNet Foundations"_.
- **Codebase:** Standalone (Python + Torch Geometric).

**TODOs:**

- [ ] **Define the Action Space:**
  - Instead of "Next Token," define "Graph Edges."
  - Actions: `AddNode(Type)`, `AddEdge(Src, Dst, Type)`.
- [ ] **Define the Reward ($R(x)$):**
  - Input: A completed ASG.
  - Output: `Is_Valid_Syntax(x) * (1.0 + Complexity_Score(x))`.
  - _Note:_ Do not check logical correctness yet, only structural validity.
- [ ] **Implement Trajectory Balance Loss ($L_{TB}$):**
  - The core GFlowNet objective. It forces the sum of probabilities flowing into a state to equal the probability flowing out.
- [ ] **The "Forward Policy" (Neural Net):**
  - Use a small **Graph Transformer** or GNN.
  - Input: Partial Graph. Output: Logits over `AddNode`/`AddEdge`.

**Outcome:** A model that spits out 100 _different_ plausible graph structures for a given problem.

### Phase 2: The "Critic" (Energy-Based Model)

**Goal:** Learn the "Physics of Logic." Define an energy function $E(x)$ where valid, correct programs have low energy.
**Why:** GFlowNets are good at diversity but bad at precision. We need a gradient field that pulls the draft toward correctness.

- **Reference Paper:** _Du et al. (2024) "Learning Iterative Reasoning through Energy Diffusion (IRED)"_.
- **Codebase:** Integrated with Phase 1 (Shared GNN backbone?).

**TODOs:**

- [ ] **Define Energy Function $E(x, y)$:**
  - $E_{syntax}$: "Are there dangling edges?" (Hard constraint).
  - $E_{semantic}$: "Does the variable flow match the type?" (Learned).
  - $E_{goal}$: "Does the execution match the I/O examples?" (Symbolic).
- [ ] **Training (Contrastive Divergence):**
  - Positive Samples ($x^+$): Real, working code.
  - Negative Samples ($x^-$): Code with subtle bugs (mutated).
  - _Objective:_ Push down energy of $x^+$, pull up energy of $x^-$.
- [ ] **Inference (Langevin Dynamics):**
  - Take a draft from Phase 1.
  - Update graph structure using $\nabla_x E(x)$.
  - _Trick:_ since graphs are discrete, use **Gibbs Sampling** or **Discrete Gradient Estimators** (like Straight-Through Gumbel).

### Phase 3: The Synergy (The "Draft-and-Refine" Loop)

**Goal:** Combine them into a single inference pipeline.

**The Workflow:**

1.  **User:** "Write a sort function."
2.  **GFlowNet:** Samples 10 diverse topological skeletons (Recursive, Iterative, Library-call).
3.  **Filter:** Prune the 5 invalid skeletons immediately.
4.  **EBM (The Refiner):** Takes the top 5 drafts. Performs "Reasoning" by minimizing their Energy (fixing logical bugs, aligning types) over $k$ steps.
5.  **Output:** The lowest-energy graph is converted to code.

### Technical Reference List

| Approach        | Key Paper                                                                | Why you need it                                                 |
| :-------------- | :----------------------------------------------------------------------- | :-------------------------------------------------------------- |
| **GFlowNet**    | [GFlowNet Foundations](https://arxiv.org/abs/2111.09266)                 | The math for training the "Drafting" model.                     |
| **EBM**         | [Implicit Generation (OpenAI)](https://arxiv.org/abs/1903.07138)         | Explains how to train Energy functions stably.                  |
| **Reasoning**   | [IRED (Iterative Reasoning)](https://energy-based-model.github.io/ired/) | **Crucial.** Shows how to sequence energy landscapes for logic. |
| **Graph Logic** | [NeuroSAT](https://arxiv.org/abs/1802.03685)                             | How to represent boolean logic as message passing.              |

### Recommendation: Independent or Integrated?

**Start Independent, Merge Later.**

1.  **Project A (GFlowNet):** Just try to generate _valid_ random graphs (Syntactically correct LISP trees).
    - _Success Condition:_ It generates trees that parse 90% of the time.
2.  **Project B (EBM):** Just try to _fix_ broken graphs.
    - _Success Condition:_ You feed it a tree with a deleted node, and it restores it based on context.
3.  **Project C (Synergy):** Feed output of A into B.
