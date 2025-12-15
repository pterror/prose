# Project Design Document: Neuro-Symbolic Graph U-Net for Code Synthesis

## 1. High-Level Objective

To build a specialized, small-scale (**~5M to 100M parameters**) generative model for program synthesis and refactoring.

- **Core Thesis:** Code is fundamentally an **Abstract Syntax Graph (ASG)**, not a linear sequence of text.
- **Differentiation:** Unlike LLMs (7B+ params) that rely on massive knowledge memorization, this model focuses purely on **reasoning structure**, **syntax manipulation**, and **logic**, enabling high performance at a fraction of the size.
- **Deployment Target:** Local consumer hardware (e.g., NVIDIA RTX 3060).

## 2. Core Philosophy & Constraints

- **Graph-Native:** We reject "Code-as-Text." Input/Output is strictly graph-structured (AST/ASG).
- **Neuro-Symbolic:** The neural network proposes actions; a symbolic engine (parser/solver) enforces validity.
- **No Hallucinations:** Constrained decoding renders syntax errors mathematically impossible.
- **Specialization:** The model does not know "trivia" (e.g., who wrote Python). It only knows "rules" (e.g., how to refactor a class).

## 3. Architecture Specification

### A. Data Representation: Abstract Syntax Graph (ASG)

Instead of a standard AST, we use an ASG to capture non-local dependencies.

- **Nodes:** AST tokens (Types, Identifiers, Operators).
- **Edges:**
  - `Child`: Standard syntactic hierarchy.
  - `Sibling`: Execution order.
  - `DataFlow`: Direct links between variable definition and usage (resolves the "long context" problem).

### B. Neural Backbone: Graph U-Net

A hierarchical Encoder-Decoder architecture adapted for non-Euclidean data.

- **Framework:** PyTorch Geometric (PyG).
- **Encoder:** Layers of **GAT (Graph Attention Networks)** or **GCN** to extract local features.
- **Pooling (`gPool`):** Learnable selection of top-$k$ nodes to downsample the graph into "Concept Clusters."
- **Bottleneck:** Deep processing of high-level logic (e.g., algorithm flow).
- **Decoder:** **Unpooling** layers restore the original graph topology using skipped indices from the encoder.
- **Elasticity:** (Optional) Implement **Nemotron-style Elastic Training** (random layer dropping) to train a "Tiny" and "Full" model simultaneously.

### C. Generative Mechanism: Discrete Tree Diffusion

- **Type:** **Forward-Learned Discrete Diffusion (FLDD)** or Absorbing State Diffusion.
- **Process:**
  - **Forward (Noise):** A learned "Curriculum of Destruction" (e.g., deleting subtrees, masking conditional logic).
  - **Reverse (Generation):** The U-Net predicts the missing nodes/edges to restore the graph.
- **Search:** **Parallel Decoding** (predicting multiple non-overlapping repairs simultaneously).

## 4. Inference & Constraints (The "Symbolic" Layer)

To ensure robustness, inference is wrapped in strict constraints:

| Constraint Level        | Mechanism          | Description                                                                                                               |
| :---------------------- | :----------------- | :------------------------------------------------------------------------------------------------------------------------ |
| **Level 1 (Syntax)**    | **Logit Masking**  | Dynamic masking of the softmax output based on the language grammar. (e.g., An `IfStmt` _must_ be followed by a `Block`). |
| **Level 2 (Logic)**     | **SMT/Z3 Solver**  | Verification of value constraints (e.g., "Integer must be > 0"). Used for refinement/rejection sampling.                  |
| **Level 3 (Structure)** | **Graph Validity** | The diffusion process operates on tree edits (Insert/Delete), guaranteeing the topology remains a valid tree.             |

## 5. Implementation Roadmap

### Phase 1: The Prototype (Validation)

- **Goal:** Beat a small Transformer on syntax repair.
- **Scale:** ~5M Parameters.
- **Data:** 1 Million synthetic ASTs (Mini-Lisp or Rust subset).
- **Hardware:** RTX 3060 (12GB VRAM).
- **Task:** Denoising Auto-Encoder (One-shot repair of corrupted trees).

### Phase 2: The Reasoner (Diffusion)

- **Goal:** Iterative refinement of logic.
- **Scale:** Scale up to ~50M-100M.
- **Task:** Full Discrete Diffusion (Iterative generation from "noise").
- **NAS:** Use Differentiable Neural Architecture Search to find the optimal mix of GAT vs. GCN layers.

## 6. Training Feasibility (RTX 3060)

- **Batching:** A batch of 512 small graphs fits easily in 12GB.
- **Bottleneck:** **CPU Data Loading**.
  - _Mitigation:_ Pre-process all dataset files into a single `.pt` (PyTorch binary) or LMDB file. Do **not** parse text to AST during the training loop.
- **Est. Time:** ~6-10 hours for a 5M param prototype (50 epochs).

## 7. Relevant Prior Art & References

- **Tree Diffusion:**
  - _Kapur et al. (2025):_ "Diffusion On Syntax Trees For Program Synthesis" (Core methodology for discrete tree edits).
- **Graph Architectures:**
  - _Gao & Ji (2019):_ "Graph U-Nets" (The `gPool`/`gUnpool` mechanism).
  - _DeepMind:_ "GraphCast" (Example of scaling GNNs for complex physics/logic).
- **Neuro-Symbolic / Tiny Models:**
  - _Ellis et al.:_ "DreamCoder" (Wake/Sleep phases for library learning).
  - _Nvidia (2025):_ "Nemotron Elastic" (Architecture for nested/elastic sub-models).
  - _Microsoft:_ "Phi-4" (Proof that high-quality data > model size).
- **Graph Generation:**
  - _Vignac et al. (2023):_ "DiGress" (Discrete Denoising Diffusion for Graphs).
