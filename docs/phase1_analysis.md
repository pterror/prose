# Phase 1 Analysis & Critical Findings

**Last Updated:** 2025-12-16  
**Status:** ⚠️ Task redesign needed - current denoising task is fundamentally limited

---

## Executive Summary

We successfully built and trained a Graph U-Net for Mini-Lisp ASG denoising, but discovered **critical limitations in the task design**. The current approach only predicts node **types** (SYMBOL, OPERATOR, etc.), not their actual **values** (`+` vs `-`, `x` vs `y`), making it of limited practical use.

**Key Achievement:**

- ✅ Expanded template diversity: 12% → 41% uniqueness (40 templates, 56 unique structures)
- ✅ Implemented position encodings (tree depth + sibling index)
- ✅ Built working Graph U-Net architecture (0.64M parameters)
- ⚠️ Model accuracy: 52.4% on masked nodes (but this reveals deeper issues)

**Critical Finding:**
The denoising task is ill-suited for real code synthesis because **graph structure alone cannot determine semantic choices** (e.g., which operator or variable name).

---

## 1. Template Diversity Expansion

### Achievements

| Metric            | Before | After | Improvement |
| ----------------- | ------ | ----- | ----------- |
| Templates         | 4      | 40    | 10×         |
| Unique structures | ~16    | 56    | 3.5×        |
| Uniqueness %      | 12%    | 41%   | 3.4×        |

### Template Breakdown

- **Core templates (7):** ArithmeticTemplate (15 patterns), RecursionTemplate, LambdaTemplate, LetBindingTemplate, etc.
- **Simple templates (3):** [simple_templates.py](file:///home/me/git/prose/src/data/simple_templates.py)
- **Extra templates (8):** [extra_templates.py](file:///home/me/git/prose/src/data/extra_templates.py)
- **Mega templates (12):** [mega_templates.py](file:///home/me/git/prose/src/data/mega_templates.py)
- **Ultra templates (10):** [ultra_templates.py](file:///home/me/git/prose/src/data/ultra_templates.py)

### Datasets Generated

```
data/processed/
├── train/   2000 samples (50 per template)
├── val/      480 samples (12 per template)
└── test/     480 samples (12 per template)
```

**Key Insight:** While we improved diversity significantly, even 41% uniqueness on 100 samples plateaus around 55 total unique structures due to template constraints.

---

## 2. Position Encodings Implementation

### What We Built

Added tree-aware position encodings to help model distinguish structurally similar nodes:

**Node Features:** `[node_type, depth, sibling_index]`

- `node_type`: 0-7 (SYMBOL, NUMBER, LIST, DEFINE, LAMBDA, IF, LET, OPERATOR)
- `depth`: Distance from root (0 = root)
- `sibling_index`: Position among siblings (0, 1, 2, ...)

**Model Integration:**

- Node type → 128-dim embedding
- Position features → normalized → 32-dim projection
- Concatenated: 160-dim input to graph layers

### Data Structure

```python
data.x.shape: [num_nodes, 3]
data.edge_index.shape: [2, num_edges]
data.edge_attr.shape: [num_edges]  # Edge type: CHILD, SIBLING, DATAFLOW
```

---

## 3. Model Architecture & Training

### Graph U-Net Configuration

```yaml
Model:
  - Input: 160 dims (128 node emb + 32 position emb)
  - Hidden: 256 dims
  - Depth: 3 layers (encoder/bottleneck/decoder)
  - Pooling ratio: 0.5
  - Graph layer: GAT (Graph Attention)
  - Output: 9 classes (8 node types + MASK token)
  - Parameters: 0.64M

Training:
  - Epochs: 20
  - Batch size: 32 (effective: 128 with gradient accumulation)
  - Learning rate: 0.001 (warmup + cosine decay)
  - Corruption rate: 20% of nodes masked
  - Optimizer: Adam (weight decay 1e-4)
```

### Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| ----: | ---------: | --------: | -------: | ------: |
|     1 |       2.65 |     14.2% |     2.51 |   17.3% |
|    10 |       1.32 |     50.1% |     1.28 |   48.6% |
|    20 |       1.15 |     55.8% |     1.23 |   50.9% |

**Best model:** Epoch 16, Val Acc 54.2%

---

## 4. Critical Bug Discovery & Fix

### The Bug

**Original code computed loss on ALL nodes**, not just masked ones:

```python
# WRONG - evaluates all nodes including uncorrupted
loss = F.cross_entropy(predictions, target_labels)
accuracy = (predictions.argmax(-1) == target_labels).mean()
```

This gave misleadingly high accuracy (~64%) because:

- Model could "cheat" by copying uncorrupted neighbors
- Only 20% of nodes were actually masked
- Metric wasn't measuring denoising capability

### The Fix

```python
# CORRECT - only evaluate masked nodes
masked_mask = (corrupted_types == mask_token_id)
masked_predictions = predictions[masked_mask]
masked_targets = target_labels[masked_mask]
loss = F.cross_entropy(masked_predictions, masked_targets)
```

**Result:** Accuracy dropped from 64% → 52.4% (the true metric)

---

## 5. Performance Analysis & The Critical Discovery

### Confusion Matrix (Validation Set, Masked Nodes Only)

```
True          Predicted Type                                    Accuracy
Type      SYMBOL NUMBER  LIST DEFINE LAMBDA    IF   LET OPERATOR  Per Class
────────────────────────────────────────────────────────────────────────────
SYMBOL       286     15    34      0      0     2     0       9     82.7%
NUMBER        50     23    22      0      0     0     0       0     24.2%
LIST          79      1   131      0      0     0     0       0     62.1%
DEFINE         0      0    15      0      0     0     0       0      0.0% ❌
LAMBDA         8      0    12      0      0     0     0       0      0.0% ❌
IF             7      0    13      0      0     0     0       0      0.0% ❌
LET            0      0    16      0      0     0     0       0      0.0% ❌
OPERATOR     111      4     5      0      0     0     0       4      3.2% ❌
────────────────────────────────────────────────────────────────────────────
Overall: 444/847 = 52.4%
```

### What This Reveals

**The model has learned to:**

- ✅ Distinguish SYMBOL (82.7% accuracy)
- ✅ Distinguish LIST (62.1% accuracy)
- ⚠️ Poorly distinguish NUMBER (24.2% - often predicts SYMBOL)
- ❌ Completely fails on OPERATOR (3.2% - almost always predicts SYMBOL)
- ❌ Never predicts structure keywords (DEFINE, LAMBDA, IF, LET)

**Root Cause:** Severe class imbalance + insufficient training

- SYMBOL: 35% of all nodes → model defaults to it
- LIST: 27% → learned reasonably
- OPERATOR: 19% → barely learned
- All others: <5% each → never learned

---

## 6. The Fundamental Problem: Task Design Flaw

### What the Data Actually Stores

**Current representation:**

```python
x = [node_type, depth, sibling_index]
#    ├─ 0=SYMBOL, 7=OPERATOR, etc.
#    ├─ Position in tree
#    └─ Order among siblings
```

**What's MISSING:** Actual node values!

- OPERATOR nodes: We know it's an operator, but NOT whether it's `+`, `-`, `*`, `/`
- SYMBOL nodes: We know it's a symbol, but NOT whether it's `x`, `y`, `foo`, etc.
- NUMBER nodes: We know it's a number, but NOT whether it's `1`, `2`, `42`, etc.

### Why This Is Problematic

**Example:** Consider `(+ 1 2)` vs `(* 1 2)`

Both have identical graph structure:

```
LIST
├── OPERATOR (depth=1, sibling=0)
├── NUMBER   (depth=1, sibling=1)
└── NUMBER   (depth=1, sibling=2)
```

**When we mask the OPERATOR:**

- Graph structure: Same
- Position encodings: Same
- Neighboring nodes: Same
- **From this information alone, both `+` and `*` are equally valid!**

### Implications

1. **The task is well-defined** - there IS a single correct node TYPE per position
2. **But it's not useful** - predicting "this should be an OPERATOR" is trivial
3. **The real challenge** - predicting WHICH operator requires:
   - Semantic understanding (not just syntax)
   - Data flow analysis
   - Or: storing actual values (which we don't)

---

## 7. Why Current Approach Has Limited Practical Value

### What the Model CAN Do

- ✅ Recognize "this position should have a SYMBOL"
- ✅ Recognize "this position should have a LIST"
- ⚠️ Sometimes recognize OPERATORs vs SYMBOLs

### What the Model CANNOT Do

- ❌ Determine which specific operator (`+` vs `-` vs `*` vs `/`)
- ❌ Determine which variable name (`x` vs `y` vs `foo`)
- ❌ Determine numeric values
- ❌ Ensure semantic correctness (e.g., type safety)

### Real-World Code Synthesis Needs

For practical code generation, we need to:

1. **Predict actual values**, not just types
2. **Maintain semantic consistency** (variables used must be defined)
3. **Respect type constraints** (can't add strings and numbers)
4. **Generate novel, correct programs**, not just reconstruct corrupted ones

---

## 8. Lessons Learned

### What Worked

1. **Template expansion was valuable** - 41% uniqueness provides diverse training data
2. **Position encodings are implemented correctly** - model can access tree structure
3. **Graph U-Net architecture is sound** - 0.64M params, trainable, converges
4. **Corruption-based training works** - model learns from self-supervised signal

### What Didn't Work

1. **Node type prediction is too limited** - doesn't capture program semantics
2. **Graph structure alone is insufficient** - can't determine semantic choices
3. **20% corruption may be too easy** - model just uses nearest neighbors
4. **Class imbalance killed performance** - SYMBOL/LIST dominate, others never learned

### Technical Debt

1. **eval script is broken** - expects Data objects but gets tensors
2. **Old checkpoints exist** - epoch_50.pt is from previous dataset
3. **Visualization not tested** - ASGVisualizer methods untested

---

## 9. Next Steps: Designing a Real Problem

### Criteria for a Practical Task

1. **Clear utility:** What real-world problem does it solve?
2. **Well-defined ground truth:** One correct answer, deterministically checkable
3. **Requires learning:** Not solvable by simple heuristics
4. **Graph structure is essential:** GNN should have advantage over other approaches

### Candidate Tasks

#### Option A: Type-Aware Code Completion

**Problem:** Given partial program + type signatures, predict next token

**Example:**

```lisp
(define (add-one x : Int) : Int
  (+ x __))  ; Predict: 1 (must be Int literal or Int variable)
```

**Advantages:**

- Clear utility (helps programmers)
- Ground truth is deterministic
- Requires type reasoning (graph structure helps)

**Challenges:**

- Need to add type annotations to Mini-Lisp
- Vocabulary can be open-ended

#### Option B: Syntax Error Correction

**Problem:** Given program with syntax error, predict fix

**Example:**

```lisp
(if (< x 0) -1)  ; Missing else branch
→ (if (< x 0) -1 1)  ; Predict: add "1"
```

**Advantages:**

- Clear utility (helps learners)
- Ground truth from correct programs
- Natural corruption model

**Challenges:**

- Need to define "plausible" errors
- May be too easy (just syntax rules)

#### Option C: Semantic Bug Localization

**Problem:** Given program with wrong output, predict buggy node

**Example:**

```lisp
(define (abs x)
  (if (< x 0) x (- x)))  ; Bug: should be (- x) in then-branch
```

**Advantages:**

- High practical value
- Requires semantic understanding
- Graph structure crucial for data flow

**Challenges:**

- Need program execution / tests
- Hard to generate buggy programs systematically

#### Option D: Program Synthesis from Spec

**Problem:** Given input/output examples, synthesize program

**Example:**

```
Input: [(1,2), (2,3), (3,4)]
Output: Second element of pair
→ Generate: (lambda (p) (cdr (car p)))
```

**Advantages:**

- Ultimate goal of code synthesis
- Clear success criterion (passes tests)
- Requires creative generation

**Challenges:**

- Extremely hard (open research problem)
- May need different architecture (decoder)
- Evaluation is expensive (running code)

### Recommendation

Start with **Option A (Type-Aware Code Completion)** or **Option B (Syntax Error Correction)** because:

1. Both have clear utility and ground truth
2. Can reuse existing infrastructure (templates, position encodings)
3. Incrementally harder than current denoising
4. Can evolve toward full synthesis later

---

## 10. Current Codebase State

### What's Implemented & Working

- ✅ Template system with 40 diverse templates
- ✅ ASG builder with position encodings
- ✅ Dataset generation (2960 balanced samples)
- ✅ Graph U-Net architecture (GAT-based)
- ✅ Training infrastructure (warmup, gradient accumulation, checkpointing)
- ✅ Denoising loss (correctly computes masked-only metrics)

### What Needs Fixing

- ⚠️ Evaluation script (expects Data but gets tensors)
- ⚠️ Visualization untested
- ⚠️ Class imbalance (if continuing with denoising)

### What Needs Rethinking

- 🔄 Task definition (node types → something more useful)
- 🔄 Data representation (add values, types, or semantics?)
- 🔄 Success metrics (what does "good" look like?)

---

## 11. Technical Details for Reference

### File Structure

```
prose/
├── src/
│   ├── data/
│   │   ├── asg_builder.py          # ASG construction + position encodings
│   │   ├── synthetic_gen.py        # 40 templates, generation logic
│   │   ├── dataset.py              # PyTorch Dataset (corruption)
│   │   ├── simple_templates.py     # 3 basic templates
│   │   ├── extra_templates.py      # 8 medium templates
│   │   ├── mega_templates.py       # 12 complex templates
│   │   └── ultra_templates.py      # 10 final diversity templates
│   ├── models/
│   │   └── graph_unet.py           # Graph U-Net with GAT layers
│   └── training/
│       ├── denoising_task.py       # Loss function (masked-only)
│       └── denoising_metrics.py    # Evaluation metrics
├── scripts/
│   ├── 01_generate_data.py         # Dataset generation
│   ├── 02_train_prototype.py       # Training script
│   └── 03_evaluate.py              # Evaluation (broken)
├── configs/
│   └── phase1_prototype.yaml       # Training configuration
└── data/processed/                 # Generated datasets
```

### Key Configuration

```yaml
model:
  in_channels: 128
  hidden_channels: 256
  depth: 3
  num_node_types: 9
  layer_type: GAT

training:
  epochs: 20
  batch_size: 32
  lr: 0.001
  corruption_rate: 0.2
  gradient_accumulation_steps: 4

data:
  mask_token_id: 8
```

### Node Type Enum

```python
class NodeType(Enum):
    SYMBOL = 0    # Variable names (x, y, foo)
    NUMBER = 1    # Numeric literals (1, 2, 42)
    LIST = 2      # S-expression lists
    DEFINE = 3    # Definition keyword
    LAMBDA = 4    # Lambda keyword
    IF = 5        # Conditional keyword
    LET = 6       # Let binding keyword
    OPERATOR = 7  # Arithmetic ops (+, -, *, /)
    # MASK = 8    # Corruption token (not in enum)
```

---

## 12. Questions to Answer Next

### Task Design

1. What real-world problem should we solve?
2. Do we need to extend Mini-Lisp (e.g., add types)?
3. Should we pivot to a different language (Python, simple imperative)?

### Data Representation

1. Should we encode actual values (tokenize operators/vars)?
2. Do we need semantic information (types, scopes, data flow)?
3. How to represent "partial" or "incomplete" programs?

### Architecture

1. Is Graph U-Net the right choice, or should we use:
   - Transformer (for sequence modeling)?
   - Graph Transformer (for long-range dependencies)?
   - Encoder-decoder (for generation tasks)?

### Evaluation

1. What metrics matter for the new task?
2. How to create a meaningful test set?
3. What baseline should we compare against?

---

## 13. Commands to Resume Work

### Regenerate Datasets (if needed)

```bash
uv run python scripts/01_generate_data.py --generate-all
```

### Train Model

```bash
uv run python scripts/02_train_prototype.py \
  --config configs/phase1_prototype.yaml \
  --epochs 50
```

### Check Model Predictions

```python
import torch
from src.models.graph_unet import GraphUNet

checkpoint = torch.load('checkpoints/best_model.pt')
model = GraphUNet(**checkpoint['config']['model'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Inspect Data

```python
import torch
data = torch.load('data/processed/train/sample_000000.pt', weights_only=False)
print(f"Nodes: {data.x.shape}, Edges: {data.edge_index.shape}")
```

---

## 14. References & Resources

### Papers

- Graph U-Net: [Gao & Ji 2019](https://arxiv.org/abs/1905.05178)
- GAT: [Veličković et al. 2018](https://arxiv.org/abs/1710.10903)
- Code representation learning: Survey papers

### Codebase

- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Mini-Lisp grammar: Simple s-expressions

---

**Next Session Goal:** Define a practical code synthesis task and redesign the data pipeline to support it.
