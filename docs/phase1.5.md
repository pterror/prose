# Phase 1.5: Test-Driven Iterative Refinement

**Goal:** Transform the Graph U-Net from type-only one-shot denoising to value-level iterative refinement guided by test execution.

**Status:** 🚧 In Progress (Infrastructure ✅ Complete)
**Target:** Bridge Phase 1 → Phase 2 (Project 2 roadmap)
**Last Updated:** 2025-12-16

---

## Executive Summary

### Current State (Phase 1)

- ✅ Graph U-Net trained on type prediction (SYMBOL, OPERATOR, etc.)
- ✅ 52.4% accuracy on masked nodes
- ❌ **Cannot predict actual values** (`+` vs `-`, `x` vs `y`)
- ❌ **Task is underspecified** - multiple semantically valid programs exist

### Target State (Phase 1.5)

- ✅ Predict **both type and value** (vocabulary of ~100-1000 tokens)
- ✅ **Test-driven refinement** - denoise programs to satisfy I/O constraints
- ✅ **Iterative architecture** - model refines over multiple passes
- ✅ **Curriculum learning** - 50% → 100% corruption (generation from scratch)

### Why This Matters

1. **Makes the task well-defined**: Tests provide ground truth for semantic correctness
2. **Enables practical use**: Programs must pass tests, not just parse
3. **Bridges to Project 2**: U-Net becomes "The Engineer" in the GFlowNet/U-Net/EBM sandwich
4. **Prepares for diffusion**: 100% corruption = unconditional generation

---

## Proposed Changes

### 1. Data Representation

#### 1.1 Add Value-Level Tokenization

**Current:**

```python
Node = {
    'type': NodeType,  # SYMBOL, OPERATOR, NUMBER, etc. (8 classes)
    'depth': int,
    'sibling_index': int
}
```

**New:**

```python
# Build vocabulary from templates
VOCAB = {
    # Operators
    '+': 0, '-': 1, '*': 2, '/': 3, '<': 4, '>': 5, '=': 6,
    # Keywords
    'define': 7, 'lambda': 8, 'if': 9, 'let': 10,
    # Common symbols
    'x': 11, 'y': 12, 'n': 13, 'acc': 14, 'foo': 15, ...
    # Numbers (discretized or special tokens)
    '0': 50, '1': 51, '2': 52, ..., '<NUM>': 99,
    # Structural
    '(': 100, ')': 101,
    # Special
    '<MASK>': 102, '<PAD>': 103
}

Node = {
    'token_id': int,        # Index into VOCAB (replaces 'type')
    'depth': int,
    'sibling_index': int,
    'prev_token_id': int,   # From previous iteration (for refinement)
    'iteration': int,       # Current iteration number (0-4)
    'test_signal': float    # 1.0 if node is in failing execution path
}
```

**Implementation:**

- [NEW] [vocabulary.py](file:///home/me/git/prose/src/data/vocabulary.py)
  - Extract all unique tokens from 40 templates
  - Build bidirectional mapping: `token ↔ id`
  - Handle unknown tokens (rare symbols/numbers)
  - Export: `VOCAB_SIZE` (estimated ~200-500 tokens)

#### 1.2 Add Test Case Representation

```python
TestCase = {
    'inputs': List[Any],      # Function arguments
    'expected_output': Any,   # Correct return value
    'actual_output': Any,     # Model's prediction output
    'passing': bool           # Test status
}

ProgramData = {
    'graph': Data,                    # PyG graph
    'tests': List[TestCase],          # 3-5 tests per program
    'execution_trace': Tensor,        # [num_nodes] - which nodes affected output
    'failing_nodes': Tensor           # [num_nodes] - 1.0 if in failing path
}
```

**Implementation:**

- [MODIFY] [synthetic_gen.py](file:///home/me/git/prose/src/data/synthetic_gen.py)
  - Each template generates: `(program, tests)` pairs
  - Tests derived from template semantics (e.g., `add-one` → test `f(5)=6`)
  - Store tests with dataset samples

---

### 2. Mini-Lisp Interpreter

> [!IMPORTANT] > **Critical Dependency**: All downstream components require test execution.

#### 2.1 Basic Interpreter

**Features:**

- Execute S-expressions: `(define ...)`, `(lambda ...)`, `(if ...)`, arithmetic
- Return output or error (syntax/runtime)
- **Fast enough** for training loop (1000s of evals/sec)

**Implementation:**

- [NEW] [interpreter.py](file:///home/me/git/prose/src/runtime/interpreter.py)
  - Recursive evaluator for Mini-Lisp AST
  - Environment for variable bindings
  - Primitive operations: `+, -, *, /, <, >, =, if, let, lambda, define`
  - Error handling: division by zero, undefined vars, type mismatches

```python
class MiniLispInterpreter:
    def eval(self, ast: AST, env: Env) -> Any:
        """Evaluate AST in environment, return result or raise error."""

    def run_tests(self, program: AST, tests: List[TestCase]) -> List[bool]:
        """Execute program on test inputs, return pass/fail for each."""

    def trace_execution(self, program: AST, inputs: List[Any]) -> Set[NodeID]:
        """Return set of node IDs visited during execution."""
```

#### 2.2 Execution Tracing

**Goal:** Identify which nodes contributed to test failure.

**Approach:**

1. Instrument interpreter to track node access during evaluation
2. Mark nodes on execution path
3. If test fails, label those nodes with `test_signal = 1.0`

**Example:**

```lisp
(define (mystery x)
  (- x 1))      ; BUG: should be (+ x 1)

; Test: f(5) = 6
; Execution trace: [define, mystery, x, -, x, 1]
; Test fails: ALL traced nodes get signal 1.0
```

**Implementation:**

- [MODIFY] [interpreter.py](file:///home/me/git/prose/src/runtime/interpreter.py)
  - Add `trace_mode: bool` parameter
  - Collect node IDs during eval
  - Return `(result, traced_nodes)`

---

### 3. Architecture Changes

#### 3.1 Iterative Graph U-Net

**Current:**

```python
class GraphUNet(nn.Module):
    def forward(self, x, edge_index):
        # x: [num_nodes, 3] - [type, depth, sibling]
        # Output: [num_nodes, 9] - logits over 8 types + MASK
```

**New:**

```python
class IterativeGraphUNet(nn.Module):
    def __init__(
        self,
        vocab_size: int = 500,
        hidden_channels: int = 256,
        depth: int = 3,
        max_iterations: int = 5
    ):
        # Node feature projections
        self.token_embedding = nn.Embedding(vocab_size, 128)
        self.prev_token_embedding = nn.Embedding(vocab_size, 32)
        self.position_projection = nn.Linear(2, 32)  # depth, sibling

        # Iteration conditioning
        self.iteration_embedding = nn.Embedding(max_iterations, 32)
        self.test_signal_projection = nn.Linear(1, 32)

        # Graph layers (unchanged)
        self.encoder = ...
        self.decoder = ...

        # Output heads
        self.token_predictor = nn.Linear(hidden_channels, vocab_size)
        self.confidence_head = nn.Linear(hidden_channels, 1)  # NEW

    def forward(
        self,
        x: Tensor,              # [num_nodes, 6] - see below
        edge_index: Tensor,
        iteration: int = 0
    ) -> Dict[str, Tensor]:
        """
        x columns:
          0: current_token_id
          1: prev_token_id (from iteration t-1)
          2: depth
          3: sibling_index
          4: iteration (redundant, but kept for compatibility)
          5: test_signal
        """

        # Embed all features
        token_emb = self.token_embedding(x[:, 0])
        prev_emb = self.prev_token_embedding(x[:, 1])
        pos_emb = self.position_projection(x[:, 2:4])
        iter_emb = self.iteration_embedding(torch.full((x.size(0),), iteration))
        test_emb = self.test_signal_projection(x[:, 5:6])

        # Concatenate: [num_nodes, 128+32+32+32+32 = 256]
        h = torch.cat([token_emb, prev_emb, pos_emb, iter_emb, test_emb], dim=-1)

        # U-Net (unchanged)
        h = self.encoder(h, edge_index)
        h = self.decoder(h, edge_index)

        # Output predictions
        logits = self.token_predictor(h)       # [num_nodes, vocab_size]
        confidence = self.confidence_head(h)   # [num_nodes, 1]

        return {
            'logits': logits,
            'confidence': torch.sigmoid(confidence)
        }
```

#### 3.2 Key Design Decisions

**Q: Why store `prev_token_id`?**
A: Model needs context of "what did I predict last time?" to learn corrections.

**Q: Why separate `confidence_head`?**
A: Enables early stopping. If `confidence.mean() > 0.95`, halt refinement.

**Q: Why embed `test_signal` directly?**
A: Attention should focus on failing nodes. Direct embedding = strong signal.

**Implementation:**

- [MODIFY] [graph_unet.py](file:///home/me/git/prose/src/models/graph_unet.py)
  - Rename `GraphUNet` → `IterativeGraphUNet`
  - Add all feature projections as above
  - Add `confidence_head`
  - Update config to handle `vocab_size`

---

### 4. Training Pipeline

#### 4.1 Trajectory Generation

**Goal:** Create realistic refinement trajectories with mistakes → corrections.

```python
def generate_trajectory(
    clean_program: AST,
    tests: List[TestCase],
    model: Optional[nn.Module] = None,
    max_iterations: int = 5
) -> List[TrajectoryStep]:
    """
    Generate training trajectory by simulating iterative refinement.

    Early training: Use random policy (no model)
    Later training: Use current model to generate realistic mistakes
    """

    trajectory = []

    # 1. Corrupt program (curriculum: 50% → 100%)
    current = corrupt(clean_program, corruption_rate=0.5)

    # 2. Simulate refinement
    for iteration in range(max_iterations):
        # Get model prediction (or random if training just started)
        if model is None or random.random() < 0.3:  # ε-greedy
            prediction = random_prediction(current)
        else:
            prediction = model.predict(current, iteration)

        # Execute tests
        test_results = interpreter.run_tests(prediction, tests)
        execution_trace = interpreter.trace_execution(prediction, tests)

        # Compute test signals
        test_signals = compute_test_signals(prediction, test_results, execution_trace)

        # Store training step
        trajectory.append({
            'input_graph': current,
            'prev_prediction': prediction if iteration > 0 else current,
            'target_graph': clean_program,
            'test_signals': test_signals,
            'iteration': iteration,
            'tests_passing': all(test_results)
        })

        # Update state for next iteration
        current = prediction

        # Early exit if all tests pass
        if all(test_results):
            break

    return trajectory
```

**Implementation:**

- [NEW] [trajectory.py](file:///home/me/git/prose/src/training/trajectory.py)

#### 4.2 Training Loop

```python
class IterativeTrainer:
    def train_step(self, batch_trajectories):
        """
        Train on multiple trajectories simultaneously.
        Each trajectory = [step_0, step_1, ..., step_T]
        """

        total_loss = 0

        for trajectory in batch_trajectories:
            for step in trajectory:
                # Prepare input features
                x = prepare_features(
                    current=step['input_graph'],
                    prev=step['prev_prediction'],
                    test_signals=step['test_signals'],
                    iteration=step['iteration']
                )

                # Forward pass
                output = self.model(
                    x=x,
                    edge_index=step['input_graph'].edge_index,
                    iteration=step['iteration']
                )

                # Compute losses
                loss = self.compute_loss(
                    predictions=output['logits'],
                    targets=step['target_graph'],
                    current=step['input_graph'],
                    confidence=output['confidence']
                )

                total_loss += loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

    def compute_loss(self, predictions, targets, current, confidence):
        """
        Multi-objective loss:
        1. Reconstruction: predict correct tokens
        2. Stability: don't change already-correct nodes
        3. Correction: fix incorrect nodes (higher weight)
        4. Confidence: high confidence on correct, low on incorrect
        """

        # Standard cross-entropy
        recon_loss = F.cross_entropy(predictions, targets.token_ids)

        # Penalize changing correct nodes
        correct_mask = (current.token_ids == targets.token_ids)
        if correct_mask.any():
            stability_loss = F.cross_entropy(
                predictions[correct_mask],
                current.token_ids[correct_mask]
            )
        else:
            stability_loss = 0

        # Reward fixing incorrect nodes (double weight)
        incorrect_mask = ~correct_mask
        if incorrect_mask.any():
            correction_loss = F.cross_entropy(
                predictions[incorrect_mask],
                targets.token_ids[incorrect_mask]
            )
        else:
            correction_loss = 0

        # Confidence calibration
        confidence_loss = F.binary_cross_entropy(
            confidence.squeeze(),
            correct_mask.float()
        )

        return (
            1.0 * recon_loss +
            0.1 * stability_loss +
            0.5 * correction_loss +
            0.2 * confidence_loss
        )
```

**Implementation:**

- [MODIFY] [denoising_task.py](file:///home/me/git/prose/src/training/denoising_task.py)
  - Replace one-shot loss with trajectory-based loss
  - Add stability/correction terms
  - Add confidence loss

---

### 5. Corruption Curriculum

> [!WARNING] > **Critical for 100% corruption**: Model must learn to generate from scratch, not just fill-in.

#### 5.1 Curriculum Stages

| Stage | Epochs | Corruption Rate | Goal                                    |
| ----- | ------ | --------------- | --------------------------------------- |
| 1     | 0-5    | 20%             | Learn basic denoising (sanity check)    |
| 2     | 6-15   | 50%             | Handle high corruption                  |
| 3     | 16-25  | 75%             | Mostly generation                       |
| 4     | 26-40  | 90%             | Nearly full generation                  |
| 5     | 41-50  | 100%            | **Unconditional generation from tests** |

#### 5.2 Corruption Strategies

**20-75% Corruption:**

- Randomly mask nodes
- **Keep structural nodes intact** (first child of DEFINE, etc.)
- Helps model learn from context

**90-100% Corruption:**

- Mask ALL tokens (except structural delimiters)
- **Tests become the ONLY signal**
- Model must reason: "If test says f(5)=6, what program could satisfy this?"

**Implementation:**

```python
def corrupt_program(program: AST, rate: float, keep_structure: bool = True):
    """
    Corrupt program by masking tokens.

    If keep_structure=True and rate < 0.9:
        Keep structural keywords (define, lambda) to preserve skeleton

    If rate >= 0.9:
        Mask everything except parentheses (full generation)
    """
    if rate >= 0.9:
        # Full generation: only keep graph structure
        return mask_all_tokens(program, keep_edges=True)
    else:
        # Partial corruption: random masking with structural preservation
        return random_mask(program, rate=rate, keep_structure=keep_structure)
```

---

### 6. Inference: Iterative Refinement Loop

```python
def generate_program(
    tests: List[TestCase],
    model: IterativeGraphUNet,
    max_iterations: int = 10,
    confidence_threshold: float = 0.95
) -> AST:
    """
    Generate program from tests using iterative refinement.

    Start: Fully masked graph (or random initialization)
    Loop: Refine based on test feedback until tests pass or max iterations
    """

    # 1. Initialize with fully masked graph
    current = create_masked_graph()

    # 2. Iterative refinement
    for iteration in range(max_iterations):
        # Run model
        output = model(
            x=prepare_features(current, iteration=iteration),
            edge_index=current.edge_index,
            iteration=iteration
        )

        # Sample predictions (greedy or sampling)
        predictions = output['logits'].argmax(dim=-1)
        confidence = output['confidence'].mean()

        # Build program from predictions
        program = graph_to_program(predictions)

        # Execute tests
        test_results = interpreter.run_tests(program, tests)

        # Check stopping criteria
        if all(test_results):
            print(f"✓ All tests pass at iteration {iteration}")
            return program

        if confidence > confidence_threshold:
            print(f"✓ High confidence ({confidence:.3f}), stopping")
            return program

        # Compute test feedback for next iteration
        execution_trace = interpreter.trace_execution(program, tests)
        test_signals = compute_test_signals(program, test_results, execution_trace)

        # Update graph with predictions and test signals
        current = update_graph(
            current,
            predictions=predictions,
            test_signals=test_signals
        )

    print(f"✗ Max iterations reached, returning best attempt")
    return program
```

**Implementation:**

- [NEW] [inference.py](file:///home/me/git/prose/src/inference/inference.py)

---

## Verification Plan

### Automated Tests

#### Unit Tests

- [NEW] `tests/test_vocabulary.py`

  - Test tokenization: `program → tokens → program` roundtrip
  - Test unknown token handling

- [NEW] `tests/test_interpreter.py`

  - Test basic evaluation: `(+ 1 2)` → `3`
  - Test lambda: `((lambda (x) (* x 2)) 5)` → `10`
  - Test scoping: let bindings, nested lambdas
  - Test errors: undefined vars, division by zero

- [NEW] `tests/test_execution_trace.py`

  - Test tracing: which nodes executed for given input
  - Test failure localization: mark failing nodes correctly

- [MODIFY] `tests/test_dataset.py`
  - Test trajectory generation
  - Test corruption at different rates (20%, 50%, 100%)

#### Integration Tests

- [NEW] `tests/test_iterative_model.py`
  - Test single refinement step
  - Test multi-step refinement converges
  - Test confidence calibration

### Manual Verification

#### Smoke Tests

1. **Generate tiny dataset** (10 samples)
2. **Train for 1 epoch** on 50% corruption
3. **Verify model outputs** reasonable predictions
4. **Check test feedback** signals propagate correctly

#### Visualization

- [MODIFY] [visualize.py](file:///home/me/git/prose/scripts/visualize.py)
  - Add trajectory visualization: show iterations 0 → T
  - Highlight: nodes changed, test signals, confidence
  - Compare: corrupted → iter1 → iter2 → ... → final

#### Full Training Run

1. **Train on curriculum** (20% → 100% over 50 epochs)
2. **Track metrics:**
   - Accuracy @ iteration 1, 2, 3, 5
   - Test pass rate vs corruption level
   - Confidence calibration (ECE score)
3. **Generate from 100% corruption:**
   - Start with only tests
   - Verify model can synthesize working programs

---

## Migration Path from Phase 1

### Step 1: Add Infrastructure (Non-Breaking)

1. Implement [vocabulary.py](file:///home/me/git/prose/src/data/vocabulary.py)
2. Implement [interpreter.py](file:///home/me/git/prose/src/runtime/interpreter.py)
3. Add tests to synthetic_gen templates
4. **Verify:** Existing code still works

### Step 2: Extend Data Representation

1. Add `token_id` field to ASG nodes (keep `type` for backward compat)
2. Generate new datasets with tokens + tests
3. **Verify:** Can load and visualize new data

### Step 3: Implement Iterative Model

1. Create [IterativeGraphUNet](file:///home/me/git/prose/src/models/graph_unet.py)
2. Add trajectory generation
3. **Verify:** Model runs forward pass on new data

### Step 4: Train Baseline

1. Train on 50% corruption (no curriculum yet)
2. **Verify:** Model learns to denoise better than random

### Step 5: Add Curriculum

1. Implement corruption scheduler
2. Train 20% → 100%
3. **Verify:** Model handles 100% corruption

### Step 6: Optimize Inference

1. Implement iterative refinement loop
2. Add early stopping
3. **Verify:** Can generate programs from tests alone

---

## Success Criteria

### Minimum Viable (Phase 1.5 Complete)

- ✅ Model predicts **token values**, not just types
- ✅ Training includes **test feedback** signals
- ✅ Model performs **iterative refinement** (2-5 iterations)
- ✅ Handles **50% corruption** with >70% test pass rate

### Stretch Goals

- ✅ Handles **100% corruption** (generation from tests)
- ✅ Test pass rate >50% on fully generated programs
- ✅ Confidence calibration ECE < 0.1
- ✅ Inference time <1s per program (5 iterations)

### Failure Modes to Watch

- ❌ **Mode collapse**: Model always outputs same program regardless of tests
- ❌ **Test overfitting**: Model memorizes test I/O instead of learning logic
- ❌ **Thrashing**: Model changes same nodes repeatedly without convergence
- ❌ **Slow interpreter**: Test execution becomes bottleneck

---

## Implementation Progress

### ✅ Phase A: Infrastructure (COMPLETE)

**Commit:** `c40f217` - "feat[_all]: phase 1.5 preparation"

#### Completed Components

1. **Vocabulary System** [`src/data/vocabulary.py`](file:///home/me/git/prose/src/data/vocabulary.py)

   - ✅ Token extraction from AST
   - ✅ Bidirectional token ↔ ID mapping
   - ✅ Special tokens (MASK, PAD, UNK)
   - ✅ Save/load functionality
   - ✅ Tests: 8/8 passing

2. **Mini-Lisp Interpreter** [`src/runtime/interpreter.py`](file:///home/me/git/prose/src/runtime/interpreter.py)

   - ✅ Full S-expression evaluation
   - ✅ Primitives: +, -, \*, /, <, >, =, if, let, lambda, define
   - ✅ Lexical scoping with environments
   - ✅ Recursion support (factorial, fibonacci tested)
   - ✅ Execution tracing for test feedback
   - ✅ Test execution interface
   - ✅ Error handling (division by zero, undefined vars, type errors)
   - ✅ Tests: 24/24 passing

3. **Test Generation** [`src/data/test_generator.py`](file:///home/me/git/prose/src/data/test_generator.py)

   - ✅ Automatic test case generation from programs
   - ✅ Manual test specifications
   - ✅ Variable extraction utilities

4. **Template Extensions** [`src/data/synthetic_gen.py`](file:///home/me/git/prose/src/data/synthetic_gen.py)

   - ✅ Added `tests` field to `ProgramMetadata`
   - ✅ Added `generate_tests()` base method

5. **Utilities**
   - ✅ [`scripts/build_vocabulary.py`](file:///home/me/git/prose/scripts/build_vocabulary.py) - Build vocab from templates
   - ✅ [`scripts/demo_phase1_5_infrastructure.py`](file:///home/me/git/prose/scripts/demo_phase1_5_infrastructure.py) - Demo all components

**Verification:**

- ✅ All unit tests passing (32/32 total)
- ✅ Demo script successful (vocabulary roundtrip, interpreter recursion, execution tracing)
- ✅ Ready for Phase B

---

### 🚧 Phase B: Data Representation (NEXT)

**Goal:** Extend node features and dataset schema for iterative refinement

#### Tasks

1. **Extend ASG Node Features** [`src/data/asg_builder.py`](file:///home/me/git/prose/src/data/asg_builder.py)

   - [ ] Add `token_id` field (from vocabulary)
   - [ ] Add `prev_token_id` field (for iteration t-1)
   - [ ] Add `iteration` field (current refinement pass)
   - [ ] Add `test_signal` field (failure feedback)
   - [ ] Keep `type` field for backward compatibility
   - [ ] Update `_to_pyg_data()` to include all 6 features

2. **Update Dataset Schema** [`src/data/dataset.py`](file:///home/me/git/prose/src/data/dataset.py)

   - [ ] Add test cases to dataset samples
   - [ ] Add execution trace storage
   - [ ] Update collate function for new features
   - [ ] Support variable-size programs in batches

3. **Generate Initial Dataset with Tests**
   - [ ] Modify data generation script to include tests
   - [ ] Generate small pilot dataset (100 samples)
   - [ ] Verify data loading and visualization

**Estimated effort:** 1-2 hours

---

### ✅ Phase C: Model Architecture (COMPLETE)

**Commit:** `bd8f473` - "feat(model): implement IterativeGraphUNet for Phase 1.5"

#### Completed Components

1. **IterativeGraphUNet** [`src/models/graph_unet.py`](file:///home/me/git/prose/src/models/graph_unet.py)

   - ✅ Implemented alongside legacy GraphUNet (backward compatible)
   - ✅ Token embedding (vocab_size → 128)
   - ✅ Prev token embedding (vocab_size → 32)
   - ✅ Position projection (2 → 32)
   - ✅ Iteration embedding (max_iterations → 32)
   - ✅ Test signal projection (1 → 32)
   - ✅ Confidence head output (sigmoid activation)
   - ✅ forward() with pooling and forward_full() without pooling

2. **Unit Tests** [`tests/test_iterative_model.py`](file:///home/me/git/prose/tests/test_iterative_model.py)
   - ✅ 10 tests covering all functionality
   - ✅ Output shape verification
   - ✅ Confidence range validation
   - ✅ Iteration conditioning
   - ✅ Test signal influence
   - ✅ Batch processing
   - ✅ Gradient flow
   - ✅ Edge cases (single node, large vocab)

**Architecture Summary:**
- Input: 256D (128 + 32 + 32 + 32 + 32)
- Parameters: ~702K (vocab=95, hidden=256, depth=3)
- Output: Logits [num_nodes, vocab_size] + Confidence [num_nodes, 1]

**Verification:**
- ✅ All 10 unit tests passing
- ✅ Tested with real Phase 1.5 data (pilot dataset)
- ✅ Gradients flow correctly

---

### 🔲 Phase D: Training Pipeline

**Goal:** Trajectory-based training with test feedback

#### Tasks

1. **Trajectory Generation** [`src/training/trajectory.py`](file:///home/me/git/prose/src/training/trajectory.py)

   - [ ] Implement corruption strategies (20%-100%)
   - [ ] Generate refinement trajectories
   - [ ] Integrate test execution feedback
   - [ ] Support curriculum learning
   - [ ] Model-based vs random sampling (ε-greedy)

2. **Multi-Objective Loss** [`src/training/denoising_task.py`](file:///home/me/git/prose/src/training/denoising_task.py)

   - [ ] Reconstruction loss (cross-entropy)
   - [ ] Stability loss (don't change correct nodes)
   - [ ] Correction loss (fix incorrect nodes)
   - [ ] Confidence calibration loss
   - [ ] Loss weighting (1.0, 0.1, 0.5, 0.2)

3. **Curriculum Scheduler**
   - [ ] Stage 1: 20% corruption (epochs 0-5)
   - [ ] Stage 2: 50% corruption (epochs 6-15)
   - [ ] Stage 3: 75% corruption (epochs 16-25)
   - [ ] Stage 4: 90% corruption (epochs 26-40)
   - [ ] Stage 5: 100% corruption (epochs 41-50)

**Estimated effort:** 4-6 hours

---

### 🔲 Phase E: Inference System

**Goal:** Iterative refinement loop for test-driven generation

#### Tasks

1. **Refinement Loop** [`src/inference/inference.py`](file:///home/me/git/prose/src/inference/inference.py)
   - [ ] Initialize with fully masked graph
   - [ ] Iterative refinement (max 10 iterations)
   - [ ] Test execution and feedback
   - [ ] Early stopping (tests pass OR high confidence)
   - [ ] Update graph with predictions + test signals

**Estimated effort:** 1-2 hours

---

### 🔲 Phase F: Evaluation & Visualization

**Goal:** Comprehensive metrics and trajectory visualization

#### Tasks

1. **Extend Metrics** [`src/eval/denoising_metrics.py`](file:///home/me/git/prose/src/eval/denoising_metrics.py)

   - [ ] Test pass rate metric
   - [ ] Iteration convergence tracking
   - [ ] Confidence calibration (ECE)
   - [ ] Per-iteration accuracy

2. **Trajectory Visualization** [`scripts/visualize.py`](file:///home/me/git/prose/scripts/visualize.py)
   - [ ] Show iteration progression (0 → T)
   - [ ] Highlight changed nodes
   - [ ] Display test signals
   - [ ] Show confidence evolution

**Estimated effort:** 2-3 hours

---

## Next Session Tasks

**Priority 1: Data Representation (Phase B)**
Start here in the next session:

1. Update `asg_builder.py` to add 6-feature node representation
2. Modify dataset to include test cases
3. Generate small pilot dataset (100 samples with tests)
4. Verify data loading

**Priority 2: Model Architecture (Phase C)**
Once data is ready:

1. Implement `IterativeGraphUNet`
2. Test forward pass
3. Run smoke test (1 epoch)

**Reference Files:**

- Implementation plan: [`implementation_plan.md`](file:///home/me/.gemini/antigravity/brain/30bb95f6-82bf-4bef-b0b5-ceb6a978a3ca/implementation_plan.md)
- Task checklist: [`task.md`](file:///home/me/.gemini/antigravity/brain/30bb95f6-82bf-4bef-b0b5-ceb6a978a3ca/task.md)
- Infrastructure walkthrough: [`walkthrough.md`](file:///home/me/.gemini/antigravity/brain/30bb95f6-82bf-4bef-b0b5-ceb6a978a3ca/walkthrough.md)

---

## Next Steps (Original)

1. **Review this plan** - Identify any gaps or concerns
2. **Implement Step 1** (Infrastructure) - Start with interpreter + vocabulary
3. **Generate pilot dataset** (100 samples with tests)
4. **Prototype iterative model** - Test single refinement step
5. **Run smoke test** - 1 epoch on 50% corruption
6. **Iterate** based on results

---

## Open Questions

### Architecture

1. Should we use **residual connections** between iterations? (Model predicts Δ instead of full state)
2. Should confidence be **per-node** or **global**?
3. How to handle **variable-size programs** in batching?

### Training

1. What's the optimal **trajectory length** distribution? (Always 5 steps? Variable?)
2. Should we use **teacher forcing** (feed ground truth at iteration t) or **free running**?
3. How to balance **exploration** (model makes mistakes) vs **efficiency** (model learns from good trajectories)?

### Data

1. How many **tests per program**? (3? 5? 10?)
2. Should tests be **held out** during training to test generalization?
3. How to handle **non-deterministic** program behavior (if we ever add randomness)?

### Interpreter

1. Python perf will suck - should we **profile** and optimize hot paths?
2. Should we **cache** execution results (memoization)?
3. How to handle **infinite loops** or **timeouts**?

> [!NOTE] > **Attention Mechanisms**: For discussion on why we don't use explicit Transformer-style attention (and how ASG structure provides implicit attention via DATAFLOW edges), see [`architecture.md`](architecture.md#attention-mechanisms-implicit-vs-explicit).

---

# 'Hybrid' numeric representation

## Representation Option: Hybrid - Types + Continuous Values

Separate **type** (discrete) from **value** (continuous):

```python
Node = {
    'type': Categorical,     # OPERATOR, SYMBOL, NUMBER, ...
    'operator': Categorical, # +, -, *, / (if type=OPERATOR)
    'symbol': Categorical,   # x, y, foo (if type=SYMBOL)
    'number': Continuous     # any float (if type=NUMBER)
}
```

**Implementation:**

```python
# Model outputs:
{
    'type_logits': [num_nodes, 8],       # SYMBOL, NUMBER, OPERATOR, etc.
    'operator_logits': [num_nodes, 7],   # +, -, *, /, <, >, =
    'symbol_logits': [num_nodes, 50],    # x, y, foo, bar, ...
    'number_value': [num_nodes, 1]       # regression head (MSE loss)
}

# At inference:
if predicted_type == NUMBER:
    return predicted_number_value  # e.g., 3.14
elif predicted_type == OPERATOR:
    return argmax(operator_logits)  # e.g., '+'
elif predicted_type == SYMBOL:
    return argmax(symbol_logits)    # e.g., 'x'
```

**Pros:**

- Handles arbitrary numbers/floats
- Clean separation of concerns
- Vocabulary stays small

**Cons:**

- More complex architecture (multiple heads)
- Need MSE loss for numbers + CE for categoricals
- Might be harder to train?

## Why?

1. **Scalable** - Handles arbitrary numbers/strings
2. **Aligns with Project 2 EBM** - Energy function can reason about numeric constraints
3. **Interpretable** - Clear separation: "This is a NUMBER with value 5"
4. **Proven** - Similar to how VAEs/diffusion models handle mixed data types

### Implementation Sketch:

```python
class IterativeGraphUNet(nn.Module):
    def __init__(self, ...):
        # Type-specific heads
        self.type_head = nn.Linear(hidden, 8)         # Node type
        self.operator_head = nn.Linear(hidden, 7)     # Which operator
        self.symbol_head = nn.Linear(hidden, 50)      # Which symbol
        self.number_head = nn.Linear(hidden, 1)       # Numeric value

    def forward(self, x, edge_index):
        h = self.graph_layers(x, edge_index)

        return {
            'type': self.type_head(h),
            'operator': self.operator_head(h),
            'symbol': self.symbol_head(h),
            'number': self.number_head(h)
        }
```

**Loss function:**

```python
# Mask by type
type_pred = output['type'].argmax(-1)

# Loss for operators (only where type == OPERATOR)
operator_mask = (target_type == OPERATOR)
operator_loss = CE(output['operator'][operator_mask], target_operator[operator_mask])

# Loss for numbers (only where type == NUMBER)
number_mask = (target_type == NUMBER)
number_loss = MSE(output['number'][number_mask], target_number[number_mask])

# Total
loss = type_loss + operator_loss + symbol_loss + number_loss
```

# Soft representations

## 1. Latent (Soft) Representations During Refinement

Instead of collapsing to discrete tokens immediately, keep **soft distributions**:

```python
# Current (bad): Discrete at every step
iteration_0: node.token = MASK (one-hot)
iteration_1: node.token = '+' (one-hot) ← Hard commitment
iteration_2: stuck with '+', can only switch discretely

# New (good): Continuous until final step
iteration_0: node.token_dist = [0.0, 0.0, ..., 1.0_MASK]  # One-hot MASK
iteration_1: node.token_dist = [0.6_'+', 0.3_'-', 0.1_'*', ...]  # Soft!
iteration_2: node.token_dist = [0.9_'+', 0.08_'-', 0.02_'*', ...] # Refining
iteration_5: node.token_dist = [0.99_'+', 0.01_'-', ...] # Near-certain
final:      node.token = argmax(...) = '+' ← Discretize only at end
```

### Why This Matters:

**Enables gradient-based refinement:**

```python
# Can compute gradients through soft assignments
energy = compute_energy(soft_graph)
gradient = ∇_soft_graph energy
soft_graph_next = soft_graph - α * gradient  # Continuous optimization!
```

**Represents uncertainty:**

- "I'm 60% sure this is `+`, but might be `-`"
- Test feedback can shift distribution without discrete jumps

### Implementation:

```python
class SoftGraphState:
    """
    Continuous graph representation for iterative refinement.
    """
    def __init__(self, num_nodes, vocab_size):
        # Soft assignments (probabilities, not one-hot)
        self.type_dist = torch.zeros(num_nodes, 8)          # SYMBOL, NUMBER, etc.
        self.operator_dist = torch.zeros(num_nodes, 7)      # +, -, *, /, ...
        self.symbol_dist = torch.zeros(num_nodes, 50)       # x, y, foo, ...
        self.number_value = torch.zeros(num_nodes, 1)       # Continuous anyway

        # Softmax-normalized distributions
        self.type_dist = F.softmax(self.type_dist, dim=-1)
        # ... etc

    def to_discrete(self):
        """Collapse to discrete program (for execution)."""
        discrete_type = self.type_dist.argmax(dim=-1)
        discrete_operator = self.operator_dist.argmax(dim=-1)
        # ...
        return DiscreteProgram(discrete_type, discrete_operator, ...)
```

Model operates on **soft states**:

```python
def forward(self, soft_state, edge_index, iteration):
    # Input: Current soft distributions
    # Output: Updated soft distributions

    # Embed soft distributions (weighted sum)
    type_emb = soft_state.type_dist @ self.type_embedding.weight
    operator_emb = soft_state.operator_dist @ self.operator_embedding.weight
    symbol_emb = soft_state.symbol_dist @ self.symbol_embedding.weight

    h = torch.cat([type_emb, operator_emb, symbol_emb, soft_state.number_value], dim=-1)

    # Process through U-Net
    h = self.graph_layers(h, edge_index)

    # Output: New soft distributions
    return SoftGraphState(
        type_dist=F.softmax(self.type_head(h), dim=-1),
        operator_dist=F.softmax(self.operator_head(h), dim=-1),
        # ...
    )
```

---

## 2. Extra State Space (Reasoning Scratchpad)

Add **auxiliary nodes** that don't correspond to AST nodes but provide "working memory":

```python
class GraphWithScratchpad:
    def __init__(self, program_nodes, num_scratch_nodes=10):
        # Real AST nodes
        self.program_nodes = program_nodes  # [N_program, features]

        # Virtual "thought" nodes (not in AST)
        self.scratch_nodes = torch.zeros(num_scratch_nodes, hidden_dim)

        # Combined graph
        self.all_nodes = torch.cat([program_nodes, scratch_nodes], dim=0)

        # Edges: AST edges + connections to scratch
        self.edge_index = build_edges_with_scratch(program_nodes, scratch_nodes)
```

### What Scratch Nodes Can Represent:

1. **Intermediate computations**: "If I choose `+` here, the result would be..."
2. **Constraint tracking**: "Test 1 requires output > 0, so operator must be..."
3. **Attention anchors**: Nodes that aggregate global context
4. **Planning tokens**: "Step 1: fix operators, Step 2: fix variables"

### Architecture:

```python
class ScratchpadGraphUNet(nn.Module):
    def __init__(self, num_scratch=10, ...):
        self.scratch_embedding = nn.Parameter(
            torch.randn(num_scratch, hidden_dim)
        )

    def forward(self, program_graph, iteration):
        # 1. Add scratch nodes to graph
        num_program = program_graph.num_nodes
        scratch = self.scratch_embedding.expand(program_graph.batch_size, -1, -1)

        # 2. Build augmented graph
        augmented_nodes = torch.cat([program_graph.x, scratch], dim=1)
        augmented_edges = add_scratch_edges(
            program_graph.edge_index,
            num_program=num_program,
            num_scratch=len(scratch)
        )

        # 3. Process with message passing
        h = self.graph_layers(augmented_nodes, augmented_edges)

        # 4. Split back
        program_output = h[:, :num_program]
        scratch_output = h[:, num_program:]

        # 5. Predictions only on program nodes
        return {
            'program_predictions': self.prediction_heads(program_output),
            'scratch_state': scratch_output  # Carries over to next iteration
        }
```

### Carry Scratch State Across Iterations:

```python
def iterative_refinement(program, tests, max_iter=5):
    soft_state = initialize_soft_graph(program)
    scratch_state = None  # Initialize

    for t in range(max_iter):
        # Model takes current state + scratch from last iteration
        output = model(
            soft_state=soft_state,
            scratch_state=scratch_state,
            iteration=t
        )

        # Update states
        soft_state = output['soft_state']
        scratch_state = output['scratch_state']  # Persist to next iteration

        # Discretize for test execution
        discrete_program = soft_state.to_discrete()
        test_results = run_tests(discrete_program, tests)

        if all_pass(test_results):
            break

    return discrete_program
```

---

## 3. Combined Architecture Sketch

```python
class IterativeReasoningUNet(nn.Module):
    """
    Graph U-Net with:
    - Soft (continuous) node representations
    - Scratchpad nodes for reasoning
    - Hybrid output types (categorical + continuous)
    """

    def __init__(
        self,
        num_node_types=8,
        num_operators=7,
        num_symbols=50,
        num_scratch_nodes=10,
        hidden_dim=256,
        depth=3
    ):
        # Embedding tables
        self.type_embedding = nn.Embedding(num_node_types, 64)
        self.operator_embedding = nn.Embedding(num_operators, 64)
        self.symbol_embedding = nn.Embedding(num_symbols, 64)

        # Scratchpad (learnable)
        self.scratch_init = nn.Parameter(torch.randn(num_scratch_nodes, hidden_dim))

        # Graph layers
        self.encoder = GraphEncoder(hidden_dim, depth)
        self.decoder = GraphDecoder(hidden_dim, depth)

        # Output heads
        self.type_head = nn.Linear(hidden_dim, num_node_types)
        self.operator_head = nn.Linear(hidden_dim, num_operators)
        self.symbol_head = nn.Linear(hidden_dim, num_symbols)
        self.number_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        soft_state: SoftGraphState,
        edge_index: Tensor,
        scratch_state: Optional[Tensor] = None,
        iteration: int = 0
    ):
        # 1. Embed soft distributions (continuous!)
        type_emb = soft_state.type_dist @ self.type_embedding.weight
        operator_emb = soft_state.operator_dist @ self.operator_embedding.weight
        symbol_emb = soft_state.symbol_dist @ self.symbol_embedding.weight

        program_features = torch.cat([
            type_emb,
            operator_emb,
            symbol_emb,
            soft_state.number_value,
            soft_state.position_features,
            soft_state.test_signals
        ], dim=-1)

        # 2. Add scratchpad
        if scratch_state is None:
            scratch = self.scratch_init.unsqueeze(0).expand(
                soft_state.batch_size, -1, -1
            )
        else:
            scratch = scratch_state

        # 3. Augment graph
        all_features = torch.cat([program_features, scratch], dim=1)
        augmented_edges = self._add_scratch_edges(edge_index, program_features.size(1))

        # 4. Graph U-Net
        h = self.encoder(all_features, augmented_edges)
        h = self.decoder(h, augmented_edges)

        # 5. Split outputs
        num_program = program_features.size(1)
        program_h = h[:, :num_program]
        scratch_h = h[:, num_program:]

        # 6. Predict (soft distributions!)
        new_soft_state = SoftGraphState(
            type_dist=F.softmax(self.type_head(program_h), dim=-1),
            operator_dist=F.softmax(self.operator_head(program_h), dim=-1),
            symbol_dist=F.softmax(self.symbol_head(program_h), dim=-1),
            number_value=self.number_head(program_h),
            confidence=torch.sigmoid(self.confidence_head(program_h))
        )

        return {
            'soft_state': new_soft_state,
            'scratch_state': scratch_h  # Pass to next iteration
        }
```

---

## Key Benefits:

1. **✅ Soft representations** → Can do gradient-based optimization (EBM!)
2. **✅ Scratchpad** → Model has "thinking space" beyond the AST
3. **✅ Hybrid types** → Numbers are continuous, operators are categorical
4. **✅ Bridge to EBM** → Energy minimization needs continuous space
