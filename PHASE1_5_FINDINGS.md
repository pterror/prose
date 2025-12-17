# Phase 1.5 Iterative Refinement: Findings & Next Steps

## Executive Summary

**Goal:** Train model to iteratively refine programs using test execution feedback
**Current Status:** Model achieves 67% one-shot accuracy but **does not perform iterative refinement**
**Root Cause:** Distribution mismatch between training and inference

---

## What We Discovered

### The Problem
Model makes predictions at iteration 0, then **never changes them** (changed=0 nodes) in iterations 1-4, even with test feedback available.

### Root Cause Analysis

1. **Training uses random policy** (`trajectory.py:106-119`):
   - Trajectories generated with random token updates
   - Model sees randomly corrupted graphs at each iteration

2. **Inference uses model's predictions**:
   - Model sees its own output from previous iteration
   - **Distribution mismatch!**

3. **Model never learned to refine its own predictions** because it never saw them during training.

---

## Approaches Attempted

### ✅ Phase 0: Baseline
- **Approach:** Train with test signals (zeros) + ground truth targets
- **Result:** 67.24% validation accuracy
- **Issue:** Model ignores test signals, makes one prediction and stops

### ❌ Phase 1: Test-Following Auxiliary Loss
- **Approach:** Add loss penalizing unchanged predictions on failing nodes
- **Loss:** `test_signals * (prediction == current_token)`
- **Result:** **25.37% validation accuracy** (catastrophic failure)
- **Conclusion:** Weak signal confuses the model

### ✅ Phase 2: Test-Guided Target Supervision
- **Approach:** Weight reconstruction loss by test signals (10x for failing nodes)
- **Loss:** `per_node_loss * (1.0 + test_signals * 9.0)`
- **Result:** **67.24% validation accuracy** (matches baseline)
- **Issue:** Still doesn't iterate - same distribution mismatch problem

---

## Key Insights

### Why Phase 2 Maintained Performance
- Test-guided weighting works: model learns to focus on failing nodes
- But only for **one-shot** prediction
- Doesn't solve iterative refinement because of distribution mismatch

### The Distribution Mismatch
```
Training:
  Iter 0: [MASK, MASK, +, MASK]  # Initial corruption
  Iter 1: [if, -, +, 3]          # Random updates
  Iter 2: [if, map, +, b]        # Random updates

Inference:
  Iter 0: [MASK, MASK, +, MASK]  # Initial corruption
  Iter 1: [if, -, +, 3]          # Model's prediction from iter 0
  Iter 2: [if, -, +, 3]          # SAME (model doesn't change anything!)
```

Model learned: "At iteration N, given random corruption X, predict Y"
Model did NOT learn: "At iteration N, given my previous prediction X, predict improvement Y"

---

## Test Results

### Iterative Refinement Performance (Phase 2)

**50% Corruption:**
- Perfect reconstructions: 0/5
- Tests passing: 0/5
- Improved from initial: 3/5
- **Issue:** changed=0 nodes after iteration 0

**75% Corruption:**
- Perfect reconstructions: 0/5
- Tests passing: 0/5
- Improved from initial: 5/5
- **Issue:** changed=0 nodes after iteration 0

**100% Corruption:**
- Perfect reconstructions: 0/5
- Tests passing: 0/5
- Improved from initial: 5/5
- **Issue:** changed=0 nodes after iteration 0

### Confidence Analysis
- Iteration 0: Low confidence (0.009-0.593)
- Iteration 1: **HIGH confidence** (0.96-0.99)
- Model learned spurious correlation: "If iteration > 0, be confident"

---

## Next Steps (Recommended)

### Option 1: Scheduled Sampling (Best)
**Fix the distribution mismatch by using model predictions during training**

```python
# In trajectory generation
def generate_trajectory(self, ...):
    for iteration in range(max_iterations):
        # Compute test signals
        test_signals = self._compute_test_signals(current_graph, tests)

        # Store trajectory step
        trajectory.append(...)

        # Use MODEL predictions (not random!) with scheduled sampling
        use_model_prob = min(0.9, iteration / max_iterations + 0.5)
        if model and random() < use_model_prob:
            # Use model's prediction
            output = model.forward_full(current_graph, iteration)
            predictions = output['logits'].argmax(dim=-1)

            current_graph.x[:, 1] = current_graph.x[:, 0]  # prev_token
            current_graph.x[:, 0] = predictions.float()    # new token
        else:
            # Use ground truth (for stability)
            current_graph.x[:, 1] = current_graph.x[:, 0]
            current_graph.x[:, 0] = target_graph.x[:, 0]
```

**Pros:**
- Fixes distribution mismatch
- Model learns to refine its own predictions
- Gradual transition (stable training)

**Cons:**
- Slower training (need model forward pass per iteration)
- Requires careful tuning of sampling schedule

### Option 2: Imitation Learning
**Train on expert demonstrations (ground truth edits)**

Instead of random/model trajectories, create "expert" trajectories:
```python
# Show model the shortest path from corrupted → clean
def expert_trajectory(corrupted, clean):
    current = corrupted.clone()
    for iteration in range(max_iterations):
        # Edit distance: which nodes need changing?
        need_fix = (current.x[:, 0] != clean.x[:, 0])

        # Fix some nodes (gradually approach target)
        fix_fraction = 0.5
        nodes_to_fix = sample(need_fix, fraction=fix_fraction)
        current.x[nodes_to_fix, 0] = clean.x[nodes_to_fix, 0]

        yield (current, clean, iteration)
```

**Pros:**
- Clear supervision signal
- Stable training

**Cons:**
- Doesn't match inference (expert always knows answer)
- May not generalize to test-driven refinement

### Option 3: Reinforce/Policy Gradient
**Treat as RL problem with test passage as reward**

**Pros:**
- Directly optimizes for test passage
- No distribution mismatch

**Cons:**
- Very hard to train (sparse rewards)
- High variance
- Slow convergence

---

## Recommendation

**Implement Option 1: Scheduled Sampling** with Phase 2's test-guided weighting.

### Implementation Plan

1. **Modify `TrajectoryGenerator.generate_trajectory()`**:
   - Accept model as parameter
   - Use model predictions with scheduled sampling
   - Keep test signals computation

2. **Update training loop**:
   - Pass model to trajectory generator
   - Use scheduled sampling probability

3. **Key insight**: Combine Phase 2 (test-guided loss) + Scheduled Sampling (fix distribution)

4. **Expected outcome**:
   - Model sees its own predictions during training
   - Learns to refine based on test feedback
   - Maintains one-shot performance (67%)
   - Gains iterative refinement capability

---

## Timeline Estimate

**Scheduled Sampling Implementation:** 2-3 hours
- Modify trajectory generation: 1 hour
- Update training loop: 30 min
- Testing & debugging: 1-1.5 hours

**Training:** 3-4 minutes (50 epochs)

**Total:** ~2-3 hours to solution

---

## Success Metrics

### Minimum Viable
- [ ] Model changes predictions across iterations (changed > 0)
- [ ] Accuracy improves with iterations (iter 4 > iter 0)
- [ ] One-shot accuracy ≥ 60% (preserve baseline)

### Target Goals
- [ ] At least 1/10 programs pass all tests after refinement
- [ ] Average +10-15% accuracy improvement over 5 iterations
- [ ] One-shot accuracy ≥ 65%

---

## Files Modified

- `src/training/trajectory.py` - Added real test execution
- `src/training/denoising_task.py` - Phase 1 & 2 loss modifications
- `src/inference/inference.py` - Real test execution + better stopping
- `scripts/train_phase1_5.py` - Integration of new losses

## Key Commits

- `1da8ec3` - Phase 1: Test-following auxiliary loss
- `e669b35` - Phase 2: Test-guided target supervision
