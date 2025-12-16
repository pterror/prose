# Phase 1.5: Next Steps for Implementation

## Quick Start (Next Session)

```bash
# 1. Verify infrastructure still works
pytest tests/test_vocabulary.py tests/test_interpreter.py

# 2. Start with data representation
# Edit: src/data/asg_builder.py
# - Add token_id, prev_token_id, iteration, test_signal fields

# 3. Update dataset
# Edit: src/data/dataset.py
# - Add tests to samples

# 4. Generate pilot data
python scripts/01_generate_data.py --num-samples 100 --with-tests

# 5. Implement iterative model
# Edit: src/models/graph_unet.py
# - Create IterativeGraphUNet class
```

---

## Progress Summary

### ✅ Completed (This Session)

**Infrastructure - All tests passing (32/32)**

1. **Vocabulary System** - Token-level encoding (~500 vocab size)

   - Files: `src/data/vocabulary.py`, `tests/test_vocabulary.py`
   - Tests: 8/8 passing

2. **Mini-Lisp Interpreter** - Full evaluation + recursion + tracing

   - Files: `src/runtime/interpreter.py`, `tests/test_interpreter.py`
   - Tests: 24/24 passing

3. **Test Generation** - Automatic test case creation

   - Files: `src/data/test_generator.py`, `src/data/synthetic_gen.py`

4. **Demo Scripts** - Complete infrastructure demonstration
   - Files: `scripts/build_vocabulary.py`, `scripts/demo_phase1_5_infrastructure.py`

**Git Commit:** `c40f217` - "feat[_all]: phase 1.5 preparation"

---

## 🎯 Next Priority: Data Representation

### Task 1: Extend Node Features (1-2 hours)

**File:** `src/data/asg_builder.py`

**Current node features** (3 features):

```python
[node_type, depth, sibling_index]
```

**Target node features** (6 features):

```python
[token_id, prev_token_id, depth, sibling_index, iteration, test_signal]
```

**Changes needed:**

1. Add `token_id` field from vocabulary
2. Add `prev_token_id` (default to MASK initially)
3. Add `iteration` field (0-4)
4. Add `test_signal` field (0.0 or 1.0)
5. Keep `type` for debugging/backward compat
6. Update `_to_pyg_data()` to output 6 features

### Task 2: Update Dataset Schema (1 hour)

**File:** `src/data/dataset.py`

**Add to dataset samples:**

```python
{
    'graph': Data,              # PyG graph (now with 6 features)
    'tests': List[TestCase],    # 3-5 tests per program
    'target_tokens': Tensor,    # Ground truth token IDs
    'metadata': ProgramMetadata # Includes template info
}
```

### Task 3: Generate Pilot Dataset (30 min)

**Command:**

```bash
python scripts/01_generate_data.py \
    --num-samples 100 \
    --with-tests \
    --output data/pilot_phase1.5/
```

**Verify:**

- Load sample and print features
- Check test cases are included
- Visualize one trajectory

---

## 🏗️ Following Steps: Model & Training

### Step 1: Iterative Model (2-3 hours)

**File:** `src/models/graph_unet.py`

Create `IterativeGraphUNet` with:

- Embeddings for all 6 features
- Confidence head output
- Iteration conditioning

### Step 2: Trajectory Generation (2-3 hours)

**File:** `src/training/trajectory.py`

Implement:

- Corruption strategies (20%-100%)
- Test execution feedback
- Refinement simulation

### Step 3: Training Loop (2-3 hours)

**File:** `src/training/denoising_task.py`

Multi-objective loss:

- Reconstruction (predict correct tokens)
- Stability (don't change correct nodes)
- Correction (fix incorrect nodes)
- Confidence calibration

### Step 4: Smoke Test (30 min)

Train for 1 epoch on pilot data to verify pipeline.

---

## 📚 Reference Documentation

All in: `/home/me/.gemini/antigravity/brain/30bb95f6-82bf-4bef-b0b5-ceb6a978a3ca/`

1. **implementation_plan.md** - Detailed architecture and design decisions
2. **task.md** - Comprehensive checklist of all tasks
3. **walkthrough.md** - Infrastructure verification results

**Main spec:** `docs/phase1.5.md`

---

## ⏱️ Time Estimates

| Phase | Component           | Estimated Time | Priority |
| ----- | ------------------- | -------------- | -------- |
| B     | Data Representation | 2-3 hours      | **P0**   |
| C     | Model Architecture  | 2-3 hours      | **P0**   |
| D     | Training Pipeline   | 4-6 hours      | **P1**   |
| E     | Inference System    | 1-2 hours      | **P1**   |
| F     | Evaluation/Viz      | 2-3 hours      | **P2**   |

**Total remaining:** ~12-17 hours

---

## 🚀 Start Next Session With

```bash
# 1. Check current state
git status
git log --oneline -5

# 2. Run existing tests
pytest tests/test_vocabulary.py tests/test_interpreter.py -v

# 3. Open key files
code src/data/asg_builder.py
code src/data/dataset.py

# 4. Review plan
cat docs/phase1.5.md
```

**First code change:** Add `token_id` field to `asg_builder.py`
