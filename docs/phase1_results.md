# Phase 1 Training Results Summary

## Training Configuration

- **Model**: Graph U-Net (0.63M parameters, 3-layer GAT)
- **Dataset**: 2,000 train / 500 val / 500 test samples
- **Epochs**: 50
- **Best Validation Accuracy**: 86.39%
- **Best Validation Loss**: 0.2930

## Performance Analysis

### Key Metrics

- **Node Accuracy**: 86.39% (model correctly predicts 8.6 out of 10 nodes on average)
- **Masked Node Reconstruction**: ~30% (model correctly predicts 30% of corrupted/masked nodes)
- **Uncorrupted Node Preservation**: ~99% (model almost always preserves nodes that weren't masked)

### Error Patterns

The model exhibits a **systematic confusion pattern**:

1. **OPERATOR ↔ SYMBOL Confusion** (dominant error)

   - Model frequently predicts `SYMBOL` when the true type is `OPERATOR`
   - Occurs at specific positions (3 and 7) in most graphs

2. **Positional Overfitting**
   - Limited template diversity (~400-500 unique templates × 4-6 duplication)
   - Model learns position-based patterns rather than semantic understanding
   - Positions 3 & 7 are consistently mispredicted

### Root Causes

1. **Dataset Artifacts**: Template repetition creates positional biases
2. **Model Capacity**: 3 layers may be insufficient for long-range context
3. **Type Ambiguity**: OPERATOR and SYMBOL are structurally similar (both leaf nodes)
4. **Limited Diversity**: Need more varied synthetic programs

## Training Infrastructure

✅ **Successfully Implemented:**

- GPU memory profiling
- Learning rate warmup (5 epochs, linear 0.0001 → 0.001)
- Cosine annealing schedule (epochs 6-50)
- Gradient norm tracking
- Checkpoint saving (every 5 epochs)
- TensorBoard logging

⚠️ **Known Issue:**

- Training ran on **CPU instead of GPU** (PyTorch has CUDA support but CUDA unavailable on system)

## Next Steps

### Priority 1: Address Error Patterns

- [ ] **Expand templates** - Reduce positional bias
- [ ] **Add position encodings** - Distinguish positional vs semantic patterns
- [ ] **Increase model depth** - Try 5-7 layers for better context

### Priority 2: Infrastructure

- [ ] **Fix CUDA setup** - Enable GPU acceleration for future runs
- [ ] Run full evaluation on test set
- [ ] Implement Transformer baseline for comparison

### Priority 3: Model Improvements

- [ ] Add focal loss for OPERATOR vs SYMBOL confusion
- [ ] Implement proper unpooling
- [ ] Add data augmentation (vary masked positions)

## Files Generated

- **Checkpoints**: `checkpoints/best_model.pt` (epoch 50)
- **TensorBoard logs**: `runs/`
- **GPU profile**: `runs/gpu_profile.json`
- **Analysis script**: `scripts/show_examples.py`

## References

See detailed analysis in:

- `docs/performance_analysis.md` (if copied from artifacts)
- Training walkthrough: `docs/training_walkthrough.md` (if copied from artifacts)
