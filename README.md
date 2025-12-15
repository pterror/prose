# prose - Program Synthesis Experiment

Graph U-Net for code synthesis using Abstract Syntax Graphs (ASG).

## Overview

This project implements a specialized neural model for program synthesis and refactoring:

- **Architecture**: Graph U-Net with Graph Attention Networks
- **Size**: ~5M parameters (Phase 1 prototype)
- **Target**: Local hardware (NVIDIA RTX 3060)
- **Approach**: Neuro-symbolic with constrained decoding

## Phase 1 Results (December 2025)

**Training Completed**: 50 epochs on 2K training samples

- **Best Validation Accuracy**: 86.39%
- **Model Size**: 0.63M parameters
- **Key Finding**: Systematic OPERATOR ↔ SYMBOL confusion due to positional overfitting
- **Root Cause**: Limited template diversity (~400-500 unique templates × 4-6 duplication)

**Next Steps**: Expand templates, add position encodings, increase model depth to 5-7 layers.

See [`docs/phase1_results.md`](docs/phase1_results.md) for detailed analysis.

---

## Quick Start

### 1. Setup Environment

```bash
# Using Nix (recommended)
direnv allow

# Or manual setup
chmod +x setup.sh
./setup.sh
```

### 2. Generate Synthetic Data

```bash
# Generate 10K samples for quick testing
python scripts/01_generate_data.py \
    --num-samples 10000 \
    --output data/processed/train \
    --balanced

# Split for validation (manually copy ~20%)
mkdir -p data/processed/val
# ... move some files ...
```

### 3. Test Data Pipeline

```bash
python scripts/test_data_pipeline.py
```

### 4. Train Model

```bash
# Quick smoke test (1 epoch)
python scripts/02_train_prototype.py \
    --config configs/phase1_prototype.yaml \
    --epochs 1 \
    --batch-size 32

# Full training (50 epochs, ~6-10 hours on RTX 3060)
python scripts/02_train_prototype.py \
    --config configs/phase1_prototype.yaml
```

### 5. Monitor Training

```bash
tensorboard --logdir runs/
```

### 6. Evaluate Model

```bash
# Evaluate best checkpoint on test set
python scripts/03_evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --test-dir data/processed/test

# Skip visualization generation (faster)
python scripts/03_evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --no-viz
```

## Evaluation Metrics

The evaluation system provides comprehensive metrics for assessing model performance:

### Exact Match Rate

Percentage of graphs that are perfectly reconstructed (all nodes and edges match).

- **Interpretation**: Strict measure of complete correctness
- **Target**: >10% for initial prototype

### Node Accuracy

Per-node classification accuracy (% of nodes with correct type).

- **Interpretation**: How well the model predicts individual node types
- **Random baseline**: ~11% (9 node types including MASK)
- **Target**: >80% for good performance

### Edge F1 Score

Precision and recall for edge prediction across all edge types.

- **Interpretation**: How well the model reconstructs graph structure
- **Components**: Precision (correct edges / predicted edges), Recall (correct edges / ground truth edges)
- **Target**: >0.7 for good structural understanding

### Syntax Validity

Percentage of reconstructed graphs that pass Mini-Lisp grammar validation.

- **Interpretation**: Whether predictions are syntactically valid programs
- **Target**: >90% (model should produce valid structures)

## Project Structure

```
prose/
├── configs/               # YAML configuration files
├── data/                  # Dataset storage
│   ├── processed/         # Generated ASG files (.pt)
│   └── raw/              # (unused in Phase 1)
├── results/              # Evaluation outputs
│   └── visualizations/   # ASG reconstruction comparisons
├── scripts/              # Training and data generation scripts
├── src/
│   ├── data/             # ASG builder, synthetic generator
│   ├── models/           # Graph U-Net architecture
│   ├── training/         # Loss functions, trainers, metrics
│   ├── utils/            # Visualization utilities
│   ├── baselines/        # Transformer baseline (TODO)
│   └── eval/             # (deprecated, merged into training/)
└── tests/                # Unit tests
```

## Architecture Details

### Abstract Syntax Graph (ASG)

Unlike standard ASTs, we use graphs with three edge types:

- **Child**: Parent-child syntactic hierarchy
- **Sibling**: Sequential evaluation order
- **DataFlow**: Variable definition → usage

This resolves the "long context" problem by directly linking related nodes.

### Graph U-Net

Hierarchical encoder-decoder:

1. **Encoder**: GAT layers + TopK pooling (3 levels)
2. **Bottleneck**: Deep processing at coarsest level
3. **Decoder**: Unpooling + skip connections (simplified in Phase 1)

### Training Task

**Denoising Auto-Encoder**:

- Corrupt 20% of nodes (replace with `[MASK]` token)
- Model predicts original node types
- Loss: Cross-entropy on node predictions

## Configuration

Edit `configs/phase1_prototype.yaml` to tune:

- Model size (`hidden_channels`, `depth`)
- Training hyperparameters (`lr`, `batch_size`, `epochs`)
- Data generation (`corruption_rate`, `num_samples`)

## Hardware Requirements

**Minimum**:

- GPU: NVIDIA RTX 3060 (12GB VRAM)
- RAM: 16GB
- Storage: ~10GB for 1M samples

**Typical Resource Usage**:

- Training: ~10GB VRAM
- Batch size: 32 (with 4x gradient accumulation = effective 128)
- Time: 6-10 hours for 50 epochs (1M samples)

## Next Steps (Phase 2)

- [ ] Implement full unpooling for true U-Net architecture
- [ ] Add Transformer baseline for comparison
- [ ] Scale to 50M-100M parameters
- [ ] Implement discrete diffusion (iterative refinement)
- [ ] Add symbolic constraints (logit masking, SMT solver)

## References

- **Graph U-Nets**: Gao & Ji (2019)
- **Tree Diffusion**: Kapur et al. (2025)
- **PyTorch Geometric**: Fey & Lenssen (2019)

## License

MIT
