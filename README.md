# prose - Program Synthesis Experiment

Graph U-Net for code synthesis using Abstract Syntax Graphs (ASG).

## Overview

This project implements a specialized neural model for program synthesis and refactoring:

- **Architecture**: Graph U-Net with Graph Attention Networks
- **Size**: ~5M parameters (Phase 1 prototype)
- **Target**: Local hardware (NVIDIA RTX 3060)
- **Approach**: Neuro-symbolic with constrained decoding

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

## Project Structure

```
prose/
├── configs/               # YAML configuration files
├── data/                  # Dataset storage
│   ├── processed/         # Generated ASG files (.pt)
│   └── raw/              # (unused in Phase 1)
├── scripts/              # Training and data generation scripts
├── src/
│   ├── data/             # ASG builder, synthetic generator
│   ├── models/           # Graph U-Net architecture
│   ├── training/         # Loss functions, trainers
│   ├── baselines/        # Transformer baseline (TODO)
│   └── eval/             # Evaluation metrics (TODO)
└── tests/                # Unit tests (TODO)
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
