#!/bin/bash
#
# Quick setup script to install dependencies and prepare environment

set -e

echo "=== Setting up prose project ==="

# Install Python dependencies
echo "Installing dependencies with uv..."
uv pip install -e .

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "1. Generate test data:"
echo "   python scripts/01_generate_data.py --num-samples 1000 --output data/processed/train"
echo ""
echo "2. Run data pipeline test:"
echo "   python scripts/test_data_pipeline.py"
echo ""
echo "3. Train model:"
echo "   python scripts/02_train_prototype.py --config configs/phase1_prototype.yaml"
