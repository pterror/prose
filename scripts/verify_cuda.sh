#!/usr/bin/env bash
# Quick script to verify CUDA setup after environment reload

echo "==================================="
echo "CUDA Setup Verification"
echo "==================================="

echo -e "\n1. Checking PyTorch CUDA detection..."
python -c "import torch; print('  PyTorch version:', torch.__version__); print('  CUDA available:', torch.cuda.is_available()); print('  Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('  Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

if [ $? -ne 0 ]; then
    echo "  ❌ PyTorch CUDA check failed"
    exit 1
fi

echo -e "\n2. Testing GPU memory allocation..."
python -c "import torch; x = torch.randn(1000, 1000).cuda(); print('  ✅ Allocated:', torch.cuda.memory_allocated() / 1e6, 'MB'); del x; torch.cuda.empty_cache()"

if [ $? -ne 0 ]; then
    echo "  ❌ GPU memory allocation failed"
    exit 1
fi

echo -e "\n3. Checking CUDA library paths..."
echo "  LD_LIBRARY_PATH contains:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -i cuda | head -n 5

echo -e "\n==================================="
echo "✅ CUDA Setup Verified!"
echo "==================================="
echo "You can now train on GPU. Expected speedup: 3-5x"
