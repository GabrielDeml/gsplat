#!/bin/bash
# Run gsplat tests on MPS (Apple Silicon)
# Usage: conda activate splats2 && bash run_mps_tests.sh

set -e

export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "=== Smoke test: import gsplat ==="
python -c "
import torch
import gsplat
from gsplat.mps._backend import _C
from gsplat.mps.build import BOOTSTRAP_SHADER_NAME
print('torch:', torch.__version__)
print('mps available:', torch.backends.mps.is_available())
print('native mps backend initialized:', _C is not None)
print('native mps kernel names:', getattr(_C, 'kernel_names', ()))
print('has_3dgs:', gsplat.has_3dgs())
print('has_2dgs:', gsplat.has_2dgs())
print('has_3dgut:', gsplat.has_3dgut())
print('has_adam:', gsplat.has_adam())
print('has_camera_wrappers:', gsplat.has_camera_wrappers())
print('has_reloc:', gsplat.has_reloc())
if torch.backends.mps.is_available():
    assert _C is not None, 'Expected native MPS backend handle on MPS hardware'
    assert BOOTSTRAP_SHADER_NAME in _C.kernel_names, 'Bootstrap Metal kernel missing'
    assert _C.get_kernel(BOOTSTRAP_SHADER_NAME) is not None, 'Bootstrap Metal kernel not loadable'
print('Import OK')
"

echo ""
echo "=== Math tests ==="
python -m pytest tests/test_math.py -v --tb=short 2>&1 || true

echo ""
echo "=== Integration tests (non-distributed, non-lidar) ==="
python -m pytest tests/test_rasterization.py -v --tb=short -k "not distributed and not lidar" --maxfail=5 2>&1 || true

echo ""
echo "=== Basic tests ==="
python -m pytest tests/test_basic.py -v --tb=short --maxfail=5 2>&1 || true

echo ""
echo "=== 2DGS tests ==="
python -m pytest tests/test_2dgs.py -v --tb=short --maxfail=5 2>&1 || true

echo ""
echo "Done."
