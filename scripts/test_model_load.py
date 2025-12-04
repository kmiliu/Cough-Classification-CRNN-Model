#!/usr/bin/env python3
"""scripts/test_model_load.py

Simple script to detect the newest .pt in output/, load the CRNN model defined in
app.py, and run a dummy forward pass to ensure compatibility.

Exit codes:
- 0: success
- 2: no model found
- 3: model load failed
- 4: dry-run failed
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    import app
except Exception as e:
    print('Failed to import dependencies:', e)
    sys.exit(3)

feats, med, mean_std = app.load_train_stats()
print('Feature count:', len(feats))

detected = app.find_latest_model_in_output()
if detected is None:
    print('No model detected under output/. Place a .pt file under output/ and try again.')
    sys.exit(2)
# Print only folder/filename to avoid exposing absolute paths
print('Detected model:', f"{detected.parent.name}/{detected.name}")

model = app.load_model_from_path(detected, len(feats))
if not model:
    print('Model load failed for:', f"{detected.parent.name}/{detected.name}")
    sys.exit(3)
print('Model loaded successfully')

# Dry-run
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy = torch.zeros(1,1,len(feats), dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(dummy)
    print('Dry-run output shape:', getattr(out, 'shape', out))
    print('Dry-run succeeded')
except Exception as e:
    print('Dry-run failed with error:', e)
    sys.exit(4)

print('OK')
sys.exit(0)
