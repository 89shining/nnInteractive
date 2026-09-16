#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/wusi/nnInteractive/MyCodes/ESO/CTV/T-20260909/Stage1-lasso-test
OUT=/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TestResults/main_deterministic/fold_0_only
mkdir -p "$OUT"
python3 - <<'PY' > "$OUT/run_manifest.json"
import json
print(json.dumps({"model":"nnInteractive","scope":"fold_0_only","physical_gpu":4,"command":"run_stage1_inference.py --fold 0 --device cuda:0 --save-predictions"}, indent=2))
PY
cd "$ROOT"
CUDA_VISIBLE_DEVICES=4 /home/wusi/miniconda3/envs/nninteractive/bin/python run_stage1_inference.py --fold 0 --device cuda:0 --save-predictions 2>&1 | tee "$OUT/run.log"
