#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/wusi/nnInteractive/MyCodes/ESO/CTV/T-20260909/Stage2-lasso-point-test
CHECKPOINT=/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage2-point/TrainResults/Workflow_K3_T0_5/fold_0/checkpoints/best.pth
DATA=/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/PreprocessDataNii/test
PLAN=/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TestResults/external_test_prompt_plans/stage1_prompt_plan.json
OUT=/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage2-point/TestResults/fold0_checkpoint/stage2-lasso-point

test -f "$CHECKPOINT"
test ! -e "$OUT"
mkdir -p "$OUT"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" /home/wusi/miniconda3/envs/nninteractive/bin/python \
  run_stage2_lasso_point_evaluation.py \
  --checkpoint "$CHECKPOINT" --data-root "$DATA" --plan "$PLAN" \
  --output-dir "$OUT" --fold 0 --device cuda:0 --save-predictions 2>&1 | tee "$OUT/run.log"
