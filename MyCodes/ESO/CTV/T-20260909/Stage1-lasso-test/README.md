# Stage 1 mask test

Generate the immutable prompt plan before running any model. The generator reads only ground-truth masks and never invokes training or inference.

Prompt-placement policy:

- Train: random one slice per SI stratum.
- Validation: fixed random SI-stratified placements.
- Main test: deterministic center of each SI stratum.
- Random robustness: multiple fixed random one-per-stratum placements.

```bash
python generate_stage1_prompt_plan.py
python generate_stage1_prompt_plan.py --placement random --seeds 0 1 2 3 4
```

Outputs default to the Stage1-lasso `TestResults/prompt_plans` directory.

After both immutable plans have been generated, run native Stage-1 test
inference. The driver uses only `joint_axial_lasso()` and
`JointLassoSession.native_predict_joint()`; it does not use Stage-2 logic or
validation placements. Each K (and each random seed/K plan) receives an
independently initialized session for each held-out fold.

```bash
# Deterministic main test
python run_stage1_inference.py --device cuda:0

# Random robustness test
python run_stage1_inference.py \
  --plan /home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TestResults/prompt_plans/stage1_random_prompt_plan_seeds_0-1-2-3-4.json \
  --device cuda:0
```

Use `--save-predictions` only when NIfTI prediction volumes are required;
metrics are always saved under the corresponding `TestResults` subdirectory.
