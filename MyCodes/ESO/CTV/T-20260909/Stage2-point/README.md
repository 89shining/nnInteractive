# ESO CTV nnInteractive Stage-2 point-correction fine-tuning

This directory is independent of `Stage1-lasso`. It reuses the same offline
Z=5-mm `[0,1]` NIfTI volumes, patient folds, Stage-1 validation plan, and
component-wise axial lasso construction.

## Protocol

- A single full-network nnInteractive model is initialized from the **matching
  fold's** Stage-1 `best.pth`.
- Training samples a feasible `K=1..5` lasso placement with a two-slice gap,
  then samples `T ~ Uniform{0,1,2,3,4,5}` correction points.
- Native hard predictions alone generate intermediate residuals and clicks.
  Clicks are the largest 26-connected FN/FP component, exclude initial lasso
  slices, and use spacing-aware EDT inner-quartile random selection.
- The terminal differentiable replay starts from a deep, isolated snapshot
  taken *after* the terminal interaction is registered and *before* native
  `_predict()` consumes it. Its raw logits receive training-only fixed-grid
  coverage completion. Neither replay nor completion can mutate the native
  session trajectory.
- The sole loss is `0.5 soft Dice + 0.5 CE` on the terminal prediction,
  excluding only initial-lasso slices.
- Validation is native only: shared fixed `K=3` placements, deterministic
  EDT-max clicks, whole-volume `D0..D5`, with best checkpoint selected by
  `mean(D0..D5)`.

## Server defaults

```text
Data: /home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/PreprocessDataNii/train
Splits: /home/intern/ftp/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260901/shared_splits.json
Validation plan: /home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults/validation_prompt_plan.json
Official model: /home/intern/ftp/wusi/nnInteractive/nnInteractive_v1.0
Stage-1 results: /home/intern/ftp/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TrainResults
Stage-2 results: /home/intern/ftp/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage2-point/TrainResults
```

For fold `N`, the default initializer is strictly:

```text
.../Stage1-lasso/TrainResults/Mixed_K1_5/fold_N/checkpoints/best.pth
```

The trainer refuses an explicit checkpoint that is not under the requested
`fold_N` path. It does not start automatically.
