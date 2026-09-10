# nnInteractive ESO CTV Stage-1 lasso experiment

This folder is deliberately independent from SAM2 training code.

## Fixed inputs

- Preprocessed data: `/home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/PreprocessDataNii/train`
- Shared folds: `/home/intern/ftp/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260901/shared_splits.json`
- Shared validation plan: `/home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults/validation_prompt_plan.json`
- Official model bundle: `/home/intern/ftp/wusi/nnInteractive/nnInteractive_v1.0`
- Results: `/home/intern/ftp/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TrainResults`

The data are already on the common Z=5 mm grid. This experiment must not
resample, resize, or window them again.

## First step: source audit only

Run this before any formal trainer is created or launched:

```bash
python audit_nninteractive.py \
  --repo /home/intern/ftp/wusi/nnInteractive \
  --model-dir /home/intern/ftp/wusi/nnInteractive/nnInteractive_v1.0
```

The static audit intentionally reports `BLOCKED` until a runtime coverage and
patch-sequence audit is completed. It is a protocol feasibility gate for human
review, not a training smoke test. The formal trainer never imports this audit
code, but it requires a retained `audit_result.json` with PASS and matching
fingerprints before formal training begins.
