# ESO CTV Stage-1 lasso fine-tuning

The trainer never imports or executes an audit script. It does require the
immutable approval record `Audit/audit_result.json` to be `PASS` and to match
the audited source/checkpoint fingerprints before it will begin formal training.
Deleting `Audit/audit_nninteractive.py` after approval does not affect training.

## Fixed server paths

```text
Data: /home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/PreprocessDataNii/train
Splits: /home/intern/ftp/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260901/shared_splits.json
Validation plan: /home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults/validation_prompt_plan.json
Model: /home/intern/ftp/wusi/nnInteractive/nnInteractive_v1.0
Results: /home/intern/ftp/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TrainResults
```

The input data are already Z=5 mm and `[0,1]`. This code never resamples,
resizes, or windows them again.

## One-time environment setup

The nnInteractive repository must be installed in the active environment so
that its package metadata and official architecture reconstruction work:

```bash
pip install -e /home/intern/ftp/wusi/nnInteractive
```

## Formal run

```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --fold 0 \
  > /home/intern/ftp/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TrainResults/nohup_fold0.log 2>&1 &
```

The command intentionally aborts if `Audit/audit_result.json` is missing,
blocked, or fingerprints a different nnInteractive source/checkpoint.

The protocol is K=1--5 positive axial closed-contour lassos, SI-stratified
sampling with a two-slice gap, joint unit-strength registration, one union ROI,
unprompted-only `0.5 soft Dice + 0.5 CE`, and fixed two-placement validation.
## Lasso prompt semantics

`K` is the number of prompted axial slices. Each selected CTV mask is split
into its 4-connected regions; each region supplies one closed-contour lasso.
All component contours across the K slices are jointly OR-merged at unit
strength before native nnInteractive prediction. Thus a prompted slice with
two disconnected CTV regions remains one prompted slice and does not change
K, sampling, loss exclusion, or the validation plan.
