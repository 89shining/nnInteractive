# nnInteractive CTV Stage-1 source audit

This report audits the source and the official model bundle only. It does not
run training, alter data, or claim full-volume coverage without a runtime probe.

- Source repository: `D:\project\nnInteractive`
- Official model bundle: `D:\project\nnInteractive\nnInteractive_v1.0`
- Static feasibility result: **BLOCKED**

| Check | Result | Evidence |
| --- | --- | --- |
| official model bundle | PASS | requires dataset.json, plans.json, and fold_0/checkpoint_final.pth |
| official lasso representation | PASS | README defines a lasso as a one-slice closed contour |
| positive-lasso interaction channel | PASS | add_lasso_interaction writes positive lasso to channel -6 |
| native interaction decay | PASS | batch wrapper may bypass sequential decay only after unit-strength verification |
| union ROI entry point | PASS | native generic image helper derives queue initialization from nonzero extent |
| last-center default risk | PASS | joint wrapper must enqueue one union item, not K native entries |
| native inference-only path | PASS | validation may use _predict; differentiable training requires a wrapper |
| raw-logit interception point | PASS | training wrapper must retain the two-class tensor before argmax |
| overwrite write-back semantics | PASS | native spatial policy is last-write-wins |
| runtime full-supervision coverage | BLOCKED | static source alone cannot prove that every intended unprompted voxel receives network-derived logits |
| training/validation patch-sequence equivalence | BLOCKED | requires a fixed-case A6000 runtime trace comparison before formal training |

## Confirmed by this audit

- **official model bundle**: requires dataset.json, plans.json, and fold_0/checkpoint_final.pth
- **official lasso representation**: README defines a lasso as a one-slice closed contour
- **positive-lasso interaction channel**: add_lasso_interaction writes positive lasso to channel -6
- **native interaction decay**: batch wrapper may bypass sequential decay only after unit-strength verification
- **union ROI entry point**: native generic image helper derives queue initialization from nonzero extent
- **last-center default risk**: joint wrapper must enqueue one union item, not K native entries
- **native inference-only path**: validation may use _predict; differentiable training requires a wrapper
- **raw-logit interception point**: training wrapper must retain the two-class tensor before argmax
- **overwrite write-back semantics**: native spatial policy is last-write-wins

## Unconfirmed or blocked assumptions

- **runtime full-supervision coverage**: static source alone cannot prove that every intended unprompted voxel receives network-derived logits
- **training/validation patch-sequence equivalence**: requires a fixed-case A6000 runtime trace comparison before formal training

## Audit fingerprints

- `repository_git_commit`: `7ee5d202d3d927e840c2a23b8e6c635bc554fbcf`
- `audit_script_sha256`: `ae9b4e0692c19403449a0a590365074467a680ae57abef9ca1da7e78af766d1d`
- `inference_session_py_sha256`: `a9f6f14f529fc5693764b99ebc5c9f50a306d0bf2d3aea10f2d4071aea53bd78`
- `crop_py_sha256`: `dd9a0b67a2e99e76047b104e1dbc9feb7accd5dac5d70893c4e8fc90693c85b7`
- `model_checkpoint_sha256`: `b3ac4421f85457bbd1aa0d87f5e67bcb7bc8e2ce6b824b6ac45077cc5d630ea9`
- `model_checkpoint_size_bytes`: `411387150`
- `model_checkpoint_mtime_ns`: `1788965001825625900`

## Go / No-Go rule

Formal training remains **BLOCKED** until a runtime audit on the target A6000
environment proves that the native union-ROI patch trajectory covers every
unprompted voxel intended for Dice+CE supervision. If this fails, do not add a
sliding-window or other coverage algorithm without revising the protocol.
