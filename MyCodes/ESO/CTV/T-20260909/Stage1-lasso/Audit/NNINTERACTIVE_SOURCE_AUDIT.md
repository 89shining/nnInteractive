# nnInteractive CTV Stage-1 source audit

This report audits the source and the official model bundle only. It does not
run training, alter data, or claim full-volume coverage without a runtime probe.

- Source repository: `/home/intern/ftp/wusi/nnInteractive`
- Official model bundle: `/home/intern/ftp/wusi/nnInteractive/nnInteractive_v1.0`
- Static feasibility result: **PASS**

| Check | Result | Evidence |
| --- | --- | --- |
| official model bundle | PASS | requires dataset.json, plans.json, inference_session_class.json, and fold_0/checkpoint_final.pth |
| official lasso representation | PASS | README defines a lasso as a one-slice closed contour |
| positive-lasso interaction channel | PASS | add_lasso_interaction writes positive lasso to channel -6 |
| native interaction decay | PASS | batch wrapper may bypass sequential decay only after unit-strength verification |
| union ROI entry point | PASS | native generic image helper derives queue initialization from nonzero extent |
| last-center default risk | PASS | joint wrapper must enqueue one union item, not K native entries |
| native inference-only path | PASS | validation may use _predict; differentiable training requires a wrapper |
| raw-logit interception point | PASS | training wrapper must retain the two-class tensor before argmax |
| overwrite write-back semantics | PASS | native spatial policy is last-write-wins |
| runtime full-supervision coverage | PASS | A6000 runtime report confirms completion coverage for every audited unprompted voxel |
| training/validation patch-sequence equivalence | PASS | A6000 runtime report found no native/wrapper trace mismatch |

## Confirmed by this audit

- **official model bundle**: requires dataset.json, plans.json, inference_session_class.json, and fold_0/checkpoint_final.pth
- **official lasso representation**: README defines a lasso as a one-slice closed contour
- **positive-lasso interaction channel**: add_lasso_interaction writes positive lasso to channel -6
- **native interaction decay**: batch wrapper may bypass sequential decay only after unit-strength verification
- **union ROI entry point**: native generic image helper derives queue initialization from nonzero extent
- **last-center default risk**: joint wrapper must enqueue one union item, not K native entries
- **native inference-only path**: validation may use _predict; differentiable training requires a wrapper
- **raw-logit interception point**: training wrapper must retain the two-class tensor before argmax
- **overwrite write-back semantics**: native spatial policy is last-write-wins
- **runtime full-supervision coverage**: A6000 runtime report confirms completion coverage for every audited unprompted voxel
- **training/validation patch-sequence equivalence**: A6000 runtime report found no native/wrapper trace mismatch

## Unconfirmed or blocked assumptions

- None

## Audit fingerprints

- `repository_git_commit`: `7ee5d202d3d927e840c2a23b8e6c635bc554fbcf`
- `audit_script_sha256`: `3d68e85bf26874edd5d936971b609d8b51d46d8d71574be664a0a35fbe53821b`
- `inference_session_py_sha256`: `28953eb070f91d9e7f07b814f09b36a259bfa36dd8bd6c6c0c3fe0396d7f6701`
- `crop_py_sha256`: `dd9a0b67a2e99e76047b104e1dbc9feb7accd5dac5d70893c4e8fc90693c85b7`
- `model_checkpoint_sha256`: `b3ac4421f85457bbd1aa0d87f5e67bcb7bc8e2ce6b824b6ac45077cc5d630ea9`
- `model_checkpoint_size_bytes`: `411387150`
- `model_checkpoint_mtime_ns`: `1788979273113191024`
- `dataset_json_sha256`: `e090394ac539b2823b2135173658190245c462cb042d3442487e4829f08252c4`
- `plans_json_sha256`: `64a30a430438302a692a3edadd345a207b3428bea2e5cdc125298757da21f4d4`
- `inference_session_class_json_sha256`: `8f43587a9d139fcf7af8776219e9432dab079f985cb0d7da72a4c851b39a4f09`
- `runtime_report_sha256`: `bb20463d63144f4258701ce01d149d0f0926244e49e843d4117552178fdca99f`
- `training_code_sha256`: `{'coordinate_adapter.py': '7dd6f32d18bee9fa71b90e7c7ad5afa0ecbb613b65c87039c4378fdd55357aba', 'dataset.py': 'e2bac6cff4ad4b0335117ed7cc614edc813f437e8b52b5f42cb0862ca30719ae', 'differentiable_buffer.py': '96a9e822d06a612483107ba126e6a0947c95f977baf049bf976a024ac23c4c25', 'interaction_builder.py': '27bb87c1fb53d1488af0066e3b35e5715b6ea23830a2f3dcefd4b58d1e76eb00', 'losses.py': 'f01ff7c7ee28cf219eb65ffa956ad95f4b4c8876f5b9fc84bdb4554841c16ea1', 'prediction_wrapper.py': '6947164edf6ed169fcf8e567dcb960ae97ce554662432e1f6550172ce1d50874', 'prompt_sampler.py': 'eae5c3595352803c66352886eecb1b64a22ebeffe72d09fbb7f767fefae1f364', 'train.py': '0ed797038ffced6d26050b10177d1e2a5548b0350cc7cb587459ed7a7f8d5243'}`

## Go / No-Go rule

Formal training remains **BLOCKED** until a runtime audit on the target A6000
environment proves native/wrapper trajectory equivalence and full unprompted
coverage after the approved training-only fixed-grid completion sweep.
