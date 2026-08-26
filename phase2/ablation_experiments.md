# Phase2 消融实验命令清单（2026-08-24，seed=20260824，已按审核修订）

> 状态：**仅代码与命令准备，未启动任何训练/完整渲染。** 等待人工审核。

## 0. 公共常量

```
PY         = D:\face_standardization_project\.venv\Scripts\python.exe
ROOT       = D:\face_standardization_project
DECA_ROOT  = D:\face_standardization_project\DECA
TRAIN_MATS = D:\face_standardization_project\DECA\results\archive_phase2_params     (10,000 base mats)
ARCFACE    = D:\face_standardization_project\results\arcface_p95_rebuilt\arcface_manifest.csv
ARCFACE_TEST = D:\face_standardization_project\results\phase2_eval_fixed_20260824_v2\arcface_fixed_test_manifest.csv
XGB_OOF    = D:\face_standardization_project\results\phase2_xgb_rebuilt_20260824\xgb_oof_phase2_manifest.csv
XGB_TEST   = D:\face_standardization_project\results\phase2_xgb_rebuilt_20260824\xgb_fixed_test_predictions.csv
EXCLUDE    = D:\face_standardization_project\results\phase2_eval_fixed_20260824_v2\base_test_ids.txt   (400 base IDs)
TEST_MF    = D:\face_standardization_project\results\phase2_eval_fixed_20260824_v2\fixed_test_manifest_v2.csv
L2CS       = D:\face_standardization_project\models\l2cs\L2CSNet_gaze360.pkl
ABLATION   = D:\face_standardization_project\results\phase2_ablation_20260824
EXT_DECA   = D:\face_standardization_project\results\phase2_eval_fixed_20260824_v2\deca_params
FIXED_MATS = %ABLATION%\fixed_test_mats
FIXED_IDS  = %ABLATION%\fixed_test_mats\fixed_test_ids.txt
```

训练参数（B–E 完全一致，**stage=3、epochs=40**）：
`--epochs 40 --batch-size 64 --lr 1e-3 --hidden-dim 256 --stage 3 --val-ratio 0.15
--device cuda --seed 20260824 --exclude-ids-file %EXCLUDE%`
（optimizer = AdamW weight_decay=1e-4，脚本内固定；**不加 early stopping**，以最低 val loss 保留 `best_model.pt`。）

> 模型 `ConditionGenerator` 的 input_dim = 99（expression 50 + pose 6 + camera 3 + light 27 + 13 个 metric 特征）。

## 1. 前置：合并固定测试的 775 张 DECA mat（只跑一次）

```
%PY% -m phase2.prepare_fixed_test_mats
```

## 2. 前置：补齐外部样本 DECA 参数（任务 A，完整 375 张，FAN 主模式）

```
%PY% -m phase2.run_fixed_external_deca --mode fan --resume --update-manifest
%PY% -m phase2.run_fixed_external_deca --mode rescue --resume           # rescue 对照，独立目录，不更新 manifest
```

## 3. 五组实验

### A. hard_zero（不训练，exp/pose 置零）

无训练/推理。hard_zero 渲染由 `render_evaluation_batch.py` 作为 any render run 的副产品产出（`hard_zero_*` 列）；评估只取 `method=hard_zero`。

```
%PY% -m phase2.render_evaluation_batch --test-manifest %TEST_MF% --phase2-manifest %ABLATION%\full\phase2_inference_manifest.csv --deca-root %DECA_ROOT% --out-dir %ABLATION%\hard_zero --device cuda
%PY% -m phase2.evaluate_rendered_outputs --test-manifest %TEST_MF% --render-manifest %ABLATION%\hard_zero\render_manifest.csv --deca-root %DECA_ROOT% --out-dir %ABLATION%\hard_zero --l2cs-weights %L2CS% --device cuda
%PY% -m phase2.write_experiment_inventory --exp-dir %ABLATION%\hard_zero --exp-name hard_zero --seed 20260824 --quality-source heuristic --alpha-mode fixed_one --augmentation false --command "hard_zero: exp=0 pose=0 (no model)"
```

### B. no_alpha（alpha_mode=fixed_one）

```
%PY% -m phase2.train_condition_generator --deca-results-dir %TRAIN_MATS% --arcface-manifest %ARCFACE% --xgb-quality-manifest %XGB_OOF% --quality-source blend --alpha-mode fixed_one --out-dir %ABLATION%\no_alpha --seed 20260824 --epochs 40 --batch-size 64 --lr 1e-3 --hidden-dim 256 --stage 3 --val-ratio 0.15 --device cuda --exclude-ids-file %EXCLUDE%
%PY% -m phase2.infer_standardize_params --deca-results-dir %FIXED_MATS% --checkpoint %ABLATION%\no_alpha\best_model.pt --arcface-manifest %ARCFACE_TEST% --xgb-quality-manifest %XGB_TEST% --quality-source blend --alpha-mode fixed_one --include-ids-file %FIXED_IDS% --out-dir %ABLATION%\no_alpha --device cuda
%PY% -m phase2.render_evaluation_batch --test-manifest %TEST_MF% --phase2-manifest %ABLATION%\no_alpha\phase2_inference_manifest.csv --deca-root %DECA_ROOT% --out-dir %ABLATION%\no_alpha --device cuda
%PY% -m phase2.evaluate_rendered_outputs --test-manifest %TEST_MF% --render-manifest %ABLATION%\no_alpha\render_manifest.csv --deca-root %DECA_ROOT% --out-dir %ABLATION%\no_alpha --l2cs-weights %L2CS% --device cuda
%PY% -m phase2.write_experiment_inventory --exp-dir %ABLATION%\no_alpha --exp-name no_alpha --seed 20260824 --quality-source blend --alpha-mode fixed_one --augmentation true --command "<train cmd above>" --checkpoint %ABLATION%\no_alpha\best_model.pt --xgb-manifest %XGB_TEST%
```

### C. no_augmentation（--no-augment）

```
%PY% -m phase2.train_condition_generator --deca-results-dir %TRAIN_MATS% --arcface-manifest %ARCFACE% --xgb-quality-manifest %XGB_OOF% --quality-source blend --alpha-mode learned --no-augment --out-dir %ABLATION%\no_augmentation --seed 20260824 --epochs 40 --batch-size 64 --lr 1e-3 --hidden-dim 256 --stage 3 --val-ratio 0.15 --device cuda --exclude-ids-file %EXCLUDE%
%PY% -m phase2.infer_standardize_params --deca-results-dir %FIXED_MATS% --checkpoint %ABLATION%\no_augmentation\best_model.pt --arcface-manifest %ARCFACE_TEST% --xgb-quality-manifest %XGB_TEST% --quality-source blend --alpha-mode learned --include-ids-file %FIXED_IDS% --out-dir %ABLATION%\no_augmentation --device cuda
%PY% -m phase2.render_evaluation_batch --test-manifest %TEST_MF% --phase2-manifest %ABLATION%\no_augmentation\phase2_inference_manifest.csv --deca-root %DECA_ROOT% --out-dir %ABLATION%\no_augmentation --device cuda
%PY% -m phase2.evaluate_rendered_outputs --test-manifest %TEST_MF% --render-manifest %ABLATION%\no_augmentation\render_manifest.csv --deca-root %DECA_ROOT% --out-dir %ABLATION%\no_augmentation --l2cs-weights %L2CS% --device cuda
%PY% -m phase2.write_experiment_inventory --exp-dir %ABLATION%\no_augmentation --exp-name no_augmentation --seed 20260824 --quality-source blend --alpha-mode learned --augmentation false --command "<train cmd above>" --checkpoint %ABLATION%\no_augmentation\best_model.pt --xgb-manifest %XGB_TEST%
```

### D. no_xgboost（不传 XGBoost manifest，quality_source=heuristic）

```
%PY% -m phase2.train_condition_generator --deca-results-dir %TRAIN_MATS% --arcface-manifest %ARCFACE% --quality-source heuristic --alpha-mode learned --out-dir %ABLATION%\no_xgboost --seed 20260824 --epochs 40 --batch-size 64 --lr 1e-3 --hidden-dim 256 --stage 3 --val-ratio 0.15 --device cuda --exclude-ids-file %EXCLUDE%
%PY% -m phase2.infer_standardize_params --deca-results-dir %FIXED_MATS% --checkpoint %ABLATION%\no_xgboost\best_model.pt --arcface-manifest %ARCFACE_TEST% --quality-source heuristic --alpha-mode learned --include-ids-file %FIXED_IDS% --out-dir %ABLATION%\no_xgboost --device cuda
%PY% -m phase2.render_evaluation_batch --test-manifest %TEST_MF% --phase2-manifest %ABLATION%\no_xgboost\phase2_inference_manifest.csv --deca-root %DECA_ROOT% --out-dir %ABLATION%\no_xgboost --device cuda
%PY% -m phase2.evaluate_rendered_outputs --test-manifest %TEST_MF% --render-manifest %ABLATION%\no_xgboost\render_manifest.csv --deca-root %DECA_ROOT% --out-dir %ABLATION%\no_xgboost --l2cs-weights %L2CS% --device cuda
%PY% -m phase2.write_experiment_inventory --exp-dir %ABLATION%\no_xgboost --exp-name no_xgboost --seed 20260824 --quality-source heuristic --alpha-mode learned --augmentation true --command "<train cmd above>" --checkpoint %ABLATION%\no_xgboost\best_model.pt
```

### E. full（**quality-source=blend、alpha-mode=learned、stage=3、augmentation enabled**）

```
%PY% -m phase2.train_condition_generator --deca-results-dir %TRAIN_MATS% --arcface-manifest %ARCFACE% --xgb-quality-manifest %XGB_OOF% --quality-source blend --alpha-mode learned --out-dir %ABLATION%\full --seed 20260824 --epochs 40 --batch-size 64 --lr 1e-3 --hidden-dim 256 --stage 3 --val-ratio 0.15 --device cuda --exclude-ids-file %EXCLUDE%
%PY% -m phase2.infer_standardize_params --deca-results-dir %FIXED_MATS% --checkpoint %ABLATION%\full\best_model.pt --arcface-manifest %ARCFACE_TEST% --xgb-quality-manifest %XGB_TEST% --quality-source blend --alpha-mode learned --include-ids-file %FIXED_IDS% --out-dir %ABLATION%\full --device cuda
%PY% -m phase2.render_evaluation_batch --test-manifest %TEST_MF% --phase2-manifest %ABLATION%\full\phase2_inference_manifest.csv --deca-root %DECA_ROOT% --out-dir %ABLATION%\full --device cuda
%PY% -m phase2.evaluate_rendered_outputs --test-manifest %TEST_MF% --render-manifest %ABLATION%\full\render_manifest.csv --deca-root %DECA_ROOT% --out-dir %ABLATION%\full --l2cs-weights %L2CS% --device cuda
%PY% -m phase2.write_experiment_inventory --exp-dir %ABLATION%\full --exp-name full --seed 20260824 --quality-source blend --alpha-mode learned --augmentation true --command "<train cmd above>" --checkpoint %ABLATION%\full\best_model.pt --xgb-manifest %XGB_TEST%
```

## 4. 每组唯一改变的变量

| 实验 | 唯一变量 | 训练? | quality_source | alpha_mode | stage | augmentation | xgb manifest |
|---|---|---|---|---|---|---|---|
| A. hard_zero | exp/pose 直接置零，不训练 | 否 | — | fixed_one 占位 | — | — | — |
| B. no_alpha | `alpha_mode=fixed_one` | 是 | blend | fixed_one | 3 | 是 | 传入 |
| C. no_augmentation | `--no-augment` | 是 | blend | learned | 3 | 否 | 传入 |
| D. no_xgboost | 不传 XGBoost manifest | 是 | heuristic | learned | 3 | 是 | 不传 |
| E. full | 参考组 | 是 | blend | learned | 3 | 是 | 传入 |

B/C/D/E 均使用相同 `--seed 20260824`、相同 train/val 划分、相同 epochs/batch/lr/hidden/
stage=3/val-ratio/optimizer、最低 val loss checkpoint、无 early stopping，且都 `--exclude-ids-file`
指向 `base_test_ids.txt`；推理/渲染/评估统一使用 `fixed_test_manifest_v2.csv`。

## 5. 每实验目录必含产物

`exact_command.txt`、`config.json`（训练/推理脚本自动保存）、`best_model.pt`、`normalizer.npz`、
`train_history.csv`、`train_summary.json`、`phase2_inference_manifest.csv`、`phase2_inference_summary.json`、
`render_manifest.csv`、`render_summary.json`、`rendered_metrics.csv`、`rendered_metrics_summary.json`、
`experiment_inventory.json`（由 `write_experiment_inventory.py` 生成）。

注：A 组（hard_zero）无 `best_model.pt`/`normalizer.npz`/`train_history.csv`（不训练）。

## 6. XGBoost 进入推理的核查

- 推理清单新增 `quality_source_requested`（= `--quality-source`）与 `quality_source_effective`
  （xgb/blend 请求但该样本无 xgb 标注时回退为 `heuristic`，绝不把 heuristic 写成 xgb score）。
- base 400 张：xgb_quality_label ∈ {high/medium/low}，xgb_quality_score 为真实分数。
- 外部 375 张：无 xgb 标注 → `xgb_quality_score=` 空、`xgb_quality_label=external_unlabeled`、
  `quality_source_effective=heuristic`（除非另行对 375 张跑 XGBoost 预测，见第 8 节）。

## 7. 待人工确认

1. 外部 375 张能否获得真实 XGBoost 分数（需先对 375 张跑 ArcFace 得到 arcface_status/score，
   再写 predict-only 路径调用已保存的 `xgb_quality_model.json`；当前尚未实现）。
2. hard_zero 组渲染复用 full 的 phase2 manifest 仅取 hard_zero 列，是否接受。
3. 外部样本 DECA mat 命名（目录=eval_id、文件=image_id）与推理 image_id 关联，是否确认。
