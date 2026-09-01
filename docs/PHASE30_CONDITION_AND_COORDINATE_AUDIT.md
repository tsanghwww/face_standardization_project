# Phase3.0 条件与坐标审计

日期：2026-09-01

状态：工程入口已建立；真实坐标方向人工审计未通过前，禁止启动 Phase3.1。

## 目的

Phase3.0 不训练扩散模型。它冻结数据划分、生成 DECA 空间条件，并验证 DECA head rotation 与 L2CS camera gaze 的坐标关系。这里的核心原则是：旋转往返可逆只是数学自洽，不等于真实轴向已经验证。

## 固定划分

- `train`：8,160，只用于优化。
- `validation`：1,440，只用于超参数、loss 权重和 checkpoint 选择。
- `fixed_test_base`：400 个 base 样本，只用于最终评估。
- `fixed_test_external`：375 个 WIDER/COFW/AFLW 困难样本，只用于最终域外评估。
- 总 fixed evaluation 为 775。rescue 继续是 audit-only，不进入主数据路径。

`scripts/prepare_phase3_splits.py` 会写入每组 ID、数量、SHA256 和两两交集；任何交集都会使命令失败。

## 坐标证据门槛

`scripts/build_phase3_gaze_manifest.py` 从 DECA `pose[:3]` 和 L2CS `gaze_{x,y,z}` 生成：

- `direct_head_to_camera` 候选：$g_{head}=R^Tg_{cam}$。
- 逆向候选，用于暴露 convention 选择而不是悄悄固定方向。
- 旋转 6D 表示和双候选 round-trip error。

默认状态始终为 `candidate_unvalidated`，且 `training_use_permitted=false`。必须按 head-pose 大小和方向分层抽样，人工检查投影方向后，才能建立如下审批文件：

```json
{
  "status": "approved",
  "convention": "direct_head_to_camera",
  "reviewer": "research_lead",
  "evidence": "pose-stratified visual audit artifact path"
}
```

## 条件缓存

`scripts/build_phase3_condition_cache.py` 从 source DECA MAT 和 Phase2 NPZ 分别生成 source/target：

- normal map；
- 16-bit depth map；
- landmark heatmap；
- face mask；
- eye-region mask。

没有坐标审批文件时，geometry 可以生成，但 gaze heatmap 保持缺失，状态为 `geometry_ready_gaze_pending`。提供合格审批文件后，才会按 `preserve_eye_in_head` 或 `canonical_camera_gaze` 生成眼区 gaze-ray heatmap。

## 5060 运行顺序

```powershell
$py = D:\face_standardization_project\.venv\Scripts\python.exe
$root = D:\face_standardization_project
$p30 = $root + "\results\phase30_20260901"

$py scripts\prepare_phase3_splits.py `
  --train-ids "$root\results\phase2_ablation_20260825\full\train_ids.txt" `
  --val-ids "$root\results\phase2_ablation_20260825\full\val_ids.txt" `
  --base-test-ids "$root\results\phase2_eval_fixed_20260824_v2\base_test_ids.txt" `
  --external-manifest "$root\results\phase2_eval_fixed_20260824_v2\fixed_test_manifest_v2.csv" `
  --external-filter-column source_dataset `
  --external-filter-values WIDER_FACE_val COFW_Color AFLW2000-3D `
  --expected-counts 8160,1440,400,375 `
  --out-dir "$p30\splits"

$py scripts\build_phase3_gaze_manifest.py `
  --phase1-manifest "$root\results\phase1_parity\phase1_master_manifest.csv" `
  --project-root $root `
  --split-registry-dir "$p30\splits" `
  --out-dir "$p30\gaze_candidates"

# 从 validation 中按旋转幅度和各轴极值选 12 张；不得从 fixed test 选样。
$py scripts\select_phase30_coordinate_audit_ids.py `
  --gaze-manifest "$p30\gaze_candidates\phase3_gaze_coordinate_candidates.csv" `
  --split validation --count 12 --out-dir "$p30\coordinate_audit"

# 渲染 pose-stratified geometry；此时不得提供 coordinate approval。
$py scripts\build_phase3_condition_cache.py `
  --phase1-manifest "$root\results\phase1_parity\phase1_master_manifest.csv" `
  --phase2-manifest "$root\results\phase2_infer_sanity_bug003_fixed_arcface_ok\phase2_inference_manifest.csv" `
  --project-root $root --deca-root "$root\DECA" `
  --split-registry-dir "$p30\splits" `
  --ids-file "$p30\coordinate_audit\coordinate_audit_ids.txt" `
  --out-dir "$p30\condition_audit" --device cuda

$py scripts\make_phase30_coordinate_audit.py `
  --phase1-manifest "$root\results\phase1_parity\phase1_master_manifest.csv" `
  --phase2-manifest "$root\results\phase2_infer_sanity_bug003_fixed_arcface_ok\phase2_inference_manifest.csv" `
  --condition-cache "$p30\condition_audit\phase3_condition_cache.csv" `
  --ids-file "$p30\coordinate_audit\coordinate_audit_ids.txt" `
  --project-root $root --out-dir "$p30\coordinate_audit\panels"

$py -m tests.test_phase30_protocol
```

首次编译 DECA standard rasterizer 时，需要从 Visual Studio x64 Developer Command Prompt 运行，或先调用 `VC\Auxiliary\Build\vcvars64.bat`；同时确保 `.venv\Scripts` 在 `PATH` 中，使 PyTorch 能找到 `ninja.exe`。

## Phase3.1 Go/No-Go

只有以下项目全部满足后才能进入 reconstruction warm-up：

1. split registry 数量符合 `8160/1440/400/375`，交集为 0。
2. source/target geometry smoke 非空、方向正常、无静默 0 填。
3. 大姿态分层人工审计批准唯一 gaze convention。
4. gaze heatmap 以眼区为锚点，head-only policy 的合成旋转行为正确。
5. 失败样本保留在完整分母，rescue 未进入 primary path。
