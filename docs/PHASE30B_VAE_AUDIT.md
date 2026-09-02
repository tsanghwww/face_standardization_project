# Phase3.0B VAE Round-trip Audit

日期：2026-09-02　状态：已完成。真实 VAE 运行在 32 张 validation 样本上完成，
全部 evaluator（ArcFace / DECA / L2CS）+ LPIPS 就绪并跑通；结果与 SHA256 产物
已写入 `results/phase30_20260901/vae_roundtrip_audit/`。

## 目的

在 32 张 pose-stratified **validation** 样本上，量化 frozen VAE（`stabilityai/sd-vae-ft-mse`）
对 identity / head pose / camera-gaze 的基础漂移，为 Phase3.1 的 reconstruction
anchor 提供证据。**不训练任何模型，不进入 500-step overfit。**

## 冻结 VAE 协议

```python
x_preprocessed = Lanczos_resize(source, 256)                # RGB, uint8
x = x_preprocessed / 127.5 - 1.0                            # [-1,1]
latent        = vae.encode(x).latent_dist.mode() * scaling_factor   # posterior mode(), NEVER sample()
reconstruction = vae.decode(latent / scaling_factor).sample
reconstruction = clamp(reconstruction, -1, 1) -> uint8
```

- 模型 `eval()`，全部参数 `requires_grad=False`；FP16 CUDA（CPU smoke 用 FP32）。
- 输入固定 RGB 256×256、范围 `[-1,1]`；resize 用 Pillow Lanczos 并写入 config。
- `scaling_factor` 严格来自 VAE config（`sd-vae-ft-mse` 默认 `0.18215`），encode/decode 两侧都使用。
- 所有指标比较 `source_preprocessed` 与 reconstruction（不把 resize 误差算给 VAE）。
- seed `20260901`，开启可行的 deterministic 设置。

## 32 张样本选择

复用 `scripts/select_phase30_coordinate_audit_ids.py` 的 pose-stratified 逻辑
（`select_pose_stratified`），从 `phase3_gaze_coordinate_candidates.csv` 中
`split=validation` 且 `status=candidate_unvalidated` 的行选 32 个唯一 ID。

程序化断言：32/32 ∈ validation_ids；与 fixed_test_ids 交集=0；ID 唯一；重跑
得到相同 ID 顺序与相同文件 SHA256（selection_hash `6fcabe54…0591`）。**不因为
gaze coordinate 未批准而排除样本**；候选 manifest 只用于 validation 与 pose
分层，不使用 head-local gaze 作真值。

该选择器不使用随机抽样；`selection_seed=20260901` 仅作为协议兼容元数据，实际
策略记录为 `deterministic_pose_extrema_no_rng`。候选不足 32 时必须 fail closed。

## 指标与 evaluator 规则

- RGB PSNR、SSIM、LPIPS（`lpips==0.1.4`，alex backbone；可用）。
- ArcFace：insightface `buffalo_l`，`det_size=640`，`det_thresh=0.1`（默认，非 0.5）。
  对 source_preprocessed 与 reconstruction 分别检测，成功才计算 cosine，
  否则 cosine 为空并记录双方 status；CSV 另记 source/reconstruction 检测分。
- DECA head pose：对两张图用**同一 FAN 检测 + `crop_to_tensor` 裁剪**（复用
  `phase2.run_fixed_external_deca`，`deca_preprocess=fan`，无 whole-image fallback）
  重新预测，计算 global rotation 的 geodesic angle；不与历史 source MAT 直接比较。
- L2CS：对两张图用同一入口重新预测，计算 camera-frame gaze angular difference；
  `pred()` 返回 `((pitch,yaw), status)`，无脸时返回 `((None,None),"no_face_detected")`，
  不 unpack 崩溃（不 0 填充）。

坐标边界（head-local gaze convention 未批准）：

- 只输出 `gaze_camera_delta_deg`，标注 `diagnostic_only`；
- `gaze_head_delta_deg` 必须为空；
- 不得把 camera-frame gaze delta 写成 eye-in-head gaze 保持；
- 不得签发 coordinate approval；
- evaluator 缺失/失败不阻断 VAE 重构，但计入完整 32 样本分母。

## 结果（2026-09-01，det_thresh=0.1）

| evaluator | 成功 / 失败 | 覆盖率 |
|---|---|---|
| VAE | 32 / 0 | 1.0 |
| ArcFace | 31 / 1 | 0.96875（source 0.96875 / reconstruction 1.0 / pair 0.96875）|
| DECA（fan）| 32 / 0 | 1.0 |
| L2CS | 32 / 0 | 1.0 |

| 指标 | count | mean | median | p05 | p95 |
|---|---|---|---|---|---|
| PSNR (dB) | 32 | 31.379 | 31.259 | 29.835 | 33.003 |
| SSIM | 32 | 0.8770 | 0.8781 | 0.8441 | 0.9072 |
| LPIPS | 32 | 0.02204 | 0.02030 | 0.01536 | 0.03259 |
| ArcFace cosine | 31 | 0.92095 | 0.91942 | 0.90125 | 0.94295 |
| head pose delta (°) | 32 | 0.657 | 0.468 | 0.152 | 1.016 |
| gaze camera delta (°) | 32 | 2.885 | 2.456 | 0.574 | 6.363 |

- GPU 峰值 5648.7 MB，wall 107.6 s；单张 runtime median 0.61 s / p95 1.17 s。
- `head_local_gaze_not_evaluated=true`；`gaze_coordinate_status=candidate_unvalidated_diagnostic_only`。
- 结论：VAE round-trip 对 identity 影响很小（ArcFace cosine 均值 ≈0.921、31/32 检出），
  对 head pose 影响 ≈0.66°、camera-frame gaze ≈2.89°，属可接受的 reconstruction anchor 漂移。

## 产物（`results/phase30_20260901/vae_roundtrip_audit/`）

```
selection/  source_preprocessed/  reconstructed/  contact_sheets/contact_sheet_all32.png
vae_roundtrip_metrics.csv  vae_roundtrip_summary.json  vae_roundtrip_failures.csv
artifact_hashes.sha256  config.json  exact_command.txt  environment.json
```

summary 以完整 32 为分母，含 VAE/evaluator 成功失败与覆盖率、各指标
count/mean/median/p05/p95、CUDA 峰值、wall time、单张 median/p95、fixed-test
overlap、模型 revision/hash、split/selection hash、`deca_preprocess`、
`arcface_det_thresh`，并显式 `head_local_gaze_not_evaluated=true`。contact sheet
是**一张** `label|source|reconstruction|abs_diff` 全 32 样本长图（非 32 个 panel），
每行标注 ID、cosine、head delta、camera-gaze delta 与失败状态。

`artifact_hashes.sha256` 在全部产物写出后递归生成（selection、config、metrics、
summary、failures、contact-sheet manifest、VAE snapshot），并排除自身。
`environment.json` 记录 transformers/accelerate/lpips 版本（未安装记为 null）。
resume 同时核对 selection、base manifest、VAE snapshot 内容哈希、scaling factor、
分辨率、dtype 与 evaluator 配置；即使路径不变，权重内容改变也会拒绝复用旧结果。

## 依赖

新增 `phase3/requirements_phase3.txt`（diffusers==0.40.0 / safetensors==0.8.0 /
huggingface-hub==1.29.0 / lpips==0.1.4），不改 PyTorch/CUDA、不改 Phase1/Phase2
requirements。已确认 lpips 安装不改变 torch 2.11.0+cu128 / torchvision 0.26.0+cu128。
transformers / accelerate 未安装（environment.json 记 null）。VAE 权重本地 snapshot
在 `models/phase3/sd-vae-ft-mse`（SHA256 已校验，`/models/` 已加入 .gitignore）。

## 运行

```powershell
$py = D:\face_standardization_project\.venv\Scripts\python.exe
$py -m py_compile phase3\audit_vae_roundtrip.py tests\test_phase3_vae_audit_protocol.py
$py -m tests.test_phase3_vae_audit_protocol
$py -m phase3.audit_vae_roundtrip --dry-run ...        # 不加载 VAE
$py -m phase3.audit_vae_roundtrip ...                  # 2-smoke → 32 full
```

## 边界

- 不使用 775 fixed test / 375 external / rescue。
- 不开始 Face Adapter、Eye Adapter、U-Net 或任何训练循环。
- smoke 通过 ≠ Phase3.1 完成。
