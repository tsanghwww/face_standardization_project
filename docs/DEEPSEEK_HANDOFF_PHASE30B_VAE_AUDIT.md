# DeepSeek 交接：Phase3.0B VAE Round-Trip Audit

将下面整段内容作为任务提示词交给 DeepSeek。执行机器是 `win-lenovo`（RTX 5060 Laptop 8GB），项目目录是 `D:\face_standardization_project`。

---

你正在接手 `face_standardization_project` 的 Phase3.0B。请完成 frozen latent-diffusion VAE 的 round-trip audit 工程实现与 32 张 validation smoke。这个任务只建立生成器之前的 VAE 基线，不训练 U-Net、Adapter 或任何 evaluator。

## 1. 基线与工作规则

- 仓库：`D:\face_standardization_project`
- 分支：`main`
- 必须从提交 `bde1603f42e66a1b53948613c911e61eaae47207` 开始；先运行 `git rev-parse HEAD` 和 `git status --short`。
- 不得切换或合并 `feature` 分支。
- 不得修改、删除或暂存现有无关文件：
  - `single_test_inputs/IMG_6033_facecrop.jpg`
  - `tools/aflw2000-3d.torrent`
  - `tools/screening_p975_p95_summary.json`
- 不要 commit，不要 push。完成后只提交报告，由 Codex 审查、修复和提交。
- 使用 `D:\face_standardization_project\.venv\Scripts\python.exe`。
- 结果只能写到 `results\phase30_20260901\vae_roundtrip_audit\`，不得覆盖 Phase1、Phase2、Phase2.1 或 Phase3.0 已有产物。
- 775 fixed test、375 external 和 rescue 路径全部禁止使用。只能从 1,440 validation 中抽取 32 张。
- 缺失和失败必须保留为空值及显式 status，禁止用 0 填充。

开始前阅读：

- `docs/PHASE3_IMPLEMENTATION_PLAN.md`，尤其 3、5.2、5.3、8、12 节；
- `docs/PHASE3_LATENT_DIFFUSION_TRAINING_PLAN.md`；
- `docs/PHASE30_CONDITION_AND_COORDINATE_AUDIT.md`；
- `tests/test_phase30_protocol.py`。

## 2. 先做只读预检

把结果写入最终报告：

1. GPU、CUDA、PyTorch 版本和峰值可用显存；
2. `diffusers`、`transformers`、`accelerate`、`safetensors`、`lpips` 是否安装及版本；
3. Hugging Face cache 中是否已有可用 `AutoencoderKL`；
4. 当前已有 ArcFace、DECA、L2CS 的实际入口、权重路径和可复用函数；
5. 下列输入文件是否存在，并记录 SHA256：
   - `results\phase30_20260901\splits\validation_ids.txt`
   - `results\phase30_20260901\splits\fixed_test_ids.txt`
   - `results\phase30_20260901\gaze_candidates\phase3_gaze_coordinate_candidates.csv`
   - Phase1/base manifest，需自行从既有脚本或文档解析其真实路径。

当前已知环境中 `diffusers=False`、`transformers=False`。若仍然缺失，允许在项目 `.venv` 中安装 VAE 审计所需的最小依赖，但必须：

- 不更换现有 PyTorch/CUDA；
- 记录精确安装命令和最终版本；
- 将 Phase3 新依赖写入新增 `phase3/requirements_phase3.txt`，使用本次实际验证的精确版本；
- 不修改 Phase1/Phase2 requirements；
- 若依赖冲突或模型下载需要登录，停止并报告，不要绕过认证或替换成不明模型。

## 3. 冻结的 VAE 协议

默认模型使用公开的 `stabilityai/sd-vae-ft-mse`，通过 `diffusers.AutoencoderKL.from_pretrained()` 加载。模型 ID 必须做成 CLI 参数 `--vae-model`，默认值为上述 ID；同时支持 `--vae-path` 指向本地 snapshot。

模型必须：

- `eval()`；
- 所有参数 `requires_grad=False`；
- FP16 CUDA 推理；CPU smoke 可使用 FP32 fake/tiny VAE；
- 输入固定为 RGB、256 x 256、范围 `[-1,1]`；
- 将原图明确 resize 成 256 x 256 后保存为 `source_preprocessed`，所有图像指标和 evaluator 都比较该图与 VAE reconstruction，避免把 resize 误差归因于 VAE；
- resize 使用固定的 Pillow Lanczos，并在 config 中记录；
- encoder 使用 posterior `mode()`，禁止随机 `sample()`；
- 严格使用 VAE config 中的 `scaling_factor`：

```python
latent = vae.encode(x).latent_dist.mode() * vae.config.scaling_factor
reconstruction = vae.decode(latent / vae.config.scaling_factor).sample
```

- 输出 clamp 到 `[-1,1]` 后再映射到 uint8；
- seed 固定为 `20260901`，并启用可行的 deterministic 设置；
- 记录模型 ID、本地 snapshot 路径或 resolved revision、config SHA256、权重文件 SHA256 清单。若整个权重过大，至少记录 snapshot commit/revision 和 snapshot 内所有模型文件的逐文件 SHA256，不得只写模型名。

## 4. 32 张样本选择

新增或复用确定性选择逻辑，从：

`results\phase30_20260901\gaze_candidates\phase3_gaze_coordinate_candidates.csv`

筛选 `split=validation` 且 `status=candidate_unvalidated`，按现有 `scripts/select_phase30_coordinate_audit_ids.py` 的 pose-stratified 逻辑选择 32 个唯一 ID，seed/排序必须确定。输出：

- `selection\vae_audit_ids.txt`
- `selection\vae_audit_selection.json`

必须程序化断言：

- 32/32 都属于 `validation_ids.txt`；
- 与 `fixed_test_ids.txt` 交集为 0；
- ID 唯一；
- 重跑得到相同 ID 顺序和相同文件 SHA256。

不要因为 gaze coordinate 仍未批准而排除这些样本。这里只用候选 manifest 做 validation 与 pose 分层，不使用候选 head-local gaze 作为训练真值。

## 5. 需要实现的文件

新增：

1. `phase3/__init__.py`
2. `phase3/audit_vae_roundtrip.py`
3. `phase3/requirements_phase3.txt`
4. `tests/test_phase3_vae_audit_protocol.py`
5. `docs/PHASE30B_VAE_AUDIT.md`

可在必要时对 `scripts/select_phase30_coordinate_audit_ids.py` 做向后兼容的小改动，但优先复用，不要复制另一套不一致的抽样器。不得开始 Face Adapter、Eye Adapter、U-Net 或训练循环实现。

`phase3/audit_vae_roundtrip.py` 至少支持：

```text
--base-manifest
--validation-ids
--fixed-test-ids
--selection-count 32
--selection-seed 20260901
--vae-model stabilityai/sd-vae-ft-mse
--vae-path
--resolution 256
--device cuda
--dtype fp16
--arcface-mode off|existing
--deca-mode off|existing
--l2cs-mode off|existing
--out-dir
--resume
--dry-run
```

`--dry-run` 不加载大模型，只验证 join、split、路径、选择、输出 schema。`--resume` 必须按逐样本状态安全恢复，不能重复覆盖成功样本或把旧失败误当成功。

## 6. 指标与 evaluator 规则

每张样本先计算：

- RGB PSNR；
- SSIM；
- 可选 LPIPS。若未安装，留空并标记 `not_available`，不得阻断主审计。

尽量复用现有 evaluator，不要重新写一套相互矛盾的人脸预处理：

- ArcFace：分别检测 `source_preprocessed` 与 reconstruction，成功时计算 cosine；任一检测失败则 cosine 为空，记录双方 status。
- DECA head pose：必须对 `source_preprocessed` 与 reconstruction 使用同一检测、裁剪和 DECA 配置重新预测，再计算 global rotation 的 geodesic angle。不要把历史 source MAT 与新 reconstruction 直接比较，因为预处理域不同。
- L2CS：对两张图使用同一入口重新预测，计算 camera-frame gaze angular difference。

坐标边界：当前 head-local gaze convention 未批准。因此：

- 本阶段只能输出 `gaze_camera_delta_deg`，字段和报告必须写明 `diagnostic_only`；
- `gaze_head_delta_deg` 必须为空；
- 不得把 camera-frame gaze delta 写成 eye-in-head gaze 保持；
- 不得签发 coordinate approval；
- evaluator 缺失或失败不应阻止 VAE 图像重构，但必须计入完整 32 样本分母。

建议的逐样本 CSV 至少包含：

```text
image_id,split,source_image,source_preprocessed,reconstruction,
vae_status,failure_reason,psnr_rgb,ssim_rgb,lpips,lpips_status,
arcface_source_status,arcface_recon_status,arcface_cosine,
deca_source_status,deca_recon_status,head_pose_delta_deg,
l2cs_source_status,l2cs_recon_status,gaze_camera_delta_deg,
gaze_head_delta_deg,gaze_coordinate_status,
runtime_seconds,gpu_peak_mb
```

`gaze_coordinate_status` 固定记录当前事实，例如 `candidate_unvalidated_diagnostic_only`。

## 7. 产物

写到：

`results\phase30_20260901\vae_roundtrip_audit\`

结构至少为：

```text
selection/
source_preprocessed/
reconstructed/
contact_sheets/
vae_roundtrip_metrics.csv
vae_roundtrip_summary.json
vae_roundtrip_failures.csv
artifact_hashes.sha256
config.json
exact_command.txt
environment.json
```

summary 必须以完整 32 为分母，包含：

- VAE 成功/失败数与覆盖率；
- evaluator 各自成功/失败数与覆盖率；
- PSNR、SSIM、LPIPS、ArcFace cosine、head pose delta、camera gaze delta 的 count/mean/median/p05/p95；
- CUDA 峰值显存、总 wall time、单张 median/p95；
- fixed-test overlap；
- 模型 revision/hash 与 split/selection hash；
- 明确写出 `head_local_gaze_not_evaluated=true`。

contact sheet 每格显示 source、reconstruction、absolute difference，并标注 ID、ArcFace cosine、head delta、camera-gaze delta 和失败状态。不要只挑成功样本。

## 8. 测试要求

`tests/test_phase3_vae_audit_protocol.py` 必须 CPU-only、无需联网、无需真实 VAE 权重。用 fake VAE/evaluator 或注入接口验证至少以下协议：

1. 只接受 validation ID，发现 fixed-test ID 时 fail closed；
2. 32 ID 唯一且确定性重跑 hash 一致；
3. posterior 使用 `mode()` 而不是 `sample()`；
4. scaling factor 在 encode/decode 两侧正确使用；
5. 缺失 evaluator 结果保留为空，不填 0；
6. head-local gaze 在 coordinate 未批准时保持为空；
7. VAE/所有 evaluator 均 frozen；
8. resume 不覆盖已成功样本；
9. summary 的总分母仍为完整选择集；
10. dry-run 不尝试下载或加载 VAE。

运行并报告：

```powershell
$py = D:\face_standardization_project\.venv\Scripts\python.exe
& $py -m py_compile phase3\audit_vae_roundtrip.py tests\test_phase3_vae_audit_protocol.py
& $py -m tests.test_phase3_vae_audit_protocol
git diff --check
```

若 `tests` 不是 package，可按仓库现有方式直接运行测试文件，但报告精确命令。

## 9. 执行顺序

1. 只读预检。
2. 实现代码、文档和 CPU protocol tests。
3. 先跑 `--dry-run`。
4. 先做 2 张 CUDA VAE smoke，确认图像非空、范围正确、显存合理。
5. 再跑 32 张 VAE reconstruction。
6. 再复用 ArcFace/DECA/L2CS 评估；若某 evaluator 无可靠复用入口，保留空值并清楚报告，不要临时伪造。
7. 生成 contact sheet、summary 和 hash。
8. 停止，不进入 500-step overfit，不训练任何模型。

## 10. 最终报告格式

请给出：

1. 基线 commit、git status 与环境/依赖；
2. 新增/修改文件逐项说明；
3. 32 张选择结果、split/hash/重复性验证；
4. VAE 模型 ID、revision、权重/hash、预处理和 latent 公式；
5. VAE 与三类 evaluator 的成功/失败/覆盖率；
6. 每项指标 count/mean/median/p05/p95；
7. GPU 峰值、耗时；
8. 测试逐项结果；
9. `git diff --stat` 和 `git status --short`；
10. 完整复现命令；
11. 尚未完成、不能声称和需要 Codex 决定的事项。

任何真实数据、依赖或 evaluator 不满足时，保留已完成的工程骨架并如实报告。不得把 smoke 通过写成 Phase3.1 已完成。

---
