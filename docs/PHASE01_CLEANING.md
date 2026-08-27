# Phase 0/1 数据准备与数据清洗总结

- **Last Updated**: 2026-07-15（Phase 1 清冾；外部评估集补充至 2026-08-24）
- **适用范围**: Phase 0（样本集准备）+ Phase 1（特征提取与数据清洗），是 Phase 2（标准化条件训练）的输入
- **Canonical 产物**: `results/phase1_parity/phase1_master_manifest.csv`（10,000 行）+ `phase1_master_summary.json`

## 1. 样本集准备（Phase 0）

### 1.1 主数据集（训练/清洗对象）

| 项 | 值 |
|---|---|
| 来源 | StyleGAN2 生成人脸（yellow-stylegan2 变体） |
| 规模 | 10,000 张 PNG（`0.png`–`9999.png`），11.6 GB |
| 唯一 ID | 10,000 / 10,000 |
| 哈希 | 全部 10,000 张记录 SHA256（`image_sha256` 列） |
| 位置 | `archive/generated_yellow-stylegan2/`（+ `archive.zip` 备份） |

### 1.2 特征提取覆盖（全部成功）

| 特征 | 脚本 | 覆盖 | 说明 |
|---|---|---|---|
| DECA / FLAME 参数 | `run_deca_batch_params.py` | 10,000 / 10,000 | `DECA/results/archive_phase2_params/{id}/{id}.mat`，含 shape/expression/pose/tex/light/camera |
| L2CS-Net 视线 | `tools/run_l2cs_batch.py` | 10,000 / 10,000 | pitch/yaw/视线向量；Gaze360 权重 SHA256 已记录；标记 `rebuilt` |
| ArcFace 身份 | `tools/extract_arcface_embeddings.py` | 9,990 / 9,990 | main pass 9,968 + retry 22（详见 §3.4） |

### 1.3 外部评估集（仅用于 Phase 2 评估/鲁棒性测试，不进训练流）

下载于 2026-08-23（aria2），2026-08-24 解压验证完毕，见 `datasets/external/EXTERNAL_DATASETS.md`：

| 数据集 | 规模 | 用途 |
|---|---|---|
| WIDER FACE | train 12,943 / val 3,289 / test 16,160 | 姿态/遮挡/模糊鲁棒性（带 blur/pose/occlusion 属性） |
| 300W-LP | 61,225 张唯一（含 flip 共 122,450 jpg） | 大姿态对照 |
| AFLW2000-3D | 2,000 张裁剪图 + pose/pts68/reannotated/roi_box 标注 | 大姿态 3D 对齐/重建评估 |
| COFW Color | ~1,007 张真实遮挡彩色人脸 | 遮挡/landmark 失效测试 |
| SCFace | ⏸ 未下载 | 低清晰度监控模糊（需机构签署 release agreement，待用户决策） |

## 2. 数据清洗流程总览

```
10,000 源图 (SHA256 全记录)
   │ ① 眼部有效性标记（非破坏性，-10）
   ▼
9,990 eye-valid
   │ ② DECA 参数空间 Mahalanobis 筛查（p95 / p97.5）
   ▼
p95: Pass 9,500 / Warn 500      p97.5: Pass 9,750 / Warn 250
   │ ③ ArcFace 训练集划分（基于 p95 Pass）
   ▼
strict train 9,482  /  full train 9,499
   │ ④ Phase2 质量 manifest（BUG-003 修复后重建）
   ▼
10,000 全覆盖：high 2,129 / medium 7,871 / low 0（带 ArcFace 特征版）
```

## 3. 各清洗步骤结果统计

### 3.1 眼部有效性排除（eye-valid）

- 历史工作站遗留的 **10 个** 眼部无效 ID：`625, 3305, 3888, 4420, 4515, 4752, 7861, 8169, 8373, 9248`
- 处理方式：**非破坏性** —— 不删除文件，在 manifest 中标记 `eye_valid=false`（ID 清单存于 `configs/phase1_eye_invalid_ids.txt`）
- 有效集 = 9,990
- 注：这 10 张仍参与 DECA/筛查打分，其中 9 张落在 p95 Warn、1 张落在 p95 Pass

### 3.2 DECA 参数空间筛查（`tools/screening_deca_params.py`，2026-07-07）

**方法**：
1. 每张图拼 86 维参数向量 = expression(50) + pose(6) + camera(3) + light(27)；10,000 个 `.mat` 维度全部一致，**0 失败**（`fail_report.json` 为空）
2. 稳健化处理：先按中位数 + ridge 逆协方差算稳健 Mahalanobis 距离，剔除 D² 超过 χ²(86) 99.9% 分位（阈值 132.277）的 **8,106 个极端样本**（不参与参考分布拟合，但仍参与打分）
3. 用剩余 1,894 个样本拟合均值/协方差，对全部 10,000 张重算 Mahalanobis 距离 D2（D2² 对应 χ²(86) 得 p 值）
4. 按 D2 分位打标：`D2 > 分位阈值 → Warn`

**结果**：

| 分支 | 分位 | D2 阈值 | Pass | Warn | 产物 |
|---|---|---|---|---|---|
| p95 | 95.0 | 25.7156 | 9,500 (95.0%) | 500 (5.0%) | `results/screening_p95/` |
| p97.5 | 97.5 | 32.2329 | 9,750 (97.5%) | 250 (2.5%) | `results/screening_p975/` |

- 嵌套关系（`results/screening_threshold_benchmark/`）：250 个 p95∩p97.5 共同 Warn；250 个仅 p95 Warn；9,500 共同 Pass
- 每个分支附带：`screening_report.json`（10,000 行按 D2 排序）、`review_manifest.csv`（分层人工复核抽样：top20/边界/中心/随机）、`pass_images/` + `warn_images/` 副本、光照分析
- 光照分析（`lighting_summary.md`）：Warn 组左右光照不平衡略高于 Pass 组（lr_abs_diff 均值 23.56 vs 16.83），提示侧光/光照不均是 Warn 的常见因素之一

### 3.3 质量分数分支（`tools/screen_percentile.py` + 阈值基准）

- 该分支按 `quality_score`（landmark 几何 + 姿态/表情范数 + 完整度加权）的底部 5% / 2.5% 打 Warn，结果与 D2 分支一致（p95: 500 Warn；p97.5: 250 Warn）
- 全量 quality_score 分布：mean 0.3816，std 0.0166，min 0.3063，max 0.4526；p2.5 cutoff 0.3472，p5 cutoff 0.3537
- ⚠️ **历史注意**：该分支生成于 BUG-003 修复（kpt 坐标反归一化）之前，`landmark_score` 恒为 0（基准报告中 both_pass 组 landmark_score mean=0.0、out_ratio≈0.717）。**Canonical 清洗标签以 D2 筛查为准（不受该 bug 影响）**，质量分分支在修复后已重建（见 §3.5）

### 3.4 ArcFace 训练集划分

| 项 | 数量 | 定义 |
|---|---|---|
| main pass 成功 | 9,968 | det_size=640, det_thresh=0.1 |
| retry 恢复 | 22 | det_thresh=0.05；组成 = 17 个 p95 Pass + 5 个 p95 Warn |
| ArcFace 最终成功 | 9,990 | 全部 eye-valid 样本成功（10 个 eye-invalid 不要求） |
| **strict train** | **9,482** | eye-valid ∧ p95 Pass ∧ main-pass 成功 |
| **full train** | **9,499** | eye-valid ∧ p95 Pass（含 retry 恢复的 17 个） |

### 3.5 Phase 2 质量 manifest（BUG-003 修复后重建）

- 输入：10,000 个 DECA `.mat` + Phase 1 master manifest（ArcFace 字段）；脚本 `phase2/build_manifest.py`
- 质量标签规则：`quality_score ≥ 0.72 → high`；`≥ 0.45 → medium`；其余 `low`；`use_for_train = high ∪ medium`

| 版本 | high | medium | low |
|---|---|---|---|
| 带 ArcFace 特征（`bug003_fixed_arcface_ok`） | 2,129 | 7,871 | 0 |
| 不带 ArcFace（`bug003_fixed`） | 732 | 9,268 | 0 |
| 修复前旧版（`phase2_real_manifest`） | 0 | 1 | 9,999 ← landmark bug 导致质量塌缩，已弃用 |

### 3.6 Phase 2 评估集划分（`phase2/build_fixed_eval_split_v2.py`）

- **v1**（2026-08-24）：625 = xgb_high 100 + xgb_medium 100 + xgb_low 100 + base_hard_pose 50 + base_low_landmark 50 + wider_pose 75 + wider_occlusion 75 + wider_blur 75（`results/phase2_eval_fixed_20260824/`）
- **v2**（最终）：775 = v1 的 625 + **cofw_occlusion 75** + **aflw_large_pose 75**（`results/phase2_eval_fixed_20260824_v2/`）
  - v1 时 COFW zip 完整性校验失败未纳入；v2 重新解压后补上
  - 基础测试集 `base_test_ids.txt` = 400 个 ID（从 10K 中按 xgb 质量分层 + hard pose + low landmark 选出）
  - COFW 遮挡比例范围 0.41–0.86；AFLW yaw 范围 ±117.6°，按 |yaw| ≥ 45° 过滤

## 4. 清洗漏斗汇总

| 阶段 | 数量 | 累计保留率 |
|---|---:|---:|
| 源图（唯一 ID） | 10,000 | 100% |
| 眼部有效（eye_valid=true） | 9,990 | 99.9% |
| p95 Pass | 9,500（eye-valid 9,499） | 95.0% |
| ArcFace strict / full train | 9,482 / 9,499 | 94.8% / 95.0% |
| DECA / L2CS 全覆盖 | 10,000 | 100% |

- NEXT.md 验收标准全部满足：10,000 唯一 ID ✓、每行含清洗标签与 DECA 状态 ✓、L2CS 10,000 ✓、ArcFace strict/full 溯源分开报告 ✓、源图 SHA256 记录 ✓

## 5. 结论与注意事项

1. **清洗是非破坏性、可复现的**：所有排除都是标签化（eye_valid / Pass-Warn / use_for_train）而非物理删除；全部由脚本 + manifest + SHA256 支撑，可在 5060 上一键重建
2. **Canonical 标签链**：D2 筛查（不受 BUG-003 影响）→ master manifest → Phase 2 训练与评估划分，全程一致（p95 为 canonical 分支，p97.5 为基准对照）
3. **BUG-003 影响面**：仅影响启发式质量分分支（landmark_score 恒 0 → 质量塌缩），已通过 `bug003_fixed` / `bug003_fixed_arcface_ok` 重建；旧 `phase2_real_manifest` 弃用
4. **冗余与存储**：`screening_p95/` + `screening_p975/` 各含 10,000 张图像副本（合计约 23 GB），建议确认后归档（见 NEXT.md 待决事项）
5. **外部数据集只用于评估**：AFLW2000-3D / 300W-LP / WIDER FACE / COFW 不进入 Phase 1/2 训练流；SCFace 是否下载待用户决策

## 6. 关键产物索引

| 产物 | 路径 |
|---|---|
| Phase 1 master manifest | `results/phase1_parity/phase1_master_manifest.csv` |
| Phase 1 汇总 | `results/phase1_parity/phase1_master_summary.json` |
| 眼部无效 ID | `configs/phase1_eye_invalid_ids.txt` |
| ArcFace retry ID | `results/phase1_parity/arcface_retry_ids.txt` |
| p95 筛查 | `results/screening_p95/`（stats/report/fail/review/lighting） |
| p97.5 筛查 | `results/screening_p975/` |
| 阈值基准 | `results/screening_threshold_benchmark/` |
| Phase2 质量 manifest（修复后） | `results/phase2_manifest_bug003_fixed_arcface_ok/` |
| 评估集 v2 | `results/phase2_eval_fixed_20260824_v2/`（775 样本） |
