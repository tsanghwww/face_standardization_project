# Phase3.1a 首次训练闭环报告

## Material Passport

- 日期：2026-09-02。
- 执行机器：RTX 5060 Laptop，8GB；Tailscale 远程执行。
- 运行类型：train-only 小样本工程诊断，不是正式训练或最终消融。
- 结论：工程执行通过，身份重构质量未通过，几何分支贡献偏弱。
- 本文数字来自真实运行产物；下方“下一步”属于建议，不是已完成结果。

## 1. 数据与边界

之前 VAE 审计的 32 张属于 validation，继续保留作审计，未用于优化。
本轮从 8,160 个 train ID 重新选择 32 张，seed=20260902。

- 8,153 张具备本轮所需输入；7 张缺少输入，明确记录并排除。
- 排除 ID：625、3305、4420、4752、7861、8169、9248。
- 新选择与 validation/fixed test 的 ID 交集为 0。
- 新样本的几何缓存 32/32 成功，JSONL 为 train=32、val=0、test=0。
- 不使用 rescue，不生成未批准的 head-local gaze，不启用 gaze loss。
- ID 文件 SHA256：`f26f74bce3cfc2cdf40d88a468a49c35e3b52c21685e8a30532224dc59870e24`。

source reconstruction 使用 source RGB 与 source normal/depth/landmark/face mask。
Phase2 target 条件不参与本轮损失；没有把原图伪装成标准化 RGB 真值。

对 8368、3476、6466 重新运行 DECA `iscrop=False`，与保存参数比较：
shape/expression/pose/camera 最大绝对差约 1.61e-4，低于 1e-3 容差。
这支持这些样本使用整图几何坐标；不等于对全体样本或所有插值细节做了完整证明。

## 2. 实际训练内容

- Face Adapter：六通道输入、四尺度 zero-conv residual。
- Identity projection：512 维 ArcFace 到四个独立身份 token。
- Identity attention：复用 Diffusers IPAdapterAttnProcessor2_0，不与文本共用身份 K/V。
- 可训练参数合计：21,552,000。
- 原始 UNet 和 VAE 冻结；未训练 Eye Adapter、LoRA、pose/gaze intervention。
- 当前唯一训练目标：source reconstruction 的 epsilon MSE。
- 未启用 ArcFace identity outcome loss，ArcFace 仅作输入条件及训练后诊断。

micro-batch=1，gradient accumulation=8，LR=1e-4。
先跑 2 optimizer steps，再从 checkpoint 恢复到总计 64 steps。
累计 512 个 micro-batches，约为这 32 张样本的 16 次遍历。
这不替代设计中的 500-step overfit gate。

两步 smoke：28.19 秒。恢复至 64 步并采样的第二段：169.43 秒。
这两个 wall time 是独立调用时长，不包括数据准备、下载或后续 ArcFace 审计。
峰值已分配显存约 2,109.45 MiB，不代表显卡总占用，也不代表加入 outcome losses 后的显存需求。

## 3. 工程验证

- CPU 测试通过：split 隔离、缺失/重复/rescue 拒绝、source-only 输入、uint16 depth。
- 零初始化与原 UNet 输出一致；关闭全部条件可恢复 baseline。
- 非重入 activation checkpointing 下梯度有限；三个可训练组均更新。
- adapter state 保存/恢复测试通过；真实 optimizer checkpoint 恢复成功。
- 64 条 optimizer-step 日志连续，无缺失/重复。
- 原始 UNet 训练前后参数哈希完全相同：
  `7e05f5a29951e0f7c0e797d73c6f3a12e640b4c03e7891e17ebc15fe7e13ff93`。
- 32/32 DDIM PNG 存在，尺寸为 256×256，非空白。
- “32/32 文件生成”不等于“32/32 有效人脸”或“身份保持”。

## 4. 固定噪声诊断

同一批 train 图像、同一组噪声、timestep=250：

| 条件 | epsilon MSE |
| --- | ---: |
| 冻结 baseline，无新条件 | 0.273807 |
| 训练后的完整条件 | 0.174841 |
| 关闭 Face Adapter | 0.175050 |
| 关闭 Identity Adapter | 0.236459 |
| 两种条件一起错配到下一张 | 0.180744 |

完整条件较 baseline 下降约 36.14%。但关闭 Face Adapter 后几乎不变。
当前证据支持“身份分支主导这项去噪改善，几何分支贡献偏弱”，不支持“已学会三维标准化控制”。
这些是同一 checkpoint 的推理敏感性对照，不是重新训练的消融，也不是泛化评估。

## 5. 真实采样与身份检查

单步 x0 估计与 20-step DDIM 从纯噪声采样分别标注。
前四张目视检查发现外观偏移，首张存在明显多人伪影。
不能用单步 x0 的视觉改善代替完整采样质量。

ArcFace 沿用 buffalo_l、det_thresh=0.1、最大候选框协议：

| 比较 | 可计算数 / 总数 | 平均 cosine | 中位数 |
| --- | ---: | ---: | ---: |
| source → VAE anchor | 32/32 | 0.917026 | 0.920336 |
| source → DDIM | 23/32 | 0.113823 | 0.094599 |
| VAE anchor → DDIM | 23/32 | 0.118075 | 0.101045 |

- DDIM 检测不到脸：9/32，28.13%。
- DDIM 检测到多个候选脸：7/32；低阈值可能包含误检，不把每个候选都视为真实独立人脸。
- 多候选图仍按最大框计算，保留标记；缺失 cosine 为空，不填 0。
- 上述平均值基于各自可计算的样本，分母不同，不能当作完整配对差值检验。
- 未设置或校准身份判定阈值；这是相似度诊断，不是身份认证结论。

当前模型没有通过 Phase3.1 的身份重构质量检查。低 epsilon MSE 不足以证明身份保持。

## 6. 下一步

暂不直接升级到 8,160 张正式训练，也不只凭 loss 下降加长训练。

1. 补 source-latent img2img 基线，与现有纯噪声采样明确分开。在固定噪声和明确 strength 下检查身份/编辑幅度，而不是把低强度复制原图当作控制成功。
2. 诊断 Face Adapter 被忽略的问题：单独打乱几何和身份，查看分层残差幅度，并在 train-only 小实验中测试条件 dropout/空间遮挡；不能给 target geometry 配 source RGB 的冲突监督。
3. 身份损失若加入训练，需要可微且冻结的 estimator，现有 ONNX ArcFace 继续只作 evaluator；不能声称 ONNX cosine 提供训练梯度。
4. 在上述检查后，再决定是否运行完整 500-step overfit，并复测身份、姿态及生成失败。
5. 视线坐标仍待验证。Eye Adapter、gaze loss 与 head/gaze 干预仍关闭，不能声称视线解耦。

## 7. 产物位置

5060 项目：`D:\face_standardization_project`。

实验根目录：`results\phase31_train_smoke_20260902`。

- `selection\selection.json`、`train_smoke_ids.txt`。
- `condition_cache\phase3_condition_cache.csv`。
- `dataset\train.jsonl`。
- `alignment_audit.json`。
- `run\config_from_step_0.json`、`config_from_step_2.json`、对应 exact_command。
- `run\training_log.jsonl`、`checkpoint.pt`、`summary.json`。
- `run\run_audit.json`、`artifact_hashes.json`、`contact_sheet.png`。
- `identity_audit\identity_metrics.csv`、`summary.json`。

checkpoint SHA256：`1670f057111e7dc75a4c8c21446b70eafa402afc6b0fde807c09ea76ea21d35d`。
训练配置保存输入/模型/训练代码哈希；运行时 Git HEAD 为 4826db2 加本轮未提交实现，不能仅凭旧 HEAD 复现本轮代码。
