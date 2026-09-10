# Phase3.1e：无参数更新梯度审计与外部训练数据方案

## Material Passport

- 日期：2026-09-09；执行机器：win-lenovo，RTX 5060 Laptop 8 GB。
- GitHub、本地与远端基础代码：`0103245fecb34eaa951623f6ed603b5755b05de4`。
- 已执行：固定 checkpoint 分项梯度审计、原 checkpoint 对照、联合调权局部方向检查。
- 未执行：optimizer 更新、validation/test 性能评估、外部数据训练或扩充。
- 验证边界：审计实际执行成功；调权后的训练效果仍为 UNVERIFIED。
- 原始记录位于 Lenovo `D:\face_standardization_project\results\phase31e_*_20260909`；统计副本同步到 Mac 同名 results 目录。

## 结论

目前不能将失败简单归因于 Face Adapter 无法反传。四个尺度都有有限、非零的分项梯度。主要证据指向几何监督主导共同标准化响应，source 重构约束受到压制，而 target/source 条件之间的差分梯度太弱。数据域扩展值得做，但单纯增加外部人脸不能修复这个目标函数问题。

优先建议：保持现有权重与结果作为对照，先降低 geometry 总权重并约束 source 侧退化，再验证条件对应关系；不要直接加长当前训练或一开始同时修改数据、损失、结构。

## 1. 审计协议与完整性

1. 对 corrected geometry-ranking step-64 checkpoint 做 56 个审计条件：全部 32 张 train 图在 t=250；按固定顺序每隔 4 张选 1 张，共 8 张，另测 t=100/400/800。
2. 原 source-reconstruction checkpoint 在同一 8 张 train 图、t=250 上作对照。这 8 张属于 Phase3.1d 的优化样本，不一定属于原 Phase3.1a 的 32 张优化样本；本报告不称其为验证集。
3. corrected checkpoint 同一 8 张、t=250 补做 12 组联合权重局部方向分析。
4. 同一 image ID 的 latent/noise/identity 在 source/target 两侧固定。梯度项为 source epsilon MSE、target pose、expression、landmark，以及 paired ranking。
5. 两侧分别重建计算图；hinge 激活时严格组装 `g_rank = g_Dtarget - g_Dsource`，不对 negative detach。CPU/实际运行环境测试确认与联合 autograd 的 active/inactive hinge 导数一致。
6. 采用 FP16 UNet autocast、FP32 VAE/DECA，梯度乘 128 再除回，以保持与训练 GradScaler 数值尺度接近。没有 optimizer，没有 clipping 操作，没有 `.backward()` 写入参数 `.grad`。
7. 三次运行均完成，adapter+UNet、VAE、DECA 的全部 state_dict 前后 SHA256 相同，输入 checkpoint 文件哈希相同；optimizer steps=0。峰值已分配显存 6672.08 MiB，低于 7.2 GiB 预算。

审计的 source MSE 与 geometry 在相同 t 上比较，用于消除噪声水平混杂；原训练 source t 来自全区间，因此这些范数不等于重放原训练每个 minibatch。此处是 one-step x0 诊断，不是 DDIM 完整轨迹或图像身份评估。没有强制确定性 CUDA kernel，重跑梯度数值存在浮点差异，不声称位级重现。

## 2. corrected 64-step：梯度结果

以下均为同一 t 内按 image ID 的中位数；不同 t 不能当成额外独立样本合并检验。

| t | 图像数 | source 梯度范数 | geometry 加权范数 | ranking 加权范数 | ranking/geometry 中位比例 |
|---:|---:|---:|---:|---:|---:|
| 100 | 8 | 0.2260 | 41.2590 | 0.01234 | 0.0299% |
| 250 | 32 | 0.2207 | 65.0565 | 0.02647 | 0.0368% |
| 400 | 8 | 0.1468 | 86.8281 | 0.05014 | 0.0449% |
| 800 | 8 | 0.02339 | 206.3351 | 0.13714 | 0.0727% |

这里采用原权重 source=1、geometry=1、ranking=0.1。t250 下 geometry 内部加权分项梯度中位数为 pose 29.92、expression 38.01、landmark 26.06；不是某个分项单独大了几百倍，而是 geometry 这整个目标组对 source 的尺度失衡。

- t250 source/geometry 梯度余弦中位数 **-0.5749**，**30/32** 为负。冻结身份分支不等于输出身份被保护；可训练 Face Adapter 仍能改坏重构。
- target/source 归一化距离梯度余弦中位数 **0.99118**；ranking 差分范数 / 两侧范数之和仅 **0.0688**。这与两种条件受到近似共同更新的解释一致，但不是“结构必然无效”的证明。
- t250 的 32 个 hinge 全部激活。ranking 弱不是因为 margin 已经满足。
- 四个尺度的 weighted ranking 梯度中位数分别为 0.00842、0.01803、0.01575、0.00297，均非零。
- 原 global clip=1 对应的局部总梯度缩放中位数约 0.0154。统一缩放不会修复分项方向冲突或相对占比失衡。实际 AdamW 的历史矩也会影响更新，不能将范数比例直接称为参数更新份额。
- t800 几何梯度很大，但 decoded RGB 的 clamp 饱和比例中位数为 0，不能把问题直接归因于 clamp 截断。高噪声 one-step 估计的语义有效性仍缺证据；“DECA 有限输出”不是人脸或姿态可信的充分条件。

## 3. 与原重构 checkpoint 的同图对照

8 张固定 train 图、t250：

| 指标 | 原 source-reconstruction checkpoint | corrected geometry step64 |
|---|---:|---:|
| source epsilon MSE 均值 | 0.20063 | 0.24770 |
| target arm 姿态误差均值 | 25.3639° | 17.5044° |
| source arm 到同一 target 的姿态误差均值 | 25.3726° | 17.5031° |
| target/source 距离梯度余弦中位数 | 0.98984 | 0.99215 |

几何训练确实使输出更接近 canonical target，但 source/target 两个条件几乎一起改善，重构 MSE 上升约 23.5%。这支持“共同标准化响应增强、条件对应关系仍未学好”的判断。

原报告 t250 的 target-source 姿态差为 +0.00503°；本次带梯度路径复测约 +0.00635°，方向一致，均远小于项目 1° 的工程门槛。这里不能把微小数值差异解读为控制成功或失败显著性。

## 4. 无更新调权检查

对固定 checkpoint 的参数梯度线性组合计算单位负梯度方向导数。负导数仅代表局部一阶下降方向，不是实际一步 AdamW、更不是 64-step 收益。8 张为探索性 train 子集，不用于对 validation 调参。

| geometry 总权重 | ranking 权重 | target 距离下降 | target-source gap 下降 | source MSE 下降 | 三项同时满足 |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.1 | 0/8 | 6/8 | 8/8 | 0/8 |
| 0.003 | 0.1 | 7/8 | 8/8 | 8/8 | 7/8 |
| 0.003 | 1.0 | 6/8 | 8/8 | 8/8 | 6/8 |
| 0.01 | 0.1 | 8/8 | 5/8 | 0/8 | 0/8 |
| 0.01 | 1.0 | 8/8 | 8/8 | 1/8 | 1/8 |

初始 geometry 分项权重仍为 `(1,75.044619,148.788535)`。上述 8 张结果只用于提出候选，不是冻结后的最终参数。

随后在**原 source-reconstruction checkpoint** 上完成了全 32 张、固定 `t={100,250,400}` 的 96 条 source/target 配对复核，并增加 source 输出回到 source 自身几何的梯度项。网格同时检查 `geometry_weight={0.001,0.003,0.01}`、`ranking_weight={0.1,0.3,1.0}`、`source_geometry_ratio={0.1,0.3,1.0}`。四项局部导数必须同时为负：target 绝对距离、target-source gap、source epsilon MSE、source 自身几何距离。

最佳组合为：

```text
source epsilon weight = 1.0
geometry_loss_weight = 0.003
source_geometry_ratio = 0.3
ranking_loss_weight = 1.0
```

| t | 图像数 | 四项同时改善 | target 绝对误差改善 | gap 改善 | source MSE 改善 | source 自身几何改善 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 32 | 26 | 30 | 32 | 32 | 28 |
| 250 | 32 | 24 | 31 | 32 | 31 | 25 |
| 400 | 32 | 23 | 31 | 32 | 30 | 25 |
| 合计 | 96 | **73** | **92** | **96** | **93** | **78** |

因此，最初 `0.003/0.1` 候选经完整复核后被 `geometry=0.003, source_ratio=0.3, ranking=1.0` 取代。这个结论仍只是固定 checkpoint 上的局部一阶方向，不等价于 AdamW 更新后的实际收益；下一步只能进行有界 smoke 和 train-only 训练，不能据此进入 validation。

单纯将原 ranking 权重放大 1000 倍，在 t250 的 32 张上仍有 30 张的局部方向使 source MSE 上升。在其他 t 还存在 source target-distance 上升的情形。ranking 可以靠恶化 negative 来改善 gap，因此必须同时要求 target 绝对误差降低、source 自身目标/重构不退化。

## 5. 应如何修改下一轮实验

按顺序进行，每次保留同预算对照：

1. **先校准损失组的梯度尺度。** 全 32 张与固定 t100/250/400 的复核已完成；下一轮冻结 `lambda_g=0.003, source_ratio=0.3, lambda_r=1.0` 作为有界候选。若采用归一化 D 替代现 geometry_total，需要重新换算和审计，不能照搬 0.003。
2. **把 source 条件也绑定到 source 自身几何目标。** 可增加 `D(G(output_source), y_source)` 并审计其梯度。target 条件对应 y_target，source 条件对应 y_source，形成明确的条件对应关系。只让两个输出都朝 y_target 改善会鼓励共同 canonicalization。
3. **处理 ranking 的 negative 退化通道。** 将 source 侧 stop-gradient 或使用固定参考可以作为单独消融，但 detach 本身不能证明读取几何；必须保留 source 自身误差与绝对 target 误差约束。不要仅凭 gap 判成功。
4. **先收窄几何监督 t 范围到待核验的 100–400。** t800 目前只有数值有限性，缺独立有效性支持。100–400 也不是最终选择，应记录同图输出质量、几何可信性与 VAE anchor 偏差。
5. **补充分离至少 10° 的同身份 counterfactual 条件。** 已实现 train-only `-10°/+10°`（两端相隔 20°）条件构建器与独立方向/排序审计器。训练前后都必须复用同一 latent/noise/timestep/identity；输出相对旋转在目标相对旋转轴上的投影必须为正，并且 own-target 总误差必须小于 crossed-target 总误差。只用接近正脸的目标，很容易学到数据集级正脸偏置。
6. **只有上述 train-only 检查通过才做下一次有限更新。** 可从原重构 checkpoint 开始同预算试验，避免将已损伤重构的 geometry checkpoint 当作唯一初始化。需要重新校准初始化处的梯度尺度。本轮没有执行任何更新。
7. **如果平衡后仍不能读条件，再改注入结构。** 当前六通道条件先从 256 缩到 32，再逐级到 16/8/4；细 landmark 信息可能被削弱，这是结构假设而非本审计结论。候选包括先高分辨率编码再下采样、target-source residual 或多尺度空间注入；先用残差敏感性和有分离度的几何条件验证。

## 6. 数据边界

本轮不扩充外部训练数据。样本多样性不足保留为论文实验局限；当前实验只修正目标函数、约束条件对应关系，并使用既有 train-only 32 张进行有界因果诊断。任何 COFW/300W-LP 候选准备均不属于本轮提交，也不得写入 canonical train registry。

## 7. 交付与测试

- 审计入口：`python -m phase3.audit_geometry_gradients --run-dir <corrected-run> --out-dir <new-output>`。
- 联合权重小样本检查：同一命令加 `--only-anchor`；原重构对照另加 `--checkpoint <original-checkpoint>`。
- 反事实条件：`python -m scripts.build_phase31f_counterfactual_conditions ... --yaw-offset-deg 10`。
- 反事实输出审计：`python -m phase3.audit_counterfactual_geometry ... --timesteps 100 250 400`。候选 gate 在 post-training 结果生成前固定为：完整分母、方向与排序同时正确至少 75%、投影变化中位数至少 0.5°。它是本阶段工程闸门，不是通用阈值。
- `python -m tests.test_geometry_gradient_attribution` 已在 Lenovo 原环境通过：joint/sequential 排名梯度等价、inactive hinge、无参数修改、加权范数与零向量余弦。
- Python 语法检查和 Git whitespace 检查通过。新脚本/报告未提交或推送到 GitHub。

首轮审计代码的精确副本保存在本地 `results/phase31e_gradient_audit_20260909/audit_code_v1.py`；后续版本仅新增联合调权方向导数，原训练代码未修改。
