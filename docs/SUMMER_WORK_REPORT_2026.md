# 2026 暑期工作报告：基于 DECA 与 ArcFace 的人脸标准化研究

**报告周期：** 2026 年 7 月 5 日至 2026 年 8 月 27 日

**项目仓库：** [tsanghwww/face_standardization_project](https://github.com/tsanghwww/face_standardization_project)

**主要实验平台：** macOS 开发机、Windows RTX 5060 Laptop GPU 实验机

**当前基线：** `main@410b867`

## 1. 工作摘要

本工作研究如何利用 DECA 提取的人脸三维参数，在有限且相对标准化的训练数据条件下，生成更稳健的人脸标准化条件。系统以 DECA 的 expression、pose、camera 和 lighting 参数为主要输入与控制对象，以 ArcFace 身份相似度、头部姿态、gaze 变化和渲染成功率作为评估信号。

暑期工作从旧实验恢复、运行环境迁移和基础错误修复开始，随后完成了 Phase1 一致性恢复、Phase2 数据协议重建、外部困难样本引入、四组正式消融训练、固定测试集评估、rescue 敏感性实验、安全 Gate 校准，以及 Phase2.1 outcome-supervised 训练框架。

截至报告日期，Phase2 v1 的预定实验和可复现交付已经完成。实验表明，学习式标准化相较 hard-zero 基线能够带来小幅但可测量的身份、姿态和 gaze 改善；但当前 Gate 对不安全结果的预测能力不足，尚不满足进入 Phase3 或正式部署的条件。因此，项目当前位于 Phase2.1 的正式数据生成与结果监督训练阶段。

## 2. 研究背景与目标

训练数据主要来自 Kaggle/base 10K 人脸集合。该数据中的人脸通常较清晰、正面且关键点容易检测，难以覆盖真实环境中的大姿态、遮挡、模糊和检测失败情况。若仅将 DECA expression 和 pose 强制设置为零，系统容易出现以下问题：

- 输入特征提取失败时，整张脸可能产生严重重建或渲染错误。
- hard-zero 忽略不同样本的可标准化程度，容易造成身份信息损失。
- 模型可能只学习到接近归零的单一策略，而不是质量相关的渐进式标准化。
- 在标准训练样本上取得的结果无法说明模型对困难样本是否稳健。

为此，项目将 Phase2 的目标定义为：根据 DECA 参数和输入质量特征预测标准化条件及强度 $\alpha$，在改善姿态和 gaze 的同时尽可能保持 ArcFace 身份一致性，并对不安全或不可处理的样本进行拒绝。

## 3. 暑期工作时间线

### 3.1 7 月 5 日至 8 日：环境迁移与管线恢复

- 建立 DECA、ArcFace、数据筛选和条件生成的基础流水线。
- 将主要实验迁移到 RTX 5060 Windows 机器。
- 修复 PyTorch/CUDA 兼容问题，并迁移 CUDA rasterizer。
- 增加 Windows 构建与运行脚本、筛选工具和结果对比可视化。
- 恢复可用的 Stage1/Stage2/Stage3 模型、推理结果和 hard-zero 输出。
- 记录旧 2060 机器数据丢失带来的恢复边界，重新建立实验来源记录。

### 3.2 7 月 8 日：修复 DECA 关键点坐标错误 BUG-003

检查发现，DECA 投影关键点位于近似 $[-1,1]$ 的归一化坐标范围，而旧代码直接将其当作像素坐标计算 landmark quality。该错误会污染关键点分数、质量标签以及由此产生的 manifest。

修复后，关键点被正确反归一化到 224 像素空间。在 200 个样本上的检查结果为：

- landmark score 均值约为 `0.943`；
- 关键点越界比例为 `0`；
- BUG-003 修复前生成的相关质量结果被判定为不可直接复用。

### 3.3 7 月 15 日：恢复 Phase1 一致性

- 重建 Phase1 parity 流程和实验记录。
- 对数据字段、质量分数和下游接口进行统一。
- 将 `main` 确立为正式实验基线，`feature/*` 仅用于小组成员的隔离开发。

### 3.4 8 月上中旬：Phase2 设计与训练协议重建

- 将 Phase2 定义为冻结 DECA 后的条件生成任务。
- 输入由 DECA 参数和质量特征组成，实际基础输入维度为 99。
- 输出包含 expression、pose、camera、lighting 的标准化条件及强度 $\alpha$。
- 引入 DECA latent-space augmentation，在 expression、pose 等参数空间制造可控扰动。
- 将验证集增强关闭，normalizer 仅由未增强训练子集计算。
- 固定随机划分，保存 `train_ids.txt`、`val_ids.txt`、配置和精确命令。
- 将随机噪声与 smooth 正则限制在训练阶段，保证验证结果可重复。

### 3.5 8 月下旬：固定测试集与外部困难样本

固定测试集共 775 张，且不参与训练或 Gate 校准：

| 来源 | 数量 | 目的 |
| --- | ---: | --- |
| Kaggle/base | 400 | 覆盖 XGBoost high、medium、low 质量层 |
| WIDER Face | 225 | 大姿态、遮挡和模糊样本 |
| COFW Color | 75 | 遮挡与低 landmark 质量样本 |
| AFLW2000-3D | 75 | 大姿态样本 |
| **合计** | **775** | 独立固定评估 |

COFW 初始压缩包曾出现解压异常，之后删除问题文件，重新下载官方 COFW Color，并完成校验、解压和 `h5py` 环境配置。

在 base 10K 中排除 400 个固定测试样本后，剩余 9,600 个样本用于训练协议，其中 8,160 个用于训练，1,440 个用于验证，固定测试集与训练/验证集重叠为零。

## 4. Phase2 v1 实施与结果

### 4.1 正式模型与基线

正式训练了四个条件生成模型：

1. `full`：learned alpha、augmentation、Stage3、XGBoost/quality blend。
2. `no_alpha`：固定 $\alpha=1$，其余与 full 相同。
3. `no_augmentation`：关闭 latent-space augmentation。
4. `no_xgboost`：仅使用 heuristic quality，不使用 XGBoost。

`hard-zero` 是固定参数归零基线，不是第五个训练模型。最终渲染评估包含 original、hard-zero 和四个训练模型，共六种方法。

### 4.2 外部样本预处理

FAN/DECA 主路径在 375 个外部样本上的结果为：

- 成功：343；
- 失败：32，全部为 WIDER 的 `fan_no_face`；
- `fallback_used=true`：0；
- 32 个失败样本保留在 manifest，并计入总失败分母。

独立 rescue 路径采用整图 warp，技术上覆盖 375/375，但不写入主 `mat_path`，也不自动替代主推理。外部 ArcFace 在检测阈值 0.1 下成功 292/375。

### 4.3 XGBoost 质量模型

- 使用排除固定测试集后的 9,600 个样本训练。
- 建立 5-fold OOF manifest，避免训练集分数泄漏。
- 固定测试集预测覆盖 743/775。
- 未覆盖的 32 个样本全部来自上游 FAN/DECA 失败，不填入伪分数。
- 模型 SHA256：`d87dd136d157467ccf7491189337711702ce10e8638b855287ff24631d1735dc`。

### 4.4 主要消融结果

在 743 个主路径可用样本上，`full` 相对 hard-zero 的配对结果为：

| 指标 | 平均变化 | 95% bootstrap CI |
| --- | ---: | ---: |
| ArcFace cosine | `+0.00359` | `[+0.00146, +0.00574]` |
| Head pose | `-0.00525` | `[-0.00961, -0.00072]` |
| Gaze | `-1.96°` | `[-3.80°, -0.18°]` |

结果说明 full 模型具有小幅但可测量的总体收益，不过绝对 ArcFace cosine 仍然偏低：full 为 `0.3688`，hard-zero 为 `0.3652`。

其他消融发现：

- learned alpha 有小幅收益，`no_alpha` 基本接近 hard-zero。
- 当前 augmentation 未获得实验支持，`no_augmentation` 的 pose/gaze 结果反而更好。
- XGBoost 主要改变样本覆盖率，尚未形成充分的安全性依据。
- full 接受 631/775 个样本，覆盖率为 `81.42%`；`no_xgboost` 接受全部 743 个主路径可用样本。

### 4.5 Rescue 敏感性结果

在 343 个 FAN 主路径和 rescue 均可用的样本上：

- 主路径与 rescue 的 ArcFace cosine 均值约为 `0.6651`；
- gaze 差异均值为 `24.12°`，95 分位为 `80.90°`；
- expression RMSE 为 `0.1203`；
- pose L2 为 `0.1910`。

这说明整图 rescue 与 FAN 对齐结果存在显著域偏移，不能视为等价输入。

对 32 个 rescue-only 样本的独立分析显示：

- 16 个为不安全结果；
- 16 个安全但无效；
- 安全且有效为 0；
- 因此主路径的科学覆盖率保持为 743/775，即 `95.87%`。

### 4.6 Gate v1 结果

在 1,440 个 validation outcomes 中：

- unsafe：167；
- safe but ineffective：1,270；
- safe and effective：3。

Gate v1 的 AUROC 为 `0.6049`，AUPRC 为 `0.159`。虽然验证集上可在约 `74.31%` 覆盖率达到经验 10% 风险，但冻结后迁移到固定测试集时：

- 覆盖率：`58.68%`；
- accepted unsafe rate：`44.27%`；
- FAR：`60.69%`；
- FRR：`42.82%`。

因此 Gate v1 不具备部署条件，也不允许在 775 固定测试集上重新调整阈值。

## 5. Phase2.1：结果监督升级

Phase2.1 将标准化后的真实结果纳入训练和决策，监督目标包括：

- ArcFace 身份保持；
- pose 改善；
- gaze 变化；
- 渲染失败概率。

已完成的工程更新包括：

- 构建逐样本 outcome supervision manifest，缺失值显式保留并使用 mask。
- 建立 identity、pose、gaze 和 render-failure 四头 surrogate。
- 使用训练子集计算 surrogate normalizer，并将其保存到 checkpoint。
- 将 identity 目标改为相对 hard-zero 的 identity delta。
- 将冻结 surrogate 接入 condition generator，使 outcome loss 能对参数和 $\alpha$ 反向传播。
- 增加 outcome 数据与 condition-generator validation ID 的重叠检查。
- Gate 使用训练集 median imputation，并增加 18 个缺失指示特征，共 36 维。
- 将 Gate 决策输入与评估标签分离，避免部署时依赖事后结果。
- rescue audit 生成独立审计文件，但不改变主路径决定。
- calibration 使用单侧 95% Wilson 风险上界；无合格阈值时输出 `threshold=null`。
- Phase2.1 协议测试目前为 10/10 通过。

真实 smoke 实验得到：

| 项目 | 结果 |
| --- | ---: |
| Surrogate identity-delta MAE | `0.0105` |
| Surrogate pose-improvement MAE | `0.0547` |
| Surrogate gaze MAE | `13.92°` |
| Gate AUROC | `0.5898` |
| Gate AUPRC | `0.1821` |
| Gate Brier | `0.1031` |
| Gate ECE | `0.0177` |

当前没有阈值能够在单侧 95% 置信度下满足 10% unsafe-risk。最佳观察点覆盖率为 `41.34%`，经验 unsafe rate 为 `7.82%`，但风险上界仍为 `11.79%`。系统因此正确输出空阈值并拒绝部署。

## 6. 主要问题与研究认识

1. **训练数据覆盖不足。** Kaggle/base 数据过于标准，难以学习真实困难条件下的稳定策略。
2. **上游检测是关键瓶颈。** ArcFace、FAN 和 DECA 都可能在大姿态或遮挡下失败，不能只报告成功样本指标。
3. **Rescue 不是无损回退。** 整图 warp 虽提高技术成功率，但会改变 DECA pose、expression 和 gaze 分布。
4. **质量分数不等于结果安全。** 输入质量和 XGBoost 标签对实际 unsafe outcome 的判别力有限。
5. **数据增强需要重新设计。** 当前 latent augmentation 没有在正式消融中表现出稳定收益。
6. **安全且有效的正样本稀缺。** 1,440 个 validation outcomes 中仅 3 个同时满足安全和有效条件。
7. **固定测试集必须保持冻结。** 阈值必须在独立 hard calibration split 上确定，不能利用测试集反复调参。
8. **负面结果同样重要。** Phase2 v1 已完整执行，但结果说明系统尚未达到进入下游生成实验的安全标准。

## 7. 两台机器与可复现性

macOS 开发机主要负责代码审查、文档维护、Git 管理和结果整理；RTX 5060 Windows 机器负责数据存储、正式训练、批量渲染与指标计算。

截至报告日期，两台机器均位于 `main@410b867`。大型数据集、模型权重、渲染输出和私有图像不提交到 GitHub；仓库保留代码、协议、轻量图表、关键指标、精确命令和 SHA256 清单。

Phase2 v1 的正式交付包含：

- 6 张最终图表；
- 66 个关键产物 SHA256 记录；
- 训练与推理精确命令；
- 固定数据划分和实验配置；
- Phase2 最终实验报告。

## 8. 当前进度

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| 环境恢复与 Phase1 | 完成 | 运行环境、数据接口和 parity 已恢复 |
| Phase2 v1 工程与实验协议 | 完成 | 四组训练、固定评估、rescue、Gate 和报告均完成 |
| Phase2 v1 科学目标 | 部分达到 | 有轻微总体改善，但安全 Gate 不合格 |
| Phase2.1 工程框架 | 完成 | outcome surrogate、Gate、校准和防泄漏协议已建立 |
| Phase2.1 正式实验 | 待完成 | 仍需重新生成无泄漏 outcome 数据并正式训练 |
| Phase3/下游生成器 | 未开始 | 必须等待 Phase2.1 通过安全与覆盖率标准 |

若以“完成 Phase2 v1 实验和可复现报告”为标准，当前完成度为 100%；若以“得到可安全进入 Phase3 的标准化系统”为标准，整体研究进度约为 70%，当前瓶颈是 outcome 数据质量、Gate 判别能力与独立校准。

## 9. 下一阶段工作

1. 从 8,160 个训练 ID 生成新的 outcome-supervision 数据，禁止复用旧 validation outcomes 训练正式模型。
2. 对每个输入生成多个标准化候选，增加 outcome 分布多样性。
3. 收集真实渲染失败正样本，解决 render-failure head 单类问题。
4. 正式训练 Phase2.1 surrogate 和 condition generator，并进行 outcome loss 权重消融。
5. 改善 gaze 监督和特征表示，降低目前约 `13.92°` 的 surrogate MAE。
6. 扩展 Gate 特征，使其直接预测标准化后的 unsafe outcome，而不是依赖输入质量代理。
7. 建立独立、困难样本富集的 hard calibration split，并冻结 Gate 与阈值。
8. 仅在模型和阈值冻结后运行一次 775 fixed test，报告安全率、覆盖率和置信区间。
9. 增加感知质量或人工评估，确认 DECA 指标改善能否转化为实际下游生成质量。
10. Phase2.1 同时通过身份安全率、姿态改善和有效覆盖率标准后，再进入 Phase3。

## 10. 相关文档

- [Phase2 最终实验报告](phase2_final_20260827/PHASE2_FINAL_REPORT_20260827.md)
- [Phase2.1 协议](PHASE21_PROTOCOL.md)
- [Phase2 rescue 与 Gate 协议](PHASE2_RESCUE_GATE_PROTOCOL_20260826.md)
- [Phase1 parity](PHASE1_PARITY.md)
- [数据集说明](DATASET.md)
- [实验记录](EXPERIMENTS.md)

## 11. 总结

本阶段完成了从环境恢复、数据协议修复到 Phase2 完整实验闭环的建设，也通过固定测试集、外部困难样本、rescue 敏感性分析和独立 Gate 校准，识别出系统当前最重要的风险：标准化结果虽然有小幅平均改善，但输入质量特征不足以可靠判断单个样本是否安全。

项目没有用提高技术成功率来掩盖不安全样本，也没有在固定测试集上重新调阈值。Phase2.1 已将研究重点从“预测看起来合理的参数”推进到“直接监督并约束标准化后的身份、姿态、gaze 和失败结果”。下一阶段的核心任务是建立无泄漏且更丰富的 outcome 数据，使这一框架接受正式训练和独立验证。
