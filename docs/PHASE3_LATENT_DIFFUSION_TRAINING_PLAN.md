# Phase3 训练方法：3D 融合控制、隐空间扩散与视线解耦

日期：2026-09-01

状态：设计冻结候选（尚未启动正式训练）

## Material Passport

- 研究主题：基于 3D 融合控制与隐空间扩散模型的人脸标准化重构及视线解耦研究
- 设计依据：`pro2.0(1).docx`、当前 `main`、Phase2/Phase2.1 已完成实验
- 已有数据：9,600 个 base 训练池、1,440 validation、775 fixed test（其中 375 个外部困难样本）
- 当前硬件：RTX 5060 Laptop，约 8 GB 显存
- 证据边界：本文件定义训练与验证协议，不代表扩散模型或视线解耦已经完成

## 1. 研究目标重新定位

Phase2 的作用是生成质量感知的标准化 3D 参数，不是最终输出模型。Phase3 才对应题目中的核心系统：

$$
x_{src}, c_{3D}, e_{id}, c_{gaze}, c_{quality}
\longrightarrow
\hat{x}_{std}.
$$

最终标准化图像应同时满足：

1. 头部姿态、表情与光照接近标准目标。
2. 视线可以独立保持或校正，不与 head pose 混为一个变量。
3. ArcFace 身份和眼周解剖特征尽量保持。
4. 生成失败、上游失败和拒绝样本继续显式记录。

本项目定义两种 gaze policy：

- `canonical_camera_gaze`：最终标准化任务使用，目标是相机方向上的标准视线。
- `preserve_eye_in_head`：用于 head-only 训练与解耦验证，改变头部时保持眼球相对头部的视线。

只训练 `canonical_camera_gaze` 会让模型学到“正脸总是直视”的相关性，不能证明解耦。因此两种 policy 都必须进入训练与消融。

## 2. 条件表示

### 2.1 空间条件分支

3D Fusion Controller 接收像素对齐条件：

$$
C_{spatial}=[N,D,L,M_f,M_e,H_g],
$$

其中：

- $N$：DECA normal map，3 通道。
- $D$：DECA depth map，1 通道。
- $L$：landmark heatmap，1 或多通道。
- $M_f$：face mask。
- $M_e$：eye-region mask。
- $H_g$：空间 gaze-ray heatmap。

$H_g$ 不能只是把三维向量通过 MLP 后广播到整张图。它应以左右眼中心为锚点，将目标视线投影成眼区方向场或热图，否则条件没有明确空间意义。

### 2.2 向量条件分支

向量条件映射为 cross-attention tokens：

$$
C_{token}=[p_{phase2},R_h^*,\mathbf g_{head}^*,l,q].
$$

- $p_{phase2}$：Phase2 输出的 expression、pose、camera、lighting 与 alpha。
- $R_h^*$：目标 head rotation，建议采用连续 6D rotation 表示进入网络。
- $\mathbf g_{head}^*$：目标 eye-in-head gaze。
- $l$：目标 lighting 参数。
- $q$：质量分数、landmark score、失败状态等诊断特征。

头部 token 与 gaze token 必须使用独立投影层和独立类型嵌入。

### 2.3 身份分支

ArcFace 512 维 embedding 经 Identity Adapter 映射为 identity tokens：

$$
T_{id}=f_{id}(e_{ArcFace}).
$$

现有 InsightFace/ONNX ArcFace 继续作为最终 evaluator。若需要反向传播 identity loss，必须增加冻结的 PyTorch 人脸识别模型；不能声称从 ONNX evaluator 获得了训练梯度。

## 3. 模型架构

在当前显存与数据规模下，不从零训练 diffusion U-Net。采用以下结构：

1. 冻结预训练 latent diffusion 的 VAE 与 U-Net 主干。
2. 训练轻量 3D Control Adapter，将空间条件注入 U-Net 多尺度残差。
3. 训练 Identity Adapter，将身份 token 注入 cross-attention。
4. 训练 Gaze Adapter，将 gaze token 与 eye-region feature 注入眼区 gated attention。
5. 只在联合微调阶段启用 rank 8 的 U-Net attention LoRA，其余骨干保持冻结。

该选择与 DiffusionRig 的像素对齐 physical buffer 结果以及 DisControlFace 的“冻结预训练骨干、单独训练显式控制网络”思想一致，但本项目增加独立 gaze/head 条件与双向 leakage 监督。

## 4. 无配对数据下的训练样本构造

当前 10K 数据没有同一身份在不同 head/gaze 下的完整配对，因此不能直接进行普通的 paired image translation。训练批次分为四类：

### A. Reconstruction batch

- 目标图像：原图 $x$。
- 条件：原始 DECA buffer、原始 head pose、原始 head-local gaze。
- 用途：学习图像重构、身份和条件读取。

### B. Eye-masked gaze batch

- 将输入 latent 的眼区遮挡或强噪声化。
- 目标仍为原图。
- gaze 条件使用原图 L2CS/DECA 转换后的 $\mathbf g_{head}$。
- 用途：迫使模型利用 gaze branch，而不是从可见眼部像素复制 gaze。

### C. Head-only intervention batch

- 改变 $R_h^*$，保持 $\mathbf g_{head}^*=\mathbf g_{head}^{src}$。
- 目标没有像素级 ground truth。
- 使用冻结的 DECA/L2CS/identity 网络对预测 $\hat{x}_0$ 施加 outcome loss。

### D. Gaze-only intervention batch

- 保持 $R_h^*=R_h^{src}$，改变 $\mathbf g_{head}^*$。
- 对 head pose 不变性、gaze target 和 identity 施加 outcome loss。

联合训练初始采样比例建议为：A 40%、B 30%、C 15%、D 15%。这是 pilot 起点，最终比例只能在 1,440 validation 上选择。

## 5. 分阶段训练

### Phase3.0：条件与坐标审计（强制 gate）

不训练模型。完成：

- DECA head rotation 与 L2CS camera gaze 的轴向、正负号和 crop 约定验证。
- 生成 source/target normal、depth、landmark、face/eye mask 和 gaze heatmap cache。
- 检查 8,160 train、1,440 validation、775 fixed test 完全隔离。
- rescue 继续 audit-only。

通过标准：合成旋转可逆；人工抽查的大姿态样本方向正确；缺失值不填 0。

### Phase3.1：Reconstruction warm-up

冻结 VAE、U-Net 和 evaluator，只训练 3D Control Adapter、Identity Adapter 与 condition encoders。

$$
\mathcal L_{3.1}=\mathcal L_{noise}+\lambda_{rec}\mathcal L_{rec}+\lambda_{id}\mathcal L_{id}.
$$

目的不是完成标准化，而是证明模型能重构输入、读取 3D 条件并保持身份。

### Phase3.2：Gaze branch warm-up

加入 eye-masked batch，只训练 Gaze Adapter、eye gate 和 gaze condition encoder；其他 adapter 可冻结或使用更低学习率。

$$
\mathcal L_{3.2}=\mathcal L_{noise}+\lambda_{eye}\mathcal L_{eye}+\lambda_g\mathcal L_{gaze\_head}+\lambda_{id}\mathcal L_{id}.
$$

通过标准：有 gaze 条件应显著优于 `no-gaze` ablation；不能只比较训练 loss。

### Phase3.3：Disentanglement intervention training

加入 head-only 与 gaze-only batch：

$$
\mathcal L_{dis}=\lambda_{h\rightarrow g}\mathcal L_{h\rightarrow g}+\lambda_{g\rightarrow h}\mathcal L_{g\rightarrow h}.
$$

其中：

- $\mathcal L_{h\rightarrow g}$：head-only 编辑后的 head-local gaze 变化。
- $\mathcal L_{g\rightarrow h}$：gaze-only 编辑后的 head pose 变化。

此阶段允许 outcome supervision，但 fixed test 不参与任何权重或阈值选择。

### Phase3.4：Joint LoRA fine-tuning

启用 U-Net attention LoRA，联合训练所有 adapter。先使用 256×256 pilot；通过后再决定是否进行 512×512 实验。

最终损失：

$$
\begin{aligned}
\mathcal L={}&\mathcal L_{noise}
+\lambda_{id}\mathcal L_{id}
+\lambda_{head}\mathcal L_{head}
+\lambda_{gaze}\mathcal L_{gaze\_head}\\
&+\lambda_{h\rightarrow g}\mathcal L_{h\rightarrow g}
+\lambda_{g\rightarrow h}\mathcal L_{g\rightarrow h}
+\lambda_{nt}\mathcal L_{non-target}
+\lambda_{bg}\mathcal L_{background}.
\end{aligned}
$$

辅助损失从 0 线性 warm-up，避免训练早期代理网络梯度压过 diffusion noise loss。初始 pilot 可使用：

- $\lambda_{id}=0.1$
- $\lambda_{head}=0.05$
- $\lambda_{gaze}=0.1$
- $\lambda_{h\rightarrow g}=0.05$
- $\lambda_{g\rightarrow h}=0.05$
- $\lambda_{nt}=0.1$
- $\lambda_{bg}=0.1$

这些值是工程起点，不是论文固定常数。需要记录每项 loss 的尺度和梯度范数，并只用 validation 调整。

## 6. 5060 有界 pilot 配置

- 分辨率：256×256。
- 精度：FP16 mixed precision。
- micro-batch：1。
- gradient accumulation：8。
- gradient checkpointing：开启。
- U-Net/VAE：冻结。
- LoRA rank：8，仅 Phase3.4 开启。
- adapter learning rate：$1\times10^{-4}$。
- LoRA learning rate：$1\times10^{-5}$。
- seed：20260901。
- 先进行 500-step overfit smoke，再进行固定预算 pilot。
- 每个 checkpoint 保存 config、exact command、Git commit、split hash 与 condition-cache hash。

正式步数不在运行前臆定为“足够”。先根据 500-step smoke 的吞吐、显存和 validation 曲线确定有界预算。

## 7. 评估与消融

### 必须报告的指标

- ArcFace cosine 的均值、中位数、低分位与失败率。
- Head target angular error。
- Head-local gaze angular error。
- $h\rightarrow g$ 与 $g\rightarrow h$ leakage。
- 非目标区域 LPIPS。
- 眼区 LPIPS/SSIM 或 landmark consistency。
- FID/KID（只作为总体分布质量，不代替身份或解耦指标）。
- 完整 split 分母上的生成失败率与 coverage。

### 必须运行的 ablation

1. Frozen latent diffusion reconstruction baseline。
2. `+3D spatial control`。
3. `+Phase2 standardized condition`。
4. `+Identity Adapter`。
5. `+camera-frame gaze`。
6. `+head-local gaze`。
7. `+eye-masked training`。
8. `+bidirectional leakage loss`（完整模型）。

每组使用相同 train/validation/fixed-test IDs。最终解耦结论必须来自 head-only 和 gaze-only 干预，不能只依靠普通重构。

## 8. 对原方案的必要修正

1. `ArcFace cosine > 0.95` 不是可直接套用的通用学术阈值。主结论使用相对 reconstruction baseline、置信区间和 validation 冻结阈值。
2. 非眼区 LPIPS 低只能说明非目标区域变化较少，不能单独证明 gaze 与 head pose 解耦。
3. L2CS camera gaze 不能直接当作 eye-in-head gaze；必须结合 head rotation 转换。
4. FFHQ/StyleGAN2 10K 不能自动支持“多族裔稳健性”结论，只能按可验证的分组报告，并把覆盖不足写入限制。
5. 文档中的 MC-pDM 全称应核对为 *Eye Gaze Correction Using Multi-Conditional Patch-Diffusion Model*，不是 “personal Diffusion Model”。
6. 当前未检索到题为 *InstaFace: 3D-conditioned Human Face Generation* 的可核验正式来源，因此暂时将其视为架构灵感，不把其模块名称写成已复现依赖。

## 9. 进入正式训练前的 Go/No-Go

Go：

- 完成真实 condition map 生成与 coverage audit。
- 完成 DECA/L2CS 坐标约定审计。
- identity、pose、gaze evaluator 能处理生成图并显式记录失败。
- 500-step overfit smoke 能降低 reconstruction loss，输出非空且身份没有明显崩坏。

No-Go：

- 在 condition map 仍为 `null` 时启动正式训练。
- 从零训练完整 diffusion backbone。
- 使用 775 fixed test 调 loss 权重、步数或阈值。
- 将 rescue 自动并入训练。
- 仅凭 gaze delta 或 LPIPS 宣称视线解耦完成。

## 10. 论文支撑

- [DiffusionRig, CVPR 2023](https://diffusionrig.github.io/)：支持使用 DECA 像素对齐 physical buffers 引导 diffusion，并显示空间条件优于简单参数向量条件。
- [DisControlFace, ACM MM 2024](https://discontrolface.github.io/)：支持冻结预训练 diffusion backbone、训练显式控制网络与随机语义遮挡。
- [L2CS-Net](https://arxiv.org/abs/2203.03339)：提供非受限环境 gaze pseudo-label，但不单独证明 disentanglement。
- [ST-ED, NeurIPS 2020](https://ait.ethz.ch/sted-gaze)：支持分别控制 gaze 与 head orientation，并进行解耦评价。
- [Few-Shot Adaptive Gaze Estimation / DT-ED, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.pdf)：支持 rotation-aware gaze/head factor separation。
