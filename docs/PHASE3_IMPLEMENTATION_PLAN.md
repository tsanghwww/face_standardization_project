# Phase3 实现方案：双流 3D 控制、隐空间扩散与视线解耦

日期：2026-09-01

状态：实现设计 v1；Phase3.0 坐标 gate 未通过，尚未启动正式生成器训练

## 1. 目标与边界

Phase3 将 Phase2 输出的质量感知 3D 标准化参数转化为标准化人脸图像：

$$
\hat{x}=G(x_{src},C_f,C_e,T_{id},T_h,T_g,q).
$$

系统需要同时实现：

1. head pose、expression 和 lighting 的标准化；
2. identity 与非目标区域保持；
3. head pose 与 eye-in-head gaze 分别控制；
4. 所有上游失败、生成失败与拒绝样本保留在完整分母。

本方案是对 GazeNeRF、ReDirTrans、DiffusionRig、DisControlFace、ControlNet 与 IP-Adapter 的工程适配，不声称复现其中任何一篇论文。

## 2. 文献驱动的设计修正

### 2.1 GazeNeRF：眼区和脸部必须分流

GazeNeRF 将 face-only 与 eye volume 分开建模，只对 eye stream 施加由目标 gaze 计算的 3D rotation，并分别监督 whole-face、face-only 与 eye outputs。其消融显示 two-stream、rotation 和 gaze functional loss 都会影响 gaze/head angular error。

对本项目的直接采用：

- 使用独立 `Face Control Adapter` 与 `Eye Gaze Adapter`。
- gaze feature 只能通过 eye mask 注入眼区。
- whole-face、face-only、eye-region 分别计算损失。
- gaze 统一在 head coordinate frame 表示。

不能直接采用：

- 普通 U-Net feature channel 不具备 GazeNeRF 中 `86 x 3` 的 3D vector 语义，因此不得直接执行 $RF$ 并声称实现了眼球刚体旋转。
- 不复现 NeRF volume rendering。原论文依赖标定 gaze 数据并报告单张 A40 约一周训练，不符合当前 8GB 预算。

### 2.2 ReDirTrans：旧状态移除、目标状态加入

ReDirTrans 的核心编辑为：

$$
\hat f_t=f_s-\Delta f_s+\Delta \hat f_t.
$$

它分别投影 gaze/head embedding，先用 source condition 逆旋转到 canonical state，再旋转到 target state：

$$
z_N^i=R^{-1}(c_s^i)z_s^i,
\qquad
\hat z_t^i=R(c_t^i)z_N^i,
\quad i\in\{h,g\}.
$$

对本项目的直接采用：

- head 与 gaze 使用不同 encoder、decoder 和类型嵌入。
- 每个 U-Net block 学习不同的 head/gaze residual gate。
- 通过 head-induced gaze error 与 gaze-induced head error评估解耦。
- source condition 从模型实际 reconstruction 重新估计，而不是盲信原图标签。

不能直接采用：

- ReDirTrans 使用同身份 source-target 图像对和准确 gaze/head 标签；当前 10K base 没有这种配对。
- 本项目不能使用 intervention image 的像素级 target loss，只能使用冻结 estimator 的 outcome loss、identity loss 与 non-target consistency。

### 2.3 其他论文的作用

- DiffusionRig：支持以 3D physical buffers 控制 diffusion，同时保留全局身份信息。
- DisControlFace：支持冻结生成骨干、训练显式 3D 控制网络，并用 semantic masking 减少显式控制与外观语义纠缠。
- ControlNet：支持冻结预训练 diffusion backbone，通过 zero-initialized residual injection 学习空间控制。
- IP-Adapter：支持把 identity image embedding 通过独立 cross-attention 注入，而不与 head/gaze token 混在同一投影中。

## 3. 数据契约

### 3.1 固定数据划分

| Split | 数量 | 用途 |
| --- | ---: | --- |
| train | 8,160 | 梯度更新 |
| validation | 1,440 | loss 权重、checkpoint、训练预算选择 |
| fixed-test base | 400 | 最终域内评估 |
| fixed-test external | 375 | 最终困难域评估 |

总 fixed evaluation 为 775。rescue 继续是 audit-only，不进入训练、阈值选择或主推理。

### 3.2 每样本输入

每个空间模态必须同时保留 `source_*` 与 `target_*`：reconstruction 使用 source 条件，intervention/standardization 使用 target 条件。当前 condition cache 已生成两套路径，但下游 JSONL builder 目前主要暴露 target 字段；进入 Phase3.1 前必须补齐 source 字段，不能用 target map 代替 source reconstruction condition。

空间条件：

$$
C_f=[N,D,L,M_f],
\qquad
C_e=[M_e,H_g].
$$

- $N\in\mathbb R^{3\times H\times W}$：target normal map。
- $D\in\mathbb R^{1\times H\times W}$：16-bit target depth，加载时归一化。
- $L\in\mathbb R^{1\times H\times W}$：target landmark heatmap。
- $M_f,M_e$：face/eye masks。
- $H_g$：以双眼区域为锚点的 target gaze direction map。

向量条件：

$$
v_h=[r_h^{6D},e_{phase2},l,\alpha_h,\alpha_e],
$$

$$
v_g=[\mathbf g_{head},\mathbf g_{head}^{*},\Delta\mathbf g,q_g],
$$

$$
v_q=[q_{xgb},q_{landmark},q_{arcface},s_{upstream}].
$$

identity 条件：

$$
e_{id}\in\mathbb R^{512}.
$$

缺失值必须为 `null`/空字段并带 status；禁止用 0 代替缺失，因为 0 是有效 pose/gaze/alpha 值。

## 4. 模型结构

### 4.1 Frozen latent diffusion backbone

- 冻结 VAE encoder/decoder。
- 冻结 U-Net 主干。
- 训练初期使用空文本或固定的人脸重构 prompt，避免文本成为隐藏控制变量。
- 256 x 256 pilot 通过后才讨论 512 x 512。

### 4.2 Face Control Adapter

输入为 $C_f$，经过四级卷积下采样，与 U-Net 的高分辨率到低分辨率 blocks 对齐。每级输出经过 zero-initialized convolution：

$$
c_{f,l}=Z_l^f(E_l^f(C_f)).
$$

Face branch 只接收 head、expression、lighting 与 geometry，不接收 target gaze。

### 4.3 Eye Gaze Adapter

Eye branch 接收 $C_e$ 与 gaze token：

$$
c_{e,l}=Z_l^e(E_l^e(C_e,T_g)).
$$

注入前将 eye mask 下采样到对应 block：

$$
\tilde c_{e,l}=\operatorname{Down}_l(M_e)\odot c_{e,l}.
$$

优先注入 64 x 64、32 x 32 的高分辨率 blocks；低分辨率全局 blocks 默认不开 gaze injection，防止 gaze 改变脸型、发型与背景。

### 4.4 Factor residual editor

分别对 source/target head 与 gaze 生成 block-wise residual：

$$
r_{h,l}=D_{h,l}(E_h(v_h^*))-D_{h,l}(E_h(v_h^{src})),
$$

$$
r_{g,l}=D_{g,l}(E_g(v_g^*))-D_{g,l}(E_g(v_g^{src})).
$$

U-Net block 的最终注入为：

$$
F_l'=F_l+c_{f,l}
+w_l^h r_{h,l}
+w_l^g\operatorname{Down}_l(M_e)\odot r_{g,l}.
$$

$w_l^h,w_l^g$ 使用 sigmoid gate，初始化接近 0。head/gaze encoder 不共享权重。

这借鉴 ReDirTrans 的 subtraction-addition 与 layer-wise weighting，但 residual 作用在 diffusion blocks，而不是 StyleGAN $W^+$。

### 4.5 Identity Adapter

ArcFace embedding 通过独立 projection 得到 identity tokens：

$$
T_{id}=P_{id}(e_{id}).
$$

identity 使用独立 cross-attention；不得与 head/gaze 向量先拼接再通过单个 MLP。该设计借鉴 IP-Adapter 的 decoupled cross-attention。

### 4.6 可训练参数

| 模块 | Phase3.1 | Phase3.2 | Phase3.3 | Phase3.4 |
| --- | --- | --- | --- | --- |
| VAE/U-Net backbone | frozen | frozen | frozen | frozen |
| Face Adapter | train | frozen/0.1x LR | train | train |
| Eye Adapter | off | train | train | train |
| Factor residual editor | off | gaze only | train | train |
| Identity Adapter | train | train | train | train |
| U-Net attention LoRA | off | off | off | optional rank 8 |
| evaluators | frozen | frozen | frozen | frozen |

## 5. 训练前审计

### 5.1 坐标 gate

必须验证 DECA head rotation 与 gaze estimator 的 axis、sign、handedness 和 crop convention。GazeNeRF 将不同数据集先做 camera/face normalization，再把 gaze 转到 head coordinate；这支持当前 gate 的必要性，但不能替本项目自动批准 DECA/L2CS convention。

坐标未批准时：

- `gaze_head_*` 必须为空；
- gaze rotation/token 必须标记为 `coordinate_status=unapproved`；
- gaze heatmap 不生成；
- Eye Adapter 和 gaze intervention 不得训练。

### 5.2 VAE round-trip audit

先生成：

$$
x_{vae}=D_{vae}(E_{vae}(x)).
$$

测量：

$$
\Delta_{id}^{vae}=1-\cos(e(x),e(x_{vae})),
$$

$$
\Delta_h^{vae}=d_R(R_h(x),R_h(x_{vae})),
$$

$$
\Delta_g^{vae}=d_\angle(g_{head}(x),g_{head}(x_{vae})).
$$

ReDirTrans发现 GAN inversion 会改变 gaze，因此使用 inversion output 的重新估计条件。Phase3 同理：intervention 的 source anchor 使用 $x_{vae}$ 或当前 reconstruction 的测量值，不能把 VAE shift 算到 adapter 身上。

### 5.3 Estimator isolation

- 训练 outcome loss：冻结的 differentiable PyTorch identity/head/gaze estimators。
- 最终评估：另一套未参与训练的 estimator。
- 现有 ONNX ArcFace 可以继续做最终 evaluator，但不能提供训练梯度。
- 如果无法获得独立 gaze evaluator，只允许写“对训练 estimator 的一致性”，不能写“独立验证的 gaze accuracy”。

## 6. 训练批次

### A. Reconstruction

$$
c^*=c^{src},\qquad x_{target}=x_{src}.
$$

学习基础重构、identity 与 Face Adapter。

### B. Eye-masked reconstruction

在 latent 或输入图眼区加入遮挡/噪声，target 仍为原图，强迫模型读取 gaze branch。

### C. Head-only intervention

$$
R_h^*\ne R_h^{src},
\qquad
g_{head}^*=g_{head}^{src}.
$$

没有像素级 target，只使用 head outcome、identity、non-target 与 $h\rightarrow g$ leakage loss。不得把原图 latent 当作 target 计算 diffusion noise loss，否则“改变 head”与“重构原图”会形成矛盾监督。

### D. Gaze-only intervention

$$
R_h^*=R_h^{src},
\qquad
g_{head}^*\ne g_{head}^{src}.
$$

只使用 gaze outcome、identity、non-target 与 $g\rightarrow h$ leakage loss。同样不计算以原图为 target 的 diffusion noise loss。

### E. Joint standardization

使用 Phase2 target head/expression 与选定 gaze policy，作为最终任务训练样本。初始比例建议：A 40%、B 30%、C 15%、D 15%；E 在 Phase3.3 后替换部分 C/D。比例只在 validation 上调整。

## 7. 损失函数

### 7.1 Diffusion 与区域重构

$$
\mathcal L_{diff}=\|\epsilon-\epsilon_\theta(z_t,t,C)\|_2^2.
$$

$\mathcal L_{diff}$ 只用于 A/B reconstruction batches。C/D/E 没有真实 target image，因此其 diffusion-loss mask 必须为 0。

对可获得像素 target 的 A/B batch：

$$
\mathcal L_{region}
=\mathcal L_{whole}
+\lambda_f\mathcal L_{face}
+\lambda_e\mathcal L_{eye}.
$$

其中每个区域同时计算 L1/LPIPS；eye loss 用 mask 面积归一化。

### 7.2 Outcome losses

$$
\mathcal L_{id}=1-\cos(e_{src},e_{out}),
$$

$$
\mathcal L_h=d_R(R_h^{out},R_h^*),
$$

$$
\mathcal L_g=d_\angle(g_{head}^{out},g_{head}^*).
$$

### 7.3 双向 leakage

$$
\mathcal L_{h\rightarrow g}
=d_\angle(g_{head}^{out},g_{head}^{src}),
$$

$$
\mathcal L_{g\rightarrow h}
=d_R(R_h^{out},R_h^{src}).
$$

ReDirTrans使用一个属性被编辑时另一个属性的 induced error 量化解耦，并以 $\epsilon\sim U(-0.1\pi,0.1\pi)$ 构造扰动。Phase3 pilot 使用更保守的 $\pm5^\circ/\pm10^\circ/\pm15^\circ$ 分层扰动，避免伪标签误差与超大编辑混淆。

### 7.4 非目标保持

$$
\mathcal L_{nt}
=\operatorname{LPIPS}((1-M_e)\odot x_{out},(1-M_e)\odot x_{anchor}).
$$

其中 $x_{anchor}$ 优先使用 VAE/reconstruction anchor。

### 7.5 总损失

$$
\begin{aligned}
\mathcal L={}&\mathcal L_{diff}
+\lambda_r\mathcal L_{region}
+\lambda_{id}\mathcal L_{id}
+\lambda_h\mathcal L_h
+\lambda_g\mathcal L_g\\
&+\lambda_{hg}\mathcal L_{h\rightarrow g}
+\lambda_{gh}\mathcal L_{g\rightarrow h}
+\lambda_{nt}\mathcal L_{nt}.
\end{aligned}
$$

所有辅助 loss 从 0 warm-up；先记录未加权 loss scale 与梯度范数，再在 validation 上确定权重。不得直接照搬 GazeNeRF/ReDirTrans 的数值权重，因为模型、分辨率、监督强度和数据均不同。

不同 batch 使用显式 loss mask：

| Batch | diffusion/region | head | gaze | identity | leakage | non-target |
| --- | --- | --- | --- | --- | --- | --- |
| A reconstruction | 是 | source consistency | source consistency | 是 | 否 | 是 |
| B eye-masked | 是 | source consistency | source consistency | 是 | 否 | 是 |
| C head-only | 否 | target | source invariance | 是 | $h\rightarrow g$ | 是 |
| D gaze-only | 否 | source invariance | target | 是 | $g\rightarrow h$ | 是 |
| E joint | 否 | target | target | 是 | 可选 | 是 |

### 7.6 Outcome loss 的显存控制

训练时不运行完整 20/50-step sampler。根据 scheduler 从 noisy latent 和预测噪声恢复单步 clean-latent estimate：

$$
\hat z_0=
\frac{z_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(z_t,t,C)}
{\sqrt{\bar\alpha_t}}.
$$

随后通过冻结但保留梯度路径的 VAE decoder 得到 $\hat x_0$，供 identity/head/gaze estimator 计算 outcome loss。初始每 4 个 optimizer steps 执行一次 outcome decode，并记录峰值显存；该频率只能在 validation 和资源预算内调整。

## 8. 分阶段实施

### Phase3.0A：坐标与 condition cache

当前状态：geometry 12/12 smoke 成功；10K gaze candidates 成功；coordinate convention 未批准。

产物：split registry、coordinate candidates、condition cache、coverage、hash、人工审计图。

### Phase3.0B：VAE round-trip audit

新增入口建议：

- `phase3/audit_vae_roundtrip.py`
- 输出 `vae_roundtrip_metrics.csv`、`summary.json`、失败清单与 contact sheet。

先在 32 张 validation pose-stratified 样本运行，再扩展到完整 validation。

### Phase3.1：32 样本 overfit smoke

- 只启用 Face Adapter + Identity Adapter。
- 500 steps，256 x 256，micro-batch 1，gradient accumulation 8。
- 通过条件：loss 明显下降；32 张均能生成；identity/pose 不比 VAE anchor 明显恶化。

### Phase3.2：Reconstruction pilot

- 8,160 train，1,440 validation。
- 加入 Eye Adapter 与 eye-masked batches。
- 仍不进行无 target 的大角度 intervention。
- 比较 `shared adapter` 与 `separate eye adapter`。

### Phase3.3：Residual intervention pilot

- 启用 source-remove/target-add editor。
- 先做 $5^\circ$，通过后再做 $10^\circ/15^\circ$。
- 加入 head-only/gaze-only 与双向 leakage loss。

### Phase3.4：可选 LoRA

只有 frozen-backbone full adapter 已在 validation 上优于 ablation，才启用 rank 8 attention LoRA。LoRA 不是 Phase3 完成的必要条件。

## 9. 8GB 5060 配置

- resolution：256 x 256 pilot；
- precision：FP16；
- micro-batch：1；
- gradient accumulation：8；
- gradient checkpointing：on；
- xFormers/SDPA：环境 smoke 通过后启用；
- VAE/U-Net：frozen；
- adapter LR：$10^{-4}$；
- LoRA LR：$10^{-5}$；
- seed：20260901；
- 每次 run 保存 config、exact command、Git commit、split hash、cache hash、GPU peak memory。

不从零训练 U-Net，不复现完整 GazeNeRF，不把 512 x 512 作为首个目标。

## 10. 代码实现清单

建议新增：

| 文件 | 作用 |
| --- | --- |
| `phase3/config.py` | 配置与 artifact schema |
| `phase3/dataset.py` | JSONL、图像、condition maps、tokens 加载 |
| `phase3/face_control_adapter.py` | geometry/head/expression 控制 |
| `phase3/eye_gaze_adapter.py` | eye mask/gaze heatmap/gaze token 控制 |
| `phase3/factor_residual_editor.py` | source-remove/target-add residual |
| `phase3/identity_adapter.py` | ArcFace tokens 与独立 attention |
| `phase3/model.py` | frozen backbone 与多分支注入 |
| `phase3/losses.py` | region/outcome/leakage losses |
| `phase3/train.py` | staged training、resume、artifact logging |
| `phase3/infer.py` | reconstruction/head-only/gaze-only/joint |
| `phase3/audit_vae_roundtrip.py` | VAE baseline drift |
| `phase3/evaluate_interventions.py` | accuracy、induced error、coverage |
| `tests/test_phase3_protocol.py` | split、mask、冻结参数、fixed-test isolation |

### 最小接口

```python
output = model(
    noisy_latent=z_t,
    timestep=t,
    source_latent=z_src,
    face_condition=face_maps,
    eye_condition=eye_maps,
    identity_embedding=arcface_embedding,
    source_head=source_head_6d,
    target_head=target_head_6d,
    source_gaze=source_gaze_head,
    target_gaze=target_gaze_head,
    quality=quality_features,
)
```

所有 gaze 参数必须带 `coordinate_status=approved`；否则 dataset 必须拒绝训练样本，而不是退回 camera-frame gaze。

## 11. 消融与评估

本科论文范围建议固定四组：

| 组 | Face/Eye 分流 | remove-add | leakage loss |
| --- | --- | --- | --- |
| A shared | 否 | 否 | 否 |
| B two-stream | 是 | 否 | 否 |
| C residual | 是 | 是 | 否 |
| D full | 是 | 是 | 是 |

共同指标：

- ArcFace cosine：mean/median/p05 与检测失败率；
- head target angular error；
- head-local gaze target angular error；
- head-induced gaze error；
- gaze-induced head error；
- eye LPIPS/SSIM；
- non-eye LPIPS；
- generation failure rate 与 coverage；
- VAE anchor-relative improvement。

训练 estimator 与最终 evaluator 必须分开。固定测试 775 只运行一次冻结配置，不参与 checkpoint、loss、扰动角度或阈值选择。

## 12. Go/No-Go

### 进入 Phase3.1

- coordinate convention 有独立证据并获批；
- 12 张 gaze heatmap smoke 方向与 eye anchor 正确；
- VAE round-trip audit 完成；
- differentiable training estimators 与独立 evaluators 已明确；
- 32 样本 condition cache 100% 可读。

### 进入 Phase3.3

- reconstruction pilot 优于 frozen VAE baseline；
- separate Eye Adapter 优于 shared adapter；
- identity 与非眼区没有明显恶化；
- estimator failure 保留在完整分母。

### 允许论文声称“视线解耦”

- head-only 与 gaze-only intervention 均已运行；
- full model 在两个 induced errors 上优于 shared/two-stream ablation；
- 使用未参与训练的 evaluator；
- 报告置信区间、失败率和数据限制。

否则只能声称“实现了独立 gaze conditioning 接口”或“观察到解耦行为”，不能声称完成稳健解耦。

## 13. 参考文献与采用关系

1. Ruzzi et al., [GazeNeRF: 3D-Aware Gaze Redirection with Neural Radiance Fields](https://arxiv.org/abs/2212.04823), CVPR 2023。采用：face/eye two-stream、eye-only rotation prior、区域损失、head-coordinate gaze；不复现 NeRF。
2. Jin et al., [ReDirTrans: Latent-to-Latent Translation for Gaze and Head Redirection](https://openaccess.thecvf.com/content/CVPR2023/html/Jin_ReDirTrans_Latent-to-Latent_Translation_for_Gaze_and_Head_Redirection_CVPR_2023_paper.html), CVPR 2023；[Supplementary](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Jin_ReDirTrans_Latent-to-Latent_Translation_CVPR_2023_supplemental.pdf)。采用：source-remove/target-add、factor-specific projection、layer-wise gates、induced error、reconstruction-anchor measurement。
3. Ding et al., [DiffusionRig: Learning Personalized Priors for Facial Appearance Editing](https://arxiv.org/abs/2304.06711), CVPR 2023。采用：DECA/3DMM physical buffers 与全局身份条件分工。
4. Jia et al., [DisControlFace: Adding Disentangled Control to Diffusion Autoencoder for One-shot Explicit Facial Image Editing](https://discontrolface.github.io/), ACM MM 2024。采用：冻结 diffusion reconstruction backbone、显式 3D control network、semantic masking。
5. Zhang et al., [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543), ICCV 2023。采用：冻结 backbone 与 zero-initialized spatial residual injection。
6. Ye et al., [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721), 2023。采用：identity 与其他条件的 decoupled cross-attention。
7. Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), CVPR 2022。采用：latent-space diffusion backbone。
8. Feng et al., [Learning an Animatable Detailed 3D Face Model from In-the-Wild Images (DECA)](https://arxiv.org/abs/2012.04012), SIGGRAPH 2021。采用：shape/expression/pose/camera/light 与 geometry buffers。
9. Abdelrahman et al., [L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments](https://arxiv.org/abs/2203.03339), 2022。采用：camera-frame gaze pseudo-label；不把其输出直接视为 head-local gaze ground truth。
10. Deng et al., [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698), CVPR 2019。采用：identity embedding 与最终身份保持评估。
11. Park et al., [Few-Shot Adaptive Gaze Estimation](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.html), ICCV 2019。采用：rotation-aware gaze/head representation。
12. Zhang et al., [On the Continuity of Rotation Representations in Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html), CVPR 2019。采用：head rotation 的连续 6D 表示。

## 14. 当前最优下一步

当前不应立即搭完整训练器。先实现 `phase3/audit_vae_roundtrip.py`，在 32 张 validation pose-stratified 样本上确定 frozen VAE 对 identity/head/gaze 的基础漂移；与此同时，用少量带标定 gaze/head 标签的 validation 数据解决 Phase3.0 coordinate gate。两项通过后，再实现 Face Adapter 的 32 样本 overfit smoke。
