# Phase3.0 进度报告（2026-09-01）

## 已完成

1. 固定 Phase3 数据划分并生成 SHA256 registry：train 8,160、validation 1,440、fixed-test base 400、fixed-test external 375，总 fixed evaluation 775；所有两两交集均为 0。
2. 为 10,000 个 base 样本生成 DECA/L2CS 坐标候选 manifest：10,000 成功，0 失败。
3. 从 validation 按 pose norm 与 x/y/z 轴极值确定性抽取 12 张坐标审计样本。
4. 为 12/12 样本成功生成 source/Phase2-target normal、16-bit depth、landmark、face mask 与 eye mask。
5. 发现并修复 normal map 脸外区域由 0 被映射成灰色 127 的问题；修复后以 face alpha 将背景严格归零。
6. 生成 direct/inverse convention 并排审计图。两候选目标方向夹角为 17.57°–59.57°，中位数 38.16°，说明 convention 选择对训练有实质影响。

## 固定划分哈希

| 分组 | SHA256 |
| --- | --- |
| train | `50c2bcc07ae74c9f86be36039782b44c8c77769298ef24c46c7856720ca17b78` |
| validation | `c9f16d8d0104ae4bf5ae1ce2f7b16df3d8bebe1db843e8b736198015f425c8c5` |
| fixed-test base | `78aef4ccabb29fba9a49b29eea8127820f7dbdf2755846928fa8edc83921e9d1` |
| fixed-test external | `d2c3d4a80d75817feecb1778006cae7c565ad309a007b95414f755bce4435365` |
| fixed evaluation 775 | `68133e2365530530f6325325896621971be213f0942df54b96e9e329599dc408` |

## 当前结论

工程链路已经从占位字段推进到真实 DECA 条件图，但 gaze coordinate 仍处于 `candidate_unvalidated`：

- 数学 round-trip 通过只能证明各候选内部自洽。
- DECA 源码将 `pose[:3]` 作为 FLAME global pose，经 Rodrigues/LBS 作用于模板顶点，这支持 `R` 是从 canonical head 到 rendered model frame 的旋转。
- 但 L2CS camera frame 与 DECA rendered frame 的轴符号/手性尚缺少独立标定或带真值的视线样本验证。

因此目前仍是 Phase3.0 **No-Go**：不签发 coordinate approval，不生成训练用 gaze heatmap，不启动 Phase3.1 reconstruction warm-up。这不是工程失败，而是避免把未经验证的伪标签注入模型。

## 下一步

1. 在小型带标定 gaze/head-pose 真值的数据上验证轴向，或构造可核验的受控采集样本。
2. 冻结唯一 convention 并保存 reviewer、证据路径与审批 JSON。
3. 用审批后的坐标生成 12 张 gaze heatmap smoke，验证 eye anchor 与 head-only 合成旋转。
4. 再批量构建 8,160/1,440/base-test 条件缓存并生成 cache hash。
5. 全部通过后进入 Phase3.1 的 500-step reconstruction overfit smoke。

外部 375 样本与 rescue 不参与训练；rescue 始终保持 audit-only。
