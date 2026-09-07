# 关节条件重构：结构化骨架通道与名称增广

> 状态：设计方案，尚未完成训练验证。
> 核心思路：让结构负责“关节在骨架中的位置”，让文本负责“关节是什么身体部位”。

## 1. 问题背景

当前模型将每个关节的名称编码为一个 T5 向量，并把它加到该关节的所有时间帧上。这个向量实际上承担了关节身份编码的职责。

现有做法有三个问题：

1. **模型过度依赖名称。** 同一副骨架只改变腿骨命名，就可能改变膝盖的弯曲方向；几何和拓扑相同并不能保证动作一致。
2. **名称文本混入了大量结构词。** `Segment Second Of 3`、`ChainEnd`、`Instance First Of 4`、`Contact` 等信息都可以由骨架数据计算，却与真正的身体部位词一起进入 T5 mean-pool。文本越长，`Calf`、`Finger`、`Wing` 等关键语义越容易被稀释。
3. **现有结构条件不能替代关节身份。** `graph_dist` 和 `joint_relations` 是成对的 attention relation；它们能描述两个关节如何连接，但不会直接告诉当前 token“我位于哪条肢体的什么位置”。模型收到的 canonical rest token 也不再保留逐关节 rest position。

因此，名称缺失、名称异常或名称分布外时，模型缺少一个稳定的结构回退信号。

## 2. 目标

本方案希望做到：

- 为每个关节提供独立、逐帧、可计算的结构身份；
- 缩短 joint name 文本，只让 T5 编码开放词表的身体语义；
- 在随机丢弃关节名时，模型仍能依靠结构生成合理动作；
- 提高模型对已知名称变体的鲁棒性；
- 保证训练和生成使用完全相同的特征算法与版本契约。

这是一项需要重训的条件重构。离线分析只能证明信号设计合理，最终收益必须由训练后实验确认。

## 3. 解决方案

方案由三个部分组成：

```text
结构通道 S：描述关节在骨架中的位置
文本瘦身 T：只保留侧别和身体部位语义
名称增广 A：训练时采样已审核的名称变体
```

S 和 T 构成主方案；A 风险更高，单独验证后再启用。

### 3.1 S：结构化骨架通道

#### 输入

结构特征只读取：

- `parents`；
- 物理 rest position，即 `cond['rest_pos_ric_hml']` 或 `cond['rest_pose'][:, :3]`；
- `contact_joints` 和 `contact_joint_source`。

不能使用传给 `InputProcess` 的 canonical rest token，也不能读取 joint name 文本。

#### 无分叉段

将骨架拆成互不重叠的无分叉段（branch-free run）：

- root 单独处理；
- root 或分叉点的每个 child 是新 run 的起点；
- 沿唯一 child 前进，直到叶子或下一个分叉点；
- 末端分叉点属于上游 run，它的 children 分别开始新的 run。

例如，人形腿可以形成：

```text
LeftThigh → LeftShin → LeftFoot → LeftToeBase
```

#### 每关节 13 维描述子

设关节位于长度为 `L` 的 run 中，第 `k` 个位置从 0 开始；`J` 是骨架关节数。几何尺度定义为：

```python
scale = max(x_span, y_span, z_span, eps)
```

| 特征 | 定义 | 作用 |
|---|---|---|
| `depth_norm` | `(k + 1) / L` | run 内相对位置 |
| `run_len_inv` | `1 / L` | 为相对位置提供有界的 run 长度信息 |
| `run_ends_contact` | run 末端或其直接 child 是否为 contact | 区分落地肢体与非落地肢体 |
| `is_contact` | 当前关节是否为 contact | 提供落地点身份 |
| `contact_known` | `contact_joint_source != 'none'` | 区分“不是 contact”和“没有 contact 标注” |
| `is_leaf` | 当前关节是否无 child | 提供末端身份 |
| `height_n` | `(y - y_min) / max(y_span, eps)` | 归一化高度 |
| `lateral_signed` | `(x - x_root) / scale` | 相对 root 的左右位置 |
| `fore_aft_n` | `(z - z_mean) / scale` | 前后位置 |
| `attach_h_n` | run 起点的 `height_n` | 肢体连接高度 |
| `subtree_n` | 子树关节数 / `J` | 分支规模 |
| `sib_rank` | 同父 children 按稳定 `(z, x, index)` 排序后的 `rank / (n-1)`；`n=1` 时为 0 | 区分重复兄弟分支 |
| `sib_n_inv` | `1 / n` | 提供有界的兄弟数量信息 |

实现必须对非法 parent、环、多 root、退化 span 和非有限值显式报错或采用统一的确定性规则。

#### 模型注入

结构特征通过独立 MLP 投影到 `latent_dim`，与 joint name embedding 并列地加到全部时间帧：

```python
self.struct_embedding = nn.Sequential(
    nn.Linear(STRUCT_DIM, latent_dim),
    nn.GELU(),
    nn.Linear(latent_dim, latent_dim),
)

struct_latent = self.struct_embedding(joint_struct.to(x.dtype))
struct_latent = struct_latent * joint_valid.unsqueeze(-1).to(struct_latent.dtype)

x = x + joints_embedded_names[None, ...]
x = x + struct_latent[None, ...]
```

必须遵守：

- 使用正常参数初始化；
- `joint_name_drop_prob` 和 `joint_name_drop_all_prob` 只丢名称，不丢结构；
- padding 必须在 MLP 投影后通过 `joint_valid` 再次清零；
- 不能用全零 name row 判断 padding，因为真实关节也可能没有有效名称。

### 3.2 T：joint name 文本瘦身

joint name 文本只保留：

- `Left` / `Right`；
- 身体部位和必要的语义限定词，例如 `Calf`、`Finger`、`Front Leg`。

以下派生信息不再写进文本，由结构通道承载：

```text
Segment / 序数 / Of / 数量
ChainStart / ChainMiddle / ChainEnd / ChainEarly / ChainLate
Instance / Contact / EndEffector
```

例如：

```text
Left Hand Thumb Segment Third Of 3 ChainEnd EndEffector
→ Left Hand Thumb
```

瘦身必须在 `build_joint_embedding_texts(..., slim=True)` 的 token 构造阶段完成，不能依靠事后字符串黑名单删除。非解剖关节继续使用空 embedding text。

S 与 T 应配套使用：T 单独上线会让原本依靠结构词区分的关节发生碰撞；S 负责补回这部分区分度。

### 3.3 A：可选的 joint name 同义增广

预处理仍将已知名称规范化为 canonical text，以保证 cond 可复现。训练时可按概率将 canonical embedding 替换成一个经过审核的同义变体 embedding，使模型不要只记住单个 T5 点。

变体离线编码到 sidecar，训练时不运行 T5：

```text
joint_name_synonym_embs.npy
canonical_text → [context-safe variant embeddings]
```

采样规则：

1. 用 `joint_name_synonym_aug_prob` 决定当前 joint 是否增广；
2. 未命中时严格使用 canonical embedding；
3. 命中时只从该文本已审核且上下文安全的 alias 中采样；
4. 每次最多替换一个语义项，避免多词变体的笛卡尔积；
5. 空文本和 padding 始终保持零。

many-to-one 规范化表不能直接反转。例如带左右含义的缩写必须与当前 `Left`/`Right` 一致；pair merge 和普通 synonym 也必须分别审核。

训练处理顺序为：

```text
同义采样
→ whole-joint name drop
→ elementwise dropout
→ species/joint FiLM
→ name projection
```

A 的目标是降低模型对 canonical 名称点的过拟合。它不能保证覆盖词表中从未登记的新拼写，因此必须通过留一和 OOD 名称实验验证。

### 3.4 版本与兼容性

结构、slim text 和 synonym sidecar 都必须有显式契约：

- `JOINT_STRUCT_FEATURE_SCHEMA_VERSION`；
- 新的 `JOINT_NAME_EMBEDDING_SCHEMA_VERSION`；
- checkpoint version；
- sidecar 中的 T5 名称、embedding 维度、slim schema、alias rule schema 和内容 fingerprint。

训练、resume 和生成发现版本或 fingerprint 不匹配时必须硬失败，不能只发 warning。`joint_struct` 建议从 cond 现有字段现场计算，不写入 cond，从而避免新增可选 key。

## 4. 实施顺序

| 步骤 | 内容 | 需要重建 cond | 需要重训 |
|---:|---|---:|---:|
| 1 | 实现 `build_joint_struct_features`，固定 run、排序、尺度和异常输入规则，并完成纯函数单测 | 否 | 否 |
| 2 | dataset 按物种计算并缓存；generation 调用同一函数 | 否 | 否 |
| 3 | collate 传递 `joint_struct`；模型增加结构 MLP 和投影后 padding mask | 否 | 是 |
| 4 | 实现 `build_joint_embedding_texts(..., slim=True)`，重建 joint name embeddings | 是 | 是 |
| 5 | 增加 struct/name/checkpoint schema 与严格兼容性检查 | 随第 4 步 | 是 |
| 6 | 训练 S + T + C1，先验证主方案 | — | 是 |
| 7 | 构建并审核 synonym sidecar，完成短程消融后决定是否加入 A | 否 | 是 |

训练和生成必须共享同一个结构 builder；任何一侧复制一份近似实现都会造成条件漂移。

## 5. 验证

### 5.1 离线验证

实现完成后、训练前必须通过：

1. **确定性与名字不变性**：同一几何和拓扑只改 joint name，`joint_struct` 必须逐位相同；重复运行结果一致。
2. **输入健壮性**：所有训练物种的特征均为 finite，范围受控；非法 parent、环和退化数据按契约处理。
3. **结构区分度**：slim text 产生的同骨架碰撞中，结构描述子未解决的 pair 不超过 5。
4. **文本瘦身**：平均文本长度不超过 2，最长不超过 4，不再出现结构派生词。
5. **padding**：结构 MLP 投影后的 padding latent 严格为零；不同关节数的 mixed batch 与单样本运行一致。
6. **路径一致性**：dataset 和 generation 对同一 cond 生成完全相同的结构特征。
7. **版本保护**：旧 cond、旧 checkpoint 或错误 sidecar 与新代码组合时必须明确失败。
8. **同义表安全性**：所有 alias 通过侧别和上下文检查；留一 alias 在 T5 空间中仍更接近正确语义组。

建议将上述统计集中到一个只读审计脚本中，并记录数据文件 hash 和各项 schema，避免文档与实现漂移。

### 5.2 训练后验证

训练后至少验证：

1. **名称扰动鲁棒性**：固定相同扩散噪声，只替换一个关键关节的名称 embedding；输出的关节角色和弯曲方向应保持稳定。
2. **名字盲测**：把全部 joint name 替换成 `unknown_joint_name`，模型仍能生成结构合理的动作。
3. **结构通道确实被使用**：记录结构 MLP 的输出 norm 和梯度 norm；关闭结构通道后，名称扰动和名字盲测应明显变差。
4. **多物种覆盖**：覆盖人形、四足、多足、翼类、蛇形以及没有 contact 标注的物种。
5. **多随机种子**：使用相同测试集和多个 seed，报告均值与方差，避免单次生成偶然通过。
6. **动作质量不退化**：对真实 clip 评估骨长误差、root height、位移、步频、关节活动幅度和 foot sliding；阈值应在训练前确定。
7. **关节方向测试**：不能只检查角度正负比例，还要设置有效屈曲幅度下限，避免接近 0° 的伪通过。
8. **A 的独立消融**：比较 S+T+C1 与 S+T+C1+A，确认名称鲁棒性提升且语料内质量没有下降。

只有离线、主方案训练和名称鲁棒性测试全部通过后，才能认为这次关节条件重构有效。
