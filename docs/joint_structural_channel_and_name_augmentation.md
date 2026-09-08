# 关节条件重构：结构化骨架通道与名称增广

> 状态：S + T 已实现并通过离线验证；A 未实现；尚未重训。
> 核心思路：让结构负责“关节在骨架中的位置”，让文本负责“关节是什么身体部位”。
> 实施状态见第 6 节。

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
- `joint_name_drop_prob` 只丢名称，不丢结构；
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
3. **多物种覆盖**：覆盖人形、四足、多足、翼类、蛇形以及没有 contact 标注的物种。
4. **多随机种子**：使用相同测试集和多个 seed，报告均值与方差，避免单次生成偶然通过。
5. **动作质量不退化**：对真实 clip 评估骨长误差、root height、位移、步频、关节活动幅度和 foot sliding；阈值应在训练前确定。
6. **关节方向测试**：不能只检查角度正负比例，还要设置有效屈曲幅度下限，避免接近 0° 的伪通过。
7. **A 的独立消融**：比较 S+T+C1 与 S+T+C1+A，确认名称鲁棒性提升且语料内质量没有下降。

只有离线、主方案训练和名称鲁棒性测试全部通过后，才能认为这次关节条件重构有效。

## 6. 实施状态

### 已落地

| 步骤 | 位置 |
|---:|---|
| 1 | `data_loaders/truebones/truebones_utils/joint_struct_features.py`：`build_joint_struct_features`，13 维、schema 1；单测 `tests/test_joint_struct_features.py` |
| 2 | dataset 在 `MotionDataset.__init__` 按物种缓存；`sample/generate.py:create_condition` 调用同一函数（审计脚本核对二者是同一个函数对象） |
| 3 | collate 传 `joint_struct`（`data_loaders/tensors.py`）；`InputProcess.struct_embedding` 为独立 MLP，投影后按 `joint_valid` 再次清零 |
| 4 | `build_joint_embedding_texts(..., slim=True)` 为默认；`JOINT_NAME_EMBEDDING_SCHEMA_VERSION` 13 → 14，meta 记录 `slim` |
| 5 | `CKPT_VERSION` 3 → 4；`joint_struct_schema_version` / `joint_name_embedding_schema_version` 写入 args.json，训练、resume、生成三处硬失败；cond schema 不匹配由 `ensure_joint_name_embeddings` 硬失败（原先只是 warning） |

结构通道恒开启、无 CLI 开关：`InputProcess.struct_embedding` 始终构建，`joint_struct` 缺失即硬失败，代码路径已固化。

### 离线验证结果

`tools/audit_joint_conditioning.py --cond dataset/merged/cond.npy`，260 个物种 / 9699 个关节：

- 确定性、改名后逐位相同、全部 finite、`max |feature| = 1.0`；
- slim 文本平均 1.898 token（上限 2），最长 4 token（`Left Lower Front Lip`），无结构派生词；
- 1724 组同文本碰撞中，结构描述子**未解决 0 对**（阈值 5）；
- padding latent 投影后严格为零，mixed batch 与单样本一致（1.2e-7）；
- 唯一未通过项：磁盘上的 cond 仍是 schema 12，需重新生成 —— 2026-09-07 已重新生成，现在 14/14 全通过。

### 训练后验证结果

cond 于 2026-09-07 20:07 重新生成（schema 14 / slim），`merged_locomotion_v6_jsc` 在其上训练满 200k 步（`joint_name_drop_prob=0.15`、struct schema 1、CKPT v4）。

`tools/verify_joint_conditioning_posttrain.py --model_path save/merged_locomotion_v6_jsc/model000200000.pt --seeds 10,11,12`，8 个物种 × 3 seed × 5 个名称变体，每个变体与 intact 共用同一份扩散噪声：

- **对照（`rerun`，同条件重采样）逐位为零**，所以任何偏差都确实来自名称通道。
- **单关节改名**（把一条腿的 `Right Calf` 换成 `Left Forearm`）：位移偏差 0.002–0.015 倍骨架尺寸，弯曲方向一致率 1.00。
- **全名盲测**（全部关节 → `unknown_joint_name`）：偏差 0.018–0.049，方向一致率 0.95–1.00，动作仍然成立。
- **左右名互换**：偏差 0.011–0.060，方向一致率 0.94–1.00；关节角色保持率与"根本没有名字"时基本相同（Dog 0.67 vs 0.67，KI_Human 0.92 vs 1.00），说明左右角色不再由名字决定。Ostrich（0.72 vs 0.86）和 Centipede（0.80 vs 0.90）仍有残余名称依赖。
- 覆盖人形、鸟形双足、四足、蛛形、多足、翼类、两种无 contact 标注的蛇形；蛇没有肢体铰链，方向/角色项按契约报 N/A 而不是伪造。

### 通道归因消融（回答"是结构通道起作用，还是名称 dropout 起作用"）

`--variants intact,rerun,blind,ood_name,struct_shuffle,blind_struct_shuffle`，同样 8 物种 × 3 seed。`struct_shuffle` 把结构描述子在有效关节之间**置换**（保持训练分布，只破坏"关节↔描述子"的对应），比置零更干净。

| 干预 | 位移偏差 dev | 承重骨误差（Dog / Eagle / Centipede） |
|---|---|---|
| `rerun` 对照 | 0.000 | — |
| `blind` 全部关节名消失 | 0.018–0.049 | 4.0 / 8.9 / 15.8 % |
| `ood_name` 全部关节名换成未登记同义词 | 0.016–0.043 | 4.1 / 9.2 / 18.5 % |
| `struct_shuffle` 结构置换、名字完好 | **0.074–0.269** | **31.9 / 41.9 / 49.0 %** |
| `blind_struct_shuffle` 两者都毁 | 0.079–0.261 | 29.1 / 47.1 / 51.7 % |

（intact 基线：3.8 / 9.0 / 16.6 %）

结论：

1. **结构通道是主要的逐关节身份信号。** 置换结构比删掉全部名字破坏大 4–10 倍；KI_Human 的弯曲方向一致率从 1.00 掉到 0.68。
2. **结构一旦被打乱，名字在不在几乎没区别**（`struct_shuffle` ≈ `blind_struct_shuffle`），所以观察到的名称鲁棒性不能只归给名称 dropout —— dropout 会把模型推向 rest_pose/拓扑，不会让它去用这个新通道。
3. **名字现在只是次要修饰。** 每物种改写 37–70 个关节名成未登记同义词，输出只移动 0.016–0.043，承重骨误差基本不变。
4. 代价是依赖翻转了：结构通道现在是单点依赖。新骨架若 `contact_joints` 检测出错或 run 划分异常，不再有名字兜底。

`struct_shuffle` 是上界估计——一个自相矛盾的信号可能比缺失信号更糟。

### 翻转后的风险：结构通道的单点依赖与语义可控性

`tools/verify_struct_dependency_risk.py`，6 物种 × 3 seed。所有结构故障都经过**真实的** `build_joint_struct_features` 重建，只替换 `joint_struct` 一路输入。

**A. 结构检测出错时的降级**（dev = 位移偏差 / 骨架尺寸；对照：删掉全部名字 = 0.018–0.049，结构置换上界 = 0.074–0.269）

| 故障 | dev 范围 | 承重骨误差（intact → 故障，最差物种） | 真实足端离地 |
|---|---|---|---|
| `contact_none` contact 全没检出 | 0.041–0.094 | 11.6 → 17.1 % | 基本不变 |
| `contact_wrong` contact 落在错误叶子 | 0.046–0.117 | 9.0 → 17.1 % | 基本不变 |
| `contact_one_side` 只检出一侧 | 0.029–0.088 | 11.6 → 13.7 % | 基本不变 |
| `rest_lean` rest pose 前倾 20° | 0.013–0.064 | 8.1 → 9.6 % | 基本不变 |
| `rest_mirror` rest pose 左右镜像 | **0.045–0.459** | **8.1 → 48.5 %** | 基本不变 |

- **contact 检测出错是温和降级**，量级与"删掉全部关节名"相当，远低于结构置换的上界；而且**足端仍然落地**——contact 标注全丢或全错，真实足端的最低高度几乎不变（KI_Human 0.023→0.026/0.035，Dog 0.017→0.009/0.011）。落地行为由几何和 rest pose 兜底，不是只靠 contact 标志。
- **唯一的悬崖是 `rest_mirror`**：KI_Human dev 0.459、承重骨误差 8.1%→48.5%，比结构置换还糟。需要注意这是**上界**：脚本只镜像了结构通道的输入，模型同时还从 `tpos_joint_embedding` 收到未镜像的 rest pose，两者互相矛盾。真实的朝向误检会让两者一起镜像，破坏应当小得多。结论是：**该防的不是"检测错了"，而是"结构通道与 rest pose 不一致"**。

**B. 名字还能不能指定语义（"这是翅膀不是手臂"）**

语料自带对照：Eagle 的翅膀链拼作 `Wing`，Ostrich 解剖学上同源的翅膀链拼作 `Forearm`。把各自改名成对方的写法：

| 改名 | 肢体竖直摆幅 intact → 改名 | 变化 | 全身 dev |
|---|---|---|---|
| Eagle 12 个 `Wing` → `Forearm` | 0.4067 → 0.4065 | −0.1 % | 0.005 |
| Ostrich 16 个 `Forearm` → `Wing` | 0.0464 → 0.0471 | +1.4 % | 0.008 |

**名字已经基本不具备语义控制力**，两个方向都动不了肢体行为，全身偏差还低于"删掉全部名字"。

但同一条肢体对**结构**有强响应：Ostrich 的翅膀摆幅在 `contact_none` 下 0.0464 → 0.0875（+89%），`contact_wrong` 下 → 0.0731（+58%）。也就是说"这条肢体是腿还是翅膀"现在由 `run_ends_contact` 回答，不由那个词回答。**新骨架上要让一条肢体表现得像翅膀，正确的杠杆是把它排除出 `contact_joints`，而不是给它改名。**

第 5 项（绝对动作质量）和第 7 项（A 消融）**不作为结论**：前者没有可比基线（上一版 checkpoint 的 cond 是旧 schema，当前代码拒绝加载；文档要求的"训练前确定阈值"也没有确定），后者依赖未实现的 A。第 5 项的统计量仍与同物种真实 clip 的同一统计量并排写进 JSON，等有基线时可直接用。

### 未做

- **A（同义增广）未实现，且按当前证据不建议做。** 它的目标是"降低模型对 canonical 名称点的过拟合"，而 `ood_name` 显示这种过拟合已经基本不存在：整副骨架改名成未登记同义词的代价与删掉全部名字同量级。A 的成本（人工审核的 alias 表 + 侧别安全性 + 新 sidecar 契约）与剩余收益不匹配。若仍要做，最小可行范围是只针对残余的左右名称依赖（Ostrich 0.72、Centipede 0.80），而不是全词表。
- **第 5.2 节第 5、7 项没有结论。** 需要一个用同一 cond schema 训练的对照 checkpoint，以及事先约定的质量阈值。
- **结构通道与 rest pose 的一致性没有守卫。** 上面的 `rest_mirror` 说明两者矛盾时输出崩坏（承重骨误差 48.5%）；`build_joint_struct_features` 与 `tpos_joint_embedding` 读的是同一个 cond 字段，所以正常管线难以产生这种矛盾，但 `process_new_skeleton` 之后值得加一条断言。
- **`_contact_flags` 用 `cond.get('contact_joints') or []` 取值**，传 ndarray 会抛 "truth value of an array is ambiguous"。现在只因为 cond 里存的是 list 才没暴露。
- **名字的语义控制力已接近零**，文档第 2 节"让文本负责关节是什么身体部位"这一半目标没有达成——身体部位实际上也由结构决定了。若确实需要按语义指定肢体角色，杠杆是 `contact_joints`。
