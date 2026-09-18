# AnyTop 模型架构

本文档描述 AnyTop 的模型结构：输入特征、条件总线、decoder、训练扰动与损失，以及参考动作在
采样侧的使用方式。这里只讲结构与数据流，不列举具体超参数——层数、宽度、各种概率和损失权重
都由训练配置决定，不在本文范围内。`docs/` 下的其他文件是历史设计记录，与本文冲突时以本文
和代码为准。

## 速览

```text
带噪动作 x_t [B, J, F, T]（F = 每个关节的动作特征维）
  + rest pose（第 0 个时间 token）
  + T5 joint-name embedding
  + joint structural embedding
  + absolute frame embedding
                    │
                    ▼
       GraphMotionDecoder × L
       ├─ graph-aware spatial attention
       ├─ full temporal attention
       ├─ cross-limb temporal block（配置的后 N 层）
       └─ FFN
                    │
                    ▼
             预测干净动作 x_0
```

每层都会重新注入同一个条件总线：

```text
diffusion timestep
  → species FiLM
  + resample-speed condition
  + canonical output-frame condition
  + loop condition
  + action-label condition
```

参考动作不经过 reference encoder 或 cross-attention。它只在采样器侧用于：

- `skip_timesteps` 的 img2img 初始化；
- 每步 clamp 已知区域的 inpainting；
- clamp 已有帧、生成新增帧的 outpainting；
- 必要时在采样前先 retarget 到目标骨架。

基础训练目标是 `MSE(predicted_x0, target_x0)`，不是噪声预测。temporal attention 是全窗口
注意力；loop 使用条件 token、圆周相位、数据增强与闭合损失，不使用 temporal mask。

下面各节展开说明。

## 1. 整体结构

AnyTop 是一个直接在动作特征上去噪的扩散 Transformer。每个“帧 × 关节”是一个 token：先在
同一帧的全部关节之间做 graph-aware spatial attention，再让每个关节沿时间做 temporal
attention，最后由因子化注意力输出的预测就是干净动作。骨架信息（rest pose、关节名称、关节
之间的成对关系与拓扑距离）以加性 embedding 和 attention bias 的形式进入网络。

在主干之上，模型还包含：

- Perceiver 式 cross-limb temporal pathway，让不同肢体交换完整的运动节奏；
- loop 条件、圆周相位、loop 数据增强与闭合损失；
- per-joint structural channel 和 whole-joint name dropout；
- 全局 species FiLM 与 per-joint species × joint FiLM；
- 四槽 action-label 条件与 action CFG；
- canonical output-frame 和 resample-speed 条件；
- subtree / temporal-span 混合噪声训练；
- full temporal attention、QK-Norm 和细分的拓扑关系码。

参考动作不进入主干：它只在扩散采样器侧用于 img2img 初始化、inpainting clamp 和
outpainting（见第 9 节）。

## 2. 总体数据流

```text
x_t [B, J, F, T]
  │
  ├─ motion projection（root / non-root 分开）
  ├─ rest-pose projection，作为第 0 个时间 token
  ├─ frozen-T5 joint-name embedding → trainable projection
  ├─ joint structural descriptor → MLP
  └─ absolute sinusoidal frame PE
  │
  ▼
tokens [T+1, B, J, D]
  │
  │  timestep condition bus
  │    sinusoidal diffusion timestep
  │      → species FiLM
  │      + resample-speed token
  │      + canonical-frame token
  │      + loop token
  │      + action-label token
  │
  ▼
GraphMotionDecoder × L
  ├─ graph-aware spatial attention（每帧、跨全部关节）
  ├─ full temporal attention（每关节、跨全部 T+1 token）
  ├─ cross-limb temporal block（配置的 active layers）
  └─ FFN
  │
  ▼
root / non-root output projections
  │
  └─ 丢弃第 0 个 rest-pose token → 预测 x_0 [B, J, F, T]
```

模型直接预测干净动作 `x_0`，基础目标是 masked MSE，而不是预测噪声 ε。关节维 `J` 由输入
骨架决定，模型不硬编码关节数：训练预处理对关节数有上限，推理张量按实际骨架的关节数动态
构造。

## 3. 输入特征

### 3.1 动作与 rest-pose token

根关节和普通关节分别使用线性投影。rest pose 也有独立的 root/non-root 投影，并作为领先
时间 token 拼到动作帧前面。因此 decoder 实际处理 `T+1` 个时间位置，输出时再删除第 0 个。

### 3.2 关节名称

T5 编码发生在预处理阶段，模型不会在每次 forward 中运行 T5。`cond.npy` 携带
`joints_names_embs`，`InputProcess.text_embedding` 再把它投影到 latent width，并沿全部
时间位置加到对应关节 token。

`joint_name_drop_prob>0` 时，训练会把整个关节名称向量替换为 learned
`unknown_joint_name`。它不同于普通 element-wise dropout：目的是真正隐藏关节名字，让模型
回退到 rest geometry、pairwise topology 和 structural channel。

### 3.3 关节结构通道

每个关节还有一个与名字无关的结构描述，包括：

- branch-free run 内的归一化位置和 run 长度；
- run 是否落到 contact、当前 joint 是否 contact、contact 是否已知；
- leaf 标记；
- 归一化高度、左右 signed offset、前后位置、附着高度；
- subtree 大小、sibling rank 和 sibling count 的倒数。

这条通道由 `parents`、物理 rest positions 和 contact annotation 确定。改关节名不会改变它。
它经独立 MLP 投影后加到 token，并在投影后重新清零 padding rows。

## 4. 条件总线

扩散 timestep 首先做 sinusoidal embedding，然后按以下顺序组合条件：

```text
timestep
  → species FiLM
  + resample_speed
  + canonical_feature_mean/std
  + is_loop
  + action_label
```

组合后的 condition 在每个 decoder layer 中经该层自己的 `embed_timesteps` 再注入残差流。

### 4.1 Species 的两条通路

| 通路 | 输入 | 作用位置 | drop 语义 |
|---|---|---|---|
| `species_cond` | species T5 descriptor | timestep 的 FiLM：`gamma*t + beta` | 可由 `species_cfg_drop_prob` bypass 到 identity |
| `species_joint_cond` | `[joint_name || species]` | 每个 joint-name embedding 的 FiLM | 始终存在，不做 CFG drop |

两条 FiLM 的最后一层都以 identity 语义初始化。需要注意：只关闭 timestep species FiLM
并不构成完全的 species-unconditional forward，因为 per-joint species FiLM 仍然开启。

### 4.2 Action label

Action label 使用受控词表，不接收自由文本。每个词先查 checkpoint 内冻结的 T5 word table，
再按四个角色槽聚合：

1. head/action；
2. direction；
3. modifier；
4. hands。

四个槽拼接后由 MLP 投影并加到 timestep condition。训练时
`action_label_cfg_drop_prob` 把部分样本送到 learned null embedding；推理时
`action_label_cfg_scale>1` 用 conditional/unconditional 两次 forward 做 CFG。

### 4.3 Canonical frame 与 resample speed

`canonical_feature_mean/std` 定义模型写入的输出坐标/标准化空间。这两组向量经
zero-init MLP 投影并始终注入，不可 CFG-drop；缺失时 forward 直接失败。

`resample_speed_cond = source_frames / internal_frames`，经另一条 MLP 注入。它告诉模型当前
时间窗口相对源动作的压缩/拉伸程度。数据侧的 `motion_speed_aug` 是另一项无显式条件的数据
增强，不应与 `resample_speed_cond` 混为一谈。

## 5. Decoder layer

每层的实际顺序是：

```text
x + embedded timestep/conditions
  → Graph Spatial Attention + residual + norm
  → Full Temporal Attention + residual + norm
  → Cross-Limb Block（若本层启用）
  → FFN + residual + norm
```

该层沿用了 `nn.TransformerDecoderLayer` 的外形，但不含 encoder-decoder memory attention：
`self_attn` 位置换成 graph/temporal 两条路径，`multihead_attn` 不使用。

### 5.1 Graph-aware spatial attention

Spatial attention 每帧在全部有效关节之间计算，不是只让图上的一跳邻居通信。它在普通
content attention 上加入两类 GRPE bias：

```text
score(i,j) = q_i·k_j
           + q_i·R_query[relation(i,j)]
           + k_j·R_key[relation(i,j)]
           + q_i·D_query[distance(i,j)]
           + k_j·D_key[distance(i,j)]
```

实际实现还对每个 head 的 Q/K 做 RMS normalization，约束 content dot product 和 graph
bias 的幅度，避免 softmax 饱和。

关系码分为两类：

- directed edge relation codes：self、parent、child、sibling、no-relation（无关系）、
  end-effector、ancestor、descendant、sibling-limb 以及按远近分档的 cousin；
- topology-distance codes：近距离 hop 精确编码；更远的 pair 再按是否共线以及
  normalized LCA depth 分档。

Q/K 的 topology/edge embedding 表由所有 decoder layers 共享。`value_emb=false` 时关系
embedding 只进入 attention score、不进入 value/output，注意力走 PyTorch SDPA 快路径。

### 5.2 Full temporal attention

Temporal attention 把张量视为 `[T+1, B*J, D]`，同一关节的每个时间 token 可以看到整个
窗口，包括 rest-pose token。窗口就是内部帧数加一，没有 causal、local 或 loop attention
mask。

### 5.3 Cross-limb temporal block

基础的因子化 spatial/temporal attention 没有一条让“某条腿的完整时间轨迹”直接观察
“另一条腿的完整时间轨迹”的短路径。Cross-limb block 用 K 个 learned latents 补上它：

```text
每帧全部 joints --cross-in--> K 个全身 latents
全身 latents ------时间自注意--> 跨时间节奏上下文
每帧全部 joints <--cross-out--- K 个全身 latents
```

该路径在窄 bottleneck `cross_limb_dim` 中运行，只在部分层启用，每个启用的层有独立 block；
latent 数量与维度由配置决定。

完整的 block 顺序（`CrossLimbTemporalBlock.forward`）：

```text
cross-in（per-joint reliability_bias 加在 logits 上）
  → latent temporal self-attention（per-frame temporal_reliability_bias 作为 additive key bias）
  → cross-K self-attention（同一帧的 K 个 latent 互相注意；Pre-Norm + 零初始化 cross_k_scale）
  → cross-out
```

“哪些区域不可靠”这条信号分三层进入网络：

- `AnyTop.unreliable_embedding`：全局一个 `d_model` 向量，在 InputProcess 之后按
  `unreliable_mask` 加到输入 token 上，整个 trunk（spatial/temporal attention、FFN，
  以及 cross-limb 的 value/query）都能看到重绘区域；
- cross-in 的标量 `reliability_bias`：做帧内的关节级选择；
- latent temporal attention 的标量 `temporal_reliability_bias`：乘以“该帧被标记的有效关节
  占比”作为 key bias，做帧级选择。只有 cross-in 的标量 bias 时，整帧被标记会被 softmax 的
  平移不变性抵消，整帧 inpaint 和 temporal span 对这条路径等于不存在。

cross-K attention 让同一帧的 K 个 latent 在 cross-out 之前直接交换信息，而不必绕
“cross-out → 下一层 spatial attention → 下一层 cross-in”。这些新增标量都零初始化，
`cross_k_norm`/`cross_k_attn` 藏在零 gate 之后，训练起点等价于没有 cross-K。

## 6. Loop 路径

Loop 不是单一布尔 token，而是模型、数据和损失共同组成的一条路径。

### 6.1 模型内

- `is_loop` 经 MLP 加到 timestep condition；
- 主 temporal path 使用 absolute PE，并在 loop 样本上额外加入 circular phase embedding；
- cross-limb latents 自己没有输入级 absolute PE，因此 loop 样本选择 circular table，非 loop
  样本选择 absolute table；
- circular table 在窗口的第一个和最后一个动作帧闭合；第 0 行 rest-pose token 为零。

两处 phase scale 都是 learned scalar，并以 0 初始化。

### 6.2 数据侧

真实 loop clip 会做随机 circular roll 和随机 tile，然后统一 resample 到内部窗口。tile count
和 phase offset 只用于诊断，不直接喂给模型。
k 份 tile 经周期重采样后是 k 份逐位相同的拷贝（周期 T/k 帧），是 loop 任务里容易的一侧；
`--loop_tile_single_prob` 给单周期窗口（推理 `--loop` + 自动长度所在的 regime）的概率设一个下限，
其余质量在 2..max 上仍均匀，默认 0.5；0 即原来的均匀抽签。

真实 loop clip 总是以 `is_loop=True` 喂给模型；唯一的降级是超出源帧预算被裁剪的 clip（环被裁开，
按非 loop 告知）。曾有的 `loop_cond_prob`（随机把 loop 标成非 loop）已删除：它没有 null 态，只是把
同一段内容按两种标签训练，稀释 `is_loop=0` 的 one-shot 语义并砍掉 loop 分支 30% 的样本，推理端
也没有任何消费者。

### 6.3 损失侧

- `loop_wrap_loss`：loop 首尾的 position、rotation 和 terminal velocity 闭合；
- `loop_root_xz_closure_loss`：translation root 的整周期 XZ velocity 积分闭合。

loop 路径不使用 loop-aware temporal mask。

## 7. 混合可靠性训练

基础扩散先按统一 timestep 生成 `x_t`，随后可对局部区域用独立噪声重新加噪。重加噪的
timestep 是一个混合分布（`--renoise_same_level_prob`）：取默认值时每个样本 `t_random = t`
（同级、只换噪声），把“被标记”与“比周围脏得多”解绑——推理时被夹住的已知区和自由区都
处于同一名义 timestep；降低该值会让一部分样本均匀取自 `[t, T)`，继续训练用可靠上下文修复
严重局部损坏。两支都不早于 `t`。

- subtree joint perturbation：随机选择预算内的非根子树；
- temporal-span perturbation：随机选择连续帧，并覆盖该样本的全部真实关节；对 k 份 tile 的 loop 窗口，
  同一相位的 span 在每一份拷贝里都重画（`y['loop_tile_count']`），否则被重加噪的段在一个周期之外有干净副本可抄；
- 两者可以取并集；
- supervision target 不变，被扰动单元仍参与 loss 和 attention；
- `cross_limb_unreliable_mask` 把位置告知整个 trunk（`unreliable_embedding`）和 cross-limb
  的两个 reliability bias；
- `unreliable_mask_drop_prob` 会在部分样本上隐藏这张图，让模型自行定位损坏区域。

这是训练阶段的扰动策略，不引入额外分支。

## 8. 损失

模型预测 `x_0`。基础 `l_simple` 是按有效关节、有效帧计算的 MSE；其余项均为可选，是否启用
与权重由训练配置决定：

| 损失 | 约束 |
|---|---|
| geodesic | 预测/目标 6D rotation 转 SO(3) 后的角距离 |
| velocity consistency | position finite difference 与 velocity channel 一致 |
| bone length | 相对 GT、按 rest length 归一化的骨长 |
| loop wrap | loop 首尾闭合 |
| loop root XZ closure | 根 XZ 速度整周期积分为零 |
| temporal-span seam | span 边界附近 position 二阶差分匹配 GT |

## 9. 参考动作与编辑（采样侧）

参考动作不进入 decoder，不产生 `reference_memory`，也不做 cross-attention。它只用于：

- plain reference generation：参考动作作为 `init_image`，从 `skip_timesteps` 指定的噪声等级
  开始 img2img；
- inpainting：被编辑区域自由生成，已知区域在每一步投影回相应噪声等级的 reference；
- outpainting：已有帧 clamp，新增帧从纯噪声生成；需要时使用两阶段流程；
- cross-skeleton reference：先在采样前 retarget 到目标骨架特征空间，再进入上述流程。

因此这些编辑能力不增加 AnyTop forward 的参数量。

## 10. 代码入口

| 主题 | 文件 |
|---|---|
| AnyTop 输入、条件与 forward | `model/anytop.py` |
| graph/temporal/cross-limb attention | `model/motion_transformer.py` |
| diffusion 构建与 `x_0` 目标 | `utils/model_util.py` |
| 训练 perturbation 与损失 | `diffusion/gaussian_diffusion.py` |
| joint structural channel | `data_loaders/truebones/truebones_utils/joint_struct_features.py` |
| refined topology codes | `data_loaders/truebones/truebones_utils/topology_relations.py` |
| loop roll/tile/resample | `data_loaders/truebones/data/dataset.py` |
| reference/inpainting/outpainting | `sample/generate.py` |
