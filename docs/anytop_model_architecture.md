# AnyTop 模型架构

本文档描述 AnyTop 的模型结构：输入特征、条件总线、decoder、训练扰动与损失，以及参考动作在
采样侧的使用方式。层数、宽度、各种概率和损失权重由训练配置决定；正文以结构与数据流为主，
实现细节以代码为准。

## 速览

```text
带噪动作 x_t [B, J, F, T]（F = 每个关节的动作特征维）
  + rest pose（第 0 个时间 token）
  + T5 joint-name embedding（可选 per-joint species FiLM 后投影）
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

每层都会重新注入同一个 timestep 条件总线（loop 使用独立通路，见 §6.1）：

```text
diffusion timestep
  + resample-speed condition
  + action-label condition
```

参考动作不经过 reference encoder 或 cross-attention。它只在采样器侧用于：

- `skip_timesteps` 的 img2img 初始化；
- 每步 clamp 已知区域的 inpainting；
- clamp 已有帧、生成新增帧的 outpainting；
- 必要时在采样前先 retarget 到目标骨架。

基础训练目标是 `MSE(predicted_x0, target_x0)`，不是噪声预测。temporal attention 是全窗口
注意力；loop 使用圆周相位、数据增强与闭合损失，不使用 temporal mask。

下面各节展开说明。

## 1. 整体结构

AnyTop 是一个直接在动作特征上去噪的扩散 Transformer。每个“帧 × 关节”是一个 token：先在
同一帧的全部关节之间做 graph-aware spatial attention，再让每个关节沿时间做 temporal
attention，最后由因子化注意力输出的预测就是干净动作。骨架信息（rest pose、关节名称、关节
之间的成对关系与拓扑距离）以加性 embedding 和 attention bias 的形式进入网络。

在主干之上，模型还包含：

- 基于全身轨迹池化的 cross-limb temporal pathway，让不同肢体交换运动节奏；
- loop 圆周相位、loop 数据增强与闭合损失；
- per-joint structural channel 和 whole-joint name dropout；
- per-joint species × joint FiLM 与 species CFG；
- 三槽 action-label 条件与 action CFG；
- resample-speed 条件；
- 可选的 global topology conditioner（骨架汇总 → 每层 AdaLN，见 §4.4）；
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
  ├─ frozen-T5 joint-name embedding → 可选 species FiLM → trainable projection
  ├─ joint structural descriptor → MLP
  └─ absolute sinusoidal frame PE
  + unreliable_embedding（按待重绘区域的 mask 加入）
  │
  ▼
tokens [T+1, B, J, D]
  │
  │  timestep condition bus
  │    sinusoidal diffusion timestep
  │      + resample-speed token
  │      + action-label token
  │    (is_loop 不进这条总线，见 §6.1)
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

输入与输出采用统一的 canonical 特征空间，每个关节有 12 个通道：位置残差 0:3、
6D rotation 3:9、速度 9:12。位置相对 rest pose 表达，并按骨架长度尺度与全语料统计量
归一化；速度只保留根关节 X/Z 两个通道，其他速度通道为零。普通关节的模型投影读写
前 9 个通道，根关节另外读写 X/Z 速度。

### 3.2 关节名称

T5 编码发生在预处理阶段，模型不会在每次 forward 中运行 T5。`cond.npy` 携带
`joints_names_embs`；启用 `species_joint_cond` 时，先经过第 4.1 节的 per-joint FiLM，
再由 `InputProcess.text_embedding` 投影到 latent width，并沿全部时间位置加到对应关节 token。

`joint_name_drop_prob>0` 时，训练会把整个关节名称向量置零（与 padding 行同为全零，
padding 已被 attention mask 和 loss 排除，不会混淆）。它不同于普通 element-wise dropout：目的是真正隐藏关节名字，让模型
回退到 rest geometry、pairwise topology 和 structural channel。

推理时处理新骨架（`process_new_skeleton` 与 server 的常驻 T5 注入）：embedding text 不在 checkpoint cond.npy 里的关节，名字直接
置空（空文本经 T5 masked mean 恰为全零行），不在线编码；原文本记在
`joints_names_embs_meta['blanked_unseen_texts']`。

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
  + resample_speed
  + action_label
```

组合后的 condition 在每个 decoder layer 中经该层自己的 `embed_timesteps` 再注入残差流。

### 4.1 Per-joint species FiLM 与 CFG

| 通路 | 输入 | 作用位置 | drop 语义 |
|---|---|---|---|
| `species_joint_cond` | `[joint_name || species]` → bottleneck（默认 128） | 每个 joint-name embedding 的 FiLM | `species_cfg_drop_prob` 按样本 bypass 到 identity |

Species descriptor 只通过这条输入层通路调制关节名，不进入 timestep condition bus。
设 T5 embedding 宽度为 `D_t5`，FiLM 头的结构为：

```text
[joint_name_emb || species_emb]
  → Linear(2 * D_t5, 128) → GELU → Linear(128, 2 * D_t5)
  → gamma_residual, beta

modulated_name = (1 + gamma_residual) * joint_name_emb + beta
  → text_embedding → latent width
```

隐藏宽度由 `species_film_bottleneck` 控制，默认 128，设为 0 时使用 `latent_dim`。
最后一层权重和偏置初始化为零，因此初始调制为 identity。

训练时，`species_cfg_drop_prob` 按样本独立抽样；被丢弃样本的所有关节绕过整条 FiLM，
直接将关节名 embedding 送入 `text_embedding`。推理时 `species_active=False` 使用同一条
identity 分支，CFG 按 `uncond + scale * (cond - uncond)` 组合预测。这里的 uncond 仅移除
显式 species 调制，关节名、骨架和其他条件仍保留。

`joint_name_drop_prob` 则按关节独立抽样，在 FiLM 之前把整条名字向量清零，FiLM 也读取
清零后的名字；它与 species CFG drop 独立。只丢名字时，FiLM 仍可从 `[0 || species_emb]`
产生 beta；两者都丢时，进入 `text_embedding` 的名字向量为零。普通 element-wise name
dropout 位于 whole-joint name drop 之后；FiLM 头读取 element-wise dropout 之前的名字，
而调制作用于 element-wise dropout 之后的名字。

### 4.2 Action label

Action label 使用受控词表，不接收自由文本。词向量表作为冻结 buffer 保存在 checkpoint
内；head 和 modifier 使用预计算的 T5 向量，direction 使用正交方向码。按三个角色槽聚合：

1. head/action；
2. direction；
3. modifier。

各槽独立计算归一化加权均值，空槽为零；head 槽的首词权重为 1.5，其余词权重为 1。
direction 槽只读取方向码的有效轴。三个槽拼接后由 MLP 投影并加到 timestep condition。训练时
`action_label_cfg_drop_prob` 把部分样本送到 learned null embedding；推理时
`action_label_cfg_scale>1` 用 conditional/unconditional 两次 forward 做 CFG。

`--action_label_adaln` 给同一个 action token 再加一路**乘性**通路：
`GraphMotionDecoder.action_adaln` 一次 matmul 出全部层的 (γ, β)，每层拿自己的
(B, 4, d) 切片，在 temporal 与 FFN 的**分支输入**上做 `(1+γ)·x + β`；残差流和 spatial 分支不碰。
零初始化，驱动向量与加性路径共用，所以 CFG 的无条件分支自动走 null。

`--action_adaln_bottleneck` 设置这个头的隐层宽度，0 表示使用 `latent_dim`。
它以 action token 为输入，经 bottleneck 生成各层的调制参数。

### 4.3 输出特征空间与 resample speed

`canonical_feature_mean/std` 定义模型写入的输出坐标/标准化空间。全语料只有一套：每个
cond entry 携带同一对向量（loader 用 `assert_single_canonical_frame` 校验），所以它不是
模型输入——输出空间对所有物种相同，也就不携带任何 body plan 信息。它们仍随 `y` collate，
供只读 `y` 的解码路径（训练 aux-loss）反标准化。

`resample_speed_cond = source_frames / internal_frames`，经另一条 MLP 注入。它告诉模型当前
时间窗口相对源动作的压缩/拉伸程度。数据侧的 `motion_speed_aug` 是另一项无显式条件的数据
增强，不应与 `resample_speed_cond` 混为一谈。

### 4.4 Global topology conditioner

`--topology_cond`（默认关）把整副骨架汇总成一个向量，乘性地调制每一层。实现是
`motion_transformer.TopologyConditioner`：

- 输入是第 0 个时间位置的 rest-pose token 加 structural latent（§3.3），**不含 joint-name
  embedding**。名字带物种身份，能还原物种的汇总会和 species descriptor 竞争——而推理时用户
  正是改 species_tags 来换步态风格。去掉名字后，这条通路只描述几何比例，改关节名不会动它；
- 若干 learned query 对全部有效关节做 cross-attention（padding 关节被 mask），过残差 FFN，
  对 query 取平均得到每个样本一个向量；
- zero-init 的头把它映射成与 action AdaLN 相同的 (B, L, 4, d) 布局，两者相加后在 temporal 与
  FFN 的分支输入上做 `(1+γ)·x + β`；
- 不做 CFG drop：它描述被驱动的骨架，不是引导方向。

动机：§4 的其他全局条件都是类别量，没有一个带连续的骨架几何；没有这条通路时，整身比例
只能靠每层的 spatial attention 逐帧重建。rest-pose token 的非 root 行在同一物种内是常数（parent-local
residual 在 rest 时为 0），所以这里的几何信息实际来自 structural channel。

启用它时，需要复查 species_tags override（Galloping vs Lumbering、Flapping vs Hovering）的
效果是否被削弱。

数据侧的 `--leaf_drop_prob`（默认 0，val/test 拒绝）给它配了拓扑增强：以该概率删掉所有符合
条件的终端叶子——名称含 Nub / end / site / Container，或（名称没有有效词元或名称向量全零）
且局部旋转静止；再随机删至多两个重复链的末段。重复链的判定是：叶子与父关节的规范名称去掉侧别词和
末尾数字后词干相同，且词干末词不是 leg / arm / foot / hoof / paw / hand；镜像对成对删。
因此可删的末段不局限于 Tail、Finger、Wing、Tongue，也可能是 Ear、Eye 等真实身体部位，
即使该关节有动画；不在重复链上的蹄、下巴、耳、眼或挂在脊柱上的翅膀不会因此被选中。
root、平移根、朝向关节、face 与 contact 标注关节始终保留。每个关节行存的是它自己的
root-relative 位置、局部旋转和速度；叶子的旋转不影响其他关节，所以删行不会改变保留关节的运动。
增强后的样本对应一副末端更短的骨架，而不一定只是去掉辅助节点。
cond 按保留关节切片、索引重映射，structural channel、relation 矩阵和长度尺度 L 都按新骨架重建
（`truebones_utils/leaf_drop.py`）。目的有两个：
对末端节点与链长变化不敏感；不让这里的池化退化成按每个物种的精确拓扑查表。

## 5. Decoder layer

每层的实际顺序是：

```text
x + embedded timestep/conditions
  → Graph Spatial Attention + residual + norm
  → [action AdaLN on the branch input] → Full Temporal Attention + residual + norm
  → Cross-Limb Block（若本层启用）
  → [action AdaLN on the branch input] → FFN + residual + norm
```

方括号里的两处只在 `--action_label_adaln` 下存在，见 §4.2。

该层使用 `nn.TransformerDecoderLayer` 的接口，但不含 encoder-decoder memory attention：
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
“另一条腿的完整时间轨迹”的短路径。Cross-limb block 用 K 条 learned pooling 轨迹补上它：

```text
每帧全部 joints --可学习加权池化--> K 个全身 latents
全身 latents ------时间自注意--> 跨时间节奏上下文
每帧全部 joints <--关节相关的 K 路混合--- K 个全身 latents
```

该路径在窄 bottleneck `cross_limb_dim` 中运行，只在部分层启用，每个启用的层有独立 block；
latent 数量与维度由配置决定。

共享一次 `proj_in` 后，`routing: Linear(d_cl, 2K, bias=False)` 的前 K 个向量
预测读取分数（softmax 沿关节轴），
后 K 个向量预测写回分数（softmax 沿 latent 轴）；两组向量独立学习。输入与 score 向量
先做 fp32 L2 归一化，余弦分数乘 `sqrt(d_cl)`，限制 logit 幅度；投影值不归一化。
两次加权混合均在窄空间内完成；读取和写回使用池化与混合，时序和 Cross-K 使用多头注意力。
读取与写回权重都使用 block 的 dropout；padding 不参与池化或帧不可靠比例统计。

latent 的帧位置信息来自关节 token。池化使用线性投影后的 value，因此可以汇聚其中的
绝对帧 PE 和主干相位信号；block 内没有独立的正弦/圆周时间表或时间表门控。

完整的 block 顺序（`CrossLimbTemporalBlock.forward`）：

```text
learned pooling（per-joint reliability_bias 加在 logits 上）
  → latent temporal self-attention（per-frame temporal_reliability_bias 作为 additive key bias）
  → cross-K self-attention（同一帧的 K 个 latent 互相注意；Pre-Norm + 零初始化 cross_k_scale）
  → joint-dependent K-way mixture → proj_out → residual + LayerNorm
```

“哪些区域不可靠”这条信号分三层进入网络：

- `AnyTop.unreliable_embedding`：全局一个 `d_model` 向量，在 InputProcess 之后按
  `unreliable_mask` 加到输入 token 上，整个 trunk（spatial/temporal attention、FFN，
  以及 cross-limb 的投影值与路由分数）都能看到重绘区域；
- 池化的标量 `reliability_bias`：做帧内的关节级选择；
- latent temporal attention 的标量 `temporal_reliability_bias`：乘以“该帧被标记的有效关节
  占比”作为 key bias，做帧级选择。帧内统一的关节偏置会被 softmax 的平移不变性抵消，
  时序偏置则能区分哪些帧需要重绘。

Cross-K attention 让同一帧的 K 个 latent 在写回前直接交换信息。这些标量都零初始化，
`cross_k_norm`/`cross_k_attn` 藏在零 gate 之后，训练起点等价于没有 cross-K。

`--cross_limb 0` 时不创建分支模块或参数；`unreliable_embedding` 仍作为主干信号保留。
`cross_limb_last_n=0` 表示所有 decoder 层启用，正数表示只启用最后 N 层。

## 6. Loop 路径

Loop 不是单一布尔 token，而是模型、数据和损失共同组成的一条路径。

### 6.1 模型内

- `is_loop` 控制窗口的时间拓扑，不进入 timestep condition bus；
- 主 temporal path 使用 absolute PE，并在 loop 样本上额外加入 circular phase embedding；
- cross-limb latents 通过池化汇聚关节 token 里的帧位置信息（§5.3），没有自己的
  loop/absolute 时间表；
- circular table 在窗口的第一个和最后一个动作帧闭合；第 0 行 rest-pose token 为零。

每层的 phase scale（`temporal_phase_scale`）是以 0 初始化的 learned scalar，控制该层
圆周相位信号的强度。

### 6.2 数据侧

真实 loop clip 会做随机 circular roll 和随机 tile，然后统一 resample 到内部窗口。tile count
和 phase offset 只用于诊断，不直接喂给模型。
roll 和 tile 只作用于**无固定相位**的 loop：label 的每个 head 都在 `PHASE_FREE_LOOP_HEADS`
（步态、振翅、呼吸、保持姿态）里，且每个 modifier 都在 `PHASE_FREE_LOOP_MODIFIERS`（步态/速度、
持续的姿态如 sit/sleep、手持物）里。其余 loop 是**有固定相位**的（attack、roar、hurt、jump 这类
从备战姿态出发又回到备战姿态的事件，以及叠在 idle/hover 上的 look、eat、shake、yawn 这类事件
modifier）：仍然闭合、仍用 circular table 和 wrap 损失，但 offset 固定为 0、tile 固定为 1——窗口
第 0 帧是备战姿态、窗口里只有一次事件，这正是推理端对这类动作开 `--loop` 时要生成的东西。
两张白名单外的词默认按有固定相位处理。`--loop auto` 用同一条规则。
k 份 tile 经周期重采样后是 k 份逐位相同的拷贝（周期 T/k 帧），是 loop 任务里容易的一侧；
`--loop_tile_single_prob` 给单周期窗口（推理端 loop 条件为真、又用自动长度时所在的 regime）的概率设一个下限，
其余质量在 2..max 上仍均匀，默认 0.5；设为 0 时在 2..max 上均匀抽签。

真实 loop clip 以概率 `--loop_cond_prob`（默认 1.0 = 总是）被告知它是 loop；其余的抽签
（`loop_uncond`）仍做它本该有的全部 loop 增广（closing-key drop、周期变速，无固定相位的再加 circular roll、tile），但按开放
clip 重采样窗口、以 `is_loop=False` 交给模型（零 circular phase、开放速度步长、无 wrap 损失）。
这是有意的标签噪声，目的与不把 loop flag 放进条件总线相同：flag 与内容相关时模型会把它
当内容键，而一个不可信的 flag 只能被当作时间拓扑来读。它不是 CFG（没有 null 态、推理端不消费），
eval loader 固定 1.0。另一处降级是超出源帧预算被裁剪的 clip（环被裁开，按非 loop 告知）。

超预算的**有固定相位** clip（label 不满足 `loop_is_phase_free`，loop 与否都算）在变速一步里有一个
速度下限 `min(L/budget, MAX_FIT_SPEEDUP)`：`MAX_FIT_SPEEDUP` 倍预算以内的被加速到恰好放进窗口、
整段进入训练（loop 因此保住 `is_loop`），也不会被 `motion_speed_aug` 再放慢出去；更长的先加速到
`MAX_FIT_SPEEDUP` 倍再裁剪，让窗口尽量多地保留事件。`motion_speed_aug` 在 `[下限, max(R, 下限)]`
里抽，两者是同一次变速。这个下限是数据准备而不是增广，eval loader 同样生效。无固定相位的 clip
（步态、呼吸等，任意一段仍是同一动作）按录制速度裁剪，不受影响。

**高频 clip 永不降采样。** 某个关节的旋转在周期不超过几帧的频段里能量足够大（逐帧锁定的昆虫
振翅、一两帧完成的急速动作；判据见 `truebones_utils/high_frequency.py`，分数随 clip 长度一起缓存在
`cache/motion_lengths.npy`），线性插值一旦把它压到更少的帧就会混叠：周期 4 帧的振翅变成慢拍频，
周期 2 帧的几乎静止。这类 clip 的源帧预算等于窗口本身：变速只减速、且不减速出窗口，tile
不超过窗口，超长的按 `resample_speed = 1` 裁剪到窗口长度而不是压进窗口，也不走上面有固定相位
clip 的强制加速（改为裁剪，事件可能不完整）。超过窗口长度的高频 loop 因此总被裁开、按非 loop
告知。代价是这类 clip 只出现在 `resample_speed <= 1` 的区间。生成端自动选长度和 `--loop auto`
用的 clip-length prior（`utils/clip_length_prior.py`）按同一规则读表：高频 clip 的长度上限是
checkpoint 的窗口，超过窗口的高频 loop 按非 loop 计票，所以不会替这类动作选出训练里没见过的长度
或 loop 条件。

**变速、roll、tile、裁剪和窗口重采样只插值一次。** 这几步都是时间的线性映射，loader 不逐步生成
中间 clip：变速只定下变速后的长度，roll、tile、裁剪只改一个帧偏移（以变速后的帧计，对源可以是
小数）和窗口长度，最后由 `resample_time_scaled_window` 按复合映射（精确有理数）从源 clip 插值一次，
速度通道沿源的位置路径采样后差分，契约与 `resample_motion_features` 相同。两次线性插值连做时，
即使两步都是放大采样，第二步也会切掉落在两个中间帧之间的源关键帧，周期几帧的内容会被削平。
不变速的 clip 不经过它：roll、tile、裁剪按整数帧取下标，窗口重采样是唯一一次插值。

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

模型预测 `x_0`。基础 `l_simple` 是按有效关节、有效帧计算的位置和 6D rotation MSE；
根速度由单独的辅助项约束。其余项均为可选，是否启用
与权重由训练配置决定：

| 损失 | 约束 |
|---|---|
| geodesic | 预测/目标 6D rotation 转 SO(3) 后的角距离 |
| root velocity | 根关节 X/Z 速度相对 GT 的误差，按物种权重缩放 |
| position step | position 相邻帧差分相对 GT 的误差，按 canonical 步长统计与物种权重缩放 |
| bone vector | 父→子骨向量相对 GT 的差（同时管长度和方向），按 rest 骨长归一化（下限为 L 的一个比例），按 `alphas_cumprod[t]` 加权；每个叶子关节另加一根局部坐标系下的单位虚拟子骨，方向各向同性（`‖R_pred − R_gt‖_F / √3`），让叶子的旋转和有子关节的关节一样受约束 |
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

**窗口到输出长度。** 模型总是生成固定的内部窗口，再换算成请求的 M 帧（`_resample_window_to_output`）。
M 小于窗口时，窗口是 loader 会把 M 帧源动作放大成的样子，所以输出按最小二乘**反解**训练时的线性
插值（`invert_resample_motion_features`，速度通道沿位置路径反解），而不是再插值一次：插值会漏掉
落在两个窗口帧之间的源帧，削平几帧周期的内容，反解对精确的窗口逐帧还原、对有噪声的窗口做平均。
M 大于窗口时插值不可逆，仍用线性插值。按帧范围 clamp 参考的 inpaint/outpaint 也保留线性插值：
它的 mask 只放开每个输出帧插值所读的那两个窗口帧，而反解的每个输出帧都以衰减权重读整个窗口，
会把重新生成的帧漏进被 clamp 的参考帧。

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
| leaf-drop 拓扑增强 | `data_loaders/truebones/truebones_utils/leaf_drop.py` |
| reference/inpainting/outpainting | `sample/generate.py` |
