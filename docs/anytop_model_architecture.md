# 当前 AnyTop 模型架构

> 本文描述 `local_root_xz` 当前代码与 `save/merged_locomotion_v10/args.json` 的实际行为。
> 它是架构现状的入口文档；`docs/` 中带“方案”“refactor”“removal”的长文是设计与迁移记录，
> 其中的“现状”、行号和旧训练命令只对文档注明的历史提交有效。

## 1. 先区分原版与当前 fork

原版 AnyTop 的核心是一个直接在动作特征上去噪的 Transformer：每个“帧 × 关节”是一个
token，先做全关节的 graph-aware spatial attention，再让每个关节沿时间做 temporal
attention。骨架条件由 rest pose、关节名称、成对关系和拓扑距离组成。

当前 fork 保留这条主干，并加入：

- Perceiver 式 cross-limb temporal pathway；
- 原生 loop 条件、圆周相位、loop 数据增强与闭合损失；
- 13 维 per-joint structural channel 和 whole-joint name dropout；
- 全局 species FiLM 与 per-joint species × joint FiLM；
- 四槽 action-label 条件与 action CFG；
- canonical output-frame 和 resample-speed 条件；
- subtree / temporal-span 混合噪声训练；
- full temporal attention、QK-Norm 和更细的拓扑关系码。

当前模型**没有** ReferencePriorEncoder、ControlNet 或 reference cross-attention 分支。
参考动作只在扩散采样器侧用于 img2img 初始化、inpainting clamp 和 outpainting。

## 2. 当前生产配置

下表来自 `save/merged_locomotion_v10/args.json`；其他 checkpoint 必须以各自的
`args.json` 为准。

| 项目 | v10 配置 |
|---|---:|
| decoder layers | 8 |
| latent width | 256 |
| FFN width | 2048 |
| attention heads | 4 |
| internal frames | 60 |
| per-joint motion features | 12（position 3 + rotation 6D + velocity 3） |
| training joint cap | 100；推理张量的关节维动态决定 |
| diffusion | cosine schedule，100 steps，预测 `x_0` |
| cross-limb | 8 latents，width 128，只在最后 4 层 |
| action/species conditions | `action_label_cond`、`species_cond`、`species_joint_cond` 全开 |
| loop condition retention | `loop_cond_prob=0.7` |
| parameter count | 16,695,080（不含 frozen action buffers） |

代码中的 `AnyTop.max_joints=143` 是遗留构造参数，不是当前训练数据的 joint cap，也不把
推理强制固定到 143 个关节。训练预处理的上限来自 `param_utils.MAX_JOINTS=100`；新骨架推理
在未启用 crop 时可以使用自己的真实关节数。

## 3. 总体数据流

```text
x_t [B, J, 12, T]
  │
  ├─ motion projection（root / non-root 分开）
  ├─ rest-pose projection，作为第 0 个时间 token
  ├─ frozen-T5 joint-name embedding → trainable projection
  ├─ 13D joint structural descriptor → MLP
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
  └─ 丢弃第 0 个 rest-pose token → 预测 x_0 [B, J, 12, T]
```

模型直接预测干净动作 `x_0`，基础目标是 masked MSE；不是“预测噪声 ε 并和真实噪声做
MSE”。

## 4. 输入 enrichment

### 4.1 动作与 rest-pose token

根关节和普通关节分别使用线性投影。rest pose 也有独立的 root/non-root 投影，并作为领先
时间 token 拼到动作帧前面。因此 decoder 实际处理 `T+1` 个时间位置，输出时再删除第 0 个。

### 4.2 关节名称

T5 编码发生在预处理阶段，模型不会在每次 forward 中运行 T5。`cond.npy` 携带
`joints_names_embs`，`InputProcess.text_embedding` 再把它投影到 latent width，并沿全部
时间位置加到对应关节 token。

`joint_name_drop_prob>0` 时，训练会把整个关节名称向量替换为 learned
`unknown_joint_name`。它不同于普通 element-wise dropout：目的是真正隐藏关节名字，让模型
回退到 rest geometry、pairwise topology 和 structural channel。

### 4.3 关节结构通道

每个关节还有一个与名字无关的 13 维结构描述，包括：

- branch-free run 内的归一化位置和 run 长度；
- run 是否落到 contact、当前 joint 是否 contact、contact 是否已知；
- leaf 标记；
- 归一化高度、左右 signed offset、前后位置、附着高度；
- subtree 大小、sibling rank 和 sibling count 的倒数。

这条通道由 `parents`、物理 rest positions 和 contact annotation 确定。改关节名不会改变它。
它经独立 MLP 投影后加到 token，并在投影后重新清零 padding rows。

## 5. 条件总线

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

### 5.1 Species 的两条通路

| 通路 | 输入 | 作用位置 | drop 语义 |
|---|---|---|---|
| `species_cond` | species T5 descriptor | timestep 的 FiLM：`gamma*t + beta` | 可由 `species_cfg_drop_prob` bypass 到 identity |
| `species_joint_cond` | `[joint_name || species]` | 每个 joint-name embedding 的 FiLM | 始终存在，不做 CFG drop |

两条 FiLM 的最后一层都以 identity 语义初始化。需要注意：只关闭 timestep species FiLM
并不构成完全的 species-unconditional forward，因为 per-joint species FiLM 仍然开启。

### 5.2 Action label

Action label 使用受控词表，不接收自由文本。每个词先查 checkpoint 内冻结的 T5 word table，
再按四个角色槽聚合：

1. head/action；
2. direction；
3. modifier；
4. hands。

四个槽拼接后由 MLP 投影并加到 timestep condition。训练时
`action_label_cfg_drop_prob` 把部分样本送到 learned null embedding；推理时
`action_label_cfg_scale>1` 用 conditional/unconditional 两次 forward 做 CFG。

### 5.3 Canonical frame 与 resample speed

`canonical_feature_mean/std` 定义模型当前写入的输出坐标/标准化空间。这两组 12 维向量经
zero-init MLP 投影并始终注入，不可 CFG-drop；缺失时 forward 直接失败。

`resample_speed_cond = source_frames / internal_frames`，经另一条 MLP 注入。它告诉模型当前
时间窗口相对源动作的压缩/拉伸程度。数据侧的 `motion_speed_aug` 是另一项无显式条件的数据
增强，不应与 `resample_speed_cond` 混为一谈。

## 6. Decoder layer

每层的实际顺序是：

```text
x + embedded timestep/conditions
  → Graph Spatial Attention + residual + norm
  → Full Temporal Attention + residual + norm
  → Cross-Limb Block（若本层启用）
  → FFN + residual + norm
```

基类 `nn.TransformerDecoderLayer` 创建的 `self_attn` 和 `multihead_attn` 已被删除；当前层不含
encoder-decoder memory attention，也不消费 `reference_memory`。

### 6.1 Graph-aware spatial attention

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

当前共有：

- 12 个 directed edge relation codes：self、parent、child、sibling、保留的
  no-relation、end-effector、ancestor、descendant、sibling-limb、三档 cousin；
- 13 个 topology-distance codes：0–4 hop 精确编码；更远的 pair 再按是否共线以及
  normalized LCA depth 分档。

Q/K 的 topology/edge embedding 表由所有 decoder layers 共享。`value_emb=True` 时还会把
对应关系 embedding 注入 value/output；v10 的 `value_emb=false`，因此当前走 PyTorch SDPA
快路径。

### 6.2 Full temporal attention

Temporal attention 把张量视为 `[T+1, B*J, D]`，同一关节的每个时间 token 可以看到整个
窗口，包括 rest-pose token。当前没有 `temporal_window`，也没有 causal、local 或 loop
attention mask。

### 6.3 Cross-limb temporal block

基础的因子化 spatial/temporal attention 没有一条让“某条腿的完整时间轨迹”直接观察
“另一条腿的完整时间轨迹”的短路径。Cross-limb block 用 K 个 learned latents 补上它：

```text
每帧全部 joints --cross-in--> K 个全身 latents
全身 latents ------时间自注意--> 跨时间节奏上下文
每帧全部 joints <--cross-out--- K 个全身 latents
```

该路径在窄 bottleneck `cross_limb_dim` 中运行，每个 active layer 有独立 block。v10 在最后
4 层使用 8 个、128 维的 latent。

训练/推理可以向 cross-in logits 加一个 learned `reliability_bias`，降低或提升标记为不可靠
的 joint/frame。当前它是标量 bias：如果某一帧所有有效关节都被同样标记，softmax 的平移
不变性会抵消它，所以它不能单独表达“整帧不可靠”。相应的改进目前只记录在
`cross_limb_reliability_cost_effective_fix.md`，尚未进入模型。

## 7. Loop 路径

Loop 不是单一布尔 token，而是模型、数据和损失共同组成的一条路径。

### 7.1 模型内

- `is_loop` 经 MLP 加到 timestep condition；
- 主 temporal path 保留 absolute PE，并在 loop 样本上额外加入 circular phase embedding；
- cross-limb latents 自己没有输入级 absolute PE，因此 loop 样本选择 circular table，非 loop
  样本选择 absolute table；
- circular table 在窗口的第一个和最后一个动作帧闭合；第 0 行 rest-pose token 为零。

两处 phase scale 都是 learned scalar，并以 0 初始化。

### 7.2 数据侧

真实 loop clip 会做随机 circular roll 和随机 tile，然后统一 resample 到内部窗口。tile count
和 phase offset 只用于诊断，不直接喂给模型。

`loop_cond_prob` 是“真实 loop 在训练时保留显式 loop 条件的概率”，不是循环强度。v10 的
0.7 表示约 30% loop-shaped 样本仍执行 roll/tile，但对模型隐藏 `is_loop`，训练无显式条件下
识别循环结构的能力。

### 7.3 损失侧

- `loop_wrap_loss`：loop 首尾的 position、rotation 和 terminal velocity 闭合；
- `loop_root_xz_closure_loss`：translation root 的整周期 XZ velocity 积分闭合。

再次强调：当前 loop 路径**不使用 loop-aware temporal mask**。

## 8. 混合可靠性训练

基础扩散先按统一 timestep 生成 `x_t`，随后可对局部区域使用独立且不早于原 timestep 的
噪声等级重新加噪：

- subtree joint perturbation：随机选择预算内的非根子树；
- temporal-span perturbation：随机选择连续帧，并覆盖该样本的全部真实关节；
- 两者可以取并集；
- supervision target 不变，被扰动单元仍参与 loss 和 attention；
- `cross_limb_unreliable_mask` 可把位置告知 cross-limb；
- `unreliable_mask_drop_prob` 会在部分样本上隐藏这张图，让模型自行定位损坏区域。

这是一种训练 curriculum，不是额外的 reference encoder。

## 9. 损失

模型预测 `x_0`。基础 `l_simple` 是按有效关节、有效帧计算的 MSE；其余项均为可选：

| 损失 | 约束 | v10 权重 |
|---|---|---:|
| geodesic | 预测/目标 6D rotation 转 SO(3) 后的角距离 | 0.1 |
| velocity consistency | position finite difference 与 velocity channel 一致 | 0.2 |
| bone length | 相对 GT、按 rest length 归一化的骨长 | 0（关闭） |
| loop wrap | loop 首尾闭合 | 0.04 |
| loop root XZ closure | 根 XZ 速度整周期积分为零 | 0.05 |
| temporal-span seam | span 边界附近 position 二阶差分匹配 GT | 0.2 |

## 10. 参考动作与编辑：采样能力，不是模型分支

当前参考动作流程不会生成 `reference_memory`，也不会在 decoder 中做 cross-attention：

- plain reference generation：参考动作作为 `init_image` 在 `skip_timesteps` 指定的噪声等级
  开始 img2img；
- inpainting：被编辑区域自由生成，已知区域在每一步投影回相应噪声等级的 reference；
- outpainting：已有帧 clamp，新增帧从纯噪声生成；需要时使用两阶段流程；
- cross-skeleton reference：先在采样前 retarget 到目标骨架特征空间，再进入上述流程。

因此 reference/inpainting/outpainting 能力不增加 AnyTop forward 的参数量。

## 11. 当前明确不存在或已删除的设计

- Reference Attention / ReferencePriorEncoder / reference memory；
- ControlNet 式并行参考分支；
- windowed temporal attention 与 `--temporal_window`；
- loop temporal attention mask；
- global-energy conditioning；
- 预测噪声 ε 的训练目标；
- action multi-hot 和整句 action-label embedding；当前是 per-word、role-slot T5 条件。

## 12. 代码入口

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
| 当前生产配置 | `save/merged_locomotion_v10/args.json` |
