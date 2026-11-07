# AnyTop 模型架构

本文档描述 AnyTop 的模型结构：输入特征、条件总线、decoder、扩散训练与损失，以及参考动作在
采样侧的编辑使用方式。层数、宽度、各种概率和损失权重由训练配置决定；正文以结构与数据流为主，
参考配置的参数预算见 §11。实现细节以代码为准。

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

`--reference_motion` 仅用于编辑，必须搭配 `--skip_timesteps`。
动作只在采样器侧用于：

- `skip_timesteps` 的 img2img 初始化；

编辑在目标骨架上进行，参考必须是目标同一套骨架：已预处理的 `.npy`，或以文件自带 bind pose 为 rest pose 的 `.glb/.fbx`（骨架层级与目标不一致时报错），不做 retarget。

基础训练目标是 `MSE(predicted_x0, target_x0)`，不是噪声预测。temporal attention 是全窗口
注意力；loop 使用条件 token、圆周相位、数据增强与闭合损失，不使用 temporal mask。

下面各节展开说明。

## 1. 整体结构

AnyTop 是一个直接在动作特征上去噪的扩散 Transformer。每个“帧 × 关节”是一个 token：先在
同一帧的全部关节之间做 graph-aware spatial attention，再让每个关节沿时间做 temporal
attention，最后由因子化注意力输出的预测就是干净动作。骨架信息（rest pose、关节名称、关节
之间的成对关系与拓扑距离）以加性 embedding 和 attention bias 的形式进入网络。

在主干之上，模型还包含：

- loop 条件、圆周相位、loop 数据增强与闭合损失；
- per-joint structural channel 和 whole-joint name dropout；
- per-joint species × joint FiLM（所有行开启）；
- 按角色分槽的 action-label 条件与 action CFG；
- resample-speed 条件（输出坐标/标准化空间全语料统一，不作为模型输入）；
- full temporal attention、QK-Norm 和细分的拓扑关系码。



## 2. 总体数据流

```text
x_t [B, J, F, T]
  │
  ├─ motion projection（全部关节共用，含根关节）
  ├─ rest-pose projection，作为第 0 个时间 token
  ├─ frozen-T5 joint-name embedding → 可选 species FiLM → trainable projection
  ├─ joint structural descriptor → MLP
  └─ absolute sinusoidal frame PE
  │
  ▼
tokens [T+1, B, J, D]
  │
  │  timestep condition bus
  │    sinusoidal diffusion timestep
  │      + resample-speed token
  │  │      + action-label token
  │    (is_loop 不进这条总线，见 §6.1)
  │
  ▼
GraphMotionDecoder × L
  ├─ graph-aware spatial attention（每帧、跨全部关节）
  ├─ full temporal attention（每关节、跨全部 T+1 token）
  └─ FFN
  │
  ▼
output projection（全部关节共用）
  │
  └─ 丢弃第 0 个 rest-pose token → 预测 x_0 [B, J, F, T]
```

模型直接预测干净动作 `x_0`，基础目标是 masked MSE，而不是预测噪声 ε。关节维 `J` 由输入
骨架决定，模型不硬编码关节数：训练预处理对关节数有上限，推理张量按实际骨架的关节数动态
构造。

## 3. 输入特征

### 3.1 动作与 rest-pose token

全部关节（含根关节）共用同一个动作线性投影：根的 position 通道是它的世界坐标（见 §4.3），与其他关节的 RIC 位置含义一致，根靠关节名区分。rest pose 也有独立的 root/non-root 投影，并作为领先
时间 token 拼到动作帧前面。因此 decoder 实际处理 `T+1` 个时间位置，输出时再删除第 0 个。

### 3.2 关节名称

T5 编码发生在预处理阶段，模型不会在每次 forward 中运行 T5。`cond.npy` 携带
`joints_names_embs`；先经过第 4.1 节的 per-joint FiLM，
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
- leaf 标记；
- subtree 大小的 log2 分桶（绝对大小，不除以 J，token 不携带关节总数）；
- sibling rank 和 sibling count 的倒数；
- 左右侧符号 side（左 +1 / 中 0 / 右 −1），读 `joint_side_labels`，与 `joint_relations` 的侧别同源；关节名文本不含左右词，侧别只从这里进入 token。

这条通道只由 `parents`、物理 rest positions 和 `joint_side_labels` 确定。改关节名不会改变它。
它经独立 MLP 投影后加到 token，并在投影后重新清零 padding rows。

## 4. 条件总线

扩散 timestep 首先做 sinusoidal embedding，然后按以下顺序组合条件：

```text
timestep
  + action_label
```

`resample_speed` 不在这条总线上，见 4.3。

组合后的 condition 在每个 decoder layer 中经该层自己的 `embed_timesteps` 再注入残差流。

### 4.1 Per-joint species FiLM

| 通路 | 输入 | 作用位置 | drop 语义 |
|---|---|---|---|
| species FiLM | `[joint_name || species_code]` → bottleneck | 每个 joint-name embedding 的 FiLM | 所有行开启；无随机 drop |

Species code 只通过这条输入层通路调制关节名，不进入 timestep condition bus。
`species_code = [T5(body plan) || variant 轴]`（`species_descriptor_table.species_code`）；
variant 是正交码（每个 variant 一根轴，取值为该 body plan 行的模长，无 variant 时全零），不经过 T5：T5 下 `Winged` 与 `Winged Fluttering` 几乎重合。
设 T5 embedding 宽度为 `D_t5`、species code 宽度为 `D_code`，FiLM 头的结构为：

```text
[joint_name_emb || species_code]
  → Linear(D_t5 + D_code, latent_dim) → GELU → Linear(latent_dim, 2 * D_t5)
  → gamma_residual, beta

modulated_name = (1 + gamma_residual) * joint_name_emb + beta
  → text_embedding → latent width
```

隐藏宽度固定为 `latent_dim`。
最后一层权重和偏置初始化为零，因此初始调制为 identity。

FiLM 在所有行上开启，没有独立的 CFG drop 或 null 模式。

`joint_name_drop_prob` 则按关节独立抽样，在 FiLM 之前把整条名字向量清零，FiLM 也读取
清零后的名字。只丢名字时，FiLM 仍可从 `[0 || species_code]`
产生 beta；两者都丢时，进入 `text_embedding` 的名字向量为零。普通 element-wise name
dropout 位于 whole-joint name drop 之后；FiLM 头读取 element-wise dropout 之前的名字，
而调制作用于 element-wise dropout 之后的名字。

### 4.2 Action label

Action label 使用受控词表，不接收自由文本。每个词先查 checkpoint 内冻结的 T5 word table，
再按角色槽聚合：

1. head/action；
2. direction（正交码，不经过 T5）；
3. modifier。

各槽的通道拼接后由 MLP 投影并加到 timestep condition。训练时
`action_label_cfg_drop_prob` 把部分样本送到 learned null embedding；推理时
`action_label_cfg_scale>1` 用 conditional/unconditional 两次 forward 做 CFG，仅作用于高噪声半程（`t ≥ T/2`），细节步退回单次 conditional forward。

启用 action label 时，同一个 action token 另有一路**乘性**通路：
`GraphMotionDecoder.action_adaln` 一次 matmul 出全部层的 (γ, β)，每层拿自己的
(B, 4, d) 切片，在 temporal 与 FFN 的**分支输入**上做 `(1+γ)·x + β`；残差流和 spatial 分支不碰。
零初始化，驱动向量与加性路径共用，所以 CFG 的无条件分支自动走 null。
设计细节见 [conditional_modulation_upgrade.md](conditional_modulation_upgrade.md) §3。

这个头的隐藏宽度固定为 `latent_dim`。

### 4.3 输出空间与 resample speed

`canonical_feature_mean/std` 定义模型写入的输出坐标/标准化空间。全语料只有一套：每个
cond entry 携带同一对向量（loader 用 `assert_single_canonical_frame` 校验），所以它不是
模型输入——输出空间对所有物种相同，也就不携带任何 body plan 信息。它们仍随 `y` collate，
供只读 `y` 的解码路径（训练 aux-loss）反标准化。

每个关节的模型通道是 position 与 rotation，没有速度。非根关节的 position 是 RIC（世界坐标减去根的
XZ）；根的 position 是它的世界坐标，XZ 从 clip 第 0 帧（预处理放在原点）起算，loader 在任何时间
阶段之前由存储的根速度积分得到，之后工作布局里不再有速度（`canonical_features.stored_to_working`），
所以所有时间阶段（变速、roll、tile、窗口重采样）都只插值位置与旋转。根的世界 XZ 分布比关节
的 RIC 位置窄得多，所以统计向量在每通道统计之后另带根 position 的一组 mean/std：X、Z 共用一个
标量（水平面保持各向同性），Y 一个。导出时 `working_to_stored` 把解码结果写回存储格式：根的
position 回到 RIC，速度行写入根的世界步长（读存储 clip 的代码只读这一路速度），其余关节的速度行为零。

`resample_speed_cond = source_frames / internal_frames`（`L = resample_speed_cond · T`）逐帧注入，
不与 action label 共用条件 token，避免帧数变成压过标签的内容索引：

- open 窗口：**逐帧物理时间**，窗口第 `i` 帧对应源帧 `i · (L - 1) / (T - 1)`（与 loader 的 resample step 一致），
  经 sinusoidal embedding 和 `tempo_projection` 加到该帧所有关节的 token 上；
- loop 窗口：**一个周期的物理长度** `L / k`，每帧相同，经独立的
  `loop_period_projection`。帧在周期内的位置由 §6.1 的循环表给出；斜坡会在接缝处跳回 0。
  两种行用不同的头，是因为同一个输入值在 open 行是帧时间、在 loop 行是周期长度。

两个头的末层都零初始化。

训练和推理始终保留 tempo 条件，open 和 loop 行都不丢弃，没有 tempo CFG。
推理的长度默认来自 `clip_length_prior`。

数据侧的 `motion_speed_aug` 改变源长度 `L`，模型经 tempo 条件看到，不应与
`resample_speed_cond` 本身混为一谈。

## 5. Decoder layer

每层的实际顺序是：

```text
x + embedded timestep/conditions
  → Graph Spatial Attention + residual + norm
  → [action AdaLN on the branch input] → Full Temporal Attention + residual + norm
  → Cross-Limb Block（若本层启用）
  → [action AdaLN on the branch input] → FFN + residual + norm
```

方括号里的 action AdaLN 见 §4.2。

该层沿用了 `nn.TransformerDecoderLayer` 的外形，但不含 encoder-decoder memory attention：
基类的 `self_attn` 换成 graph/temporal 两条路径，`multihead_attn` 已删除。

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
  leaf（无子关节的关节在对角线上的标记）、ancestor、descendant、sibling-limb、按归一化 LCA 深度分档的 cousin，
  以及双侧镜像伴侣 mirror_twin（在存储矩阵中以 flag 叠加在拓扑码上，训练和推理均完整保留）；
  no-relation 码保留编号但不再产生；每个 edge code 再按 `joint_side_labels` 乘上侧别
  （neutral / same_side / opposite_side，同样以 flag 存储，嵌入索引为 `code + NUM_EDGE_CODES * lateral`），
  否则左手到左脚与到右脚的拓扑码完全相同，只交换一个肢带左右时注意力偏置不变；
  训练和推理均完整保留侧别信息；
- topology-distance codes：近距离 hop 精确编码；更远的 pair 再按是否共线以及
  normalized LCA depth 分档。

Q/K 的 topology/edge embedding 表由所有 decoder layers 共享。`value_emb=false` 时关系
embedding 只进入 attention score、不进入 value/output，注意力走 PyTorch SDPA 快路径。

### 5.2 Full temporal attention

Temporal attention 把张量视为 `[T+1, B*J, D]`，同一关节的每个时间 token 可以看到整个
窗口，包括 rest-pose token。窗口就是内部帧数加一，没有 causal、local 或 loop attention
mask。

## 6. Loop 路径

Loop 不是单一布尔 token，而是模型、数据和损失共同组成的一条路径。

### 6.1 模型内

- `is_loop` **不**进 timestep condition。语料里这个 flag 与内容强相关（loop 类以
  stationary idle 为主；同一 species-tag 簇里 loop 标注的步态与 one-shot 标注的步态不同）。
  一旦把它作为全局 token 加到 condition bus，模型就会把它学成内容键：推理时 `--loop on`
  会借同簇其它 loop 的小幅度、相位散乱步态，而不是本物种 one-shot 的步态。
  一个闭合窗口是时间拓扑的陈述，phase table 是唯一能表达它的地方；
- 窗口的周期数 `k = y['loop_tile_count']`（loader 的 tile 数；loop clip 只在整条长度上闭合，
  所以一份 tile 就是一个周期），推理由 `--loop_cycles` 给出；open 行不读 k；


推理端 k 与长度的解析规则
（`sample/output_lengths.resolve_loop_cycles`）见 `docs/loop_cycle_condition.md` §4。

### 6.2 数据侧

真实 loop clip 会做随机 circular roll 和随机 tile，然后统一 resample 到内部窗口。tile count
在 loop 条件生效的行上就是模型的周期数 k（§6.1）；phase offset 只用于诊断。
roll 和 tile 只作用于**无固定相位**的 loop：label 的每个 head 都在 `PHASE_FREE_LOOP_HEADS`
（步态、振翅、呼吸、保持姿态）里，且每个 modifier 都在 `PHASE_FREE_LOOP_MODIFIERS`（步态/速度、
持续的姿态如 sit/sleep、手持物、细碎次级动作如 look/eat/yawn/scratch/sniff）里。其余 loop 是**有固定相位**的（attack、roar、hurt、jump 这类
从备战姿态出发又回到备战姿态的事件，以及叠在 idle/hover 上的 bite、throw、taunt 这类一次性事件
modifier）：仍然闭合、仍用 circular table 和 wrap 损失，但 offset 固定为 0、tile 固定为 1——窗口
第 0 帧是备战姿态、窗口里只有一次事件，这正是推理端对这类动作开 `--loop` 时要生成的东西。
两张白名单外的词默认按有固定相位处理。`--loop auto` 用同一条规则。
k 份 tile 经周期重采样后是 k 份逐位相同的拷贝（周期 T/k 帧），是 loop 任务里容易的一侧；
`--loop_tile_single_prob` 给单周期窗口（推理端 loop 条件为真、又用自动长度时所在的 regime）的概率设一个下限，
其余质量在多份 tile 的取值范围上仍均匀；设为 0 时全部在多份 tile 上均匀抽签。

真实 loop clip 以概率 `--loop_cond_prob` 被告知它是 loop（取 1 即总是）；其余的抽签
（`loop_uncond`）仍做它本该有的全部 loop 增广（closing-key drop、周期变速，无固定相位的再加 circular roll、tile），但按开放
clip 重采样窗口、以 `is_loop=False` 交给模型（零 circular phase、无回绕步、无 wrap 损失）。
这是有意的标签噪声，目的与不把 loop flag 放进条件总线相同：flag 与内容相关时模型会把它
当内容键，而一个不可信的 flag 只能被当作时间拓扑来读。它不是 CFG（没有 null 态、推理端不消费），
eval loader 总是告知。另一处降级是超出源帧预算被裁剪的 clip（环被裁开，按非 loop 告知）。

**高频 clip 永不降采样。** 某个关节的旋转在周期不超过几帧的频段里能量足够大（逐帧锁定的昆虫
振翅、一两帧完成的急速动作；判据见 `truebones_utils/high_frequency.py`，分数随 clip 长度一起缓存在
`cache/motion_lengths.npy`），线性插值一旦把它压到更少的帧就会混叠：周期只有几帧的振翅变成慢拍频，
更短周期的几乎静止。这类 clip 的源帧预算等于窗口本身：变速只减速、且不减速出窗口，tile
不超过窗口，超长的按 `resample_speed = 1` 裁剪到窗口长度而不是压进窗口。超过窗口长度的高频
loop 因此总被裁开、按非 loop 告知。代价是这类 clip 只出现在 `resample_speed <= 1` 的区间。生成端自动选长度和 `--loop auto`
用的 clip-length prior（`utils/clip_length_prior.py`）按同一规则读表：高频 clip 的长度上限是
checkpoint 的窗口，超过窗口的高频 loop 按非 loop 计票，所以不会替这类动作选出训练里没见过的长度
或 loop 条件。

**变速、roll、tile、裁剪和窗口重采样只插值一次。** 这几步都是时间的线性映射，loader 不逐步生成
中间 clip：变速只定下变速后的长度，roll、tile、裁剪只改一个帧偏移（以变速后的帧计，对源可以是
小数）和窗口长度，最后由 `resample_time_scaled_window` 按复合映射（精确有理数）从源 clip 插值一次。两次线性插值连做时，
即使两步都是放大采样，第二步也会切掉落在两个中间帧之间的源关键帧，周期几帧的内容会被削平。
不变速的 clip 不经过它：roll、tile、裁剪按整数帧取下标，窗口重采样是唯一一次插值。

### 6.3 损失侧

- `loop_wrap_loss`：loop 首尾的 rotation 闭合；
- position step 的回绕步 `pos[0] - pos[-1]`：loop 首尾的 position 闭合，根的世界 XYZ 在内，
  所以根轨迹的整周期闭合也由它监督。

loop 路径不使用 loop-aware temporal mask。

## 7. 扩散训练

训练只使用统一 timestep 的标准扩散加噪 `q_sample(x_0, t)`，不做局部区域重加噪，也不提供可靠性掩码条件。

TODO：以后训练专门的 inpaint/outpaint 模型；motion diffusion 本身不再承载这些能力。

## 8. 损失

模型预测 `x_0`。基础 `l_simple` 是按有效关节、有效帧计算的 MSE，覆盖 position 与 rotation 通道
（根的世界坐标在内）；速度不是模型通道。其余项是否启用与权重由训练配置决定：

| 损失 | 约束 |
|---|---|
| position step | 每个关节逐帧位置步长对 GT 的 MSE（loop 样本含首尾回绕步）；根的步长是它的世界 XYZ 步长，所以也监督根轨迹 |
| position acceleration | 位置误差的二阶差分，按位置步长 RMS 与物种权重归一，loop 含回绕二阶差分 |
| root position | 根位置 XYZ 在 canonical 标准化单位下的 MSE，不按关节数摊薄、不乘物种权重 |
| root step / root acceleration | 单独取根的 XYZ 步长及其二阶差分，单位与物种权重同 position step |
| geodesic | 预测/目标 6D rotation 转 SO(3) 后的角距离 |
| bone vector | 父→子骨向量相对 GT 的差（同时管长度和方向），按 rest 骨长归一化（下限为 L 的一个比例），按 `alphas_cumprod[t]` 加权；每个叶子关节另加一根局部坐标系下的单位虚拟子骨，方向各向同性（`‖R_pred − R_gt‖_F / √3`），让叶子的旋转和有子关节的关节一样受约束 |
| loop wrap | loop 首尾 rotation 闭合 |

## 9. 参考动作与编辑

编辑用法（带 `--skip_timesteps`）不把参考送入 decoder，参考必须与目标同物种，用途为：

- plain reference generation：参考动作作为 `init_image`，从 `skip_timesteps` 指定的噪声等级
  开始 img2img；

整段参考动作仅作为扩散初始状态，不作为 decoder 条件。

跨骨架参考动作迁移由 [motion autoencoder](motion_autoencoder.md) 承担。扩散模型不编码参考动作，动作标签、species FiLM 与 loop 周期条件始终遵循各自的常规规则。

**窗口到输出长度。** 模型总是生成固定的内部窗口，再换算成请求的 M 帧（`_resample_window_to_output`）。
M 小于窗口时，窗口是 loader 会把 M 帧源动作放大成的样子，所以输出按最小二乘**反解**训练时的线性
插值（`invert_resample_motion_features`），而不是再插值一次：插值会漏掉
落在两个窗口帧之间的源帧，削平几帧周期的内容，反解对精确的窗口逐帧还原、对有噪声的窗口做平均。
M 大于窗口时插值不可逆，仍用线性插值。

## 10. 代码入口

| 主题 | 文件 |
|---|---|
| AnyTop 输入、条件与 forward | `model/motion_diffusion.py` |
| graph/temporal attention | `model/motion_transformer.py` |
| diffusion 构建与 `x_0` 目标 | `utils/model_util.py` |
| 扩散训练与损失 | `diffusion/gaussian_diffusion.py` |
| joint structural channel | `data_loaders/truebones/truebones_utils/joint_struct_features.py` |
| refined topology codes | `data_loaders/truebones/truebones_utils/topology_relations.py` |
| loop roll/tile/resample | `data_loaders/truebones/data/dataset.py` |
| 参考动作 img2img | `sample/generate.py` |
| 共享静态骨架几何 | `data_loaders/reference.py` |

## 11. 宽度分配与设计依据

参数预算由训练配置（`train_diffusion_full.bat`）决定：主干宽度、FFN 宽度、层数、
`last_layer_ff`。action AdaLN 和 species FiLM 的隐藏宽度均固定为 `latent_dim`。冻结的 T5 词表以 buffer 形式保存，
不计入可训练参数。训练启动时会打印各模块的参数量。

参数主要分布在以下模块：decoder 主干（占绝大部分）、`action_adaln` 头、
`action_label_projection`、`input_process`（per-joint species FiLM、joint-name 投影、structural 投影）、
`tempo_projection`，以及可忽略的 `output_process` 与各层共享的 relation/topology embedding 表。

各处宽度的分配依据，来自训练后权重的秩与单元参与率：

- **中间层是饱和的**：中间层 FFN 的隐单元几乎全部参与，注意力 q/k/v/o 的有效秩接近满宽。
  宽度没有余量，因此主干保持全宽。
- **末层 FFN 坍缩**：末层 FFN 的单元增益远低于其它层，大量隐单元不参与。末层之后只有
  `LayerNorm → Linear` 的线性读出，不需要全宽 FFN，因此用 `last_layer_ff` 收窄。
- **`action_label_projection` 只读紧凑通道**：direction 槽的正交码只占少数几个轴，其余列的输入
  恒为 0，是结构性死参数；投影只读 `compact_slot_channels`，不读完整槽宽。
