# AnyTop 条件调制升级方案：additive bus → 分支级 affine modulation

> 状态：**设计建议，尚未实施**
> 范围：`action_label`、`resample_speed_cond`、`canonical_feature_mean/std`、`is_loop`
> 目标：保留当前稳定的 additive condition bus，同时验证哪些条件值得进一步用于
> hidden-state / residual-branch 的乘性或仿射调制。
> 非目标：本方案不改变动作特征格式、扩散目标、action-label 词表、CFG 语义或数据标注。

---

## 1. 结论

当前方案不是结构性错误。把全局条件投影后加到 timestep embedding，并由每个 decoder layer
独立投影、重复注入，是有效且常见的全局条件基线。条件随后进入 attention、LayerNorm 和带
GELU 的 FFN，因此模型仍能在深层形成非线性的条件交互；不能把它理解成“条件只能改变最终
输出偏置”。

真正的限制是：**注入点本身只有 shift，没有显式的 `condition × activation` 交互**。
模型若要根据条件放大某些通道、改变 temporal/spatial 分支的相对强度，必须在后续层间接学出。
对于天然描述尺度、速率和增益的条件，这条路径表达效率偏低。

建议采用渐进式升级，而不是把所有 additive token 一次性替换成乘法：

1. 保留 additive bus，作为 timestep、离散语义和旧行为的稳定主路径；
2. 第一优先给 `log(resample_speed_cond)` 增加 temporal-branch affine modulation；
3. 第二优先把 canonical mean/std 用于显式 affine modulation；
4. action label 保留 additive shift，再通过 per-layer scale/shift/gate 增强控制；
5. loop 继续以 circular phase / 边界结构为主，不优先改成普通通道 FiLM；
6. 所有新增调制 head 以 identity 语义初始化，并通过等参数量消融决定是否保留。

## 2. 当前实现边界

当前条件总线位于 [`AnyTop.forward`](../model/anytop.py)：

```text
sinusoidal diffusion timestep
  → species FiLM
  + resample_speed token
  + canonical-frame token
  + loop token
  + action-label token
```

组合结果在每个 `GraphMotionDecoderLayer` 中经过该层独立的 `embed_timesteps`，再广播加到所有
帧、关节 token：

```python
x = x + self.embed_timesteps(timesteps_emb).view(1, bs, 1, self.d_model)
```

需要区分以下事实：

- `action_label`、`resample_speed_cond` 和 canonical frame 当前主要走 additive token；
- `species_cond` 已经对 timestep embedding 做 `gamma * t + beta`；
- `species_joint_cond` 已经对每个 joint-name embedding 做 species × joint FiLM；
- `is_loop` 除了 additive token，还控制每层 temporal attention 使用的 circular phase signal；
- `action_group` 在服务端用于选择专用 checkpoint，不是模型 forward 内的 condition token。

因此本方案针对的不是“AnyTop 完全没有乘性条件”，而是：**decoder hidden state 及其
spatial / temporal / FFN 残差分支没有条件相关的乘性调制。**

## 3. Additive bus 的能力与限制

令各条件投影之和为 `c`，第 `l` 层接收：

```text
b_l = W_l(t + c)
x_l' = x_l + b_l
```

由于 `W_l` 为线性层：

```text
W_l(t + c_action + c_speed + ...) = W_l(t) + W_l(c_action) + W_l(c_speed) + ...
```

### 3.1 它能做到什么

- 每层拥有独立的 `W_l`，同一条件可以在不同深度产生不同方向的 shift；
- shift 会改变 attention 的 query/key/value 以及后续 FFN 输入，并非只改变输出均值；
- attention、normalization 和 GELU 可以在深层形成条件与 token content 的非线性交互；
- 离散、全局的 action 语义用 additive token 表达是合理的。

### 3.2 它不擅长什么

- 没有直接表示 `(1 + gamma(c)) * x` 的短路径；
- 多种条件先在同一个 latent 空间求和，可能发生幅值竞争或语义纠缠；
- 同一个向量广播到所有帧和关节，无法直接定位 action 对哪个 limb、哪个时间区间生效；
- speed 与 diffusion timestep 之间的交互只能由后续层间接恢复；
- 条件若主要定义特征尺度，网络仍需从 additive token 推断应该缩放哪些 hidden channels。

这些是学习效率和可控性问题，不是表达能力的绝对缺失。足够深、足够宽的现有网络原则上仍可
近似相同映射，因此是否升级必须由消融而非结构直觉决定。

## 4. 条件与调制机制的匹配

| 条件 | 语义类型 | 建议机制 | 优先级 |
|---|---|---|---:|
| `resample_speed_cond` | 连续时间尺度、步频 | additive token + temporal affine modulation；输入改用 `log(speed)` | 1 |
| canonical mean/std | 输出空间的平移与尺度定义 | affine modulation，或长期统一到单一规范空间 | 2 |
| `action_label` | 离散、可组合语义 | 保留 additive shift，再增加 per-layer scale/shift/gate | 3 |
| `species_cond` | 全局形态与动力学先验 | 现有 timestep FiLM 保留；可纳入 block modulation 消融 | 4 |
| `is_loop` | 边界拓扑、周期相位 | circular phase / loop-aware temporal structure 为主 | 不优先 |
| direction / hands 等 action slot | 空间局部语义 | 后续考虑 joint/limb-aware conditioning 或 cross-attention | 独立课题 |

### 4.1 Resample speed

`resample_speed_cond = source_frames / internal_frames` 表示窗口相对源动作的时间压缩或拉伸，
最接近“增益/频率”型条件，因而是乘性调制的首选。

建议先使用：

```text
s = log(clamp(resample_speed_cond, min=eps))
e_speed = MLP(FourierFeatures(s))  # 小数据时也可先只用 MLP(s)
```

`log` 空间使互为倒数的快放/慢放围绕 0 分布，也避免原始正数比例的明显偏态。第一阶段只让
`e_speed` 调制 temporal attention 与 FFN，不调制 spatial attention，以减少作用面并提高
消融可解释性。

但必须避免过度承诺：速度不仅改变 hidden-channel 幅度，也改变时间频率。若 temporal FiLM
收益有限，下一步应让 speed 调制 temporal positional frequency / phase，而不是无限扩大 FiLM。

### 4.2 Canonical frame

canonical mean/std 定义模型写入的坐标空间，其中 std 本身就是尺度。只用 additive token 时，
网络必须从一个全局向量间接推断每个 feature channel 的 gain。

短期建议把 `[mean || log(std)]` 同时保留在 additive bus，并送入每层 affine modulation。
mean 更自然地影响 shift，std 更自然地影响 scale，但不要用硬编码把二者完全隔离；由独立 MLP
学习 `gamma/beta`，再通过监控验证是否符合预期。

长期更干净的方向是让所有样本在同一规范化空间训练，并在模型外做确定性的坐标变换。如果
canonical frame 可以完全从模型目标中移除，优先选择确定性变换，而不是用更复杂的条件网络
补偿混合坐标系。

### 4.3 Action label

action label 需要引入“walk / attack / turn”等新语义，因此不能只靠乘法：纯 gate 只能重标定
已有 feature，缺少自然的语义 shift。建议保留现有四槽 additive token，同时让 action
embedding 参与 affine modulation。

如果目标是提升 `left/right`、`hand1/hand2` 等局部可控性，全局 AdaLN 仍可能太粗。那类问题
应单独设计 slot → joint/limb 的定向注入，或者引入 action-token cross-attention，而不是继续
堆叠全局 FiLM。

### 4.4 Loop

loop 的核心不是“放大哪些通道”，而是窗口首尾具有周期邻接关系。当前 circular phase signal
已经比普通 FiLM 更贴合问题。建议保留 additive loop token 和 circular phase 路径，优先验证：

- circular phase 是否真正被每层的 `temporal_phase_scale` 使用；
- loop wrap loss 与采样期闭合处理是否一致；
- 是否需要 loop-aware relative position / circular temporal attention。

只有证据显示 loop token 的全局控制不足时，才将它纳入 affine head。

## 5. 建议结构

### 5.1 条件保持分源，调制允许晚融合

不要仅保留一个不可解释的总和。先分别构造：

```text
e_t       = timestep embedding
e_action  = existing four-slot action projection
e_speed   = projection(log(resample_speed_cond))
e_frame   = projection(canonical mean, log std)
e_loop    = loop projection
e_species = species descriptor projection
```

additive bus 可以继续使用：

```text
e_add = e_t + e_action + e_speed + e_frame + e_loop
```

同时为 affine path 显式 concat 或分源求和：

```text
e_mod = concat(e_t, e_action, e_speed, e_frame[, e_species, e_loop])
```

concat 让 modulation head 知道信息来自哪个条件，避免在总和中先丢失来源。若参数预算不允许
concat，可以让每种条件先经过独立线性层，再求和；不要直接复用未经分源的 `e_add`。

### 5.2 每层、每分支的 affine modulation

理想结构采用 Pre-Norm 风格。对第 `l` 层的分支
`m ∈ {spatial, temporal, ffn}`：

```text
(gamma_lm, beta_lm, gate_lm) = ModHead_lm(e_mod)
z_lm = (1 + gamma_lm) * Norm(x) + beta_lm
x = x + (1 + gate_lm) * Branch_lm(z_lm)
```

三个分支使用不同参数，因为条件的作用不应被强制相同：

- speed 预计主要改变 temporal 与 FFN；
- morphology / canonical scale 可能同时影响 spatial 与 temporal；
- action 可能改变三者，但不同 action slot 的最优分配未知。

当前 `GraphMotionDecoderLayer` 是 Post-Norm 顺序。直接改成完整 Pre-Norm 会同时改变训练动力学，
使“条件调制收益”与“Norm 架构变化”无法区分。因此首轮实验建议采用最小侵入版本：

```text
x_cond = (1 + gamma_lm) * x + beta_lm
branch = Branch_lm(x_cond)
x = Norm(x + (1 + gate_lm) * branch)
```

先在现有 Post-Norm 主干中验证调制价值。只有确认有效后，再单独比较 Pre-Norm 重构。

### 5.3 Identity 初始化

新增 head 的最后一个 Linear 必须零初始化：

```text
gamma = 0
beta  = 0
gate  = 0
```

这里 residual multiplier 使用 `1 + gate`，从而 fresh model 初始行为与当前分支一致。不要在现有
主干上直接采用 `gate * branch` 且把 gate 零初始化，否则所有 transformer branch 会在初始时
被关闭，比较的不再只是条件调制。

必要时可用有界参数化限制 OOD 条件：

```text
scale = 1 + a * tanh(gamma)
gate  = 1 + b * tanh(gate_residual)
```

默认先不用 clamp；只有监控发现 scale 爆炸或自定义 action/species 输入导致 OOD 崩坏时再加。

## 6. 分阶段实现计划

### 阶段 A：仅 speed → temporal modulation

- 保留当前 speed additive token；
- speed 输入改为或额外加入 `log(speed)`；
- 每层只给 temporal branch 增加 `gamma/beta`；
- 暂不增加 residual gate，避免一次引入三个自由度；
- 所有 head zero-init；
- 与等训练配置、等 seed 的 additive baseline 对照。

这是最小、因果最清楚的一步。若没有收益，不应继续把同一设计无差别铺到所有条件。

### 阶段 B：canonical frame → affine modulation

- 输入使用 `[mean || log(std)]`；
- 先调制 spatial/temporal 的输入 hidden state；
- 保留原 canonical additive token，做 `additive only / affine only / both` 三臂消融；
- 检查不同 object subset 下的 feature-channel 误差，而不只看全局 loss。

### 阶段 C：action → scale/shift/gate

- 保留现有四槽 additive action token 和 CFG null embedding；
- conditional 与 unconditional forward 必须同时切换 action modulation；
- 先把四槽拼接后的 action embedding 送进 modulation head；
- 若 direction/hands 仍弱，再研究 slot-specific、joint-aware 路径。

### 阶段 D：统一 block conditioner（可选）

只有 A–C 中至少一项被消融证实有效后，才考虑将 timestep、action、speed、frame、species
统一成 block conditioner，并为 spatial / temporal / FFN 一次性产生 modulation 参数。

## 7. CFG 与 condition dropout 约束

新增 action modulation 后，CFG 的 unconditional branch 不能只把 additive action token 换成
`action_label_null_emb`；所有由 action 驱动的 `gamma/beta/gate` 也必须读取同一个 null 状态。
否则 conditional/unconditional 两次 forward 的差异不再只代表 action，CFG 语义会被破坏。

建议把“是否 active”的处理放在 action embedding 构造阶段：

```text
action_repr = active ? projected_action : learned_null_action
```

随后 additive path 与 modulation path 都只读取 `action_repr`，避免维护两套 drop 判断。

`resample_speed_cond`、canonical frame 和 loop 当前不是 action CFG 要引导掉的对象；action CFG 的
unconditional branch 必须保持这些条件完全一致。

## 8. Checkpoint 与训练兼容性

任何新增 per-layer head 都会产生新 state-dict keys。当前严格加载逻辑下：

- 旧 checkpoint 不能无条件加载到新结构；
- resume 与 inference 必须根据 checkpoint `args.json` 重建相同架构；
- 如果新功能由默认 `False` 的显式 flag 控制，旧 checkpoint 可继续走完全不创建新参数的旧路径；
- 不能只靠 `strict=False` 静默吞掉调制 head，因为这会让实验误以为加载了已训练能力。

建议新增单一实验开关时按作用域命名，例如：

```text
--speed_temporal_film
--canonical_block_film
--action_block_adaln
```

这些名字只用于首轮消融。确定最终方案后再考虑合并成统一配置，避免在尚无证据时引入一个
语义宽泛、难以复现实验的 `--use_film`。

## 9. 消融矩阵

首轮不要直接做“全条件 AdaLN vs 当前模型”，否则即使结果变化也无法定位来源。推荐顺序：

| Arm | Additive bus | Speed temporal affine | Canonical affine | Action affine/gate |
|---|---:|---:|---:|---:|
| A0 baseline | ✓ | — | — | — |
| A1 speed | ✓ | ✓ | — | — |
| B1 frame | ✓ | — | ✓ | — |
| C1 action | ✓ | — | — | ✓ |
| D1 combined | ✓ | 仅纳入已胜出的项 | 仅纳入已胜出的项 | 仅纳入已胜出的项 |

若要判断提升来自“乘性”还是单纯增加参数，再为获胜 arm 加一个等参数量 additive MLP 对照。

## 10. 验收指标

### 10.1 通用训练指标

- 同 step 的 validation `l_simple`、velocity、geometry、loop-wrap loss；
- 收敛速度与最终质量同时记录，避免只看某个中途 checkpoint；
- 参数量、单 step 时间、峰值显存和采样延迟；
- `gamma/beta/gate` 的均值、std、分位数及按层范数；
- 条件置换测试：固定噪声和骨架，只替换一个条件，其他输入完全不动。

### 10.2 Speed 专项

- 目标 speed 与生成 motion 的周期、步频、root velocity 的单调性；
- 未见过的中间 speed 插值；
- speed reciprocal 对（如 `0.75` 与 `1/0.75`）是否表现近似对称；
- action identity、步幅、接触稳定性是否随 speed 改变而意外崩坏；
- temporal modulation 的 `gamma` 是否随 `log(speed)` 有系统变化，而非始终接近 0。

### 10.3 Canonical frame 专项

- 分 object subset、分 feature channel 的去标准化后误差；
- 不同 canonical std 下的骨长、关节角和 root displacement 一致性；
- scale 是否主要响应 std、shift 是否主要响应 mean；这是诊断信号，不设为硬约束。

### 10.4 Action 专项

- 固定 skeleton/seed 下的 action adherence；
- head、direction、modifier、hands 四槽分别置换；
- 未见组合的 compositional generalization；
- CFG scale sweep，观察更强 guidance 是否只增强动作语义而不破坏几何；
- 对称词（left/right、hand1/hand2）的关节局部响应。

### 10.5 Loop 专项

- 首尾 position、rotation、velocity discontinuity；
- loop/non-loop 条件置换后的实际闭合变化；
- circular phase scale 的层间分布；
- 不以“普通 FiLM 参数不为零”替代真实闭合质量。

## 11. 采纳与回退标准

建议在训练前冻结明确门槛，至少满足：

1. 对目标条件的专项指标稳定优于 additive baseline；
2. 提升超过等参数量 additive 对照，而不只是参数增多；
3. 非目标条件、几何质量和 loop 质量无显著回退；
4. modulation 参数确实偏离 identity，且变化与条件有可解释相关性；
5. 推理成本与显存增量在服务预算内。

出现以下情况应回退：

- `gamma/beta` 长期接近 0：模型不需要该路径；
- 参数明显变化但专项指标不变：调制被模型用于无关补偿；
- action adherence 上升但 skeleton geometry 或 motion smoothness 明显下降；
- speed 控制只改变动作幅度，不改变实际周期/速率；
- 效果只能由更高参数量解释。

## 12. 推荐的最终判断

- **不是缺陷修复，而是有明确目标的容量重分配。** 当前 additive bus 应保留为 baseline。
- **speed 是最值得先试的乘性条件**，但重点应放在 temporal branch，并准备进一步调制时间编码。
- **canonical std 天然适合 scale、mean 天然适合 shift**；更长期的方案是统一坐标空间。
- **action 不应由 additive 全面替换为 pure gate**；推荐 additive semantic shift 与 affine
  modulation 并存。
- **loop 优先解决周期结构，不优先解决通道增益。** 当前 circular phase 路径比普通 FiLM
  更符合其语义。
- 若分支级调制实施，spatial / temporal / FFN 应分开产参、identity-init，并严格维护 CFG
  conditional/unconditional 的同源条件表示。

