# Cross-Limb 可靠性与 Latent 通信：高性价比修复方案

> 状态：已实现（2026-09-12，`CKPT_VERSION` 9 → 10，从头重训）。实现记录见第 10 节。
> 范围：缺陷 1（整帧可靠性信号失效）、缺陷 2（latent 间缺少直接通信），以及可靠性 flag 的训练/推理语义对齐

## 1. 目标

在不把 Cross-Limb block 改造成重型 Perceiver、也不引入 `(L × K)` 全局注意力的前提下：

1. 让模型真正感知哪些关节、哪些帧正在被重绘；
2. 保留已有的关节级可靠性选择能力；
3. 让同一帧内的 `K` 个 latent 能直接交换信息；
4. 避免模型把 flag 简化理解成“这里总是比当前 timestep 脏很多”；
5. 新增路径以 no-op 起步，不改变原有结构的初始行为。

本文统一使用以下记号，避免把序列长度和 diffusion 总步数都写成 `T`：

- `F = nframes`：运动帧数；
- `L = F + 1`：Transformer 序列长度，索引 0 是 T-pose token；
- `T_diff`：diffusion 总步数；
- `K`：Cross-Limb latent 数量。

推荐的最小完整结构为：

```text
输入 token + per-joint unreliable embedding
    ↓
Cross-in（保留现有关节级 reliability bias）
    ↓
Temporal latent attention（新增帧级 reliability key bias）
    ↓
Cross-K attention（新增，一次，Pre-Norm + gated residual）
    ↓
Cross-out（保持现状）
```

## 2. 缺陷 1：可靠性信号的根治方案

当前 `unreliable_mask` 只作为 cross-in logits 的标量偏置。当某一帧所有有效关节均被标记时，每个有效 key 得到相同偏置，softmax 的平移不变性会将其精确抵消。因此，整帧 inpaint 和 temporal span mask 对该路径等价于没有提供可靠性信息。

性价比最高的修复由两个互补改动组成。

### 2.1 在输入 token 上加入 unreliable embedding

在 `InputProcess` 之后、进入 Transformer trunk 之前，将 mask 转成可学习的 token 类型嵌入：

```python
self.unreliable_embedding = nn.Parameter(torch.zeros(d_model))

x = x + unreliable_mask.unsqueeze(-1) * self.unreliable_embedding
```

模型内部统一后的 `unreliable_mask` 采用 `[L, B, J]`，其中 `L = nframes + 1`：

- 不可靠位置为 `1`，可靠位置为 `0`；
- `unreliable_mask[0]` 即 T-pose 行，固定为 `0`；
- padding joint 固定为 `0`。

训练侧仍可传入原始 `[B, F, J]` mask，但必须先拼接可靠的 T-pose 行并转成 `[L, B, J]`，再用于 embedding、temporal key bias 和各 Cross-Limb block。推理侧如果已经传入 prepared mask，不得重复拼接。

作用：

- trunk 的 spatial attention、temporal attention 和 FFN 都能感知重绘区域；
- cross-in 的 value、cross-out 的 query 也间接包含可靠性信息；
- 即使整帧所有关节都被标记，也不会因 softmax 平移不变性而消失；
- 新增参数仅为 `d_model` 个。

初始值设为零，可保证新增路径在初始化时是 no-op。

### 2.2 给 latent temporal attention 增加帧级 key bias

从关节 mask 汇总出每帧的不可靠度，只统计有效关节：

```python
valid = ~joints_key_padding_mask                 # [B, J]

frame_unreliable = (
    (unreliable_mask * valid.unsqueeze(0)).sum(dim=-1)
    / valid.sum(dim=-1).clamp_min(1).unsqueeze(0)
)                                                # [L, B]
```

然后将它转换为 temporal attention 的 additive key bias：

```python
self.temporal_reliability_bias = nn.Parameter(torch.zeros(1))   # 与 block 内既有 time_emb_scale / reliability_bias 同形

temporal_key_bias = (
    self.temporal_reliability_bias
    * frame_unreliable.transpose(0, 1)[:, None, :]
)                                                # [B, 1, L]

temporal_key_bias = (
    temporal_key_bias.expand(B, K, L)
    .reshape(B * K, L)
)                                                # [B*K, L]
```

不需要为它新开接口。`SelectiveMultiheadAttention` 的 `key_padding_mask` 传 float 时本身就是 additive key bias：
`need_weights=True` 分支的 `_apply_key_padding_mask` 和 SDPA 分支的 `_key_padding_mask_to_additive` 都做 `scores + key_padding_mask[:, None, None, :]`，只有 bool 输入才走 `-inf` 的 padding 语义。因此直接把 `[B*K, L]` 的 float bias 作为 `key_padding_mask` 传入即可：

```python
zt, _ = self.temporal_attn(
    zt_in, zt_in, zt_in,
    attn_mask=None,
    key_padding_mask=temporal_key_bias,      # float [B*K, L]，additive；None 时行为不变
    need_weights=False,
)
```

`unreliable_mask is None` 时传 `None`，与现状完全一致。如果只是嫌 `key_padding_mask` 这个名字在这里有误导性，可以给 `SelectiveMultiheadAttention.forward` 加一个 `key_bias=` 关键字别名并映射到同一条路径；不要另写一条并行的 additive 代码路径，那只会复制现有逻辑并多出一处需要和 SDPA/非 SDPA 分支保持一致的地方。

作用：

- temporal attention 能降低对不可靠帧作为信息源的依赖；
- temporal span 和整帧 inpaint 的标记会沿时间维产生非均匀偏置，不再被 softmax 抵消；
- 现有 cross-in bias 继续负责关节级选择，新增 temporal bias 负责帧级选择。

如果所有时间、所有关节都不可靠，那么时间 bias 仍可能成为均匀偏置并被抵消。这是合理行为：此时不存在可供选择的可靠信息源。

### 2.3 为什么两个改动都保留

两者解决的问题不同：

| 改动 | 解决范围 | 不能单独覆盖的部分 |
| --- | --- | --- |
| unreliable embedding | 让所有 trunk/token 路径知道目标位置正在被重绘 | 不直接约束 temporal attention 应少读哪些帧 |
| temporal key bias | 让 latent 时间流显式区分可靠帧与不可靠帧 | 不能让普通 joint trunk 全程感知 mask |

只增加 embedding 已能修复“模型完全看不到整帧 mask”的核心问题；再增加一个 temporal 标量 bias，成本极低，却能补上显式的信息源选择机制，因此建议一起实现。

### 2.4 同批修复 flag 的训练/推理语义差

增强 flag 之后，必须同时处理现有训练分布与 inpaint 推理分布的差异：

- 训练时，标记区域被替换为 `q_sample(x_0, t_random, fresh_noise)`，且当前实现令 `t_random` 均匀取自 `[t, T_diff)`；
- 推理时，每轮输入模型的自由区域与已夹住区域都处于当前同一名义 timestep。自由区域来自反向过程自身的状态，已知区域来自参考动作在同一级别的 `q_sample`；
- `skip_timesteps > 0` 的第一步更极端：自由区域起初也是同一级别的加噪参考，却仍被标记。

100-step、训练 `t` 均匀采样时，当前分布满足 `E[t_random - t] = 24.75`。它会使 flag 与“额外高噪声”形成很强的训练相关性。输入 embedding 落地后，模型更容易利用这一相关性，因此语义对齐必须与可靠性可见性一起实施。

推荐第一版把 `_sample_renoise_timesteps` 改成混合分布：

```python
same_level_prob = 0.5
same_level = torch.rand(t.shape, device=device) < same_level_prob

# 保留现有的困难样本分支：UniformDiscrete[t, T_diff)
span = (self.num_timesteps - t).clamp(min=1).to(torch.float32)
offset = (
    torch.rand(t.shape, device=device, dtype=torch.float32) * span
).to(t.dtype)
t_hard = (t + offset).clamp(max=self.num_timesteps - 1)

t_random = torch.where(same_level, t, t_hard)
```

这使训练同时覆盖：

- `50%` 的“被标记但与全局同级”，解除 flag 与严重额外噪声的绑定；
- `50%` 的原始困难重加噪，继续训练利用可靠上下文修复局部损坏。

`0.5` 是首选消融起点，不视为理论最优值。`t_random = t` 且只更换独立高斯噪声时，其单步边缘分布与普通 `q(x_0, t)` 相同；它的主要作用是纠正 flag 的条件语义，而不是完整模拟反向过程自由区域的历史误差。

也可以试验把困难分支收窄为 `[t, min(t + Δ, T_diff))`，但不建议第一版只保留窄区间，否则可能削弱严重遮挡/缺失区域的修复训练。更稳妥的消融顺序是先使用“同级 + 原始困难分支”，确认收益后再调 `same_level_prob` 或 `Δ`。

该改动不增加 state-dict key，但它改变了训练语义；按照本项目的 checkpoint contract，仍必须随本批修改升级 `CKPT_VERSION`。

## 3. 缺陷 2：latent 通信的高性价比方案

不建议直接将 temporal attention 改成对全部 `(L × K)` token 做全局注意力。更合适的是保留现有沿时间的 axial attention，并在其后增加一次沿 `K` 维的 self-attention：

```python
self.cross_k_norm = nn.LayerNorm(d_cl)
self.cross_k_attn = SelectiveMultiheadAttention(d_cl, nheads, dropout=dropout)
self.cross_k_scale = nn.Parameter(torch.zeros(1))
```

forward 伪代码：

```python
# z: [L, B, K, d_cl]，已经完成 temporal attention
zk = z.permute(2, 0, 1, 3).reshape(K, L * B, d_cl)

zk_norm = self.cross_k_norm(zk)
delta, _ = self.cross_k_attn(
    zk_norm,
    zk_norm,
    zk_norm,
    need_weights=False,
)

zk = zk + self.cross_k_scale * delta
z = zk.reshape(K, L, B, d_cl).permute(1, 2, 0, 3)
```

这样，每一帧中的 `K` 个 latent 在进入 cross-out 前可以直接交换时间摘要和肢体相位信息，而不必依赖“cross-out → 下一层 spatial attention → 下一层 cross-in”的间接路径。

推荐采用：

- 一次 cross-K attention；
- Pre-Norm；
- 零初始化 residual scale；
- attention dropout 与 block 内其余三个 attention 对齐；
- 暂不增加 latent FFN。

其中 `dropout` 是 `CrossLimbTemporalBlock.__init__` 已有的形参，由 `GraphMotionDecoder` 以 `decoder_layer.dropout1.p` 传入，cross-in / temporal / cross-out 三个 attention 都用它。cross-K attention 必须显式传同一个值：`SelectiveMultiheadAttention` 的 `dropout` 默认是 `0.0`，漏传不会报错，只会让这一个子层在 `--dropout_prob 0.1` 下静默地不做 attention dropout，与同一 block 内其余子层的正则强度不一致。

以 `d_cl=128`、4 个 Cross-Limb block 为例，新增参数约 26.5 万。cross-K attention 相对原 temporal attention 的理论注意力开销约为 `K/L`；当 `K=8、L=61` 时约为 13%，占整个模型的额外开销通常更低。

### 为什么暂不加 latent FFN

latent FFN 可以提高表达能力，但它不直接解决“K 个 latent 完全不通信”这个主要限制。现有 joint trunk 和投影层已经提供大量非线性变换，因此第一版先增加 cross-K attention；只有消融实验显示其表达瓶颈仍明显时，再补一个小型 latent FFN。

## 4. 明确不做的设计

第一版不建议加入以下结构：

- `(L × K)` 全局 self-attention；
- 每层重复注入 unreliable embedding；
- cross-out target gate；
- latent FFN；
- 多层 latent processor；
- 复杂的 source/value 双重门控。

这些设计可能有收益，但会显著扩大参数、算力、调参空间或改变旧模型分布，并非修复当前两个问题所必需。

## 5. 实施顺序

建议分三组 commit 或消融项落地：

1. **语义对齐**：`t_random=t` 与原始困难重加噪的混合分布；
2. **可靠性可见性**：输入 unreliable embedding + temporal frame key bias；
3. **latent 通信**：单层 cross-K attention。

这样可以分别回答：

- 整帧/时间段 mask 是否真正生效；
- 同级 flag 样本是否减少 inpaint 后期的过度重猜、平均化或幅度衰减；
- latent 直接通信是否改善肢体相位与跨肢体协调。

## 6. 初始化与版本

新增参数建议初始化为：

```python
unreliable_embedding = 0
temporal_reliability_bias = 0
cross_k_scale = 0
```

在该初始化下，新结构与不含这三条路径的网络输出一致或只存在浮点误差。

### 6.1 版本升级

本方案同时改变 state-dict 布局和训练语义，必须升级 `CKPT_VERSION`。以本文编写时的源码为准，当前值是 9，因此落地版本应升为 10；如果实现前版本已经变化，则从届时版本继续加一。新模型从头训练，不从旧权重迁移。

如果担心 `cross_k_scale=0` 使 attention 分支早期梯度较弱，可以将其初始化为很小的正数（如 `1e-3`）。

## 7. AdamW 参数分组

当前训练代码把所有参数放进一个 AdamW 参数组，因此新增的可靠性参数、residual gate、LayerNorm gain/bias 也都会承受 weight decay。零值本身不会继续被衰减，但参数学成非零后会持续被拉向零。

推荐对以下参数关闭 weight decay：

- 本方案新增的：`temporal_reliability_bias`、`cross_k_scale`、`cross_k_norm.weight` 与 `cross_k_norm.bias`；
- Cross-Limb block 内既有的同类零初始化 gate 标量：`reliability_bias`、`time_emb_scale`；
- `GraphMotionDecoderLayer` 内同类的 `temporal_phase_scale`；
- 第一版可同时对 `unreliable_embedding` 关闭 weight decay，并把是否衰减作为消融项。

把既有的三个标量一并纳入，是因为它们和 `temporal_reliability_bias` 是同一种参数（零初始化、学成非零后被 WD 拉回零与初始化方向相反），只免衰减新加的那一个而让同一个 block 里的兄弟继续衰减，规则不自洽；而本批改动本来就升级 `CKPT_VERSION`、从头重训，多纳入这三个既有标量不增加任何兼容成本。是否进一步扩展到全模型所有 norm gain/bias（含 QKNorm gain，它们被拉向 0 而不是 1），仍作为独立训练改动另行记录和消融，本批不做。

实现上优先使用受控的参数名规则（按 `named_parameters()` 的名字后缀匹配），不要仅以 `param.ndim` 判断：上述标量统一采用 `torch.zeros(1)`，与 `nn.LayerNorm` 的 gain/bias 同为 1-D，按维度无法把它们和不该免衰减的参数区分开。

改变参数组会使旧 optimizer state 与新 optimizer 不兼容，这与从头重训的策略一致。

## 8. 必须通过的验证

### 8.1 零初始化测试

- `unreliable_embedding=0`、`temporal_reliability_bias=0`、`cross_k_scale=0` 时，新实现与旧实现输出一致或只存在浮点误差；
- 在上述初始化下，mask 与 no-mask 输出一致，这是预期的 no-op；
- 无 mask 与全零 mask 输出一致。

### 8.2 可靠性机制测试

以下测试必须先把被测的新参数显式设为非零，不能在 no-op 初始化下断言 mask 有影响：

- 将 `unreliable_embedding` 设为非零后，整帧 mask 相比无 mask 必须产生非零输出差异；
- 将 `temporal_reliability_bias` 设为负值后，包含可靠帧和不可靠帧的 temporal span mask 必须改变输出；
- 单关节 mask 的原有行为必须保留；
- T-pose token 与 padding joint 不得被误标；
- 不同 batch sample 的 mask 不得互相泄漏。

### 8.3 latent 通信测试

- 在 temporal attention 后扰动某一个 latent；
- 将 `cross_k_scale` 显式设为非零并开启 cross-K 后，其余 latent 输出应发生变化；
- 将 `cross_k_scale` 设为零后，其余 latent 的变化应重新归零。

### 8.4 训练分布测试

- 固定随机种子验证 `t_random >= t` 且 `t_random < T_diff`；
- 验证 same-level 分支始终返回 `t_random=t`；
- 大样本统计中 same-level 比例应接近配置值，默认约为 `0.5`；
- 困难分支仍应覆盖明显高于 `t` 的 timestep；
- cond-dropout 只隐藏发布给模型的 flag，不应改变实际重加噪分布。

### 8.5 回归、质量与性能测试

- 对比训练和推理显存、吞吐与耗时；
- 分别评估 joint inpaint、整帧 in-betweening 和 temporal span corruption；
- 至少对 `same_level_prob ∈ {0.0, 0.5, 1.0}` 做消融；
- 重点观察 masked-region error、已知区域漂移、边界速度/加速度突变、跨肢体相位一致性、生成多样性与动作幅度；
- “后期过度重猜、平均化、幅度衰减”作为待验证风险，不预先写成确定结论。

## 9. 最终建议

首选方案不是全面重写 Cross-Limb，而是四个配套改动：

1. 训练重加噪采用“同级 + 困难样本”的混合分布；
2. 一个全局共享的 per-joint unreliable embedding；
3. 每个 Cross-Limb block 一个 temporal reliability 标量；
4. 每个 Cross-Limb block 一次轻量 cross-K attention。

它们分别补齐“flag 训练/推理语义不过度错位”“mask 对整网可见”“不可靠帧可被时间注意力识别”“latent 可直接通信”四个缺口。新增网络路径通过零初始化安全接入；训练语义改变则通过版本升级管理，模型从头重训。若消融结果仍显示 latent 表达不足，再考虑加入小型 FFN；在此之前没有必要引入完整 Perceiver 或 `(L × K)` 全局注意力。

## 10. 实现记录（2026-09-12）

四个改动全部按第 9 节落地；与方案正文的差异只在下面标注的几处。

| 方案项 | 代码位置 | 备注 |
| --- | --- | --- |
| 同级 + 困难混合重加噪 | `GaussianDiffusion._sample_renoise_timesteps`；`renoise_same_level_prob` 构造参数 | 训练 flag `--renoise_same_level_prob`（默认 0.5，training 组）；0 / 1 走短路，不多消耗一次 rand |
| per-joint unreliable embedding | `AnyTop.unreliable_embedding`，在 `input_process` 之后相加 | **不**受 `cross_limb` 开关门控：它是 trunk 级信号，`cross_limb=False` 的模型同样生效。mask 的 raw/prepared 归一化抽成 `AnyTop._prepare_unreliable_mask` |
| temporal frame key bias | `CrossLimbTemporalBlock.temporal_reliability_bias`；float `key_padding_mask` 走既有 additive 路径 | 没有新增 `key_bias=` 别名，直接传 `key_padding_mask` 并在调用处注释语义 |
| cross-K attention | `cross_k_norm` / `cross_k_attn` / `cross_k_scale`，在 temporal attention 之后、cross-out 的 `(K, T*B, d)` 布局上直接做 | dropout 显式传 block 的 `dropout` |
| AdamW 分组 | `train.training_loop.build_optimizer_param_groups`，按名字后缀 `NO_WEIGHT_DECAY_PARAM_SUFFIXES` | 覆盖第 7 节全部条目，含 `unreliable_embedding`；LayerNorm 其它 affine 照旧衰减 |
| 版本 | `CKPT_VERSION = 10` | 旧 `save_dir` 的 `--auto_resume` 被拒，新 `save_dir` 从头训 |

第 8 节的验证对应 `tests/test_cross_limb_temporal.py`（8.1–8.3 的 block 级）与
`tests/test_cross_limb_reliability_fix.py`（8.1 的模型级、8.2 的 embedding、8.4、AdamW 分组）。
8.5 的训练/推理质量消融未在代码内完成；开销实测（RTX PRO 6000，eager bf16，B=16 J=40 F=60，
train.bat 模型形状）：参数 14.659M → 14.925M（+0.266M），fwd+bwd 90.1 → 94.5 ms/step（+4.9%），
峰值显存 9.43 → 9.55 GiB（+1.3%）。新 block 在 Dynamo 下单图、无 graph break。

一个测试期发现值得记下：cross-K 是 Pre-Norm，对某个 latent 的**全通道等量**扰动会被
LayerNorm 精确抹掉，其它 latent 看不到；验证 latent 通信必须用通道不均匀的扰动。

