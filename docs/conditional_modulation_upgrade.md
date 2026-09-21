# AnyTop 条件注入改造：loop 相位周期修正 + action 条件强度

> 状态：§2 loop 修正**已实施（CKPT 12），待重训**，见 2.6；§3 action AdaLN
> **已实施（2026-09-20，`--action_label_adaln`）**，见 3.5 —— 但 §3.3 的消融**没有跑**，
> 所以"改注入方式是否有效"至今**未被验证**。
> 范围：`is_loop`、`action_label`。其余条件经实测现状机制无需改造，已从本方案移除（证据见附录）。
> 证据来源：2026-09-14 在 `save/merged_locomotion_v14_fp16/model000200000.pt`（用该 run 自带的
> `cond.npy` 快照）上做的 teacher-forced 探针与生成探针。只测了 locomotion 组。

---

## 1. 结论

| 条件 | 实测问题 | 改造 | 需要 |
|---|---|---|---|
| `is_loop` | 生成的 loop 首尾几乎重合（接缝处停一帧），训练数据从不如此 | 环形相位表周期 `motion_frames-1 → motion_frames`；`loop_wrap_loss` 的 pose/rot 项改为接缝连续性 | CKPT bump + 重训 |
| `action_label` | 未见过的 物种×方向 组合在 cfg=1 下不跟随标签；靠 cfg=3 才能跟随，但骨长误差、jerk 都变差约 20% | action 条件加一路逐层 AdaLN（temporal / FFN 分支输入），identity 初始化 | flag（已落地）+ 三组消融（未跑） |

顺序：**先做 loop 修正并重训，得到新 baseline，再在它上面做 action 消融**。loop 修正改变的是所有
loop 样本的训练语义，放在同一次消融里会和 action 调制的效果混在一起。

## 2. `is_loop`：环形相位周期与数据约定差一帧

### 2.1 现象

闭合比 = `‖x[末帧] − x[首帧]‖ / 窗口内相邻帧步长中位数`（非根关节的 pos + rot 通道）。

| 窗口 | 低 speed | 高 speed |
|---|---:|---:|
| 真实加载器输出的 loop 窗口，按 speed < 1.25 / ≥ 1.25 分组（p10 / 中位数，n=258 / 186） | 0.70 / 1.32 | 0.44 / 0.64 |
| 按加载器逻辑复现、跨度对齐的 GT 窗口，s = 1 / 2（中位数） | 0.76 | 0.44 |
| 生成，`is_loop=True`，s = 1 / 2（中位数） | **0.14** | **0.11** |

生成结果比训练数据的 p10 还小 4–5 倍：末帧几乎是首帧的复制，平铺播放时接缝处接近静止一帧。

### 2.2 原因

两处仍在沿用"保留 closing key（末帧 = 首帧）"的旧约定，而加载器从 `471743e` 起已经在加载时
丢掉了 closing key（[`dataset._drop_loop_closing_frame`](../data_loaders/truebones/data/dataset.py)），
所以窗口末帧是首帧的**前一步**：

1. [`circular_phase_embedding`](../model/motion_transformer.py) 用 `period = motion_frames - 1`，
   首、末两帧的相位编码完全相同。主干每层 temporal attention 前加的相位
   （`GraphMotionDecoderLayer.temporal_phase_scale`）用这张表（cross-limb 曾经的 loop 时间表
   `_loop_aware_time_embedding` 已于 2026-09-21 删除，见 anytop_model_architecture.md §5.3）。
   函数 docstring 里 "the closing key every stored loop keeps"已经过时。
2. [`loop_wrap_loss`](../diffusion/gaussian_diffusion.py) 的 pose 项和 rot 项把首帧和末帧往**相等**拉；
   同一个函数里的 terminal velocity 项要求 `首帧 − 末帧 = 末帧速度 × step`，符合新约定。两类项互相矛盾。

### 2.3 归因实验（推理时替换，不重训）

在同一 checkpoint、相同物种和 seed 下，只替换相位表（8 个物种 × 3 个样本，`walk, forward`）：

| 变体 | 闭合比 s=1 / s=2 | 周期 / GT | 骨长误差 | jerk（含接缝） s=1 / s=2 |
|---|---:|---:|---:|---:|
| 现状（周期 59） | 0.14 / 0.11 | 1.01 / 1.02 | 0.046 / 0.049 | 0.0198 / 0.0621 |
| 周期 60 | **0.84 / 0.42** | 1.04 / 1.08 | 0.047 / 0.047 | 0.0192 / 0.0548 |
| 主干相位 scale 置 0 | 1.25 / 0.64 | — | — | — |

只改周期，闭合比就回到数据分布内，步频和骨长不变，含接缝的 jerk 下降。说明过度闭合主要来自
相位表。loss 的 pose/rot 项贡献多少，推理实验无法分离；它和数据矛盾，一并修正。

该 checkpoint 的 `temporal_phase_scale` 只有第 0 层明显非零（0.28，其余层绝对值 ≤ 0.023）。

### 2.4 改造

1. `circular_phase_embedding`：相位改为 `2π·f·t / motion_frames`，即"末帧的下一帧"和首帧同相。
   cross-limb 时间表调用同一函数，自动一致。同步更新 docstring。
2. `loop_wrap_loss`：
   - 删除 pose 项。terminal velocity 项已经按正确约定约束了位置接缝。
   - rot 项从 `geodesic(末帧, 首帧) → 0` 改为接缝处与相邻帧步长连续，例如
     `geodesic(末帧, 首帧) ≈ geodesic(倒数第二帧, 末帧)`。
3. （建议同批做）加载器 loop 窗口的重采样用的是 `resample_motion_features` 的 `linspace(0, L-1, T)`
   端点映射：接缝步长是 1 个源帧，窗口内步长是 `(L-1)/(T-1)` 个源帧，二者只在 `L = T` 时相等。
   实测训练窗口闭合比的 p10–p90 横跨 0.44–2.58，说明周期性本身就不均匀。loop 窗口改为按
   `t·L/T`（不含端点）环形插值后，周期严格等于 `T`。
   注意：这会改变 loss 端从 `resample_speed_cond` 反推的 step scale（现为 `(L-1)/(T-1)`），
   `_physical_velocity_step_scale` 要对 loop 样本同步改成 `L/T`。
4. `CKPT_VERSION` 11 → 12：相位表语义变化，旧 loop checkpoint 不能在新代码上运行。之后重训。
5. （可选，现有 checkpoint 的止血方案）推理时切换到周期 `T` 的相位表。2.3 的实测没有退化，
   但对训练时的相位通路而言属于分布外，只能临时用，不作为最终方案。

### 2.5 验收

- 生成 loop 的闭合比落在训练窗口分布内。做了第 3 项后应接近 1，并且不随 `resample_speed` 变化。
- 周期 / GT、骨长误差与修正前持平；含接缝的 jerk 不高于修正前。
- 非 loop 生成不受影响（同样的指标，`is_loop=False`）。

### 2.6 实施记录（2026-09-14）

- 第 1–4 项已实施，`CKPT_VERSION` 11 → 12。第 5 项没做：版本号升级后，旧 checkpoint 会直接被版本检查拒绝。
- 第 2 项的 rot 项：取接缝步长与**两侧**相邻步长均值之差的绝对值，
  `|geo(R[-1], R[0]) − ½·(geo(R[-2], R[-1]) + geo(R[0], R[1]))|`。量纲仍是弧度，`--lambda_loop_wrap 0.04`
  不用改。训练日志里不再有 `loop_wrap_pose`。
- 第 3 项实际改了三处，缺一处周期都不均匀：
  - 窗口重采样 `resample_motion_features(periodic=True)`，只用于 `loop_condition_active` 的窗口（loop 片段被裁剪
    降级为非 loop 时仍按端点重采样）；
  - loop 片段的速度增广 `time_scale_motion_features(periodic=True)`。否则平铺后窗口内部仍有不均匀的接缝；
  - `_physical_velocity_step_scale` 按 `y['is_loop']` 取 `L/T`，非 loop 仍是 `(L-1)/(T-1)`。
- 生成端同步。显式 `--loop on` / `--loop off` 是“这个窗口是一个环”的声明，和 reference 本身是不是 loop
  无关；默认的 `--loop auto` 相反，有 `--reference_motion` 时按 reference 自己的闭合判决解析
  （`loop_verdict` 的存储判决行，没有则用端点检测器），只有 `--action_label` 时按该 label 训练片段里
  loop 的多数，两者都没有时取 `off`。窗口 ↔ 输出之间只有
  一个映射（`_resample_window_to_output`，`periodic` 取解析后的 loop 条件），三处都走它：
  - 纯 loop 生成（没有 reference）导出到 M ≠ T 帧时按环形重采样，否则导出结果的接缝步长又会不均匀；
  - 带 reference 时同样按环形重采样，并且 reference 本身也按环形放进窗口（`_prepare_img2img_reference_bundle`）：
    先丢 closing key（如果它带），再按 `t·L/T` 重采样。这样窗口第 t 帧在两个方向上都对应 reference 源时刻
    `t·L/T`，往返是恒等映射，`--inpaint_frames` 的范围落在它指名的 reference 姿态上，模型读到的
    `resample_speed_cond = L/T` 也正是 reference 实际的步长。反过来（reference 端点重采样 + 导出端点重采样）
    虽然自洽，但接缝步长是 1 个源帧、窗口内是 `(L-1)/(T-1)`，对 clamp 来说这个不均匀接缝会直接进入输出；
  - `_reground_inpaint_joint_y` 用的 reference 必须和导出用同一个映射（它逐帧配对两者），否则竖直 reseat 的
    均值取在错位的帧上。`_map_frame_ranges_to_internal` 同理，`--loop` 时用 `T/M` 而不是 `(T-1)/(M-1)`。
  这样导出结果**结构上**一定是 loop：接缝是一步普通步长，不再有停一帧。内容闭合得多好仍取决于模型，而
  `--skip_timesteps` 越高输出越贴着 reference——`--loop` 不强制内容闭合，它只保证不再由导出环节把环破坏掉。
- `tools/sample_augmented_bvh.py --real-time` 没有 reference，按 `loop_applied` 选重采样方式。
- 加载器实测。样本是 biped / multiped / quadruped 三个子集的全部 loop clip，每个抽 6 次；表中是闭合比中位数，三个数依次对应这三个子集：

  | | speed < 1.25 | speed ≥ 1.25 |
  |---|---:|---:|
  | 旧（端点重采样） | 1.06 / 1.25 / 1.06 | 0.66 / 0.64 / 0.64 |
  | 新（环形重采样） | 1.06 / 1.08 / 1.01 | 0.98 / 1.01 / 1.01 |

  高 speed 组从 0.64 回到 ≈1，不再随 speed 变化，训练数据这一侧符合 2.5 第一条。生成结果的验收要等重训后再测。

## 3. `action_label`：未见组合上的条件强度

### 3.1 实测

方向从肢体运动学判断（locomotion 的根 XZ 位移在预处理时被拉回原地，不能看根速度）：
`D_fb = corr(足端高度, 足端 z 速度)`，`D_lr = corr(足端高度, 足端 x 速度)`，足端取静止姿态中高度最低三分之一的叶关节。
分类器用 GT 各方向标签的中心点，按最近中心归类。

| 场景 | cfg=1 | cfg=3 |
|---|---:|---:|
| 训练中有 walk 前/后/左/右 的 12 个物种，方向准确率（GT 自身用此指标上限 63%） | 59% | 65% |
| 从没有后退/横移 clip 的 7 个四足骨架：forward / backward / left / right | 100 / 33 / 0 / 0 % | 100 / **100** / 0 / 29 % |
| 8 个常规物种：walk vs run 步频比 ÷ GT 步频比（1 = 与 GT 同样分开） | 1.08 | 1.09 |
| 第一行 12 个物种中能测出周期的 9 个（多为 unitybundles 怪物骨架），run/walk 周期比（GT 0.67） | **0.98** | 0.67 |
| 骨长误差（常规物种，中位数） | 0.054 | 0.065 |
| jerk / GT | 1.16 | 1.42 |

结论：
- 训练中见过的组合，cfg=1 基本够用（方向准确率接近指标上限，常规物种 walk/run 分得开）。
- 模型**有**这些条件的知识：cfg=3 能让四足后退从 33% 升到 100%，也能让怪物物种分开 walk/run。
  但 cfg=1 时条件被物种先验盖过。靠 cfg=3 补偿，要付出骨长误差 +20%、jerk +22% 的代价。
- 四足横移在 cfg=3 下依然失败。这类步态在数据里基本不存在，不属于本方案的目标，任何注入方式都
  无法指望解决。

"cfg=1 条件太弱"是否由加性注入方式造成，无法从推理实验判断，也可能来自 CFG drop 比例或数据分布。
所以本节是一个**需要消融验证**的改造，不是确认的缺陷。

### 3.2 改造

在现有加性 action token 之外，加一路逐层 AdaLN：

```text
action_repr = where(active, action_label_projection(channels), action_label_null_emb)   # 每次 forward 只算一次
x_t_in  = (1 + γ_l^temporal(action_repr)) * x + β_l^temporal(action_repr)   # 只喂 temporal 分支输入，残差仍用 x
x_ff_in = (1 + γ_l^ffn(action_repr))      * x + β_l^ffn(action_repr)        # 只喂 FFN 分支输入
```

- **位置**：当前 decoder 是 Post-Norm（`GraphMotionDecoderLayer`），temporal 和 FFN 分支的输入正好是
  `norm1` / `norm2` 的输出，所以这就是只作用于分支输入的 AdaLN，不改 Norm 结构。spatial 分支的输入
  已经带了 `embed_timesteps` 的加性偏移，不再重复。temporal 分支的 γ/β 在环形相位相加
  （`GraphMotionDecoderLayer.temporal_phase_scale`）**之前**施加，避免缩放相位。
- **初始化**：每个 head 的最后一层 Linear 零初始化，新模型一开始与现状完全一致。
  不引入 residual gate。
- **同一次 Bernoulli**：训练时 `AnyTop._resolve_action_label_active` 会抽 `torch.rand`。加性路径和
  调制路径必须共用同一个 `action_repr`，不能各自调用一次 `_action_condition`，否则两条路径看到的
  drop 掩码不同。
- **CFG**：[`ClassifierFreeActionModel`](../model/cfg_sampler.py) 的无条件分支只把
  `action_label_active` 置 False。只要调制读的是同一个 `action_repr`，它就自动走 null，其余条件
  在两次前向中保持一致。`feat/all_group` 上的 group token 仍只走加性路径（label CFG 的无条件分支会保留它）。
- **精度**：head 用 `run_in_fp32`，与其他条件投影一致。训练使用 `--amp_dtype fp16`。
- **参数量**：每层两个 `Linear(256 → 512)`，共 8 层，约 2.1M，是现有 17.0M 的 +12%。
- **开关**：`--action_label_adaln`，默认关。关闭时不创建任何参数，旧 checkpoint 按原路径加载，
  不需要 CKPT bump。不要用 `strict=False` 吞掉新 key。

（2026-09-20 已实施，见 §3.5。`feat/all_group` 的 group token 那句已作废：`train/v22` 不再有 group token。）

### 3.3 消融

以 §2 修正后重训的模型作为 baseline，同一数据、同样的 step 数：

| Arm | 说明 |
|---|---|
| A0 | baseline，跑 2 个 seed，用来估计 seed 噪声 |
| A1 | + `--action_label_adaln` |
| A2 | 与 A1 相同的 head，但强制 γ≡0（只有 β）。用来分离"乘性调制"和"多一处注入点"两种效果 |

### 3.4 采纳 / 回退

采纳 A1 需同时满足：

1. 四足后退迁移准确率在 cfg=1 下达到 A0 cfg=3 的水平，超出 A0 两个 seed 的差异范围；
2. 怪物物种 walk/run 步频比在 cfg=1 下接近 GT；
3. 训练中见过的组合的方向准确率不低于 A0；
4. cfg=1 下骨长误差和 jerk 不高于 A0 cfg=1。本方案的意义就是免掉 cfg=3 的质量代价；
5. 提升明显大于 A2；
6. 推理时把 head 置零后，上述提升消失。

回退条件：A1 与 A0 或 A2 无法区分。此时 action 条件强度问题的现实解法就是提高 cfg，代价见 §3.1 表格。

### 3.5 实施记录（2026-09-20，分支 `train/v22`）

按 §3.2 落地，开关名 `--action_label_adaln`，默认关。与设计稿的出入只有一处：**没有实现 A2 臂**
（γ≡0 的 β-only 变体），因为本轮不跑 baseline 对照 —— 用户决定直接带着这个改动重训
（见 [unified_action_group_training.md](unified_action_group_training.md)）。§3.3 / §3.4 因此**未执行**，
留作日后要做归因时的方案。

代码位置：

- head 在 [`GraphMotionDecoder`](../model/motion_transformer.py)（`self.action_adaln`），
  不在 layer 上：**一个** head 一次 matmul 出全部 8 层的 (γ,β)，切片喂给每层。
  参数量 `d² + 4·L·d²` 与"每层一个 head"完全相同，但 kernel launch 从 2L 次降到 1 次 ——
  这一步部分是 launch 绑定的。实测 256: 2.17M / 384: 4.88M。
- 施加点在 [`GraphMotionDecoderLayer.forward`](../model/motion_transformer.py) 的 `_modulate`：
  只改 temporal 与 FFN 的**分支输入**，残差流保持未调制的 post-norm 值，所以逐层增益不会相乘。
  spatial 分支不碰（它的输入已带 `embed_timesteps` 的加性偏移）。
- temporal 的 γ/β 在 `_temporal_mha_block_sin_joint` **之外**施加，而环形相位是在该函数内部加的，
  所以 loop 样本的相位不会被 label 缩放 —— §3.2 那条要求由调用顺序自动满足。
- 驱动向量就是加性路径用的那个 `action_label_token`（`_build_action_label_token` 的返回值），
  一次 Bernoulli、一个 `action_repr`。CFG 的无条件分支只把 `action_label_active` 置 False，
  调制随之走 null 嵌入，**不存在绕过 head 的路径**（被退役的 global_energy null 就是栽在这里）。
- 精度按 §3.2 用 `run_in_fp32`；该 helper 已从 `model/anytop.py` 移到
  `model/motion_transformer.py`（decoder 自己要用，而 anytop 导入 motion_transformer），
  `model.anytop.run_in_fp32` 仍然可用，是同一个函数。
- 零初始化的回归测试在 [`tests/test_action_label_adaln.py`](../tests/test_action_label_adaln.py)：
  拷贝权重后两个模型逐位一致（head 会消耗 RNG，光靠同 seed 不够）。
- **CKPT_VERSION 不 bump**：默认关，关闭时不创建参数，state_dict 不变。

## 4. 探针方法（复现用）

- **周期**：非根关节 pos + rot 通道去均值后，把各通道的自相关相加，取最高峰 85% 以上的第一个峰，
  再做抛物线插值；乘以 `(round(s·60)−1)/59` 换算成源帧周期。在 GT 窗口上，源帧周期对 s 不变。
- **GT 窗口**：按加载器逻辑复现：drop closing key → tile → 截取 `round(s·60)` 帧 → `resample_motion_features`。
- **生成**：DDPM 100 步，fp32。speed / loop 实验每个物种 2–4 个样本。
  Elephant、Raptor 在 GT 上也测不出周期，已排除。
- **CKPT**：v14_fp16 是 CKPT 10，加载时绕过了版本检查。它与 11 的差别只有 9 个 2–8 关节小骨架的 L 下限，
  探针没有用到这些骨架，并且使用该 run 自带的 cond 快照。

---

## 附录：已排除的条件（现状机制无需改造）

| 条件 | 证据 |
|---|---|
| canonical frame（`canonical_feature_mean/std`） | 实际只有 7 组取值，position 的 std 所有 subset 共享（0.821），本质上是 7 类类别变量。teacher-forced 把 frame 换成别的 subset，x0 误差：pos ×2.2–4.2，rot ×1.7–6.4；用真实 frame 时每个 frame 的直流偏差均值 0.003–0.010（不计只有 3 个样本的 frame），换 frame 后 0.019–0.050。加性 token 已经把输出空间分清 |
| `species_cond` FiLM | 关掉后 x0 误差 ×1.00–1.07。信息与关节名通路重复，加调制也不会让重复的信息变有用 |
| `resample_speed_cond` | 8 个物种、s = 0.75–2.0，生成周期（源帧）/ GT 中位数 0.85–1.09。s=1.5 / 2 各 48 个样本，77% / 88% 在 ±30% 以内（训练本身就有 ±30% 节奏增广），保持 s=1 步频不变的只有 12% / 0%。训练中 s∈[1.25, 2] 占 45%。不存在忽略 speed 的系统性问题 |
