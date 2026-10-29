# Loop 周期数条件与循环时间编码

> 本文是方案，已实现（CKPT_VERSION 56）。目标是修复 phase-free loop（run / fly / swim / idle / hover 等）生成中的
> **周期混叠**（一个周期没结束，新周期已经开始，表现为某个肢体按双倍频率运动）和
> **速度时快时慢**。
> 相关代码：`data_loaders/truebones/data/dataset.py`（loop roll / tile / 窗口重采样）、
> `model/anytop.py`（绝对帧 PE、tempo PE）、`model/motion_transformer.py`（`circular_phase_embedding`）、
> `sample/output_lengths.py`、`utils/clip_length_prior.py`（自动长度）。

---

## 1. 诊断结论（merged_all_v62）

测试用例 Horse_RunLoop、Buffalo_RunLoop、Eagle_FlyLoop，每组 8 个 seed，`--loop on`，
长度分别取 GT 长度、1.25×GT、2×GT。

- **混叠确实存在，后期改善主要来自记忆。** 判据是：在 1 周期窗口里，某个运动关节的二次谐波能量超过基频。
  Horse 在 GT 长度（N=31）下，40k–60k 时 8 个样本中有 3–4 个混叠，240k 降到 0 个；
  但同一个 240k 在 N=39 和 N=62 时仍有 3 个。这两个长度经速度增广后同样在训练分布内。
  模型记住的是「这个物种、这个长度 → 1 个周期」，而没有学会这个物种的周期。
- **速度忽快忽慢分两部分。** 用样本自身 PCA 平面里的相位角计算相位速度 CV：
  - Horse 从 20k 的 0.71 降到 240k 的 0.35，GT 本身是 0.15。
  - 带双倍频肢体的样本，CV 约 2，与混叠同源。
  - 不带双倍频的样本，CV 仍是 GT 的 2–3 倍，这是残余部分。
- **接缝影响很弱。** 相位速度偏差在首尾接缝附近只略高（Horse），Buffalo 没有趋势。
- **v62 几乎没用上循环编码。** loop 样本同时收到三种时间信号：
  - 非循环的绝对帧 PE；
  - 非循环的 tempo PE（线性斜坡 `i·L/T`），投影很强；
  - 循环相位表。它每层的 `temporal_phase_scale` 只在前两层明显非零，其余接近 0。
    绝对 PE 已经给了位置，循环表对模型是冗余信号，学不起来是预期之内的。

**根因**：窗口里的总周期数从未告诉过模型。loader 把一条 loop clip 拼接 k 份，再重采样到窗口 T，
模型只看到时长 `L = resample_speed_cond · T`。同一个 label 下，各物种的周期相差约 2 倍
（`run, forward` 的 clip 长度从 19 帧左右到 37 帧左右）。因此「run, forward，窗口 31 帧」在训练集里既可能是
长周期物种的 1 份 tile，也可能是短周期物种的 2 份 tile。模型只能靠记住物种来区分，记不住时就把两者混在一起。

**数据事实**：每条 loop clip 只在整条长度上闭合。即使翅膀振动多次（如 Insect_Bee_Fly01、Giantbee_Fly），
根节点和部分关节（尾巴）的周期也是整条 clip，换任何首尾帧裁剪都无法闭合。
因此 clip 自身的周期数恒为 1，**窗口的周期数 k 就等于 tile 数**，不需要预处理测量。
翅膀在一个周期内振动几次是物种属性，不随变速和 tiling 改变，由模型从物种学习，不进入 k。

---

## 2. 条件定义

`k = loop_tile_count`，只对 loop 条件生效的窗口（loader 的 `loop_condition_active`）定义：

| 窗口类型 | k |
|---|---|
| phase-free loop | loader 抽到的 tile 数 |
| 相位锁定 loop（`loop_is_phase_free` 为假） | 固定为 1（这类 loop 不做 roll / tile） |
| `loop_uncond`（loop clip 按开放 clip 训练，content 可能已 tile）、超长被裁剪的 loop、非 loop | 不定义，走开放窗口路径 |

k 不做训练时 drop。参考动作激活的行（`ref_active`）在训练和推理里都把 k 读作 1（§3.3）。

**k 的训练分布**：`_sample_loop_tile_count` 保证 `k · P_s ≤ max_source_length`（`P_s` 为变速后的 clip 长度；
预算对普通 clip 是 `MAX_SOURCE_FRAMES_MULT · T`，对高频 clip 是 `T`），k=1 的概率至少为 `loop_tile_single_prob`。
因此每个 (物种, label) 下训练见过的 k 是 `1..⌊预算 / P_s⌋`，推理时的 k 必须在这个范围内（§4）。

---

## 3. 模型改动（只改 loop 样本，开放窗口不变）

### 3.1 位置编码：循环替换绝对，周期为 T/k

loop 样本**不再加绝对帧 PE**，改为在输入层加循环相位表，作为它唯一的帧位置信号：

- 周期为 `T/k`：相位 `2π·f·k·i/T`，沿用 `circular_phase_embedding` 现有的整数频率 f；k=1 就是周期为 T 的表。
- 每层叠加的循环相位（`temporal_phase_scale`）使用同一张表。
  cross-limb 的 latent 通过 cross-in 继承输入层的帧编码，不用另外改。
- 窗口最后一帧仍是第 0 帧的前一步（`_drop_loop_closing_frame` 的约定不变），所以周期取 `T/k`，不是 `(T−1)/k`。

依据：

- **周期 T/k 与数据严格一致。** loop 窗口按周期重采样，窗口第 i 帧取源帧 `i·L/T`，每份 tile 占 `P_s` 个源帧，
  所以在窗口里恰好占 `T/k` 帧。k 整除 T 时，各份 tile 逐位相同，它们的编码也逐位相同，
  「各份拷贝相同」就从需要学习的规律变成了结构上的约束。
- **去掉绝对 PE 才能逼模型使用循环编码。** 只要绝对 PE 和斜坡还在，模型就能绕开循环表（§1 的 `temporal_phase_scale`）。
  绝对 PE 在接缝处也是不连续的（第 T−1 帧和第 0 帧编码差很远），这和 loop 窗口的拓扑相矛盾。
- **现有的表就够用，不需要另做频谱设计。** 这张表原始的点积近似 delta（相邻帧点积约 0.05），
  但 attention 看到的是 `PEᵢᵀ W_qᵀ W_k PEⱼ`。这张表在 T 帧上的秩是 T，是循环群 Z_T 上完整的 Fourier 基，
  所以 W 对各频率加权，就能表示任意循环平移不变的核，包括平滑、局部的核。它的作用等价于可学习的位置嵌入。
  与之相比，绝对 PE（D=384、T=60）秩只有 38，大部分通道在窗口内近似常数。
- **不同 k 共用同一组编码。** 相位 `k·i/T mod 1` 总落在 1/T 的格点上：gcd(k, T)=1 时，
  第 i 帧的编码就是 k=1 那张表的行换了顺序；否则是其中 T/gcd 行各重复 gcd 次。
  因此对各频率学到的权重对所有 k 通用，不存在小数相位。

实现：周期因样本而异，表按 `[T, B, D]` 在设备上用 `k` 张量直接算，不按 k 分支（compile 下没有 graph break，也没有同步）。
开放行仍用现有只在 batch 维建一行的绝对 PE，两者用 `is_loop` 选择。

### 3.2 tempo 编码：loop 样本改为每帧相同的周期长度

loop 样本不再用 tempo 斜坡 `i·L/T`，改为每帧相同的嵌入，内容是**物理周期长度** `L/k`
（一个周期对应多少源帧）。它通过**独立的** `loop_period_projection` 进入模型，
结构与 `tempo_projection` 相同（`create_sin_embedding` 加两层 MLP），最后一层零初始化。

依据：

- **没有信息损失。** 相对时间已经由 §3.1 的循环表给出，T 固定，k 由表的周期给出，所以 `L = k · (L/k)` 可以还原。
  `L/k` 不是新的速度条件，只是把已经给出的时长按周期数做了归一化。tempo drop 的行例外，见本节末尾。
- **`L/k` 直接就是模型要学的量。** 同一物种的 loop 在训练里的 `L/k` 就是变速后的周期 `P_s`，
  落在 `P̂/1.3 … 1.3·P̂` 内（`motion_speed_aug=1.3`），与 tile 数无关。因此「物种 + `L/k`」能直接对应到步速。
  斜坡把同一信息拆成「每帧递增多少」，并且和 tile 数混在一起。
- **斜坡和循环拓扑矛盾。** 斜坡在接缝处从 `(T−1)·L/T` 跳回 0，换成每帧相同的值后，这个跳变就消失了。
- **为什么要独立投影。** 开放行的 tempo 输入是第 i 帧的源时间，速度藏在相邻帧之间的变化率里；
  loop 行的输入是一个每帧相同的周期长度，速度体现在它的绝对值里。共用一个逐帧 MLP 时，同一个输入值 30
  在两种行里含义不同，只能靠下游从 PE 类型去分辨。独立投影只多一组 2·D² 参数，就消除了这个歧义。

tempo drop 不变：drop 的行把输出乘以 `tempo_active`，`loop_period_projection` 的输出也乘以同一个 `tempo_active`。
tempo drop 只在训练时触发，作为正则；推理始终给出 L，没有 tempo CFG。drop 的行真的丢了 L：
§3.1 的循环表不受 tempo drop 影响，模型仍知道窗口里有 k 个周期、每个占 T/k 帧，
只是不知道一个周期对应多少源帧，步速只能从物种和 label 的先验推断。

### 3.3 k 只作为周期进入模型

k 没有自己的 token：它只是 §3.1 循环表的周期 `T/k` 和 §3.2 loop tempo 的 `L/k`，模型只能把它读成时间。

依据：merged_all_v65 的推理消融显示 k 的作用全在周期上——单独去掉 k 的 token，倍频样本数、相位速度 CV
和 `--loop on/off` 的抬腿差别都不变；只有把表和 tempo 退回 k=1 才退回 v62 水平。

训练：k 来自 batch 里已有的 `y['loop_tile_count']`（也用于诊断和 temporal-span 掩码），只在 loop 条件生效的行使用。

参考动作激活的行把 k 读作 1，训练和推理相同，与 action label、species FiLM 在这些行被绕过一致：
参考动作给出动作内容，周期也由它自己决定。训练里参考与目标共用一条时间轴、一起 tile，
z 逐帧对齐，k 份拷贝的结构由 z 给出。这些行的表和 tempo 按一个周期算，即使内容已 tile k 份；
如果训练时 ref 行能看到 k，模型会靠 k 抄近路而不读 z。

### 3.4 不改的部分

- 开放窗口（包括 `loop_uncond`）：绝对帧 PE 和 tempo 斜坡保持原样。
- 参考动作编码器（`motion_encoder`）的时间编码：本方案不涉及。
- 损失函数：不新增；wrap 相关损失照旧。
- temporal-span 掩码：已经按 `loop_tile_count` 的周期重复，不变。

改了输入编码，需要 bump `CKPT_VERSION` 并重新训练。cond 不需要重新生成。

---

## 4. 推理端

记 `P̂` 为目标物种在该 action_label 下 loop clip 长度的中位数，和现有自动长度用的是同一个先验
（`auto_num_frames`；loop clip 已经去掉闭合帧，没有该 label 的 clip 时取最相似物种的）。
由于每条 clip 是一个周期，`P̂` 就是周期的估计。推理窗口仍是 checkpoint 的 T，N 只决定 `L = N`。

| 用户给出 | k | N |
|---|---|---|
| 都不给（`num_frames` 为 auto） | **1** | `P̂`（与现有自动长度一致） |
| 只给 `--loop_cycles k` | k | `k · P̂` |
| 只给 `--num_frames N` | 普通 label：在 log 空间里取离 `N / P̂` 最近的整数；高频 label：`⌊N / P̂⌋`；至少为 1 | N |
| 两者都给 | k | N |

没有周期先验时，未给出的 k 取 1。

高频 label 向下取整的依据：高频 clip 是 `slow_only`，训练里 `P_s ≥ P`，也就是 `L/k ≥ P̂`。
取最近整数可能得到 `N/k < P̂`，这在训练里从未出现。

训练分布检查（`k` 和 `L/k` 都要在训练见过的范围内）：

- `L/k` 超出该 clip 的速度范围时警告。普通 clip 为 `[P̂ / R, R · P̂]`，高频 clip 为 `[P̂, R · P̂]`，
  其中 R 取 checkpoint 的 `motion_speed_aug`。例如 `R=1.3` 时，`N/P̂ ∈ (1.3, 1.54)` 这一段，
  k=1 和 k=2 都覆盖不到。
- `L` 超过源帧预算时警告：普通 clip 为 `MAX_SOURCE_FRAMES_MULT · T`，高频 clip 为 `T`。超出预算的窗口在训练里会被裁成开放片段，
  从未按 loop 训练过。例如 Horse 给 `--loop_cycles 4`，L≈124，速度正常，但超过了 120。
- 相位锁定的 label 只允许 k=1，给出更大的 k 直接报错（训练时这类 loop 从未拼接过）。

参考动作路径（encode / 编辑 / outpaint）：N 默认取参考长度 R，k 一律为 1，给出 `--loop_cycles` 直接报错。

- 推理时假定参考动作都不在训练集里，它有几个周期并不知道，窗口里的周期由参考决定：默认的拟合把整条参考
  重采样到 N 帧，只改速度、不复制周期；`--outpaint` 追加的帧从噪声生成，周期数也不确定。
- 参考动作激活的行在模型里本来就把 k 读作 1（§3.3）；没有 ref encoder 的 img2img / inpaint 同样给 1，规则统一。

非 loop 生成不涉及 k。

---

## 5. 预期效果与不解决的部分

| 问题 | 预期 |
|---|---|
| k=1 窗口的周期混叠（实测的主因） | 由 §3.1 的 T/k 表和 §3.2 的 `L/k` 区分「拼 2 份」的那批训练窗口（v65 非 GT 长度倍频 16/24 → 4/24） |
| k≥2 时各周期之间不一致 | 由 §3.1 的 T/k 周期从结构上约束 |
| 混叠带来的局部双倍速 | 随混叠一起消失 |
| k=1 窗口里残余的忽快忽慢 | **不解决**。相位原点（roll 偏移）的对称性仍在，各段可能各自定相位。要解决需要规范化的相位原点（例如对齐到落地事件），另行评估 |
| 高频振翅 loop 长于 T | **不解决**。这类 clip 的源帧预算等于窗口，超长时会被裁成开放片段，从未按 loop 训练（例如 Giantbee_Fly）。可选的独立修复：在适度范围内改用周期性 FFT 重采样代替线性插值，让它们保持 loop |

与 `86e6823` 之前「相位表周期 = (T−1)/k、推理时按物种周期表反推 k」的设计相比，这次有三点不同：
- tempo 以 `L/k` 给出周期长度，和表的周期一致；
- 非循环的绝对 PE 和 tempo 斜坡不再给 loop 样本，模型没有绕开循环编码的路；
- 周期取 T/k，消除了接缝处卡一帧的问题。

---

## 6. 验证

新 checkpoint 与 merged_all_v65 比较，指标：

- **周期混叠**：每个用例 8 个 seed，统计「存在二次谐波能量超过基频的运动关节」的样本比例。
  长度取 GT、1.25×GT、2×GT；关键看非 GT 长度是否接近 GT 长度的水平。
- **速度均匀性**：PCA 相位速度 CV，与 GT 以及 GT 加 ±30% / ±60% 正弦变速的基线比较。
- **幅度**：生成样本的姿态 RMS 与 GT 之比。
- **局部平滑**：loop 样本的帧间加速度，与 GT 和 A 组比较。C 组的 loop 行只剩循环表这一种位置信号，
  这一项用来检查局部时间结构有没有变差。
- **不退化**：相位锁定 loop（attack / idle 带事件）和开放动作的现有 eval 指标；loop 闭合误差。
- **k 的控制力**：同一物种用 `--loop_cycles 1/2/3`，检查生成的周期数是否等于 k，并检查步态是否仍是该物种的
  （与 k=1 的步态比较单周期姿态序列的距离），排除 k 被当成内容键的情况。
- **loop 内容键**：同一 label、同一 N 下 `--loop on` 与 `off` 的抬腿、能量之比（v65：Horse + Buffalo 低 10–15%）。
- 以 eval 指标判断，不以训练 loss 判断。

## 7. 实现清单

1. loader：只在 `loop_condition_active` 的行把 `loop_tile_count` 作为 k 输入模型；相位锁定 loop 的 k 为 1；
   开放窗口不给 k。
2. 模型：
   - loop 行的输入层用循环表（周期 `T/k`）替换绝对帧 PE，按 `[T, B, D]` 计算；每层循环相位用同一张表；
   - loop 行的 tempo 改为 `L/k` 经 `loop_period_projection` 的每帧相同嵌入，乘以 `tempo_active`；
   - 参考动作激活的行把 k 读作 1。
3. 推理：`--loop_cycles`（正整数）、§4 的长度与 k 解析规则（普通 label 用 log 最近整数，高频 label 向下取整）、
   速度范围和源帧预算警告、相位锁定 label 的检查、参考路径的 k 一律为 1。
4. bump `CKPT_VERSION`；更新 `docs/anytop_model_architecture.md` 中 loop 相关章节，
   以及 `circular_phase_embedding` 的 docstring（它现在写的是「不替换绝对 PE、不编码周期数」）。
5. 测试：
   - k 整除 T 时，k 份 tile 的窗口在 T/k 循环表下各份编码逐位一致；k 与 T 互素时，编码是 k=1 表的行置换；
   - 参考动作激活的行 k 读作 1，开放行不读 k；
   - tempo drop 的行，`loop_period_projection` 的输出为零；
   - §4 的推理解析表逐行覆盖，包括高频 label 向下取整和两种警告。
