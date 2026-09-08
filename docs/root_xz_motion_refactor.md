# root XZ 重构：删掉模糊带 + 标志位 + 生成期一视同仁

> 状态：**已实施（2026-09-08）**。C1 / C2 / C3 / C4 / c6 全部落地，c5 按 §2.5 不做。
> 仍待执行：**全量重新预处理 + 重训**（C1/C2/c6）。C3/C4 已对现有 checkpoint 生效。
> `CKPT_VERSION` 4 → 5，旧 checkpoint 会被明确拒绝。
>
> 实施中发现两处方案与代码不符（§7.1、§7.2）；C2 的语义与推理开关按用户反馈定稿，见 **§7.5**。

目标：

1. 预处理时**保留小范围的 root XZ 运动**（注意是 **effective translation root**，不一定是 joint 0），
   让模型学会原地小位移。数据集里大部分是 in-place 的 locomotion，带微弱 XZ 漂移；
   全剥会降低动作质量。
2. 生成时**能正确重建 root motion**，包括 `--loop` 的动作。

方案（四条主改 + 一条附带；c5 不做，见 §2.5）：

| | 改动 |
|---|---|
| **C1** | 删掉 `[0.08, 0.6]` 模糊带（门 B）。只留 `> 0.6` 全剥（门 A），`<= 0.6` 原样保留 |
| **C2** | 加一个标志位告诉模型"这段被 strip 过"，走现成的 `Linear(1,d)` 条件通路（**语义见 §7.5：provenance，不是 content**） |
| **C3** | 生成期不再对 loop 强制置零，一视同仁地积分重建 root XZ |
| **C4** | `motion_np[:, R, 0/2] = 0` 的 RIC 清理改成**无条件**执行（今天只在 `--loop` 下做） |
| ~~c5（附带）~~ | ~~`root_xz_is_closed` 移出 `is_loop` 的与条件~~ —— **不做**（§2.5：只会加 loop 标签，误标 19 条单次动作） |
| c6（附带） | 把 `loop_wrap_loss` 的 terminal 项在 root 的 ch0/ch2 上 mask 掉（terminal 速度约定不变，§1.4） |

---

## 1. 现状

### 1.1 特征布局：root XZ 只有一个载体

每关节 13 维：`[0:3]` RIC 位置、`[3:9]` 6D 旋转、`[9:12]` 局部速度、`[12]` 接触。

`get_rifke`（[features.py:147](../data_loaders/truebones/truebones_utils/features.py#L147)）
把**所有**关节减去 translation root 的 XZ，**包括 root 自己**：

```python
positions[..., 0] -= positions[:, R:R+1, 0]
positions[..., 2] -= positions[:, R:R+1, 2]
```

所以对 translation root `R`：

| 通道 | 内容 |
|---|---|
| `ch0`, `ch2` | **恒等于 0**（构造性，任何 clip 都一样，不携带信息） |
| `ch1` | root 世界高度 |
| `ch9`, `ch11` | **世界 root XZ 轨迹的唯一载体** |

`r_rot` canonical 化之后是 identity
（[features.py:185](../data_loaders/truebones/truebones_utils/features.py#L185) 的长注释），
所以 `ch9/ch11` 就是世界系下的每帧 XZ 位移。重建端对称
（[features.py:897](../data_loaders/truebones/truebones_utils/features.py#L897)）：

```python
r_pos[..., 1:, [0, 2]] = translation_features[..., :-1, [9, 11]]
r_pos = np.cumsum(r_pos, axis=-2)
```

即 `root_world_xz = cumsum(vel_xz[:-1])`。链路本身自洽、精确可逆。
注意 **terminal 行（最后一帧）不参与重建**（用的是 `[:-1]`）。

`R` 是 `translation_root_index`，**不一定是 0**：truebones 有 269 条 clip 是 1、43 条是 2
（Horse 的 `Bip01`），unitybundles 有 278 条是 1。全链路已经正确地按这个索引取值。

### 1.2 两道门

[features.py:640-670](../data_loaders/truebones/truebones_utils/features.py#L640)
`extract_motion_features_from_aligned_anims`：

```python
xz_extent      = xz_locomotion_extent(export_anim, R)      # 离中心原点的最大 XZ 距离
has_locomotion = xz_extent > ROOT_XZ_STRIP_THRESHOLD       # 门 A: > 0.6

if not has_locomotion and 0.08 <= xz_extent <= 0.6:        # 门 B: 模糊带
    ...  wrap_gap_p75 = p75(dist(root_rel_pos[-1], root_rel_pos[0]))
    if wrap_gap_p75 < LOOP_DETECTION_STEP_MIN:             # < 0.02 → 当成"循环行进"
        has_locomotion = True

if has_locomotion:
    motion_anim        = strip_translation_root_xz(new_anim, R)
    motion_export_anim = strip_translation_root_xz(export_anim, R)
...
if has_locomotion:
    local_vel[:, R, [0, 2]] = 0.0
    terminal_local_vel[R, [0, 2]] = 0.0
```

阈值 `0.6` 在 HML 归一化单位下约等于 **43% 的 body span**
（`HML_REF_MAX_SPAN = 1.389`，
[param_utils.py:167](../data_loaders/truebones/truebones_utils/param_utils.py#L167)）。

**门 B 是问题所在。** 它挑的是"位移在 0.08~0.6 之间、但首尾姿态高度闭合"的 clip ——
这正是原地挥击 / idle / 出去再回来的摆动的画像。§2 的实测显示它 **90% 打在 stationary 上**。

`strip_translation_root_xz`
（[animation_utils.py:842](../data_loaders/truebones/truebones_utils/animation_utils.py#L842)）
本身没问题，它正确处理了 `R != 0` 的骨架。**要改的是调用条件，不是实现。**

### 1.3 loop 判据

[animation_utils.py:425](../data_loaders/truebones/truebones_utils/animation_utils.py#L425)：

```python
is_loop = wrap_gap <= effective_tolerance and root_xz_is_closed     # 净位移 <= 0.08
```

一个可证明的耦合：**门 B ⟹ is_loop = True**。
门 B 的条件是 `wrap_gap_p75 < 0.02`，而 `effective_tolerance = clamp(2.2*envelope, 0.02, 0.08) >= 0.02`，
且两处 `wrap_gap` 算的是同一个量（`get_rifke` 只减 XZ，和门 B 的 `root_rel_pos` 一致）；
剥零后 `root_xz_is_closed` 又恒真。所以门 B 的 clip 一定被标成 loop。

反过来说：**删掉门 B 之后，这些 clip 保留了净位移，其中净位移 > 0.08 的会丢掉 `is_loop` 标签**。
这是删门 B 唯一的"代价"；§2.5 的实测显示该代价可接受（受影响的大多是
单次动作，本来就该非循环），最初考虑的 c5 因此不做。

### 1.4 terminal velocity 约定

[features.py:126](../data_loaders/truebones/truebones_utils/features.py#L126)
`_compute_terminal_local_velocity`：loop 时最后一行写 `pos[0] - pos[-1]`。
注意传入的是 `global_positions`（**世界系**，[features.py:686](../data_loaders/truebones/truebones_utils/features.py#L686)），不是 RIC。

对非 root 关节，这是步态周期闭合 delta，与逐帧速度通道（ch9/ch11 = `r_rot·Δworld_pos`，
也是世界系）一致。对 translation root 的 XZ，它等于 clip 的**净位移** —— 对整周期步态
就是"周期的最后一步"。`_tile_loop_motion` 只拼接行、位置是速度的 `cumsum`，
所以 tile 接缝**天然连续**：接缝处的 wrap delta 正好接上下一个 tile 的第一步
（周期序列里它精确等于 `v[0]`）。**这个约定本身是正确的，不需要改。**
今天它被门 A/B 的置零掩盖；C1 保留位移后，步态 loop 的 root XZ terminal 变成非零
（最后一步），唯一暴露出的真问题是 `loop_wrap_loss` 的 terminal 项（未 mask root ch0/ch2）
退化成"把 terminal 速度压向 0"，与真值最后一步对着拉。这就是 c6（只改 loss 侧）。

（terminal 行不参与重建（重建用 `[:-1]`），只影响训练：`loop_wrap_loss` 的 terminal 项、
`velocity_consistency_loss`、`resample_motion_features(loop_terminal=True)`。）

### 1.5 生成期

`sample/generate.py` 只在 `--loop` 时做一次后处理
（[generate.py:926](../sample/generate.py#L926)、[:1627](../sample/generate.py#L1627)），
调 `_close_loop_root_xz_via_velocity`（[generate.py:1840](../sample/generate.py#L1840)）：

```python
motion_np[:, R, 0] = 0.0; motion_np[:, R, 2] = 0.0      # RIC 恒等式清理
drift = sum(vel[:-1, R, [9, 11]])
vel[:-1, R, [9, 11]] -= drift / (T - 1)                 # 强制净位移 = 0
vel[-1,  R, [9, 11]]  = 0.0
```

两件事混在一个函数里，其中**只有第一件是无条件正确的**：
RIC root XZ 结构性为 0，模型在那儿输出的噪声会让导出的 root 和积分出的 `r_pos` 打架
（`recover_from_bvh_ric_np` 里 `世界位置_j = RIC_j + r_pos`，root 的 RIC XZ 非零就等于凭空偏移）。
第二件（强制净位移归零）则让 `--loop` 永远不可能行进。

### 1.6 数据增强：不用动

| 操作 | 位置 | 说明 |
|---|---|---|
| `_circular_roll_motion` | [dataset.py:192](../data_loaders/truebones/data/dataset.py#L192) | 纯 index roll ✅ |
| `_tile_loop_motion` | [dataset.py:200](../data_loaders/truebones/data/dataset.py#L200) | 拼接速度序列 = 行进累加，物理正确 ✅（与 c6 无关，c6 只改 loss 侧） |
| `resample_motion_features` | [dataset.py:133](../data_loaders/truebones/data/dataset.py#L133) | 已经是"积分→插值→差分÷step_scale" ✅ |

---

## 2. 实测：门 A / 门 B 各吃掉多少

方法：对两个数据集的**全部源文件**（GLB）重新加载，按管线的算术重算
`xz_extent = scale_factor * max_t ||(q ⊗ (p_R(t) - p_R(0)))_xz||`
和门 B 的 `wrap_gap_p75`（限定在 cond 保留的关节集上），再按两道门分类。
`translation_root_index` 取自 `motion_metadata`，按关节名回到源文件里定位。

**校验**：把重算结果和 `motions/*.npy` 里 `max|vel_xz| <= 1e-8` 的真值做混淆矩阵，
两个数据集的"应为零"总数都**逐条对上**（truebones 87+213+78 = 378，unitybundles 454+579+350 = 1383），
和 §1 直接从 npy 数出来的剥零数完全一致。所以下表是实测，不是估计。

### 2.1 两道门各吃掉多少

| | truebones（970） | unitybundles（2585） |
|---|---|---|
| 门 A 剥零（extent > 0.6） | 87 （9.0%） | 454 （17.6%） |
| **门 B 剥零（[0.08, 0.6] 且首尾闭合）** | **213 （22.0%）** | **579 （22.4%）** |
| 源文件 root XZ 本来就是 0 | 78 （8.0%） | 350 （13.5%） |
| 今天保留且确有 root XZ | 592 （61.0%） | 1202 （46.5%） |

**门 B 的杀伤在两个数据集上都是 22%，是门 A 的 1.3~2.4 倍。**

### 2.2 门 B 打在谁身上

按 `action_group` 拆（门 A / 门 B / 保留）：

| | truebones | unitybundles |
|---|---|---|
| stationary | 43 / **199** / 324 | 116 / **509** / 803 |
| locomotion | 18 / 17 / 189 | 385 / 70 / 152 |
| transition | 33 / 0 / 147 | 28 / 0 / 522 |

**门 B 的 787 条里 708 条（90%）是 stationary，transition 一条都没打中。**
它就是一个专杀原地动作的门 —— 需求里"in-place 的 walk/run 也有轻微 XZ 位移，全剥会降质量"
说的正是这批。

门 A 则相反：unitybundles 的 529 条里 385 条是 locomotion，extent 中位数 1.439（超过一个
body span），最大 11.7；truebones 门 A 的 extent 中位数 0.907、最大 25.1。**它剥的是真行进，
保留门 A 是对的。**

活样本：`Alligator_Bite1` extent = 0.592、wrap_p75 = 0.000 → 被门 B 整段剥零。
一个咬合动作的前扑位移，正是该保留的东西。

### 2.3 门 B 的 clip 其实是"出去再回来"

| 门 B 的 | truebones | unitybundles |
|---|---|---|
| extent p50 / p90 / max | 0.179 / 0.448 / 0.598 | 0.226 / 0.504 / 0.599 |
| **净位移 p50 / p90 / max** | **0.002** / 0.014 / 0.598 | **0.000** / 0.201 / 0.595 |

门 B 挑的是**姿态**闭合，而这些 clip 的**位置**也基本闭合（净位移中位数 ≈ 0）。
它们是挥击、闪避、摆动这类"出去再回来"的动作，不是行进步态。

这条实测有两个直接后果：

1. **不需要做趋势/残差分解**（本文档 v1 的方案）—— 没有趋势可分，直接保留即可。
2. **c6 的影响面很小；c5 不做（见 §2.5）**。改动后保留 root XZ 的全体里：

| | truebones | unitybundles |
|---|---|---|
| 保留 root XZ 的 clip | 876 / 970 （90.3%） | 2056 / 2585 （79.5%） |
| 其中 `is_loop` | 600 | 1463 |
| **其中净位移 > 0.08（c6：loss terminal 项与真值非零 terminal 速度冲突）** | **5 （0.8%）** | **66 （4.5%）** |

c6 是真 bug、3 行 mask、值得修（terminal 速度约定本身正确，见 §1.4）。c5 最初也考虑了，实测后不做 —— §2.5。

### 2.4 标志位的 epsilon（决定 C2 怎么写）

源文件 root XZ 恒为 0 的 clip，经过 `get_hml_aligned_anim` 的局部位置反解之后，
npy 里会留下**数值残差**：truebones 150 条里有 73 条、unitybundles 581 条里有 231 条
超过 `1e-8`（残差 p90 ≈ 1e-7，最大 3.7e-7）。

| eps | 残差被正确归零 | 真运动被误判为零 |
|---|---|---|
| `1e-8` | 427 / 731 | 1 / 1491 |
| **`1e-6`** | **731 / 731** | 44 / 1491 |
| `1e-5` | 731 / 731 | 97 / 1491 |

⇒ **C2 的判据用 `1e-6`，不能用 `1e-8`**。`1e-8` 会把 304 条实际原地的 clip 错标成
"有 root 位移"，正好把零尖峰重新混回去，等于白改。`1e-6` 下被"误判"的那 44 条，
其 `max|vel_xz| < 1e-6`（相对 body span 1.389 是 1e-6 量级），叫它原地本来就是对的。

### 2.5 c5 不做：实测翻转集合（2026-09-08）

在 C1 下对**全部源文件**重跑管线（3555 条 clip，其中 8+235 条 joint-count 不匹配跳过），
比较两种 `is_loop` 定义，翻转集合如下：

| | truebones | unitybundles |
|---|---|---|
| `is_loop`（现状，保留 `root_xz_is_closed`） | 642 | 1668 |
| `is_loop`（c5，移出） | 647 | 1734 |
| **翻转数** | **5** | **66** |

三个结构性事实：

1. **c5 只能单向翻转：非循环 → 循环**。去掉一个合取项只会扩大集合
   （`(wrap ≤ tol) ⊇ (wrap ≤ tol) ∧ closed`）。
   所以本文档原话"删掉门 B 会顺手把一批真循环降级成非循环，c5 来防止"是**自相矛盾**的：
   c5 删不掉任何 loop 标签，只能加。"降级"指的是 C1 之后这些 clip 的 `is_loop`
   天然从 True（剥零后恒真）变 False（净位移 > 0.08），与 c5 无关。
2. **71 条翻转全部来自原门 B 的 clip**（kept 集合 0 条），且全部是
   "姿态闭合（wrap ≤ tol）+ 位置不闭合（净 XZ 位移 > 0.08）"。
3. 71 条分两类：**52 条真步态**（Walk/Run/Strafe/Crawl/Fly，可平铺的单步幅）
   + **19 条单次动作**（Attack/Damage/Stand/BackUp：`MB_TigerDrago_Attack`、
   `IAC_Caveman_Attack1/2`、`IAC_Mammoth_Attack*`、`FireAnt_AntBackUp`、`Bear_Stand` 等）。
   `wrap ≤ tol` 无法区分这两类 —— 姿态闭合与 XZ 漂移可以并存。

结论：c5 是把钝刀。52 条步态被标成 loop 算"对"（前提是 c6 也做）；
19 条单次动作被误标成 loop，包括已人工确认非循环的 `MB_TigerDrago_Attack`。
保留 `root_xz_is_closed` 在 `is_loop` 里是安全的选择：单次动作正确保持非循环，
唯一代价是带净漂移的步态不参与 tiling 增强 —— 小损失，不是正确性问题。

若将来需要"步态可平铺"的语义，正确做法是加一个由 `action_group` / 步态白名单
决定的独立"可平铺"标志位，而不是去掉 `root_xz_is_closed`。

---

## 3. 方案

### C1 · 删掉门 B（`features.py`）

[features.py:650-665](../data_loaders/truebones/truebones_utils/features.py#L650) 整个模糊带分支删除，
只保留：

```python
has_locomotion = xz_extent > ROOT_XZ_STRIP_THRESHOLD
```

`LOOP_DETECTION_ROOT_XZ_TOLERANCE` 和 `LOOP_DETECTION_STEP_MIN` 在 `features.py` 里
只被门 B 用到（[:657](../data_loaders/truebones/truebones_utils/features.py#L657)、
[:664](../data_loaders/truebones/truebones_utils/features.py#L664)），import 随之移除；
两个常量在 `animation_utils.py` 里另有用途，定义保留。

被门 A 剥掉的那部分**接受信息丢失** —— 它们是真行进（§2 显示 extent 中位数接近或超过一个
body span），模型本来也不该在缺少行进条件的情况下去拟合它们。C2 的标志位负责告诉模型
"这一段的 root XZ 是零"，避免它把两类样本混起来平均。

### C2 · 标志位（`features.py` → metadata → collate → `anytop.py`）

**语义建议用 content 而不是 provenance**：判据是"这段的 root XZ 是不是 0"，
而不是"这段有没有被 strip 过"。

```python
ROOT_XZ_ZERO_EPS = 1e-6     # 见 §2.4：1e-8 会漏掉局部位置反解留下的数值残差
motion_labels['root_xz_zero'] = bool(
    np.max(np.abs(local_vel[:, R, [0, 2]])) <= ROOT_XZ_ZERO_EPS
)
```

理由：一行判据，且**自动覆盖源生 in-place 的那批**（§2.1：truebones 78 条、unitybundles 350 条
源文件 root motion 本来就被作者烘掉了）。这些 clip 今天走的是"保留"分支，
但内容和被剥的一模一样 —— 用 provenance 语义会把它们错分到 flag=0，
反而把零尖峰重新混进"有位移"那一类。

**epsilon 必须是 `1e-6`，不能是 `1e-8`**（§2.4）：局部位置反解会给"源文件恒为零"的 clip
留下 ~1e-7 量级的数值残差，`1e-8` 会把其中 731 条里的 304 条错标成"有位移"。

链路（全都是现成的）：

1. `dataset_pipeline._build_motion_metadata_entry` 写进 `motion_metadata.json`；
   `_copy_required_motion_metadata`（[dataset.py:62](../data_loaders/truebones/data/dataset.py#L62)）
   是整字典拷贝，**不需要改**。
2. [tensors.py:307 / :376](../data_loaders/tensors.py#L307) 两个 key 列表里加
   `'root_xz_zero'`，让它过 collate。
3. `model/anytop.py` 照抄 `loop_condition_projection`
   （[:129-136](../model/anytop.py#L129) 定义、[:904-911](../model/anytop.py#L904) 注入）：

```python
self.root_motion_projection = nn.Sequential(
    nn.Linear(1, self.latent_dim), nn.GELU(),
    nn.Linear(self.latent_dim, self.latent_dim),
)
...
timesteps_emb = timesteps_emb + self.root_motion_projection(root_motion_condition)
```

4. 生成期：`--in_place` 开关喂 1，默认 0。

> `nn.Linear(1, d)` 天然吃连续标量。将来如果二值太粗，直接换成"行进速度"标量即可，
> **不用改结构、不用改 checkpoint 形状**。这是把门留着，不是现在就做。

### C3 · 生成期一视同仁（`generate.py`）

删掉两处 `if getattr(args, 'loop', False): _close_loop_root_xz_via_velocity(...)`
（[:926](../sample/generate.py#L926)、[:1627](../sample/generate.py#L1627)）。
root XZ 从此只由 `ch9/ch11` 积分出来，loop 和非 loop 走同一条路。

### C4 · RIC 清理改成无条件（`generate.py`）

把 `_close_loop_root_xz_via_velocity` 里那两行拆出来成一个小函数，
在两处导出前**无条件**调用：

```python
def _zero_root_ric_xz(motion_np, R):
    """RIC 里 translation root 的 X/Z 结构性为 0（get_rifke 的构造）。
    模型在这两个通道上的噪声会和积分出的 r_pos 打架，导出前清掉。"""
    motion_np[:, R, 0] = 0.0
    motion_np[:, R, 2] = 0.0
```

这是今天唯一被 `--loop` 挡住的正确行为。

### c5 · ~~`root_xz_is_closed` 移出 `is_loop`~~ —— **不做**

§2.5 实测：c5 只能翻转非循环 → 循环（去掉合取项只扩大集合），71 条翻转全部是
原门 B 的"姿态闭合 + 位置不闭合"clip，其中 19 条单次动作（Attack/Damage，
包括人工确认非循环的 `MB_TigerDrago_Attack`）会被误标成 loop。原话"c5 防止真循环
被降级成非循环"自相矛盾 —— c5 删不掉任何 loop 标签。

**`is_loop` 保持现状**：`wrap_gap <= effective_tolerance and root_xz_is_closed`，不动。
`root_xz_total_disp` / `root_xz_is_closed` 本来就是诊断量，
`tools/compute_loop_unclosure_error.py` 继续报。

（可选：顺手删掉 [animation_utils.py:493](../data_loaders/truebones/truebones_utils/animation_utils.py#L493)
的 `root_xz_steps`（算了没用）。）

### c6 · loss mask（terminal 速度约定保持不变）

terminal 速度约定**保持现状**（§1.4）：世界系 wrap delta 就是"步态周期的最后一步"
的真值，tile 接缝天然连续，不存在"反向跳变"。**不改**
`_compute_terminal_local_velocity`，也不改 `utils/npy_roundtrip_utils.py:79`。
（改成"平均步长"反而会让接缝与步态第一步不一致。）c6 只改 loss 侧：

**(a) `loop_wrap_loss` 的 terminal 项**
（[gaussian_diffusion.py:652](../diffusion/gaussian_diffusion.py#L652)）：

```python
terminal_residual = first[:, :, 0:3, 0] - last[:, :, 0:3, 0] - last[:, :, 9:12, 0] * step_scale
```

对 root 的 `ch0/ch2`，`first - last` 恒为 0，该项退化成把 terminal 速度压向 0 ——
与 C1 后步态 loop 写进去的真值 terminal 速度（最后一步，非零）直接对着拉。
pose 项已经 mask 了 root 的 ch0/ch2（[gaussian_diffusion.py:623-624](../diffusion/gaussian_diffusion.py#L623)），
terminal 项只用了 `joint_weight`、没有 mask —— 照 `pose_weight` 的写法 mask 掉这两个通道即可。

**(b) `velocity_consistency_loss`（值得修，效应可达 ~7%）**
（[gaussian_diffusion.py:440](../diffusion/gaussian_diffusion.py#L440)，`--lambda_vel 0.2` 现在开着，
作用在 `model_output_physical` 上，[gaussian_diffusion.py:1786](../diffusion/gaussian_diffusion.py#L1786)）
同理在 root 的 `ch0/ch2` 上退化成"把速度压向 0"的正则（root 的 x/z 位置预测 ≈ 0，
`finite_diff ≈ 0`，loss ≈ `(vel·step_scale)²`）。

把两项的等效曲率（对 root vXZ 速度 `v`）和 `l_simple` 比一比。注意 **`L` 会抵消**：
`l_simple` 对 `N = 13·J·L` 个 entry 取平均，root ch0 速度占其中 `L` 个，
曲率 = `2/(std_v²·13·J)`（无 `L`）；`vel_loss` 分母 `J·(L-1)·3`，帧求和的 `L-1` 也抵消，
曲率 = `2·step_scale²/(J·3)`（无 `L`）。平衡点（`step_scale ≈ playspeed ≈ 1`）：

```
相对收缩 = λ_vel · (13/3) · step_scale² · std_v²
         = 0.2 · 4.333 · std_v²  =  0.867 · std_v²
std_v ∈ [0.055, 0.285]   =>   收缩 [0.26%, 7.0%]
=> 平衡点上 root vXZ 约为真值的 93% ~ 99.7%
```

**原"<0.5%"的估计多乘了一个 `L²`（约 20 倍低估）。实际效应是 root vXZ 速度最多收缩 ~7%，
即 loop 步态的 root XZ 轨迹被压缩 ~7% —— 是真实、可累积的偏差。**
与 (a) 一并修：照 `pose_weight` 的写法 mask 掉 root 的 ch0/ch2。

---

## 4. 实施清单与代价

| # | 改动面 | regen | 重训 |
|---|---|---|---|
| C1 | `features.py`（删一个分支） | ✅ **全量** | ✅ |
| C2 | `features.py` / `dataset_pipeline.py` / `tensors.py` / `anytop.py` / `parser_util.py` | ✅ 随 C1 | ✅ |
| C3 | `generate.py`（删两处调用） | ❌ | ❌ |
| C4 | `generate.py`（拆函数 + 无条件调用） | ❌ | ❌ |
| ~~c5~~ | **不做**（§2.5） | ❌ | ❌ |
| c6 | `gaussian_diffusion.py`（两处 loss mask：loop_wrap terminal 项 + velocity_consistency） | ❌（只改训练侧，不 regen） | ✅ 随 C1 重训 |

一次全量重新预处理 + 一次重训覆盖 C1/C2；c6 是训练侧 loss 改动、不 regen，随 C1 的重训一起上；
C3/C4 可以先上、对现有 checkpoint 立即生效。

**验证器**（`utils/validate_anytop_dataset.py`
[:475](../utils/validate_anytop_dataset.py#L475)）：`--root-motion-threshold` 的语义不变
（仍然是 `ROOT_XZ_STRIP_THRESHOLD = 0.6`），因为改动后保留的 clip 按定义都 `<= 0.6`。
建议新增一条断言：`root_xz_zero == True` 的 clip 其 `max|vel_xz|` 必须是 0，
反之必须非 0 —— 这是 C2 标志位有没有正确落盘的直接判据。

**受影响的测试**：
`tests/test_recover_animation_translation_root.py`（引用 `ROOT_XZ_STRIP_THRESHOLD`）、
`tests/test_generate_inpainting.py`（`test_close_loop_root_xz_distributes_velocity_residual` /
`test_close_loop_root_xz_noop_for_invalid_root` —— 被测函数在 C3/C4 后不再存在，
改成测 `_zero_root_ric_xz`）。
新增：门 B 删除后模糊带 clip 的 root XZ 非零；`root_xz_zero` 标志位与特征内容一致；
loop clip tile 2 次后 root XZ 路径无跳变（验证 §1.4 的连续性结论）。

---

## 5. 验收判据

1. **保留率**：stationary 组里 root XZ 非零的 clip 占比，从今天的水平升到 §2 表里
   "kept + 门 B" 的合计比例（两个数据集都应 > 85%）。
2. **标志位自洽**：`root_xz_zero` 与 `max|vel_xz| > 0` 严格互补，无例外。
3. **loop 数量**：c5 不做、`is_loop` 定义不变；C1 之后 71 条（5/66）原门 B clip
   由 loop（剥零后恒真）变为非循环（净位移 > 0.08，见 §2.5），人工抽检 20 条
   确认标注合理（单次动作保持非循环）。
4. **roundtrip**：特征 → `recover_animation_from_motion_np` → BVH，
   root XZ 世界轨迹与源文件（减去门 A 剥离量）的最大误差 < 1e-4。
5. **tile 连续性**：loop clip tile 3 次，root XZ 路径无一帧跳变（现有约定已满足，
   §1.4；c6 只改 loss，验收时确认无回归）。
6. **生成质量（最终判据）**：重训后对 in-place 的 walk / idle 目视对比 ——
   脚步应该带出对应的 root 微位移，而不是脚在动、root 焊死。
   `--in_place` 开与关都要能正常出片，`--loop` 下首尾无缝。

---

## 6. 不做什么

- **不改 terminal 速度约定**（原 c6(a)）：世界系 wrap delta 就是"步态周期最后一步"
  的正确真值，tile 接缝天然连续（位置是 `cumsum`，无跳变）；改成"平均步长"
  反而会让接缝与步态第一步不一致。c6 只改 `loop_wrap_loss` 的 terminal 项 mask。
- **不做 c5**：§2.5 实测翻转集合表明 c5 只会加 loop 标签（非循环 → 循环），
  且会把 19 条单次动作（含 `MB_TigerDrago_Attack`，人工确认非循环）误标成 loop。
  若需要"步态可平铺"语义，用独立的"可平铺"标志位，而不是去掉 `root_xz_is_closed`。
- **不做趋势/残差分解**（本文档 v1 的方案）。实测显示模糊带里的 clip 净位移中位数接近 0 ——
  它们是"出去再回来"，本来就闭合，没有趋势可分。为一个不存在的问题引入
  per-clip 趋势向量、三档降级阶梯和生成期趋势注入是过度设计。
- **不动 `strip_translation_root_xz` 的实现**：它对 `R != 0` 的骨架处理是对的。
- **不动 `get_rifke` / RIC 的定义**：root 的 `ch0/ch2` 恒为 0 是刻意的。
  想把 root XZ 塞进位置通道会和 `_apply_rest_pos`、`bone_length_consistency_loss`、
  导出端的 RIC override 全部冲突。
- **不动三个数据增强**（roll / tile / resample）：§1.6 已确认在新方案下依然正确。
- **不加新特征通道**：13 维布局和 `CANONICAL_FEATURE_SPACE`、`cond` 的 key set
  稳定性绑定（改 key set 会触发 `torch.compile` 重编译，
  见 `cond_key_set_must_be_stable_for_compile`）。标志位走条件 token，不进特征维度。


---

## 7. 实施记录（2026-09-08）

按 §3 落地时对每条主张做了代码核对与实测复算，两处与方案不符：

### 7.1 C2 的链路少了一步（已补）

§C2 第 2 步只写了 `tensors.py:307 / :376` 两个 key 列表。实际上这两处只把
`root_xz_zero` 放进 **item**；要变成 `cond['y']` 里的张量还必须加进
[tensors.py:190](../data_loaders/tensors.py#L190) 的 `for key in ('is_loop', 'loop_full_cycle')`
布尔 collate 循环 —— 否则 `y.get('root_xz_zero')` 永远是 `None`，条件通路静默失效。
三处都已加。

另外在 [dataset.py](../data_loaders/truebones/data/dataset.py) 的 `_prepare_sample` 里把
标志位显式归一化（`bool(...get(..., False))`），保证 key 在每个 item 上都存在 ——
key 时有时无会触发 `torch.compile` 重编译（见 `cond_key_set_must_be_stable_for_compile`）。

### 7.2 species-level translation root 契约（2026-09-08 已修复）

实测三个数据集的 `motions/*.npy`：按 `motion_metadata.json` 记的
`translation_root_index` 取 RIC ch0/ch2，最大值是 **1.516**（unitybundles）
/ 0.437（truebones），不是 0。

旧流程在特征生成后才按物种众数统一 metadata，没有重编码 tensor，导致少数派 clip
的 metadata R ≠ 提取 R：

| | 不一致 clip 数 |
|---|---|
| truebones zoo | 47 / 970（Dog、Dog-2、SabreToothTiger、Spider、Crow…） |
| truebones zoo_upgrade | 31 / 247（Bear 13 条最多） |
| unitybundles | 172 / 2585（KI 126、MB 40、RMW 6） |

现已改为三阶段契约：

- **阶段 1**扫描物种的全部源动作，按检测票数确定一个固定 root（平票取较小 index）。
- **阶段 2**把固定 root 显式传给每条动作和 rest pose 的特征提取；cond 与每条
  motion metadata 在写出时已经一致。
- `regenerate_dataset_artifacts.py` 只验证契约，不再修改 root provenance；发现旧式
  不一致数据会要求 `--overwrite` 全量重建。
- **阶段 3**对每条 clip（不再仅限 `root_xz_stripped=True`）验证声明 root 的 RIC XZ
  为零，同时验证 motion metadata root 与 cond root 相等。

truebones zoo 全量重建后的 970 条 clip 均满足 metadata root = cond root，且声明 root
的 `max|RIC XZ|` 为 0；原 Dog / Dog-2 五条警告已消失。

### 7.3 实测复核

- **§2.4 的 epsilon**：在现有三个数据集上重算 `max|vel_xz|` 分布，确认
  `(0, 1e-6]` 有一个密集残差簇（unitybundles 441 / truebones 60 / zoo_upgrade 14），
  `(1e-6, 1e-4]` 是空谷，`> 1e-4` 才是真运动。`1e-6` 落在谷里，`1e-8` 会把
  几百条实际原地的 clip 划到"有位移"。**方案正确**。
- **§2.1 的剥零总数**：truebones 实测 364 条恒为 0 + 14 条 ≤1e-8 = 378，与文档的
  87+213+78=378 **逐条对上**。
- **C1 实地重跑**：抽样 80 条"当前被剥零"的 truebones clip 重跑管线，45 条恢复了
  root XZ（`Alligator_Bite3` 0.052、`Eagle_Strike2` 0.151、`Tyranno_HeadButt` 0.115…
  净位移全部 ≈0，正是 §2.3 说的"出去再回来"），35 条仍被门 A 剥零；0 条失败。
  恢复的 45 条 `is_loop` 全部保持 True（与 §2.5 "truebones 只有 5 条翻转"一致）。
- **验收判据 4（roundtrip）**：恢复的 clip 特征 → `recover_animation_from_motion_np`
  → 世界 root XZ，与源动画最大误差 **~1e-8**（判据是 < 1e-4）。
- **c6 两处 mask**：单元验证 root 的 ch0/ch2 速度任意放大都不再改变 loss，
  而非 root 关节、以及 root 的 ch1（高度）仍然被正常监督。

### 7.4 其它

- `CKPT_VERSION` 4 → 5。C1 改了训练数据语义、C2 加了
  `root_motion_projection`、C3/C4 改了生成语义，v4 checkpoint 载入会静默跑错。
- `--in_place` 加在 `add_base_options` 的 `--loop` 旁边，经 `create_condition(in_place=)`
  写进 `root_xz_zero`。
- 新测试 `tests/test_root_xz_motion.py`（20 条）。
  `tests/test_generate_inpainting.py` 的两条改成测 `_zero_root_ric_xz`。
  `tests/test_recover_animation_translation_root.py` 无需改动（它只用门 A 的阈值）。
- 全套 633 + 20 测试通过。


---

## 7.5 C2 定稿：provenance 语义，且推理期没有开关（2026-09-08，用户反馈后）

§C2 原本主张"用 content 而不是 provenance"（判据 = "这段的 root XZ 是不是 0"），
并配 `--in_place` 推理开关。两条都**推翻**：

### 判据改回 provenance：`root_xz_stripped`

> 假设训练时所有动作的 root XZ extent < 0.6（永不触发 strip），则根本不需要这个标志 ——
> 所有动作的物理含义一样（都或多或少带 root XZ 偏移），没有被手工强制置零过。
> 标志位唯一的目的是让模型知道**"这个动作被 strip 过，不要相信它的 root XZ，
> 不要让它污染其它正常样本"**。

推论：**作者本来就烘成原地的 clip 是诚实数据** —— 它的 root 确实不动，这就是这段动画的
真相，模型该照学，没有什么要警告的。只有**本代码写进去的那个 0 才是假的**。
content 判据把这两类混为一谈，等于给诚实样本贴上"不可信"的标签。

落地：

- `motion_labels['root_xz_stripped'] = has_locomotion`，直接取
  `extract_motion_features_from_aligned_anims` 的门 A 判定并沿
  `get_motion` 返回（元组 8 → 9，顺带修了 `utils/auto_retarget.py` 两处
  `*_unused, x, y` 尾部解包 —— 追加字段会被它们静默吃掉）。
- **`ROOT_XZ_ZERO_EPS` / `motion_root_xz_is_zero` 全部删除**。provenance 是管线自己
  知道的确定布尔量，不需要阈值；§2.4 那整套 epsilon 讨论随之作废。
- 模型侧模块改名 `root_motion_projection` → `root_xz_strip_projection`。
- resample 重提取路径拿到的是**已经被第一遍剥过**的 anim，二次门读到 extent≈0 永不触发，
  所以保留第一遍的判定（取 OR）。

实测（truebones 抽样 120 条）：10 条 `stripped=True`（全部精确等于 0.0），
110 条 False，其中 **17 条是"源生原地"**（`Hamster_Walk`、`Camel_IdleLoop`、
`Spider_Walker`、`BrownBear_RunLoop`…）—— 这批在旧的 content 判据下会被错标成
"不可信"，现在正确地标为诚实数据。

### 推理期不加开关

> `--loop`、`--action_label` 等已经隐含了这个动作的 root XZ 是什么；
> 再加一个 in-place 开关容易组合出 OOD。

`--in_place` 已删除，`create_condition` 的 `in_place=` 参数也删除。生成期
**恒定喂 `root_xz_stripped: False`** —— 生成永远要诚实的 root 轨迹，没有理由去要
那个被删掉的伪影，所以这个通道本来就不该暴露给调用方。

### 验证器随之简化

`_validate_root_xz_stripped_flag` 变成**单向且精确**的检查：strip 写的是字面 `0.0`，
所以 `stripped=True` ⟹ `max|root vel XZ|` 必须**恰好等于 0**（无 epsilon）。
反向不检查 —— 源生原地的 clip 落在 0 上是合法的，这正是 provenance 判据的意义。
"kept 但 extent 超阈值" 由既有的 `_validate_root_motion_extent` 覆盖，两者合起来是完备的。
