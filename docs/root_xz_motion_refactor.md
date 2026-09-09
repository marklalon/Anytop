# root XZ 运动处理规则

预处理保留小范围的原地 root XZ 运动（模型学会"脚在动、root 微移"），把真正走出去的步态
拉回原地，并给所有位移封顶；生成期一视同仁地积分重建 root motion。
`CKPT_VERSION = 6`，v5 及更早的 checkpoint 会被明确拒绝。

---

## 1. 特征布局：root XZ 只有一个载体

每关节 13 维：`[0:3]` RIC 位置、`[3:9]` 6D 旋转、`[9:12]` 局部速度、`[12]` 接触。
`get_rifke` 把所有关节（含 root 自己）减去 translation root 的 XZ，所以对 root `R`：

- `ch0 / ch2`（RIC XZ）**结构性恒为 0**，不携带信息；
- `ch1` 是 root 世界高度；
- **`ch9 / ch11` 是世界 root XZ 轨迹的唯一载体**，重建端 `root_world_xz = cumsum(vel_xz[:-1])`。

预处理把 wrapper 折叠后 `R` 恒为骨骼 0（§2.5）。loop 判据保持
`is_loop = wrap_gap <= effective_tolerance and root_xz_is_closed` 不变；
terminal 速度约定（loop 时最后一行 = 世界系 wrap delta，即"步态周期的最后一步"）不变，
tile 接缝天然连续。

---

## 2. 预处理规则

### 2.1 漂移扁平化（朝向帧运输量）

**判据：`locomotion 标签` AND `朝向帧净运输量 > 0.08`，缺一不可。**

- **标签**（`load_locomotion_clip_names` 读 `action_labels.jsonl` 的 `action_group == 'locomotion'`）
  只给"许可"：姿态闭合 + 结束时位移，单周期步态 take 和扑击在几何上无法区分，
  纯测量判据在真实数据上不成立。数据集未标注时集合为空 ⇒ 一条都不改，并打印显式警告。
- **测量**：作者本来就烘成原地的 locomotion 不少，对它们施加算子只会引入浮点噪声。

阈值 `ROOT_XZ_DRIFT_THRESHOLD = 0.08`（≈ 5.8% body span）。drift 分布平滑无谷，
这是政策选择而非数据分界。

**算子**（`flatten_root_xz_drift`，[animation_utils.py](../data_loaders/truebones/truebones_utils/animation_utils.py)）：

在世界系里转向步态的速度方向随角色一起转，没有可减的直线趋势——端到端线拟合会留下
整段弧矢（`MB_Unka_GlideLeft` 0.371、`IAC_Cavewoman_RunTurnRight` 0.732，分别是 27%、
53% 个 body span）。转到 **root 自身朝向帧**后，同一运动是近似恒定的前进速度，减掉
一个常数正是旧线拟合本来在做的事，只是选对了坐标系：

1. `root_xz_heading` 取 root 每帧世界朝向（+Z 前轴在 XZ 平面的角，`np.unwrap` 展开成连续
   斜坡）；动画须先 `collapse_translation_root_chain` 把惰性 wrapper 折到 root 上，
   root 的世界旋转才带上 wrapper 的 yaw。
2. `_linear_fit` 对该朝向做最小二乘直线拟合（匀速转向在 yaw 上就是一条直线）——
   拟合出的线是角色真正跟随的行进朝向，把每步骨盆抖动滤掉。
3. 帧间位移按**负朝向**旋进朝向帧 → 减其均值（= 持续运输量）→ 按朝向旋转回世界系 →
   从 frame 0 累加。`correction` 即累积运输量，`flattened = traj - correction`，
   **frame 0 留在原点**，周期内的涌动与侧摆全部保留。

`drift = ||correction[-1]||`，取**终点**而不是最大偏离：只有终点能区分"走了不回来"
和"出去再回来"。闭合路径（含整圈绕行）按定义是"原点附近的周期性运动"，一律保留。

**校验器从特征张量读回同一朝向**（`recover_from_bvh_rot_np` + `root_xz_heading`）：
旋转直接取自 6D 通道，offsets 只摆关节位置、不动朝向。恢复动画带 identity orient，
朝向可能整体差一个常量；但旋进帧、减均值、再旋转回，对帧的**常量旋转等变**，
这个常量精确抵消，所以管线与校验器对同一条 clip 读出同一个 drift。

**foot contact 在 root XZ 编辑之前读取**（`get_contact_state` 读变换前的
`positions_global(new_anim)`）。扁平化等于给每个关节加上步速，踩实的脚随之动起来 ——
在变换后算接触会静默丢掉整个支撑相。终端行仍取变换后的位置（描述平铺接缝）。

### 2.2 root XZ 软 clamp（无门控，所有 clip）

扁平化只处理"走出去的步态"，剩下的位移原样保留但**必须封顶**：
表示层扛不住 `Oscafish_Atk` 冲出 4.79（3.4 个 body span）这类量级。

**0.6 以内完全不动，0.6 往上软压，0.8 是渐近的天花板**
（`soft_clamp_extent` / `soft_clamp_root_xz`）：

```
g(r) = r                            r ≤ 0.6
g(r) = L - w² / (r - k + w)         r > 0.6      k=0.6, L=0.8, w=0.2
```

四条性质缺一不可（这个双曲线是同时满足它们的最简形式）：

| 性质 | 为什么 |
|---|---|
| knee 以下是恒等映射 | 停在原点附近的 clip 逐位不变，不引入浮点噪声 |
| knee 处值和斜率都连续 | 否则 0.6 处出现速度台阶，模型会把它当成数据特征学走 |
| 严格单调 | 硬 clamp 把所有超限位移压成同一个值，1.0 的扑击和 10.0 的摔落变得不可区分 |
| L 是渐近线而非取值 | 远端平滑减速而不是撞墙 |

不用指数 knee：float64 下超过约 34 个 knee 宽度后它精确等于 0.8，最深处的一批位移
（1.5~4.79）会挤进 1e-3 带子里互相分不开；双曲线按 `1/r` 衰减把它们撑开。

**按帧作用在半径上**，不是整条轨迹乘系数：全局缩放会让"别处的一次扑击"按同一比例压掉
"原点附近的碎步"。按帧映射是位置的纯函数，方向逐帧不变，闭合路径仍然闭合。

**顺序：先扁平化再 clamp，两步合并成一次 FK。** 先 clamp 会把步态每一步都压进天花板，
交到扁平化手里时已经不像步态了。

**显式 opt-in（`clamp_root_xz_extent`），因为它不幂等**：对已 clamp 过的 clip 再跑会二次压缩 ——
retarget、重采样第二遍、NPy 往返都会重新提特征，只有从源动画出发的那一遍才打开它。

### 2.3 物种根：取最深 transport carrier

同一个物种的不同 clip 可能把 root motion 写在**不同关节**上（TigerDrago 的骨架
`Cg(0) → Pelvis(1) → Spine(2)`：`GetHitR` 动 Cg，`Run` 动 Pelvis）。物种根必须让每条 clip
的位移都看得见：

- 根在所有 carrier 的下方（或就是它）⇒ 每条 clip 的位移都看得见；
- 根在任何一个 carrier 的上方 ⇒ 那条 clip 的位移**一点都看不见**。

没有折中，也轮不到多数票。但"动过"不等于"承载 transport"：Crow 的 `Spine` 在 idle 时
晃 0.026，而它的 `Pelvis` 飞 0.809 —— 按条数统计救不了（TigerDrago 走 Pelvis 的 clip 占 36%
必须赢，Crow 走 Spine 的占 40% 必须输），**只有幅度能分**。

判据（`chain_xz_travel` / `select_transport_carrier`）：

1. `chain_xz_travel` 量候选链上每个关节**局部 XZ** 相对首帧的最大位移。只看水平 ——
   只上下点头的关节不承载 transport。
2. 单条 clip 的 carrier = **最深的**、同时满足的关节：
   `travel ≥ 0.5 × 本 clip 链上峰值`（`ROOT_TRANSPORT_CARRIER_SHARE`）
   且 `travel ≥ 0.08`（`ROOT_TRANSPORT_MIN_TRAVEL`）。都不满足 ⇒ 这条 clip 不投票
   （Tukan 七条里五条完全不动，旧规则把"只能返回点什么"的答案当成票，压掉了两条真在飞的）。
3. 物种根 = 所有 clip 的 carrier 里最深的那个；一条都没有 ⇒ 层级根。

`find_translation_root` 保持原样：它回答"**这条 clip** 的 root motion 在哪"，
和"**这个物种**的 transport 控制器是哪个"不是同一个问题。

**修正量施加在 carrier 上**（`set_translation_root_xz` + `_transport_carrier_index`：
取"层级根 → R"链条上最靠上的、世界系 XZ 不静止的关节），不是无条件施加在 R 上：
R != 0 且祖先携带位移的骨架（Dog / Dog-2 / KI_*），只安放 R 会把祖先和 R 撕开，
凭空制造相对运动。carrier 下面的一切（含 R）刚性平移，R 依然精确落在目标上。
R == 0 与"静止 wrapper"两种情况都退化回原行为。

### 2.4 垂直 clamp 读冻结根

`clamp_vertical_trajectory` 接收 `process_anim` 传下来的 `translation_root_index` ——
高度和轨迹读在**同一个冻结关节**上。不传时回落到逐 clip 检测：裸 rest pose 和 retarget
的源动画没有物种根可用，回落是对的。

### 2.5 无效根骨：wrapper 折叠，real root 恒为骨骼 0

R 之上的控制节点（`Cg` / `Ctrl` / `All` / 光杆 `Root`）是纯控制节点：链上无分叉，
transport 在 R 或更下面，它们相对 R 的偏移不承载任何运动。放着不管，它们的 RIC 通道
装的是每 clip 一个任意常量 + 一条反向的 root 轨迹 —— 模型必须在 RIC 通道里学会忽略的垃圾。

**预处理层直接把 wrapper 折进 real root**（`promote_translation_root_to_hierarchy_root`
→ `promote_root_once`，[root_collapse.py](../motion_lib/root_collapse.py)），让 translation root
恒等于骨骼 0。不在 FBX/BVH 加载层做：那层是保守兜底，只能靠名字猜，而 34 个 wrapper 里
33 个恰恰叫 `Hips` / `Root` / `Body`，名字分不开；carrier 测量分得开，但它要等 phase 1
把每条 clip 都量过之后才知道。

**两遍收敛**（`_prepare_object_outputs`）：第一遍量出物种根 R；R≠0 就把上面 R 个关节折进
real root，用折完的骨架重跑 phase 1；第二遍落在 0 就是不动点（最多 3 遍，不收敛则
`DatasetPreprocessingError`）。增量预处理从 cond 读回 `root_promote_depth`，第一遍即收敛。
raw 导入结果走 realpath 索引的缓存，几遍之间共用。

**旋转烘焙**：child 的旋转外乘上被丢关节的旋转（`root ⊗ child`），offset 转进父帧再相加，
带动画的位移同理，rest `orients` 同样复合 —— 少了最后一条，bind pose 会相对动画转过去
（Alligator / Scorpion / Deer 上那个 90° 滚转）。34 个 wrapper 物种里 9 个真的在 wrapper 上
写了旋转，最严重的 clip 全是转身，折错 = 转身全丢。折叠前后**留下的每个关节世界变换逐位不变**。

**归一化顺序（关键）**：drop prop socket → drop end site → **fold wrapper** → crop。

- 必须在 drop 之后：祖先只要有第二个子节点就拒绝折；挂在 wrapper 上的 prop socket
  要等 drop 才消失，折在前面会把整个物种判死。
- 必须在 crop 之前：否则 `MAX_JOINTS` 名额先被即将折掉的控制节点占掉，卡在上限的骨架
  每有一个 wrapper 关节就赔掉一根真骨头。

深度本身在完全归一化的骨架上量，比测量点早一个 crop 被应用是安全的：
`select_cropped_joint_indices` 只删当前叶子且永不删根，任何留下身体的 crop 都不可能缩短
根链（`test_crop_never_shortens_the_root_chain`）。
`_load_motion_source` 和 `get_common_features_from_rest_pose` 必须同序，
否则 clip 与 rest pose 对不上关节集会硬失败。

**护栏**：链上任一祖先有第二个子节点就整个跳过（防手工/陈旧索引）。

**重采样分支从源 anim 重新提取**：再提取跑 `source_new_anim` / `source_export_anim`
（变换前的 anim），而不是把第一遍的输出再喂回去 —— 否则会在"刚被去掉漂移的轨迹"上重新
测漂移、在"已经滑脚的 clip"上重读接触。决策一次、施加一次。

**保留待清理**（等全量重新预处理跑通、确认所有物种都落在 0 之后再删）：
`_transport_carrier_index`、`translation_root_ancestor_chain`、
`collapse_translation_root_chain`、`set_translation_root_xz` 的祖先分支、phase 2 的少数派重新对齐、
冻结根的祖先校验，以及 cond / metadata / 验证器里的 `translation_root_index`。

---

## 3. 生成期

- root XZ 只由 `ch9/ch11` 积分出来，**loop 和非 loop 一视同仁**，没有 `--loop` 特判、
  没有 `--in_place` 开关（动作语义已经由 `--action_label` / `--loop` 隐含）。
- 导出前**无条件**调 `_zero_root_ric_xz`（[generate.py:1838](../sample/generate.py#L1838)）：
  RIC 里 root 的 XZ 结构性为 0，模型在这两个通道上的噪声会和积分出的 `r_pos` 打架，清掉。
- 训练侧 `loop_wrap_loss` 的 terminal 项与 `velocity_consistency_loss` 在 root 的
  `ch0/ch2` 上 mask（[gaussian_diffusion.py:675](../diffusion/gaussian_diffusion.py#L675)）——
  这两个通道恒为 0，不 mask 会把真值 terminal 速度（步态最后一步）压向 0。

---

## 4. 验证器不变量（`utils/validate_anytop_dataset.py`）

| 检查 | 范围 | 判据 |
|---|---|---|
| `_validate_root_motion_drift` | 仅 locomotion clip | 用与管线**完全相同**的算术（解码器重建轨迹 + 同一朝向信号），扁平化后净运输量 ≤ `ROOT_XZ_DRIFT_THRESHOLD` |
| `_validate_root_xz_ceiling` | **每一条** clip | root XZ extent ≤ `ROOT_XZ_SOFT_CLAMP_LIMIT` + `1e-3`。超了说明 tensor 来自 clamp 之前 |
| `_validate_root_transport_carrier` | 每一条 clip | 逐帧取非 root 关节 RIC 位移的最小值 = 刚性整体平移的下界；超过天花板 0.8 且 root 自己轨迹不到它的一半 ⇒ 报警（位移被写在了物种根看不见的关节上） |

`--root-motion-threshold` 默认值 = `ROOT_XZ_DRIFT_THRESHOLD`（0.08）。

**预处理硬前提**：`action_labels.jsonl` 与 `species_tags.jsonl` 必须存在（fast-fail，
`load_motion_metadata` 是严格的）。未标注的数据集跑不了预处理，先标注。

**冻结根校验（增量）**：新源的 carrier 在冻结根**之上**是允许的（FK 会把那段平移折进
冻结根的世界位置，特征照样看得见）；在**之下**才是真丢了，FAIL 并要求 `--overwrite`。

---

## 5. 阈值速查

| 常量 | 值 | 含义 |
|---|---|---|
| `ROOT_XZ_DRIFT_THRESHOLD` | 0.08 | 扁平化门限（≈ 5.8% body span），= loop 闭合容差 |
| `ROOT_XZ_SOFT_CLAMP_KNEE` | 0.6 | 软 clamp 不动区上限 |
| `ROOT_XZ_SOFT_CLAMP_LIMIT` | 0.8 | 软 clamp 渐近天花板 |
| `ROOT_TRANSPORT_CARRIER_SHARE` | 0.5 | carrier 须达本 clip 链上峰值的 50% |
| `ROOT_TRANSPORT_MIN_TRAVEL` | 0.08 | carrier 绝对下限（低于它下游不会对这点位移做任何事） |
| `ROOT_XZ_CEILING_TOLERANCE` | 1e-3 | 天花板检查容差 |
| `CKPT_VERSION` | 6 | v5 及更早 checkpoint 被拒绝 |

metadata 里 `root_xz_flattened` 是纯人读的溯源字段，不进模型。

---

## 6. 主要文件

| 文件 | 职责 |
|---|---|
| [animation_utils.py](../data_loaders/truebones/truebones_utils/animation_utils.py) | 全部 root XZ 算子：`flatten_root_xz_drift`、`soft_clamp_*`、`select_transport_carrier`、`set_translation_root_xz`、`promote_translation_root_to_hierarchy_root` |
| [dataset_pipeline.py](../data_loaders/truebones/truebones_utils/dataset_pipeline.py) | 两遍收敛、carrier 物种根、locomotion 门控、`root_promote_depth` 持久化 |
| [features.py](../data_loaders/truebones/truebones_utils/features.py) | `extract_motion_features_from_aligned_anims(flatten_root_travel=, clamp_root_xz_extent=)`；contact 在变换前读取 |
| [root_collapse.py](../motion_lib/root_collapse.py) | `promote_root_once`（旋转/offset/orient 复合） |
| [validate_anytop_dataset.py](../utils/validate_anytop_dataset.py) | 三条不变量 |
| [generate.py](../sample/generate.py) | 无条件 `_zero_root_ric_xz` |

测试：`tests/test_root_xz_motion.py`（扁平化 / clamp / carrier）、
`tests/test_preprocess_skeleton_normalization_order.py`（归一化顺序）、
`tests/test_crop_skeleton_max_joints.py::test_crop_never_shortens_the_root_chain`、
`tests/test_vertical_root_clamp.py`（冻结根 clamp）、
`tests/test_recover_animation_translation_root.py`、
`tests/test_preprocess_incremental_update.py`（`root_promote_depth` / 冻结根祖先校验）。

---

## 7. 仍待执行

**全量重新预处理 + 重训。** `CKPT_VERSION` 5 → 6，v5 checkpoint 不能直接续用。
