# 骨架统计档案 + 动作微调运行时：执行方案

> 状态：M1（Profile 提取器）、M2（分解器 + 严格回放 + UI 基础）、M3（时间 / locomotion / 根 / 幅度参数、锁脚 IK、UI 接触修正）已实现；M4 起未实现。
> 范围：从训练动作数据中统计每个骨架的运动学属性（Skeleton Profile），把 AnyTop 的输出分解为可编辑的中间层（Edit Package），并在微调端用一个固定、确定性的运行时（Edit Runtime）合成最终动作，提供即时、可预期的参数化微调。
> 不在范围内：训练"骨架 → 属性"预测模型；把属性作为 AnyTop 的条件输入。两者都要等本方案的 M7（约束后处理评估）给出结论后再立项。

## 1. 目标与原则

1. **模型只负责生成，微调不再依赖重新抽卡**：AnyTop 输出 `(F, J, 12)` 之后，用户所有的调整都在微调端（微调 UI 或 `apply_edit` 命令行）完成，不重新调用模型，参数相同则结果逐位一致。
2. **代码固定，数据特化**：只有一套带版本号的通用运行时；每个骨架、每个动作的差异全部放在 Profile 和 Edit Package 这两份数据里，不生成任何特化代码。
3. **不碰模型与 cond**：Profile 单独存成文件，不写进 `cond.npy`，所以不需要 regen 也不需要重训，cond 的 key 集合也保持稳定（compile 依赖它）。
4. **默认参数严格回放**：所有参数为默认值时，运行时的输出必须与 `restore_animation_from_features(restore_space="hml", fullbody_ik=k, stretch_factor=s)` 的结果一致，其中 `k`、`s` 是该 package 分解时使用的 `fullbody_ik`、`stretch_factor`（第 4.1 节）。在这个前提之上才谈编辑。
5. **编辑只作用在增量上**：每一层都只叠加"编辑引起的变化"，不重写原结果中与编辑无关的部分。锁脚保留原结果自带的滑步残差，次级运动只叠加新旧父运动的模拟结果之差，着地只补偿编辑引起的高度变化。这样参数从默认值出发连续变化，滑杆在默认值附近不会突变。"修掉原结果的滑步"是一个显式开关（`foot_lock`），不附带在其他参数里。
6. **超出范围就报告，不静默出错**：例如腿伸不到、找不到事件帧，都要作为诊断信息返回，并把对应参数夹住。
7. **验收以人工判断为主**：现有评分机制不可靠，不作为验收依据。自动测试只检查机械不变量（回放、确定性、连续性、闭合、骨长、锁脚），效果好坏由人看渲染结果来判断（第 6 节）。
8. **性能暂不考虑**：先保证正确和可预期，不设耗时指标。

### 1.1 数据前提：locomotion 是原地动作，其他动作保留真实位移

预处理只对 `action_group == locomotion` 的 clip 做根 XZ 去趋势，所以：

- **locomotion 是 in-place**：根的 XZ 只有局部小范围晃动，没有真实的水平位移；
- **非 locomotion 动作保留真实的水平位移**：攻击的突进、受击的击退、死亡的倒地、transition 的位移都在根 XZ 里；
- **Y（高度）对所有动作都带真实位移**，比如跳跃、起身、下落；只有净高度偏移很大的竖直 locomotion（向上飞、向下潜）做过竖直去趋势。

对 locomotion：

- **没有世界坐标下的固定落点**。支撑期的脚相对身体以"隐含地速"向后滑动，就像在跑步机上；
- **速度和步幅从脚的运动推出来，不从根推**：隐含地速 `v_g` = 支撑期接触关节水平速度的共同分量（取反）；步幅 = `v_g` × 周期；
- **"不滑步"的定义**：支撑期内各接触关节的水平速度都等于同一个 `v_g`（这一帧所有支撑脚一致，并且在支撑期内保持平稳）。等价的检查方法是：把位置加上 `v_g` 的累积位移变回"地面坐标系"，再看落点有没有漂移；
- **接触区间不能用"速度接近 0"来判定**：支撑脚在水平方向是匀速后移的，判定规则见第 4.1 节。

对非 locomotion 动作（攻击、受击、idle、起身等），`v_g ≈ 0`：地面坐标系就是世界坐标系，支撑脚的落点就是固定在地面上的点，根的真实位移由身体承担。

**接触、锁脚和 IK 对所有动作都适用，不只针对 locomotion**。攻击时加大幅度、力度、蓄力，或者调整 posture，都会带动髋部和躯干，支撑脚必须靠 IK 留在原地，否则一样会滑步。

## 2. 总体流程

```text
离线（训练数据）
  motions/*.npy + cond.npy + action_labels.jsonl (+ contact_overrides.json)
      → motion_edit.build_profiles → <dataset_root>/skeleton_profiles.json

服务端（每次生成）
  AnyTop → (F, J, 12) features
      → decompose_motion(features, cond 子集, profile, stretch_factor)
          （内部：npy_restore 解码 + fullbody IK 刚体化）
      → edit_package (.npz + manifest.json)

微调（ui/serve.py 或 apply_edit.py 进程内）
  edit_package + 用户参数 → EditRuntime.apply → Animation → BVH / GLB
```

模块划分：所有代码统一放在 `Anytop/motion_edit/` 下，命令行入口用 `python -m motion_edit.<cmd>` 从 `Anytop/` 目录运行。

```text
motion_edit/
  __init__.py
  rotations.py              四元数 / rotvec 工具（profile 与运行时共用）
  contacts.py               接触区间检测（第 4.1 节第 2 步，profile 与分解器共用）
  profile/                  统计提取器：data / skeleton / joints / spring / gait / build / report
  decompose.py              生成结果 → Edit Package；接触关节 / 接触区间的修改
  package.py                Edit Package 的读写与版本号（不导入 torch）
  runtime.py                EditRuntime：package + 参数 → Animation（第 5 节）
  ik.py                     肢体 IK（第 5.2 节第 4 步）
  ui/                       微调 UI：serve.py + index.html + vendor/（第 5.4 节）
  build_profiles.py         命令行：批量构建 Profile 并输出审查报告
  decompose_clip.py         命令行：npy + cond → Edit Package（--stretch_factor、--no_fullbody_ik）
  apply_edit.py             命令行：package + 参数 JSON → BVH / GLB
tests/test_motion_edit_profile.py、tests/test_motion_edit_runtime.py
```

运行时复用仓库已有的实现：四元数和 FK 用 `motion_lib`（`Quaternions`、`Animation`），导出走 `npy_restore` 的 exporter 路径；重采样和肢体 IK 是运行时自己的（第 5.3 节）。运行时不导入 torch，也不读 cond 或数据集目录，所需的骨架数据全部来自 package。v1 不考虑运行时的性能。

## 3. 阶段 A：Skeleton Profile 统计提取器

### 3.1 输入

- cond entry：`parents`、`offsets`、`kinematic_chains`、`contact_joints`、`symmetry_partner_indices`、`joint_side_labels`、`species_tags`、`translation_root_index`、`scale_factor`、`forward_joint_index`、`forward_base_joint_index`。
- `contact_overrides.json`（可选，第 3.4 节）：用户对接触关节集合的手动增删。
- `passive_confirmations.json`（可选，第 3.5 节）：用户确认要做次级运动的叶链关节。
- 该物种所有 clip 的 `motions/*.npy`，形状 `(F, J, 12)`，通道为 pos 3 + rot6D 6 + vel 3。
- `action_labels.jsonl`：`action_group`、`action_label`、`is_loop`，用于按动作族分组统计。

解码统一走 `build_skeleton_only_context` 加 `restore_animation_from_features(restore_space="hml")`，得到局部四元数和全局位置，不另写一条解码路径。训练数据本身就是刚体骨架导出的，统计时不需要开 `fullbody_ik`。所有长度类统计量都除以**腿长**做归一化，这样不同物种之间可以比较。腿长按肢体计算：cond 的 `contact_joints` 通常包含整条足部链（踝、趾、趾尖），所以先把接触关节按所在肢体（肢体链根，即髋或肩）分组，每条肢体取从链根到该肢体静息最低的接触关节的骨长之和，再对肢体取平均；没有腿的骨架改用 `axial_avg_len × scale_factor`。

### 3.2 统计权重

一个 clip 内的所有帧合计权重为 1（clip 内按帧均分），然后再按动作族（`action_group` + 主词）做均衡，避免一个长 idle clip 主导统计结果。每个属性都要记录**覆盖度**：参与统计的 clip 数，以及来自哪些动作族。

### 3.3 每个关节的属性

| 属性 | 算法 | 运行时用途 |
|---|---|---|
| `dof_class` ∈ {fixed, hinge, planar, ball} | 局部旋转相对于该关节的鲁棒均值姿态取 log map，得到 rotvec，再做加权 PCA。总角度标准差低于阈值判为 fixed；第一主成分方差占比 ≥ 0.85 判为 hinge；前两个主成分合计 ≥ 0.9 判为 planar；其余判为 ball | IK 约束、幅度缩放的轴 |
| `principal_axes` (3×3, 父关节坐标系) | PCA 的特征向量 | 同上 |
| `hinge_flex_sign` | 仅对有子骨的 hinge：从数据里读屈曲方向。锚点取最近的、静息位置与该关节不重合的祖先（零长骨会让父关节和它重合），对所有帧计算锚点到子关节的距离与主轴投影角的加权协方差；距离随角度增大而缩短，+主轴就是屈曲方向（+1），反之为 −1；距离几乎不变（数据里从未弯过）为 0。不在均值姿态附近做局部探测，因为接近伸直的均值姿态正处在距离的极大值上，两个方向分不出来 | 原结果接近伸直时 IK 的 pole 方向（第 5.2 节第 4 步） |
| `ang_speed` | 局部角速度的 q50 / q95（rad/s） | 力度参数的上限参考、诊断 |
| `role` ∈ {root, axial, support, swing, passive, other} | 见第 3.4 节 | 决定哪些编辑作用在哪个关节上 |
| `spring` {source, k, c, g_ang, g_lin, g_grav, natural_hz, damping_ratio, swing_length} | 仅对叶链上的候选关节：拟合通过时取拟合值（source = fit），否则取默认值（source = default），见第 3.5 节；拟合的诊断数据另存在 `spring_fit` | 次级运动模拟 |
| `confidence` | 由覆盖度与稳定性（第 3.7 节）综合得出 | 低置信度时回退到默认值 |

### 3.4 接触关节与关节角色

- **接触关节集合不做自动检测**：默认取 cond 的 `contact_joints`，再应用 `contact_overrides.json` 中用户的手动增删。例如 `truebones/zoo/Alligator` 的 cond 里，`contact_joints` 只有后脚 `R_ashi`/`L_ashi`，前脚 `te` 既不在 `contact_joints` 里也不在 `end_effector_joints` 里；用户在微调 UI 里发现前脚滑步时，手动把前脚标成接触关节即可（第 5.4 节）。
- `contact_overrides.json` 放在 `<dataset_root>/` 下，按物种名记录 `{"add": [...], "remove": [...], "skeleton_hash": ...}`，用关节名而不是索引。`skeleton_hash` 对不上时整条覆盖失效并在报告中列出。cond 本身不改；如果要把覆盖写回 cond，走正常的 regen + 重训流程，单独决策。
- 肢体：从一个带左右标记（`joint_side_labels` 为 left / right）的关节往上，直到父关节换了标记为止，最上面那个关节是肢体链根。
- `support`：接触关节，以及从它往上直到肢体链根的所有关节。
- `axial`：其余标记为 center 的关节（躯干、头、尾）。
- `swing`：不接触地面的左右肢体（手臂、翅膀等）；肢体全长不到腿长 0.25 的小附件（耳、嘴角、须）记为 `other`。
- `passive`：用户在 `passive_confirmations.json` 中确认的叶链关节（尾、耳、毛发、触须、翼膜、披风等），不在 support 链上。叶链是从一个叶关节往上、直到最近的分叉关节为止的那一段。

gait 统计（第 3.6 节）需要训练 clip 上的逐帧接触区间，按第 4.1 节的区间检测规则在上述接触关节上计算。报告列出在 locomotion 帧中接触占比低于 5% 的接触关节，供用户判断是否要从接触集合中去掉。

### 3.5 次级运动：候选、确认与弹簧参数

**候选与确认**：每条叶链上、不属于 support 链、且有子骨可摆动的关节都是次级运动候选。报告按叶链列出候选；要启用哪些关节由用户决定，写在 `<dataset_root>/passive_confirmations.json`，按物种记录 `{"confirmed": [...], "skeleton_hash": ...}`（关节名）。确认的关节 `role` 改为 `passive`，运行时才对它做模拟；确认了非候选关节（比如脚）的条目不生效，写进报告。

**弹簧参数**：每个候选关节都带一组参数，来源有两个：

1. **拟合**：数据里确实有被身体带动的运动时，用拟合出的参数（`source = fit`）。
2. **默认**：其余情况都用默认参数（`source = default`）。数据里大部分尾巴、毛发是动画师手 K 的，和身体运动无关，拟合通不过，但这不妨碍它们在编辑后获得跟随感：运行时只叠加新旧父运动的模拟结果之差（第 5.2 节第 5 步），手 K 的原曲线原样保留，弹簧只负责编辑带来的那部分变化。

默认参数把关节下方的链看成一根绕关节摆动的均匀杆：

```text
L       = 关节下方链的静息长度（swing_length 记录 L / 腿长）
f       = 1.5 Hz × (L / 腿长)^(−1/2)，夹在 0.5–4 Hz        （链越长摆得越慢，像单摆）
k       = (2π f)²，c = 2 ζ √k，ζ = 0.3
g_ang   = 1               （父关节转动时，杆按 1:1 落后）
g_lin   = 3 / (2L)        （父关节横向加速时，均匀杆绕端点的角加速度）
g_grav  = 0               （手 K 的姿态已经是动画师要的下垂位置，不再叠加重力）
```

用户还可以用 `secondary.stiffness / damping` 滑杆整体调节。

**拟合**：对叶链上的关节 j（父关节 p）：

```text
θ_j(t)  = 局部 rotvec 相对均值的偏差（p 的坐标系）
α_p(t)  = 父关节在世界坐标中的角加速度，转换到 p 的局部坐标系
a_p(t)  = 父关节在世界坐标中的线加速度，转换到 p 的局部坐标系
ĝ_p(t)  = 世界竖直向下方向，转换到 p 的局部坐标系
b       = 关节 j 在均值姿态下的骨向（p 的坐标系）
模型：  θ̈ + c·θ̇ + k·θ = −g_ang·α_p − g_lin·(b × a_p) − g_grav·(b × ĝ_p) + 常数项
```

三个轴共用一组参数，堆叠成一个线性最小二乘。数值微分用 Savitzky–Golay（窗口 9、三阶）；loop clip 按周期边界处理。每个 clip 末尾 20% 的帧留出，用来计算 R²。

只看 R² 判别不了被身体带动的运动：`θ̈ + c·θ̇ + k·θ = 0` 能完美拟合任何正弦，一条手 K 的周期摆尾即使和父运动无关，R² 也会很高。所以拟合通过要求三条同时满足：

1. 留出帧上 R² ≥ 0.6；
2. 驱动项的贡献：完整模型的留出 R² 比去掉驱动项（g_ang = g_lin = g_grav = 0）的模型高出 `delta_r2` ≥ 0.2；
3. 驱动系数至少有一个显著不为零（系数 / 标准误 > 3），固有频率 √k / 2π 在 0.2–8 Hz 之间，且 c ≥ 0。

拟合通过与否只决定参数来源，不决定是否启用；报告里标出拟合通过的关节，供用户确认时参考。

### 3.6 全局属性与 locomotion 属性

- `leg_length`、`hip_height`（静息姿态）、`axial_length`。
- `gait`：对每个 locomotion clip，按肢体统计（肢体中任一接触关节着地就算该肢体着地；两段着地之间，如果肢体最低的接触关节没有抬到它在该 clip 中摆动高度（q95 − q05）的 25%，就算同一次支撑里的抖动，合并成一段，不算新的一步）：周期 T（帧，取同一肢体相邻两次着地的间隔的中位数，loop clip 也算首尾环绕的那一次）、占空比（duty factor）、各肢体相对第一条肢体的着地相位（以肢体最低的接触关节编号为键）、隐含地速 `v_g`（支撑期均值，腿长 / 秒）、步幅（`v_g` × T / fps，腿长）。按 `action_label` 分组取中位数。步幅只能从脚推出来，因为 locomotion 根的 XZ 没有真实位移。
- `stride_speed_fit`：同一物种所有 locomotion clip 上拟合 `stride = a · v_g^b`。描述该物种步幅随速度变化的规律，供诊断参考；运行时不使用。clip 数不足 3 时不做拟合，固定 b = 0.5。
- `vertical`：对 Y 带真实位移的 clip（根的净高度变化 ≥ 0.25 腿长，或最低关节离地 ≥ 0.15 腿长），按 `action_label` 记录根的净位移和最低关节的离地高度（/ 腿长）的最小、中位、最大值，作为 `jump_height` 参数的参考范围。

### 3.7 提取器自身的验证

1. **稳定性**：把一个物种的 clip 分成两半分别统计，每个动作族的 clip 都分散到两半里（不分层的话，几条 clip 的物种会变成攻击对比 idle）。两半的主轴方差占比相差不超过 0.1（不直接比 dof_class，避免占比恰好落在类别阈值两侧时误报），两半都是 hinge 时主轴夹角 < 15°。不满足的关节 `confidence` 减半，原因写进关节的 `unstable` 字段。
2. **对称性**：只在 locomotion clip 上检查（单手攻击、转身本来就左右不同）。`symmetry_partner_indices` 给出的左右配对，以 X = 0 平面镜像后比较：方差占比相差不超过 0.1；hinge 的转轴、planar 的平面法向夹角 < 20°。全部结果写进 JSON 的 findings；报告只列出镜像轴夹角 ≥ 30° 的，这一类才可能是骨架左右定义本身的问题。
3. **名字合理性**：原名或规范名是 knee/elbow/hiza/hiji 一类的关节应当被判为 hinge。不是的话写进报告，不自动修改。
4. **confidence** = 覆盖度 × 稳定性：覆盖度 = min(1, clip 数 / 8) × (0.5 + 0.5 × min(1, 动作族数 / 3))；不稳定的关节乘 0.5。
5. **报告**：`skeleton_profiles_report.md`，开头是按物种的计数表，下面按物种列出：解码失败、覆盖文件问题（过期、关节名不存在、确认了非候选关节）、次级运动候选叶链（标出拟合通过和已确认的关节）、几乎不着地的接触关节、明显的左右镜像轴不一致、名字和自由度冲突。不稳定和低置信度只在表里给数量：它们主要反映物种的 clip 少、动作族单一，已经体现在 confidence 里。

### 3.8 输出格式

下面只示意结构，其中的数值都是虚构的，不是实际统计结果。

```json
{
  "schema_version": 1,
  "profiles": {
    "truebones/zoo/Alligator": {
      "skeleton_hash": "sha1(parents + offsets)",
      "source": {"clips": 31, "action_families": {"locomotion|walk": 6, "...": 0}},
      "global": {"leg_length": 0.226, "has_legs": true, "hip_height": 0.99, "axial_length": 6.3},
      "contacts": {"cond": [10, 13], "override_add": [18, 21], "override_remove": [], "used": [10, 13, 18, 21]},
      "joints": [
        {"index": 9, "name": "R_hiza", "role": "support", "dof_class": "hinge",
         "principal_axes": [[...], [...], [...]], "hinge_flex_sign": 1,
         "variance_ratio": [0.91, 0.07, 0.02], "angle_std_deg": 18.8,
         "ang_speed": {"q50": 1.1, "q95": 6.3}, "confidence": 0.92}
      ],
      "gait": {"walk, forward": {"clips": 3, "period": 32, "duty": 0.68, "v_g": 1.3,
               "phase": {"10": 0.0, "13": 0.5, "18": 0.25, "21": 0.75}, "stride": 1.4}},
      "stride_speed_fit": {"a": 0.9, "b": 0.5, "n_clips": 6, "fitted": true},
      "vertical": {"jump": {"clips": 2, "net": [-0.01, 0.0, 0.02], "peak": [0.4, 0.5, 0.6]}}
    }
  },
  "findings": {"truebones/zoo/Alligator": {"...": "报告的原始数据"}}
}
```

次级运动候选关节另有 `spring`、`spring_fit` 和 `passive_confirmed`；拆半不稳定的关节另有 `unstable`。

`python -m motion_edit.build_profiles [--dataset <namespace>] [--species <glob>] [--workers N]` 读 `dataset/datasets.jsonl` 里的每个数据集，写 `<dataset_root>/skeleton_profiles.json` 和报告；带过滤条件时只替换本次构建的物种，其余行保留。

`skeleton_hash` 用来检测 Profile 过期：骨架改了（重新预处理、关节重命名或删除），哈希就会对不上，运行时拒绝使用并提示重建。这条规则和 orientation_quat 过期的教训是同一个道理。

### 3.9 没有动作数据的新骨架

`tools/process_new_skeleton.py` 处理过的新骨架没有动作统计，v1 采用回退策略：

- 拓扑、接触关节、对称性取自 cond（接触关节同样可以手动增删）；
- `dof_class`、`hinge_flex_sign`：按规范关节名（`canonical_joint_names`），从 `species_tags` 相同的已知物种中取多数的 dof_class；匹配不到时用 ball。`confidence` 一律为 0；
- 次级运动候选照常按叶链列出，参数用默认值；用户确认后同样可以启用。

以后用预测模型替换这个回退策略，但要等 M7 的结论。

## 4. 阶段 B：分解器（生成结果 → Edit Package）

### 4.1 步骤

1. **解码并刚体化**：调用 `restore_animation_from_features(restore_space="hml", fullbody_ik=k, stretch_factor=s)`，`k` 默认开。生成结果的 pos 通道和旋转 FK 不完全一致时，普通解码会给关节解出逐帧的局部平移（骨长伸缩）；fullbody IK 把这部分位置信息改由旋转来表达，骨长被限制在静息长度的 `[1 − s, 1 + s]` 之内。IK 只重建肢体：躯干（cond 接触关节所在肢体的肢体根——髋、锁骨这类同侧链最上端——的所有祖先，例如根、骨盆、脊柱到颈）保留解码的旋转和逐帧局部平移，躯干关节的直接子关节（肢体根、尾根、头）保留解码的挂接位置。生成结果常在肢体分叉处弯背，刚体躯干表达不了，误差会让整条肢体平移，脚悬空或陷地；mesh 形变问题主要出在远端肢体，躯干保留解码不明显。没有接触关节或侧别标签的骨架只保留根。
   关掉 IK（`--no_fullbody_ik` 或 UI 开关）时保留逐帧局部平移，`s` 不起作用。运行时、锁脚 IK 和导出都按逐帧骨长工作，可以照常编辑；但 `amp.*` 只缩放旋转偏差，存在平移里的那部分姿态（骨骼方向偏离静息方向的部分）不受幅度参数影响。骨骼本来接近刚性的 clip 两种方式差别很小，pos 与旋转偏差大的骨架（Biped 前腿、挂件骨）用 IK 更合适。
   - `s` 是分解参数，默认取 `utils/fullbody_ik.py` 的 `DEFAULT_IK_STRETCH_FACTOR`，与 `sample/export.py`、`tools/restore_glb_from_npy.py` 的默认导出一致。用户在加载或导入动作时可以改：调到 0 是严格刚体，调大则更贴近 pos 通道；
   - 得到局部旋转 `R[F, J]`、逐帧局部平移 `T[F, J]`（骨长在静息长度的 `[1 − s, 1 + s]` 之内，`s = 0` 时就是静息 offsets）、全局位置 `P[F, J]`、根轨迹；
   - IK 残差（`ik_error` 的均值和最大值）写进诊断；
   - package 同时保存原始特征和解码所需的 cond 子集，这样用户之后换一个 `s` 时可以在服务端重新分解，不需要回到数据集目录。
2. **接触区间检测**：对接触关节集合（cond 的 `contact_joints` 加上 `contact_overrides.json` 的增删，或用户在 UI 上对这个 package 的增删）中的每个关节，在 clip 内按下面的规则判定接触帧。**所有动作都做这一步**。
   1. 预处理：位置先做 3 帧中值滤波。水平速度在 σ = 1 帧的高斯平滑后求差分；竖直速度取向前、向后两个单帧差分中绝对值较小的那个，因为快跑的支撑期可能只有 2 帧，每一帧都紧挨着落地或离地帧，居中差分或平滑后的差分永远读不到静止。loop clip 按周期边界处理（第 F−1 帧之后接第 0 帧）。
   2. 地面高度：世界坐标的 `ground_height`，默认 0（预处理后的 clip 站在 y = 0 上），存在 manifest 里，UI 上可以改。每个关节取自己高度的 q05 作为该关节的地面（踝着地时本来就比趾高）；关节的地面比 `ground_height` 高出 0.3 × 腿长以上的，整段都不算接触，这样悬停、飞行、游泳 clip 里垂着或收起的脚不会被判成接触；
   3. 候选帧：关节高度 < 该关节地面 + 0.05 × 腿长，且竖直速度 < 0.5 × 腿长 / 秒；
   4. 估计每帧的隐含地速 `v_g(t)`：取满足严格阈值的候选关节水平速度的中位数（取反），再沿时间做 σ = 1 帧的平滑。只用严格阈值，是因为贴地前摆的脚能通过宽阈值，会把中位数拉偏；平滑窗口不能更大，因为跳跃步态的支撑脚速度在一个短支撑期内会变化一倍以上；
   5. 确认接触：候选关节的水平速度与 `−v_g(t)` 的差 < max(0.3 × 腿长 / 秒, 0.3 × |v_g|)。跑动时支撑脚的后移速度在支撑期内本身就有起伏，所以容差随地速放大。原地 idle、攻击这类动作的 `v_g ≈ 0`，这时条件自然退化成"速度接近 0"；
   6. 去抖：上面的阈值放宽到 1.5 倍作为退出阈值（滞回），宽阈值下的连续段只要包含一帧满足严格阈值、且至少 2 帧，就是一个接触区间。

   规则实现在 `motion_edit/contacts.py`，Profile 统计和分解器共用。上面的阈值是在 Truebones（Horse 走 / 跑、Alligator）和 UnityBundles（Caveman）的 locomotion 上对过的：Horse 慢走的占空比约 0.7、跑步约 0.3。

   检测出的区间只是初值：用户可以在 UI 时间轴上增加、删除、拖动区间（第 5.4 节）。区间被手动改过时，`v_g` 改用手动区间里着地关节的水平速度中位数估计（同样做中值滤波和 σ = 1 帧的平滑），落点、残差、事件按新区间重新计算，不重新解码。飞行、游泳这类根本没有地面接触的 clip，如果仍被误判出区间，也由用户在 UI 上清掉。

   同时在地面坐标系中测量每个接触区间内落点的漂移量，也就是生成结果自带的滑步量，作为基线写进诊断（T6 使用）。
3. **识别事件**：
   - **loop locomotion**：由接触时序得到周期 T 和各接触关节的着地帧、离地帧，并检查与 Profile 中 gait 相位的偏差，偏差写进诊断。生成的 loop 窗口是周期性的：第 F−1 帧的下一帧就是第 0 帧，末尾没有重复的收尾帧，周期 T 按这个约定计算。周期只对 locomotion 计算，并且要求着地间隔重复（间隔对中位数的相对偏差的中位数 ≤ 0.2）；原地攻击、起身这类 loop 只是脚在原地重新站位，没有周期。
   - **根轨迹**：根的 Y 和 XZ 都拆成趋势和振荡两部分。趋势是低通分量（有步态周期的 locomotion loop 取一个周期，其余 loop 和 one-shot 取 0.5 秒；loop 的窗口不能取整段，环绕滤波下每帧的窗口都覆盖全部帧，趋势变成常数，sway / bounce 会缩放根的全部运动），振荡是剩下的部分。Y 的趋势就是净位移和跳跃弧线；XZ 的趋势在 locomotion 上接近 0，在攻击突进、击退、倒地等动作上就是真实位移。Y 带真实位移时记录起跳帧和落地帧。
   - **one-shot**（攻击、受击、跳跃、死亡等）：接触区间照常识别。攻击中常见的垫步、跨步、踩踏，就是中途离地、在新位置落地的接触区间，按普通落点处理。取能量（角速度平方和）最高的那条链作为**主动链**，其末端速度峰值帧为 `impact`；`impact` 之前，末端沿击打方向位移的局部极小值为 `windup`；`impact` 之后能量回落到峰值 15% 的帧为 `recover`。每个事件都附带置信度。
   - 事件只是**可编辑的标记**：微调 UI 允许用户拖动事件帧，拖动后重新计算依赖它的层。
4. **分层**：
   - `base`：每帧局部旋转和局部平移，作为兜底的细节层；
   - `root`：XZ 趋势、XZ 振荡、Y 趋势、Y 振荡、yaw 曲线；
   - `chains`：按肢体链分组，每个关节相对**链参考姿态**的 rotvec 偏差曲线。参考姿态取全段均值（loop 和 one-shot 相同），偏差最小，摆动围绕中心缩放。rotvec 沿时间 unwrap（每帧取与上一帧最近的等价分支，角度可以超过 π），所以任何增益下偏差曲线都连续。loop 中 unwrap 后首尾相差一整圈的关节（持续自转）缩放会破坏接缝，整段增益固定为 1，并写进诊断；
   - `plants`：每个接触区间的 `[start, end]`、`v_g` 曲线、接触点在"地面坐标系"（位置加上 `v_g` 的累积位移）中的落点（水平取区间均值，高度取区间中位数：均值会被落地、抬脚的边缘帧抬高，最低帧会落在长支撑期刚体拟合缓慢起伏的波谷上），以及区间内脚相对落点的残差曲线（原结果自带的滑步）。落点同时保存为相对于根的偏移；
   - `events`：上述事件帧和周期。

### 4.2 Package 格式

```text
<clip>.edit/
  manifest.json   # runtime_version、profile_schema、skeleton_hash、object_type、clip、fps、frame_count、
                  # is_loop、action_group、action_label、translation_root_index、forward(_base)_joint_index、
                  # stretch_factor、fullbody_ik、leg_length、dataset_root（来自数据集时有，写物种覆盖用）、
                  # input_notes（关于输入的诊断，如物种覆盖行过期；重新分解和接触编辑时保留，改写物种覆盖时清掉）、
                  # profile（status: ok / fallback / missing、该动作的 gait 行、stride_speed_fit）、
                  # contacts（使用的关节、来源 cond / species_add / species_remove / package_add / package_remove、
                  #   接触区间是否被手动改过）、events（周期、各肢体着地帧、离地段、与 Profile gait 的相位偏差）、
                  # facts（is_loop、locomotion、has_plants、turning、airborne、has_passive、chain_groups：
                  #   决定哪些参数可用）、
                  # params（每个参数的默认值、范围、分组、对该动作是否可用、分解时的运行时是否已实现；
                  #   UI 服务端按当前运行时改写"已实现"）、
                  # diagnostics（IK 残差、滑步基线、增益固定的关节、逐条诊断）
  data.npz        # 骨架：parents, names, bvh_names, sides, orients, anim_offsets, skeleton_offsets, skeleton_rest_rotations
                  # base：base_rot (F,J,4), base_pos (F,J,3)            即解码 + IK 的结果本身
                  # root：root_trend, root_osc (F,3), root_yaw (F,), root_tilt (F,4)
                  # chains：chain_reference (J,4), chain_offsets (F,J,3)（unwrap 后），chain_gain_locked (J,), chain_group (J,)
                  # 接触：contact_joints (K,), contact_mask (F,K), ground_velocity (F,2)
                  # plants：plant_intervals (N,3) = (接触列, start, end)，plant_anchor / plant_root_offset (N,3)，
                  #   plant_residual (F,K,3), plant_id (F,K), plant_drift (N,)
                  # profile 子集：profile_role / dof / axes / flex_sign / spring / passive / confidence
                  # 重新分解用：source_features (F,J,12), source_cond（cond 子集的 JSON；运行时不读）
```

旋转按 `R = exp(chain_offsets) · chain_reference` 重组，根旋转按 `R_root = yaw(root_yaw) · root_tilt` 重组，根平移是 `root_trend + root_osc`。跨越 loop 接缝的接触区间 `end > F`，帧号对 F 取模。

Package 是自包含的：微调端不需要 cond.npy，也不需要数据集目录。`runtime_version` 不兼容时拒绝加载。

## 5. 阶段 C：微调运行时

### 5.1 v1 参数集

| 参数 | 默认值 | 范围 | 作用对象 | 保证 |
|---|---|---|---|---|
| `tempo` | 1.0 | 0.5–2.0 | 全局时间 | loop 动作保持闭合，每周期帧数取整（实际倍率写进诊断）；one-shot 按 `i × tempo` 采样，末尾不足一个采样间隔的部分舍去 |
| `stride` | 1.0 | 0.6–1.6 | locomotion：隐含地速 `v_g` 与脚的前后摆动距离（根 XZ 不动，仍是原地动作）；转弯动作不提供 | 支撑期各接触关节的速度都等于新的 `v_g` |
| `bounce` | 1.0 | 0–2 | 根 Y 的振荡分量 | 支撑脚锁定，不穿地 |
| `jump_height` | 1.0 | 0.5–1.8 | one-shot 的离地段：根 Y 相对起跳帧—落地帧连线的弧线，起跳和落地帧不动 | 落地帧接触关节回到地面 |
| `sway` | 1.0 | 0–2 | 根 XZ 的振荡分量；XZ 趋势（突进、击退等真实位移）不受影响 | 支撑脚锁定 |
| `amp.legs / arms / axial / tail / wings` | 1.0 | 0–2 | 各类链的偏差幅度 | 支撑脚锁定 |
| `posture` | 0 | −0.3–0.2（× 腿长） | 髋高偏移（蹲 / 伸） | 支撑脚锁定，腿部 IK 重解 |
| `force` | 1.0 | 0.5–2.0 | one-shot：组合下面三项 | 事件顺序不变；支撑脚锁定 |
| `windup_depth` | 1.0 | 0–2 | windup 段主动链的偏差 | 支撑脚锁定 |
| `strike_speed` | 1.0 | 0.5–2.0 | windup→impact 的时间压缩 | 时间单调 |
| `overshoot` | 1.0 | 0–2 | impact→recover 段的过冲幅度 | 支撑脚锁定 |
| `impact_shift` | 0 | ±0.3（× 动作时长） | impact 事件在时间上的位置 | 事件顺序不变 |
| `secondary.stiffness / damping` | 1.0 | 0.25–4 | passive 关节的弹簧参数倍率 | 只作用于已确认的 passive 关节 |
| `foot_lock` | 关 | 开 / 关 | 所有带接触区间的动作 | 开：接触区间内落点漂移为 0（够不到的帧除外）；关：保留原结果自带的滑步 |
| `soft_stretch` | 0.1 | 0–0.2 | 所有带接触区间的动作：IK 重解时腿链骨长允许的最大伸缩比例 | 只在落点逼得腿接近伸直或远比原姿态更弯时起作用；0 时骨长与 package 一致 |

"支撑脚锁定"对所有动作成立：只要某个参数改变了支撑链或根的姿态，第 5.2 节的第 4 步就会对支撑脚重解 IK，让落点跟着编辑走而不是跟着身体走。主动链本身就是腿的时候（踢、踩踏），该腿在主动段不是支撑脚，不受锁定，只有它落地之后的接触区间才锁定。

`force` 不是独立的运算，它只是同时改变 `windup_depth`、`strike_speed`、`overshoot` 的一个预设组合；美术也可以分别调整这三项。

没有单独的速度参数：locomotion 的地速倍率就是 `tempo × stride`，由用户自己组合这两项。

`foot_lock` 是唯一一个会改变"原结果中与编辑无关部分"的参数，所以它是开关而不是滑杆，打开时结果允许与原结果有明显差别。

`soft_stretch` 本身不产生编辑：它只规定第 5.2 节第 4 步重解 IK 时腿可以伸缩多少，没有别的参数触发 IK 时它不起作用，所以默认值不为 0 也不影响严格回放。可用性由 package 的 facts 现算，旧 package 不需要重新分解。

参数取了非默认值，但当前运行时版本还没实现它、或者它对该动作不可用时，运行时直接报错，不静默忽略。UI 只显示可用的参数，未实现的滑杆显示为禁用。

可用性由分解时记录的 `facts` 决定：

- `stride`：locomotion、有接触区间、且不是转弯动作（根 yaw 的变化范围 ≤ 25°）；
- `jump_height`：one-shot 且检测到离地段（两个着地段之间没有任何接触的连续帧）。loop 的跳跃弧线在根 Y 的振荡里，由 `bounce` 调；起身这类没有离地段的竖直位移不提供；
- `posture`、`foot_lock`：有接触区间；
- `force` 系列：one-shot；`secondary.*`：有已确认的 passive 关节。

### 5.2 运算顺序（固定）

```text
0. 所有参数为默认值的层直接跳过（这样才能保证严格回放）
1. 时间：每个输出帧在源时间 s 上采样（旋转 slerp，平移线性插值，整数 s 原样取关键帧）。
   时间倍率 = tempo，步幅倍率 S = stride。
   loop：每周期帧数 F' = round(F / 倍率)，s = i·F/F'，插值在首尾之间环绕（第 F−1 帧之后接第 0 帧）；
   one-shot：s = i × 倍率。
   M4 起改为以事件为节点的单调分段三次插值（PCHIP），t' = w(t)
2. 幅度：chain_offsets 按每段的增益缩放（amp.* × windup_depth / overshoot 的分段曲线，段与段之间平滑过渡）；
   采样后的偏差取与 chain_offsets 插值最近的分支再缩放；chain_gain_locked 的关节增益为 1
3. 根（在源帧上算好，再按第 1 步采样）：sway 缩放 XZ 振荡；bounce 缩放 Y 振荡；
   jump_height 缩放每个离地段内根 Y 相对"起跳前最后一个着地帧—落地帧"连线的偏离（两端固定）。
   缩放的是趋势加振荡的整条 Y：0.5 秒的低通窗口和一次起跳差不多长，只缩放趋势会漏掉大半条弧线；
   posture 平移 Y（× 腿长）；XZ 趋势保持不变
4. 锁脚 + IK（所有带接触区间的动作都执行）：
   - 触发条件：除 tempo 以外任何参数不是默认值，或者 foot_lock 打开。只改 tempo 时只重采样，不重解
   - 输出帧的接触：前后两个源帧都在同一个接触区间里才算着地（落地前半帧还是正在落下的脚）
   - 支撑期目标（每个着地的接触关节）：
       W = P(u) − (S − 1)·(D(U) − D(U_mid)) − foot_lock·残差(u)
     P(u) 是原结果在源时间 u 的位置，D 是 v_g 的累积位移。U 是沿输出时间轴展开的源时间
     （loop 每过一圈加一个周期的位移），U_mid 是这只脚（同一肢体的所有接触关节共用）这段支撑的中点。
     所以 S = 1 时目标就是原轨迹；S ≠ 1 时脚绕支撑中点多走 (S − 1) 倍的地速位移，支撑中点处相对身体的位置不变。
     变回地面坐标系（加上 S·D）后，落点是原地面坐标落点加一个常数，所以漂移不超过原结果。
     foot_lock 打开时减去原结果自带的滑步残差，落点锁死。残差按脚取一份：这只脚的参考关节（着地关节里最深的，
     即脚掌滚离地面时最后离地的脚趾）的残差，同一只脚的所有着地关节一起减。脚跟绕着地的脚趾抬起（heel peel）
     因此照原样保留，只去掉脚趾的滑步；如果每个关节各自锁死，抬起中的脚跟会被按回地面，蹬地末期腿够不到。
     每段支撑从第一帧参考关节的残差开始，之后逐帧累加"前后两帧都在同一接触区间里的最深着地关节"的残差增量，
     所以参考关节交接（脚趾落地、区间断开一帧）时目标不跳变
   - 肢体：接触关节按所在肢体分组（同侧关节一直往上到肢体链根）。肢体的"脚"是包含该肢体全部接触关节的最深关节，
     IK 链是肢体链根到脚的父关节；脚的子树按刚体移动。没有链可解的接触关节（中轴上的、链根就是脚的）写进诊断
   - 每帧每个肢体：着地关节里最深的是 pivot（就是 foot_lock 的参考关节），离 pivot 最远的着地关节是 distal。脚的目标位置让 pivot 精确落在目标上；
     脚的世界朝向保持编辑后姿态的朝向，再转一个让 pivot→distal 对准目标方向的最小旋转。
     其余着地关节随脚部刚体移动，不再单独约束
   - 摆动期：支撑帧上的修正量（脚的平移和对齐旋转）在相邻两段支撑之间线性插值（loop 环绕，one-shot 两端保持），
     叠加在编辑后的摆动轨迹上，抬脚高度和弧线形状照原样保留
   - 求解：阻尼最小二乘（DLS），从编辑后的姿态出发迭代，每步每关节最多 0.2 rad。
     关节自由度来自 Profile，但只是偏好不是硬约束：hinge 绕第一主轴自由转动，planar 绕前两个主轴，
     其余方向（fixed 关节则是全部方向）按 0.1（fixed 0.05）的权重开放。
     硬 hinge 会让一条转轴接近平行的 hinge 链完全无法横向移动脚（Horse 左后腿的三个关节都被判成 hinge）。
     置信度 < 0.3 或没有 Profile 的关节当作 ball。
     从当前姿态出发，膝、肘就弯向原来弯的那一侧；只有链已经伸直（伸展 > 链长的 0.995）又必须缩短时，
     先把 hinge 沿 Profile 的 flex_sign 方向预弯 0.1 rad。
     目标超出链长的 0.995 倍时，先沿链根→目标方向拉回到这个距离再解：DLS 瞄准伸直以外的点不会收敛，
     链会来回摆，停在哪一步就是哪个姿态，够不到的帧和身体下降量随之乱跳。拉回后腿稳定地伸直指向目标，
     误差仍按原目标计算，竖直缺口交给第 6 步。IK 只转动关节，骨长取 package 的逐帧局部平移，
     只有下面的软伸缩会改变腿链的骨长
   - 软伸缩（`soft_stretch` = s）：求解前把肢体链根以下的骨（链根之后的链关节和脚的局部平移）整体乘一个系数 k。
     伸展度 = 链根到目标的距离 / 链长。目标要求的伸展度超过 max(0.96, 原姿态自身的伸展度) 时拉长，
     低于原姿态伸展度的 0.9 倍时缩短；超出的比例 x 经 k = 1 + s·tanh(x / s) 缓入上限，所以 |k − 1| < s。
     拉长让腿不必伸直锁死也够得到；缩短让 posture 压低时膝盖不必弯到远超原动作的程度。
     k 只在支撑帧上计算，摆动段在相邻支撑之间线性插值；再把拉长量和缩短量分别取 0.2 s 升余弦斜坡的上包络，
     骨长逐帧平滑变化。包络让邻近帧的腿比必需的略长或略短，膝盖会把差吃掉，落点照样精确。
     拉到上限仍够不到的部分才交给第 6 步的身体下降
   - 够不到（pivot 误差 > 1e-4 × 腿长）的帧写进诊断
5. 次级运动：已确认的 passive 关节只叠加增量：
   θ_out = θ_orig + sim(新父运动, k', c') − sim(原父运动, k, c)
   其中 k、c、g_* 取 Profile 的 spring（拟合值或默认值），k'、c' 是乘上 secondary.* 倍率后的值。
   两次模拟用同一个积分器、同一组初值。同一条叶链从链根往链尖逐个关节模拟，
   下游关节的"新父运动"已经包含上游关节的模拟增量
   loop：预热两个周期，取第三个周期作为结果，保证首尾闭合
6. 着地：支撑目标本身就取原结果的高度，第 4 步把脚放回去，原结果自带的悬空或穿地保持不变。
   只有 IK 够不到、pivot 停在目标上方时（比如 posture 抬高、bounce 压平后腿不够长），
   才把这些帧的身体按最大的差降下来。下降量取差的上包络：每帧的差向前后各展开 0.2 s 的升余弦斜坡，
   身体缓降缓升，不会在单帧上下沉；离地段在支撑帧之间线性插值（跳跃弧线不被压平）。
   然后从编辑后的姿态（不是第一次 IK 的结果）重新解一次 IK：第一次 IK 在支撑帧上没够到的误差
   不能再被插值进摆动帧，否则支撑结束那一帧脚会跳。第二次沿用第一次的软伸缩系数：下降量是按它量出来的，
   身体降低后重新计算会得到更小的拉长，腿又差一点够不到
7. 输出 Animation；导出复用 npy_restore 的 exporter 路径（BVH / GLB）。
   可选输出 root motion：把 v_g' 积分到根 XZ，供需要根位移的引擎使用；默认仍输出原地动作，与训练数据一致
```

### 5.3 运行时用到的算子与复用关系

| 算子 | 来源 |
|---|---|
| 四元数 / rotvec 运算、FK | `motion_lib` 的 `Quaternions`、`Animation` |
| 重采样 | `runtime.py` 的 `sample_quat` / `sample_linear` / `sample_angle`：周期模式下第 F−1 帧之后接第 0 帧；整数时间原样返回关键帧，T1b 才能逐位一致。`npy_restore.resample_animation` 不动 |
| PCHIP 时间扭曲 | 新写，放在 `runtime.py` |
| 肢体 IK | `motion_edit/ik.py`：所有帧一起批量求解的 DLS（第 5.2 节第 4 步）。不放进 `utils/fullbody_ik.py`：那里是让每根骨对齐目标方向的全身求解，这里要的是"末端到点 + 软自由度 + 保持弯曲侧"，两者没有可共用的部分，放在一起只会让导出路径背上运行时的改动风险 |
| 二阶弹簧积分 | 新写，放在 `runtime.py` |

`profile/` 统计 hinge 主轴时用到的 log map 与运行时用同一套四元数实现，避免两边对 rotvec 的约定不一致。

### 5.4 微调 UI：本地 serve 页面

运行时之上是一个本地网页，用户在页面上调参数、拖事件标记、修正接触，并立即看到结果。它与 `dataset/review/serve.py` 采用同一种形态：标准库 `ThreadingHTTPServer` 加一个静态页面，不引入 web 框架。

**进程与职责**

```text
motion_edit/ui/serve.py          python -m motion_edit.ui.serve --packages <dir> [--port 8770]
  ├─ 加载 package，调用 EditRuntime.apply（运行时只在服务端跑，页面不重写任何编辑逻辑）
  ├─ 加载时可选 stretch_factor / fullbody_ik：与 package 当前值不同时，用 package 内的原始特征重新分解
  ├─ 接触修正：改接触关节集合时重新检测区间；改区间时重新计算落点和残差
  ├─ 把结果以逐帧全局位置 + 局部旋转的形式返回给页面
  └─ 导出 BVH / GLB（复用 npy_restore 的 exporter 路径）
motion_edit/ui/index.html        单页：3D 视图 + 参数面板 + 时间轴 + 诊断面板
motion_edit/ui/vendor/           three.js 等前端依赖放在本地，离线可用
```

所有编辑都只在服务端的 `EditRuntime` 里算，页面只负责显示。这样页面看到的结果和导出文件、命令行 `motion_edit.apply_edit` 的结果是同一份，不会出现"预览和导出不一致"。

**HTTP 接口**

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/packages` | 列出 `--packages` 目录下的 package（clip 名、物种、is_loop、stretch_factor、诊断摘要） |
| GET | `/api/package/<id>` | manifest：参数默认值与范围、事件、接触关节集合及来源、接触区间、骨架（parents / offsets / names）、哪些参数对该动作可用（按服务端当前的参数集重建） |
| POST | `/api/load` | 输入：package id + stretch_factor（可选）+ fullbody_ik（可选）；任一项与当前值不同时重新分解并覆盖 package，返回新的 manifest。重新分解会重置手动改过的接触区间，页面在执行前要求确认 |
| POST | `/api/contacts` | 输入：package id + `joints`（完整的接触关节集合）+ `species`（是否应用到该物种），或 package id + `mask`（(K, F) 的 0/1 接触区间），或 package id + `ground_height`（地面高度，重新检测区间）。关节集合变了就重新检测区间，区间变了就重新计算 v_g、落点和残差，都不重新解码，结果覆盖 package。`species` 时把相对 cond 的增删（关节名）写进 `<dataset_root>/contact_overrides.json`（增删都为空时删掉该行），package 的增删随之转成物种来源；package 没有 `dataset_root` 时报错 |
| POST | `/api/apply` | 输入：package id + 参数 + 事件帧（可选，用户拖动过时才带）。UI 一律走完整路径（第 0 步的默认值跳过关闭，即 T1b），默认值下的输出也是分解再重组的结果；输出：逐帧全局位置、局部旋转、`v_g'`、每个输出帧采样的源时间、输出时间轴上的着地掩码、支撑目标、够不到的掩码、诊断（夹紧、够不到、身体下降、增益固定、tempo 取整）、与原始结果的最大位置差（UI 放在诊断列表里） |
| POST | `/api/export` | 输入同 `/api/apply`，外加格式（bvh / glb）、是否输出 root motion；写入文件并返回下载链接 |
| GET / POST | `/api/presets/<id>` | 读写该 package 的参数预设（JSON，与 package 放在一起） |

v1 不考虑性能：参数滑杆在松开时才请求 `/api/apply`，拖动过程中不发请求；新请求到达时丢弃尚未返回的旧请求的结果。

**页面布局**

1. **3D 视图**（three.js）：
   - 骨架用线段加关节球显示；接触关节在接触区间内高亮；
   - 视口里左键点选关节，把它加入或移出接触关节草稿（中键环绕不受影响）；侧栏列出草稿和每个关节的来源（cond / 物种 / 本 package），也可以从下拉框添加。确认后"应用到本 package"或"应用到该物种"（走 `/api/contacts`）；
   - 编辑结果每帧的支撑目标画成黄圈，够不到的标红；原始骨架按编辑结果当前帧采样的源时间对齐，tempo 改了也能逐帧对比；
   - 地面网格放在 `ground_height` 上，不随动画或相机移动；侧栏改地面高度时网格立即跟着移动，点"应用"后才重新检测接触区间；可以切换成"root motion 视图"，把 `v_g'` 积分到根上，让角色真正前进；
   - **对比**：同时显示原始结果（半透明）和编辑后的结果；也可以切换成左右并排；
   - 播放、暂停、逐帧、播放速度、循环开关；相机可以环绕和跟随。
2. **参数面板**：按第 5.1 节分组（时间 / locomotion / 根 / 幅度 / 力度 / 次级运动 / 着地）。对当前动作不可用的参数直接隐藏（比如 attack 不显示 stride，没有 passive 关节就不显示 secondary）。每个滑杆都有"恢复默认"按钮；面板顶部有"全部恢复默认"，恢复后结果必须与原始结果完全一致（T1 在 UI 上的体现）。顶栏可以设置 stretch_factor 和 fullbody IK 开关（关掉 IK 时 stretch_factor 禁用），点"重新分解"生效。
3. **时间轴**：按源帧显示，播放头在编辑结果当前帧采样的源时间上。显示事件标记（windup / impact / recover、起跳 / 落地、loop 周期边界），以及每个接触关节一行的接触区间色条（左侧是关节名）。事件标记可以拖动（M4）。接触区间在草稿上编辑：空白处拖动新增，拖动两端调整（loop 可以跨接缝），点选后按 Delete 删除；新增的格子绿色、删掉的格子淡红；"应用区间修改"才发给服务端，"放弃"恢复。点标尺行或关节名列只移动播放头。
4. **诊断面板**：列出本次 apply 的夹紧、够不到、增益固定、事件低置信度、分解时的 IK 残差等信息，点一条就跳到对应的帧并高亮对应的关节。
5. **导出栏**：格式、是否输出 root motion、文件名；以及预设的保存和加载。

**v1 不做的事**

- 不显示蒙皮网格，只显示骨架。有网格的角色导出 GLB 后，用外部工具查看；
- 不在页面里调用 AnyTop 生成新动作。package 由服务端生成后放进 `--packages` 目录；
- 不做多用户：单机本地使用，参数预设和接触覆盖只写在本地。

## 6. 验收

现有的评分机制不可靠，**效果验收全部由人工判断**。自动测试只负责机械不变量：这些项目只能判断"结果错没错"，判断不了"效果好不好"。

### 6.1 自动测试（机械不变量，必须通过）

| 编号 | 项目 | 判据 |
|---|---|---|
| T1 | 严格回放 | 默认参数下，与 `restore_animation_from_features(restore_space="hml", fullbody_ik=k, stretch_factor=s)` 的输出相比，局部旋转最大误差 < 1e-5 rad，位置误差 < 1e-6 |
| T1b | 分解往返 | 关闭第 0 步的跳过，强制走"分解 → 所有增益为 1 → 重组"的完整路径，与 T1 相同的阈值。T1 在默认值下跳过了所有层，测不到分解器，这一条才能 |
| T2 | 确定性 | 同一 package、同一组参数，两次运行的输出逐位一致 |
| T4 | loop 闭合 | 所有参数组合下，环绕步（第 F'−1 帧 → 第 0 帧）不超过内部最大的一步。运行时的每一层都按周期构造，接缝只是普通的一步；和原结果比"环绕步 / 步长中位数"没有意义，编辑会改变步长分布 |
| T5 | 骨长 | 每帧每根骨的长度与 package 的 `base_pos` 一致（`stretch_factor = 0` 时就是静息长度），误差 < 1e-6；编辑不改变骨长，只有 IK 肢体链根以下的骨按 `soft_stretch` 伸缩，比例不超过 1 ± soft_stretch；`soft_stretch = 0` 时所有骨误差 < 1e-6 |
| T6 | 锁脚 | 所有动作（包括攻击、受击等非 locomotion 动作）、所有参数组合下，地面坐标系（位置加 S·D）中每个接触区间内的落点漂移，不超过原结果在同一组源时间上的漂移：pivot 多出 < 1e-4 × 腿长，其余着地关节多出 < 0.015 × 腿长（它们随脚部刚体移动；amp.legs = 0 压平了脚趾自己的滚动，会多出一点）。`foot_lock` 打开时 pivot 在它连续当 pivot 的每一段帧上漂移 < 1e-4 × 腿长（脚趾抬起后脚跟重新当 pivot 时，中间绕脚趾滚过，分段算），并且每个着地关节相对参考关节的偏移等于原结果的偏移。够不到的帧不计；基线必须在同一组帧上算，排除帧后子集的均值会变，和整段的基线比不成立 |
| T7 | 连续性 | 对每个滑杆参数，取默认值 ± ε（ε = 范围的 1e-3），输出与默认输出的位置差（/ 腿长，比较共同的帧）不超过 20·ε。用来防止某一层在偏离默认值时一下子全量生效。one-shot 的 tempo 常数最大（约 7）：采样时间的偏移随帧号累积 |

T6 不是"效果好"的判据，只是防止运行时把滑步弄得比原来更糟。

### 6.2 人工审查（效果验收）

微调 UI 做好后，由人在 UI 上挑几条动作试用，不做批量渲染和逐项打分。挑选时尽量覆盖 loop locomotion、one-shot attack（带突进位移的）、Y 带真实位移的动作（跳跃或起身）。也可以混入一条训练集的原始 clip（不经过模型生成），用来区分是分解器和运行时的问题，还是生成结果本身的问题。

对每个参数，把滑杆从最小拖到最大，结合半透明的原始结果对比和支撑目标标记来看：

1. **名实相符**：参数的效果和名字说的是同一件事（stride 调大就是步子变大，而不是腿抬得更高）；
2. **单调**：效果随参数连续、单调变化，中间没有跳变；
3. **无瑕疵**：没有比原结果更多的滑步、膝肘反弯、穿地、帧间跳变（pop）、loop 接缝；
4. **极值可用**：最小和最大值不要求自然，但不能崩坏。崩坏就收窄该参数的范围。

发现的问题按"修复、收窄范围、作为已知限制"之一处理。

## 7. 里程碑

| 里程碑 | 内容 | 完成标准 |
|---|---|---|
| M1 | Profile 提取器 + 报告 | 所有物种的 Profile 生成完毕；第 3.7 节的检查全部跑完；人工看过报告（要做次级运动的关节可以随时写进 `passive_confirmations.json`，不阻塞后续里程碑） |
| M2 | 分解器（含 fullbody IK 刚体化、stretch_factor、接触区间检测）+ 严格回放 + UI 基础（server、3D 视图、对比、参数滑杆、加载时设置 stretch_factor） | T1、T1b、T2、T5 通过；UI 能加载 package 并做往返回放。之后 M3–M5 的人工审查都在这个 UI 上进行 |
| M3 | 运行时：tempo / stride / bounce / sway / jump_height / amp / posture / foot_lock + 锁脚 IK；UI 上的接触修正（标记 / 取消接触关节，增删、拖动接触区间，应用到该物种） | T4–T7 通过；在原地 locomotion、Y 位移动作、带突进的攻击和 idle 上人工审查通过 |
| M4 | one-shot 事件 + force 系列参数 | 人工抽查 attack clip 的事件帧，正确率 ≥ 80%；attack / hurt 上 T6、T7 通过；force 系列参数人工审查通过（试用的动作中要有带垫步、跨步的攻击 clip） |
| M5 | 次级运动弹簧（增量模拟，拟合参数或默认参数） | 已确认的 passive 关节启用；T4、T7 通过；人工审查通过（审查的动作中要有用默认参数的手 K 尾巴） |
| M6 | 微调 UI 完整版（第 5.4 节）：事件拖动、诊断面板、导出、预设、root motion 视图 | UI 上"全部恢复默认"与原始结果完全一致；UI 导出的文件与命令行用同一组参数导出的文件逐位一致；美术试用，收集参数是否够用、是否直观的反馈 |
| M7 | Profile 用于生成后处理（hinge 投影），不改模型 | 对同一批生成结果做处理前后的盲审 A/B（左右位置随机），记录偏好；结论决定是否立项"预测器"和"Profile 作为 AnyTop 条件" |

M1 与 M2 可以并行；M3 依赖两者；M4、M5 可以并行；M6 可以在 M3 之后的任意时间开始，随 M4、M5 增加参数分组。

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| 生成结果抖动，接触区间和事件检测不稳 | 中值滤波加最短持续时间；接触区间和事件都允许用户在 UI 上手动修正 |
| cond 的 `contact_joints` 有漏标（如 Alligator 前脚） | 用户在 UI 上手动标记，可以只对一个 package 生效，也可以写进 `contact_overrides.json` 对整个物种生效；cond 修正走正常的 regen + 重训流程，单独决策 |
| 飞行、游泳 clip 被误判出接触区间 | 关节自己的地面离 `ground_height` 太高时整段不算接触；生成结果整体浮起或下沉时在 UI 上调地面高度；剩下的误判由用户在时间轴上清掉 |
| fullbody IK 刚体化后偏离 pos 通道 | 分解时记录 IK 残差并显示在诊断面板；用户可以在加载时调大 stretch_factor |
| 次级运动参数不合适 | 拟合只在驱动项有显著贡献时采用，其余用默认参数；是否启用由用户确认；`secondary.*` 滑杆可整体调节 |
| 链偏差越过 π，缩放后翻转 | 偏差沿时间 unwrap，曲线连续；loop 中每周期自转一整圈的关节增益固定为 1，并写入诊断 |
| 原地 locomotion 的接触判定依赖隐含地速估计，转弯、侧移时各脚速度不一致 | 先只支持直行：根 yaw 变化超过 25° 的 locomotion 不提供 stride（写进诊断）。以后 `v_g` 改为每帧的二维向量加 yaw 角速度（刚体平面运动），按每只脚的位置分别计算其应有速度 |
| 多个接触关节同时着地时，刚体脚部只能精确放下 pivot | distal 方向对齐，其余关节随刚体移动；T6 对它们给出单独的容差。脚趾在支撑期自己的滚动被 amp 压平时最明显 |
| 编辑把落点推到腿够不到的地方（posture 抬高、stride 很大） | 先按 `soft_stretch` 拉长腿；拉到上限仍够不到时 IK 伸直到最近处并标红，pivot 停在目标上方时身体下降（第 5.2 节第 6 步），所以 posture 往上调可能被"下降"部分抵消，以诊断为准。腿的伸缩在蒙皮上可见，5–8% 基本看不出，接近 20% 时腿明显变细变长 |
| 无腿、飞行、游泳类骨架没有落点 | 自动禁用 stride 的落点重算，只做根轨迹缩放；`bounce` 改为作用在主轴振荡上 |
| 时间扭曲导致 loop 周期不是整数帧 | 第 1 步强制整数帧周期重采样；T4 兜底 |
| hml 空间与 native 空间的换算 | 运行时全程在 hml 空间工作，长度用腿长归一化；导出时沿用 npy_restore 的 `restore_space` 逻辑 |
| Profile 与骨架不同步 | `skeleton_hash` 校验（Profile 和 `contact_overrides.json` 都校验）；把 `motion_edit.build_profiles` 作为 `regenerate_dataset_artifacts` 的可选步骤 |

## 9. 后续（本方案之外）

- **M7 有收益时**：做 AnyTop 的 oracle 实验，用真实 Profile 作为条件、在留出物种上评估。只有 oracle 有提升，才训练"骨架 → Profile"预测器，并用**预测值**加噪声去训练 AnyTop。
- **编辑后修顺**：参数超出运行时能处理的范围时，把编辑结果作为约束送回服务端，做低噪声的 inpainting 修顺（使用已有的 joint / temporal mask 机制），固定种子。
