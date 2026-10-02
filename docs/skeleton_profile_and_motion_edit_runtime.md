# 骨架统计档案 + 动作微调运行时：执行方案

> 状态：方案（未实现）。
> 范围：从训练动作数据中统计每个骨架的运动学属性（Skeleton Profile），把 AnyTop 的输出分解为可编辑的中间层（Edit Package），并在微调端用一个固定、确定性的运行时（Edit Runtime）合成最终动作，提供即时、可预期的参数化微调。
> 不在范围内：训练"骨架 → 属性"预测模型；把属性作为 AnyTop 的条件输入。两者都要等本方案的 M7（约束后处理评估）给出结论后再立项。

## 1. 目标与原则

1. **模型只负责生成，微调不再依赖重新抽卡**：AnyTop 输出 `(F, J, 12)` 之后，用户所有的调整都在微调端（微调 UI 或 `apply_edit` 命令行）完成，不重新调用模型，参数相同则结果逐位一致。
2. **代码固定，数据特化**：只有一套带版本号的通用运行时；每个骨架、每个动作的差异全部放在 Profile 和 Edit Package 这两份数据里，不生成任何特化代码。
3. **不碰模型与 cond**：Profile 单独存成文件，不写进 `cond.npy`，所以不需要 regen 也不需要重训，cond 的 key 集合也保持稳定（compile 依赖它）。
4. **默认参数严格回放**：所有参数为默认值时，运行时的输出必须与 `restore_animation_from_features(restore_space="hml", fullbody_ik=True, stretch_factor=s)` 的结果一致，其中 `s` 是该 package 分解时使用的 `stretch_factor`（第 4.1 节）。在这个前提之上才谈编辑。
5. **编辑只作用在增量上**：每一层都只叠加"编辑引起的变化"，不重写原结果中与编辑无关的部分。锁脚保留原结果自带的滑步残差，ROM 夹紧只压超出原结果自身幅度的部分，次级运动只叠加新旧父运动的模拟结果之差，着地只补偿编辑引起的高度变化。这样参数从默认值出发连续变化，滑杆在默认值附近不会突变。"修掉原结果的滑步"是一个显式开关（`foot_lock`），不附带在其他参数里。
6. **超出范围就报告，不静默出错**：例如腿伸不到、关节超出 ROM、找不到事件帧，都要作为诊断信息返回，并把对应参数夹住。
7. **验收以人工判断为主**：现有评分机制不可靠，不作为验收依据。自动测试只检查机械不变量（回放、确定性、连续性、ROM、闭合、骨长、锁脚），效果好坏由人看渲染结果来判断（第 6 节）。
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
  profile/                  统计提取器（依赖 numpy、motion_lib、npy_restore）
  decompose.py              生成结果 → Edit Package；接触区间检测
  runtime.py                EditRuntime：package + 参数 → Animation（第 5 节）
  ui/                       微调 UI：serve.py + index.html + vendor/（第 5.4 节）
  build_profiles.py         命令行：批量构建 Profile 并输出审查报告
  decompose_clip.py         命令行：npy + cond → Edit Package（--stretch_factor）
  apply_edit.py             命令行：package + 参数 JSON → BVH / GLB
tests/test_motion_edit_profile.py、tests/test_motion_edit_runtime.py
```

运行时复用仓库已有的实现：四元数和 FK 用 `motion_lib`（`Quaternions`、`Animation`），重采样用 `npy_restore.resample_animation`，IK 在 `utils/fullbody_ik.py` 的基础上扩展（第 5.3 节），导出走 `npy_restore` 的 exporter 路径。运行时不导入 torch，也不读 cond 或数据集目录，所需的骨架数据全部来自 package。v1 不考虑运行时的性能。

## 3. 阶段 A：Skeleton Profile 统计提取器

### 3.1 输入

- cond entry：`parents`、`offsets`、`kinematic_chains`、`contact_joints`、`symmetry_partner_indices`、`joint_side_labels`、`species_tags`、`translation_root_index`、`scale_factor`、`forward_joint_index`、`forward_base_joint_index`。
- `contact_overrides.json`（可选，第 3.4 节）：用户对接触关节集合的手动增删。
- 该物种所有 clip 的 `motions/*.npy`，形状 `(F, J, 12)`，通道为 pos 3 + rot6D 6 + vel 3。
- `action_labels.jsonl`：`action_group`、`action_label`、`is_loop`，用于按动作族分组统计。

解码统一走 `build_skeleton_only_context` 加 `restore_animation_from_features(restore_space="hml")`，得到局部四元数和全局位置，不另写一条解码路径。训练数据本身就是刚体骨架导出的，统计时不需要开 `fullbody_ik`。所有长度类统计量都除以**腿长**（接触关节到髋部所在链根的静息链长之和；没有腿的骨架改用 `axial_avg_len`）做归一化，这样不同物种之间可以比较。

### 3.2 统计权重

一个 clip 内的所有帧合计权重为 1（clip 内按帧均分），然后再按动作族（`action_group` + 主词）做均衡，避免一个长 idle clip 主导统计结果。每个属性都要记录**覆盖度**：参与统计的 clip 数，以及来自哪些动作族。

### 3.3 每个关节的属性

| 属性 | 算法 | 运行时用途 |
|---|---|---|
| `dof_class` ∈ {fixed, hinge, planar, ball} | 局部旋转相对于该关节的鲁棒均值姿态取 log map，得到 rotvec，再做加权 PCA。总角度标准差低于阈值判为 fixed；第一主成分方差占比 ≥ 0.85 判为 hinge；前两个主成分合计 ≥ 0.9 判为 planar；其余判为 ball | IK 约束、幅度缩放的轴 |
| `principal_axes` (3×3, 父关节坐标系) | PCA 的特征向量 | 同上 |
| `hinge_flex_sign` | 仅对 hinge：沿主轴往哪个方向转会**缩短**祖父关节到子关节的距离，就把那个方向定为屈曲 | IK 的 pole 方向，防止膝盖反弯 |
| `rom` | 每个主轴上投影角度的 q01 / q99；加上子骨方向相对均值方向的摆动锥角 q99 | 软夹紧 |
| `ang_speed` | 局部角速度的 q50 / q95 | 力度参数的上限参考、诊断 |
| `role` ∈ {root, axial, support, swing, passive, other} | 见第 3.4 节 | 决定哪些编辑作用在哪个关节上 |
| `spring` {k, c, g_ang, g_lin, r2, delta_r2} | 仅对 passive 候选：见第 3.5 节 | 次级运动模拟 |
| `confidence` | 由覆盖度与稳定性（第 3.7 节）综合得出 | 低置信度时回退到默认值 |

### 3.4 接触关节与关节角色

- **接触关节集合不做自动检测**：默认取 cond 的 `contact_joints`，再应用 `contact_overrides.json` 中用户的手动增删。例如 `truebones/zoo/Alligator` 的 cond 里，`contact_joints` 只有后脚 `R_ashi`/`L_ashi`，前脚 `te` 既不在 `contact_joints` 里也不在 `end_effector_joints` 里；用户在微调 UI 里发现前脚滑步时，手动把前脚标成接触关节即可（第 5.4 节）。
- `contact_overrides.json` 放在 `<dataset_root>/` 下，按物种名记录 `{"add": [...], "remove": [...], "skeleton_hash": ...}`，用关节名而不是索引。`skeleton_hash` 对不上时整条覆盖失效并在报告中列出。cond 本身不改；如果要把覆盖写回 cond，走正常的 regen + 重训流程，单独决策。
- `support`：接触关节，以及从它往上直到肢体链根的所有关节。
- `axial`：`kinematic_chains` 中包含根的那条链，以及头尾方向的主链。
- `swing`：不接触地面的肢体链（手臂、翅膀等）。
- `passive`：由第 3.5 节的弹簧拟合判定；只在叶链上判定（尾、耳、毛发、触须、翼膜），不在 support 链上判定。

gait 统计（第 3.6 节）需要训练 clip 上的逐帧接触区间，按第 4.1 节的区间检测规则在上述接触关节上计算。

### 3.5 被动关节的弹簧拟合

对叶链上的关节 j（父关节 p）：

```text
θ_j(t)  = 局部 rotvec 相对均值的偏差
α_p(t)  = 父关节在世界坐标中的角加速度，转换到 p 的局部坐标系
a_p(t)  = 父关节在世界坐标中的线加速度（含重力方向），转换到 p 的局部坐标系
模型：  θ̈ + c·θ̇ + k·θ = −g_ang·α_p − g_lin·(a_p 在垂直于骨向的分量)     （每个轴独立）
```

对 k、c、g_ang、g_lin 做线性最小二乘（数值微分前先做 Savitzky–Golay 平滑），留出 20% 的帧计算 R²。

只看 R² 判别不了被动关节：`θ̈ + c·θ̇ + k·θ = 0` 能完美拟合任何正弦，一条手 K 的周期摆尾即使和父运动无关，R² 也会很高。所以判据是三条同时满足：

1. 留出帧上 R² ≥ 0.6；
2. 驱动项的贡献：完整模型的留出 R² 比去掉驱动项（g_ang = g_lin = 0）的模型高出 `delta_r2` ≥ 0.2；
3. g_ang、g_lin 至少有一个显著不为零（系数 / 标准误 > 3），且 k 在合理区间内。

满足时标记为 `passive`，并写入 `spring`。这组参数在运行时驱动次级运动模拟，也就是第 5.2 节的第 6 步。

这一步有误判风险，所以它的结果只作为**候选**，在报告中单独列出供人工确认。未确认的 passive 关节在运行时默认不启用模拟。

### 3.6 全局属性与 locomotion 属性

- `leg_length`、`hip_height`（静息姿态）、`axial_length`。
- `gait`：对每个 `is_loop` 的 locomotion clip，得到每个接触关节的着地相位、各接触关节之间的相对相位、占空比（duty factor）、周期 T（帧）、隐含地速 `v_g`（支撑期均值）、步幅（`v_g` × T / 腿长）。按 `action_label` 分组记录。步幅只能从脚推出来，因为 locomotion 根的 XZ 没有真实位移。
- `stride_speed_fit`：同一物种所有 locomotion clip 上拟合 `stride = a · v_g^b`。运行时用户只调"速度"时，按这个物种自己的规律在步幅和步频之间分配。clip 数不足 3 时不做拟合，固定 b = 0.5。
- `vertical`：对 Y 带真实位移的 clip（跳跃、起身、下落等），记录 Y 的净位移和峰值高度（/ 腿长），作为 `jump_height` 参数的参考范围。

### 3.7 提取器自身的验证

1. **稳定性**：把一个物种的 clip 随机分成两半分别统计。hinge 主轴夹角 < 15°，ROM 区间的 IoU > 0.7，dof_class 一致。不满足的关节把 `confidence` 降级。
2. **对称性**：`symmetry_partner_indices` 给出的左右配对，镜像后主轴夹角 < 20°，ROM 接近。不一致的写进报告。注意对称性不一致也可能是骨架左右定义本身的问题。
3. **名字合理性**：名字是 knee/elbow/hiza/hiji 一类的关节应当被判为 hinge。不是的话写进报告，不自动修改。
4. **报告**：`skeleton_profiles_report.md`，按物种列出：低置信度关节、passive 候选、对称性不一致、名字和自由度冲突、失效的 `contact_overrides` 条目。

### 3.8 输出格式

下面只示意结构，其中的数值都是虚构的，不是实际统计结果。

```json
{
  "schema_version": 1,
  "profiles": {
    "truebones/zoo/Alligator": {
      "skeleton_hash": "sha1(parents + offsets)",
      "source": {"clips": 31, "action_families": {"locomotion|walk": 6, "...": 0}},
      "global": {"leg_length": 0.41, "hip_height": 0.33, "axial_length": 1.2},
      "contacts": {"cond": [10, 13], "override_add": [18, 21], "override_remove": [], "used": [10, 13, 18, 21]},
      "joints": [
        {"index": 9, "name": "R_hiza", "role": "support", "dof_class": "hinge",
         "principal_axes": [[...], [...], [...]], "hinge_flex_sign": 1,
         "rom": {"axis_deg": [[-5, 95], [-8, 8], [-6, 6]], "cone_deg": 98},
         "ang_speed": {"q50": 1.1, "q95": 6.3}, "confidence": 0.92}
      ],
      "gait": {"locomotion|walk, forward": {"period": 32, "duty": 0.68, "v_g": 1.3,
               "phase": {"10": 0.0, "13": 0.5, "18": 0.25, "21": 0.75}, "stride": 1.4}},
      "stride_speed_fit": {"a": 0.9, "b": 0.5, "n_clips": 6}
    }
  }
}
```

`skeleton_hash` 用来检测 Profile 过期：骨架改了（重新预处理、关节重命名或删除），哈希就会对不上，运行时拒绝使用并提示重建。这条规则和 orientation_quat 过期的教训是同一个道理。

### 3.9 没有动作数据的新骨架

`tools/process_new_skeleton.py` 处理过的新骨架没有动作统计，v1 采用回退策略：

- 拓扑、接触关节、对称性取自 cond（接触关节同样可以手动增删）；
- `dof_class`、ROM、`hinge_flex_sign`：按"规范关节名 + 链位置"，从 `species_tags` 相同的已知物种中取中位数；匹配不到时用宽松默认值（ball，±90°），并把 `confidence` 设为 0；
- 不启用 passive 模拟。

以后用预测模型替换这个回退策略，但要等 M7 的结论。

## 4. 阶段 B：分解器（生成结果 → Edit Package）

### 4.1 步骤

1. **解码并刚体化**：调用 `restore_animation_from_features(restore_space="hml", fullbody_ik=True, stretch_factor=s)`。生成结果的 pos 通道和旋转 FK 不完全一致时，普通解码会给关节解出逐帧的局部平移（骨长伸缩）；fullbody IK 把这部分位置信息改由旋转来表达，骨长被限制在静息长度的 `[1 − s, 1 + s]` 之内。
   - `s` 是分解参数，默认取 `utils/fullbody_ik.py` 的 `DEFAULT_IK_STRETCH_FACTOR`，与 `sample/export.py`、`tools/restore_glb_from_npy.py` 的默认导出一致。用户在加载或导入动作时可以改：调到 0 是严格刚体，调大则更贴近 pos 通道；
   - 得到局部旋转 `R[F, J]`、逐帧局部平移 `T[F, J]`（骨长在静息长度的 `[1 − s, 1 + s]` 之内，`s = 0` 时就是静息 offsets）、全局位置 `P[F, J]`、根轨迹；
   - IK 残差（`ik_error` 的均值和最大值）写进诊断；
   - package 同时保存原始特征和解码所需的 cond 子集，这样用户之后换一个 `s` 时可以在服务端重新分解，不需要回到数据集目录。
2. **接触区间检测**：对接触关节集合（cond 的 `contact_joints` 加上 `contact_overrides.json` 的增删，或用户在 UI 上对这个 package 的增删）中的每个关节，在 clip 内按下面的规则判定接触帧。**所有动作都做这一步**。
   1. 地面高度：取每帧所有接触关节中最低高度的 q05，作为该 clip 的地面高度。不用"该关节在这个 clip 里的最低高度"，否则飞行、游泳 clip 里收起来的脚每个周期都会被判成接触；
   2. 候选帧：关节高度 < 地面高度 + 0.05 × 腿长，且竖直速度 < 0.1 × 腿长 / 秒；
   3. 估计每帧的隐含地速 `v_g(t)`：取所有候选关节水平速度的中位数（取反），再沿时间做平滑；
   4. 确认接触：候选关节的水平速度与 `−v_g(t)` 的差 < 0.1 × 腿长 / 秒。原地 idle、攻击这类动作的 `v_g ≈ 0`，这时条件自然退化成"速度接近 0"；
   5. 生成结果比训练数据抖，先做中值滤波，再用滞回加最短持续 3 帧去抖。

   检测出的区间只是初值：用户可以在 UI 时间轴上增加、删除、拖动区间（第 5.4 节）。飞行、游泳这类根本没有地面接触的 clip，如果被误判出区间，也由用户在 UI 上清掉。

   同时在地面坐标系中测量每个接触区间内落点的漂移量，也就是生成结果自带的滑步量，作为基线写进诊断（T6 使用）。
3. **识别事件**：
   - **loop locomotion**：由接触时序得到周期 T 和各接触关节的着地帧、离地帧，并检查与 Profile 中 gait 相位的偏差，偏差写进诊断。生成的 loop 窗口是周期性的：第 F−1 帧的下一帧就是第 0 帧，末尾没有重复的收尾帧，周期 T 按这个约定计算。
   - **根轨迹**：根的 Y 和 XZ 都拆成趋势和振荡两部分。趋势是低通分量（loop 动作的窗口取一个周期，one-shot 取 0.5 秒），振荡是剩下的部分。Y 的趋势就是净位移和跳跃弧线；XZ 的趋势在 locomotion 上接近 0，在攻击突进、击退、倒地等动作上就是真实位移。Y 带真实位移时记录起跳帧和落地帧。
   - **one-shot**（攻击、受击、跳跃、死亡等）：接触区间照常识别。攻击中常见的垫步、跨步、踩踏，就是中途离地、在新位置落地的接触区间，按普通落点处理。取能量（角速度平方和）最高的那条链作为**主动链**，其末端速度峰值帧为 `impact`；`impact` 之前，末端沿击打方向位移的局部极小值为 `windup`；`impact` 之后能量回落到峰值 15% 的帧为 `recover`。每个事件都附带置信度。
   - 事件只是**可编辑的标记**：微调 UI 允许用户拖动事件帧，拖动后重新计算依赖它的层。
4. **分层**：
   - `base`：每帧局部旋转和局部平移，作为兜底的细节层；
   - `root`：XZ 趋势、XZ 振荡、Y 趋势、Y 振荡、yaw 曲线；
   - `chains`：按肢体链分组，每个关节相对**链参考姿态**的 rotvec 偏差曲线。loop 动作的参考姿态取周期均值；one-shot 取首帧，因为一般是起始站姿。偏差的模长接近 π 时（> 0.8π），缩放会让 rotvec 翻到另一侧，这些帧写进诊断，运行时对这些帧把该关节的增益夹到 1；
   - `plants`：每个接触区间的 `[start, end]`、`v_g` 曲线、接触点在"地面坐标系"（位置加上 `v_g` 的累积位移）中的落点，以及区间内脚相对落点的残差曲线（原结果自带的滑步）。落点同时保存为相对于根的偏移；
   - `events`：上述事件帧和周期。

### 4.2 Package 格式

```text
<clip>.edit/
  manifest.json   # runtime_version、profile_schema、skeleton_hash、fps、is_loop、
                  # action_label、stretch_factor、接触关节集合及其来源（cond / 物种覆盖 / 本 package 手动）、
                  # 接触区间是否被手动改过、各参数的默认值与范围、诊断信息
  data.npz        # parents, offsets, names, forward_joint_index, forward_base_joint_index,
                  # base_rot, base_pos, root_*, chain_offsets, plants, events,
                  # 运行时用到的 profile 子集（dof/axes/rom/flex_sign/role/spring），
                  # 以及服务端重新分解用的原始特征和 cond 子集（运行时不读）
```

Package 是自包含的：微调端不需要 cond.npy，也不需要数据集目录。`runtime_version` 不兼容时拒绝加载。

## 5. 阶段 C：微调运行时

### 5.1 v1 参数集

| 参数 | 默认值 | 范围 | 作用对象 | 保证 |
|---|---|---|---|---|
| `tempo` | 1.0 | 0.5–2.0 | 全局时间 | loop 动作保持闭合，结果帧数为整数 |
| `stride` | 1.0 | 0.6–1.6 | locomotion：隐含地速 `v_g` 与脚的前后摆动距离（根 XZ 不动，仍是原地动作） | 支撑期各接触关节的速度都等于新的 `v_g` |
| `speed` | 1.0 | 0.6–1.8 | locomotion：按 `stride_speed_fit` 在步幅和步频之间分配 | 同上 |
| `bounce` | 1.0 | 0–2 | 根 Y 的振荡分量 | 支撑脚锁定，不穿地 |
| `jump_height` | 1.0 | 0.5–1.8 | 根 Y 的趋势分量（跳跃弧线、起身高度），起跳和落地帧不动 | 落地帧接触关节回到地面 |
| `sway` | 1.0 | 0–2 | 根 XZ 的振荡分量；XZ 趋势（突进、击退等真实位移）不受影响 | 支撑脚锁定 |
| `amp.legs / arms / axial / tail / wings` | 1.0 | 0–2 | 各类链的偏差幅度 | ROM 软夹紧；支撑脚锁定 |
| `posture` | 0 | −0.3–0.2（× 腿长） | 髋高偏移（蹲 / 伸） | 支撑脚锁定，腿部 IK 重解 |
| `force` | 1.0 | 0.5–2.0 | one-shot：组合下面三项 | 事件顺序不变；支撑脚锁定 |
| `windup_depth` | 1.0 | 0–2 | windup 段主动链的偏差 | ROM；支撑脚锁定 |
| `strike_speed` | 1.0 | 0.5–2.0 | windup→impact 的时间压缩 | 时间单调 |
| `overshoot` | 1.0 | 0–2 | impact→recover 段的过冲幅度 | ROM；支撑脚锁定 |
| `impact_shift` | 0 | ±0.3（× 动作时长） | impact 事件在时间上的位置 | 事件顺序不变 |
| `secondary.stiffness / damping` | 1.0 | 0.25–4 | passive 关节的弹簧参数倍率 | 只作用于已确认的 passive 关节 |
| `foot_lock` | 关 | 开 / 关 | 所有带接触区间的动作 | 开：接触区间内落点漂移为 0（够不到的帧除外）；关：保留原结果自带的滑步 |

"支撑脚锁定"对所有动作成立：只要某个参数改变了支撑链或根的姿态，第 5.2 节的第 5 步就会对支撑脚重解 IK，让落点跟着编辑走而不是跟着身体走。主动链本身就是腿的时候（踢、踩踏），该腿在主动段不是支撑脚，不受锁定，只有它落地之后的接触区间才锁定。

`force` 不是独立的运算，它只是同时改变 `windup_depth`、`strike_speed`、`overshoot` 的一个预设组合；美术也可以分别调整这三项。

`foot_lock` 是唯一一个会改变"原结果中与编辑无关部分"的参数，所以它是开关而不是滑杆，打开时结果允许与原结果有明显差别。

### 5.2 运算顺序（固定）

```text
0. 所有参数为默认值的层直接跳过（这样才能保证严格回放）
1. 时间扭曲：以事件为节点做单调分段三次插值（PCHIP），构造 t' = w(t)
   loop：时间扭曲后按周期重采样到整数帧数；周期约定为第 F−1 帧之后接第 0 帧，插值在首尾之间环绕
2. 幅度：chain_offsets 按每段的增益缩放（amp.* × windup_depth / overshoot 的分段曲线，段与段之间平滑过渡）；
   第 4.1 节标出的接近 π 的帧，该关节增益夹到 1
3. ROM 软夹紧（只压超出原结果自身幅度的部分）：
   对每个关节、每个主轴，设原结果在该帧的投影角为 x(t)，编辑后为 x'(t)，
   通过阈值 a(t) = max(rom 的膝点, |x(t)|)，上限 b(t) = max(rom 边界, |x(t)|)；
   |x'| ≤ a 时原样通过，超过 a 的部分用 tanh 渐近到 b。x' = x 时输出就是 x，所以从默认值出发是连续的。
   hinge 关节先把编辑引起的增量投影到主轴上，再做夹紧
4. 根：sway 缩放 XZ 振荡；bounce 缩放 Y 振荡；jump_height 缩放 Y 趋势（起跳到落地之间的弧线，弧线两端固定）；
   posture 平移 Y；XZ 趋势保持不变
5. 锁脚 + IK（所有带接触区间的动作都执行；在地面坐标系里做，最后再变回原坐标）：
   - 触发条件：第 2–4 步中任何改动了支撑链或根的参数不是默认值，或者 stride / speed 不是默认值，或者 foot_lock 打开
   - 新地速 v_g' = v_g × stride（或按 speed 和 stride_speed_fit 分配后的值）；非 locomotion 动作 v_g' = v_g ≈ 0
   - 落点：stride / speed 为默认值时，直接使用 package 里记录的原始地面坐标落点（攻击、受击、posture、amp 这类编辑都走这条路径）；
     改了 stride / speed 时，用区间中点处相对根的原偏移重新推出落点，沿前进方向
     （由 forward_base_joint_index → forward_joint_index 给出）按 stride 缩放
   - 支撑期目标 = 新落点 + 原结果的滑步残差（第 4.1 节 plants 里保存的残差曲线）；
     foot_lock 打开时残差取 0，落点锁死。变回原坐标后，脚以 −v_g' 匀速后移（v_g' = 0 时就是不动）
   - 摆动期：由前后落点做插值，加上原轨迹相对插值线的残差（保留抬脚高度和弧线形状）
   - 求解腿链 IK：骨长取 package 的逐帧局部平移（stretch_factor = 0 时就是静息 offsets），IK 不改变骨长；
     每个 hinge 关节以主轴和 hinge_flex_sign 作为 pole 约束；两节的腿用解析的两骨 IK，
     三节及以上的腿（如四足后腿 髋-膝-跗-足）用带 hinge 约束的迭代链 IK；
     每次迭代后对链上关节做第 3 步的 ROM 软夹紧。够不到时伸直并夹紧，写入诊断
6. 次级运动：passive 关节只叠加增量：
   θ_out = θ_orig + sim(新父运动, k', c') − sim(原父运动, k, c)
   其中 k'、c' 是乘上 secondary.* 倍率后的弹簧参数。两次模拟用同一个积分器、同一组初值
   loop：预热两个周期，取第三个周期作为结果，保证首尾闭合
7. 着地：只补偿编辑引起的高度变化。在接触帧上，计算编辑后与原结果的最低接触关节高度之差，
   对 Y 做相应补偿（规则与 retarget 的 --ground 一致：以最低的接触关节为准）。原结果自带的悬空或穿地保持不变。
   Y 带真实位移的动作（跳跃、下落），只对齐起跳前和落地后的接触段，不把整段压平
8. 输出 Animation；导出复用 npy_restore 的 exporter 路径（BVH / GLB）。
   可选输出 root motion：把 v_g' 积分到根 XZ，供需要根位移的引擎使用；默认仍输出原地动作，与训练数据一致
```

### 5.3 运行时用到的算子与复用关系

| 算子 | 来源 |
|---|---|
| 四元数 / rotvec 运算、FK | `motion_lib` 的 `Quaternions`、`Animation` |
| 重采样 | `npy_restore.resample_animation`，给它加一个 `periodic` 选项：周期模式下第 F−1 帧之后接第 0 帧，插值在首尾之间环绕。现有调用方不受影响 |
| PCHIP 时间扭曲 | 新写，放在 `runtime.py` |
| 腿链 IK | 在 `utils/fullbody_ik.py` 中扩展：两节的腿用解析两骨 IK，三节及以上用迭代链 IK，都加 hinge pole 约束和逐帧骨长；`run_basic_inverse_kinematics_with_constraints` 的逐关节迭代可作为链 IK 的起点 |
| tanh 软夹紧、二阶弹簧积分 | 新写，放在 `runtime.py` |

`profile/` 统计 hinge 主轴时用到的 log map 与运行时用同一套四元数实现，避免两边对 rotvec 的约定不一致。

### 5.4 微调 UI：本地 serve 页面

运行时之上是一个本地网页，用户在页面上调参数、拖事件标记、修正接触，并立即看到结果。它与 `dataset/review/serve.py` 采用同一种形态：标准库 `ThreadingHTTPServer` 加一个静态页面，不引入 web 框架。

**进程与职责**

```text
motion_edit/ui/serve.py          python -m motion_edit.ui.serve --packages <dir> [--port 8770]
  ├─ 加载 package，调用 EditRuntime.apply（运行时只在服务端跑，页面不重写任何编辑逻辑）
  ├─ 加载时可选 stretch_factor：与 package 当前值不同时，用 package 内的原始特征重新分解
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
| GET | `/api/packages` | 列出 `--packages` 目录下的 package（clip 名、物种、action_label、is_loop、stretch_factor、诊断摘要） |
| GET | `/api/package/<id>` | manifest：参数默认值与范围、事件、接触关节集合及来源、接触区间、骨架（parents / offsets / names）、哪些参数对该动作可用 |
| POST | `/api/load` | 输入：package id + stretch_factor（可选）；stretch_factor 与当前值不同时重新分解并覆盖 package，返回新的 manifest。重新分解会重置手动改过的接触区间，页面在执行前要求确认 |
| POST | `/api/contacts` | 输入：package id + 接触关节集合（可选）+ 接触区间（可选）+ 是否"应用到该物种"；关节集合变了就重新检测区间，区间变了就重新计算落点和残差；"应用到该物种"时把关节增删写进 `contact_overrides.json` |
| POST | `/api/apply` | 输入：package id + 参数 + 事件帧（可选，用户拖动过时才带）；输出：逐帧全局位置、局部旋转、接触区间、`v_g'`、诊断（夹紧、够不到、超 ROM、rotvec 接近 π） |
| POST | `/api/export` | 输入同 `/api/apply`，外加格式（bvh / glb）、是否输出 root motion；写入文件并返回下载链接 |
| GET / POST | `/api/presets/<id>` | 读写该 package 的参数预设（JSON，与 package 放在一起） |

v1 不考虑性能：参数滑杆在松开时才请求 `/api/apply`，拖动过程中不发请求；新请求到达时丢弃尚未返回的旧请求的结果。

**页面布局**

1. **3D 视图**（three.js）：
   - 骨架用线段加关节球显示；接触关节在接触区间内高亮，并在地面上标出落点；
   - 点选关节可以把它标记为接触关节或取消标记（走 `/api/contacts`）；
   - 地面网格：非 locomotion 动作的网格静止；原地 locomotion 的网格按 `v_g'` 滚动，可以切换成"root motion 视图"，让角色真正前进；
   - **对比**：同时显示原始结果（半透明）和编辑后的结果；也可以切换成左右并排；
   - 播放、暂停、逐帧、播放速度、循环开关；相机可以环绕和跟随。
2. **参数面板**：按第 5.1 节分组（时间 / locomotion / 根 / 幅度 / 力度 / 次级运动 / 锁脚）。对当前动作不可用的参数直接隐藏（比如 attack 不显示 stride，没有 passive 关节就不显示 secondary）。每个滑杆都有"恢复默认"按钮；面板顶部有"全部恢复默认"，恢复后结果必须与原始结果完全一致（T1 在 UI 上的体现）。加载对话框里可以设置 stretch_factor。
3. **时间轴**：显示当前帧、事件标记（windup / impact / recover、起跳 / 落地、loop 周期边界），以及每个接触关节的接触区间色条。事件标记可以拖动；接触区间可以增加、删除、拖动两端；拖动后带着新的数据重新请求。
4. **诊断面板**：列出本次 apply 的夹紧、够不到、超 ROM、rotvec 接近 π、事件低置信度、分解时的 IK 残差等信息，点一条就跳到对应的帧并高亮对应的关节。
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
| T1 | 严格回放 | 默认参数下，与 `restore_animation_from_features(restore_space="hml", fullbody_ik=True, stretch_factor=s)` 的输出相比，局部旋转最大误差 < 1e-5 rad，位置误差 < 1e-6 |
| T1b | 分解往返 | 关闭第 0 步的跳过，强制走"分解 → 所有增益为 1 → 重组"的完整路径，与 T1 相同的阈值。T1 在默认值下跳过了所有层，测不到分解器，这一条才能 |
| T2 | 确定性 | 同一 package、同一组参数，两次运行的输出逐位一致 |
| T3 | ROM | 任何参数组合下，每个关节每个主轴的投影角不超过 `max(rom 边界 × 1.1, 原结果在该帧的投影角)`；IK 输出同样检查 |
| T4 | loop 闭合 | 环绕步（第 F−1 帧 → 第 0 帧）与步长中位数之比，不超过原生成结果的同一比值 |
| T5 | 骨长 | 每帧每根骨的长度与 package 的 `base_pos` 一致（`stretch_factor = 0` 时就是静息长度），误差 < 1e-6；编辑不改变骨长 |
| T6 | 锁脚 | 所有动作（包括攻击、受击等非 locomotion 动作）、所有参数组合下，地面坐标系中每个接触区间内的落点漂移，不超过原结果自带的滑步基线（第 4.1 节）；`foot_lock` 打开时漂移 < 1e-4 × 腿长（诊断为够不到的帧除外） |
| T7 | 连续性 | 对每个滑杆参数，取默认值 ± ε（ε = 范围的 1e-3），输出与默认输出的差（旋转、位置）不超过 C·ε。用来防止某一层在偏离默认值时一下子全量生效 |

T6 不是"效果好"的判据，只是防止运行时把滑步弄得比原来更糟。

### 6.2 人工审查（效果验收）

微调 UI 做好后，由人在 UI 上挑几条动作试用，不做批量渲染和逐项打分。挑选时尽量覆盖 loop locomotion、one-shot attack（带突进位移的）、Y 带真实位移的动作（跳跃或起身）。也可以混入一条训练集的原始 clip（不经过模型生成），用来区分是分解器和运行时的问题，还是生成结果本身的问题。

对每个参数，把滑杆从最小拖到最大，结合半透明的原始结果对比和地面落点标记来看：

1. **名实相符**：参数的效果和名字说的是同一件事（stride 调大就是步子变大，而不是腿抬得更高）；
2. **单调**：效果随参数连续、单调变化，中间没有跳变；
3. **无瑕疵**：没有比原结果更多的滑步、膝肘反弯、穿地、帧间跳变（pop）、loop 接缝；
4. **极值可用**：最小和最大值不要求自然，但不能崩坏。崩坏就收窄该参数的范围。

发现的问题按"修复、收窄范围、作为已知限制"之一处理。

## 7. 里程碑

| 里程碑 | 内容 | 完成标准 |
|---|---|---|
| M1 | Profile 提取器 + 报告 | 所有物种的 Profile 生成完毕；第 3.7 节的检查全部跑完；人工看过报告，处理完 passive 候选 |
| M2 | 分解器（含 fullbody IK 刚体化、stretch_factor、接触区间检测）+ 严格回放 + UI 基础（server、3D 视图、对比、参数滑杆、加载时设置 stretch_factor） | T1、T1b、T2、T5 通过；UI 能加载 package 并做往返回放。之后 M3–M5 的人工审查都在这个 UI 上进行 |
| M3 | 运行时：tempo / stride / speed / bounce / sway / jump_height / amp / posture / foot_lock + 锁脚 IK；UI 上的接触修正（标记 / 取消接触关节，增删、拖动接触区间，应用到该物种） | T3–T7 通过；在原地 locomotion、Y 位移动作、带突进的攻击和 idle 上人工审查通过 |
| M4 | one-shot 事件 + force 系列参数 | 人工抽查 attack clip 的事件帧，正确率 ≥ 80%；attack / hurt 上 T6、T7 通过；force 系列参数人工审查通过（试用的动作中要有带垫步、跨步的攻击 clip） |
| M5 | 次级运动弹簧（增量模拟） | 已确认的 passive 关节启用；T4、T7 通过；人工审查通过 |
| M6 | 微调 UI 完整版（第 5.4 节）：事件拖动、诊断面板、导出、预设、root motion 视图 | UI 上"全部恢复默认"与原始结果完全一致；UI 导出的文件与命令行用同一组参数导出的文件逐位一致；美术试用，收集参数是否够用、是否直观的反馈 |
| M7 | Profile 用于生成后处理（hinge 投影 + ROM 软夹紧），不改模型 | 对同一批生成结果做处理前后的盲审 A/B（左右位置随机），记录偏好；结论决定是否立项"预测器"和"Profile 作为 AnyTop 条件" |

M1 与 M2 可以并行；M3 依赖两者；M4、M5 可以并行；M6 可以在 M3 之后的任意时间开始，随 M4、M5 增加参数分组。

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| 生成结果抖动，接触区间和事件检测不稳 | 中值滤波加最短持续时间；接触区间和事件都允许用户在 UI 上手动修正 |
| cond 的 `contact_joints` 有漏标（如 Alligator 前脚） | 用户在 UI 上手动标记，可以只对一个 package 生效，也可以写进 `contact_overrides.json` 对整个物种生效；cond 修正走正常的 regen + 重训流程，单独决策 |
| 飞行、游泳 clip 被误判出接触区间 | 地面高度取所有接触关节的共同低分位，不取单个关节的最低点；剩下的误判由用户在时间轴上清掉 |
| fullbody IK 刚体化后偏离 pos 通道 | 分解时记录 IK 残差并显示在诊断面板；用户可以在加载时调大 stretch_factor |
| 物种 clip 少、动作族单一，ROM 偏窄 | 记录覆盖度；`confidence` 低时 ROM 放宽到同 `species_tags` 物种的分布；ROM 夹紧本身只压超出原结果幅度的部分，不会削掉原结果 |
| passive 拟合误判 | 判据要求驱动项有显著贡献（第 3.5 节）；只作为候选，人工确认后才启用 |
| 链偏差接近 π，缩放后翻转 | 分解时标出这些帧，运行时把该关节在这些帧上的增益夹到 1，并写入诊断 |
| 原地 locomotion 的接触判定依赖隐含地速估计，转弯、侧移时各脚速度不一致 | `v_g` 改为每帧的二维向量加 yaw 角速度（刚体平面运动），按每只脚的位置分别计算其应有速度；先只支持直行，转弯类动作在 M3 中只做 tempo / amp，不开放 stride |
| 无腿、飞行、游泳类骨架没有落点 | 自动禁用 stride / speed 的落点重算，只做根轨迹缩放；`bounce` 改为作用在主轴振荡上 |
| 时间扭曲导致 loop 周期不是整数帧 | 第 1 步强制整数帧周期重采样；T4 兜底 |
| hml 空间与 native 空间的换算 | 运行时全程在 hml 空间工作，长度用腿长归一化；导出时沿用 npy_restore 的 `restore_space` 逻辑 |
| Profile 与骨架不同步 | `skeleton_hash` 校验（Profile 和 `contact_overrides.json` 都校验）；把 `motion_edit.build_profiles` 作为 `regenerate_dataset_artifacts` 的可选步骤 |

## 9. 后续（本方案之外）

- **M7 有收益时**：做 AnyTop 的 oracle 实验，用真实 Profile 作为条件、在留出物种上评估。只有 oracle 有提升，才训练"骨架 → Profile"预测器，并用**预测值**加噪声去训练 AnyTop。
- **编辑后修顺**：参数超出运行时能处理的范围时，把编辑结果作为约束送回服务端，做低噪声的 inpainting 修顺（使用已有的 joint / temporal mask 机制），固定种子。
