# Skeleton Profile 与 Motion Edit 运行时：设计文档

本文说明当前 `motion_edit` 系统如何把 AnyTop 的动作输出转换成可交互编辑、可导出的骨架动画。新开发者可以先读“数据流”和“编辑如何生效”，再按末尾的代码索引定位实现。

## 1. 系统职责与数据流

AnyTop 生成形状为 `(F, J, 12)` 的动作特征，通道为位置 3、旋转 6D 6、速度 3。Motion Edit 在生成后工作：它把特征解码为动画，提取接触、事件和可编辑层，随后在本地根据参数重新合成动作。调整参数不调用生成模型。

```text
训练数据 motions/*.npy + cond.npy + motion_metadata.json
    │  build_profiles：按骨架统计关节、步态、次级运动属性
    ▼
<dataset_root>/skeleton_profiles.json
    │
生成特征 (F, J, 12) + cond 条目 + 可选 Profile
    │  decompose_clip / decompose_motion：解码、检测、分层
    ▼
<clip>.edit/{manifest.json, data.npz}  （可带 mesh/）
    │
    ├─ EditRuntime.apply(params, events) → Animation → UI 预览
    └─ apply_edit → BVH / GLB；可选 root motion、蒙皮 mesh
```

系统有三类数据，作用不同：

| 数据 | 来源 | 用途 |
|---|---|---|
| `cond` 条目 | 骨架数据集或另行指定的 `cond.npy` | 关节层级、静息骨架、左右标记等骨架定义 |
| 关节部位与接触 | 从骨架预填，物种级 / Package 级人工校准（第 2.1 节） | 编辑分组、次级运动通道、IK 肢体、接触集合、发力链先验 |
| Skeleton Profile | 同一骨架的训练 clip 离线统计 | 关节自由度与主轴、角色、弹簧参数、步态参考；分解时取出该 clip 所需子集 |
| Edit Package | 对某条生成或训练动作分解得到 | 运行时需要的全部数组和元数据；可独立加载并重复编辑 |

Profile 是可选输入。缺失时，分解器从 `cond` 推导角色和腿长，为次级运动候选生成默认弹簧；骨架不匹配的 Profile 由 `skeleton_hash` 拒绝。Profile 不写入 `cond.npy`，运行时也不读取数据集或模型。

### 原地 locomotion 与真实位移

数据预处理使 `action_group == locomotion` 的水平根位移以原地动作为主。支撑脚会相对身体向后移动；系统从接触脚速度推算隐含地速 `v_g`，而非从根 XZ 读取前进速度。其他动作的根位移保留在动画中；根 Y 也承载跳跃等竖直位移。编辑器默认保留输入动作的位移表达。UI 的 root motion 视图和导出时的 `--root_motion` 把 `v_g` 的积分位移加到顶层关节 XZ，使原地 locomotion 前进。

## 2. Skeleton Profile：骨架的统计描述

入口是 `python -m motion_edit.build_profiles`。它发现已处理数据集，读取 `cond.npy`、`motions/*.npy` 和 `motion_metadata.json`，为各物种写入 `skeleton_profiles.json`，并生成 `skeleton_profiles_report.md`。训练 clip 的解码使用 `utils.npy_restore` 的 `build_skeleton_only_context` 与 `restore_animation_from_features(restore_space="hml")`。

统计按动作族均衡：每个动作族总权重相同，同族每条 clip 权重相同，clip 内帧均分该权重。Profile 主要包含：

- **全局尺度**：腿长、髋高、轴向长度。腿长用于接触阈值、IK 与参数的尺度归一化；无腿骨架以轴向长度回退。
- **逐关节描述**：`dof_class`（fixed / hinge / planar / ball）、`principal_axes`、`hinge_flex_sign`、角速度、`role`、`confidence`。自由度由局部旋转相对均值姿态的 rotvec 分布及 PCA 得到；IK 把主轴和屈曲方向当作偏好约束。
- **次级运动**：候选关节的 `spring` 和拟合信息。数据能支持受父运动驱动的拟合时使用拟合参数，否则用按悬挂长度计算的默认弹簧。是否启用 passive 由关节选择决定，和弹簧拟合是否通过无关。
- **动作参考**：locomotion 的周期、占空比、着地相位、隐含地速、步幅，以及竖直动作统计。分解器使用匹配 `action_label` 的步态行比较当前动作并产生诊断；编辑以当前动作自身的事件和曲线为基准。

接触关节集合和 passive 关节都由第 2.1 节的部位决定；接触发生在哪些帧由动作计算。Passive 候选须满足子树内没有支撑关节，部位为 `soft` 的候选启用。没有训练 clip 的骨架可按相同 `species_tags` 和规范关节名借用其他 Profile 的自由度信息，置信度为 0；没有可用 Profile 时仍可分解。

### 2.1 关节部位与接触

每个关节有一个部位和一个接触位（`motion_edit/profile/parts.py`）。默认值由 `data_loaders/truebones/truebones_utils/joint_parts.py` 的 `prefill_joint_parts` 预填：先按 embedding 文本里的部位词，再沿树继承，再做肢体细化（着地的肢体末端改为 hand/foot、翼链整条改为 wing、多足的步行肢改为 leg/foot、四足按前后分 arm/leg），最后用几何兜底；接触来自 `prefill_contacts`（先几何后名字），名字是 Foot / Hand 且下面还有趾、指的关节不算接触。

人工校准分两层，后一层覆盖前一层，每层只记与下层不同的关节（`{关节名: {"part", "contact"}}`）：

- **物种层**：数据集根目录的 `joint_parts_overrides.json`，按 cond key 存，带 `skeleton_hash`，骨架变化后整行忽略并写诊断；
- **Package 层**：存在 manifest 的 `parts.package` 里，只影响这一个 Package。

Profile 构建和分解读同一份「预填 + 物种层」。预填只看 cond 子集（`profile.build.COND_FIELDS`），Package 重建时用存下的子集重算，结果与首次分解一致。

| 部位 | 含义 / 判定 |
|---|---|
| `trunk` | 骨盆、脊柱、胸腔；蛇、鱼、蠕虫的主体段 |
| `neck` | 颈椎链 |
| `head` | 头及其附属：下颌、眼、耳、舌、角、喙 |
| `arm` | 前肢从肩 / 锁骨到腕之前 |
| `hand` | 腕及以下：掌、指、爪 |
| `leg` | 后肢从髋到踝之前；多足动物步行足的近端段 |
| `foot` | 踝及以下：跖、趾、蹄、爪尖 |
| `wing` | 整条翼链，包括翼指和受驱动的羽毛骨 |
| `tail` | 尾链 |
| `fin` | 鳍 |
| `soft` | 被动附属物：触须、毛发、鬃、衣物、披风、饰物、被动触手、植物叶片 |
| `helper` | 非解剖节点：包装根、locator、IK target、道具挂点；不能是接触 |

歧义判定：

- 四肢按前后位置判定，不按用途。四足的前肢是 `arm/hand`，后肢是 `leg/foot`，前肢着不着地都一样，承重由接触位表达。节肢、多足动物的步行足一律 `leg/foot`，螯、钳、须肢是 `arm/hand`。
- 翼、鳍、`soft` 按功能判定。蝙蝠的「手指」是 `wing`；章鱼主动划水的腕是 `arm`；水母被动拖曳的触手是 `soft`。海龟、海豹主动划水的大鳍肢归 `arm/leg`，小的稳定鳍归 `fin`。
- 尾巴末端受尾链驱动、自身无主动运动的饰物（毛团、尾羽）归 `soft`。爪长在 hand 上的归 `hand`，长在 foot 上的归 `foot`。
- 接触是正常站立或行走时会着地的关节；四足前肢的指同样可以是接触。

部位驱动的编辑：

| 机制 / 参数 | 由什么决定 |
|---|---|
| 接触集合、`foot_lock`、`soft_stretch`、`jump_height`、`stride` | 接触位 |
| IK 肢体 | 从接触关节向上到同侧、同一肢体类（arm/hand、leg/foot、wing）的最高关节 |
| `amp.legs` / `amp.arms` / `amp.wings` / `amp.fins` / `amp.axial` | `leg+foot` / `arm+hand` / `wing` / `fin` / `trunk+neck+head`；翻译根单列为 root，`soft`、`helper` 不受幅度滑杆控制 |
| `tail_weight` / `tail_stiffness` | `tail` 中能摆动的关节 |
| `passive_*`、`gravity` | `soft` 中能摆动的关节 |
| `spread.arms` / `spread.legs` | arms / legs 组里挂在其他组关节上的肢体；带 IK 肢体的从 IK 肢体根部转、移动其着地目标 |
| 发力链的标签先验 | bite / headbutt / firebreath / spit → `head`、`neck`；kick → `leg`、`foot`；punch / swat / slash / stab / catch / smash → `arm`、`hand`，没有再找 `leg`、`foot`；sting / whip → `tail` |

## 3. Edit Package：动作的可编辑表示

`decompose_clip.py` 接收生成特征 `--npy` 和骨架 `--object_type`，或数据集训练动作 `--clip <namespace>:<motion file>`。它读取 Profile 与覆盖，再调用 `decompose_motion`。后者先用 `restore_animation_from_features` 解码；默认启用 full-body IK，`stretch_factor` 默认 0.2。`--no_fullbody_ik` 保留解码结果的逐帧局部位移，此时 `stretch_factor` 不起作用。

例如，从 `Anytop/` 将生成结果制成 Package：

```bash
python -m motion_edit.decompose_clip --npy out/sample0.npy --object_type truebones/zoo/Horse --action_group locomotion --action_label "walk, forward" --is_loop --out outputs/edit_packages
```

分解器随后依次建立以下信息：

1. **接触**：`contacts.detect_contacts` 对指定接触关节计算 `(F, K)` 布尔 mask 与逐帧 XZ 隐含地速。判定条件是关节接近自己的最低高度且接近地面、竖直速度低、水平速度与其他支撑脚的共同地速一致；迟滞和最短帧数把候选帧连成区间。循环动作跨首尾处理。
2. **着地锚点**：每个接触区间在地面坐标中有 `plant_anchor`。原动画在该区间相对锚点的漂移保存在 `plant_residual`，供运行时保留原有滑步或显式锁脚。
3. **事件**：循环 locomotion 提取步态周期和着地时刻；接触 mask 给出腾空区间；可识别的攻击/受击动作提取 `windup`、`impact`、`recover` 及候选发力链。低置信度结果进入诊断。
4. **可编辑层**：根位置拆为低频 `root_trend` 与 `root_osc`，根旋转拆为世界 yaw 与 tilt；其余关节以均值姿态 `chain_reference` 和连续展开的旋转向量 `chain_offsets` 表示。循环中转满一圈的关节锁定幅度增益，避免接缝断裂。
5. **动作事实与参数**：`facts` 记录循环、locomotion、着地、转向、腾空、passive、strike 和关节组情况；据此确定本动作可用的参数。

一个 Package 是 `<clip>.edit/` 目录：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 运行时版本、骨架与动作元数据、Profile 状态、接触和 passive 来源、事件、可用参数、诊断 |
| `data.npz` | 骨架、解码后的基准动画、分解层、接触 mask 与锚点、Profile 子集；另存 `source_features` 与 JSON 编码的 `source_cond` 供重新分解 |
| `mesh/`（可选） | `--tpose_mesh` 指定的 T-pose FBX/GLB 的预览与标定数据，用于蒙皮预览和导出 |

`package.py` 负责读写，当前 `RUNTIME_VERSION = 6`；加载时要求版本一致。`runtime.py` 只读取 Package，不导入 torch，也不走特征解码路径。编辑部位与接触、地面高度或接触区间时，`decompose.py` 的 `with_parts` / `with_ground_height` / `with_contact_mask` 会用 Package 内保存的数据重建相关层；部位编辑不改变接触集合时，手工接触区间保留。修改解码设置走 `redecompose`，从保存的原始 features 重新解码；它会重新检测接触区间，手工区间编辑不会保留。

## 4. EditRuntime：参数如何作用于动画

`EditRuntime(package).apply(params, events=...)` 先补齐默认值，再将数值参数夹到范围内，并拒绝未知参数或对当前动作不可用的非默认参数。`runtime.PARAM_SPECS` 定义参数、范围和分组；`available_params(facts)` 控制 UI 呈现。输出 `EditResult` 包含 `Animation`、全局关节位置、输出地速、源时间映射、接触目标和诊断。

运行时按固定顺序合成：

1. **时间映射**：`tempo` 改变源时间采样。循环动作保持整数输出帧数并周期采样；one-shot 按源帧采样。存在 strike 事件时，`windup_speed`、`strike_speed`、`recover_speed` 可分别改变事件段速度，并用单调 PCHIP 插值形成时间映射。UI 可移动三个事件和选择发力链；事件是力度编辑的时间锚点。
2. **关节幅度与发力**：对各组的 rotvec 偏移使用 `amp.legs`、`amp.arms`、`amp.axial`、`amp.wings`、`amp.fins` 等增益（分组见第 2.1 节）。`force` 调整蓄力、出击、恢复相关参数；`windup_depth` 和 `overshoot` 对发力链、躯干和根施加相对接触姿态的变化，并带动身体倾斜和位移。`spread.arms`、`spread.legs` 取 −1 到 1：每条有左右之分的手臂或腿在肢体根部转动，使肢端沿远离身体中线的水平方向移动“取值 × 比例 × 肢长”，负值向内收拢；比例是组的比例（arms 1.0、legs 0.3），着地的肢体一律用着地比例 0.3；外向方向取左右肢体根部连线的水平方向。四足的前肢属于 arms、后肢属于 legs，可分别调节。着地的肢体从 IK 肢体根部转动、移动其着地目标，脚掌保持原世界朝向；不着地的肢体从链顶转动、移动最远的关节。肢体首段过短或名为锁骨/肩胛时从下一关节转。某组没有左右成对的肢体时不提供该参数。肢体已指向正外或正内、或越过中线时写入诊断。
3. **根运动**：`bounce` 缩放 Y 振荡，`sway` 缩放 XZ 振荡，`jump_height` 缩放腾空区间的离地高度：取最低接触关节高出其 floor（该关节在本片段中的最低支撑高度，从未支撑的关节取地面高度）的距离，减去起跳帧到落地帧两端值的连线（不低于 floor），根在竖直方向上移动 (k−1) 倍这个离地量；因此最低的脚始终不会低于该连线和 floor，k<1 不会把脚压进地面，区间两端也保持不动；循环动作的腾空区间可以跨越接缝，`posture` 按腿长偏移根高度。
4. **着地目标与 IK**：对原结果的支撑脚位置施加编辑引起的变化。`stride` 改变原地 locomotion 的隐含前进速度和步幅；`spread.legs` 把每段支撑的着地目标沿该段中点的外向方向平移同样距离，整段支撑内不变，因此脚不滑动；默认保留原动画的滑步残差，`foot_lock` 打开时才移除。关节幅度、力度、根等会移动身体的编辑触发肢体 IK；`soft_stretch` 控制接近伸展或压缩极限时的软骨长调整。够不到的目标和修正量写入诊断，必要时对身体做着地高度补偿。
5. **次级运动**：`tail_weight`、`passive_weight` 在 0–1 区间缩放部位自身曲线；超过 1 时，在编辑后的身体运动上叠加弹簧响应。对应的 `*_stiffness` 改弹簧频率，`gravity` 控制重力驱动项。次级运动在着地修正后计算，因此弹簧读到最终的身体轨迹。

默认参数有一条回放路径：`apply()` 直接返回 Package 保存的 `base_rot` / `base_pos`，即分解时 `restore_animation_from_features` 的结果。`apply(compose=True)` 强制从可编辑层重组；UI 的 `/api/apply` 使用这条路径，CLI 可通过 `--compose` 指定。编辑层尽量表达相对原动画的变化：未打开 `foot_lock` 时保留原有滑步，次级运动只增加弹簧响应，着地补偿针对编辑造成的偏移。

单独移动 strike 事件只改变后续力度编辑使用的锚点；力度参数保持默认值时不会改变姿态。

参数是否可用取决于动作事实：例如 `stride` 只提供给有接触且不明显转向的 locomotion，`jump_height` 需要动作标签含 `jump` 且存在腾空区间（循环与否均可），力度参数需要 strike 事件，passive 参数需要对应关节。参数取值范围和完整清单以 `runtime.PARAM_SPECS` 为准。

## 5. UI、覆盖与导出

`python -m motion_edit.ui.serve --packages outputs/edit_packages` 启动本地页面（默认 `127.0.0.1:8770`）。页面可选择 Package，比较原动作和编辑动作，查看骨架或可选蒙皮 mesh、时间轴、接触目标、root motion 和诊断；参数改动通过 `/api/apply` 调用同一个 `EditRuntime`。

页面右栏「部位与接触」显示每个关节的部位、接触和来源（预填的名字 / 继承 / 几何，或物种 / 本包），视口可按部位着色、接触关节画白环。视口单击选关节，Shift+单击选整棵子树，Ctrl+单击增减；色板或按键设部位（`1`–`9` 为 trunk…tail，`0` soft，`F` fin，`H` helper），`C` 切换接触，`M` 把左侧关节复制到镜像关节，`Ctrl+Z` 撤销；「重新预填」把草稿恢复成预填结果。草稿可只应用到当前 Package（`POST /api/parts`），也可写入数据集的 `joint_parts_overrides.json`，作为该物种后续 Profile 构建和分解的默认值。接触区间和地面高度是当前 Package 的编辑。解码设置通过 `/api/load` 重新分解。`PackageStore` 按文件变更重载 Package，并在写入、重建和导出之间使用编辑锁。

导出使用 `python -m motion_edit.apply_edit <clip>.edit --bvh out.bvh` 或 `--glb out.glb`。参数可来自 `--params` JSON、`--set name=value`，或 UI 生成的 sidecar（`params` 加 `events`）。默认 GLB 是骨架动画；`--mesh` 使用 Package 的 T-pose mesh 蒙皮；`--root_motion` 把输出地速积分到根。UI 导出会写 sidecar 并在子进程运行同一 CLI，因此预览与文件共用运行时合成逻辑。

## 6. 代码索引

| 路径 | 主要职责 |
|---|---|
| `motion_edit/profile/`、`build_profiles.py` | 训练动作解码、骨架与关节统计、步态和弹簧拟合、Profile 构建 |
| `motion_edit/contacts.py` | 接触 mask、隐含地速与循环接触区间 |
| `motion_edit/decompose.py`、`decompose_clip.py` | 解码生成特征、分解层与事件、构建及重建 Package |
| `motion_edit/package.py` | Package 格式、版本检查和磁盘读写 |
| `motion_edit/runtime.py`、`ik.py`、`rotations.py` | 参数解析、时间/姿态合成、肢体 IK、旋转工具 |
| `motion_edit/ui/serve.py`、`ui/index.html` | 本地预览、编辑接口、交互页面 |
| `motion_edit/apply_edit.py`、`mesh.py` | BVH/GLB 导出、root motion、蒙皮 mesh |

所有命令行示例均从 `Anytop/` 目录运行。最短的阅读路径是 `decompose_clip.main` → `decompose_motion` / `assemble_package` → `EditRuntime.apply` → `apply_edit.main`；需要理解接触或次级运动时，再进入对应模块。
