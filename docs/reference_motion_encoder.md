# 参考动作编码器：免 retarget 的跨物种动作迁移

> 本文描述参考动作编码器（encode 路径）的当前设计与实现。
> [anytop_model_architecture.md](anytop_model_architecture.md) §9 描述同物种的 img2img / inpainting / outpainting 编辑，
> 与本文的 encode 路径由命令行参数区分、并存。
> 文中不列具体超参与数值；默认值以 `utils/parser_util.py` 和代码为准。

---

## 0. 目标与边界

**目标**：给一段参考动作，加上一副目标骨架，生成「目标骨架在做参考里的动作」。
encode 推理不经过 retarget，也不需要关节对应表；可支持的物种与 body plan 范围由 §6 的参考交换实验确定，不预先承诺任意物种可用。
「人走路的参考 + 四足骨架 → 四足用后腿站起来走」是跨 body plan 的高级验收目标，不是自重建成功后的自然推论。

**同时保持**：
- 纯文本生成（action_label + species_tags）不退化；
- 支持「文字 + 参考」组合生成。

**硬性规则**：
- **合成数据永不作为生成目标。** 所有训练目标都是真实 clip。retarget 之类的合成产物最多只出现在编码器输入侧（§4.3）。

**不追求**：参考的逐关节姿态复现。跨物种时能迁移的是运动内容（§1），不是关节角。

---

## 1. 设计动机：运动与骨架分开提供

借鉴运动与外观分开提供的思路，本方案利用三件事：

1. **自监督重建**：从同一段数据提供运动信号 M 和骨架条件 A，再重建原动作；不要求跨物种动作配对作为生成目标。
2. **M 有容量限制**：降低逐关节复制的便利程度，鼓励模型利用 A 中已有的骨架信息。容量限制本身不能保证身份被剥离。
3. **组合泛化假设**：训练后交换来自不同骨架的 M 和 A，希望运动内容可组合。这必须通过交换实验验证，不能由自重建误差推出。

映射到 AnyTop：

| 视频模型 | AnyTop | 来源 |
|---|---|---|
| 外观 A | 骨架身份：拓扑、骨长比例、关节名、rest pose、structural channel | 目标骨架的现有条件 |
| 运动 M | 运动内容：根轨迹、躯干朝向与直立程度、几条肢体在支撑、落脚相位、节奏、发力时序 | **参考编码器输出 z** |
| 文本 | action_label / species_tags | 现有条件，可选 |

**z 的设计目标是减少「骨架身份」依赖，保留「运动风格」。** 「直立、两肢交替支撑」属于运动内容，正是这里要迁移和验收的东西。

---

## 2. 待验证假设：z 的运动语义能否跨骨架复用

**自重建不能辨识唯一的跨骨架运动表示。** 即使解码器已知骨架，编码器仍可使用骨架专属的坐标表达运动；这不需要额外的 latent 维度。

反例：设运动为 m，骨架 S 对应一个可逆变换 A_S。编码器输出 `z=A_S·m`，解码器根据目标骨架使用 `A_S⁻¹` 恢复 m。同骨架重建完全正确、z 的维度也没有增加，但把源骨架 S 的 z 交给骨架 U 后得到 `A_U⁻¹·A_S·m`，运动语义可能错误。即使只用一维的正负号变换，也存在这个问题。

因此需要分别验证三件事：

- **容量是否合适**：控制逐关节复制的便利程度（§3.1），不作为去身份的证明。
- **形态扰动是否鲁棒**：§4.2 让编码器适应骨长、末端拓扑和名字缺失，但重建同一目标不要求增强前后的 z 相同，也不直接建立跨 body plan 的语义对应。
- **参考交换是否成立**：§6 在未见骨架上固定目标、交换参考，分别验收节奏、姿态和支撑方式，据实际效果确定可支持的迁移范围。

同骨架重建成功但交换失败时，启用 §4.3 的跨骨架输入训练作为补救实验；其效果仍需重新通过交换验收。

---

## 3. 模型结构

### 3.1 运动编码器 E（`model/motion_encoder.py`）

**结构**：
- 输入：只有与源骨架 rest 无关的量（见下文「为什么不读 rest」）。逐帧逐关节读 `ref_pose [J_s, 4, T]`：父→子骨骼单位方向 + 关节绝对高度 / L；根关节额外读 canonical 特征里的根 XZ 速度。逐关节静态读关节名 embedding 与 structural channel 的拓扑列（链上位置、链长、是否叶子、子树大小、兄弟序、兄弟数）加左右侧符号。投影权重独立，不复用 InputProcess。
- 少量层 spatial + temporal attention，复用 `GraphMotionDecoder` 的层实现，宽度小于主干（构造时校验）。spatial attention 不带拓扑 / 边关系偏置（`graph_relations=False`），走普通 SDPA；骨架身份只经关节名、structural channel 的拓扑列与骨骼方向进入 E。
- 每帧用 K 个 learned query 对全部有效关节做 cross-attention 池化，输出 `z [T, K, d_z]`。

E 不读取 species descriptor，也不启用自己的 species FiLM。`ref_pose` 由 `data_loaders/reference.py` 的 `reference_sample` 统一构造，训练 loader 和推理共用这一个函数。根关节和与父关节重合的 helper 关节的骨骼方向为零。

**为什么用骨骼方向和绝对高度**：物理位置位于骨架规范化后的静止朝向系（所有 rest 朝同一方向；逐帧不做 root yaw 去旋转），高度为绝对值，因此一个方向向量在所有骨架上表示同一个世界方向，「脚着地」在所有骨架上都是高度≈0。除以 L 让不同体型的高度可比。

**为什么不读 rest**：position 通道是相对各自 rest 的残差，rotation 是局部旋转，rest token、rest 行和 structural 的几何列（高度、横向 / 前后偏移、附着高度）都描述源骨架的静止姿态。只要 E 能看到源 rest，自重建就奖励「编码相对自己 rest 的偏移」（§2 的 `A_S·m`），解码器再把这个偏移施加到目标 rest 上。实测：人跑参考（T-pose → 跑步是双手大幅内收）驱动猫时，猫同名的 `UpperArm/Forearm/Hand` 前肢照搬内收而左右交叉；人的躯干「保持 rest」，猫的躯干也保持水平。E 只读绝对量后，「前臂朝下」对人和猫是同一个值，从 rest 换算到目标残差的工作只能由解码器按目标 rest 完成。代价是单位方向不含骨骼扭转，同骨架重建会略损失；本方案本来不追求逐关节复现。左右只保留符号，没有名字的关节仍能区分左右而不暴露横向偏移量。

**为什么用 K 个 query 池化**：z 的形状与关节数无关，任何源骨架都映射到同一个空间。cross-limb block 已经用这个结构表达「全身节奏」，被验证过可用。

**为什么输出逐帧 z**：参考和输出处在同一个时间窗口里，帧对齐是恒等映射。逐帧 z 让模型不必学习时间对齐，时间上下文由编码器的 temporal attention 写进每一帧。

**为什么不需要关节顺序增强**：模型里没有关节 index 位置编码（关节身份只来自名字、几何和成对关系），所以「attention + query 池化」天然对关节排列不变。

**容量**：每帧动作输入维度随有效关节数线性增长（按 `InputProcess` 实际读取的通道计，不是 padding 后的 J，也不是存储张量的通道数）。z 的总维度 `K·d_z` 应明显小于小骨架的输入维度，否则起不到压缩作用。选择 `(K, d_z)` 的做法：

- 按训练集与评测集的有效关节数分布，对每个候选报告压缩比 `r=(K·d_z)/D_in` 的分布，并逐骨架列出压缩不明显的骨架；
- 从小规模候选起步，高容量只作对照；
- 选择依据是重建质量与未见骨架交换质量的共同表现，不能只按重建误差选最大容量。

压缩比既不等于信息容量，也不保证解耦：输入本身存在冗余，连续 z 仍可编码身份或骨架专属坐标。

### 3.2 解码器：共享主干 + 参考专用分支

两种模式共享现有 spatial / temporal attention、cross-limb block、FFN、输入输出投影及文字条件模块，全部参与联合训练。底层运动规律继续由主干学习；每层只额外增加一个独立的 `ref cross-attn`，负责读取参考。

- **ref cross-attn**：放在 spatial attention 之后，第 t 帧的每个关节 token 作为 query，读取 z[t] 的 K 个槽，经独立标量 gate 加入残差。按折叠形式参数化：每个头把槽的 key / value 直接映射到 d_model，打分与读取各是一次矩阵乘法，没有 D×D 的 query / output 投影；它能表达每个头秩 ≤ d_z 的全部 query-key 双线性形式。
- 每层的参考分支各自持有参数（含独立归一化）；gate 初始化为零，attention 本身采用普通初始化，避免 gate 与分支输出同时为零。参考分支的残差仅对 `ref_active=True` 的样本生效。

层内顺序为：`共享 spatial → ref cross-attn → 共享 temporal → 共享 cross-limb（若启用）→ 共享 FFN`。原有共享子层及其归一化保持原结构。

**理由**：
- **可学习的读取关系**：query 带着目标关节的名字和几何，可以学习从 z 提取运动信息。但 learned query 槽没有预设的「支撑肢」或「上肢」语义，同骨架训练也不保证槽位在不同 body plan 间一致。后腿能否读取正确的支撑运动，由 §6 的交换实验验收，不以 attention 图作为语义对应成立的证明。
- **为什么放在 spatial 之后**：同一帧内先完成关节间通信，再读参考，之后的 temporal attention 负责把读到的内容沿时间整合。
- **不设参考专用 FFN**：参考信息只能经 cross-attn 进入残差，专用 FFN 读的是隐藏状态，不增加信息。它只对参考样本生效，会学成一个与参考内容无关的「参考模式」偏置；CFG 的 `s_ref·(x0(ref)−x0(∅))` 在 s_ref>1 时会连这个偏置一起放大。非线性解码由共享 FFN 承担。
- **零初始化 gate**：给定相同主干参数，新增分支在训练起点为零贡献。是否真正利用参考，以更换参考后的输出变化判断，gate 大小仅作辅助观察。
- **为什么不复用 cross-limb block**：cross-limb 只在后几层启用，语义是「目标肢体之间交换节奏」。把 z 混进去，两者的作用就无法分开测量，也无法单独关闭。

### 3.3 Null 语义：硬旁路

`ref_active=False` 时，ref cross-attn 的残差置零，不使用 learned null token。混合 batch 按样本屏蔽整个新增残差，包含其 bias；不能只把 z 置零。全 masked 的 attention 用安全的零哨兵避免 NaN。

**理由与边界**：没有参考时，仅使用原有共享路径；参考分支的参数不直接影响该次输出。共享主干仍会同时收到有参考和无参考样本的梯度，因此这能增加容量并隔离专属参数，但不能严格保证纯文本不退化，仍按 §6 做基线对比。

**工程约束**：`ref_feats` / `ref_mask` / `ref_active` 每个 batch 都存在。没有参考时用全 False，不传 None，保持 cond key 集合稳定（compile 契约）。

---

## 4. 训练

### 4.1 目标与 dropout

- **编码器输入**：同一个 clip 的同一时间窗，干净的物理特征（经过 §4.2 的编码器侧增强）按 §3.1 转成编码器输入。**解码器目标**：同一个 clip，按主干自己的 `--leaf_drop_prob` / `--bone_length_aug_prob` 独立抽签增强。
- `--ref_cfg_drop_prob` 对 ref 做硬丢弃。参考生效的行：species FiLM 一律旁路（`species_active=False`），action_label 按 `--ref_label_drop_prob` 丢弃；无参考的行仍按 `--action_label_cfg_drop_prob` 与 `--species_cfg_drop_prob` 抽签。
- 每个 batch 随机保留固定数量（由丢弃概率换算）的参考行，编码器只对这些行前向（形状静态，可编译）；被丢弃的行被硬旁路，其 z 不会被读取。
- 共享主干不冻结、不对参考样本做 stop-gradient：两种模式共同更新共享参数；E、ref cross-attn 及其 gate 仅由参考激活样本训练。

**理由**：
- 训练分布对齐推理：encode 固定关闭 species，所以「参考 + species」不训练；独立抽签时「参考 + species」会占掉大量样本，而 encode 的默认组合（参考、无 label、无 species）占比很小。因此参考生效的行强制旁路 species，并让 label 按独立概率丢弃。「只有参考」「参考 + 文字」「只有文字」「无条件」四种组合仍都被训练，推理时任选。
- 自重建中参考、label 与 descriptor 来自同一 clip，训练时并无冲突；旁路 species 也不阻止 z 编码源骨架身份（E 本来不读 descriptor），它只让 body plan / 步态由 z 与目标骨架结构提供。
- ref 丢弃概率不能太低。z 携带的信息远比 label 丰富，有 z 时 label 通路几乎拿不到梯度。label 通路要靠 z 缺席的样本继续学习，纯文本质量取决于此。

### 4.2 骨架增强（编码器输入与目标各自独立抽签）

| 增强 | 现有实现 | 打破的是什么 |
|---|---|---|
| leaf-drop | `truebones_utils/leaf_drop.py` | 扰动拓扑末端，鼓励减少对精确关节数和末端链长的依赖 |
| 骨长比例 | `truebones_utils/bone_length_aug.py` | 扰动身体比例，鼓励减少对源骨长的依赖 |
| 整关节名置空 | 与 `joint_name_drop_prob` 同一机制 | 隐藏部分名字，鼓励结合几何和运动识别部位 |

编码器侧的概率由 `--ref_leaf_drop_prob`、`--ref_bone_length_aug_prob` 等参数控制，与目标侧互相独立。

**理由与限制**：这些增强保留时间对应，鼓励编码器从变化后的输入恢复原始运动；它们提供的是形态扰动鲁棒性，不强制 z 不变，也不建立不同 body plan 的对应。骨长增强保留局部旋转，但会改变末端位置、速度和可能的接触状态；leaf-drop 也会丢失被删末端的运动。因此必须控制强度，并比较增强前后的动作内容与交换结果，不能假定运动信息完全无损。

**目标侧也做**：主干在非参考训练里本来就用 leaf-drop 和骨长增强；参考训练沿用同一组参数，两侧抽签互相独立。于是源和目标的形态差异两个方向都有：参考关节比目标少（编码器侧 leaf-drop），也有参考关节比目标多（目标侧 leaf-drop），后者对应推理时源骨架比目标多出尾巴、手指的情形。

**实现要点**：
- 目标侧增强在时间采样前作用于存储 clip（与非参考训练完全相同）；编码器输入从存储骨架出发，复用目标的 closing-key 判定和同一时间窗，再做自己的增强。
- 编码器侧没有增强生效时，直接复用目标的 canonical 特征与 structural channel（`reference_sample` 可接收已算好的这两项），只有 leaf-drop 或骨长增强生效时才重算。
- 参考可能比目标宽，collate 按两者中较宽的关节数选 joint bucket；存储骨架的关节数决定 sampler 的 bucket，所以仍落在同一 bucket 里。

**不做时间增强**：编码器输入与目标必须逐帧对齐（§3.1）。

### 4.3 条件阶段：retarget 作为编码器输入

**启用条件**：同骨架重建可用，但 §6 的跨骨架参考交换失败。失败既包括不响应 z，也包括响应了 z 却迁移错姿态、肢体角色或支撑相位；不要求同 body plan 已成功才允许做此实验。

**做法**：离线把真实 clip（物种 A）retarget 到随机的物种 B，作为**编码器输入**。解码器骨架和目标仍是 A 的原始 clip。

**理由与限制**：§4.2 不改变 body plan；本步骤给出跨 body plan 的关联输入，尝试减少骨架专属编码。retarget 的瑕疵只出现在输入侧，不违反硬性规则，但也可能丢掉直立姿态、支撑方式等关键信息。此时多个不同原动作可能得到相似输入，不能要求编码器从中恢复已丢失的内容。该训练不是跨物种迁移成功的保证，也不能替代真实桥接样本。

**过滤**：除倒地、翻转外，还要检查待迁移的姿态、节奏、支撑数量与相位是否保留；无法可靠判断的样本人工抽查。保留原始输入训练样本，单独报告加入跨骨架输入前后的交换结果，避免编码器转而依赖 retarget 的固定伪影。

### 4.4 检查点

- 参数包括编码器（含骨骼方向投影）、每层 ref cross-attn、独立归一化及 gate；训练启动时打印完整参数预算。
- 这些参数写入 state_dict，由 `CKPT_VERSION` 区分是否兼容；不对旧 checkpoint warm start，从头训练。
- 验证同时计算同噪声、同 timestep 的 ref-null 与 reference 两种 loss：前者保留原 `Val/*` 名称，后者为 `Val/reference_*`。

### 4.5 训练入口

`train/train_anytop.py` 加 `--ref_encoder` 启用；同时需要 `--species_joint_cond`（§5.4）。编码器规模由 `--ref_slots`、`--ref_dim`、`--ref_width`、`--ref_layers` 控制，ref 丢弃由 `--ref_cfg_drop_prob`、`--ref_label_drop_prob` 控制，编码器侧增强由 `--ref_*` 系列增强参数控制。

---

## 5. 推理

### 5.1 输入

- 参考在**它自己的骨架上**转成特征，不 retarget。
  - npy 来源：现成可用；`--reference_source_type` 可显式指定 NPY 源骨架，encode 不会静默把未知源当成目标骨架；
  - 原始 FBX/GLB/GLTF：复用预处理的根检测、wrapper 折叠、姿态对齐与特征提取，使用 checkpoint 的全局统计和关节名向量缓存，未知关节名置空，不加载 T5。
- 帧长：沿用现有 reference 的「重采样到窗口」规则。

### 5.2 参数
- 无 `--reference_mode`：带 `--skip_timesteps` / `--inpaint_*` / `--outpaint` 的 `--reference_motion` 走同物种编辑，否则走 encode。
- `--ref_cfg_scale` 控制参考引导强度，只作用于 `t >= T/2` 的高噪声步（躯干朝向、身高等全局姿态在这里定下）；低噪声细节步固定为 1（`ClassifierFreeReferenceModel.REF_CFG_SWITCH` / `REF_CFG_LATE_SCALE`）。这样能拿到高 scale 的姿态，细节步不带高 scale 的伪影；无 label 时后段只需一次前向。

### 5.3 CFG（组合式，x0 上的代数与现有实现相同）

```
out = x0(∅, ∅)
    + s_ref · (x0(ref, ∅)   − x0(∅, ∅))
    + s_txt · (x0(ref, txt) − x0(ref, ∅))
```

- 不给 label 时：去掉第三项，通常两次前向；`ref_cfg_scale=1` 时仅计算参考分支，一次前向；为 0 时仅计算无条件分支。
- 给 label 时：按 `(1-s_ref)·base + (s_ref-s_text)·ref + s_text·full` 跳过系数为零的分支，其余顺序执行，最多三次前向。两个 scale 都为 1 时仅计算 full，一次前向。
- 实现为一个独立 wrapper（`model/cfg_sampler.py`），不与现有 action / species wrapper 嵌套，避免前向次数相乘。

### 5.4 species 调制

**训练时开启 species conditioning，干净迁移推理时关闭 species 调制。** 两个阶段使用不同开关：

- **训练**：传入 `--species_joint_cond`，构造并训练 species FiLM；保持 `species_cfg_drop_prob > 0`，让模型同时学习调制开启和硬旁路两种状态。参考生效的行由模型强制旁路 species FiLM（§4.1），species FiLM 只由无参考的行训练。
- **推理**：加载上述 checkpoint，保留其 `species_joint_cond=True`；encode（不带编辑参数的 `--reference_motion`）固定关闭 species FiLM，让参考提供运动内容。encode 与 `--species_active` 参数互斥，传入 true 或 false 都会报错；无需另传关闭参数。不要把 checkpoint 的 `species_joint_cond` 改为 False。
- encode 默认不给 action_label；编辑与无参考生成默认 `species_active=True`，可用 `--species_active true/false` 控制。

**契约**：唯一的 species FiLM 是 `InputProcess.species_film_j`。它读取关节名 embedding 与 species descriptor 的拼接；`species_joint_cond` 控制是否构造/启用模块，`species_active` 控制每个样本是否应用整个调制。`species_active=False` 会旁路全部 per-joint 调制，保留原始关节名、rest pose 和 structural channel。

**依据与边界**：目标 descriptor 包含 body plan 和步态信息，可能与参考中的直立姿态、支撑方式等运动内容冲突。关闭是减少条件冲突的实验起点，尚未证明迁移质量更好。关闭 FiLM 不会消除关节名、几何及共享主干已有的运动先验。E 的输入不读取 descriptor。未来若做开关对照，应使用独立实验入口。

**CFG 行为**：encode 的所有 reference CFG 分支固定关闭 species，不嵌套 species guidance；其它模式下 `--species_active false` 时同样忽略 `species_cfg_scale`，不构造 species CFG wrapper，不检查 species CFG 的训练前提。关闭 species 不改变 reference/text CFG 的前向次数。

---

## 6. 评测与验收

| 项 | 方法 | 回答的问题 |
|---|---|---|
| 纯文本不退化 | ref-null 模式下，与基线比较 MotionQualityScorer 和 val l_simple | 联合训练后纯文本质量是否保持 |
| 同物种重建 | val clip 自编码自重建，误差对比 ref-null | z 是否携带了运动 |
| z 是否被使用 | 每层 ref gate 的大小；固定目标和随机种子，换不同参考完整生成，看输出是否相应变化 | 解码器有没有走「无视 z」的捷径 |
| 跨物种迁移 | 固定一组评测对（双足→四足、四足→不同体型的四足、飞行→飞行）。比较源与输出之间骨架无关的量：按 L 归一化的根速度曲线、躯干俯仰（直立程度）、全身运动能量随时间的曲线。同时用 scorer 打合理性分，并人工看姿态、支撑方式和脚滑 | 迁移了多少运动内容，结果是否合理 |

跨物种没有 GT，所以只比较骨架无关的量，不比较关节角。选少量固定评测对，包含未见骨架，与 ref-null 对照；同骨架重建好不等于跨骨架迁移成功。

固定评测对由 [tiny_tasks.json](../eval/tiny_tasks.json) 中的 `ReferenceMotion` 任务提供（同骨架、犬→马、鸟→其他物种），目前用于 encode 路径的工程冒烟检查。真正的未见骨架验收需要独立的 val 数据来源，训练骨架不能当作未见骨架。

### 已知限制

- **幅度被目标骨架的数据分布约束。** 跨骨架时参考的方向可以迁移，但幅度会被压回目标骨架自己的数据分布（例如猫的尾巴大幅上翘，迁到训练里尾巴很少上翘的犬骨架时，仰角明显低于参考）。这是数据覆盖问题，不是参考通道没有信息。
- **直立度会被长附属链稀释。** 尾巴、翅膀、披风、武器等长链按 rest 长度计入质量分布，同类动作的直立度在不同 rig 间会有明显散布；蛇、鱼、蜘蛛等奇异骨架的数值合理，但游泳 / 飞行 clip 的最低关节可低于地面，所以最低高度被截断、loss 用 Huber。
- **容量不是首要瓶颈。** 诊断显示 z 的信息量足以线性回归出尾巴、头颈等部位的运动；有参考相对无参考的去噪增益主要随训练步数上升，而不是随 z 容量上升。

---

## 7. 明确不做

| 不做 | 原因 |
|---|---|
| 对 species_tags 做对抗约束 | tags 描述的 body plan 和步态属于运动内容，剥掉它们就剥掉了要迁移的东西 |
| 把 z 拆成两层 | 先验证单层表示的容量与交换能力；没有证据保证一层足够，也没有证据说明拆层能解决 §2 的不可辨识问题 |
| 合成数据作为目标 | 扩散模型会把 retarget 的瑕疵当作合理分布学进去（硬性规则） |
| 每帧 z 带额外的时间 PE 或对齐模块 | 窗口内帧对齐是恒等映射（§3.1） |
| 编码器侧做关节顺序增强 | 编码器结构上对关节排列不变（§3.1） |
| 参考专用 FFN | 只带来与参考内容无关的模式偏置，且被 CFG 放大（§3.2） |
| 跨物种 retarget 的编辑推理 | img2img / inpaint / outpaint 限定同物种，无需 retarget；跨物种由 encode 承担 |

---

## 8. 代码入口

| 文件 | 内容 |
|---|---|
| `model/motion_encoder.py` | 编码器 E |
| `model/motion_transformer.py` | decoder layer 中的 ref cross-attn、独立 LayerNorm 与零初始化 gate |
| `model/anytop.py` | 调用 E；解析 `ref_active`；硬旁路 |
| `model/cfg_sampler.py` | 组合式 ref/text CFG wrapper |
| `data_loaders/reference.py`、`data_loaders/truebones/data/dataset.py` + collate | 编码器侧增强副本、`ref_pose`（骨骼方向 + 高度）、`ref_*` 字段、ref dropout |
| `utils/parser_util.py` | 训练侧 `--ref_*` 参数；推理侧 `--ref_cfg_scale` |
| `sample/generate.py`、`sample/reference_motion.py` | encode 模式：源骨架上的特征 → E → z |
| `eval/` | 跨物种评测对 |

## 9. 验证流程

1. 小规模训练，扫描容量，按 §6 检查参考控制和跨物种效果，并对比 per-joint species FiLM 开关。
2. 视结果决定是否启用 §4.3：同骨架可控但跨骨架不响应或语义错误时均可启用，再用相同评测对检查效果。
