# 参考动作编码器：免 retarget 的跨物种动作迁移

> 状态：第三版已实现（训练性能优化，CKPT 33），需从头重训；首版记录见 §11，第二版改动见 §12，第三版见 §13。
> [anytop_model_architecture.md](anytop_model_architecture.md) §9 保留同物种的 img2img / inpainting / outpainting 编辑，
> 本文的 encode 路径与其并存。§10 是实现前的方案核验记录。

---

## 0. 目标与边界

**研究目标**：给一段参考动作，加上一副目标骨架，生成「目标骨架在做参考里的动作」。
encode 推理不经过 retarget，也不需要关节对应表；可支持的物种与 body plan 范围由 §6 的参考交换实验确定，不预先承诺任意物种可用。
「人走路的参考 + 四足骨架 → 四足用后腿站起来走」是跨 body plan 的高级验收目标，不是自重建成功后的自然推论。

**同时保持**：
- 现有的纯文本生成（action_label + species_tags）不退化；
- 支持「文字 + 参考」组合生成。

**硬性规则**：
- **合成数据永不作为生成目标。** 所有训练目标都是真实 clip。retarget 之类的合成产物最多只出现在编码器输入侧（§4.2）。

**不追求**：参考的逐关节姿态复现。跨物种时能迁移的是运动内容（§1），不是关节角。

---

## 1. 设计动机：运动与骨架分开提供

借鉴运动与外观分开提供的思路，本方案希望利用三件事：

1. **自监督重建**：从同一段数据提供运动信号 M 和骨架条件 A，再重建原动作；不要求跨物种动作配对作为生成目标。
2. **M 有容量限制**：降低逐关节复制的便利程度，鼓励模型利用 A 中已有的骨架信息。容量限制本身不能保证身份被剥离。
3. **组合泛化假设**：训练后交换来自不同骨架的 M 和 A，希望运动内容可组合。这必须通过交换实验验证，不能由自重建误差推出。

映射到 AnyTop：

| 视频模型 | AnyTop | 来源 |
|---|---|---|
| 外观 A | 骨架身份：拓扑、骨长比例、关节名、rest pose、structural channel | 目标骨架的现有条件，不改 |
| 运动 M | 运动内容：根轨迹、躯干朝向与直立程度、几条肢体在支撑、落脚相位、节奏、发力时序 | **新增的 z** |
| 文本 | action_label / species_tags | 现有条件，可选 |

**z 的设计目标是减少「骨架身份」依赖，保留「运动风格」。** 「直立、两肢交替支撑」属于运动内容，正是这里要迁移和验收的东西。

---

## 2. 待验证假设：z 的运动语义能否跨骨架复用

**自重建不能辨识唯一的跨骨架运动表示。** 即使解码器已知骨架，编码器仍可使用骨架专属的坐标表达运动；这不需要额外的 latent 维度。

反例：设运动为 m，骨架 S 对应一个可逆变换 A_S。编码器输出 `z=A_S·m`，解码器根据目标骨架使用 `A_S⁻¹` 恢复 m。同骨架重建完全正确、z 的维度也没有增加，但把源骨架 S 的 z 交给骨架 U 后得到 `A_U⁻¹·A_S·m`，运动语义可能错误。即使只用一维的正负号变换，也存在这个问题。

因此需要分别验证三件事：

- **容量是否合适**：§3.1 的容量扫描控制逐关节复制的便利程度，不作为去身份的证明。
- **形态扰动是否鲁棒**：§4.2 让编码器适应骨长、末端拓扑和名字缺失，但重建同一目标不要求增强前后的 z 相同，也不直接建立跨 body plan 的语义对应。
- **参考交换是否成立**：§6 在未见骨架上固定目标、交换参考，分别验收节奏、姿态和支撑方式。据实际效果确定可支持的迁移范围。

同骨架重建成功但交换失败时，启用 §4.3 的跨骨架输入训练作为补救实验；其效果仍需重新通过交换验收。

---

## 3. 模型改动

### 3.1 运动编码器 E（新模块 `model/motion_encoder.py`）

**结构**：
- 输入：参考 clip 的干净特征 x0 `[J_s, F, T]`，以及参考骨架自己的 rest token、关节名 embedding、structural channel。复用 InputProcess 的同类投影，权重独立。
- 少量层 spatial + temporal attention，复用 `GraphMotionDecoder` 的层实现，宽度小于主干。spatial attention 不带拓扑 / 边关系偏置（`graph_relations=False`），走普通 SDPA；骨架身份只经关节名、rest 几何、structural channel 与骨骼方向进入 E。
- 每帧用 K 个 learned query 对全部有效关节做 cross-attention 池化，输出 `z [T, K, d_z]`。

沿用当前模型通道契约：普通关节读取 position + rotation 共 9 维，仅 translation root 额外读取 XZ velocity 两维；不把派生速度当作额外独立输入。E 不读取 species descriptor，也不启用自己的 species FiLM。

此外，E 读取每个关节在骨架规范化静止朝向系下的父→子骨骼单位方向 `ref_bone_dir [J_s, 3, T+1]`：第 0 行是 rest pose，其余行逐帧对应动作。它经独立线性投影后加到同一 token 上。根关节和与父关节重合的 helper 关节为零。该通道由 `data_loaders/reference.py` 的 `reference_sample` 统一构造，训练 loader 和推理共用这一个函数。

**为什么加骨骼方向**：position 通道是相对各自 rest 的残差，rotation 是局部旋转，同样是「尾巴朝上」，在不同骨架上数值不同。物理位置位于骨架规范化后的静止朝向系（所有 rest 朝 +Z；逐帧不做 root yaw 去旋转），高度为绝对值，因此一个方向向量在所有骨架上表示同一个世界方向。rest 行让 E 能对照源骨架自己的静止姿态。

**理由**：
- **K 个 query 池化**让 z 的形状与关节数无关，任何源骨架都映射到同一个空间。cross-limb block 已经用这个结构表达「全身节奏」，被验证过可用。
- **为什么输出逐帧 z**：参考和输出处在同一个时间窗口里，帧对齐是恒等映射。逐帧 z 让模型不必学习时间对齐。时间上下文由编码器的 temporal attention 写进每一帧。
- **为什么不需要关节顺序增强**：模型里没有关节 index 位置编码（关节身份只来自名字、几何和成对关系），所以「attention + query 池化」天然对关节排列不变。

**容量扫描**：按 `InputProcess` 的实际读取通道计算，每帧动作输入为 `D_in=9·J_valid+2`（见 `canonical_features.py` 的 `JOINT_MODEL_CHANNELS` / `ROOT_MODEL_CHANNELS`），不使用 padding 后的 J，也不使用存储张量的 `12·J`。例如 20 关节只有 182 维，原建议 `K=8, d_z=32` 的 256 维已超过输入。

动手前先按训练集与评测集的有效关节数统计 min / P10 / P50 / P90 / max，并报告每个候选的 `r=(K·d_z)/D_in` 分布。小规模起点为 `(K,d_z)=(2,8)、(4,8)、(4,16)`，分别为 16 / 32 / 64 维；`(8,32)` 只保留为高容量对照，不作为默认值。若最小相关骨架上仍没有明显压缩，继续缩小候选或明确标记该骨架不满足压缩假设。

上述数值是实验起点，不是已验证配置。维度比既不等于信息容量，也不保证解耦：输入本身存在冗余，连续 z 仍可编码身份或骨架专属坐标。选择依据是重建质量与未见骨架交换质量的共同表现，不能只按重建误差选最大容量。

### 3.2 解码器：共享主干 + 参考专用分支

两种模式共享现有 spatial / temporal attention、cross-limb block、FFN、输入输出投影及文字条件模块，全部参与联合训练。底层运动规律继续由主干学习；每层只额外增加一个独立的 `ref cross-attn`，负责读取参考。

- **ref cross-attn**：放在 spatial attention 之后，第 t 帧的每个关节 token 作为 query，读取 z[t] 的 K 个槽，经独立标量 gate 加入残差。按折叠形式参数化：每个头把槽的 key / value 直接映射到 d_model，打分是一次 `[D, H·K]` 乘法、读取是一次 `[H·K, D]` 乘法，没有 D×D 的 query / output 投影；它能表达每个头秩 ≤ d_z 的全部 query-key 双线性形式。
- 每层的参考分支各自持有参数；gate 初始化为零，attention 本身采用普通初始化，避免 gate 与分支输出同时为零。参考分支的残差仅对 `ref_active=True` 的样本生效。

层内顺序为：`共享 spatial → ref cross-attn → 共享 temporal → 共享 cross-limb（若启用）→ 共享 FFN`。原有共享子层及其归一化保持原结构。

**理由**：
- **可学习的读取关系**：query 带着目标关节的名字和几何，可以学习从 z 提取运动信息。但 learned query 槽没有预设的「支撑肢」或「上肢」语义，同骨架训练也不保证槽位在不同 body plan 间一致。后腿能否读取正确的支撑运动，由 §6 的交换实验验收，不以 attention 图作为语义对应成立的证明。
- **为什么放在 spatial 之后**：同一帧内先完成关节间通信，再读参考，之后的 temporal attention 负责把读到的内容沿时间整合。
- **不设参考专用 FFN**：参考信息只能经 cross-attn 进入残差，专用 FFN 读的是隐藏状态，不增加信息。它只对参考样本生效，会学成一个与参考内容无关的「参考模式」偏置；CFG 的 `s_ref·(x0(ref)−x0(∅))` 在 s_ref>1 时会连这个偏置一起放大。非线性解码由共享 FFN 承担。
- **零初始化 gate**：给定相同主干参数，新增分支在训练起点为零贡献。是否真正利用参考，以更换参考后的输出变化判断，gate 大小仅作辅助观察。
- **为什么不复用 cross-limb block**：cross-limb 只在后几层启用，语义是「目标肢体之间交换节奏」。把 z 混进去，两者的作用就无法分开测量，也无法单独关闭。

### 3.3 Null 语义：硬旁路

`ref_active=False` 时，ref cross-attn 的残差置零，不使用 learned null token。混合 batch 按样本屏蔽整个新增残差，包含其 bias；不能只把 z 置零。

**理由与边界**：没有参考时，仅使用原有共享路径；参考分支的参数不直接影响该次输出。共享主干仍会同时收到有参考和无参考样本的梯度，因此这能增加容量并隔离专属参数，但不能严格保证纯文本不退化，仍按 §6 做基线对比。

**工程约束**：`ref_feats` / `ref_mask` / `ref_active` 每个 batch 都存在。没有参考时用全 False，不传 None，保持 cond key 集合稳定（compile 契约）。

---

## 4. 训练

### 4.1 目标与 dropout

- **编码器输入**：同一个 clip，干净的 x0（经过 §4.2 的增强）。**解码器目标**：同一个 clip 的原始版本。
- 新增 `--ref_cfg_drop_prob`，对 ref 做硬丢弃，建议 0.3–0.5。与现有的 `--action_label_cfg_drop_prob` 和 `--species_cfg_drop_prob` 各自独立抽签。每个 batch 随机保留固定的 `B − round(B·p)` 行，编码器只对这些行前向（形状静态，可编译）；被丢弃的行被硬旁路，其 z 不会被读取。
- 共享主干不冻结、不对参考样本做 stop-gradient：两种模式共同更新共享参数；E、ref cross-attn 及其 gate 仅由参考激活样本训练。ref 丢弃时旁路参考子层。

**理由**：
- 独立 dropout 让「只有参考」「参考 + 文字」「只有文字」「无条件」四种组合都成为训练过的模式，推理时任选。
- ref drop 不能太低。z 携带的信息远比 label 丰富，有 z 时 label 通路几乎拿不到梯度。label 通路要靠 z 缺席的样本继续学习，纯文本质量取决于此。

### 4.2 编码器侧增强（只作用于编码器输入，目标永不改变）

| 增强 | 现有实现 | 打破的是什么 |
|---|---|---|
| leaf-drop | `truebones_utils/leaf_drop.py` | 扰动拓扑末端，鼓励减少对精确关节数和末端链长的依赖 |
| 骨长比例 | `truebones_utils/bone_length_aug.py` | 扰动身体比例，鼓励减少对源骨长的依赖 |
| 整关节名置空 | 与 `joint_name_drop_prob` 同一机制 | 隐藏部分名字，鼓励结合几何和运动识别部位 |

**理由与限制**：这些增强保留时间对应，鼓励编码器从变化后的输入恢复原始运动；它们提供的是形态扰动鲁棒性，不强制 z 不变，也不建立不同 body plan 的对应。骨长增强保留局部旋转，但会改变末端位置、速度和可能的接触状态；leaf-drop 也会丢失被删末端的运动。因此必须控制强度，并比较增强前后的动作内容与交换结果，不能假定运动信息完全无损。

**实现要点**：loader 为编码器单独生成一份增强副本（独立的 cond 切片），解码器侧的 cond 和目标不受影响。leaf-drop 只删不增，副本的关节数不超过原骨架，可以落在同一个 joint bucket 里。

**不做时间增强**：编码器输入与目标必须逐帧对齐（§3.1）。

### 4.3 条件阶段：retarget 作为编码器输入

**启用条件**：同骨架重建可用，但 §6 的跨骨架参考交换失败。失败既包括不响应 z，也包括响应了 z 却迁移错姿态、肢体角色或支撑相位；不要求同 body plan 已成功才允许做此实验。

**做法**：离线把真实 clip（物种 A）retarget 到随机的物种 B，作为**编码器输入**。解码器骨架和目标仍是 A 的原始 clip。

**理由与限制**：§4.2 不改变 body plan；本步骤给出跨 body plan 的关联输入，尝试减少骨架专属编码。retarget 的瑕疵只出现在输入侧，不违反硬性规则，但也可能丢掉直立姿态、支撑方式等关键信息。此时多个不同原动作可能得到相似输入，不能要求编码器从中恢复已丢失的内容。该训练不是跨物种迁移成功的保证，也不能替代真实桥接样本。

**过滤**：除倒地、翻转外，还要检查待迁移的姿态、节奏、支撑数量与相位是否保留；无法可靠判断的样本人工抽查。保留原始输入训练样本，单独报告加入跨骨架输入前后的交换结果，避免编码器转而依赖 retarget 的固定伪影。

### 4.4 参数与版本

- 新增参数包括编码器（含骨骼方向投影）、每层 ref cross-attn、独立归一化及 gate。实现时汇总完整参数预算。
- state_dict 新增键，CKPT 版本 +1。
- 从头训练，不 warm start。

---

## 5. 推理

### 5.1 输入
- 参考：在**它自己的骨架上**转成特征，不 retarget。
  - npy 来源：现成可用；
  - 原始文件：走新骨架预处理生成 cond 和特征。
- 帧长：沿用现有 reference 的「重采样到窗口」规则。

### 5.2 新参数
- 无 `--reference_mode`：带 `--skip_timesteps` / `--inpaint_*` / `--outpaint` 的 `--reference_motion` 走同物种编辑，否则走 encode。
- `--ref_cfg_scale`。

### 5.3 CFG（组合式，x0 上的代数与现有实现相同）

```
out = x0(∅, ∅)
    + s_ref · (x0(ref, ∅)   − x0(∅, ∅))
    + s_txt · (x0(ref, txt) − x0(ref, ∅))
```

- 不给 label 时：去掉第三项，通常两次前向；`ref_cfg_scale=1` 时仅计算参考分支，一次前向；为 0 时仅计算无条件分支。
- 给 label 时：按 `(1-s_ref)·base + (s_ref-s_text)·ref + s_text·full` 跳过系数为零的分支，其余顺序执行，最多三次前向。两个 scale 都为 1 时仅计算 full，一次前向。
- 实现为一个新 wrapper，不与现有 action / species wrapper 嵌套，避免前向次数相乘。

### 5.4 推荐默认（干净迁移）

**训练时开启 species conditioning，干净迁移推理时关闭 species 调制。** 两个阶段使用不同开关：

- **训练**：传入 `--species_joint_cond`，构造并训练 species FiLM；保持 `species_cfg_drop_prob > 0`（默认 0.15），让模型同时学习调制开启和硬旁路两种状态。
- **推理**：加载上述 checkpoint，保留其 `species_joint_cond=True`；encode（不带编辑参数的 `--reference_motion`）固定关闭 species FiLM，让参考提供运动内容。encode 与 `--species_active` 参数互斥，传入 true 或 false 都会报错；无需另传关闭参数。不要把 checkpoint 的 `species_joint_cond` 改为 False。
- action_label 不给；
- 编辑与无参考生成仍默认 `species_active=True`，可用 `--species_active true/false` 控制。

**当前实现契约**：全局 species FiLM 已移除，唯一的 species FiLM 是 `InputProcess.species_film_j`。它读取关节名 embedding 与 species descriptor 的拼接；`species_joint_cond` 控制是否构造/启用模块，`species_active` 控制每个样本是否应用整个调制。`species_active=False` 会旁路全部 per-joint 调制，保留原始关节名、rest pose 和 structural channel；不存在「关闭全局 FiLM，同时保持 per-joint FiLM」这一组合。

**推荐依据与验证边界**：目标 descriptor 包含 body plan 和步态信息，可能与参考中的直立姿态、支撑方式等运动内容冲突。关闭是减少条件冲突的实验起点，尚未证明迁移质量更好。encode 的正式推理入口固定关闭；未来若做开关对照，应使用独立实验入口。旁路应由非零 `species_cfg_drop_prob` 的训练覆盖。关闭 FiLM 不会消除关节名、几何及共享主干已有的运动先验。参考编码器 E 的输入仍按 §3.1，不读取 descriptor。

**CFG 行为**：encode 的所有 reference CFG 分支固定关闭 species，不嵌套 species guidance。encode 模式或其它模式下 `--species_active false` 时忽略 `species_cfg_scale`，不构造 species CFG wrapper，不增加 species 前向，也不检查 species CFG 的训练前提。reference/text CFG 按上述分支系数执行 1–3 次前向，关闭 species 不改变其次数。

---

## 6. 评测与验收

| 项 | 方法 | 回答的问题 |
|---|---|---|
| 纯文本不退化 | ref-null 模式下，与基线比较 MotionQualityScorer 和 val l_simple | 联合训练后纯文本质量是否保持 |
| 同物种重建 | val clip 自编码自重建，误差对比 ref-null | z 是否携带了运动 |
| z 是否被使用 | 每层 ref gate 的大小；固定目标和随机种子，换不同参考完整生成，看输出是否相应变化 | 解码器有没有走「无视 z」的捷径 |
| 跨物种迁移 | 固定一组评测对（双足→四足、四足→不同体型的四足、飞行→飞行）。比较源与输出之间骨架无关的量：按 L 归一化的根速度曲线、躯干俯仰（直立程度）、全身运动能量随时间的曲线。同时用 scorer 打合理性分，并人工看姿态、支撑方式和脚滑 | 迁移了多少运动内容，结果是否合理 |

跨物种没有 GT，所以只比较骨架无关的量，不比较关节角。选少量固定评测对，包含未见骨架，与 ref-null 对照；同骨架重建好不等于跨骨架迁移成功。


---

## 7. 明确不做

| 不做 | 原因 |
|---|---|
| 对 species_tags 做对抗约束 | tags 描述的 body plan 和步态属于运动内容，剥掉它们就剥掉了要迁移的东西 |
| 首版把 z 拆成两层 | 先验证单层表示的容量与交换能力；当前没有证据保证一层足够，也没有证据说明拆层能解决 §2 的不可辨识问题 |
| 合成数据作为目标 | 扩散模型会把 retarget 的瑕疵当作合理分布学进去（硬性规则） |
| 每帧 z 带额外的时间 PE 或对齐模块 | 窗口内帧对齐是恒等映射（§3.1） |
| 编码器侧做关节顺序增强 | 编码器结构上对关节排列不变（§3.1） |
| 保留 img2img / inpaint / outpaint | 限定同物种后无需 retarget，与 encode 由参数区分；跨物种 retarget 推理路径已删除 |

---

## 8. 改动清单

| 文件 | 改动 |
|---|---|
| `model/motion_encoder.py` | 新增：编码器 E |
| `model/motion_transformer.py` | decoder layer 加入 ref cross-attn、独立 LayerNorm 与零初始化 gate |
| `model/anytop.py` | 调用 E；解析 `ref_active`；硬旁路；CKPT 版本 +1 |
| `model/cfg_sampler.py` | 组合式 ref/text CFG wrapper |
| `data_loaders/reference.py`、`data_loaders/truebones/data/dataset.py` + collate | 编码器侧增强副本、骨骼方向、`ref_*` 字段、ref dropout |
| `utils/parser_util.py` | `--ref_encoder`、`--ref_slots`、`--ref_dim`、`--ref_width`、`--ref_layers`、`--ref_cfg_drop_prob`、编码器侧增强概率；推理侧的 `--ref_cfg_scale` |
| `sample/generate.py`、`sample/reference_motion.py` | encode 模式：源骨架上的特征 → E → z |
| `eval/` | 跨物种评测对 |

## 9. 实施顺序

1. **§3 + §4.1 + §4.2 小规模训练**。扫描容量，按 §6 检查参考控制和跨物种效果，并对比 per-joint species FiLM 开关。
2. **视结果决定是否启用 §4.3**：同骨架可控但跨骨架不响应或语义错误时均可启用，再用相同评测对检查效果。

## 10. 本轮 P1 核验记录（2026-10-18）

- **表示假设**：§2 给出同维可逆变换反例；一维正负号版本可实现零同骨架重建误差而跨骨架交换错误。修复的是不成立的推论与实验推进规则，尚未证明实际模型能跨物种迁移。
- **实际输入容量**：核对 `canonical_features.py` 的 `JOINT_MODEL_CHANNELS=0..8`、`ROOT_VEL_CHANNELS=(9,11)` 及 `InputProcess.forward` 的切片，确认每帧读取 `9·J_valid+2` 维。20 关节输入 182 维，256 维 z 的比值约 1.407，原起点不满足明显压缩。
- **species 契约**：核对 `AnyTop.forward` 与 `InputProcess.forward`，仅存在 per-joint species FiLM；运行 `python -m unittest discover -s tests -p test_species_cond_hybrid.py -v`，13 项全部通过，覆盖整个调制旁路、训练 dropout 和 species CFG。§5.4 已按此修正，未新增或恢复全局 FiLM。
- **验证边界**：本轮为方案与现有实现契约核验，未实现 E、未运行容量扫描或迁移训练；数据分布与跨骨架迁移效果仍待实施阶段取得。

## 11. 首版实施记录（2026-10-18）

### 已实现

- `MotionEncoder` 独立投影、较窄的 spatial/temporal decoder、按有效关节池化的 K 个 query。
  默认 `ref_width=64`、`ref_layers=2`、`ref_slots=2`、`ref_dim=8`；编码器宽度必须小于共享主干。
- 每层独立的 reference cross-attention、Pre-LN FFN（默认 256 宽）和各自的零初始化 gate。
  `ref_active=False` 屏蔽两条完整残差；null batch 使用安全的零哨兵避免全 masked attention 产生 NaN。
- loader 在完成时间采样后复制源数据，仅增强编码器侧；目标及目标 cond 保持原样。
  默认编码器 leaf-drop / 骨长增强概率为 0.2 / 0.2，骨长幅度 0.1，整关节名丢弃概率 0.15。
  reference 训练拒绝原有 decoder 侧 leaf-drop / 骨长增强参数，防止合成形态进入目标。
  所有 collate batch 都有固定的 `ref_*` tensor 字段；参考 CFG dropout 默认 0.4，在模型内独立抽样。
- NPY 按源 cond 编码；原始 FBX/GLB/GLTF 复用预处理的根检测、wrapper 折叠、姿态对齐与特征提取，
  使用 checkpoint 的全局统计和关节名向量缓存，未知关节名置空，不加载 T5，不经过 retarget。
  `--reference_source_type` 可显式指定 NPY 源骨架；encode 不会静默把未知源当成目标骨架。
- encode 使用独立的顺序 CFG，跳过系数为零的分支，species 状态固定。
  encode 固定关闭目标 species FiLM，且与 `--species_active` 参数互斥；其它模式默认开启，可用 `--species_active true/false` 控制。
- CKPT_VERSION 从 30 提升到 31；不对旧模型做 warm start。训练打印完整参数预算。
  验证同时计算同噪声、同 timestep 的 ref-null 和 reference loss，前者保留原 `Val/*` 名称，
  后者使用 `Val/reference_*`。

### 本地容量审计

本轮审计的训练数据含 4964 个
clip、476 副骨架。clip 加权的有效关节数 min / P10 / P50 / P90 / max 为
`2 / 16 / 43 / 80.8 / 100`；对应输入维度 `20 / 146 / 389 / 729.2 / 902`。

| K × d_z | r 的 min / P10 / P50 / P90 / max | z 维度不小于输入的 clip 数 |
|---|---|---:|
| 2 × 8 | .0177 / .0220 / .0411 / .1096 / .8 | 0 |
| 4 × 8 | .0355 / .0439 / .0823 / .2192 / 1.6 | 45 |
| 4 × 16 | .0710 / .0878 / .1645 / .4384 / 3.2 | 118 |
| 8 × 32 | .2838 / .3513 / .6581 / 1.7534 / 12.8 | 1279 |

报告也逐骨架统计，并列出默认 r > .5 的 `weak_compression_at_default` 骨架；它们不满足明显压缩
假设。默认 2 × 8 只是工程起点，未按短训重建误差选择更大容量。

**数据缺口**：当前 `dataset/datasets.jsonl` 的所有来源都是 train，val 为空；未移动真实数据、未把
训练骨架当成未见骨架。当前 [tiny_tasks.json](../eval/tiny_tasks.json) 中的 `ReferenceMotion` 任务
覆盖同骨架、犬→马和鸟→其他物种的 encode 模式工程冒烟检查。未见骨架验收仍需要独立 val 来源。

### 已运行的实验与边界

使用项目 `../.venv/Scripts/python.exe` 和 RTX PRO 6000，在全部真实训练 clip 上从头运行四个候选，
每个 100 步：D=128、4 层、FFN=256、window=32、batch=4、BF16、20 diffusion steps、seed=10。
checkpoint 保存在 `save/reference_encoder_smoke_20261018_{2x8,4x8,4x16,8x32}`。
每个模型执行 5 组固定任务 × species on/off × ref/null，共 80 个生成 clip（10-step DDIM），并运行
MotionQualityScorer。另跑通原始 `Dog-Attack.glb` + `attack, bite` 文字 → Horse 的三分支 encode 推理。

逐任务报告保存在
`outputs/reference_encoder_eval_20261018_<候选>/report.json`，对应 NPY 与 BVH 留在同目录树中。

| K × d_z | 总参数 | 同骨架生成 MSE：ref / null | 固定马骨架交换犬参考的输出 RMS |
|---|---:|---|---:|
| 2 × 8 | 2,694,128 | 1.6770 / 1.6501 | .00558 |
| 4 × 8 | 2,694,256 | 1.0707 / 1.0624 | .00296 |
| 4 × 16 | 2,702,968 | 1.0486 / 1.0523 | .00472 |
| 8 × 32 | 2,720,648 | 1.1216 / 1.1035 | .00939 |

MSE 按 canonical 的 9·J+2 个模型通道计算，来自一个训练骨架上的完整生成样本，**不是 val l_simple**。
RMS 是生成物理特征的数值差，只确认参考改变了输出；不证明运动语义被正确迁移。100 步后参考 gate
已经非零，但自重建尚未稳定优于 null，不能据此选容量、断言纯文本不退化或迁移成功。
未启用 §4.3：当前还没有满足「同骨架重建可用」的证据。

参数预算：D=384、8 层、共享 FFN=2048、最后一层 FFN=256、cross-limb 后4层宽128、
action/species head bottleneck=128、ref 2×8 时，总计 30,992,258；共享 26,761,768；
编码器 219,466；全部参考分支 4,011,024（含 attention、FFN、归一化、gate、bias）。

测试覆盖 gate 起点、混合 batch 的完整硬旁路、null 的零参考梯度、激活样本的共享主干与编码器梯度、
有效关节池化、非根关节排列不变性、padding、9/11 维通道契约、独立参考 dropout、2/3 次 CFG、
species 状态、增强隔离、源骨架 resample 与配对验证。`torch.compile` 的 eager 后端
fullgraph 跟踪同时通过 reference/null；实际 GPU 训练验证使用 eager BF16，未据此宣称 Inductor 性能。

最终回归：Anytop 全套 1412 项测试、61 项子测试通过，另有 7 项服务端兼容测试通过。
仅目标骨架的窄 cond + checkpoint 源 cond 回退的 encode 推理也已实跑通过。
同骨架默认 retarget 模式的 img2img 生成与导出也已实跑通过。
旧路径的跨骨架 retarget 实跑在已有的关节匹配 LLM 服务处连接被拒绝，未完成该项端到端验证；
没有为本改动替换其关节匹配或服务配置。

### 训练入口

```powershell
../.venv/Scripts/python.exe train/train_anytop.py --cond_path dataset/cond.npy --save_dir save/ref_2x8_new --objects_subset all --action_group all --action_label_cond --action_label_adaln --species_joint_cond --ref_encoder --ref_slots 2 --ref_dim 8 --latent_dim 128 --ff_size 256 --layers 4 --cross_limb_last_n 2 --num_frames 32 --batch_size 4 --diffusion_steps 20 --num_steps 100 --save_interval 100 --log_interval 25 --amp_dtype bf16 --seed 10
```

延长训练时使用新的 save_dir，从头训练，并保持 decoder 侧 `leaf_drop_prob=0`、
`bone_length_aug_prob=0`，用 `ref_leaf_drop_prob` / `ref_bone_length_aug_prob` 调整编码器侧增强。
下一步需要独立 val 骨架、充分收敛的候选与同训练预算的文本基线，按 §6 验收后才能选择容量或触发 §4.3。

## 12. 第二版改动（2026-10-18）

**依据**：在 `zoo_locomotion_tiny_v11`（50k 步）、`v12` / `v13`（20k 步）上做了诊断，全部使用训练 clip 和 EMA 权重。

- **参考容量不是瓶颈。** 在同为 20k 步的点上，v11 / v12 / v13 有参考与无参考的分部位去噪误差差值基本相同。从 z 线性回归，尾巴/头颈运动的 R² 达到 0.84–0.94，v13 的 z 携带的信息更多，但去噪增益没有变化。v11 训到 50k 后，t=90 时的增益从约 18% 升到约 50%，尾巴与四肢的相对增益接近。
- **Cat_Walk → Dog 尾巴不翘的原因是跨骨架姿态。** 测量尾根→尾尖的仰角：Cat_Walk +89°，Cat→Cat encode +107°；Dog 无参考 −24°，ref_cfg_scale=1 时 −6°，=3 时 0°；Dog 的 33 个训练 clip 最高 +49°。参考的方向能迁移过去，但幅度被压回 Dog 的数据分布。Cat 与 Dog 在 rest pose 下的尾巴方向几乎相同，Cat_Walk 中尾巴朝上；这一差别在新的骨骼方向通道中可以直接读出。
- **参考专用 FFN 主要提供模式偏置。** 换参考时，ref cross-attn 的输出完全改变（相对变化约 1.29），ref FFN 只变 30–62%。

**改动**：E 增加 `ref_bone_dir` 输入（§3.1）；删除每层 ref FFN 及 `--ref_ff_dim`（§3.2）；CKPT_VERSION 31 → 32。推理的参考样本改为与 loader 共用 `reference_sample`。

**待验收**：与 v11 相同设置、相同步数重训，复测 Cat_Walk → Dog 的尾巴仰角、分部位去噪增益和交换测试，与 v11 对比。

## 13. 第三版改动：训练性能（2026-10-18）

**依据**：train_tiny 配置（B=16、60 帧、D=256、8 层、`--compile default`）实测，开启参考后真实 step 从 110.4 ms 升到 124.6 ms。消融定位：

- **编码器 E 约 8 ms**：graph attention 的拓扑 / 边偏置需要 4 次 `[T·B, H, J, J]` gather 与 fp32 累加，完整加性 mask 也让 SDPA 用不上融合 kernel；这部分与宽度无关，E 的 2 层约等于主干 2 层。且 `ref_cfg_drop_prob` 丢弃的行仍完整前向。
- **数据侧约 5 ms**：每个样本重算 joint_struct、rest token；未增强时参考的 canonical 特征与目标完全相同却再算一遍。取数线程每 batch 多 12 ms CPU，经 GIL 拖慢训练线程。
- **ref cross-attn 约 1 ms**，另占约 3 GB 显存：K·d_z 只有 32 维，却做 D×D 的 query / output 投影。

**改动**：
1. 数据侧：未增强的参考直接复用目标的 canonical 特征、rest token 与 structural channel；只有 leaf-drop 或骨长增强生效时重算。`reference_sample` 可接收已算好的这三项。
2. 参考 dropout 改为每 batch 固定保留行数，编码器只对保留行前向（§4.1）。每个样本被丢弃的概率为 `round(B·p)/B`。
3. E 的 spatial attention 去掉拓扑 / 边关系偏置（§3.1），`ref_graph_dist` / `ref_joints_relations` 字段删除。
4. ref cross-attn 改为折叠参数化（§3.2）。

CKPT_VERSION 32 → 33，需从头重训。
