# AnyTop：Motion Autoencoder 与 Latent Diffusion

> Motion Autoencoder（AE）、共享动作重建损失及 Latent Diffusion 基础生成链路已实现。本文按当前实现说明数据流、条件语义与支持边界。
> 本文不维护模型容量、损失权重或训练配置。实现完成不代表重建质量、生成质量或泛化能力已通过验收。

## 1. 当前整体结构

AE 将完整 canonical 动作编码为逐帧连续 latent，再结合静态骨架条件重建动作。它使用独立训练和评估入口，与现有“参考编码器 + 动作空间 diffusion”路径并存。

```text
动作 x + 静态骨架 s + 时间条件
  → 动作投影 + 骨架/时间嵌入
  → Encoder 图时空主干
  → 每帧 learned-query 关节池化
  → 连续 latent z
  → 静态骨架 query 读取同帧 latent
  → Decoder 图时空主干
  → 可选时间残差卷积 → 动作重建 x_hat
```

动作内容仅通过 z 从 Encoder 传到 Decoder；没有逐关节 skip、原动作输入或额外 reference 动作旁路。AE 不使用动作标签、物种 embedding 或 diffusion timestep；复用主干接口时使用固定零调制。

AE 的目标是完整动作重建。现有 `ReferenceEncoder` 服务于参考迁移，输入内容和表示契约不同，不能直接替代 AE 编码器，其输出也不能与 AE latent 互换。

## 2. 输入、条件与 latent 契约

### 2.1 动作表示

输入和输出均为 `x [B,J,C,T]`，C 对应 `MODEL_CHANNELS` 中的位置与 rotation-6D 通道，不学习工作容器中的 velocity 通道。物理转换和导出前由适配器补齐工作容器。

沿用现有 canonical 坐标、尺度归一化和统计契约：根位置使用专用统计，根 XZ 表示世界轨迹，其他关节位置遵循 RIC 约定；canonical 骨对齐全局旋转不能直接当作物理 local rotation。

模型支持变关节数及有效帧 mask，在输入、池化、latent 和输出处处理无效元素。**当前共享重建损失要求等长有效窗口，不支持时间 padding**；模型的 mask 能力不等同于训练损失已支持任意变长 batch。

### 2.2 静态骨架条件

模型读取的骨架字段限于：

- `n_joints`：有效关节范围。
- `joints_names_embs`、`rest_geom`、`joint_struct`：投影后相加，形成关节静态特征。
- `graph_dist`、`joints_relations`：供图关系注意力使用。

offsets、尺度、canonical 统计等数据由损失与物理转换链路使用，不作为额外动态内容传入模型。自重建模式使用同一骨架视图，`--dual_view` 模式使用独立增强的骨架视图。两种模式下关节名 drop 都在 Encoder/Decoder 独立采样，逐关节几何 drop 仅在 Encoder 生效；损失始终读取完整的目标骨架条件。

### 2.3 时间条件

`loop_metadata` 携带 `is_loop`、`loop_tile_count` 和 `resample_speed_cond`。后者表示重采样前源窗口长度与有效内部长度之比，使统一内部帧数后的动作仍保留原物理时长信息。

Encoder 使用开放序列的位置编码或循环相位编码。两侧主干均加入物理时间嵌入：开放动作使用源帧时间，循环动作使用物理周期长度。相关投影以零输出初始化并参与学习；有效长度取自 frame mask。

可选 temporal RoPE 在两侧主干中按开放/循环语义处理时间关系。启用时，Encoder 额外接收循环相位内容；Decoder 不加入额外的绝对/循环相位表，使用 latent、物理时间及主干时间机制恢复动作。

### 2.4 连续 latent

latent 布局为 `z [B,T,K,d_z]`。每帧通过固定数量的 learned queries 池化有效关节，槽数不随骨架关节数变化，也不对应特定关节。

当前仅进行关节维度的槽化，**不做时间降采样或解码前时间插值**。解码帧数等于 latent 帧数，显式 frame mask 必须与其匹配。latent 不保证对所有骨架都构成数值压缩，也不保证骨架身份完全解耦。

## 3. Encoder 与 Decoder

### 3.1 Encoder

完整动作通道经线性投影，与静态骨架及时间位置特征相加，送入图时空主干。主干复用 `GraphMotionDecoder` 层实现，处理关节空间关系和跨帧依赖，并支持可选跨肢体交互。

主干输出逐帧执行 learned-query attention pooling，再投影为 latent。模型支持两种表示：

- 确定性 AE：直接使用连续编码结果。
- 对角高斯 VAE：分别输出均值和对数方差，训练时重参数化采样；方差运算使用稳定数值路径并限制范围，默认评估使用均值。

### 3.2 Decoder

每个目标关节的静态骨架特征作为 query，通过 cross-attention 读取同帧 latent 槽。读取结果与静态 query 相加后进入独立的图时空主干，最后投影回完整动作通道。

Decoder 可在各层继续读取 latent，也可在输出投影前使用逐关节时间残差卷积细化局部动态。卷积保持帧数不变，循环动作按有效长度环绕，开放动作延拓边界。

Encoder 与 Decoder 的深度、宽度和前馈容量可独立设置。宽度相同时共享骨架和物理时间投影，宽度不同时分别投影；latent 接口保持一致。

### 3.3 已实现接口

```python
mu, logvar = ae.encode_posterior(x, skeleton, frame_mask, time_metadata)
z = ae.encode(x, skeleton, frame_mask, time_metadata)
x_hat = ae.decode(z, skeleton, frame_mask, time_metadata)
```

`encode` 直接返回 latent 张量，不返回 posterior 对象；默认返回均值，VAE 可通过 `sample=True` 采样。`encode_posterior` 返回均值及可选对数方差。`forward` 串联编码和解码，VAE 在训练模式采样、评估模式使用均值，也可返回 posterior 信息供 KL 计算。

## 4. 重建监督与数据链路

### 独立双视图跨重建

训练入口支持 `--dual_view`；未指定时使用单视图自重建。全量和 tiny 的 VAE→latent diffusion 批处理入口已开启双视图。每步仅编码源视图一次、解码到目标视图一次。checkpoint 记录 `dual_view_objective=source_to_target_v1`；旧的双向 cross/self 训练 checkpoint 仍可推理，但续训时会拒绝静默切换训练目标，需使用新的 run 名称。

每个样本从同一原始动作、原始骨架生成 A/B 两个视图。rest variant、拓扑、骨长、rest repose 分别独立抽样，各视图保留与自身骨架一致的动作、结构特征、canonical 转换上下文和损失条件。变速、循环闭合帧判定、相位偏移、平铺、裁剪、最终重采样及 loop 条件标签共享同一个时间映射，不通过重放随机数模拟同步。两侧按共同的关节 bucket 上界 padding。

普通增强路径执行 `E(A)→zA→D(zA, skeleton(A′))`，用独立增强后的 A′ 动作监督。这是广义自重建，A′ 可以具有不同拓扑、骨长和 rest pose；不再额外执行严格的 A→A 重建。各项几何损失只使用目标视图自己的完整条件；VAE 的 KL 只计算源视图 posterior，再按原权重及 warmup 加入。已移除 `--dual_view_self_weight`。

双视图与原单视图一样，每步只有一次编码、一次解码、一次重建损失和一次反向传播，不使用激活检查点重算。额外开销来自目标视图的独立数据增强、collate 与传输；共同 padding 上界也可能因目标骨架更大而增加计算量。

`joint_name_drop_prob` 使用同一概率，对每个视图的 Encoder 条件和 Decoder 条件分别独立采样；目标关节名也可能不可靠，因此不复用源侧的名字 mask。`rest_geom_joint_drop_prob` 只对 Encoder 的逐关节几何独立采样，Decoder 始终保留完整几何。整副几何、镜像关系和侧别 drop 已移除，普通网络 dropout 保持原配置。干净条件及损失转换上下文不受任何 drop 影响。旧 checkpoint 的已移除 drop 配置仅在推理加载时过滤，不影响权重加载。

验证保留原始自重建及 latent 使用诊断；启用 `--reference_pairs` 时，另以固定随机种子在 val 配对动作上评估单向源→目标迁移并报告 `reference_pair_cross_mse`。不再报告双视图通用 MSE、样本数或额外严格自重建指标。验证条件不做 drop，时间处理沿用 val 设置。

`--reference_pairs` 自动开启双视图训练，复用各数据源 `reference_pairs.jsonl` 的同动作逐帧配对组：有配对的 clip 从完整组内均匀抽取 B（包含自身；N 条动作各占 1/N），无配对的 clip 继续使用同一原始动作的独立增强视图。两侧始终使用自身动作和骨架，独立骨架增强、共享完整时间映射，每步仅执行 A→B；B→A 随后续源/目标抽样自然覆盖，不额外执行 A→A、B→B 或 B→A。不增加 `L_align`，也不启用 diffusion 的 reference encoder。组只包含当前 split、sample filter 等筛选后仍存在的成员，并校验原始长度和 loop 标记一致；关节分桶覆盖组内最大骨架。配对验证固定排除自身，仅衡量真实跨样本迁移；没有可用配对时不报告指标。全量/tiny 批处理默认开启此选项；续训禁止切换配对训练开关。

`MotionReconstructionLosses` 直接计算干净动作的重建监督，与 diffusion 共享动作几何损失组件，不构建 diffusion 对象，也不使用噪声时间衰减。

| 监督类别 | 当前实现 |
|---|---|
| 基础重建 | 有效关节的 canonical 位置与旋转 MSE |
| 局部动态 | 位置步长、位置加速度、旋转增量 |
| 根运动 | 根位置、步长与加速度，覆盖完整根位置通道 |
| 旋转与骨架 | 旋转测地距离、相对目标的骨向量一致性 |
| 肢体协同 | 末端运动相关性 |
| 循环接缝 | 循环动作自身的旋转接缝连续性，不要求首尾姿态相同 |
| 可选 VAE 正则 | 有效帧上的对角高斯 KL，按样本归约 |

几何项复用 canonical 到物理空间的转换及缓存，并沿用各自的归约语义。加速度监督对齐目标动态，骨向量监督对齐目标骨架表现。局部重新加噪补全的接缝损失不属于当前 AE 任务。

数据沿用共享 loader 的动作窗口、采样和骨架增强链路。速度、裁剪、循环拼接、拓扑、骨长及 rest 变化在编码前完成，增强后的动作重建自身，动作和骨架条件保持配对。基础 AE 不继承参考迁移中源/目标独立增强的规则，也不使用动作破坏与重新加噪任务。

## 5. 实现入口与当前边界

| 文件 | 当前职责 |
|---|---|
| `model/motion_autoencoder.py` | AE/VAE、逐帧池化、时间条件、encode/decode 与工作容器适配 |
| `train/motion_reconstruction_losses.py` | 共享动作几何损失与 AE 重建目标 |
| `train/train_autoencoder.py` | 独立训练、val 集评估、latent 置零/交换诊断与 checkpoint |
| `sample/eval_autoencoder.py` | 显式预处理动作文件的整段重建、指标与导出 |
| `train_tiny_vae_diffusion.bat`、`train_all_vae_diffusion.bat` | tiny / 全量数据的 VAE → latent diffusion 顺序训练入口 |
| `eval_autoencoder.bat` | 独立重建诊断入口 |

训练入口遍历独立 val 集的 60 帧窗口，包含正常重建、latent 置零和批内交换对照，并记录姿态、动态、根运动及骨架误差。val 不继承训练集筛选条件，关闭训练增强；为空或没有满足最小长度的片段时跳过评估，不回退到训练集。训练入口只保存指标和 checkpoint，不导出重建动作。这些诊断用于检查瓶颈依赖，不直接证明表示解耦或未见骨架泛化。

整段诊断将输入动作重采样到 checkpoint 的内部窗口后重建，再恢复原输入帧数；Encoder 与 Decoder 使用同一份时长比例及循环条件。该流程中的重采样属于评估适配，不是 AE 的时间压缩层。

Checkpoint 保存模型架构、权重、latent 布局、canonical 与解码内容版本、条件数据校验信息及续训状态。当前解码内容版本为 `latent_time_v2`；缺少物理时间投影的旧版本不能直接续训或诊断加载。

AE 自重建与独立 latent diffusion 链路均已实现。跨骨架 latent 交换、未见骨架泛化、长尾动作细节和潜空间可生成性仍需实验验证；实现与接口测试不代表生成质量已经验收。

## 6. Latent Diffusion

Latent Diffusion 在冻结 AE 的连续 latent 上学习条件生成，与原动作空间 diffusion 使用独立的模型和 checkpoint。沿用现有动作标签、骨架、species、参考动作及时间条件的语义，改变去噪对象和条件注入位置。

### 6.1 数据流与训练目标

```text
增强后的目标动作 + 目标骨架 + 真实时间条件
  → 冻结 Encoder → latent → 固定标准化 → z0
  → 加噪 → zt
  → 条件去噪器 → z0_hat
  → latent MSE

生成：latent 噪声 → 条件去噪 → 逆标准化
  → 冻结 Decoder（目标骨架、真实时间条件）
  → canonical 动作 → 物理转换 → 恢复输出长度与导出
```

Encoder 和 Decoder 始终保持 eval 模式并冻结权重。训练目标由 Encoder 在线、无梯度地产生；Decoder 不进入训练图。去噪器预测干净的标准化 latent，损失先对每个样本的时间、槽和通道维求 MSE，再对 batch 平均。训练使用均匀 timestep 采样，不计算动作空间的位置、旋转、骨骼或接缝损失。

潜空间扩散复用现有前向加噪、反向后验、噪声日程和采样步映射，支持普通反向采样与 DDIM。标准化 latent 不套用动作空间的数值裁剪；最终生成完成后逆标准化，并只解码一次。

### 6.2 固定窗口与时间语义

**所有动作都重采样到同一个内部窗口，当前为 N=60；内部每一帧均有效。** 原始帧数不同不意味着内部存在 padding，因此没有时间维的无效帧，也不需要有效帧 mask：时间 attention 读取整个窗口，加噪和 MSE 覆盖全部帧。入口会校验内部窗口长度，拒绝带时间 padding 的输入。

`resample_speed_cond` 表达重采样前源窗口与内部窗口的时长比例，保留原物理时长；`is_loop` 和 `loop_tile_count` 表达开放/循环状态及周期数。开放动作使用绝对位置与源帧时间，循环动作使用循环相位与物理周期长度；可选 temporal RoPE 处理时间关系。这些信息与表示噪声等级的 diffusion timestep 分开。

去噪器沿用参考优先级：参考生效时，周期数按单周期处理，并应用相应的 RoPE 策略；参考关闭后恢复无参考时的时间条件。**这些门控仅作用于去噪器，冻结 AE 编码目标和最终 Decoder 始终使用真实目标时间条件。**

骨架关节数仍可变化，需要排除 padding 关节；这是骨架有效性处理，与时间 padding、关节编辑 mask 或参考隐藏帧均不同。

### 6.3 去噪主干

输入、输出均为逐帧 latent `z [B,T,K,d_z]`，时间维不压缩。输入投影后加入 learned slot identity、帧位置和物理时间特征。槽 identity 只区分 AE 的池化槽，不代表具体关节或肢体。

每层依次执行：

1. 同帧槽间 self-attention。
2. 以 latent 为 query，cross-attention 读取目标静态骨架。
3. 参考生效时，逐帧读取独立参考槽，并通过可学习 gate 加入残差。
4. 每槽沿整个窗口执行时间 attention。
5. FFN 更新。

目标骨架在每次条件前向中编码一次，供各层共享。真实关节图关系只用于骨架分支，不用于 latent 槽间注意力。主干保留逐层 timestep 注入和 action AdaLN，cross limb block 关闭。输出头直接预测标准化 latent，不经过动作空间的输出容器适配。

### 6.4 条件来源与参考优先级

动作、参考和时间的共用编码逻辑由 `motion_conditioning.py` 提供，旧 AnyTop 与新 latent 模型复用同一套语义。

| 条件 | 当前作用 |
|---|---|
| 动作标签 | 沿用受控词表及 head/direction/modifier 角色槽；聚合为动作 token，同时加入 timestep embedding 并驱动各层 action AdaLN |
| 静态骨架 | 关节名称、结构与 rest geometry 构成关节 token，经真实骨架图关系编码后供 latent 读取；`n_joints` 排除 padding 关节 |
| species | 描述码与关节名共同产生 FiLM，在名称投影前调制；不作为独立的全局 timestep 条件 |
| 参考动作 | 在自身源骨架上构造参考特征，经独立、可训练的 `ReferenceEncoder` 生成逐帧参考槽；与待去噪的 AE latent 相互独立 |
| loop 与物理时长 | 决定帧位置、循环相位、物理时间嵌入及时间注意力策略 |

缺标签、标签无效、整条动作 CFG drop 或参考生效时，动作条件都进入同一个 learned null；加法注入与 AdaLN 使用同一个 token。角色槽 dropout 仅隐藏对应槽，与整条动作条件置空分别处理。

参考是否最终生效，由调用者开关、源关节有效性及训练时的 reference drop 共同决定。参考读取 gate 从零初始化；参考关闭或被 drop 后，源动作内容不进入 latent 残差，动作、species 和时间门控据此重新计算。

| 最终状态 | 动作 token | species FiLM | 参考读取 |
|---|---|---|---|
| 无参考或参考被 drop | 有效标签或 action null | 启用该功能时生效 | 旁路 |
| 参考有效 | action null | 旁路 | 生效 |
| reference CFG 的 base 分支 | 强制 action null | 启用该功能时生效 | 旁路 |

species override 只替换描述码，不改变目标骨架、rest 或 canonical 统计。生成侧的编码参考与动作标签互斥；不通过冻结 AE 编码源动作来替代独立参考分支。

### 6.5 数据增强、标准化与 CFG

训练沿用共享 loader，速度、循环拼接、拓扑、骨长及 rest 等增强在目标编码前完成，增强后的动作与目标骨架保持配对。去噪器的名称、geometry、tempo 和 CFG dropout 只改变去噪条件，不改变 AE 的编码目标或最终解码条件。当前使用在线编码，不维护 latent 缓存。

训练入口支持 `--compile None|default|reduce-overhead`，默认 `None`；tiny 启动脚本启用 `--compile reduce-overhead`。编译范围为冻结 AE 的目标编码与去噪器训练前向，使用静态 shape、完整图；各关节桶首次使用时会编译。AE 编码保持 FP32，去噪器沿用 `amp_dtype`。归一化统计和 val 集验证使用 eager；优化器、EMA 和 checkpoint 始终使用原始模块，可在续训时切换编译模式。

latent 标准化统计在训练开始时从训练增强分布采集，沿 batch 和时间维归约，保留槽与通道维；后续保持固定，不对每段动作单独消除均值或幅度。确定性 AE 使用编码结果；VAE 支持 posterior 均值或采样，两者均要求统计与训练分布一致。采样模式的统计计入 posterior 方差，退化维度通过标准差下限处理。

CFG 在标准化 latent 的干净样本预测上组合，然后统一逆标准化和解码：

- 动作 CFG：条件与 base 分支仅切换动作标签的启用状态，其余条件一致。
- 参考 CFG：参考分支读取源动作；base 分支关闭参考并强制 action null，同时恢复无参考时的 species、周期数与 RoPE 策略，因此 base 仍保留目标骨架和物理时间条件。

两种 CFG 不叠加，使用引导前会检查对应条件和训练时的 null 分支支持。引导按实际噪声日程分段，低噪声阶段回到普通条件预测。最终 Decoder 的目标骨架与真实时间条件不随 CFG 分支变化。

## 7. Latent Diffusion 实现入口与边界

| 文件 | 职责 |
|---|---|
| `model/latent_motion_denoiser.py` | 槽时空去噪主干、静态骨架分支与参考读取 |
| `model/motion_conditioning.py` | 共用动作槽、reference 优先级、时间条件及条件 dropout |
| `diffusion/latent_diffusion.py` | latent 加噪、干净样本 MSE 和无动作域裁剪的采样 |
| `utils/latent_util.py` | 冻结 AE、真实条件筛选、标准化与 checkpoint 绑定校验 |
| `train/train_latent_diffusion.py` | 在线编码训练、val 集验证、EMA 与保存续训 |
| `train/latent_validation.py` | VAE 与 latent diffusion 共用的独立 val 加载器及空集处理 |
| `sample/generate_latent.py` | 条件解析、CFG、输出长度恢复及动作导出 |
| `train_tiny_vae_diffusion.bat`、`train_all_vae_diffusion.bat` | VAE 成功训练后使用其 checkpoint 训练 latent diffusion |

checkpoint 保存去噪器及续训状态，绑定 AE 文件 hash、权重选择、架构、canonical/解码内容版本、latent 布局与标准化统计，以及动作词表/角色槽、species 描述表、参考/时间合同和扩散预测语义。species 描述表随模型目录保存，AE 本体保持独立。加载和续训会校验相应绑定；新 checkpoint 与旧动作空间 diffusion checkpoint 不兼容。

生成沿用现有动作标签、species、参考预处理、长度先验、loop 判断和周期解析。参考先对齐输出时域，再重采样到内部窗口；生成动作由冻结 Decoder 解码后，转换到物理空间、恢复所需输出长度并导出。

当前支持无动作标签生成、动作条件生成、species 条件和编码参考生成。**不支持时间 padding、joint mask、temporal-span mask、renoise、img2img、inpaint joints/frames 或 outpaint。** 不接受编辑请求；共享 loader 携带的全零参考隐藏帧字段会被移除，实际隐藏帧请求会被拒绝。

验证遍历独立 val 集，使用中间噪声去噪结果，按样本数汇总 latent MSE、解码 canonical MSE 和 AE 重建基线；保留末尾不足一个 batch 的样本，空 val 集跳过验证。标准化统计仍只从训练分布采集。这不是完整反向采样的质量验收；条件响应、动作细节、根轨迹、循环接缝及未见骨架泛化仍需生成实验评估。续训恢复模型、优化器、AMP 和主进程 RNG，worker/prefetch 数据流重新启动，不保证逐 batch 精确复现；旧 checkpoint 中的训练评估批不再使用。

## 8. 参考

- [当前参考编码器设计](reference_encoder.md)：参考迁移表示与完整动作重建的区别。
- [当前模型结构](anytop_model_architecture.md)：现有主干与动作空间生成路径。
- [MLD: Executing Your Commands via Motion Diffusion in Latent Space](https://arxiv.org/abs/2212.04048)：连续潜空间扩散路线参考。
