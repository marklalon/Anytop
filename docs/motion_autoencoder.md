# AnyTop：Motion Autoencoder

> Motion Autoencoder（AE）及共享动作重建损失已实现。本文按当前实现说明数据流、条件语义与支持边界。
> 本文不维护模型容量、损失权重或训练配置。实现完成不代表重建质量、迁移质量或泛化能力已通过验收。

## 1. 当前整体结构

AE 将完整 canonical 动作编码为逐帧连续 latent，再结合静态骨架条件重建动作。它使用独立训练和评估入口，并承载参考动作的跨骨架迁移；motion diffusion 负责动作标签与骨架条件生成，不再编码参考动作。

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

AE 的目标是完整动作重建与跨骨架动作迁移。源动作经 Encoder 得到 latent，再结合目标骨架条件由 Decoder 重建。

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

offsets、尺度、canonical 统计等数据由损失与物理转换链路使用，不作为额外动态内容传入模型。训练始终使用独立增强的源/目标骨架视图。关节名 drop 都在 Encoder/Decoder 独立采样，逐关节几何 drop 仅在 Encoder 生效；损失始终读取完整的目标骨架条件。

### 2.3 时间条件

`loop_metadata` 携带 `is_loop`、`loop_tile_count` 和 `resample_speed_cond`。后者表示重采样前源窗口长度与有效内部长度之比，使统一内部帧数后的动作仍保留原物理时长信息。

Encoder 使用开放序列的位置编码或循环相位编码。两侧主干均加入物理时间嵌入：开放动作使用源帧时间，循环动作使用物理周期长度。相关投影以零输出初始化并参与学习；有效长度取自 frame mask。

两侧主干的时间注意力始终使用 temporal RoPE，按开放/循环语义处理时间关系。Encoder 额外接收循环相位内容；Decoder 不加入额外的绝对/循环相位表，使用 latent、物理时间及主干时间机制恢复动作。

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

训练入口固化 `dual_view`、`reference_pairs`、`layerwise_latent` 为始终开启，移除对应命令行开关及单视图训练分支；Decoder 始终逐层读取 latent。每步仅编码源视图一次、解码到目标视图一次。checkpoint 记录 `dual_view_objective=source_to_target_v1`；旧的双向 cross/self 训练 checkpoint 仍可推理，但续训时会拒绝静默切换训练目标，需使用新的 run 名称。

每个样本从同一原始动作、原始骨架生成 A/B 两个视图。rest variant、拓扑、骨长、rest repose 分别独立抽样，各视图保留与自身骨架一致的动作、结构特征、canonical 转换上下文和损失条件。变速、循环闭合帧判定、相位偏移、平铺、裁剪、最终重采样及 loop 条件标签共享同一个时间映射，不通过重放随机数模拟同步。两侧按共同的关节 bucket 上界 padding。

普通增强路径执行 `E(A)→zA→D(zA, skeleton(A′))`，用独立增强后的 A′ 动作监督。这是广义自重建，A′ 可以具有不同拓扑、骨长和 rest pose；不再额外执行严格的 A→A 重建。各项几何损失只使用目标视图自己的完整条件；VAE 的 KL 只计算源视图 posterior，再按原权重及 warmup 加入。已移除 `--dual_view_self_weight`。

双视图与原单视图一样，每步只有一次编码、一次解码、一次重建损失和一次反向传播，不使用激活检查点重算。额外开销来自目标视图的独立数据增强、collate 与传输；共同 padding 上界也可能因目标骨架更大而增加计算量。

`joint_name_drop_prob` 使用同一概率，对每个视图的 Encoder 条件和 Decoder 条件分别独立采样；目标关节名也可能不可靠，因此不复用源侧的名字 mask。`rest_geom_joint_drop_prob` 只对 Encoder 的逐关节几何独立采样，Decoder 始终保留完整几何。整副几何、镜像关系和侧别 drop 已移除，普通网络 dropout 保持原配置。干净条件及损失转换上下文不受任何 drop 影响。旧 checkpoint 的已移除 drop 配置仅在推理加载时过滤，不影响权重加载。

验证保留原始自重建及 latent 使用诊断；另以固定随机种子在 val 配对动作上评估单向源→目标迁移并报告 `reference_pair_cross_mse`。不再报告双视图通用 MSE、样本数或额外严格自重建指标。验证条件不做 drop，时间处理沿用 val 设置。

训练始终复用各数据源 `reference_pairs.jsonl` 的同动作逐帧配对组：有配对的 clip 从完整组内均匀抽取 B（包含自身；N 条动作各占 1/N），无配对的 clip 继续使用同一原始动作的独立增强视图。两侧始终使用自身动作和骨架，独立骨架增强、共享完整时间映射，每步仅执行 A→B；B→A 随后续源/目标抽样自然覆盖，不额外执行 A→A、B→B 或 B→A。不增加 `L_align`。组只包含当前 split、sample filter 等筛选后仍存在的成员，并校验原始长度和 loop 标记一致；关节分桶覆盖组内最大骨架。配对验证固定排除自身，仅衡量真实跨样本迁移；没有可用配对时不报告指标。续训要求 checkpoint 的三个固化配置均已开启。

`MotionReconstructionLosses` 直接计算干净动作的重建监督，与 diffusion 共享动作几何损失组件，不构建 diffusion 对象，也不使用噪声时间衰减。

Diffusion 同样支持 `lambda_acc`（全关节位置加速度）与 `lambda_root_pos`（根位置），两项直接复用 AE 的计算，不做噪声时间衰减。两侧根步长和根加速度均监督 XYZ，并对三个通道取平均；相较旧 diffusion 的 XZ 均值，Y 轴现在参与监督，纯 X/Z 误差的数值也随分母变化。新增权重的 CLI 默认值为零，diffusion 的全量与 tiny 训练脚本均显式开启这两项。

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

数据沿用共享 loader 的动作窗口、采样和骨架增强链路。速度、裁剪、循环拼接、拓扑、骨长及 rest 变化在编码前完成，源/目标视图独立进行骨架增强、共享时间映射，目标动作与目标骨架条件保持配对（见双视图与参考配对训练说明）。AE 不使用扩散模型的动作破坏与重新加噪任务。

## 5. 实现入口与当前边界

| 文件 | 当前职责 |
|---|---|
| `model/motion_autoencoder.py` | AE/VAE、逐帧池化、时间条件、encode/decode 与工作容器适配 |
| `train/motion_reconstruction_losses.py` | 共享动作几何损失与 AE 重建目标 |
| `train/train_autoencoder.py` | 独立训练、val 集评估、latent 置零/交换诊断与 checkpoint |
| `sample/eval_autoencoder.py` | 显式预处理动作文件的整段重建、指标与导出 |
| `train_autoencoder_tiny.bat`、`train_autoencoder_all.bat` | tiny / 全量数据的独立 AE/VAE 训练入口 |
| `eval_autoencoder.bat` | 独立重建诊断入口 |

训练入口遍历独立 val 集的 60 帧窗口，包含正常重建、latent 置零和批内交换对照，并记录姿态、动态、根运动及骨架误差。val 不继承训练集筛选条件，关闭训练增强；为空或没有满足最小长度的片段时跳过评估，不回退到训练集。训练入口只保存指标和 checkpoint，不导出重建动作。这些诊断用于检查瓶颈依赖，不直接证明表示解耦或未见骨架泛化。

整段诊断将输入动作重采样到 checkpoint 的内部窗口后重建，再恢复原输入帧数；Encoder 与 Decoder 使用同一份时长比例及循环条件。该流程中的重采样属于评估适配，不是 AE 的时间压缩层。

Checkpoint 保存模型架构、权重、latent 布局、canonical 与解码内容版本、条件数据校验信息及续训状态。当前解码内容版本为 `latent_time_v2`；缺少物理时间投影的旧版本不能直接续训或诊断加载。

AE 自重建与跨骨架迁移链路已实现。跨骨架 latent 交换、未见骨架泛化、长尾动作细节仍需实验验证；实现与接口测试不代表迁移质量已经验收。
