# AnyTop 改造方案：Motion Autoencoder + Latent Diffusion

> 状态：7.A 已实现确定性 AE 基线及共享动作损失；其余为设计提案，新接口和参数以实际代码为准。
> 重点：第一阶段在混合数据、多骨架条件下保真重建动作，尤其是 shake 等局部高频动作。
> 第二阶段：冻结自编码器，在连续 latent 上训练 diffusion，训练目标以 latent MSE 为主，不经过解码器反向。

## 1. 目标与成功条件

当前现象：shake-only 小数据可过拟合出局部细节；扩展动作池后，即使匹配 shake 曝光次数，生成细节仍可能退化。

本改造希望把问题分为两个可独立检查的任务：

1. **表示与重建**：是否存在一个可压缩的表示，能同时保留各类动作的姿态、根轨迹、局部振幅、频率和关节间时序？
2. **条件生成**：给定动作与骨架条件，生成模型能否学好这些表示的分布？

第一阶段通过不代表第二阶段必然通过，也不保证类别不平衡消失。latent MSE 小不等于所有动作细节误差都小。本方案提供任务分解和诊断边界，不预先承诺清晰度或速度收益。

第一版优先级：重建保真 > 未见骨架泛化 > 潜空间可生成性 > 压缩率和性能。先通过混合数据重建验收，再冻结自编码器开展第二阶段。

## 2. 当前架构与改造边界

当前代码入口：

- `model/anytop.py`：目标动作输入投影、动作空间去噪主干、输出头，以及可选参考动作分支。
- `model/reference_encoder.py`：参考动作编码器，输出逐帧 `z_ref [B,T,K,d]`。
- `model/motion_transformer.py`：时空主干及逐帧参考 cross-attention。
- `diffusion/gaussian_diffusion.py`：对 canonical 动作加噪、预测 x0、计算动作损失。
- `data_loaders/truebones/truebones_utils/canonical_features.py`：动作表示、标准化和物理空间转换。
- `data_loaders/truebones/data/dataset.py`：动作窗口、骨架增强及动作采样。

当前是“参考编码器 + 动作空间条件去噪”的联合训练。目标动作 x_t 本身可以携带细节，参考 latent 不是动作重建的唯一路径。

新方案的数据流：

```text
阶段一：
  原动作、骨架 → 一致的数据增强 → x, s
  x, s → Encoder → z → Decoder(z, s) → 重建动作 x_hat
                                            ↓
                                 姿态、动态、骨架损失

阶段二：
  x, s → 冻结 Encoder → z → 固定标准化 → z0
  z0 → latent 加噪 → zt → 条件去噪网络 → z0_hat
                                      ↓
                                  latent MSE

生成：
  latent 噪声、动作条件、骨架条件 → latent 去噪
  → 逆标准化 → 冻结 Decoder → canonical 动作 → 现有导出链路
```

保留当前训练与推理入口作为对照，新增独立入口和 checkpoint 类型。第一版不改写现有模型 checkpoint，不把旧 reference encoder 权重自动视为新自编码器权重。

## 3. 第一阶段的输入输出契约

### 3.1 动作表示

沿用当前 canonical working layout，不重新发明坐标系：

- 主模型有效通道由 `MODEL_CHANNELS` 决定；当前为位置 3 维与旋转 6D，共 9 维。
- 使用 canonical 骨对齐全局旋转，不能误当作直接的物理 local rotation。
- 根位置按当前 root 专用统计处理，根 XZ 存储世界轨迹；其他关节位置遵循当前 RIC 约定。
- 沿用 L 尺度归一化、全局 canonical 统计和窗口根 XZ 重定位。
- 当前 12 维工作容器中的 velocity 通道不是学习目标；输入输出适配器应按现有转换契约补齐，而非新增速度输出任务。

输入与重建目标形状约定：`x [B,J,9,T]`；进入现有物理转换函数前恢复所需工作容器。`J`、`T` 可以变化。

### 3.2 静态骨架条件 s

编码器和解码器读取有效关节 mask、父子拓扑、关系表、关节名 embedding、joint_struct、rest geometry、offsets、L、根索引及 canonical 转换所需统计。

条件中不能混入目标动作的逐帧位置、旋转、速度、参考动作特征或从目标提取的动态统计。第一版不向解码器提供 action label / species embedding；骨架形态来自 s，运动内容必须来自 z。是否加入额外静态语义条件单独做消融。

长度、真实 frame mask、loop 状态和周期元数据独立传入。解码长度不能由 padding 猜测；周期信息不得误把自由动作强制首尾闭合。

## 4. 第一阶段模型：完整动作编码器 + 骨架条件解码器

### 4.1 为什么不能直接使用当前 ReferenceEncoder

当前参考编码器的逐帧输入主要是单位骨方向、高度和根 XZ，另有静态 rest 上下文；它没有完整读取每个关节的旋转扭转信息。其目标是迁移动作内容，而不是完整动作保真压缩。

新编码器可以复用图时空层、关节池化和 mask 处理，但必须读取完整的有效 canonical 动作通道。现有参考条件瓶颈的容量不能默认足够编码全部动作细节。

### 4.2 Encoder

建议第一版采用连续 VAE，以便在第二阶段开始前就检查重建与潜空间正则化的权衡：

1. 对每个关节的完整动作通道做输入投影；加入静态骨架条件与时间位置编码。
2. 采用少量 spatial + temporal blocks，支持真实关节数与时间 mask。
3. 每帧用 K 个 learned queries 池化关节 token，得到与关节数无关的槽。
4. 每槽预测 `mu, logvar [B,T,K,d_z]`。
5. 训练采样 `z = mu + exp(0.5 * logvar) * eps`；评测分别报告 mu 重建和采样重建。

先用逐帧 latent 验证空间瓶颈对 shake 的影响，再比较时间压缩方案，并重新验收高频、相位和非整除长度。

当前实现支持可选时间压缩：在关节槽池化后、posterior heads 前，先做时间滤波，再降采样。解码前线性插值恢复完整时间长度，并通过时间残差卷积细化；所有 decoder latent 注入均使用恢复后的表示，无完整帧率内容旁路。open 边界钳制，loop 按有效周期均匀采样并跨首尾插值，padding 不参与聚合。输出长度不能由压缩后的长度唯一确定时，须向 `decode` 传入原始 `frame_mask`；省略时输出长度为 latent 长度乘 stride。VAE 采样和 KL 使用压缩后的 mask，位置编码及 RoPE 保持完整输出帧率。该方案不保证消除抖动，仍需比较真实快速动作细节与重建加速度误差。

压缩率改变必须使用新 run，不能续训旧目标配置的 checkpoint。支持的压缩配置以模型实现和启动脚本为准。

容量消融分别比较槽数 K 与每槽维度 d_z，观察细节保真和潜空间正则化的权衡。latent 对小关节骨架可能不构成数值压缩；必须报告各骨架的实际表示量和主干计算量，不能仅凭“latent”声称压缩。

VAE 的 logvar 数值范围需要稳定处理；KL 不得强迫 posterior 方差塌到不可计算的数值区间。另设同容量 deterministic AE 对照，区分瓶颈损失与 KL 带来的损失。

### 4.3 Decoder

采用静态骨架 query + latent cross-attention + 时空主干：

1. 为每个目标关节构造静态 query：关节名、结构、rest geometry 与时间位置。
2. 每帧关节 query 读取该帧 K 个 z 槽。
3. 用时空 blocks 恢复关节协同与跨帧关系。
4. 输出完整位置与 rotation-6D 通道，根轨迹遵循同一动作输出契约。

可复用 `GraphMotionDecoder` 的层实现；现有 reference cross-attention 可作为读取槽的起点，但新模块必须有独立接口和明确的 latent 激活规则。

**禁止的旁路**：原始或带噪目标动作输入、逐帧目标姿态 query、encoder 到 decoder 的逐关节 skip、额外高维 reference 动作。否则重建可以绕过 z，第一阶段指标无法证明瓶颈保留细节。

解码器不需要 diffusion timestep；如复用底层接口，传显式固定的零调制或移除该入口，不将训练步数伪装成噪声时间。

第一版不做 encoder 到另一副任意骨架 decoder 的无配对重建。自重建不能证明跨物种 latent 可交换，也不能保证骨架身份完全解耦。

### 4.4 为什么先选连续 latent

第二阶段目标是连续 latent MSE，因此第一版不引入 VQ 码本。VQ-VAE + token 生成、残差量化、分层高低频 latent 均列为后续方案，不与第一版同时改动。

VAE 正则化并不自动保证易生成。若 KL 足够小才能保细节，但第二阶段始终学不好，应同时评估 latent 尺度、条件歧义、生成容量与 latent 结构，再决定是否改换 AE 家族。

## 5. 第一阶段损失：迁移动作监督，去掉 diffusion 专用语义

总目标：

```text
L_AE = L_pose
     + lambda_dpos     * L_dpos
     + lambda_drot     * L_drot
     + lambda_root_dpos* L_root_dpos
     + lambda_root_acc * L_root_acc
     + lambda_geo      * L_geo
     + lambda_bone     * L_bone
     + lambda_limb_corr* L_limb_corr
     + lambda_loop_wrap* L_loop_wrap
     + beta            * L_KL
```

重建项比较重建动作与增强后的真实目标；loop_wrap_loss 额外约束预测自身的接缝连续性，不比较 GT。保留现有损失归约方式：bone_loss 按 batch 内有效元素平均，有效项更多的样本权重更大。沿用有效关节、短序列及 loop 语义；若引入时间 padding，需显式排除无效帧。

| 损失 | 第一阶段用途 | 迁移注意点 |
|---|---|---|
| L_pose / 当前 l_simple 对应项 | canonical 位置、旋转重建 | 独立命名为重建损失；保持 root 专用统计，屏蔽 padding |
| dpos_loss | 位置一阶差分、局部运动振幅 | 复用现有 corpus step 标度与样本权重，保留有效 loop wrap |
| drot_loss | 旋转增量与局部抖动 | 使用现有训练的全局旋转增量定义；保留 drot_joint_boost 作为消融 |
| root_dpos_loss | 根 XZ 步长 | 不被全关节平均稀释 |
| root_acc_loss | 根 XZ 二阶差分 | 对齐目标加速度，不是把加速度压向零 |
| geodesic_loss | 旋转姿态准确性 | 沿用稳定旋转转换及 fp32 数学路径 |
| bone_loss | 骨向量长度、方向和末端旋转细节 | 当前是相对 GT 的骨向量一致性，不改成所有帧严格贴 rest 长度 |
| limb_corr_loss | 肢体间协同 | 单独观察；它对幅度不敏感，不能替代 dpos/drot |
| loop_wrap_loss | 真实 loop 的旋转接缝连续性 | 不要求最后一帧等于第一帧；开放窗口不计算 |
| temporal_span_seam_loss | 第一版关闭 | 这是局部重新加噪补全的接缝项；基础 AE 没有对应任务 |

### 5.1 不沿用 diffusion 时间衰减

当前 drot、bone 等非线性辅助项具有噪声相关的 `alphas_cumprod[t]` 衰减。第一阶段是干净动作编码与重建，不存在同样的 t：相关调用使用 `t=None` 或独立的纯重建接口，不随机采一个 diffusion t 来压低损失。

不要为复用损失而构建整套 diffusion 对象并调用 `training_losses`。建议提取共享的动作几何损失组件；原 diffusion 调用处负责噪声衰减，新 AE 调用处负责重建权重。

### 5.2 KL 与权重策略

KL 使用对角高斯到 N(0,I) 的解析形式，先对每个样本的有效 T、K、d_z 求平均，再 batch 平均。记录每槽、每维 KL、mu 方差和 posterior 方差，避免仅总均值掩盖塌陷。

- 先用 beta=0 做短期重建能力诊断；这段只能证明近似 AE 的能力，不作为最终 VAE 通过标准。
- 再对 beta 做 warm-up，至少比较两档非零强度；必须在目标 beta 下重新验收。
- 当前 train_tiny 的动作 lambda 可作初始量级参考，但不直接认定最优：去掉噪声衰减后有效强度已经改变。
- 用各项未加权值、加权值和抽样的梯度范数判断权重是否失衡；不只看总 loss。
- 不把所有类的振幅统一拉大；静止动作的多余抖动必须同时受罚。

### 5.3 可选频域监督

第一版优先使用现有姿态和动态项。若重建平均误差下降但 shake 高频仍明显损失，再增加目标相对的多尺度频域项，优先比较角增量或位置步长的频谱。

仅匹配功率谱会忽略相位与关节关系，因此只能补充时域监督。开放窗口使用一致的有效长度和窗函数，频率按实际 fps 换算；不得把 padding、人工拼接或 loop 重复产生的谱峰当作真实细节。

## 6. 第一阶段数据与增强

### 6.1 使用同一条数据契约

沿用现有训练 split、sample_filter、多数据集元数据及采样权重。先按源 clip / 骨架定义 train、val，再生成窗口或增强；同一 clip 的增强版本不能跨 split。严格的未见骨架评测需要额外保留骨架组，不能由 clip split 代替。

两阶段采用同一版本的数据转换和统计。验证集默认关闭随机增强，增强鲁棒性另设可复现的验证集合。

### 6.2 增强分配

| 当前增强 | 第一阶段建议 | 第二阶段接口约束 |
|---|---|---|
| motion_speed_aug、裁剪、循环相位、loop tiling | 保留，在编码前执行，增强结果重建自身 | 编码前可继续增强，不能在 latent 上随意插值冒充时间重采样 |
| topology_aug | 保留，动作与拓扑、索引、mask 同步变更 | 新骨架必须在 AE 验证范围内 |
| bone_length_aug | 保留，目标为增强后的动作 | 不能仍用原骨长条件；当前增强无 contact IK，不以其产物证明接触正确 |
| rest_repose、rest_variant | 保留，使用更新后的 rest 与转换信息 | 不假定 latent 完全不变，除非额外验证了不变性 |
| 名称或静态条件 dropout | Encoder/Decoder 共享置空后的名称与 rest geometry；可选关系丢弃及网络 dropout | 用于实际读取这些条件的网络；概率由实验配置决定 |
| action/species CFG dropout | 第一版 AE 不使用这些生成条件 | 第二阶段使用 |
| ref_cfg_drop_prob | 不适用于新 AE 的唯一路径 latent | 参考生成分支单独设计 |
| joint_mask、temporal_span_mask、unreliable_mask_drop、renoise | 基础 AE 关闭 | latent 局部补全必须重新设计，不能按原关节索引直接 mask 全局槽 |
| loop_cond_prob 降级 | AE 保留可配置的 loop→open 降级 | 源周期属性用于物理增强；最终窗口的 loop 条件同时用于 Encoder、Decoder 和损失，降级窗口不计算 wrap |

增强后的 x' 和 s' 必须成对送给 encoder、decoder 和 loss。编码器与解码器不能各自独立随机出两个不同拓扑或骨长，再无配对地重建。

当前参考迁移路径允许源参考与目标骨架独立增强，那是不同任务；第一版 AE 不继承这条独立采样规则。

不直接启用输入破坏→干净目标的 denoising AE，以免把“细节是否能穿过瓶颈”和“能否补全缺失动作”混成一个实验。

### 6.3 尾部采样

阶段一也存在长尾问题。保留 rare_head_word_floor / head_word_weights 的机制，统计实际第一 head word 分组的概率、曝光量与增强分布。

不把“所有尾部统一过采样”当作充分条件。每类报告重建指标，针对困难动作补预算，同时观察常见动作是否退化。保持原始权重分布与统一托底两组对照。

当前 sample_loss_limit 按 diffusion timestep 维护基准，不能原样移植。AE 基线关闭该机制，先检查异常数据；若确需鲁棒降权，另设计与噪声 t 无关的版本，并报告 shake 等困难样本的降权比例。

## 7. 第一阶段训练流程

### A. 最小自重建与旁路检查

- 在固定小集合上训练无随机增强、beta=0 的 AE，覆盖 shake、静止、位移和不同骨架。
- 确认完整旋转、根轨迹、有效长度、loop 接缝、padding 和变关节数契约。
- 固定 s，打乱或置零 z，动作内容应明显改变；若仍能复制原目标，检查动态信息泄漏。该测试只能发现部分泄漏，不能单独证明表示解耦。
- 小集合成功仅作为实现与容量检查，不作为混合训练通过标准。

已实现入口：`train/train_motion_autoencoder.py`，运行 `train_tiny_autoencoder.bat [run_name]`。
按 sample filter 选取训练 clip，评估时每个 clip 使用固定窗口；支持确定性 AE 和对角高斯 VAE，以及可选时间压缩。模型容量、模块开关、损失权重、增强概率、训练步数及精度和编译配置以代码、启动脚本和各 run 保存的配置为准，本文不维护具体实验参数。

重建使用有效 canonical 通道的 MSE，并可配置位置步长、加速度、旋转、根轨迹、骨向量、肢体协同及 loop 接缝损失；不做 diffusion 噪声衰减，保留现有归约与 loop 语义。当前共享 AE 损失要求无时间 padding 的固定窗口。

VAE 训练采样 posterior，评估与导出使用 mu；KL 按每样本有效时间、槽和维度平均。KL 与 dpos/drot/acc 支持线性 warm-up；日志记录原始项、加权项及 warm-up 比例。续训沿用绝对步数，并校验架构与训练目标配置；切换 AE/VAE 需新建实验。AMP FP16 的损失与评估保持 FP32；GradScaler 在裁剪前 unscale，溢出重试同一批次，checkpoint 保存 scaler 状态。

训练逐步在线采样，输入与目标为同一增强样本。Encoder/Decoder 共享置空后的名称与 geometry（含 present flag），保留物理转换和损失所需统计与 offsets；关系丢弃及网络 dropout 仅训练生效。loop 降级窗口采用开放重采样与开放损失。固定无增强窗口仅用于评估；按绝对步数派生增强随机种子，恢复训练可复现。

输出位于 `save/<run_name>/`：`autoencoder.pt` 支持续训并保存固定输入；TensorBoard 日志记录训练各项原始/加权 loss 与定期评估；`reconstruction_metrics.json` 记录逐样本重建、置零/跨样本打乱 latent 的误差和角步长 RMS；`reconstructions/` 保存物理空间 NPZ，并按导出配置保存原始/重建 NPY、BVH。旁路检查只提供诊断，不自动判定通过。

### B. 混合数据保真基线

- 使用完整选定动作池，仍不做时间下采样。
- 开启选定骨架与时间增强，逐项记录对重建的影响。
- 比较空间瓶颈容量和采样方案；优先通过 shake 与尾部重建门槛。

### C. 正则化与未见数据验收

- 开启 KL warm-up，在最终 beta 下稳定训练并验收。
- 分别评测 mu、posterior sample，确认不是只有确定性均值重建好。
- 按源 clip、未见骨架、动作、活动部位、长度及 loop/open 分组。
- 若 KL 导致细节丢失，先调正则强度和容量，不以第二阶段“也许会补回来”为理由通过。

### D. 冻结并发布 AE 契约

- 固定 checkpoint 与 eval 模式，统计训练 latent 的 mean/std，不使用 val 计算统计。
- 统计方法必须明确：VAE posterior sample 或 mu、有效 token、按维归约方式、数据权重、增强分布和固定随机种子。
- 第二阶段若使用 posterior sample，标准差统计须包含 posterior 的方差贡献；不能只统计 mu 后当作同一分布。
- 使用 `z_norm=(z-mean)/std`，对近零 std 作数值保护并报告退化维度；统计一经发布固定不再漂移。
- 统计后的标准化要可逆，不给每段动作单独减均值或除 RMS，以免抹掉根轨迹、幅度或节奏。

## 8. 第一阶段验收：不能只看总重建 loss

### 8.1 必须覆盖的集合

1. 固定 shake 样本及未见 shake clip，包含不同参与部位和骨架。
2. 其他尾部动作，避免只把 shake 修好。
3. 常见周期运动与快速肢体动作。
4. idle / 静止动作，检查虚假高频。
5. 根位移明显的动作、短序列、开放窗口和真实循环窗口。
6. rest variant、合法拓扑编辑、骨长增强，以及未见骨架组。

未见组中没有足够样本时明确标记“证据不足”，不使用训练集重建代替泛化结果。

### 8.2 指标与可视检查

| 方面 | 指标 |
|---|---|
| 姿态 | canonical MSE、物理位置误差/L、旋转角误差 |
| 局部动态 | 活动关节的角增量/位置步长 RMS、振幅保留比、主频偏差和有效频带能量 |
| 时序 | 目标与重建的时间偏移、肢体间相位/相关性 |
| 根运动 | XZ 轨迹误差、步长与加速度误差 |
| 骨架一致性 | GT 相对骨向量误差；单独查看短骨、末端与无效旋转输出 |
| loop | 跨 wrap 的位置步长、旋转增量、接缝异常 |
| 静止 | 相对 GT 的多余角速度和高频能量 |

活动关节与频带应由 GT 和固定规则定义，不由预测结果挑选。低能量目标使用绝对误差，避免接近零的分母制造巨大比例。真实 GT 本身的抖动是否属于数据噪声需要另行审核，不能无差别强化全部高频。

建议初始工程门槛：对有充分运动能量的 shake 关节，振幅比优先控制在 0.9–1.1，主频偏差不超过当前有效窗口的一个频率分辨单元；短窗或非周期动作不强套主频门槛。以上是待样本校准的验收起点，不是研究结论。姿态、静止噪声和未见骨架指标需预先固定可接受阈值，不能只凭振幅通过。

按动作类先求均值再汇总 macro 指标，同时保留按样本加权的总体指标；报告分位数、最差样本及有效样本数。导出原始/重建并排视频，保证同 fps、相同相位、相同相机。

### 8.3 决策规则

- **重建已糊**：检查输入是否缺信息、瓶颈容量、时间下采样、KL 和损失分配；暂停第二阶段。
- **训练清晰、val 模糊**：优先处理记忆、数据覆盖或泛化，不增加压缩率。
- **mu 清晰、sample 模糊**：检查 posterior 方差与正则化，不能仅用 mu 成绩批准 sample 训练。
- **AE 全部通过、latent 生成模糊**：重点检查第二阶段类别预算、噪声权重、条件和生成容量。
- **AE 不比现有路线更有效**：保留当前动作空间模型，不因架构名称而强制迁移。

## 9. 第二阶段的最小设计

冻结 E/D，encoder 使用 eval + no_grad，decoder 不进入训练图。VAE 基线使用 posterior sample 作为干净 latent，采样和统计约定固定；mu 训练作为独立消融。

保留当前 x0-prediction 的思路，只将加噪对象替换为标准化 latent：

```text
z0 = normalize(sample(E(x, s)))
zt = sqrt(alpha_bar[t]) * z0 + sqrt(1-alpha_bar[t]) * eps
z0_hat = G(zt, t, action_condition, skeleton_condition, length, loop_condition)
L_gen = mean_valid(w(t) * (z0_hat-z0)^2)
```

`mean_valid` 在 latent 的真实时间 mask 上聚合；不再使用动作关节 mask 归约 latent。空间槽 K 固定，t 由 latent 噪声日程定义。

G 使用 latent token 主干，不能为了兼容把 K 个槽伪装成有真实骨长和父子的关节。原图关系注意力、canonical 动作标准化、dpos/drot/bone 不直接用于 latent。

G 仍需骨架条件：不能假定 E 已完全剥离骨架身份。可将变长静态骨架 token 用 cross-attention 提供给 G；最终 D 同时读取完整目标骨架条件。先完成同骨架生成，再独立验证跨骨架参考迁移。

第二阶段继续使用经 AE 验证的编码前增强与尾部采样。在线增强需要在线编码；单个原始 clip 的缓存 latent 不能覆盖所有增强。缓存必须绑定 AE hash、canonical 版本、骨架、窗口、增强参数与 latent 统计版本。

训练只使用 latent MSE；验证时允许解码计算动作指标，不反向。生成结束后只解码一次。flow matching 可在此契约上作为独立后续对照，不与第一阶段首次实现绑在一起。

## 10. 参考迁移与局部编辑的兼容边界

第一版不承诺现有所有编辑能力无损迁移：

- 当前参考 z_ref 与新动作 latent z 是不同契约，维度即使相同也不能互换；第二阶段若加入参考条件，使用独立参考编码分支或显式适配器。
- 自重建学到的 z 不保证跨物种语义一致；参考交换需要独立验收，不用未经校正的自动 retarget 输出替代真实目标。
- 全关节池化槽一般不对应某个特定关节。原 joint inpainting 不能直接变成“mask 相同槽索引”。
- 帧对齐 latent 有利于设计时间编辑，但时序网络使修改一帧 latent 的影响可能跨帧，不能承诺原有边界精确性。
- 若局部编辑是发布必需项，必须在替换旧服务前补齐专门设计与验收；第一版生成实验通过不等于可直接替换生产推理。

## 11. 拟议实现拆分与 checkpoint

| 拟议文件 | 职责 |
|---|---|
| `model/motion_autoencoder.py` | 新 Encoder/Decoder、posterior、encode/decode 接口 |
| `train/motion_reconstruction_losses.py` | 从 diffusion 中提取纯动作重建损失及共享转换缓存 |
| `train/train_motion_autoencoder.py` | 第一阶段独立训练入口、采样与指标 |
| `train/eval_motion_autoencoder.py` | 固定重建、扰动、未见骨架评测与导出 |
| `model/latent_motion_denoiser.py` | 第二阶段 latent 主干与骨架条件 |
| `train/train_latent_anytop.py` | 第二阶段冻结 AE、latent 加噪与 MSE |
| `sample/generate_latent.py` | latent 采样、逆标准化、解码与现有导出适配 |

接口草案：

```python
posterior = ae.encode(x, skeleton, frame_mask)
z = posterior.sample()       # 验证另用 posterior.mean
x_hat = ae.decode(z, skeleton, frame_mask, loop_metadata)
losses = reconstruction_losses(x_hat, x, skeleton, frame_mask, loop_metadata)
```

第一阶段 checkpoint 至少记录：encoder/decoder 权重及可选 EMA、架构、latent 布局、canonical/channel 版本、归一化统计来源、VAE/KL 约定、数据 split/filter、增强配置、采样权重、优化器及训练状态。

latent 统计作为绑定到指定 AE checkpoint 的版本化产物保存。若采用 EMA，统计、缓存、第二阶段训练和推理必须使用同一套 AE 权重。

第二阶段 checkpoint 额外绑定 AE 的不可变 hash、latent mean/std 与采样约定、噪声日程、条件契约。加载不匹配时明确报错，不静默用另一套 E/D。AE 更新后必须重新编码或更新第二阶段，不能把旧 latent 模型接上新 decoder 直接发布。

## 12. 验证与性能评估计划

永久回归测试覆盖：有效 mask、极短长度、变关节数、batch padding 不影响有效重建、增强后骨架契约、loop/open 边界、完整旋转通道、checkpoint/hash 不匹配拒绝加载，以及共享损失提取后旧 diffusion 行为不变。

性能至少比较三条路径：

1. 现有动作空间 diffusion。
2. latent diffusion，仅 latent MSE。
3. 诊断用 latent diffusion + 解码后动作监督；不是第一版默认训练方案。

分别报告第一阶段训练成本、第二阶段在线编码/缓存成本、稳态每步时间、峰值显存、同模型调用次数及同墙钟预算下的生成质量。计时固定硬件、batch、长度、骨架组成、精度、compile 设置并排除编译预热。

表示压缩率不等于训练加速倍数；必须计算第一阶段摊销成本。一次解码较便宜也不能预先推导实际速度收益。

一次性诊断脚本、计时结果和 smoke 输出放在 agent-owned 临时目录；正式模型产物沿用项目训练输出约定，不将临时基准文件加入仓库。

## 13. 推荐执行顺序

1. 实现完整通道、无旁路、无时间下采样的 AE；完成小集合重建和接口回归。
2. 提取并迁移动作损失，完成混合数据与增强下的重建基线。
3. 做槽容量、KL、尾部采样消融，按 shake 和其他动作共同验收。
4. 冻结通过验收的 AE，发布 latent 统计与 checkpoint 契约。
5. 实现只用 latent MSE 的第二阶段，比较最终动作质量与端到端性能。
6. 有证据后再研究时间压缩、flow matching、分层 latent、局部编辑和跨物种参考迁移。

## 14. 参考

- [MLD: Executing Your Commands via Motion Diffusion in Latent Space](https://arxiv.org/abs/2212.04048)：动作 VAE + 潜空间扩散的代表方案，不证明其人体配置可直接用于多拓扑动物。
- [T2M-GPT](https://arxiv.org/abs/2301.06052)：离散动作表示路线，作为后续方案对照，非本文首版实现。
- [当前参考编码器设计](reference_encoder.md)：参考迁移目标与新保真压缩任务的区别。
- [当前模型结构](anytop_model_architecture.md)：现有主干与动作空间生成路径；细节以代码为准。
