# AnyTop：Motion Autoencoder 架构现状与 Latent Diffusion 后续设计

> Motion Autoencoder（AE）及共享动作重建损失已实现；Latent Diffusion 尚属后续设计。
> 本文说明当前 AE 的数据流、模块职责与接口边界，不维护模型容量、损失权重或训练配置。实现完成不代表重建质量或泛化能力已通过验收。

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

offsets、尺度、canonical 统计等数据由损失与物理转换链路使用，不作为额外动态内容传入模型。Encoder 和 Decoder 使用同一份增强后的骨架条件；名称及 rest geometry 的条件置空也共享结果。

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
| `train/train_motion_autoencoder.py` | 独立训练、固定窗口评估、latent 置零/交换诊断与 checkpoint |
| `sample/eval_autoencoder.py` | 显式预处理动作文件的整段重建、指标与导出 |
| `train_tiny_autoencoder.bat`、`eval_autoencoder.bat` | 训练与诊断启动入口 |

训练入口的固定窗口评估包含正常重建、latent 置零和交换对照，并记录姿态、动态、根运动及骨架误差。这些诊断用于检查瓶颈依赖，不直接证明表示解耦或未见骨架泛化。

整段诊断将输入动作重采样到 checkpoint 的内部窗口后重建，再恢复原输入帧数；Encoder 与 Decoder 使用同一份时长比例及循环条件。该流程中的重采样属于评估适配，不是 AE 的时间压缩层。

Checkpoint 保存模型架构、权重、latent 布局、canonical 与解码内容版本、条件数据校验信息及续训状态。当前解码内容版本为 `latent_time_v2`；缺少物理时间投影的旧版本不能直接续训或诊断加载。

当前实现完成的是自重建链路。跨骨架 latent 交换、未见骨架泛化、长尾动作细节和潜空间可生成性仍需实验验证；尚未包含 latent diffusion 主干、latent 标准化产物或最终 latent 生成入口。

## 6. Latent Diffusion 后续设计（未实现）

后续冻结 E/D，encoder 使用 eval + no_grad，decoder 不进入训练图。确定性 AE 直接使用编码结果；若采用 VAE，需明确使用 posterior sample 还是均值，并使训练与统计约定一致。

保留当前 x0-prediction 的思路，只将加噪对象替换为标准化 latent：

```text
z0 = normalize(ae.encode(x, s, frame_mask, time_metadata))
zt = sqrt(alpha_bar[t]) * z0 + sqrt(1-alpha_bar[t]) * eps
z0_hat = G(zt, t, action_condition, skeleton_condition, length, loop_condition)
L_gen = mean_valid(w(t) * (z0_hat-z0)^2)
```

`mean_valid` 在 latent 的真实时间 mask 上聚合；不再使用动作关节 mask 归约 latent。空间槽 K 固定，t 由 latent 噪声日程定义。

G 使用 latent token 主干，不能为了兼容把 K 个槽伪装成有真实骨长和父子的关节。原图关系注意力、canonical 动作标准化、dpos/drot/bone 不直接用于 latent。

G 仍需骨架条件：不能假定 E 已完全剥离骨架身份。可将变长静态骨架 token 用 cross-attention 提供给 G；最终 D 同时读取完整目标骨架条件。先完成同骨架生成，再独立验证跨骨架参考迁移。

第二阶段继续使用经 AE 验证的编码前增强与尾部采样。在线增强需要在线编码；单个原始 clip 的缓存 latent 不能覆盖所有增强。缓存必须绑定 AE hash、canonical 版本、骨架、窗口、增强参数与 latent 统计版本。

训练只使用 latent MSE；验证时允许解码计算动作指标，不反向。生成结束后只解码一次。flow matching 可在此契约上作为独立后续对照，不与第一阶段首次实现绑在一起。

### 6.1 参考迁移与局部编辑的兼容边界

第一版不承诺现有所有编辑能力无损迁移：

- 当前参考 z_ref 与新动作 latent z 是不同契约，维度即使相同也不能互换；第二阶段若加入参考条件，使用独立参考编码分支或显式适配器。
- 自重建学到的 z 不保证跨物种语义一致；参考交换需要独立验收，不用未经校正的自动 retarget 输出替代真实目标。
- 全关节池化槽一般不对应某个特定关节。原 joint inpainting 不能直接变成“mask 相同槽索引”。
- 帧对齐 latent 有利于设计时间编辑，但时序网络使修改一帧 latent 的影响可能跨帧，不能承诺原有边界精确性。
- 若局部编辑是发布必需项，必须在替换旧服务前补齐专门设计与验收；第一版生成实验通过不等于可直接替换生产推理。

## 7. 后续阶段的衔接约束

进入 latent diffusion 前，需要选定 AE checkpoint，并发布与该权重绑定的 latent 统计。标准化必须可逆，基于有效 token，明确确定性编码、VAE 均值或采样的分布约定；不对每段动作单独消除幅度与节奏。使用采样分布时，统计需包含 posterior 方差贡献。

第二阶段应与 AE 共用物理时间条件：训练时编码与去噪使用配对的时长比例；生成时将输出长度换算为内部窗口的时长比例，同时传给去噪网络与 Decoder，再恢复目标帧数。循环状态与拼接周期信息同步传递。

拟新增的模块包括 latent 去噪主干、冻结 AE 的训练入口及 latent 采样入口。第二阶段 checkpoint 与缓存需绑定 AE 权重、latent 统计和条件契约；AE 更新后不能直接沿用旧 latent 模型与缓存。

质量评估需同时覆盖姿态、局部振幅与频率、相位、根轨迹、循环接缝和静止噪声，按动作与骨架分组查看。AE 重建质量与 latent 生成质量分别验收；生成性能收益需计入编码、缓存和解码成本。

## 8. 参考

- [当前参考编码器设计](reference_encoder.md)：参考迁移表示与完整动作重建的区别。
- [当前模型结构](anytop_model_architecture.md)：现有主干与动作空间生成路径。
- [MLD: Executing Your Commands via Motion Diffusion in Latent Space](https://arxiv.org/abs/2212.04048)：后续连续潜空间扩散路线参考。
