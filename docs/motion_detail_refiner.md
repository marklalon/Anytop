# 运动残差校准 Refiner

Refiner 是冻结扩散主干之后独立训练的运动校准器。它读取主干生成的动作、骨架、动作条件和独立时序噪声，一次前向预测非根关节的局部旋转残差，使输出的全频段运动统计接近同条件 GT。训练和推理均不更新主干。

当前保留模型为 `save/detail_v67_v3_200/refiner_latest.pt`，底座为 `save/zoo_locomotion_tiny_v67/model000010000.pt`（优先 EMA），缓存为 `outputs/cache/detail_v67`。正式验收 **121/124 条件组通过，整体 passed=false**，它是当前选用的候选，尚未完成动作质量验收。推理需显式传入权重路径。

## 快速开始

在 `Anytop` 目录、安装项目依赖的 Python 环境中执行。底座、条件文件和采样配置必须与缓存及 checkpoint 的契约一致。

### 使用现有模型

```bat
python -m sample.generate --model_path save/zoo_locomotion_tiny_v67/model000010000.pt --object_type Eagle --action_label "fly, forward" --refiner_path save/detail_v67_v3_200/refiner_latest.pt --output_dir outputs/refiner/generated
```

`--refiner_alpha` 在 0–1 之间控制修正强度，`--refiner_seed` 控制独立噪声。不传 `--refiner_path` 或设 `--refiner_alpha 0` 时直接返回基础动作。启用后，`refiner_comparisons/` 按相同的文件名保存主干（未细化）动作的 `.npy` 与 `.bvh`，每次采样的复现信息（两个种子、alpha、refiner 路径与哈希、逐样本长度/循环/残差幅度）写入同目录的 `batch_XXXXX.json`；常规导出使用细化动作。

支持普通单骨架和 all-species 生成。reference、img2img、inpaint、outpaint 的细化尚不支持。

### 缓存与训练

已有缓存可以直接复用。需要重建时选择一个空目录：

```bat
python -m tools.cache_motion_detail --cache-dir outputs/cache/detail_v67_new
```

缓存只使用原生 `train` split，每条件默认生成 4 个训练主干种子、1 个保留验证种子，记录底座、条件文件、采样器与时间契约及文件指纹。生成可通过 `--resume` 恢复；更换底座或采样配置需使用新的缓存。

从头训练，选择空保存目录：

```bat
python -m train.train_motion_detail --cache-dir outputs/cache/detail_v67 --save-dir save/refiner_new
```

默认配置对应当前选用方案：

| 配置 | 默认值 |
|---|---|
| 网络宽度 / 层数 | 64 / 3 |
| 每条件更新数 | 200 |
| 学习率 | 0.0003 |
| 日程 | cosine，预热 2 次条件遍历，最低学习率为峰值的 1% |
| 旋转残差上限 | 0.30 rad，每关节整窗口统一缩放 |
| 统计分桶边界 | 4.5 / 6.75 Hz |
| 残差输出频率上限 | 参考 30 fps 的 Nyquist，即 15 Hz；导出时再取实际上限的较小值 |
| 损失权重 | stat / keep / geo / identity = 1，reg = 0.01 |
| 周期验证 | 关闭（`--eval-every 0`）；训练后独立评估 |

总更新数为条件组数乘以每组预算。训练只读取缓存和 refiner，不加载主干。默认批量处理同一条件下的主干种子，在 CUDA 上使用 CUDA Graph；`--serial-groups` 可运行逐样本参考路径。

同一保存目录再次运行命令会自动续训；`--resume PATH` 可以从指定 checkpoint 恢复。恢复时检查缓存、架构、损失权重、条件顺序和已完成预算，并恢复优化器、随机数与采样进度。延长已完成的 cosine 训练保留原日程，在最低学习率继续；需要更长的完整 cosine 日程时应从头训练。

`--eval-every N` 可启用周期及末步验证。训练参数、每步 JSONL、TensorBoard 和 `refiner_latest.pt` 保存在保存目录。判断收敛优先看 `train/epoch_*` 的完整条件遍历均值；`grad_norm` 为裁剪前范数，`grad_clip_scale` 和 `epoch_clip_fraction` 反映梯度裁剪情况。梯度余弦与冲突率仅用于诊断，优化采用加权目标的梯度求和。

### 评估

```bat
python -m eval.eval_motion_detail --cache-dir outputs/cache/detail_v67 --refiner save/detail_v67_v3_200/refiner_latest.pt --output outputs/refiner/evaluation.json
```

评估使用保留主干种子和至少两个未参与训练的细节种子，按条件报告基础误差、细化误差、GT 匹配输入变化、几何、边界及种子敏感性。阈值来自训练前复制到保存目录的 `acceptance.json`；维护源为 [motion_calibration_acceptance.json](motion_calibration_acceptance.json)，可在新训练时通过 `--acceptance` 指定。

## 模型与运动表示

```text
冻结主干 → 基础动作 x_base
                    ↓
动作 + 骨架 + 条件 + 独立时序噪声
                    ↓
时序卷积与骨架邻接传播网络
                    ↓
全频段旋转残差 → 输出频率限制 → 整窗口幅度控制
                    ↓
局部旋转合成 + 当前骨向量运输 → 最终动作 x_out
```

公开张量布局为 `[B, J, F, T]`。网络按有效关节和帧处理变长骨架及窗口，读取局部旋转及速度、根运动上下文、rest offsets、结构/关节嵌入、动作条件、真实时间尺度和 loop 标记。另有 102 维输入动作上下文：身体坐标系全局旋转与观测位置及速度 18 维，以及局部旋转、身体全局旋转、虚拟 FK 位置的均值和三频带 RMS 共 84 维。上下文不读取 GT、clip ID 或目标统计。

输出为有效非根关节每帧三维切空间旋转残差，按 `R_out = R_base × Exp(alpha × delta)` 合成。根轨迹、根旋转、padding 和未用通道保持原值。频率按真实时间步长计算；loop 使用 FFT，开放窗口使用偶延拓的 DCT 边界。4.5 Hz 只是统计分桶边界，残差允许 DC、低频和高频修正。

### 位置与导出契约

实际位置使用 `current_bone_transport_v2`：用基础全局旋转将当前每帧实际骨向量变换到父关节局部坐标，再按细化后的父关节全局旋转传播。运输使用最终合成 6D 旋转的实际解码结果，零残差在非单位 6D 输入上也保持原 tensor。

该操作保持输入每帧的骨长，包括输入已有的骨长变化，不修复基础动作的骨长错误。RIC 根 XZ 是世界轨迹，其余关节 XZ 已是根相对量；计算身体骨向量时先清零根 XZ。

普通与 all-species 导出均在最终重采样网格上重新执行骨运输，再进入 `working_to_stored`，保证导出几何相对于重采样后的基础动作保持一致。

## 训练目标

```text
L = stat + keep + geo + 0.01 × reg + identity
```

- **stat**：相位无关的条件统计匹配，观察非根局部旋转、身体坐标系全局旋转及虚拟 FK 位置。特征包括 DC、三个频带的逐轴 RMS、包络分位数、父子/同父兄弟协方差和跨频带包络耦合。虚拟位置由旋转及 rest offsets 计算，使监督量处于旋转 refiner 的可控范围。
- **keep**：相对于基础动作的全频段位置、局部及全局旋转改动代价。所有频段使用相同代价。
- **geo**：输入与输出骨长差的均方，主要监测运输实现的数值误差。边界加速度另作诊断，不将开放窗口端点修正为零或循环首末修正相等作为训练目标。
- **reg**：切空间残差平方代价。
- **identity**：将训练 GT 作为已匹配输入，约束统计与物理改动，并同时训练完整窗口、2/3 与 1/2 中心裁剪。位置/尺度 RMS 和局部旋转矩阵 RMS 超过 0.004 时增加软惩罚；裁剪保留原物理 dt，按开放窗口处理，各裁剪权重为 0.5。

同一 clip 的多个窗口先合并为一个统计观测。只有一条独立 GT 时使用 robust moment；有多条独立 clip 时使用多带宽 RBF MMD 加 robust moment。GT RMS 归一化下限为 0.02。每个条件独立计算目标，不混合不同骨架、动作和时间设置。

独立生成样本与 GT 不做逐帧波形重建。GT identity 的软惩罚也不保证所有未知输入严格满足验收阈值。

## 当前验收状态与边界

保存目录中的 `acceptance_evaluation.json` 是当前模型的正式验收报告，使用细节种子 `2**40` 与 `2**40+1`。124 组中有 3 组未通过：

| 条件 | 未通过项 |
|---|---|
| DNL_Leonopteryx，roll/fly/left | 统计改善不足 5% |
| DNL_Leonopteryx，roll/fly/right | 统计改善不足 5% |
| DNL_FireDragon，fly/forward，speed≈0.6833 | GT 局部旋转变化约 0.0059，超过 0.005 |

当前候选训练 24,800 步；原生长度导出的骨长增量均值约 `4.66e-8`，根误差为零。未参与训练的 42/52 帧裁剪保真检查为 114/124 组通过，仍有尾部超限。

缓存共有 124 组，其中 121 组只有一条独立 GT，3 组有两条。保留主干/细节种子的评估衡量采样泛化，不能证明未见 GT clip 的质量或真实分布覆盖。统计相似也不能代替相位、协调、动作语义、蒙皮及最终导出动画的人工检查。更换底座、采样契约或幅度预算需重新评估。

## 代码导航与保留文件

| 文件 | 职责 |
|---|---|
| `model/motion_detail_refiner.py` | 配置、网络、SO(3) 合成、频率投影与骨运输 |
| `sample/detail_refinement.py` | checkpoint/底座契约校验、推理接入、最终网格骨运输 |
| `tools/cache_motion_detail.py` | 冻结主干采样与 train split 缓存 |
| `train/train_motion_detail.py` | 数据准备、GT 裁剪、训练与续训入口 |
| `train/detail_objectives.py` | 统计特征、目标聚合、最小改动与保真代价 |
| `train/detail_runtime.py` | 同条件批量训练，与公开逐样本路径保持一致 |
| `train/detail_cuda_graph.py` | CUDA Graph 捕获、重放与 eager 回退 |
| `train/detail_gradients.py` | 加权梯度求和及梯度诊断 |
| `train/detail_checkpoint.py` | 条件调度、随机状态、续训和学习率日程 |
| `train/detail_metrics.py` | 物理变化、身体振幅、骨长与边界指标 |
| `eval/eval_motion_detail.py` | 保留种子验收与逐条件报告 |

唯一支持的 checkpoint 架构标识为 `calibration_version=3`。固定的合成、运动上下文和几何字段作为契约元数据保存在 checkpoint，加载和续训不会自动解释其他架构。

本地保留 `save/detail_v67_v3_200/` 中的权重、参数、阈值、训练日志、TensorBoard 和正式验收报告，以及 `outputs/cache/detail_v67/` 的冻结缓存。回归测试覆盖旋转与骨长保持、根与 padding、统计不变性、批量/逐样本梯度、CUDA Graph、真实训练、续训及导出。
