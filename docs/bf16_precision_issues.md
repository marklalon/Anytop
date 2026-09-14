# bf16 混合精度：数值精度问题与方案

> 状态：调研完成（2026-09-13）；5.2 的 A + B + D 已实施（2026-09-13，见 5.5），C 待实验。第 2、4 节描述的是实施前的状态。
> 后续：训练已于 2026-09-13 改用 fp16（`train.bat`）。fp16 与 bf16 的逐层对比见 [fp16_vs_bf16_precision.md](fp16_vs_bf16_precision.md)——每层误差是 bf16 的 1/8，本文第 4 节的问题 1、3 相应降一个数量级。
> 测量对象：`save/merged_locomotion_v14/model000005000.pt`，locomotion 训练集 8 个 batch × 16 条（128 条样本）。
> 环境：RTX PRO 6000 Blackwell（WDDM），torch 2.11.0+cu128，与 `train.bat` / `generate.bat` 相同的参数。

## 1. 结论摘要

训练、推理、评估目前都跑在 bf16 autocast 下。实测发现四个问题：

| # | 问题 | 关键数字 | 严重程度 |
|---|---|---|---|
| 1 | 模型输出（x0 预测）精度损失 | 位置 p99 误差 0.99% L（1.8 m 人形约 4.7 mm），旋转 p99 0.17°；fp32+TF32 只有其 1/13 | 低：绝对值小 |
| 2 | 输出带白噪声式高频抖动，**评估指标被抬高** | 打分器的 jerk +52%、频谱平坦度 +64%、snap +8% | **高**：评估数字不可信 |
| 3 | 零初始化门控标量的梯度被舍入噪声主导 | `cross_limb_blocks.0.time_emb_scale` 梯度误差是真实梯度的 5.6 倍、minibatch 噪声的 4.4 倍 | 中：对训练质量的影响未验证 |
| 4 | bf16 下无法做数值等价性验证 | 13 个输出梯度元素 1e-8 级的变化 → 参数梯度相对差中位 2e-3 | 中：影响排查与重构验证 |

建议（第 5 节展开）：

1. **推理与评估改用 fp32+TF32**：前向只慢约 5%，直接消除问题 1、2 对输出和评估的影响。
2. **`OutputProcess` 固定在 fp32**：改动约 5 行，bf16 训练的输出误差降约 40%、梯度全局误差减半；速度代价不超过 4.4%。
3. **训练暂时保留 bf16**：fp32+TF32 训练慢 44%、显存多 73%。问题 3 先用短程 A/B 实验确认是否影响质量，再决定。
4. **数值验证一律在 fp32 确定性模式下做**（第 5.2 节 D）。

## 2. 现状：bf16 覆盖了哪些计算

> 本节是实施 A + B 之前的状态，实施后的变化见 5.5。

### 2.1 使用位置

| 场景 | 精度来源 |
|---|---|
| 训练 | `train.bat` 的 `--amp_dtype bf16`，`TrainLoop._autocast_context()`（[training_loop.py](../train/training_loop.py)） |
| 训练中评估 | `TrainLoop.evaluate()` 复用训练的 `_autocast_context()`，与 `--amp_dtype` 绑定 |
| 推理 | `generate.bat` 的 `--amp_dtype bf16`（[generate.py](../sample/generate.py) `_resolve_inference_amp_dtype`） |
| checkpoint 评估 | [eval_checkpoint.py](../eval/eval_checkpoint.py) 两处硬编码 `--amp_dtype bf16` |

### 2.2 autocast 下哪些算子在 bf16

- **bf16**：所有 `nn.Linear` / `matmul`，包括 FF 层、注意力 Q/K/V 投影、cross-limb block 的 `proj_in` / `proj_out`，以及最终输出投影 `OutputProcess`（[anytop.py](../model/anytop.py) `class OutputProcess`）。
- **已显式保护为 fp32**：
  - QK-norm（`QKNorm.forward` 把输入转 fp32 再做 `rms_norm`）；
  - 注意力 score 与 softmax（`SelectiveMultiheadAttention`、`GraphMultiHeadAttention` 中的 `.float()`）；
  - 所有 loss 计算（`GaussianDiffusion._fp32_math_context` 关闭 autocast）。
- **只是事后 `.float()`**：注意力输出投影（`F.linear(...).float()`）和图注意力的 `output_layer(x).float()`。这些线性层本身仍在 bf16 里算，结果转 fp32 前已经被量化，所以残差流虽然是 fp32，每个子层加进来的增量都带 bf16 舍入。
- **没有保护**：`OutputProcess`。模型的 x0 预测直接以 bf16 输出，进入 loss 前才转 fp32。

### 2.3 格式本身的精度

| 格式 | 尾数位 | 相对量化间隔 | 就近舍入最大相对误差 |
|---|---|---|---|
| bf16 | 7 | 0.39%–0.78% | 0.2%–0.39% |
| fp32 | 23 | ~1e-7 | ~6e-8 |
| TF32（仅 cuBLAS 矩阵乘内部，NVIDIA 定义为 10 位尾数，结果仍以 fp32 存储） | 10 | 0.05%–0.1% | 0.02%–0.05% |

bf16 的指数位与 fp32 相同，不会溢出，所以不需要 loss scaling。但它只有 7 位尾数，每次矩阵乘都会引入千分之几的相对舍入误差。这是下面所有问题的根源。

## 3. 测量方法（摘要）

- **参照组**：fp32、关闭 TF32、SDPA 只用 math 后端、`torch.use_deterministic_algorithms(True)`。同一配置跑两次结果逐位一致。
- **对比配置**（同一 batch、同一随机种子）：

  | 配置 | 说明 |
  |---|---|
  | fp32+TF32 | fp32，开 TF32 与快速 SDPA（`--amp_dtype fp32 --compile` 的实际形态） |
  | bf16 | 现状 |
  | bf16 + math SDPA | 与 bf16 结果相同，说明误差来自 dtype 本身，与注意力 kernel 选择无关 |
  | bf16 + 输出层 fp32 | 只把 `OutputProcess` 放进 fp32 |
  | bf16 + 输出层与 cross-limb fp32 | 再加上 4 个 `CrossLimbTemporalBlock` |

- **dropout 全部置 0**：dropout 的随机 mask 会随 SDPA 后端和 tensor dtype 变化，开着 dropout 时连 fp32+TF32 都与参照差 10%，这与精度无关。条件 drop、joint mask 等用 float32 随机数生成，与 dtype 无关，保留。
- **L 是骨架静息姿态关节位置的 RMS 散布**（`_length_scale_from_rest`）。260 个物种的 L/身高中位数为 0.318，KI_Human 为 0.263，所以 1.8 m 人形的 L ≈ 0.47 m。

## 4. 问题详述

### 4.1 问题 1：模型输出精度

训练步前向（128 条样本，覆盖全部 timestep），与 fp32 参照组的差：

| 配置 | 输出相对 L2 误差 | 位置 p99 | 位置最大 | 旋转 p99 | 旋转最大 | l_simple 相对差 |
|---|---|---|---|---|---|---|
| fp32+TF32 | 2.2e-4 | 0.067% L | 0.18% L | 0.020° | 0.055° | 7.8e-5 |
| **bf16（现状）** | **2.8e-3** | **0.99% L** | **2.5% L** | **0.17°** | **0.71°** | **6.3e-4** |
| bf16 + 输出层 fp32 | 1.75e-3 | 0.56% L | 1.65% L | 0.11° | 0.61° | 3.5e-4 |
| bf16 + 输出层与 cross-limb fp32 | 1.65e-3 | 0.55% L | 1.55% L | 0.11° | 0.59° | 3.3e-4 |

- 按 1.8 m 人形换算，bf16 的位置 p99 约 4.7 mm，最大约 12 mm。
- 误差和 timestep 无关（t<10：3.2e-3；10≤t<50：2.8e-3；t≥50：3.2e-3），所以推理的最后几步同样受影响。
- 只把最终输出投影放进 fp32，误差就降约 40%；剩下的来自网络内部各层的 bf16 舍入。

单看数值，这个误差不大：l_simple 只差万分之六。真正的问题是它的频谱形态，见 4.2。

### 4.2 问题 2：高频抖动与评估指标偏差

推理的最终输出就是最后一步（t=0）的 x0 预测。在 eval 模式下对 t=0 做一次前向：

| 配置 | 位置 p50 | 位置 p99 | 旋转 p99 | 误差的 jerk 能量 / 真值的 jerk 能量 |
|---|---|---|---|---|
| fp32+TF32 | 0.025% L | 0.062% L | 0.020° | 0.02% |
| **bf16（现状）** | 0.27% L | 1.0% L | 0.17° | **4.5%** |
| bf16 + 输出层 fp32 | 0.18% L | 0.55% L | 0.10° | 0.69% |
| bf16 + 输出层与 cross-limb fp32 | 0.17% L | 0.55% L | 0.10° | 0.54% |

输出层量化产生的是逐帧独立的白噪声。时间差分会成倍放大白噪声：3 阶差分的能量增益是 20 倍，4 阶是 70 倍；而真实运动的高阶导数很小。所以位置上毫米级的误差，在 jerk 上变成了真值 jerk 能量的 4.5%。从数据看，网络内部的舍入误差对 jerk 的贡献小得多（推测是经过时间注意力后偏低频），因此单独把输出层放进 fp32 就能把这个比例降到 1/6.5。

对评估打分器（[scorer.py](../eval/motion_quality/scorer.py) `_compute_features`，训练中评估和 `eval_checkpoint.py` 都用它）的实际影响：

| 相对 fp32 参照 | jerk_norm | snap_norm | 频谱平坦度 |
|---|---|---|---|
| fp32+TF32 | +0.2% | +0.1% | +0.9% |
| **bf16（现状）** | **+52%** | **+8%** | **+64%** |
| bf16 + 输出层 fp32 | +7% | +2% | +17.5% |
| （参考：真值相对 fp32 预测） | −42% | −42% | +30% |

表中是逐关节特征均值的变化。bf16 带来的偏差已经和"预测与真值之间的差距"是同一个量级，所以现在训练中评估和 `eval_checkpoint` 给出的 Jerk、Snap、SpectralFlatness 分量，有相当一部分反映的是 bf16 量化噪声，而不是模型本身。不同 checkpoint 之间横向比较也会受影响：量化噪声的大小取决于输出值的幅度，不是一个固定常数。

> 这组数字是在"真值加微量噪声后做一次 t=0 前向"的条件下测的。完整采样的最后一步是同一机制，但生成结果本身的 jerk 水平可能不同，偏差的百分比会随之变化，方向不变。

### 4.3 问题 3：门控标量的梯度被舍入噪声主导

整体梯度与 fp32 参照组的差距：

| 配置 | 全局相对误差 | 余弦相似度 | 逐张量相对误差 中位 / p90 | 误差 / minibatch 噪声（中位） |
|---|---|---|---|---|
| fp32+TF32 | 1.2e-3 | 0.9999993 | 0.11% / 0.15% | 0.0017 |
| **bf16（现状）** | **1.8e-2** | **0.99985** | **1.1% / 2.2%** | **0.017** |
| bf16 + 输出层 fp32 | 8.8e-3 | 0.99996 | 0.82% / 1.2% | 0.013 |
| bf16 + 输出层与 cross-limb fp32 | 7.7e-3 | 0.99997 | 0.75% / 0.99% | 0.011 |

"minibatch 噪声"是 fp32 下 8 个 batch 之间逐张量梯度的标准差，它和梯度均值之比的中位数是 0.77。对绝大多数参数张量，bf16 约 1% 的梯度误差比 SGD 本身的 minibatch 噪声小约 60 倍（中位比值 0.017），没有影响。

问题集中在 `CrossLimbTemporalBlock` 里零初始化的门控标量（`nn.Parameter(torch.zeros(1))`，不做 weight decay，负责逐步"打开"新加的通路）：

| 参数 | bf16 误差 / 真实梯度 | bf16 误差 / minibatch 噪声 | fp32+TF32 误差 / 真实梯度 |
|---|---|---|---|
| `cross_limb_blocks.0.time_emb_scale` | **559%** | **4.4×** | 4.5% |
| `cross_limb_blocks.2.cross_k_scale` | 38% | 0.37× | 8.0% |
| `cross_limb_blocks.3.reliability_bias` | 19% | 0.15× | 4.5% |
| `cross_limb_blocks.2.time_emb_scale` | 12% | 0.12× | 1.5% |
| `cross_limb_blocks.3.temporal_reliability_bias` | 12% | 0.07× | 0.5% |

**机制（推断）**：标量门的梯度是上千万个"上游梯度 × 激活"乘积之和。门接近平衡位置时，这些项大量正负相消，真实梯度很小；而每一项里 bf16 舍入带来的千分之几的误差互相独立，不会相消，于是在总和里占了主导。

**选择性 fp32 救不回来**：把 cross-limb block 整体放进 fp32 后，`time_emb_scale` 的误差只从 559% 降到 173%，`cross_k_scale` 没有改善。舍入噪声主要来自上游主干层 bf16 算出的激活和梯度，只有整体 fp32 才能根治。

**对训练质量的实际影响尚未验证**。AdamW 的二阶矩会吸收一部分噪声，但信号被噪声主导时，门的有效学习速度会变慢，轨迹会随机游走。这些门正好对应 cross-limb 可靠性修复（CKPT v10）新开的通路，需要实验确认（5.3 节）。这组数据还有两个局限：只来自一个 5000 步的 checkpoint，而且只用了 8 个 batch。

### 4.4 问题 4：bf16 下无法做数值等价性验证

在训练步性能优化中，FK loss 的链式累乘改成了指针倍增，结合顺序不同。结果是 loss 对模型输出的梯度在 115.2 万个元素里只有 13 个发生变化，最大差 1.5e-8（梯度最大值是 1.3e-4）。

同样的对比：

- **bf16 下**：参数梯度逐张量相对差（按最大绝对值计）中位数 2.1e-3，最差 7.7e-3。
- **fp32 确定性模式下**：中位数 1.5e-7，最差 8e-7，纯属 float32 舍入。

原因是 bf16 的舍入在任何比特级扰动下都会重新"洗牌"。同一份代码在默认（非确定性）模式下连续跑两次，梯度也会出现同类差异。所以在 bf16 下看到 1e-3 级的梯度差异，既证明不了改动有问题，也证明不了没问题。

## 5. 方案

### 5.1 方案对比

| 方案 | 解决 | 训练速度 | 推理速度 | 显存 | 需要重训 |
|---|---|---|---|---|---|
| A. 推理与评估改 fp32+TF32 | 问题 1、2 在推理和评估侧 | 不变 | +5%（38.9 → 40.9 ms/前向） | 未测 | 否 |
| B. `OutputProcess` 固定 fp32 | 问题 1、2 的大部分；问题 3 小幅改善 | ≤ +4.4%（未单独测，是 C' 的子集） | ≈ 不变 | ≈ 不变 | 否 |
| C. 训练整体 fp32+TF32 | 问题 1–4 全部 | **+44%**（156.0 → 224.9 ms/步，6.41 → 4.45 it/s） | — | **+73%**（19.4 → 33.6 GiB） | 否（权重本来就是 fp32），但训练动态会变 |
| C'. bf16 + 输出层与 cross-limb fp32 | 同 B，问题 3 只有少量改善 | +4.4%（162.9 ms/步） | ≈ 不变 | +2% | 否 |
| D. 数值验证规范 | 问题 4 | — | — | — | 否 |

速度都在编译模式、真实数据加载器下测得。推理按 `generate.bat` 的形态测：eager、batch 8、单次前向。

### 5.2 建议立即做：A + B + D

**A. 推理与评估改 fp32+TF32**

- 把 `generate.bat` 和 `eval_checkpoint.py` 两处的 `--amp_dtype bf16` 改为 `fp32`。
- `TrainLoop.evaluate()` 的采样不再复用训练的 `_autocast_context()`，固定用 `torch.autocast(device_type, enabled=False)`。`_compute_eval_losses` 可以保持现状。
- fp32 推理要显式开 TF32（`torch.set_float32_matmul_precision('high')`）。目前只有 `--compile` 训练路径在 `_compile_forward_model` 里设置了它，`generate.py` 没有设置；上面 +5% 的数字是开了 TF32 测得的。
- TF32 的误差（t=0 位置 p99 0.062% L，jerk 能量 0.02%）远低于可感知和打分器敏感的量级。

**B. `OutputProcess` 固定 fp32**

```python
def forward(self, output):
    # The x0 prediction must not be quantized to bf16: its per-frame rounding is
    # white noise that time derivatives amplify (docs/bf16_precision_issues.md).
    with torch.autocast(device_type=output.device.type, enabled=False):
        output = output.float()
        root_data = self.root_dembedding(output[:, :, 0])
        all_joints = self.joint_dembedding(output[:, :, 1:])
    ...
```

- 只涉及两个 256→12 的线性层，计算量可以忽略，与 checkpoint 完全兼容。
- 对 bf16 训练：输出误差降 40%，l_simple 相对差从 6.3e-4 降到 3.5e-4，梯度全局误差减半。
- 对仍在用 bf16 的推理：jerk 偏差从 +52% 降到 +7%。

**D. 数值验证规范**

凡是"改动前后结果是否一致"的验证，统一按下面的设置做：

1. `--amp_dtype fp32`，`torch.backends.cuda.matmul.allow_tf32 = False`；
2. `torch.use_deterministic_algorithms(True)`，设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
3. SDPA 只开 math 后端；
4. 所有 dropout 置 0，或确认两边 dropout mask 相同；
5. 每次运行前重设 torch / cuda / numpy / random 种子；
6. 先让同一份代码跑两次，确认逐位一致，再做新旧对比。

bf16 只用于最终的速度测量。

### 5.3 建议通过实验再决定：C

- **对照**：从同一 checkpoint（或从头、同一种子）分别用 bf16 和 fp32+TF32 训练 2–3 万步，其余参数与 `train.bat` 相同。两边都先落地 B。
- **记录**：
  1. 各物种族的 `l_simple_*`、`fk_angle_deg`、`loop_*` 指标曲线；
  2. 第 4.3 节表中门控标量的取值轨迹（需要额外加日志）；
  3. 按方案 A 在 fp32 下算的评估 Score。
- **判定**：
  - fp32 组的指标或门控标量轨迹与 bf16 组的差异超出同精度不同种子的波动，才考虑换 fp32。
  - 可以考虑折中做法：大部分步数用 bf16，最后 10–20% 的步数切到 fp32 收尾，只多付这一段的 44% 速度代价。
  - 差异不显著，就保持 bf16。

### 5.4 不建议：C'

C' 比 B 多付约 4% 速度，前向和输出抖动上的收益与 B 几乎相同（t=0 jerk 能量 0.54% 对 0.69%）；它想解决的门控标量梯度问题也没解决（第 4.3 节）。

### 5.5 实施记录：A + B + D（2026-09-13）

| 方案 | 落点 |
|---|---|
| A | `generate.bat` 改 `--amp_dtype fp32`；[eval_checkpoint.py](../eval/eval_checkpoint.py) 两处改为共用常量 `_COMMON_GENERATE_ARGS`（fp32）；`TrainLoop.evaluate()` 的采样固定 `torch.autocast(..., enabled=False)`；[generate.py](../sample/generate.py) `_resolve_inference_amp_dtype` 在 CUDA 上设 `set_float32_matmul_precision('high')` |
| B | [anytop.py](../model/anytop.py) `OutputProcess.forward`，与 5.2 的代码一致 |
| D | 新增 [utils/numerical_verification.py](../utils/numerical_verification.py)：`enable_numerical_verification_mode()`（第 1–3 步）、`disable_dropout(model)`（第 4 步）、`reseed(seed)`（第 5 步）；`--amp_dtype fp32` 和第 6 步由调用方负责 |

实施中多发现并处理了两处：

- **`fixseed()` 会关掉 TF32。** [utils/fixseed.py](../utils/fixseed.py) 原来强制 `matmul.allow_tf32 = False`，而 `generate.py` 在每个 batch 采样前都调用 `fixseed(seed)`。只在加载运行时开 TF32 的话，会被第一个 batch 悄悄关掉。现在 `fixseed` 只设种子（和 `cudnn.benchmark = False`），TF32 由各调用方的精度策略决定。对训练没有影响：eager 路径的 `matmul.allow_tf32` 默认本来就是 False，`--compile` 路径在 `fixseed` 之后自己设 `'high'`；`cudnn.allow_tf32` 只影响卷积，模型里没有卷积。
- **`eval_checkpoint` 的增量复用看不到精度变化。** 任务校验和原本只覆盖每个任务自己的参数，`--batch_size` / `--amp_dtype` 不在内（docstring 写的是在内）。只改 dtype 的话，旧的 bf16 结果会被当作有效输出直接复用、重新打分。现在 `_COMMON_GENERATE_ARGS` 也计入校验和，所以**之前跑过的 checkpoint 评估目录在下次运行时会全部重新生成**。例外：早于校验和功能（2026-05-31）、没有 `task_params.json` 的旧目录仍按原规则复用。

验证：

- 新增 [tests/test_bf16_precision_policy.py](../tests/test_bf16_precision_policy.py)（12 项）；全量测试 833 项通过。
- 用本文的 checkpoint 实跑 `generate.py`（Buffalo，batch 8，DDPM 100 步）：fp32 下 100 次模型调用全部是 TF32 `'high'`、autocast 关闭；bf16 下 `OutputProcess` 的输入和输出都是 float32。
- bf16 autocast 下 `torch.compile` `OutputProcess`：0 个 graph break，输出为 float32，与 eager 逐位一致。

影响：

- 训练中评估的 Score（尤其 Jerk、Snap、SpectralFlatness 分量）和 `eval_checkpoint` 报告，与实施前的数字不可直接比较。
- B 改变了 bf16 训练的数值（输出误差降约 40%，见 4.1），但不改变 checkpoint 格式，不需要重新生成数据，也不强制重训。

## 6. 附录：测量细节

- **梯度对比**：每个配置用同一 batch、同一种子（t 采样、噪声、joint / temporal mask 全部相同）跑一次 `training_losses` + `backward`，在优化器更新前取梯度，与参照组逐张量比较 L2 相对误差。
- **minibatch 噪声**：参照组在 8 个 batch 上的逐张量梯度标准差，除以 8 个 batch 梯度均值的范数。
- **位置和旋转误差**：两边输出都经 `canonical_to_physical_hml` 解码后比较；位置用 L 归一，旋转用 6D → 矩阵后的测地角。只统计有效关节（去掉 padding）。
- **jerk 能量比**：`Σ|Δ³(误差位置)|² / Σ|Δ³(真值位置)|²`，只统计有效关节。
- **打分器特征**：直接调用 `eval/motion_quality/scorer.py` 的 `_compute_features`（`nperseg = min(64, T)`），逐样本逐关节计算后比较。
- **速度**：编译模式稳态，预热 150 步，每个配置 3–5 个块、每块 60 步取中位数；显存是 `torch.cuda.max_memory_allocated()` 的峰值。
- **一个已排除的干扰**：开 dropout 时，SDPA 后端（math 与 flash / mem-efficient）和 tensor dtype 都会改变 dropout mask，导致不同配置之间出现约 10% 的输出差异，与精度无关。
