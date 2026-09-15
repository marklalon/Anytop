# fp16 与 bf16 混合精度：逐层精度对比

> 状态：测量完成（2026-09-14）；第 5 节的两个问题已按第 9 节的 A + B + C + D 修复（2026-09-14）。
> 第 3–5 节描述的是修复前的状态。
> 承接 [bf16_precision_issues.md](bf16_precision_issues.md)，该文只比较了 bf16 与 fp32；
> `train.bat` 现在跑 `--amp_dtype fp16`（commit 80b99ef），本文测的是 fp16 与 bf16 之间的差别。
> 测量对象：`save/merged_locomotion_v14_fp16/model000200000.pt`（主）与 `save/merged_locomotion_v14/model000005000.pt`（早期训练对照）。
> 每个 checkpoint 4 个 batch × 8 条，locomotion 训练集，参数同 `train.bat`。
> 环境：RTX PRO 6000 Blackwell，torch 2.11.0+cu128。

## 1. 结论摘要

| # | 项目 | fp16 | bf16 | 说明 |
|---|---|---|---|---|
| 1 | 逐层"层内"量化误差 | 2–6e-4 | 2–5e-3 | 每一层都是 **7.4–8.2 倍**，与尾数位数 10 vs 7 完全吻合 |
| 2 | 模型输出累积误差（相对 L2） | 2.3e-4 | 1.9e-3 | p99 逐元素误差 0.16% vs 1.4%（相对输出均值） |
| 3 | 参数梯度全局相对误差 | 3.1e-3 | 1.8e-2 | 余弦 0.9999953 vs 0.9998433；fp32 重跑噪声底 1.0e-7 |
| 4 | 零初始化门控标量的梯度误差 | 0.05%–1.2% | 1.7%–8.7% | bf16 的"问题 3"在 fp16 下降一个数量级 |
| 5 | 前向溢出风险 | 余量 ×3272 | 无 | 最大激活 20.0，fp16 上限 65504 |
| 6 | 反向溢出/下溢 | **依赖 GradScaler** | 无 | 关掉 loss scale 后 fp16 梯度误差 0.34（不可用） |
| 7 | 步时 / 显存 | 241 ms / 21.4 GiB | 248 ms / 21.4 GiB | eager、bs=16、40 步；fp32 为 364 ms / 32.8 GiB |

一句话：**在 AnyTop 的每一层上，fp16 的数值误差都是 bf16 的 1/8，代价和显存一样，前提是 GradScaler 必须开着**。
唯一属于 fp16 的新风险是下溢，它只在"梯度小到 1e-10 量级"的通路上出现——也就是刚开始训练、零初始化门还没打开的那几千步（第 5.2 节）。

## 2. 测量方法

同一个 batch、同一个 t、同一份噪声、同一套 mask，跑四种配置的完整训练步（前向 + 反向）：

| 配置 | 说明 |
|---|---|
| fp32（参照） | 无 autocast，TF32 关闭 |
| fp16 | `torch.autocast(float16)` + loss scale 2^17（实跑 run 的稳定值，见 5.1） |
| fp16_noscale | 同上但 loss scale = 1，用来量化 GradScaler 的作用 |
| bf16 | `torch.autocast(bfloat16)` |

- **层内（隔离）误差**：在 fp32 参照跑的 forward hook 里，把该层**原封不动的 fp32 输入**再喂给它一次，外面套 fp16 / bf16 autocast，直接比较该层自己的输出。这样得到的是这一层自身的量化贡献，不含上游累积。
- **累积误差**：同一层在完整 fp16 / bf16 跑里的输出，与 fp32 参照跑同一层输出相比。
- dropout 全部置 0（`utils/numerical_verification.disable_dropout`）；条件 drop / joint mask 用 fp32 随机数，与 dtype 无关，保留，每个配置前重设种子。
- **可复现性底噪**：fp32 跑两次，参数梯度相对差 **5e-8 ~ 1.8e-7**。下面所有 1e-3 量级的数字都远在底噪之上。
- 155 个模块被 hook（每个 `nn.Linear` / `nn.LayerNorm` / 两种注意力 / cross-limb block / InputProcess / OutputProcess）。
- `InputProcess` 容器本身不做隔离重放：它内部要抽 whole-joint name-drop 掩码，重放会换一套掩码，比的就不是 dtype 了。它的各个子线性层是纯函数，单独测。

脚本：`precision_probe.py` / `analyze_probe.py`（本次放在会话临时目录，未入库；需要复用再搬进 `tools/`）。

## 3. 前向：逐层精度

### 3.1 层内量化误差（相对 L2，4 batch 中位数，200k checkpoint）

| 层 | 实跑 dtype | fp16 | bf16 | bf16/fp16 |
|---|---|---|---|---|
| `input_process.joint_embedding` | fp16 | 4.06e-4 | 3.29e-3 | 8.1 |
| `input_process.text_embedding` | fp16 | 3.52e-4 | 1.80e-3 | 5.1 |
| `input_process.struct_embedding.0` | fp16 | 3.54e-4 | 2.84e-3 | 8.0 |
| `layers.N.embed_timesteps` | fp16 | 3.16e-4 | 2.48e-3 | 7.9 |
| `layers.N.spatial_attn`（整块） | fp32 输出 | 6.03e-4 | 4.46e-3 | 7.4 |
| `layers.N.spatial_attn.linear_q/k` | fp16 | 2.9e-4 | 2.1e-3 | 7.4 |
| `layers.N.spatial_attn.linear_v` | fp16 | 4.18e-4 | 3.10e-3 | 7.4 |
| `layers.N.temporal_attn`（整块） | fp32 输出 | 4.57e-4 | 3.66e-3 | 8.0 |
| `layers.N.linear1` | fp16 | 2.42e-4 | 1.93e-3 | 8.0 |
| `layers.N.linear2` | fp16 | 3.18e-4 | 2.53e-3 | 8.0 |
| `cross_limb_blocks.N`（整块） | fp32 输出 | 1.53e-4 | 1.20e-3 | 7.8 |
| `cross_limb_blocks.N.cross_k_attn` | fp32 输出 | 4.41e-4 | 3.62e-3 | 8.2 |
| `layers.N.norm1/2/3`、`norm_cl`、`cross_k_norm` | fp32 | **0** | **0** | — |
| `output_process`（整体） | fp32 | **0** | **0** | — |
| `output_process.*_dembedding`（假设不钉 fp32） | — | 2.4e-4 | 1.9e-3 | 8.0 |

- **比值在所有层上都是 7.4–8.2**，没有任何一层对 fp16 更不利。bf16 尾数 7 位、fp16 尾数 10 位，相对量化间隔差 8 倍，实测就是这个数。
- 归一化层（LayerNorm、QKNorm）误差**严格为 0**：autocast 把它们留在 fp32，两种 dtype 下走的是同一段代码。
- `output_process` 整体为 0，是 [bf16_precision_issues.md](bf16_precision_issues.md) 5.2-B 那次修复的效果（`OutputProcess.forward` 自己关 autocast）。表里最后一行是"如果不钉 fp32 会付多少"的反事实值——fp16 下这一行的代价也只有 bf16 的 1/8。
- 5000 步 checkpoint 上重测，同一张表逐行吻合（比值 7.4–9.3），说明这是格式属性，与训练阶段无关。

### 3.2 累积误差与深度

每层 `norm3` 输出（残差流）相对 fp32 参照：

| 解码层 | fp16 | bf16 | bf16/fp16 | 该层 absmax |
|---|---|---|---|---|
| 0 | 4.01e-4 | 3.11e-3 | 7.7 | 5.95 |
| 2 | 4.78e-4 | 3.68e-3 | 7.7 | 9.50 |
| 4 | 5.05e-4 | 3.98e-3 | 7.9 | 10.42 |
| 6 | 4.02e-4 | 3.19e-3 | 7.9 | 9.78 |
| 7 | 1.93e-4 | 1.50e-3 | 7.8 | 8.48 |

误差随深度只在 0→4 层缓慢上升然后回落，两种 dtype 的比值全程稳定在 7.7 左右——**误差不会因为层数多而对某一种格式不利**。

模型最终输出（`output_process`）：fp16 相对 L2 **2.32e-4**，bf16 **1.88e-3**；逐元素 p99（以输出均值归一）fp16 0.16%、bf16 1.37%。
bf16 这个数与旧文 4.1 节实施 B 之后的 1.75e-3 一致，互为交叉验证。

### 3.3 前向动态范围（fp16 是否会溢出）

| 项 | 值 |
|---|---|
| 全网最大激活 | 20.0（`layers.0.spatial_attn.linear_q`），其次 17.4（`cross_limb_blocks.3.norm_cl`） |
| fp16 余量 | ×3272 |
| fp16 / bf16 前向非有限值计数 | 0 / 0 |

注意力 logits 有 QK-norm 约束、score 与 softmax 显式 fp32，loss 在 `_fp32_math_context` 里算，所以前向没有任何接近 65504 的地方。**fp16 的前向溢出在这个模型上不是风险**。

## 4. 反向：参数梯度

### 4.1 全局

200k checkpoint（4 batch 中位数）：

| 配置 | 全局相对 L2 | 余弦 | 逐张量中位 | 逐张量 p90 |
|---|---|---|---|---|
| fp32 重跑（底噪） | 1.0e-7 | 1.0000000 | 1.5e-7 | 2.5e-7 |
| **fp16（scale 2^17）** | **3.1e-3** | **0.9999953** | **2.5e-3** | **3.9e-3** |
| **bf16** | **1.8e-2** | **0.9998433** | **1.7e-2** | **2.5e-2** |
| fp16 无 loss scale | 3.4e-1 | 0.9416615 | 5.8e-1 | 9.8e-1 |

5000 步 checkpoint：fp16 1.1e-3 / bf16 7.9e-3 / fp16 无 scale 2.0e-1，比值同样约 7.5。

旧文 4.3 节把 bf16 梯度误差与 minibatch 噪声比，结论是"对绝大多数张量无影响，问题集中在零初始化门控标量"。fp16 把两头都改善了：

| 参数 | bf16 | fp16 | 倍数 |
|---|---|---|---|
| `action_label_null_emb` | 8.7e-2 | 3.3e-3 | 26.7 |
| `cross_limb_blocks.1.cross_k_scale` | 6.8e-2 | 2.8e-3 | 24.2 |
| `cross_limb_blocks.0.reliability_bias` | 5.7e-2 | 1.1e-2 | 4.9 |
| `cross_limb_blocks.3.time_emb_scale` | 2.6e-2 | 2.3e-3 | 11.3 |
| `layers.3.temporal_phase_scale` | 2.3e-2 | 1.7e-3 | 13.4 |

这些标量的梯度是上千万个正负相消项之和，舍入噪声不相消、在总和里占比被放大，所以它们的相对误差比普通张量高一个数量级；换成 fp16 之后，最差的也回到了 1% 以内。
（旧文给出的 `time_emb_scale` 559% 不能与本表直接比：那是另一个 `model000005000.pt`——同名文件已被 2026-09-13 18:23 那次重训覆盖——而且是 `OutputProcess` 钉 fp32 之前的代码。）

### 4.2 GradScaler 不是可选项

`fp16_noscale` 一行是全文最重要的数字：**关掉 loss scale，fp16 的参数梯度中位相对误差是 58%，p90 是 98%，余弦掉到 0.94**。
原因见 5.1 的梯度量级表：主干层的激活梯度中位数在 1e-7–1e-8，fp16 的最小规格化数是 6.1e-5，不放大就整片落进非规格化区甚至清零。
bf16 指数位和 fp32 一样，完全不需要这套机制——这是 bf16 唯一真正的优势，而在本仓库里这个优势已经由 `MixedPrecisionTrainer` 的 `th.amp.GradScaler` 兑付掉了（`diffusion/fp16_util.py`，`amp_dtype == 'fp16'` 时才启用）。

## 5. fp16 专有的两个风险

### 5.1 loss scale 的窗口有多宽

以 200k checkpoint、典型 batch 计：

| 约束 | 值 | 允许的最大 scale |
|---|---|---|
| 激活梯度最大值 | 3.52e-3 | 2^24.2 |
| fp16 层里最大 \|dW\|（`species_film.2.weight` 4.9e-3） | — | 2^23.7 |
| 5000 步 checkpoint 同项 | 1.2e-2 | 2^22.4 |

实跑 run 的 scaler 状态（从 `opt*.pt` 读出）：

| step | scale | `_growth_tracker` |
|---|---|---|
| 5000 | 2^15 | 277 |
| 50000 | 2^17 | 945 |
| 100000 | 2^18 | 318 |
| 150000 | 2^17 | 578 |
| 200000 | 2^17 | 1720 |

典型 batch 的天花板是 2^23–2^24，而 scaler 稳在 2^17–2^18，即比典型上限低 64–128 倍。差距来自离群 batch：`save/merged_locomotion_v14_fp16/spikes/` 里 10 个 dump 的 `grad_norm_preclip` **全部是 `Infinity`**，即全部是 fp16 溢出导致的跳步（`_optimize_amp` 在非有限梯度时把 `last_grad_norm` 置 inf 并跳过该步），batch 内最大单样本 loss 达 1.06，是普通 batch 的 20–100 倍。scaler 每遇到一次就砍半，所以稳态就停在这些离群 batch 的上限附近。

两个副作用：

1. **离群 batch 被整步丢弃**。bf16 下这些 batch 会被 `clip_grad_norm_(max_norm=1.0)` 裁剪后照常更新，fp16 下直接跳过。已知事件 ≥10 次（step 4333–30381），相对 195k 步可以忽略，但"最难的样本被系统性丢掉"这一点在改数据配比时要记得。
2. **spike 探针被占满**。`_maybe_capture_spike` 用 `not np.isfinite(grad_norm)` 当触发条件，于是每次 scaler 溢出都写一份 dump，`--spike_max_dumps 10` 在 step 30381 就用光了，之后真正的梯度尖峰不再被记录（只打印一次提示）。**建议把 scaler 溢出与真实尖峰分开计数**，否则 fp16 训练里这个诊断设施在 3 万步后就等于关掉了。

### 5.2 关着的门后面会下溢

5000 步 checkpoint 上，`cross_limb_blocks.3` 的 cross-k 通路（`cross_k_scale` 还没打开）：

| 位置 | 激活梯度 absmax | 中位（非零） | ×2^17 后 |
|---|---|---|---|
| `cross_limb_blocks.3.cross_k_attn` | 5.06e-10 | 1.10e-11 | 6.6e-5 / **1.4e-6** |
| `cross_limb_blocks.3.cross_k_norm` | 1.79e-10 | 3.93e-12 | 2.3e-5 / **5.2e-7** |
| （对照）`layers.7.linear2` | 2.90e-5 | 1.24e-7 | 3.8e0 / 1.6e-2 |

fp16 最小规格化数 6.1e-5、最小非规格化数 6.0e-8。放大 2^17 之后这条通路**整片落在非规格化区**，只剩几个尾数位。结果：

| 参数（5000 步 ckpt） | fp32 参照 \|dW\| | fp16 相对误差 | bf16 相对误差 |
|---|---|---|---|
| `cross_limb_blocks.3.cross_k_attn.q_norm.weight` | 4.37e-9 | **26.0%** | 1.6% |
| `cross_limb_blocks.3.cross_k_attn.k_norm.weight` | 4.37e-9 | **26.6%** | 1.6% |
| 同两项在 200k ckpt | 4.12e-5 | 0.24% | 2.1% |

这是全部测量里唯一 fp16 输给 bf16 的地方，而且它会自愈：门一打开，梯度涨回 1e-5 量级，fp16 立刻恢复到 bf16 的 1/9。

**什么时候要小心**：新加一条零初始化门控通路（`time_emb_scale`、`temporal_phase_scale`、`cross_k_scale`、`reliability_bias` 都是这种）并从头训练时，最初几千步这条通路的梯度可能就在 1e-10 量级。此时 fp16 给它的梯度基本是噪声，门可能迟迟打不开。对策：这种改动的前几千步用 `--amp_dtype bf16` 跑，或者按本文方法量一次该通路的 `激活梯度中位数 × 当前 scale`，确认 ≳1e-3。

## 6. 速度与显存

eager 模式（非 `--compile` 路径）、bs=16、预热 10 步后取 40 步均值，交替顺序重复两轮：

| 配置 | 步时 | 相对 bf16 | 峰值显存 |
|---|---|---|---|
| fp16 | 240.1 / 241.9 ms | −2.8% | 21.4 GiB |
| bf16 | 247.6 / 247.8 ms | — | 21.4 GiB |
| fp32 | 363.5 ms | +45.8% | 32.8 GiB |

fp16 与 bf16 在 Blackwell 上 tensor core 吞吐相同，显存也相同，2.8% 的差是稳定但很小的量级（GradScaler 的 unscale + 检查抵不过别处的差异）。fp32 的 +45.8% / +53% 显存与旧文 5.1 的 +44% / +73% 一致。

> **不要被两次训练的日志步时误导**：bf16 run（`run_20260913_182323`）稳态 195.1 ms/step，fp16 run（`run_20260913_213151`）稳态 150.6 ms/step，这 23% 的差**不是 dtype 带来的**——中间隔着 commit c8cfb09 `perf(train): remove per-step host-device syncs`（19:58 落地），正是它把步时从 ~185 降到 ~151。

## 7. 训练质量的旁证

fp16 run 是从 bf16 run 的 5000 步 checkpoint 续训的，两边在 5000–5400 这段用同一份数据顺序：

| step | 5000 | 5100 | 5200 | 5300 | 5400 |
|---|---|---|---|---|---|
| `loss` bf16 | 0.1343 | 0.1327 | 0.1255 | 0.1177 | 0.1342 |
| `loss` fp16 | 0.1341 | 0.1325 | 0.1253 | 0.1170 | 0.1355 |

400 步内两条 loss 曲线的差属于逐步舍入分叉的正常范围，没有系统性偏向。fp16 run 随后跑满 200k 步，`l_simple` 从 0.1117 降到 0.0307，没有发散。

## 8. 建议

1. **训练保持 `--amp_dtype fp16`**。逐层误差是 bf16 的 1/8，步时和显存不变，旧文"问题 3"（门控标量梯度被舍入噪声主导）随之降一个数量级。
2. **绝不要在没有 GradScaler 的路径上用 fp16**。`MixedPrecisionTrainer` 只在 `amp_dtype == 'fp16'` 且 CUDA 时开 scaler；任何绕过 `mp_trainer.backward()` 的自定义训练/调试脚本都会拿到 0.3 量级的梯度误差。
3. **改一下 spike 探针**：把 scaler 溢出（`last_grad_norm == inf`）与真实梯度尖峰分开计数、分开限额，否则 fp16 训练下这个诊断在 3 万步后失效（5.1）。
4. **新增零初始化门控通路时，前几千步用 bf16 或先量一次梯度量级**（5.2）。
5. **推理 / 评估维持 fp32 + TF32 不变**（旧文 5.2-A）。`--amp_dtype` 在生成侧只接受 `fp32|bf16`，与本文无关。
6. 数值等价性验证仍然按 `utils/numerical_verification.py` 的 fp32 确定性协议做。fp16 的底噪（3e-3）虽然比 bf16（1.8e-2）小，但依然比 fp32 的 1e-7 大 4 个数量级，同样不能用来判断"改动前后是否一致"。

## 9. 实施记录：A + B + C + D（2026-09-14）

### 9.1 问题的形状：一个窗口的两端

把 loss scale 当自变量扫一遍（5000 步 checkpoint，2 batch，每个 scale 与 fp32 参照比）。扫 scale 等价于扫梯度量级，所以这张表同时回答了"梯度是平时 32 倍的那种 batch 会怎样"：

| loss scale | 全局参数梯度误差 | `cross_k_attn.q_norm.weight` | 溢出参数数 |
|---|---|---|---|
| 2^10 | 1.5e-3 | 100% | 0 |
| 2^14 | 1.13e-3 | 82–128% | 0 |
| **2^17（旧实跑值）** | 1.12e-3 | **6.8%** | 0 |
| 2^19 | 1.12e-3 | 0.78% | 0 |
| 2^21 | 1.12e-3 | 0.12% | 0 |
| **2^23** | 1.15e-3 | 0.09% | **9（溢出墙）** |
| 2^25 | — | — | 92 |

- 除那条被门关住的支路以外，误差在 **2^14–2^22 完全平坦**：窗口上沿一分钱都不值。
- 第 5.2 节的 QKNorm 要 scale ≥2^19 才干净，而离群 batch 把有效溢出墙压到 2^17–2^18 ——**两个目标在原结构下直接冲突**，`GradScaler` 选了 2^17，代价就是 6.8%。
- 最先溢出的是 `canonical_frame_projection.2` / `loop_condition_projection.2` / `resample_speed_projection.2` / `species_film.2` / `cross_in_attn.in_proj_bias`：它们的输出要广播到 ~1e6 个 token，反向对这些 token 求和，所以权重梯度是全网最大的。

另外，第 5.1 节说的"周期性溢出"是 `GradScaler` 的设计而不是意外：它每 `growth_interval`（2000 步）翻一倍直到溢出，稳态下按节律制造跳步。要"不产生 spike"就必须停掉这个探顶动作。

### 9.2 改动

| | 改动 | 落点 |
|---|---|---|
| A | cross-K 子路做成 fp32 孤岛 | [motion_transformer.py](../model/motion_transformer.py) `CrossLimbTemporalBlock.forward` |
| B | GradScaler 的 loss scale 封顶并从 2^15 开始，并记录 `amp_overflow` | [fp16_util.py](../diffusion/fp16_util.py) `GRAD_SCALER_MAX_SCALE`、`_cap_loss_scale`、`_optimize_amp` |
| C | scaler 溢出与真实梯度尖峰分开分类、分开限额、分开文件名 | [training_loop.py](../train/training_loop.py) `classify_grad_event`、`AMP_OVERFLOW_MAX_DUMPS`、`_maybe_capture_spike` |
| D | 广播型条件头固定 fp32 | [anytop.py](../model/anytop.py) `run_in_fp32` + 8 处调用 |

- **A**：`cross_k_scale` 是模型里唯一一个背后挂着整张子网络的零初始化门（其余 `time_emb_scale` / `temporal_phase_scale` / `*_reliability_bias` 只门住一个加法项，门自身的梯度并不小）。子路只有 K 个 token、瓶颈宽度，fp32 实测 0 成本。
- **B**：当前 `GradScaler` 显式以 2^15 初始化并封顶，避免新 run 在第一步探测 2^16；真正溢出时仍会下降。**恢复旧 run 时**保存的更高 scale 会在第一次 optimizer update 后被拉回 2^15（并重置 growth tracker）。
- **C**：旧逻辑用 `not np.isfinite(grad_norm)` 当尖峰触发条件，而 `_optimize_amp` 恰好用 `inf` 标记"因非有限梯度跳过"。没有 scaler 的 bf16 / fp32 下 `inf` 仍然算真实故障，照样 dump。
- **D**：8 处 = `species_film`、`resample_speed_projection`、`loop_condition_projection`、`action_label_projection`、`canonical_frame_projection`、`input_process.{species_film_j, text_embedding, struct_embedding}`。`InputProcess` 的返回本来就在末尾被 fp32 的 pos_emb 提升，所以下游看到的 dtype 没有变化。

### 9.3 效果（实测）

QKNorm（5000 步 checkpoint，扫 2^12–2^24）：

| | 2^12 | 2^16 | 2^17 | 2^21 | 2^23 |
|---|---|---|---|---|---|
| 改前 | ~100% | 35–104% | 6.8% | 0.12% | 0.09% |
| **改后** | **8.9e-4** | **8.9e-4** | **8.9e-4** | **8.9e-4** | **8.9e-4** |

即与 loss scale 彻底解耦，并且比同一 batch 的 bf16（7.0e-3）好 8 倍。

全局（200k checkpoint，4 batch，fp16 在各自的实跑 scale 上）：

| 指标 | 改前（scale 2^17） | 改后（scale 2^16） |
|---|---|---|
| 参数梯度全局相对误差 | 3.07e-3 | **2.76e-3** |
| 逐张量中位 / p90 | 2.47e-3 / 3.88e-3 | **1.91e-3 / 3.13e-3** |
| 余弦 | 0.9999953 | **0.9999969** |
| 模型输出累积误差 | 2.32e-4 | **1.97e-4** |
| 最差参数张量 | 26%（5k ckpt 的 QKNorm） | **5.8e-3**（`action_label_null_emb`） |
| bf16 同项（也吃到 D 的好处） | 1.77e-2 | 1.55e-2 |

5000 步 checkpoint 上全局误差 1.12e-3 → 7.0e-4（−37%），来自 D。

**逐张量对拼**（两个 checkpoint 各 4 batch，取中位）：改后 **429/429 个参数梯度张量、140/140 个前向层，fp16 都优于 bf16**，没有例外。

| | bf16/fp16 误差比 min | p10 | 中位 | p90 | max |
|---|---|---|---|---|---|
| 参数梯度（429 个张量，200k） | 1.04 | 6.45 | 7.91 | 10.17 | 29.85 |
| 参数梯度（429 个张量，5k） | 1.10 | 7.14 | 8.87 | 11.67 | 27.31 |
| 前向激活（140 层，200k） | 6.97 | — | 7.87 | — | 8.50 |

前向优势严格是 7–8.9 倍（尾数位数之差）；梯度侧最小只有 1.04 倍，落在 `reliability_bias` / `temporal_phase_scale` 这类正负相消主导的标量门上——那里的误差不由尾数决定，所以 fp16 只是打平。
注意标量 loss 本身两边都已经在 1e-4 相对量级、彼此互有胜负，那是 fp32 归约的噪声底，不构成反例。

成本：

| | 改前 | 改后 |
|---|---|---|
| fp16 步时（eager，bs=16，交替两轮） | 240.1 / 241.9 ms | 240.2 / 242.5 ms |
| 峰值显存 | 21.4 GiB | 21.4 GiB |
| `torch._dynamo.explain` 整个 AnyTop 前向 | — | **1 graph / 0 graph break** |

步时差在噪声内（≤1%）。作为对比，把整个 `CrossLimbTemporalBlock` 钉成 fp32 是 +3.8%（251.2 ms），收益与只钉 cross-K 相同——所以孤岛要小。

### 9.4 验证

- 新增 [tests/test_fp16_precision_policy.py](../tests/test_fp16_precision_policy.py)（16 项）：孤岛在 fp16/bf16 autocast 下都是 fp32 且只覆盖 cross-K、封顶逻辑（超上限拉回 / 未超不动 / scaler 关闭时不报错 / 上限落在实测平坦窗口内）、溢出与尖峰的分类、`run_in_fp32` 与纯 fp32 调用逐位相等。全量 **849 项通过**。
- 40 步真实 fp16 训练（eager，`--ml_platform_type NoPlatform`）：历史 2^16 配置下 `loss_scale_log2` 稳定 16、`amp_overflow` 恒 0；当前配置预期稳定在 15。
- 改动只动数值，不动 checkpoint 格式：现有权重可直接 resume，不需要重新生成数据，也不强制重训。但它确实改变了训练数值，所以跨此改动的 loss 曲线逐步对比没有意义（整体水平可比）。

### 9.5 上线后看什么

1. `amp_overflow`（tensorboard，每步记录的 0/1 均值）应长期为 0。若出现非零，说明离群 batch 的梯度超过了 2^15 的余量，应优先检查数据/loop seam；必要时可继续降到 2^14。
2. `save/<run>/spikes/` 下现在区分 `spike_step*.json`（真实梯度尖峰，限额 `--spike_max_dumps`，默认 10）和 `overflow_step*.json`（scaler 溢出，限额 2）。只有前者才需要按尖峰的老路子查 `top_param_grad_norms`。
