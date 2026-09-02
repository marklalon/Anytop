# temporal_window → 全注意力

> 状态：**完成（2026-09-02）** —— 代码已实施（2026-09-01）、
> 全量重训 200k 步已跑完（`merged_locomotion_v4_fullattn`）、
> §6 的四条判读全部通过，**新 ckpt 转正**（判读数据见 **§6.1**，实施记录见 §10）
> 改动范围：**删除 `--temporal_window` 及整条 temporal mask 通路**（36 处引用 / 11 个文件），另加 **checkpoint 版本守卫**（替代并删除两个既有 deprecation 守卫）。
> **无 cond.npy 变化、无数据重生成、`state_dict` 结构不变。**
> 目标：去掉 temporal 自注意力的滑动窗口，让每个 token 看到整段。
> **旧 checkpoint 不再兼容**，必须加 checkpoint 版本守卫（§3.2），否则会静默跑错。
> 验证方式：**全量重训 200k 步，重训完成后人工跑几条常规用例验证**（口径见 §6，
> 实际判读见 §6.1 —— 四条里三条改用了可复现的自动量测，人工判读作抽查）。

---

## 0. 现状

[`create_temporal_mask_for_window`](../data_loaders/truebones/data/dataset.py#L125-L136) 生成一个
`(T+1, T+1)` 的 0/1 模板，经 [collate](../data_loaders/tensors.py#L95-L99) 进 `y['mask']`，在
[`anytop.py:658-660`](../model/anytop.py#L658-L660) 变成加性 mask 喂给 temporal attention：

```python
temporal_template = (1.0 - temp_mask.reshape(bs, -1, nframes+1, nframes+1)[:, :1].float()) * -1e4
temporal_template = temporal_template.expand(-1, self.num_heads, -1, -1)
temporal_mask = temporal_template.unsqueeze(1).expand(-1, njoints, -1, -1, -1).reshape(-1, nframes+1, nframes+1)
```

三个必须先讲清楚的事实：

1. **它是 dense additive mask，不是稀疏 kernel。** 全尺寸注意力矩阵照算，窗外位置加 `-1e4`。
   窗口**不省任何 FLOPs**。
2. **token 0 是 T-pose token，且被全体可见**（`mask[:, 0] = 1`，
   [dataset.py:128](../data_loaders/truebones/data/dataset.py#L128)）。
3. **序列长度恒为 `num_frames+1 = 61`，训练推理完全一致。** 推理端 `--num_frames M` 只经
   [`_finalize_output_lengths`](../sample/generate.py#L244-L255) 变成 `playspeed = M/60`，
   采样仍在 61 个 token 上进行（[generate.py:595](../sample/generate.py#L595) 的注释：
   "Fixed-window model: always run at native window length"），采完再 resample。
   **不存在长度外推问题**，窗口注意力的经典理由之一在本仓库不成立。

生产配置是 `--temporal_window 41`（见 [species_joint_film_upgrade.md](species_joint_film_upgrade.md) §6），
margin=20，每帧看 42/61 个 token。

---

## 1. 先排除三条不是理由的理由（全部实测）

这一节的结论是**否定性**的：性能、显存、kernel 后端都与本改动无关，既不构成收益也不构成代价。
写下来是为了避免以后再拿它们当论据。

### 1.1 开窗口不花时间

真实 AnyTop 训练步（latent 256 / ff 2048 / 8 层 / 4 头，bs=16、60 关节、60 帧、
bf16 autocast、fwd+bwd+AdamW、eager、RTX PRO 6000 Blackwell、torch 2.11.0+cu128）：

| 配置 | ms/step |
|---|---|
| `window=41`（现状） | 132.88 |
| `window=121`（等价全注意力） | 133.26 |

噪声内相等 —— 如 §0 事实 1 所述，两者 FLOPs 相同。

### 1.2 模型是 launch-bound，temporal attention 不在关键路径上

同一配置的 profile：

```
8639 次 kernel launch / step
最大单项 aten::copy_ 占 9.9%（1821 次调用/step），没有任何 op 超过 10%
```

把 mask 从 per-sample 物化改成 `(1, H, T+1, T+1)` 广播 buffer，在隔离测试里省 0.5 ms
（8 层 fwd+bwd：4.16 → 3.65 ms），放进 133 ms 的真实步里是 **0.4%**。
**想提速应该去合并那 1821 次 copy，不是动 mask。**

### 1.3 flash / 对齐 / 显存都不成立

- 本机 wheel **没有编译 flash attention**（`can_use_flash_attention` 报
  "Torch was not compiled with flash attention"），连 `attn_mask=None` 都走不了 flash。
  可用后端只有 `EFFICIENT`（全 dtype）和 `CUDNN`（仅 bf16）。
- `T+1 = 61` **不会被 `EFFICIENT_ATTENTION` 拒绝**，不存在退回 `MATH` 的问题。
  61 未对齐只导致 SDPA 每次调用做一份 padded bias 拷贝（每层多约 23 MiB，8 层约 180 MiB）；
  换广播 bias 后拷贝降到约 30 KiB/层，实测峰值 612.3 MiB vs 无 bias 的 612.1 MiB，测不出差别。
- 因此 **`num_frames` 不需要改成 63/64**（详见 §5）。

---

## 2. 真正的理由

### 2.1 T-pose token 是一个"只看开头"的全局广播寄存器 —— 最硬的一条

`mask[:, 0] = 1` 让**每一帧都读 token 0**。但 row 0 自己也被同样的窗口限制：

| window | 全部可见 pair | row 0（T-pose token）能看到 |
|---|---|---|
| 31 | 46.8% | tokens 0..16（17 / 61） |
| 41 | 58.1% | tokens 0..21（22 / 61） |

**整个序列里唯一一个广播给所有帧的 token，自身只聚合了 clip 的前三分之一。**

这不是"可能有收益"，是一个结构性不对称：token 0 的内容被所有帧读取，却是一个偏向开头的
局部摘要。全注意力下它立刻变成真正的整段摘要，代价为零。

### 2.2 窗口正好切在反相点上，再往外仍有信号

在 400 条 locomotion clip 上量的自相关（先按训练管线 resample 到 60 帧，
原始 BVH 通道，去均值后按通道方差加权平均）：

```
lag  1: +0.952     lag 20: -0.285     lag 35: -0.198
lag  5: +0.599     lag 25: -0.287     lag 40: -0.128
lag 10: +0.140     lag 30: -0.244     lag 45: -0.036
```

lag 20~25 的强负相关是左右肢反相点。`window=41` 的 row `i` 覆盖 `i-20 .. i+21`，
**刚好压线把反相点包进来** —— 这大概能解释 41 为什么比默认 31 好用。
但 lag 30~40 的相关绝对值仍有 0.13~0.27，全在窗外。

连通性本身不是瓶颈（忽略 token-0 hub，frame1→frame60 在 window=41 下 3 跳可达，模型有 8 层），
问题是这些关系目前只能靠多层复合，而经过 value 混合的复合是有损的。

### 2.3 但收益上限有限：clip 太短

```
locomotion clip 原始帧数：median 31（p10 20, p90 61）→ 多数 clip 是被 UPSAMPLE 到 60 的
400 条里只有 36 条能检出重复周期
lag 1 自相关 = 0.95
```

大多数 clip 只装得下**一个步态周期甚至更少**，61 个 token 实际只承载约 30 帧的独立信息。
窗口并没有切掉"很多个周期"，能捞回来的是一个周期内的远端相位关系，不是跨周期的长程结构。

**期望要放对：这是修一个结构缺陷（§2.1）+ 补一段中程相位关系（§2.2），不是能力解锁。**

### 2.4 §2.3 同时削弱了主要风险

丢掉窗口 = 丢掉"时间局部性"这个归纳偏置。但相邻帧自相关 0.95，局部性在数据里满得溢出来，
模型不需要 mask 去发现它；何况 base 配置已有 `--lambda_vel 0.2`
（速度-位置一致性）直接惩罚时间不连贯。风险等级：中 → 低。

---

## 3. 方案

### 3.1 直接删除 `--temporal_window`，不保留窗口路径

不做开关、不留 `window > 0` 的对照路径。§1 已证明保留窗口路径零性能收益，
它的唯一价值是可回退性，§4 的取舍已接受放弃这一点。

删除面：`temporal_window` 36 处引用 / 11 个文件（含 `train.bat`），
`create_temporal_mask_for_window` 10 处。

删除后 temporal attention 恒为 `attn_mask=None`，整条
`dataset → collate → y['mask'] → anytop` 的 mask 通路一起消失。
`y['mask']` 的真实消费者只有 [`anytop.py:588`](../model/anytop.py#L588) 一处
（`nframes` 来自 `x.shape`，不依赖 mask），所以这条链可以整体拆掉。

### 3.2 必须加的守卫 —— 删除后唯一真正的风险：checkpoint 版本检查

`parse_and_load_from_model` 只回填**当前 parser 里存在**的键
（[parser_util.py:19-25](../utils/parser_util.py#L19-L25)）。所以 flag 删掉之后，
旧 `args.json` 里的 `temporal_window: 41` 会被**静默忽略**：checkpoint 照常加载，
然后以全注意力运行一个按 `window=41` 训练的模型。

这不是理论风险，**这个失配场景已经实测过**（§5 末行）：窗外 logit 从未被任何梯度约束，
放进 softmax 分母后逐层放大，表现为异常姿态 —— 看起来像"质量退化"而不是"不兼容"。

**用 checkpoint `version` 号取代所有按 key 的守卫**（两个既有的
`assert_action_conditioning_not_deprecated` / `assert_global_energy_not_deprecated` 一并删除，
其由来见 [global_energy_removal.md](global_energy_removal.md) §2）：

- 新增常量 `CKPT_VERSION = 1`（`parser_util.py`）；
- 训练存档时把版本号写进 `args.json`（[train_anytop.py:206](../train/train_anytop.py#L206) 的 `json.dump(vars(args), ...)` 旁）；
- `extract_args`（[parser_util.py:111](../utils/parser_util.py#L111) 旁）加载时检查：
  **缺失** `version` → 版本化之前的旧 checkpoint，拒绝；`version != CKPT_VERSION` → 拒绝。
  两者都 `SystemExit`，消息指向对应文档；
- resume 路径复用同一检查（`train_anytop.py` 的 `assert_resume_keeps_action_group` 旁），
  否则用新代码 resume 旧 `save_dir` 仍会"加载成功但跑错"。

版本检查**严格强于**旧守卫：旧守卫只在特定 key 出现时触发，版本检查拒绝**所有**
升级前 checkpoint（已核实：现有 checkpoint 的 `args.json` 均无 `version` 字段）。
今后任何改变训练/推理语义的改动（哪怕 `state_dict` 结构不变），
**只需 bump `CKPT_VERSION`**，不再每次新增一个 assert。

### 3.3 删除清单

| # | 位置 | 改动 |
|---|---|---|
| 1 | [`parser_util.py:188`](../utils/parser_util.py#L188) | 删 `--temporal_window` |
| 2 | [`parser_util.py:488`](../utils/parser_util.py#L488)、[`514-518`](../utils/parser_util.py#L514-L518) | 删 `generate_args` 里的 CLI-override 保留逻辑；`_cli_flag_explicit` 随之成为死代码，一并删 |
| 3 | [`parser_util.py:35`](../utils/parser_util.py#L35)、[`56`](../utils/parser_util.py#L56)、[`111-112`](../utils/parser_util.py#L111-L112) | **删除**两个既有 `assert_*_not_deprecated`；**新增** checkpoint 版本检查（`CKPT_VERSION` + 存档写 `version` + `extract_args` 检查 + resume 复用，§3.2） |
| 4 | [`anytop.py:588`](../model/anytop.py#L588)、[`658-660`](../model/anytop.py#L658-L660) | 删 `temp_mask = y['mask']` 与整段 mask 构造；`temporal_mask = temporal_template = None` |
| 5 | [`motion_transformer.py:533-534`](../model/motion_transformer.py#L533-L534) | cross-limb 的 `tt` 直接删，`temporal_attn(..., attn_mask=None)`（该模块已支持 `None`，见 [`motion_transformer.py:143`](../model/motion_transformer.py#L143)） |
| 6 | [`dataset.py:125-136`](../data_loaders/truebones/data/dataset.py#L125-L136)、[`799-820`](../data_loaders/truebones/data/dataset.py#L799-L820)、[`993-994`](../data_loaders/truebones/data/dataset.py#L993-L994) | 删 `create_temporal_mask_for_window`、两个模板、`_get_temporal_mask`、`circular_mask`（§3.4） |
| 7 | [`tensors.py:85-99`](../data_loaders/tensors.py#L85-L99)、`264`、`286`、`304-307` | 从 batch 元组和 `cond['y']` 里删 `temporal_mask` / `mask`；**注意 batch 元组是位置索引，删一项要同步改下标注释** |
| 8 | `get_data.py`、`dataset.py`、`train_anytop.py`、`training_loop.py` | 删 `temporal_window` 形参透传（8 处） |
| 9 | [`generate.py:26`](../sample/generate.py#L26)、`875`、`1454`、`2233`、`2265` | 删 import、`create_condition` 的形参与两处调用 |
| 10 | `tools/sample_augmented_bvh.py:245`、`tools/visualize_action_separability.py:311` | 删传参 |
| 11 | `train.bat` | 删 `--temporal_window 41` |
| 12 | tests | `test_dataset_loop.py`（9 处）、`test_generate_inpainting.py`（2 处）、`test_native_loop.py:46,91-97`；5 处 `'mask'` fixture（`test_action_label_cfg_sampler.py:72` 等）随 #7 调整 |

### 3.4 circular mask 分支：随之消失

[`dataset.py:130-133`](../data_loaders/truebones/data/dataset.py#L130-L133) 的环形窗口在
`loop_full_cycle` 时让首尾互见。**全注意力下 circular 与 linear 完全等价。**

准确的说法：**不是丢连接**（全注意力下首尾当然互见），而是**这条通路不再携带"本段是循环"这个 bit**。
loop 还有另外两路信号：
[`loop_condition_projection`](../model/anytop.py#L620-L623) 和
[`circular_phase_embedding`](../model/motion_transformer.py#L1097)，后者是更强的信号。

因为窗口路径整体删除，circular 分支不会留下死代码 —— 但
[`test_native_loop.py:91-97`](../tests/test_native_loop.py#L91-L97) 的 linear≠circular 断言
必须一起删（列在 §3.3 #12），它是这条通路唯一的显式契约。

---

## 4. 预期收益与代价

### 收益

| 项 | 依据 | 把握 |
|---|---|---|
| T-pose token 从"前 1/3 摘要"变成整段摘要，且它被所有帧读取 | §2.1，结构性，非经验 | **高** |
| 一个周期内 lag 30~40 的相位关系从"3 层复合"变成直连 | §2.2，相关绝对值 0.13~0.27 | 中 |
| 删掉 circular 分支 + 窗口模板，dataset/collate 少一条状态 | §3.3 | 高（但价值小） |
| 训练吞吐 | —— | **零。§1.1 实测 132.88 vs 133.26 ms** |
| 显存 | —— | **零。§1.3** |

### 代价

| 项 | 量级 | 缓解 |
|---|---|---|
| 一次全量重训 200k 步 | **主要成本** | 无 |
| **所有旧 checkpoint 报废** | **本方案最大的不可逆代价**：`window>0` 的推理路径不复存在，既有 ckpt 只能当历史结果读，不能再跑 | 无。版本守卫（§3.2）只保证它**报错而不是跑错** |
| 丢失时间局部性归纳偏置，可能出现高频抖动 | 低（§2.4：相邻帧 ac=0.95，且有 `--lambda_vel 0.2`） | 出现则上 §5 的 ALiBi 备选（**需再重训一次**，因为窗口路径已删，无法回退对照） |
| loop 的 mask 拓扑信号消失 | 低（另有两路信号，且 phase embedding 更强） | §6 单列 loop 缝合判读 |
| 删除面 36 处 / 11 文件，batch 元组是位置索引 | 中，机械但易错 | §3.3 #7 的下标注意事项 + §8 步骤 3 的回归 |

**一句话**：主要成本是一次重训加旧 ckpt 报废；收益的下限是 §2.1 那个确定的结构修复，
上限被 §2.3 的 clip 长度压住。

**注意这个取舍随"不保留窗口路径"变了性质**：保留开关时这是个"低风险可回退"的改动，
删干净之后**没有回退路径** —— 若 §6 判读为负，只能再重训一次（上 ALiBi 或恢复窗口）。
接受这一点的前提是 §1 已经证明保留窗口路径**不带来任何性能收益**，它的唯一价值就是可回退性。

---

## 5. 本轮明确不做

| 项 | 内容 | 不做的理由 |
|---|---|---|
| ALiBi 软衰减 | 把硬窗口换成按时间距离线性递减的加性偏置，每 head 一个斜率 | **列为 §4 抖动风险的备选**。它引入一组需要调的斜率超参，买到的"局部先验"正是数据以 0.95 自相关免费提供的东西。先上朴素全注意力，出问题再加 —— 但注意窗口路径已删，届时是**再重训一次**，不是回退 |
| `num_frames` 60 → 63/64 | 让 `T+1` 对齐到 64，消除 SDPA 的 padded bias 拷贝 | §1.3 实测：广播 bias 下拷贝只有约 30 KiB/层，测不出差别。而改它要重训，还会改变 `playspeed = M/num_frames`（[generate.py:254](../sample/generate.py#L254)）的语义，已标定的 playspeed 全要重对 |
| 广播 `(1,H,T+1,T+1)` bias 重构 | 把 per-sample 物化改成 buffer | 只在保留 mask 时才有意义；本方案是**没有 mask**，自然消失。作为独立优化收益 0.4%（§1.2），不值得单做 |
| 时间位置编码升级（相对位置 / RoPE / 逐层重注入） | 现在只在 [`InputProcess`](../model/anytop.py#L786-L788) 加一次绝对正弦 PE | 序列恒长 61 且不外推（§0 事实 3），绝对 PE 够用。**若 §6 判读显示相位仍混模态，这是下一个该动的地方，不是 mask** |
| 推理端单独开窗口 | 用旧 ckpt 跑 `--temporal_window 121` 看效果 | **已试过，出现异常姿态，且该实验对本问题零信息量**：窗外 logit 从未被任何梯度约束过，推理时打开 mask 等于把未训练的任意 logit 放进 softmax 分母，并逐层放大。这测的是 mask 失配，不是注意力范围的效果 |

---

## 6. 验证：全量重训 + 人工常规用例验证

### 配置

本 arm 沿用 [species_joint_film_upgrade.md](species_joint_film_upgrade.md) §6 的 base 配置，
去掉 `--temporal_window 41` 一项，其余超参与 seed 逐项对齐。
跑完用 `args.json` 逐键 diff 复核，只允许 `save_dir` 不同、`temporal_window` 缺失、
新增 `version` 字段。

### 验证方式

**不做 base arm 对照** —— 重训完成后，人工跑几条常规用例，看结果：

> **实际执行时四条都做成了可复现的量测**（§6.1），人工判读降级为抽查。
> 另外"不做 base arm 对照"这个决定的代价在 §6.1 开头具体化了：
> 本轮重训同时含 [action_label_keyword_refactor.md](action_label_keyword_refactor.md)
> 的关键词化改动，四条判读都无法单独归因给全注意力。

1. **步态相位一致性** —— 采几条 locomotion 动作（walk / gallop 等），看左右肢反相关系
   与相位是否随时间漂移。这是 §2.2 的直接靶子，也是当初 S11 提出这条的动机。
2. **loop 首尾缝合** —— 采 loop 用例，看首尾衔接。这是本改动**唯一可能变差**的地方
   （§3.3 丢掉的 mask 拓扑信号），必须单独看，不能混在总体质量里。
3. **高频抖动** —— §4 的主要风险。看 `--lambda_vel` 项的收敛值和逐帧位置二阶差分（或肉眼判读）。
4. **整体质量** —— 常规用例自然，无异常姿态。

### 结论口径

代码已经删干净，所以每一档的"下一步"都是**往前**，没有回退档：

- 1/2/3/4 均通过 → 完成，新 ckpt 转正。
- 1 通过 + 2 退化 → 保留全注意力，把 loop 信号补强（优先调 phase embedding 的 scale，
  **不是**恢复 mask —— 恢复 mask 等于把 §2.1 的结构修复一起退掉）。
- 1 不变 + 3 退化 → 上 §5 的 ALiBi 备选，再重训一次，重跑本节。
- 1 不变 + 3 不变 → 保留全注意力（§2.1 的结构修复仍成立，且零成本），
  把相位问题转交给"时间位置编码升级"（§5 末行），不要再在 mask 上找答案。

**不能用 l_simple 作判据** —— 同 [global_energy_removal.md](global_energy_removal.md) §4：
这类通路改动对训练 loss 的影响远小于其对可控性的影响。

### 6.1 判读结果（2026-09-02，`merged_locomotion_v4_fullattn` / `model000200000.pt`）

#### 先做 §6 要求的配置核对

`args.json` 逐键 diff，**四处不同**：

| 键 | v3 | v4 | §6 是否允许 |
|---|---|---|---|
| `save_dir` | `save/merged_locomotion_v3` | `save/merged_locomotion_v4_fullattn` | ✅ |
| `temporal_window` | `41` | 缺失 | ✅ |
| `version` | 缺失 | `1` | ✅ |
| `action_label_coarse_prob` | `0.3` | 缺失 | ❌ **计划外** |

第四条来自
[action_label_keyword_refactor.md](action_label_keyword_refactor.md) §2.5 ——
那一轮的删除和本轮的重训**合并成了同一次训练**。
所以 §6"不做 base arm 对照"的代价在这里具体化了：
**本 ckpt 同时含两项改动，下面四条判读都不能单独归因给全注意力。**
能做的归因只是"哪一条改动瞄的是哪个靶子"，写在各条里。
其余超参与 seed 逐项一致。

#### 1. 步态相位一致性 —— **通过**，且是四条里改善最明显的

用 `tools/direction_following.py phase` 量左右脚接触相位的组内 circvar
（同一提示 16 条样本，0 = 完全一致），对着 v3 的同一张表逐格比
（原始表在 `outputs/direction_following/R0_phase.txt` / `R1_phase.txt`）。
cfg∈{2,3} 均值，v3 → v4（括号是该格的语料 circvar）：

| | forward | backward | left | right |
|---|---|---|---|---|
| run | 0.467 → **0.110** (0.002) | 0.611 → **0.205** (0.000) | 0.506 → 0.876 (0.591) | 0.283 → **0.138** (0.448) |
| walk | 0.264 → **0.142** (0.117) | 0.715 → **0.441** (0.730) | 0.173 → **0.044** (0.552) | 0.191 → **0.113** (0.428) |

8 格里 7 格改善。`run, forward` 从 0.467 掉到 **0.110**（语料 0.002）——
这正是 §2.2 的靶子：一个周期内 lag 30~40 的相位关系不再跨三层复合。
唯一变差的 `run, left` 该格语料本身就散（0.591，n=3），误差棒宽，不单独下结论。

**归因说明**：相位一致性是本改动瞄的靶，关键词化瞄的是朝向；
朝向那一栏同轮也大幅改善（见
[action_label_keyword_refactor.md §4.3](action_label_keyword_refactor.md)），
两者各自命中自己的靶，没有互相矛盾的迹象 —— 但严格说仍分不开。

#### 2. loop 首尾缝合 —— **通过**（本改动唯一可能变差的地方）

对 `eval_checkpoint` 电池里全部 **9 个带 `--loop`** 的任务，
按 `tools/compute_loop_unclosure_error.py` 的 `wrap_gap` 定义
（逐关节 ‖pos_last − pos_first‖ 的 p75，特征通道 0-2）逐任务量，v3 → v4：

| 任务 | v3 | v4 |
|---|---:|---:|
| Basic/task2 (run, 30f) | 0.0175 | **0.0139** |
| Basic/task3 (walk, 30f) | 0.0098 | **0.0040** |
| Basic/task4 (run, 60f) | 0.0086 | **0.0051** |
| Basic/task5 (walk, 60f) | 0.0041 | **0.0025** |
| Basic/task6 (run, 120f) | 0.0092 | **0.0056** |
| Basic/task7 (walk, 120f) | 0.0046 | **0.0025** |
| ConvertLoop/task1 | 0.0272 | 0.0277 |
| InpaintJoints/task1 | 0.0147 | **0.0117** |
| NewSkeleton/task1 | 0.0066 | **0.0055** |

**全部持平或收紧**，唯一的 + 是 ConvertLoop 的 +0.0005（量级上是噪声）。
§3.3 丢掉的 mask 拓扑信号没有造成缝合退化 ——
§4 代价表里"另有两路信号，且 phase embedding 更强"这条判断成立。

#### 3. 高频抖动 —— **语料内通过，域外有一个真实回归**

指标用逐帧位置二阶差分的 RMS 除以每帧步长 RMS（无量纲，rig 尺度被约掉）。
17 项电池里 **16 项在 ±0.03 内浮动**（基线 0.17~0.39），无系统性上升；
`jerk_norm` / `snap_norm` 分量在这 16 项上分别 +0.004 / +0.012（见第 4 条）。

**例外是 `NewSkeleton/task1`**（dragon + `--action_label fly --loop`，域外骨架）：
抖动比 0.42 → **1.07**，`jerk_norm` 0.725 → **0.166**，总分 0.788 → 0.533，
8 条里 6 条中招（per-file 0.46 / 0.46 / 0.50 / 0.47 / 0.49 / 0.72 / 0.45 / 0.71）。

拆分探针（v4，同一 cond、同一 seed，`outputs/direction_following/dragon_probe/`）
定位到是**三者叠加**才炸：

| 提示 | `jerk_norm` |
|---|---:|
| `fly`，不带 `--loop` | **0.995** |
| `walk --loop` | 0.720 |
| `fly --loop` | **0.166** |

即 **loop × 域外骨架 × 域外动作**（`fly` 在 locomotion group 的边缘）三者同时成立才复现；
语料内的 6 个 loop 任务（上表 Basic/task2~7）全部正常。
v3 侧无法回跑同样的探针 —— §3.2 的版本守卫按设计直接拒绝了 v3 的 `args.json`
（**顺带是这条守卫的第一次实战验证，行为符合设计：报错，不是跑错**）。

按 §6 的口径这一条判为**通过但挂一个待办**：它不是"整体高频抖动"，
不触发 §5 的 ALiBi 备选（那要求语料内也抖）。

#### 4. 整体质量 —— **通过**

`eval_checkpoint.py` 的 17 项电池，v3 / v4 同任务同种子逐项对比
（报告：`outputs/eval_checkpoint/merged_locomotion_v4_fullattn/model000200000/eval_report.html`）：

- **16/17 项持平或改善**，均值 0.8868 → **0.8971**
- 四个分量（去掉第 3 条那个域外任务后）全部上升：
  `jerk_norm` +0.004、`snap_norm` +0.012、`spectral_flatness` +0.014、`bone_length` +0.014
- 唯一下降的就是第 3 条的 `NewSkeleton/task1`
- 两份 `cond.npy`（v3 save / v4 save / `dataset/merged`）MD5 逐字节相同，排除 cond 的账

#### 结论

按 §6 的结论口径：**1 / 2 / 3 / 4 均通过 → 完成，新 ckpt 转正**。
`temporal_window` 的删除保留，不上 §5 的 ALiBi。

一个待办（不阻塞转正）：**域外骨架 + `--loop` + 域外动作** 三者叠加时的高频抖动，
见第 3 条。它落在"新骨架 loop 生成"这条独立线上，
不是本改动的判据所覆盖的范围，需要单独立项时再查。

---

## 7. 影响面清单

| 改动 | regen `cond.npy` | regen 数据集 | 旧 ckpt 兼容 | retrain |
|---|:---:|:---:|:---:|:---:|
| §3.1 删除 `--temporal_window` 及 mask 通路 | 否 | 否 | **否**（版本检查直接 `SystemExit`） | **是** |
| §3.2 新增 checkpoint 版本检查（并删除两个既有 deprecation 守卫） | 否 | 否 | —— | 否 |
| §3.4 circular 分支随删 | 否 | 否 | 否 | 随上 |

**参数量、特征维度、条件维度全部不变**，`state_dict` 结构不变 —— 旧 ckpt 的**权重**
其实仍可加载，不兼容的是它训练时依赖的 mask 语义。守卫存在的意义正在于此：
不加守卫就会"加载成功但跑错"（§3.2）。

---

## 8. 执行顺序

1. §3.2 的版本守卫（先加版本检查并删掉两个旧守卫，再删代码，
   避免中间态出现"能加载但跑错"的窗口期）
2. §3.3 的 12 项删除 + §3.4 清理
3. 补一个测试：`seqTransDecoder` 收到的 `temporal_mask is None`，且 cross-limb 块不炸；
   再补一个：`version` 缺失或不匹配的 `args.json` 触发 `SystemExit`
4. 跑一遍现有测试套件 —— 重点是 §3.3 #7 的 batch 元组下标（位置索引，删一项会静默错位）
5. 启动重训（`RUN_NAME=merged_locomotion_v4_fullattn`）—— ✅ 200k 步已跑完
6. 按 §6 判读（人工常规用例），回填"实施记录"一节 —— ✅ 判读见 §6.1，记录见 §10 末

---

## 9. 附：测量方法

本文所有数字来自以下测量，脚本未入库（一次性）：

| 测量 | 方法 |
|---|---|
| §1.1 步时 | 构造生产超参的 `AnyTop`，随机 `y`，fwd+bwd+AdamW，8 步 warmup + 30 步计时。注意 `AnyTop._apply` 不返回 `self`，`.to(dev)` **不能链式调用** |
| §1.2 profile | `torch.profiler`，按 `self_device_time_total` 排序 |
| §1.3 后端 | `torch.nn.attention.sdpa_kernel([backend])` 排他执行 + `torch.backends.cuda.can_use_*_attention(params, True)` 取拒绝原因 |
| §2.1 mask 统计 | 直接对 `create_temporal_mask_for_window(w, 60)` 取行/列；跳数用邻接矩阵幂（先把 token-0 hub 置零） |
| §2.2 / §2.3 自相关 | 直接解析 `dataset/*/bvhs/*.bvh` 的 MOTION 块（4028 个文件，924 个 locomotion 名字，抽 400），按训练管线线性 resample 到 60 帧，逐通道归一化自相关后按方差加权平均 |

**已知口径限制**：自相关是在 BVH 原始通道（euler 角 + root 位移）上算的，不是 13 维 HML 特征，
方向可信，数值不能直接当作模型输入特征的相关系数。步时是 eager，未开 `--compile`；
compile 只会让 launch-bound 更明显，不改变 §1 的结论。

---

## 10. 实施记录

**2026-09-01：§8 步骤 1~4 完成，代码已删干净。**
**2026-09-02：步骤 5（重训 200k 步）与步骤 6（§6 判读）完成，新 ckpt 转正 —— 见本节末与 §6.1。**

### 与方案的差异

三处，都是方案没写细但实施时必须定的：

1. **`temporal_template` 与 `temporal_mask` 两条参数链一并删除，不只是传 `None`。**
   §3.3 #4 写的是 `temporal_mask = temporal_template = None`，但两者删窗口后都恒为
   `None`：`temporal_template` 的唯一用途就是 §3.3 #5 删掉的那个 `tt`；
   `temporal_mask` 删窗口后全仓库无任何非 `None` 供给方，留着就是一个死形参
   （`attn_mask=None` 与参数不存在逐 bit 相同）。所以它们从 `AnyTop.forward` →
   `GraphMotionDecoder.forward` → `GraphMotionDecoderLayer.forward` →
   `CrossLimbTemporalBlock.forward` 的签名里全部删除；`_temporal_mha_block_sin_joint`
   直接调 `temporal_attn(x, x, x, key_padding_mask=...)`，不再接收 `attn_mask`。
   代价：将来若要恢复时间先验（如 §5 的 ALiBi）需重新加回这条参数链。

2. **args.json 的写入抽成 `write_args_json(args, args_path)`**（`train_anytop.py`）。
   `args.version = CKPT_VERSION` 直接内联在 `run_training` 里没法测（要跑整个 `run_training`），
   抽出来之后写入侧和读取侧可以做一次真正的往返测试。

3. **`train.bat` 的 `RUN_NAME` 从 `merged_locomotion_v3` 改成 `merged_locomotion_v4_fullattn`**
   （§8 步骤 5 的运行名）。不改的话，`--auto_resume` 下一次运行会去 resume v3 的 save_dir，
   然后被新的 resume 版本守卫拦下——正确但没有意义。

另有两处顺带清掉的死代码：`AnyTop.forward` 里 `assert 'n_joints' in y`
（它在 70 行之前就已经被 `y['n_joints']` 解引用过，永远触发不了），
以及 `parser_util` 里随 `_cli_flag_explicit` 一起变得无用的 `import sys`。

### 版本守卫的落点

- `CKPT_VERSION = 1`、`assert_checkpoint_version` 在 `utils/parser_util.py`；
  `extract_args` 用它替换了原来的两个 `assert_*_not_deprecated`。
- 写入：`train/train_anytop.py` 的 `write_args_json`。
- resume：`train/train_anytop.py` 的 `assert_resume_checkpoint_version`，
  紧挨 `assert_resume_keeps_action_group` 调用，两者都在 `prepare_save_dir` 之后、
  args.json 被覆写之前。这条是必须的：resume 会整份重写 args.json，
  等于把新版本号盖到旧语义训出来的权重上，是生成侧守卫唯一看不见的口子。

### 测试

现有 445 个测试全绿。新增/改写的部分：

| 文件 | 内容 |
|---|---|
| `tests/test_diffusion_loss_precision.py` | `test_anytop_forward_reuses_shared_temporal_template_for_masks` 换成两条：`..._leaves_temporal_attention_unmasked`（decoder 既不收 `temporal_mask` 也不收 `temporal_template`，只有 spatial mask 进 decoder）与 `..._runs_the_real_cross_limb_decoder_without_a_temporal_mask`（跑真 decoder，不用 capture stub，确保 cross-limb 块不炸） |
| `tests/test_action_group_checkpoint_binding.py` | 新增 `VersionIsBoundToTheCheckpoint`（当前版本放行 / 无 `version` 拒绝 / 版本不符拒绝 / 训练写入的版本号生成侧认得）与 `ResumeCannotCrossAVersion` |
| `tests/test_native_loop.py` | 删 `test_circular_temporal_mask_wraps_motion_frames_only`（§3.4：这是窗口通路唯一的显式契约） |
| `tests/test_cross_limb_temporal.py` | 删 `_template` 及全部 `tt` 传参。注意 `test_full_batch_equals_per_sample_sliced` 的强度略降：原来靠 per-batch 的 mask 图案抓 batch 维转置，现在靠 per-batch 的 `x` / `kpm` / `unreliable`，仍然抓得住 |
| batch 元组下标（§3.3 #7 的风险点） | `temporal_mask` 原在位置 `[5]`，删后其后各项整体前移一位。已同步：`tensors.py` 的布局注释与全部 `b[...]`、`tools/sample_augmented_bvh.py` 的解包、`tests/test_dataset_loop.py` 的 `sample[12], sample[13]` → `sample[11], sample[12]`、`tests/test_canonical_features.py` 的 collate fixture |

### 重训与判读（2026-09-02）

`train.bat` 跑完 200k 步（`RUN_NAME=merged_locomotion_v4_fullattn`），
按 §6 的四条判读，**全部通过，新 ckpt 转正** —— 数据与归因说明见 **§6.1**。

两件实施期才知道的事：

1. **本轮重训与关键词化那一轮合并了**，所以 `args.json` 的 diff 比 §6 预期多一项
   `action_label_coarse_prob`，四条判读都带一个归因混淆项（§6.1 开头）。
2. **§3.2 的版本守卫第一次实战触发**：想拿 v3 的 ckpt 回跑一条对照探针时被直接拒绝
   （`args.json` 无 `version`），行为符合设计 —— 报错，不是跑错。
   代价是 v3 侧再也做不了新的对照实验，只能读它已有的产物（§4 代价表里
   "旧 ckpt 只能当历史结果读"这条，实测就是这个手感）。

遗留一个待办，**不阻塞转正**：域外骨架 + `--loop` + 域外动作三者叠加时的高频抖动
（§6.1 第 3 条）。语料内不复现，不触发 §5 的 ALiBi 备选。
