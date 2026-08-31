# species 条件通路改造方案：per-joint 加性 → per-joint FiLM

> 状态：**方案 A 代码已实施（2026-08-31），重训与验证（§6）未开始**
> 改动范围：仅 `InputProcess`（[anytop.py](../model/anytop.py)），`species_cond` 不动，cond.npy 无需重生成。
> 目标：把 `--species_joint_cond` 从"统一加性偏置"升级为"per-joint 乘性调制（FiLM）"。
> 验证方式：**全量重训 200k 步，与既有 checkpoint `save/merged_locomotion_v1` 直接对照**。

---

## 0. 现状：两条物种条件通路

两条通路消费的是**同一个** `species_emb`（每物种一个、由 T5 编码 `species_tags.jsonl` 的
motion-relevant tags、烘焙进 cond.npy 的固定 768 维向量）：

| 通路 | 开关 | 输入 | 作用对象 | 形式 | drop |
|---|---|---|---|---|---|
| timestep FiLM | `--species_cond` | 纯 `species_emb` | timestep token（`_apply_species_film`，`anytop.py:707`） | 乘性 `gamma*t+beta`，**每个 decoder 层重新注入**（`motion_transformer.py:1189`） | `--species_cfg_drop_prob`（默认 0.15） |
| per-joint 加性 | `--species_joint_cond` | `species_emb`（**仅此**） | 每个关节名 T5 embedding（`InputProcess.forward`，`anytop.py:915` 起） | 加性 `joints_emb + species_proj(species_emb)`，**仅输入层** | 无，**始终在场**（缺 `species_emb` 直接报错） |

两条通路正交（作用空间不同）：

- **timestep FiLM** = 信号/时间空间：全局 body-plan 动力学（节奏、能量、跨关节协调）。
- **per-joint** = 语义空间：把"每个关节是什么"的语义往该物种的 body-plan 方向偏移。

## 1. 动机：加性路的退化是可证的

`text_embedding` 是纯 `nn.Linear`，`species_proj` 与它之间没有任何非线性，所以：

```
E·(j_i + W·s + b_W) + b_E  =  E·j_i  +  [E·W·s + E·b_W + b_E]
                                        └──── 与关节 i 无关，也与帧无关 ────┘
```

结论（构造上成立，非经验判断）：

1. **现状 per-joint 路的 per-joint 分化量严格为零**——它等价于给每个关节 token 加**同一个
   per-species 常向量**，且沿帧广播后对时间也是常量。"per-joint"的 per 只体现在**加在哪**。
2. **那 590K 参数的有效自由度是一个秩 ≤256 的线性映射** `R^768→R^256`，`species_proj`
   与 `text_embedding` 复合后可被一个 `Linear(768,256)` 完全替代。绝大部分是浪费的。

FiLM 版 `gamma_j ⊙ joint_emb + beta_j`（`gamma/beta = F(species_emb, joint_emb)`）：
每个关节的缩放/偏移都是 `species × joint` 交互的函数，且 gamma≠1 提供缩放自由度。

### 1.1 期望要放对：这是"把交互挪到输入层"，不是"能力解锁"

必须诚实记录的折价——**加性路在端到端层面并没有堵死 `s×j` 交互**。
token = `E·j_i + c(s)`，每个 token 都同时携带 j 和 s，下游 8 层 / 16M 参数的 transformer
原则上可以自己算出 `f(j_i, s)`；它撞的只是"求和把两者纠缠在一起、难以分离"这个障碍
（与 §4 里 FiLM 输入必须用 concat 而非 sum 的理由完全同源）。

所以 FiLM 的真实贡献是：**把一个原本要在深层绕远路才能学到的交互，挪到输入层、变便宜、变显式。**
这是"小幅提升"的画像，不是台阶式跃升。方案的先验成功率应按此校准，
不要拿"加性做不到"当作必然收益的论据。

### 1.2 已知的反向证据

- `species_cond_no_lsimple_effect`：species 条件此前对 l_simple 无影响，诊断的**第一条原因是
  "与 joint-name embedding 冗余"**——FiLM 不解决这条。模型已看到每关节完整 T5 名字嵌入、
  拓扑边关系、rest pose、骨长；`species_emb` 在很大程度上是这些信息的有损摘要。
- **描述符碰撞**：`dataset/merged/cond.npy` 中 260 个物种只有 **121 个不同的 `species_emb`**，
  196 个（75%）与他人共用描述符。这压低了任何 species 通路的天花板。
  （但对 FiLM 的**相对**优势是中性偏正：Bat 与 Crow 共用 `s`，加性路发给它们字面同一个向量，
  FiLM 因 `j` 不同仍能分开。）
- 本通路**不做 CFG drop**，零初始化下没有任何东西逼 gamma 离开 1。而既有 `species_film`
  结构相同、还多了 0.15 的 drop 提供压力，仍未打出 l_simple 效果。

## 2. 关键设计决策：为什么是"两个 FiLM 并存"而不是"一个统一 FiLM"

曾考虑让**一个** FiLM 同时调制 timestep token 和 joint embedding，被否决：

1. **输入语义不匹配**：timestep 是全局的，需要**纯物种**全局先验；joint 需要 `species×joint`
   分化输入。硬塞 `species+joint` 进 timestep 要么 broadcast 掉 joint 维度（丢信息），
   要么拿"所有关节平均"的调制（语义错位）。
2. **drop 一致性**：timestep FiLM 可 drop（CFG），per-joint 是物种保底、不可 drop。
   一个 head 无法对两个对象做独立 drop 决策。
3. **输出维度不同**：timestep FiLM 输出 `2*latent_dim`，joint FiLM 输出 `2*t5_out_dim`。

> 两条通路共享 `species_emb` 只是**信息源共享**，不是功能冗余。

## 3. 最终方案（A：原地替换）

| 通路 | 输入 | 作用对象 | 形式 | 状态 |
|---|---|---|---|---|
| `species_cond` | 纯 `species_emb` | timestep token | FiLM（现状） | **保留，不动** |
| `species_joint_cond` | `[joint_emb ‖ species_emb]` concat | 每个关节名 embedding | **加性 → FiLM** | **唯一改动点** |

`--species_joint_cond` 开关语义不变，内部实现从加性换成 FiLM；**不新增开关**，
不保留加性路径。

**这是一道单向门：** `species_proj` 键消失、`species_film_j` 键新增，
[model_util.py:73-81](../utils/model_util.py#L73-L81) 的 `load_model` 对
unexpected 和 missing **两边都断言**（只豁免 `.q_norm/.k_norm`），
所以改完之后**所有既有 `species_joint_cond=True` 的 checkpoint 都不再可加载**
（包括 resume 和 `sample/generate.py`）。既然走全量重训，这是可接受的代价，
但要保留旧代码路径（git tag / 分支）以便回放旧 checkpoint。

参数量（实测 `save/merged_locomotion_v1`，模型总计 16.43M）：

| | 参数量 | 净变化 |
|---|---|---|
| 现状 `species_proj` | 590,592 | — |
| 新 `species_film_j`（concat + latent_dim 瓶颈） | 788,224 | **+197,632（+1.2%）** |

cond.npy 无需重生成（`species_emb` 与 `joints_names_embs` 均已烘焙）。
`--species_tags` 推理换风格的开关判定不变（判 `species_cond OR species_joint_cond`）。

## 4. 实现（`InputProcess`，`anytop.py:877` 起）

三个实现细节是必须的，不是可选优化：

**(a) FiLM 输入必须 concat，不能求和。** 方案 A 删掉了加性路，没有 `W(s)` 残差兜底，
所以"新方案至少能表达旧方案"这条底线**只能由 concat 提供**。
若输入用 `j + s`，则 `beta` 只看得到和，要复现 `W(s)` 就必须在同一个和上区分不同 (j,s) 来源，
只能靠记忆一个高度不光滑的映射，且对 `--species_tags` 的新 `s` 完全不泛化。
concat 下 `gamma_res=0, beta([j,s])=W(s)` 平凡可表示。

**(b) FiLM 的条件输入必须取 dropout 之前的 joint embedding。**
`joints_names_dropout`（p=`--dropout_prob`，默认 0.1）在 `anytop.py:926` 先于融合执行。
加性路免疫（`species_proj` 只吃 s）；FiLM 若吃 dropout 后的 j，则训练时用随机置零并按
1/0.9 缩放的 j 算 gamma/beta、推理时用干净 j，**调制参数本身**产生 train/eval 分布错位。
dropout 只作用于被调制的那一份。

**(c) 隐藏层用 `latent_dim` 瓶颈，不用 `t5_out_dim`。**
与既有 `species_film`（`anytop.py:187-191`）房规一致，且把净增参数从 +1.77M 压到 +0.2M。

```python
# __init__（替换 self.species_proj）
self.species_film_j = nn.Sequential(
    nn.Linear(2 * t5_output_dim, latent_dim),   # (a) concat -> 2*768 入
    nn.GELU(),
    nn.Linear(latent_dim, 2 * t5_output_dim),   # (c) latent_dim 瓶颈
) if species_joint_cond else None
if self.species_film_j is not None:
    nn.init.zeros_(self.species_film_j[-1].weight)   # 零初始化 -> identity 起步
    nn.init.zeros_(self.species_film_j[-1].bias)

# forward（替换加性融合）
joints_clean = joints_embedded_names.to(x.device)          # (b) dropout 前的干净副本
joints_embedded_names = self.joints_names_dropout(joints_clean)
if self.species_joint_cond:
    if species_emb is None:
        raise ValueError(
            "species_joint_cond is enabled but species_emb was not passed to "
            "InputProcess (expected y['species_emb'])."
        )
    # joints_clean: [B, J, t5]; species_emb: [B, t5] -> broadcast to [B, J, t5]
    species_broadcast = species_emb.to(device=x.device, dtype=joints_clean.dtype).unsqueeze(1).expand(-1, joints_clean.shape[1], -1)
    gamma_residual, beta = self.species_film_j(
        torch.cat([joints_clean, species_broadcast], dim=-1)
    ).chunk(2, dim=-1)
    joints_embedded_names = (1.0 + gamma_residual) * joints_embedded_names + beta
joints_embedded_names = self.text_embedding(joints_embedded_names)
```

`InputProcess.__init__` 签名里已有 `latent_dim`，调用点（`anytop.py:109`）无需改动。
`species_cond`、`_resolve_species_active`、`_apply_species_film` 一行不改。
`utils/parser_util.py:196` 的 `--species_joint_cond` help 文案需同步改写（当前描述的是加性行为）。

## 5. 风险

**R1 — 下限不再是精确的（方案 A 特有）。**
concat 保证 FiLM 的**最优解**至少不劣于加性，但零初始化起步意味着它要**从头学出** `W(s)`，
而不是继承。训练早中期完全可能短暂劣于 base。判读曲线时不要用中途 checkpoint 下结论。

**R2 — `--species_tags` 的 OOD 外推（原方案未列）。**
这是 `species_emb` 训练中**没见过**的唯一路径（[generate.py:1490+](../sample/generate.py#L1490)
用 T5 现场编码任意 tags）。今天它只过一个线性 `W(s)`，外推平缓、影响有界；
改后它驱动非线性 MLP 产生**乘性 gamma**，直接缩放关节语义本身，OOD 的 s 可能给出破坏性 gamma。
concat 让物种保持独立通道、已大幅缓解。**若实测 `--species_tags` 出现崩坏**，
再把 `gamma = 1 + gamma_res` 改为 `1 + tanh(gamma_res)` 夹紧到 (0,2)
（默认不加，与既有 `species_film` 保持一致）。

**R3 — 零初始化 FiLM 停在 identity**（退化回"无 per-joint 注入"，比加性略弱）。
缓解：concat 输入里 `joint_emb` 是 per-joint 变化的，天然自带分化信号。
但"是否真学到分化"必须靠 §6 的 gamma 指标判定，不能只看总 loss。

**R4 — 旧 checkpoint 全部失效**（见 §3 单向门）。重训前打 git tag。

## 6. 验证：全量重训 vs 既有 checkpoint

### base arm（已存在，无需重跑）

`save/merged_locomotion_v1` —— 200k 步已完成（08-30 23:36 → 08-31 10:10，约 10.6h），
`species_joint_cond=True`（加性），跑在 08-30 重生成的 `dataset/merged/cond.npy` 上，
**晚于 scale-normalization 修复，不是陈旧 checkpoint**。

### film arm（本次重训）

除 `InputProcess` 的代码改动外，**所有超参与 seed 必须逐项对齐 base**，否则不构成对照。
base 相对 parser 默认值的全部非默认项：

```
--action_group locomotion --action_label_cond --action_label_coarse_prob 0.3
--amp_dtype bf16 --auto_resume --balanced --compile default
--cond_path dataset/merged/cond.npy
--cross_limb_dim 128 --cross_limb_last_n 4
--ema_rate 0.995 --use_ema --ff_size 2048 --layers 8 --latent_dim 256
--global_energy_cond --global_energy_cfg_drop_prob 0.3
--joint_mask_prob 0.3 --temporal_span_mask_prob 0.3
--temporal_span_seam_loss_weight 0.2 --temporal_window 41
--lambda_geo 0.1 --lambda_loop_wrap 0.04 --lambda_vel 0.2
--loop_cond_prob 0.5 --weight_decay 0.01
--main_process_prefetch_batches 64 --motion_cache_size 32768
--ml_platform_type TensorboardPlatform
--num_steps 200000 --save_interval 5000
--species_cond --species_joint_cond
```

（`seed 10`、`objects_subset all`、`batch_size 16`、`lr 1e-4` 等均为 parser 默认，
无需显式传。`--save_dir` 由 `train.bat` 的 `RUN_NAME=merged_locomotion_v2` 决定，即 `save/merged_locomotion_v2`。）

跑完后用 args.json 逐键 diff 复核，只允许 `save_dir` 不同：

```python
import json
a = json.load(open('save/merged_locomotion_v1/args.json'))
b = json.load(open('save/merged_locomotion_v2/args.json'))
allow = {'save_dir'}
d = [k for k in set(a) | set(b) if k not in allow and a.get(k) != b.get(k)]
print("对照被破坏的键:", d or "无 — 配置严格可比")
```

### 判读

**先看机制，再看质量。** 只看总 loss 无法区分"FiLM 学到了分化"与"FiLM 停在 identity"，
而 §1.2 已记录 species 通路历史上对 l_simple 不敏感。

| 指标 | 含义 | 判据 |
|---|---|---|
| **同一物种内 gamma 跨关节的 std** | 直接检验 per-joint 分化——方案的全部立论 | 接近 0 ⇒ FiLM 退化为加性的等价物，方案作废 |
| 固定关节、gamma 跨物种的 std | 物种分化 | 接近 0 ⇒ 只是学了个与物种无关的关节重标定 |
| `species_film_j[-1].weight` 范数 | 偏离 identity 的程度 | 必要不充分：范数涨但跨关节 std≈0 说明没学到点上 |
| 跨物种 locomotion 生成质量 | 最终目标（步态节奏/关节幅度） | 主观，须配合下面的物种选择 |

**评测物种必须避开描述符碰撞。** 260 个物种中只有 64 个描述符唯一；
原方案选的"跨物种步态差异"正是被碰撞压平的那根轴
（Bat/Bird/Crow/Parrot/Parrot2/Pigeon 共用一个向量；Elephant/Mammoth/Rhino/Stego/Tricera 共用一个）。
对照物种请从唯一描述符集合中挑，重生成列表：

```bash
python -c "
import numpy as np, collections
c=np.load('dataset/merged/cond.npy',allow_pickle=True).item()
h=collections.defaultdict(list)
for k,v in c.items(): h[np.asarray(v['species_emb'],dtype=np.float32).tobytes()].append(k)
print('\n'.join(sorted(g[0] for g in h.values() if len(g)==1)))"
```

建议的 locomotion 对照对（体态差异大、描述符互异）：
`truebones/zoo/Cat`、`truebones/zoo/Centipede`、`truebones/zoo/Flamingo`、
`truebones/zoo/SpiderG`、`truebones/zoo_upgrade/Donkey`、`unitybundles/MU04_Earthworm`。

### 结论口径

- 若 **gamma 跨关节 std ≈ 0**：FiLM 停在加性等价点，方案作废，回滚到 git tag。
- 若 **gamma 有分化但生成质量与 base 无差**：per-joint 分化在学但不影响目标，
  说明瓶颈在 §1.2 的冗余/碰撞，应转向修描述符碰撞而非继续改架构。
- 若 **film > base**：采纳，同时按 R2 复测 `--species_tags` 的 OOD 行为。

## 7. 附：相关结论速查

- `--species_cfg_drop_prob` 不是"让 species_cond 正常工作"所必需的，它的目的是
  (a) 给零初始化 FiLM 学习激励，(b) 训练出无条件模式供 CFG。不做 species CFG 时设 0 即可。
- per-joint 通路（加性或 FiLM）是**唯一不可 drop** 的物种保底通道。
  推论：`--species_cfg_drop_prob` 的"无条件"分支并非真正物种无条件，species CFG 的引导强度
  被稀释；改成 FiLM 后这条常开路更强，**稀释会加剧**。不是阻断项，但用到 species CFG 时要知道。
- 两条通路正交性的真正来源是**输入不同**（纯 species vs species×joint）+ **作用对象不同**
  （timestep vs joint），而非"机制强弱"。
