# action_label 关键词化 + 去 multi-hot 方案

> 状态：**§8 步骤 1-5 已实施（2026-09-01），待训 R1**
>
> 已完成：
> 1. §5 词表小修 —— `strafe` 移出 `turn`，`sprint`/`shuffle`/`strafe` 进
>    `ACTION_VOCAB`。**外加一条同类修正**：`backward`/`backwards` 也从 `retreat`
>    的表面形式里移出（它当初在那里只是因为没有方向轴，正在 54 个不是撤退的
>    clip 上点亮 `retreat`：`knocked backward tumbling`、`collapses backward`）。
> 2. §2.6 迁移 —— 一次性迁移工具已重写三份 `action_labels.jsonl`（4028 条 →
>    610 个不同 label），并在各数据集目录写出 `*_action_label_review.tsv`
>    供人工过目；三份 `action_label_embs.npy` 已重建。实测新 label 的余弦与
>    §1.1 预测一致（`walk, forward` vs `walk, left` = 0.690，vs `run, forward`
>    = 0.694，L/R = 0.775）。
> 3. §2.5 `action_label_coarse_prob` 全链路删除。
> 4. §3.2 + §3.4 去 multi-hot；`action_label_projection` / `action_label_null_emb` 保留。
> 5. §4 评测工具 [`eval/direction_following.py`](../eval/direction_following.py)
>    （`sweep` / `sheet` / `score` / `phase`）。
>
> **未实现（按判据，都要等评测结果才决定）**：
> §3.3 的 FiLM 备选方案 —— 没有 flag，没有代码；
> §3.5 的 L/R 两位硬输入兜底。
>
> 待办：§2.6 第 3/4 步的**人工过目与方向补标**（见迁移脚本的 review TSV）、
> **跑 R0 基线**、训 **R1**、按 §4 的 scale 曲线分流。
>
> ---
>
> 原状态：**方案已定，待实施**（FiLM 一项列为 §3.3 的备选方案）
>
> 背景：`merged_locomotion_v3` 训练后暴露两个问题。问题 2（生成动作在时间维上
> 时快时慢）已定位为推理端从不填 `loop_phase_length`、导致 `(playspeed, phase_len)`
> 落在训练流形之外，已由 b30fefe 修复。问题 1（**方向性塌缩**）在补上推理端
> action-label CFG 之后（00d5abb，契约见
> [action_group_label_refactor.md §9.6](action_group_label_refactor.md)）
> 步伐混乱有明显改善，但**朝向的指令遵循性依然弱**。本文只处理这个残留。
>
> 三项决定（已拍板，不再比选）：
> 1. **action_label 全面关键词化**，不保留短语描述（§2）；
>    `action_label_coarse_prob` 及相关代码**直接删除**（§2.5）
> 2. **去掉 multi-hot 机制**（连同 core word 的概念），action_label 只走 frozen-T5（§3）。
>    注入方式**保持加性 token**；FiLM 列为备选方案（做成 flag），**默认关**（§3.3）
> 3. 词表小修：`strafe → turn` 等三处误映射（§5）
>
> 影响面：重新生成 `action_labels.jsonl` + `action_label_embs.npy`，**重训**。
> `cond.npy` 不受影响。

---

## 0. 为什么 CFG 没有把方向救回来

CFG 放大的是"条件 vs 无条件"之差，它**不能凭空造出条件里不存在的区分度**。
方向的区分度在两个地方被吃掉了：

1. **硬路径（multi-hot）里没有方向**。[`ACTION_VOCAB_CORE`](../data_loaders/truebones/truebones_utils/motion_labels.py)
   的 24 个核心词全是动作，没有方向词。实测：

   ```
   walk, strides forward with arms swinging    → multihot ['walk']
   walk, strides left with arms swinging       → multihot ['walk']    ← 完全相同
   walk, strides right with arms swinging      → multihot ['walk']    ← 完全相同
   walk, retreats backwards with arms swinging → multihot ['walk','retreat']  ← 只有后退偶然可分
   ```

   三个方向在硬路径上是**同一个向量**，差为 0，乘任何 CFG scale 还是 0。

2. **软路径（frozen-T5 mean-pool）把方向稀释成噪声**。见 §1。

CFG 只能放大第 2 条里那 0.05 的余弦差，救不了第 1 条。这解释了观察到的现象：
"步伐混乱（模态混合）有改善，朝向遵循性没有"。

**注意这个诊断指向输入端的稀释，不指向注入强度** —— 这是 §3.3 把 FiLM
单独拎出来列为备选方案、而不是跟着这一轮一起上的原因。

---

## 1. 证据：整句 label 把方向稀释了多少

全部数字用训练实际使用的 `t5-base` 现算，mean-pool 后取余弦。

### 1.1 整句 vs 关键词

| 度量 | 现状（整句 label） | 关键词化后 | 变化 |
|---|---|---|---|
| 方向对比 `walk,forward` vs `walk,left` | **0.933** | **0.690** | 对比度 0.067 → 0.310，**放大 4.6×** |
| 方向对比 vs `walk,right` | 0.953 | 0.727 | |
| 方向对比 vs `walk,backward` | 0.885 | 0.775 | |
| 动作对比 `walk,forward` vs `run,forward` | 0.831 | 0.694 | |

关键读数：**现状下方向差异（0.933）比 walk/run 的动作差异（0.831）还小** ——
条件向量里方向的显著性低于动作。关键词化之后两者拉平（0.690 vs 0.694），
方向变成和动作同等量级的信号。

### 1.2 查询侧同样被放大

| 查询 | label 集合 | top-1 | 与第二名的差 |
|---|---|---|---|
| `walk, moves forward` | 整句 label | `…strides forward…` 0.706 | **0.036**（第二名 `…right…` 0.670） |
| `walk forward` | 关键词 label | `walk, forward` 0.808 | **0.130**（第二名 `walk, backward` 0.678） |

用户自己造句时的判别裕度从 0.036 提升到 0.130（**3.6×**）。
另外 `--action_label "walk"` 在整句集合上对所有 walk 标签余弦 ≈ 0.36 且几乎等距 ——
这是"短 query 必然拿到各方向均值"的直接原因。

### 1.3 一个反例：方向词一律用裸形

| 词对 | 裸词余弦 | 放进 label 后 |
|---|---|---|
| `left` / `right` | 0.562 | 0.775 |
| `leftward` / `rightward` | **0.845** | **0.941** |
| `sideways` vs `backward` | 0.602 | – |

**用 `forward` / `backward` / `left` / `right` 的裸形**，不要 `leftward`/`rightward`
这类派生形容词（T5 把它们压成近同义词），也不要 `sideways`（语义糊，和 backward
0.602、和 leftward 0.645）。

---

## 2. action_label 全面关键词化

### 2.1 新语法

label = 受控词表里的词，按固定顺序、逗号分隔，**不写短语描述**：

```
walk, forward
run, sprint, forward, left
walk, crouch, retreat, backward
walk, strafe, left
idle
attack, bite
```

顺序固定为 **动作词（`ACTION_VOCAB` 顺序）→ 方向词（forward, backward, left, right）**。
固定顺序保证"一种词组合只有一种拼法"，沿用 §2.6 已有的设计意图。
空串仍然是无条件状态（§2.5 不变）。

`_validate_action_label_entry` 相应收紧：从"非空时须命中 ≥1 个受控词"改成
**每个逗号分隔的 token 都必须在词表里，且顺序必须是规范顺序**。
关键词化之后这是可以严格校验的，写错会当场报错而不是静默变成一个奇怪的 T5 向量。

### 2.2 需要补三个 modifier 词，否则关键词化会制造新的塌缩

关键词化会把"同一个动作词下靠短语区分的不同风格"合并。KI_Human locomotion
实测塌缩：

| 关键词 label | 合并的 clip |
|---|---|
| `run, forward` | `Run01Forwards`, `Sprint01Forwards` |
| `run, forward, left` | `Run01ForwardsLeft`, `Sprint01ForwardsLeft` |
| `run, left` | `Run01Left`, `Sprint01Left` |
| `run, right` | `Run01Right`, `Sprint01Right` |
| `walk, left` | `ShuffleLeft`, `Walk01Left` |
| `walk, right` | `ShuffleRight`, `Walk01Right` |

而且 §5 修完 `strafe → turn` 之后，`Strafe01Left` 会从 `walk, turn, left` 掉进
`walk, left`，变成三路塌缩。

**补救不需要新机制**，沿用 `flap` 的既有做法（它同时是 `fly` 的表面形式、
又是独立词；`vocab_words_in` 的等长匹配规则让同一 span 的两个词都保留）：
把 `sprint` / `shuffle` / `strafe` 加进词表，它们本来就已经是 `run` / `walk`
的表面形式，加完自动得到 `run, sprint` / `walk, shuffle` / `walk, strafe`。
去掉 multi-hot 之后词表是扁平的（§3.2），新增词**不改任何维度、不需要额外槽位**。

### 2.3 关键词化顺手消灭三个解析 bug

现在 `vocab_words_in` 是对**散文**做正则匹配，误命中三处：

| label | 误命中 | 原因 |
|---|---|---|
| `walk, strafe left with arms swinging` | `turn` | `strafe*` 被登进 turn 的表面形式。横移是纯平移、不改朝向，不是转身。locomotion 里点亮 turn 的 88 个 clip **有 27 个（31%）是横移** |
| `swim, glides left with arms and legs cycling` | `fly` | `glide*` 是 fly 的表面形式，在游泳 clip 上误触发 |
| `swim, crawl forward with arms reaching` | `crawl` | 自由泳的 "crawl" ≠ 四肢爬行 |

关键词化之后 label 本身就是词表里的词，**散文匹配这一步整个消失**，
`_VOCAB_SURFACE_FORMS` 退化成"迁移期一次性转换表"。三个 bug 自然消失。
迁移脚本跑之前仍要先修 `strafe` 的归属（§5.1），否则会把错的词写进新 label。

### 2.4 364 条没有方向词的 clip：不伪造，也不做几何推导

locomotion 1007 条里 **364 条（36%）** label 里没有方向词。实测**从 clip 名里
一条也捞不回来（0/364）** —— 它们主要是 zoo / zoo_upgrade 的动物 clip
（`Horse_Run`、`Bear_Walk`），名字里本来就不带方向。

**不做**从动画几何反推方向：语料里大量 clip 本身就是原地动作（in-place，
没有 root motion），`strip_translation_root_xz` 之前也量不出行进方向。
**方向只能靠人工标注，也只能靠人工标注来验证**（评测协议见 §4）。

因此：**"无方向"是一个合法的独立状态，label 就写 `run`**，和 `run, forward` 并存。
这和"空 label = 无条件"是同一种设计 —— 部分指定是合法的，模型学到的是
`run` 的边缘分布。

> 这不会重新引入塌缩。用户只说 `run` 时拿到各方向的均值是**正确**行为；
> 现在的问题是用户**明确指定了** forward 却仍然拿到均值。两者不是一回事。

需要方向可控的物种（KI_Human 这类方向齐全的），标注必须补齐；
只有一条 `Horse_Run` 的动物，`run` 就够了。**按需人工补标，不要全量补。**

### 2.5 删除 `action_label_coarse_prob` 及全部相关代码

关键词化之后 label **本身就是 coarse 串**，训练期的 coarse 增强不再有任何意义 ——
而且它现在是有害的：`coarse_label_from_words` 会把方向词再剥掉一次，
这正是当前 30% 的样本方向被抹除的来源（叠加 `action_label_cfg_drop_prob=0.2`
的硬丢弃，约 **44% 的 batch 完全没有方向信息**，且那 30% 是在**显式训练
"短 query = 各方向均值"**）。

**直接删除，不是设为 0**。删除清单：

| 文件 | 内容 |
|---|---|
| [`utils/parser_util.py:235`](../utils/parser_util.py) | `--action_label_coarse_prob` 参数 |
| [`train/train_anytop.py:180`](../train/train_anytop.py) | 传参 |
| [`data_loaders/get_data.py:64,79,101,119`](../data_loaders/get_data.py) | 两处 loader 的形参与透传 |
| [`data_loaders/truebones/data/dataset.py`](../data_loaders/truebones/data/dataset.py) | `__init__` 形参（672）、`self.action_label_coarse_prob`（674, 1180）、透传（1235）、`_resolve_action_label_condition` 里的 coarse 分支（1064-1066）、`coarse_label_from_words` 的 import（20） |
| [`motion_labels.py:433`](../data_loaders/truebones/truebones_utils/motion_labels.py) | `coarse_label_from_words` 函数本体 |
| [`tools/build_action_label_embeddings.py:49,70`](../tools/build_action_label_embeddings.py) | 不再为每条 label 额外烘焙一份 coarse 串，一条 label 只出一个向量 |
| [`tests/test_action_label_cond.py:19,123,130`](../tests/test_action_label_cond.py) | 对应用例 |

`_resolve_action_label_condition` 删完只剩"取 label → 查 emb"，
空 label 仍然走 null 路径（这条不变）。

### 2.6 迁移步骤

1. §5.1 + §5.2 词表小修（必须先做，否则会把错词写进新 label）。
2. 写一次性脚本：读现有三份 `action_labels.jsonl` 的散文 label，
   用 `vocab_words_in` 抽词 + 方向词抽取，按 §2.1 的顺序重拼成关键词 label，
   原地覆写 `action_label` 字段（`clip` / `action_group` 不动）。
3. 人工过一遍 swim 开头的 label（`glide` / `crawl` 误命中，只有 KI_Human 系几条）。
4. **人工补标方向**：只补需要方向可控的物种（§2.4）。
5. 用 [`tools/build_action_label_embeddings.py`](../tools/build_action_label_embeddings.py)
   重建三份 `action_label_embs.npy`。
6. 执行 §2.5 与 §3.4 的代码删除。

---

## 3. 去掉 multi-hot：action_label 只走 frozen-T5

### 3.1 选它的依据：T5 空间里"方向"是一个可组合的线性方向

跨 8 个动作词（walk/run/fly/swim/jump/crawl/retreat/turn）两两配对，
比较差向量的一致性：

| 差向量 | 平均余弦（28 个动作对） |
|---|---|
| `emb(a, backward) − emb(a, forward)` | 0.683（min 0.589, max 0.788） |
| `emb(a, left) − emb(a, forward)` | 0.665（0.494–0.741） |
| `emb(a, right) − emb(a, forward)` | 0.640（0.409–0.789） |
| 参照：`emb(a1, d) − emb(a2, d)`（动作轴） | 0.653 |

**方向轴的一致性和动作轴一样好。** 进一步做 leave-one-action-out 读出：
用**其它**动作词的样本建方向质心，去判一个没见过的 (动作, 方向) 组合 ——
**32/32 = 100%**。即：**一个线性读出头**就能从 T5 向量里恢复方向，
并且能泛化到训练集里没出现过的动作×方向组合。

这一点很关键，因为语料的组合覆盖是稀疏的：locomotion 的"动作词 + 方向"组合
共 **139 种，其中 105 种不足 5 条 clip**。硬编码的槽位对没见过的组合零泛化，
T5 的组合性正好补上这个洞。

扩展性同理：以后新增任何修饰轴（`sprint` / `crouch` / `uphill` / `wounded`）
**不改维度、不改任何 index layout**，只要训练集里出现过该词，推理时直接可用。

> "线性读出头就够"这条结论，同时也是 §3.3 的关键输入：
> 如果方向在输入向量里是线性可读的，那么一个线性投影 + 加性注入在**表达力**上
> 已经够用，FiLM 提供的是更大的函数类，不是更大的增益。

### 3.2 设计（默认路径）

**词表扁平化。** `ACTION_VOCAB_CORE` / `ACTION_VOCAB_DETAIL` 的区分**本来就是
"有没有 multi-hot 槽位"**，槽位没了区分也就没了意义 —— 合并成一个
`ACTION_VOCAB`（元组顺序 = §2.1 的规范拼写顺序），
再加一个 `DIRECTION_VOCAB = ("forward", "backward", "left", "right")`
排在动作词之后。`GROUP_MULTIHOT_MASK` 整个删除。

**注入方式保持现状（加性 token）。** `_build_action_label_token` 只是少掉
`action_multihot_projection(...)` 那一个加数，其余不变：

```
action_emb = action_label_projection(t5_emb)                    # 少了 + multihot_projection(...)
token      = where(active & valid, action_emb, action_label_null_emb)
timesteps_emb = timesteps_emb + token
```

`action_label_null_emb` **保留** —— 它仍然是无条件模式的落点，
[`ClassifierFreeActionModel`](../model/cfg_sampler.py) 在 uncond pass 上强制
`action_label_active=False` 的现有做法**一行都不用改**。

### 3.3 FiLM：**备选方案**（默认不上，留作后备）

结论：**列为备选方案，不纳入本轮范围**。实现成 `--action_label_film` flag、默认关闭；
仅当主方案（加性 token）按启用判据被证明不够时才启用，对应 §4 的 R2。
理由：

1. **诊断不指向注入强度。** §0/§1 定位的根因是**输入端稀释**（方向只有 0.05 余弦
   的差），关键词化把它放大 4.6× 直接对症。"加性 token 表达力不够"从来没有被证据支持过。
   而且 §3.1 已经证明方向在输入里是**线性可读**的 —— 线性投影 + 加性注入
   在表达力上就够。
2. **species FiLM 的先例不能平移过来。** 当初 species 从加性 token 换成 FiLM，
   原因很具体：那个零初始化的加性 token 和 per-joint 名字路径**冗余**，
   拿不到梯度而变成死通道（见 `species_cond_no_lsimple_effect` /
   [species_joint_film_upgrade.md](species_joint_film_upgrade.md)）。
   action_label **没有任何冗余路径** —— 骨架、关节名、物种描述都不能告诉模型
   这是走还是跑。当前模型对 walk/run 明显有响应，说明这条加性通道是活的。
   死通道那个失败模式在这里不成立。
3. **要"更大的增益"已经有更便宜的旋钮。** `--action_label_cfg_scale` 是推理期
   免费的增益调节，不用重训。FiLM 给的是更大的函数类，不是更大的增益 ——
   这两件事不要混。先把 CFG scale 扫一遍再谈换注入方式。
4. **两级 FiLM 是有真实风险的改动。** `species_film` 已经在乘性调制同一个
   timestep token，再叠一级 action FiLM 会让两者乘性复合；同时 CFG 在乘性通道上的
   响应曲线和加性通道不同（`out = uncond + s*(cond - uncond)` 仍然成立，
   但最优 s 会变）。这需要测，不能假设。
5. **归因。** 这一轮已经同时改了 label（关键词化）和条件路径（去 multi-hot）。
   再加第三个变量，结果不好时无法定位。

**如果要上，设计如下**（与 [`species_film`](../model/anytop.py) 同构）：

```
action_film: Linear(t5_out_dim, latent) → GELU → Linear(latent, 2*latent)   # 末层零初始化
gamma = 1 + gamma_residual;  timesteps_emb = gamma * timesteps_emb + beta
```

- 注入顺序：`species_film` → `action_film` → 再加各加性 token
  （playspeed / loop / canonical frame）。两级都零初始化，起点是恒等，
  初始化时不会互相打架。
- 分两级而不是合并成一个头，是因为 `species_cfg_drop_prob`(0.15) 和
  `action_label_cfg_drop_prob`(0.2) 是**独立**的丢弃掩码，且 CFG 只引导 action
  通道；合并成一个头就得为被丢的那一半再造一个 learned null 向量，反而更复杂。
- 开 FiLM 时 `action_label_null_emb` 不再需要，无条件样本 **bypass 到恒等**
  （gamma=1, beta=0），与 `species_film` 的 hard-drop 处理一致。
  `ClassifierFreeActionModel` 仍然不用改。

**启用判据（何时从备选转正）**：R1（§4）跑完并扫完 CFG scale 之后，
若指令遵循性**随 scale 上升而饱和在一个明显不够的水平**
（提高 scale 只增加伪影、不增加遵循性），说明瓶颈是表达力而不是增益 ——
这时才启用备选方案（跑 R2 试 FiLM）。若遵循性随 scale 正常提升，备选方案作废。

### 3.4 删除清单

| 文件 | 内容 |
|---|---|
| [`motion_labels.py`](../data_loaders/truebones/truebones_utils/motion_labels.py) | `ACTION_VOCAB_CORE` / `ACTION_VOCAB_DETAIL` 合并为 `ACTION_VOCAB`；删 `GROUP_MULTIHOT_MASK`、`group_multihot_mask`、`action_multihot_words`、`action_multihot_vector`、`vocab_words_in` 的 `core_only` 形参及相关 assert |
| [`data_loaders/tensors.py:5-31,153`](../data_loaders/tensors.py) | `_build_action_multihot_batch` 及 batch 里的 `action_multihot` 字段 |
| [`model/anytop.py`](../model/anytop.py) | `action_multihot_projection`(135,148)、`_build_action_multihot`(264)、`_coerce_action_multihot`(291)、`self.action_vocab` / `self.action_word_to_index`(119-121)。**`action_label_projection` 和 `action_label_null_emb` 保留**（§3.2）；FiLM 走 flag 时才另建 `action_film`（§3.3） |
| [`tools/visualize_action_separability.py:53,81`](../tools/visualize_action_separability.py) | `action_multihot_words` → `vocab_words_in` |
| [`tests/test_action_label_cond.py`](../tests/test_action_label_cond.py) | multi-hot 相关用例删除；补一条"删掉 multi-hot 后加性 token 与 null 契约不变"的回归用例 |

### 3.5 唯一的真实风险：left / right

关键词化之后 `walk, left` vs `walk, right` = **0.775**，比 forward/left 的 0.690
还近（裸词 0.562）。T5 把 left/right 当作同一句法槽的近义词 ——
**这是本方案唯一的已知失败点**。

因此 §4 的评测**必须把 `forward/backward` 和 `left/right` 分开报**，不能只看总体。

**兜底（只有 L/R 明显不达标时才启用）**：只为 `left` / `right` 保留 2 个 0/1
硬输入位（一个 2 维输入的小 MLP，加到 timestep token 上）。这是最小干预 ——
不恢复整套 multi-hot 词表，只解决 T5 分不开的这一对。

> 注意这个兜底和 §3.3 的 FiLM 是**两种不同的失败模式的解药**，不要混用：
> L/R 分不开是**输入端两个向量太近**（FiLM 救不了）；
> 全面遵循性随 scale 饱和才是**表达力不足**（硬位救不了）。
> 先看 §4 的分组指标是哪一种。

其它需要盯的次生风险：

- 失去"一个词一个学习列"的干净归纳偏置，小样本词（`crawl` 24 条、`retreat` 53 条）
  可能变弱。
- `GROUP_MULTIHOT_MASK` 原本承担"阻止跨 body plan 记忆"的闸门作用
  （见 [action_group_label_refactor.md §2.4.1](action_group_label_refactor.md)），
  T5 路径没有这个闸门。评测里要盯住 `swim` 这类只有 2 个 body plan 承载的词
  是否开始绑定物种。

### 3.6 方向词在语料里的覆盖（决定哪些方向值得期待可控）

| 方向 | clips | species | 覆盖的 body plan（≥3 物种） |
|---|---:|---:|---:|
| forward | 375 | 176 | 6 |
| right | 121 | 39 | 5 |
| left | 120 | 38 | 5 |
| backward | 90 | 36 | 5 |
| up | 10 | 5 | 1（仅 winged） |
| down | 8 | 5 | 0 |
| sideways | 3 | 2 | 0 |

四个平面方向的覆盖是够的（对照原 multi-hot 的入选门槛 clips≥10 / species≥5 /
body plan≥3，四个都过线）。**up / down / sideways 数据太薄**，
标注时可以写，但不要期待它们可控 —— 也不要为它们做专门的评测。

---

## 4. 验证与评测：方向只能人工标注

自动几何判定不可用（§2.4：大量 clip 是原地动作，量不出行进方向），
所以方向指标走**人工盲评**。

**轮次**

| 轮次 | 配置 | 回答的问题 |
|---|---|---|
| **R0** | 现有 v3 checkpoint | 基线 before 数字（不训练，只标注） |
| **R1** | 关键词化 label + 去 multi-hot + 加性 token | 主方案够不够 |
| **R2（备选）** | R1 + `--action_label_film` | 备选方案：只在 §3.3 的启用判据成立时才考虑 |

**提示集**：方向齐全的物种（KI_Human 等）× {`walk`, `run`} × {`forward`,
`backward`, `left`, `right`}，每个提示固定 seed 采 N=16~32 条。

**协议**：渲染成视频网格，**不显示提示词**，标注者为每条选一个标签
（forward / backward / left / right / 混合 / 无法判断）。

**主指标**：方向 top-1 准确率，**分开报 `forward+backward` 与 `left+right` 两组**；
"混合"单独计数 —— 它是塌缩的直接读数。

> **R0 基线必须先做。** 它便宜（不用训练），但没有它就无法判断新模型是真的好了
> 还是错觉，也无法给 §3.3 / §3.5 的两个兜底判据定阈值。

**可自动的次指标**：左右脚接触相位差的分布方差。contact 通道（channel 12）
不受 root XZ 剥离影响，**原地动作也能算**，是"左右脚交叉 / 猫步"的可测量代理，
适合做训练期的快速回归监控。

**CFG scale 扫描**：`--action_label_cfg_scale ∈ {1, 2, 3, 4, 6}`，每轮都扫。
**遵循性 vs scale 的曲线形状就是 §3.3 的判据本身** ——
单调提升 = 增益问题（不需要 FiLM）；早早饱和 + 伪影增加 = 表达力问题（启用 R2 备选方案）。

**其它**：[`eval/evaluate_motion_quality.py`](../eval/evaluate_motion_quality.py)
的既有指标不得退化。

---

## 5. 词表小修（随 §2 一起做）

### 5.1 `strafe` 从 `turn` 移出

[`motion_labels.py` `_VOCAB_SURFACE_FORMS["turn"]`](../data_loaders/truebones/truebones_utils/motion_labels.py)
里删掉 `"strafe", "strafes", "strafing"`，把 `strafe` 加进 `ACTION_VOCAB`。
横移是纯平移、不改朝向，和 turn 是两回事。`circle` / `bank` 留在 `turn`
（它们确实改朝向）。

**必须在迁移脚本之前做** —— 否则 27 个横移 clip 会被写成 `…, turn, …`。

### 5.2 `sprint` / `shuffle` 加进 `ACTION_VOCAB`

保留它们在 `run` / `walk` 表面形式里的位置不动 —— 等长匹配会让两个词都保留，
得到 `run, sprint` / `walk, shuffle`。解决 §2.2 的塌缩。

### 5.3 `glide` / `crawl` 的跨模态误命中

迁移脚本跑完后人工过一遍 swim 开头的 label（只有 KI_Human 系几条）。
关键词化之后散文匹配消失，这条规则不需要长期维护。

---

## 6. 本轮明确不做

| 项 | 内容 | 不做的理由 |
|---|---|---|
| S9 | 把行进方向作为模型条件通道（保留 root XZ，或补 body-local 方向向量） | 大量 clip 本身就是原地动作，没有 root motion 可保留；方向只能来自人工标注（§2.4） |
| S10 | 收紧 `_sample_loop_tile_count` 的平铺上限 | 问题 2 已由 `loop_phase_length` 补传解决（b30fefe），没必要再动数据增强 |
| S11 | `temporal_window ≥ num_frames`，或加全局时间 token | 同上；且 00d5abb 已让显式 `--temporal_window` 生效，需要时可在推理端单独试 |
| — | 方向专用 multi-hot 槽位 | 已否决：每加一个修饰轴就要改 index layout + 重训一次，且对 105/139 的稀疏组合零泛化。只保留 §3.5 的 L/R 兜底 |
| — | action_label FiLM | 列为备选方案（§3.3），默认不上；仅当 §4 的 scale 曲线显示表达力不足时才启用 |

---

## 7. 影响面清单

| 改动 | regen `cond.npy` | regen `action_labels.jsonl` | regen `action_label_embs.npy` | retrain |
|---|:---:|:---:|:---:|:---:|
| §2 关键词化 + 删 coarse | 否 | **是** | **是** | **是** |
| §3 去 multi-hot（加性 token 不变） | 否 | 否 | 否 | **是**（条件维度变，旧 ckpt 不兼容） |
| §3.3 FiLM（备选方案，可选） | 否 | 否 | 否 | **是**（仅当备选启用） |
| §5 词表小修 | 否 | **是**（迁移脚本读它） | **是** | 随 §2 |

`cond.npy` 全程不受影响 —— action label 的 embedding 在独立 sidecar
`action_label_embs.npy` 里。

---

## 8. 执行顺序

1. §5.1 + §5.2 词表小修
2. §2.6 迁移：脚本重写三份 `action_labels.jsonl` → 人工补标方向 → 重建三份 embs
3. §2.5 删除 `action_label_coarse_prob` 全链路
4. §3.2 + §3.4 去 multi-hot（`action_label_projection` / `action_label_null_emb` 保留）
5. 补 §4 的评测（人工盲评协议 + 相位方差脚本），**先跑 R0 基线**
6. 训 **R1**，按 §4 评测 + CFG scale 扫描
7. 按曲线形状分流：
   - 遵循性随 scale 正常提升，且 L/R 达标 → 收工
   - L/R 单独不达标 → §3.5 的 2 位硬输入兜底
   - 整体随 scale 早早饱和 → 启用 §3.3 的 FiLM 备选方案，训 **R2**
