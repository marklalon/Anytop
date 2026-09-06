# action_label 关键词化 + 去 multi-hot 方案

> **已作废（2026-09-06）** —— 本文是实现记录，保留作设计史：
> 它描述的**整句 T5 向量**条件、**每数据源**的 label-keyed `action_label_embs.npy`、
> surface-form 正则匹配与旧词表（含 `shuffle`）均已被 per-word 槽方案取代。
> 现行实现与标签契约以
> [action_label_per_word_pooling.md](action_label_per_word_pooling.md) 为准
> （词表 103 token 闭集、精确 token 解析、全局 word-keyed sidecar、checkpoint v2）。
> §0/§1（动机）与 §3（T5 而非 multi-hot 的决策）仍是现行方案的地基；
> §4 的 R0/R1 评测针对 v1 checkpoint，新代码（`CKPT_VERSION=2`）已拒绝加载该产物，
> 数字仅作历史参考。
>
> 状态：**方案完成（2026-09-02）** —— §8 步骤 1~7 全部走完，
> R1 已训（`merged_locomotion_v4_fullattn`）并评测，**主方案达标、两个备选方案都不启用**，
> 结论见 **§4.3**。
>
> 已完成：
> 1. §5 词表小修 —— `strafe` 移出 `turn`，`fast`/`shuffle`/`strafe` 进
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
> 5. §4 评测脚本 [`tools/direction_following.py`](../tools/direction_following.py)
>    （`sweep` / `score` / `sheet` / `phase`；方案执行期的临时验证工具，不在
>    常驻 eval 套件里）。
> 6. **R0 基线已跑**（KI_Human，640 clip），数据与结论见 **§4.2**。
>    同一轮把**主指标从 top-1 换成了连续角度误差** —— top-1 吸附到最近象限，
>    系统性欠转在它上面是隐形的（§4.2 的第一段就是这个更正）。
> 7. **R1 已训并评测**（同一提示集 640 clip，逐格对比 §4.2），结果见 **§4.3**：
>    cfg 2 上四个方向的角度误差全部进入个位数（四向均值 15.0° → **5.2°**），
>    `left` 的系统性欠转消失，`run` 的相位离散度同向收敛，
>    既有质量电池 16/17 项持平或改善。
>
> **§4 的方向指标现在可以自动判定，不必人工盲评**（见下面的 §4.1 补充）。
>
> **两个备选方案都不启用**（判据见 §4.3 末）：
> §3.3 的 FiLM —— 曲线形状不是表达力签名，**本轮作废**；
> §3.5 的 L/R 硬输入 —— R0 指向它的那笔账（`left` 欠转 24~34°）在 R1 上自己好了，
> 剩下的 `right` 在可用区间内只有 8.2°，不值一次重训，**留作观察项**。
>
> 待办：§2.6 第 3/4 步的**人工过目与方向补标**（见迁移脚本的 review TSV）；
> 推理端默认 `--action_label_cfg_scale` 定在 **2**（§4.3 第 3 条）。
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

> **R0 量完之后，这一段要打个折（§4.2）。** "朝向遵循性没有改善"是在
> `--action_label_cfg_scale` 默认 1.0（= 关）的情况下看的；真扫上去，
> forward 的角度误差 15.8° → 4.9°、right 15.6° → 6.4°，**CFG 确实把方向救回来了不少**，
> 那 0.05 的余弦差比这里预计的能放大。
> 站得住的只剩**一个**方向：`left` 全程 24~34° 不降。
> 所以本节"CFG 救不了方向"应读成"CFG 救不了 `left`"，
> 而结论（关键词化把方向的显著性提上来）不受影响 —— 它本来就该做。

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
run, forward, left, fast
walk, crouch, retreat, backward
walk, left, strafe
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

若不补 modifier，关键词化会把“同一个动作词下靠短语区分的不同风格”合并。
KI_Human locomotion 最终用下面的规范标签保住差别：

| 关键词 label | 对应 clip |
|---|---|
| `run, forward` | `Run01Forwards` |
| `run, forward, fast` | 高速向前 clip |
| `run, forward, left, fast` | 高速向左前 clip |
| `run, left, fast` | 高速向左 clip |
| `run, right, fast` | 高速向右 clip |
| `walk, left` | `Walk01Left` |
| `walk, left, shuffle` | `ShuffleLeft` |
| `walk, right` | `Walk01Right` |
| `walk, right, shuffle` | `ShuffleRight` |

`Strafe01Left` 则写成 `walk, left, strafe`，避免与 walk / shuffle 再次合并。

**补救不需要新机制**：把 `fast` / `shuffle` / `strafe` 加进扁平词表；标注迁移把来源名称里的
高速动作统一写成 `fast`，得到 `run, fast` / `walk, shuffle` / `walk, strafe`。
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

扩展性同理：以后新增任何修饰轴（`fast` / `crouch` / `uphill` / `wounded`）
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
若**角度误差随 scale 下降到某点就不再下降**、再推只增加伪影
（判据按方向逐条看，不是看总体均值），说明瓶颈是表达力而不是增益 ——
这时才启用备选方案（跑 R2 试 FiLM）。若误差随 scale 正常下降，备选方案作废。

> **R0 的读数（§4.2）不支持启用**：forward / right 的误差随 scale 正常下降
> （15.8→4.9、15.6→6.4），只有 `left` 一条不降 —— 而**单独一个方向词失灵是输入端的
> 症状，不是表达力不足**（若是表达力不足，right 会一起烂）。R0 因此把判据的天平
> 推向 §3.5 而不是本节。R1 之后重判。

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

因此 §4 的评测**必须把四个方向分开报**，不能只看总体，
**也不能把 left 和 right 合成一个 `left/right` 轴** —— 见下面 R0 的读数。

> **R0 实测（§4.2）：失败模式比预期更窄，而且轴平均会把它抹掉。**
> `right` 干净（5~7°），只有 `left` 系统性欠转 24~34° 且不随 scale 改善。
> 两者平均成一个 LR 轴是 ~16°，看着像"这一对都差一点"，
> 完全指错方向。**问题是 `left` 这一个词，不是这一对。**
> 注意 R0 有分布外混淆项（v3 训的是散文 label，见 §4.2 末），
> 所以这条证据**指向**兜底但还不足以启用它。
>
> **R1 实测（§4.3）：那个混淆项就是主因，兜底不启用。**
> 消除拼法不匹配之后 `left` 自己好了（bias −24.2 → +0.6 @cfg2）。
> 同一个签名换到了 `right` 上（bias 随 scale +5.0 → +20.2，`|bias| ≈ mean|e|`），
> 但量级小一档：可用区间 cfg 2 上是 8.2°，仍在个位数。
> **结论：本轮不上硬输入**；若人工判读仍看得见 `right` 偏侧，
> 届时按下一段的写法**只给 `right` 一位**。

**兜底（只有 L/R 明显不达标时才启用）**：只为 `left` / `right` 保留 2 个 0/1
硬输入位（一个 2 维输入的小 MLP，加到 timestep token 上）。这是最小干预 ——
不恢复整套 multi-hot 词表，只解决 T5 分不开的这一对。
若 R1 复现 R0 的不对称（只有 `left` 坏），可以退到**只给 `left` 一位**。
—— R1 复现了不对称，但坏的是 `right`，所以这一位要给 `right`（§4.3）。

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

## 4. 验证与评测

> ⚠️ **本节开头这条前提在实施时被推翻了，见 §4.1。**
> "自动几何判定不可用"只对 **root motion** 成立；**步态**里还留着方向，
> 而且在本节实际要用的那批物种上是 100% 可判的。
> 下面保留原文以便对照，实际按 §4.1 执行：主指标默认走 `score --auto`，
> 人工盲评降级为 `--auto` 拒绝的物种的兜底和抽查。

~~自动几何判定不可用（§2.4：大量 clip 是原地动作，量不出行进方向），
所以方向指标走人工盲评。~~

**轮次**

| 轮次 | 配置 | 回答的问题 |
|---|---|---|
| **R0** | 现有 v3 checkpoint | 基线 before 数字（不训练）。**已跑完，见 §4.2。** 注意 v3 权重里有 `action_multihot_projection`，当前代码已删掉这一路，`load_model_wo_clip` 会因 unexpected key 直接报错 —— R0 必须在重构前的 commit `00d5abb` 上跑（开一个 git worktree），`score` 两条路径都不碰模型，在哪边跑都行 |
| **R1** | 关键词化 label + 去 multi-hot + 加性 token | 主方案够不够。**已跑完，见 §4.3**；注意这个 ckpt 同时含全注意力改动，两项功劳分不开 |
| **R2（备选）** | R1 + FiLM 注入 | 备选方案：只在 §3.3 的启用判据成立时才考虑；**代码未实现**，要用得先写 |

**提示集**：方向齐全的物种（KI_Human 等）× {`walk`, `run`} × {`forward`,
`backward`, `left`, `right`}，每个提示固定 seed 采 N=16~32 条。

**协议**：渲染成视频网格，**不显示提示词**，标注者为每条选一个标签
（forward / backward / left / right / 混合 / 无法判断）。

**主指标**：~~方向 top-1 准确率~~ → **实测行进方向与提示方向的夹角**（带符号，
度），**按四个方向分开报，left 和 right 不许合并**。理由见 §4.2 第一段：
top-1 把测到的角度吸附到最近象限，±45° 内一律算命中，
所以"每条 left 都落在前左对角线上"这种**系统性欠转**在它上面完全看不见 ——
R0 上正是这个失败模式，top-1 报 96~100%，实际差 24~34°。
每格报 `mean|e| / median / p90 / bias`，`bias` 是带符号均值：
`|bias| ≈ mean|e|` 就说明整组一致地偏，不是散开，采再多条也平均不掉。

top-1 保留为**粗读**，也是人工盲评那条路唯一能产出的东西；
"混合"仍然单独计数 —— 它是塌缩的直接读数。

**读数必须对着语料地板看**：同一个估计器量该物种自己的真实 clip，
KI_Human 是 0.1~0.9°（§4.2）。所以个位数度才算"贴轴"，两位数就是模型没听指令。
`score --auto` 会把这一行自动打在表下面。

> **R0 基线必须先做。** 它便宜（不用训练），但没有它就无法判断新模型是真的好了
> 还是错觉，也无法给 §3.3 / §3.5 的两个兜底判据定阈值。
> —— **已做，见 §4.2**；而且它第一件事就是推翻了本节原来的主指标。

**可自动的次指标**：左右脚接触相位差的分布方差。contact 通道（channel 12）
不受 root XZ 剥离影响，**原地动作也能算**，是"左右脚交叉 / 猫步"的可测量代理，
适合做训练期的快速回归监控。

**CFG scale 扫描**：`--action_label_cfg_scale ∈ {1, 2, 3, 4, 6}`，每轮都扫。
**角度误差 vs scale 的曲线形状就是 §3.3 的判据本身** ——
单调下降 = 增益问题（不需要 FiLM）；不再下降甚至回升 + 伪影增加 = 表达力问题
（启用 R2 备选方案）。**曲线要按方向逐条看**：R0 上 forward/right 正常下降、
left 一路不降反升，两者混在一起会互相抵消（§4.2）。

**其它**：[`eval/evaluate_motion_quality.py`](../eval/evaluate_motion_quality.py)
的既有指标不得退化。

### 4.1 补充（实施时的修正）：方向可以自动判定 —— 用步态，不用 root motion

本节原来的前提是"自动几何判定不可用"。**这条只对 root motion 成立，对步态不成立。**

`strip_translation_root_xz` 把每个 locomotion clip 的 root XZ 剥成 0
（实测 `KI_Human_Walk01` 四个方向的 root 轨迹逐字节相同，span 全 0），
所以从 root 上确实读不出方向。但**剥掉 root 恰恰把 clip 变成了跑步机**：
支撑脚踩在不动的地面上，身体被钉住之后，那只脚必然以 **−行进速度** 滑动。
对着地脚的水平速度取平均再取反，就是身体坐标系下的行进方向。

两个实现要点：

* **支撑脚要用高度判定，不能用 contact 通道**。channel 12 是「速度低于阈值
  且贴近地面」，它恰好把这里要测的滑动筛掉了 —— 用它测出来的水平速度恒等于 0。
  改成「离该关节自己的 5% 分位地面 ≤ 0.05」。
* 规范帧里 **+Z = 前，+X = 左**（`process_anim` 把所有骨架转到面向 +Z，
  特征帧 `r_rot` 是 identity）。

**在全语料上验证过**（每个只带一个方向词的 locomotion clip）：

| 物种类别 | n | forward/backward | left/right |
|---|---:|---:|---:|
| 四方向人形 rig（KI_Human / KI_Archer / KI_Soldier / KI_Warrior / KI_CasterMage / LH_Hero / RMW_Skeleton） | 68 | **100%** | **100%** |
| 龙 / 四足（MB_TigerDrago、MB_Unka、Trex） | 30 | 100% | **0%** |

**后一行不是噪声，是词义不同**：四足和飞龙不会横移，它们的 `left` clip 是
**转弯**，身体坐标系下的行进方向确实是 forward。所以这个估计器
**只在 §4 实际要用的那批物种上成立**（本节的提示集本来就写的是
"方向齐全的物种（KI_Human 等）"），不能无条件推广。

因此 `score --auto` 的设计是**先自校准再打分**：
拿该物种自己的带标注 clip 跑一遍估计器，达不到阈值（默认 90%）就**拒绝打分**
并要求人工（`truebones/zoo/Scorpion-2` 就是这样被抓出来的 —— 它的左右是反的）。
另外三类一律跳过：label 里带 `turn`（转向 ≠ 行进方向）、带 `swim`/`fly`
（没有支撑脚）、以及行进速度低于地板值的（原地动作，没有方向可读）。

人工盲评（`sheet` + `score`）**保留**，用于 `--auto` 拒绝的物种和抽查。
唯一测不到的是 "mixed"：逐 clip 的几何测量只能给一个方向，看不出模态混合 ——
那一项仍然要么看 `phase` 的组内离散度，要么人工。

### 4.2 R0 基线结果（2026-09-01）

**跑法**：`merged_locomotion_v3` 的 `model000200000.pt`，在 `00d5abb` 的 git
worktree 里跑 `sweep`（该树没有 `DIRECTION_VOCAB` / `canonical_action_label`，
把这两个符号逐字内联进那份 tool 拷贝，保证 R0 和 R1 的提示词拼法一字不差，
其余文件不动）。KI_Human × {`walk`, `run`} × 四方向 × cfg{1,2,3,4,6} × 16 条
= **640 clip**，seed 10，DDPM 100 步。`score --auto` / `phase` 在主树跑。
原始输出在 `outputs/direction_following/R0{,_score.txt,_phase.txt}`。

只跑 KI_Human 一个物种（校准两轴都 OK，FB 7/7、LR 10/10）。
代价是每个 (方向, cfg) 格子只有 32 条，L/R 那一栏误差棒偏宽；
KI_Archer / KI_CasterMage / KI_Soldier / KI_Warrior 校准同样两轴 OK，
需要收窄时从它们里补（LH_Hero / RMW_Skeleton 语料里没有 L/R clip，LR 不可信）。

#### 先说指标本身被推翻了

R0 的 top-1 是这样的：cfg 1 是 81%/83%，**cfg 2 起就 97~100%**，cfg 6 略回落。
照这张表读，方向遵循性根本没问题、两个兜底方案都该作废 —— **这个结论是错的**。

人工抽查 `KI_Human__run__forward__cfg2` 时发现 16 条里有 5~6 条明显偏侧。
逐条量出来：`+17.3° +15.6° −11.8° −8.2° −7.9° +6.7°`，其余 10 条在 ±3.7° 内 ——
**正好 6 条**，和人眼数出来的一致。而 top-1 把这 16 条全记成命中，
因为没有一条跨过 45° 那条线。**top-1 是阈值判定，不是可累加的误差**，
它对系统性欠转是结构性盲的。主指标因此改成连续角度（§4 已改）。

#### 语料地板

同一个估计器量 KI_Human 自己的真实 clip，对着它们自己的标注方向：

| dir | n | mean\|e\| | p90 |
|---|---:|---:|---:|
| forward | 4 | **0.4°** | 0.9° |
| backward | 3 | **0.1°** | 0.2° |
| left | 5 | **0.6°** | 1.1° |
| right | 5 | **0.9°** | 1.6° |

真实片子贴着轴，所以下面每一度都是模型的账，不是测量噪声，
也不是"这个 rig 本来就走得歪"。

#### 角度误差（度，+ = 偏向角色左侧）

| cfg | forward | backward | left | right |
|---:|---:|---:|---:|---:|
| 1 | 15.8 | 30.8 | 37.7 | 15.6 |
| 2 | 7.2 | 22.2 | 24.3 | 6.4 |
| 3 | **4.9** | 19.1 | 26.7 | 7.2 |
| 4 | 5.5 | **15.6** | 28.0 | 7.3 |
| 6 | 8.2 | 27.5 | 33.8 | **4.8** |

四条结论：

1. **`left` 没有被 CFG 修好，而且越推越差**（24 → 27 → 28 → 34）。
   它的 `bias` 几乎等于 `mean|e|`（cfg 4：−28.0 对 28.0），
   即**整组一致地欠转**，全落在前左对角线上 —— 不是散开，采样平均不掉。
2. **`right` 是干净的（5~7°）**。所以这**不是** §3.5 预期的"L/R 这一对分不开"，
   是 **`left` 这一个词单独放不准**。右侧同时干净这件事顺带排除了"注入强度不够"：
   若是增益问题，right 会一起烂。按 §3.5 的分诊，这笔账在**输入端**，FiLM 救不了。
3. **cfg 可用区间是 2~3**。cfg 6 上 forward 从 4.9 退回 8.2、backward 从 15.6
   炸到 27.5（p90 64°），和人工看到的"cfg>3 动作明显变形"一致；
   人工同时判定 **cfg 2 动作质量最好**。cfg 4 只在 backward 上换来一点角度，
   代价是画质，不值。
4. **R1 的对比口径**：拿这张角度表逐格比，**不要**比"扫描后的 top-1 峰值" ——
   top-1 在 cfg≥2 已经饱和在 100%，没有 headroom，比不出东西。

#### 但有一个混淆项，所以 §3.5 现在还不能上

v3 是拿**散文 label** 训的（`walk, strides left with arms swinging`），
R0 却喂它**关键词** `walk, left` —— 对这个 checkpoint 是分布外输入。
`left` 的欠转有多少来自"训练/推理拼法不匹配"、多少来自 T5 放不准 `left`，
**R0 分不开**，而消除这个不匹配正是 R1 要做的事。
所以顺序不变：**先训 R1，再看 left 这一栏**；
若 R1 之后 `left` 仍然系统性欠转、且随 scale 不降，才启用 §3.5 的两位硬输入
（届时也可以只给 `left` 一位，而不是原方案的 L/R 两位）。

#### 次指标：相位离散度（`phase`）

同一提示 16 条样本的左右脚相位一致性，对照真实语料（circvar，0 = 完全一致）：

| 提示 | 生成 | 语料 |
|---|---:|---:|
| run, forward | 0.32 ~ 0.53 | **0.002** (n=4) |
| run, backward | 0.09 ~ 0.72 | **0.000** (n=3) |
| walk, forward | 0.18 ~ 0.52 | 0.117 (n=6) |
| walk, left / right | 0.08 ~ 0.66 | 0.43 ~ 0.55 |

真实 run clip 之间相位几乎完全一致，生成的散得很开：
**方向对了，步态仍在混模态** —— 这是"步伐混乱"的可测量残余，
也是 R1 值得盯的另一栏。注意语料 n 只有 3~4 条，circvar 会偏低，
别把 0.002 当硬阈值，看的是数量级差。

### 4.3 R1 结果（2026-09-02）

**跑法**：`merged_locomotion_v4_fullattn` 的 `model000200000.pt`，**主树直接跑**
（R0 那份 worktree 只是为了让旧 ckpt 能加载，R1 不需要）。提示集、seed、采样器
与 §4.2 逐项相同：KI_Human × {`walk`, `run`} × 四方向 × cfg{1,2,3,4,6} × 16 条
= **640 clip**，seed 10，DDPM 100 步。`score --auto` 校准与 R0 同样两轴通过
（FB 100% / LR 100%，地板 0.6°）。原始输出在
`outputs/direction_following/R1{,_sweep.log,_score.txt,_phase.txt}`。

> **读这一节前先记住的混淆项**：R1 的 checkpoint 同时含**两项**改动 ——
> 本文的关键词化，以及
> [temporal_window_full_attention.md](temporal_window_full_attention.md) 的全注意力。
> 逐格对比 R0 **无法把功劳分开**。两条线各自命中了自己的靶子（朝向角误差 ↔ 关键词化，
> `run` 的相位一致性 ↔ 全注意力），没有互相矛盾的迹象，但下面每一个数都要带着这一条读。

#### 角度误差（度），逐格对比 §4.2

R0 → R1，括号是差值，负号 = R1 更好：

| cfg | forward | backward | left | right |
|---:|---|---|---|---|
| 1 | 15.8 → **3.7** (−12.1) | 30.8 → **6.6** (−24.2) | 37.7 → **11.8** (−25.9) | 15.6 → 15.5 (−0.1) |
| **2** | 7.2 → **3.6** (−3.6) | 22.2 → **5.5** (−16.7) | 24.3 → **3.4** (−20.9) | 6.4 → 8.2 (+1.8) |
| 3 | 4.9 → **3.8** (−1.1) | 19.1 → **7.2** (−11.9) | 26.7 → **7.2** (−19.5) | 7.2 → 8.6 (+1.4) |
| 4 | 5.5 → **3.9** (−1.6) | 15.6 → **8.9** (−6.7) | 28.0 → **7.3** (−20.7) | 7.3 → 11.2 (+3.9) |
| 6 | 8.2 → **4.0** (−4.2) | 27.5 → **16.1** (−11.4) | 33.8 → **7.4** (−26.4) | 4.8 → 20.3 (+15.5) |

**cfg 2 上四个方向全部进入个位数**（3.6 / 5.5 / 3.4 / 8.2，对着 §4.2 的语料地板
0.4 / 0.1 / 0.6 / 0.9），四向均值 15.0° → **5.2°**。p90 同向收窄
（left cfg2 34.4 → 8.9，backward cfg1 55.2 → 13.1）。
top-1 前后向已是五档全 100%，如 §4.2 第 4 条所料没有 headroom，不要拿它比。

四条结论：

1. **`left` 的系统性欠转彻底消失**：bias 从 −24.2 变成 **+0.6**（cfg 2），
   而且不再随 scale 单调恶化。§4.2 末尾那个混淆项（v3 拿散文 label 训、却喂它关键词）
   **就是 `left` 那 24~34° 的主因**——消除拼法不匹配之后它自己好了。
   **§3.5 的硬输入不需要为 `left` 而上。**
2. **同一个失败签名换到了 `right` 身上，但小一档**：bias
   +5.0 → +8.3 → +11.2 → +20.2（cfg 2→6），`|bias| ≈ mean|e|`，
   是整组一致欠转、采样平均不掉，和 R0 的 `left` 结构相同、左右互换。
   区别是量级：cfg 2 上 8.2°（R0 的 left 是 24.3°），仍在"个位数贴轴"内。
3. **CFG 可用区间从 2~3 收窄到 2**。forward 全程平（3.6~4.0，已到地板附近，
   §3.6 的覆盖也最厚）；backward / left / right 三者都在 **cfg 2 取极小**，
   之后回升，cfg 6 上 backward 16.1°、right 20.3°。
   这**不是** §3.3 的表达力签名（那要求各方向一起早早停止下降 + 伪影增加），
   是各方向已逼近地板之后 CFG 过推的常规退化。**推理默认建议定在 2。**
4. `unclear`（行进速度低于地板、读不出方向）在 LR 轴上 1.6~4.7%，
   都出现在 `right`/`left` 上，与 R0 同量级，不是新问题。

#### 次指标：相位离散度（`phase`）

cfg∈{2,3} 的均值，R0 → R1（括号是同一格的语料 circvar）：

| | forward | backward | left | right |
|---|---|---|---|---|
| run | 0.467 → **0.110** (0.002) | 0.611 → **0.205** (0.000) | 0.506 → 0.876 (0.591) | 0.283 → **0.138** (0.448) |
| walk | 0.264 → **0.142** (0.117) | 0.715 → **0.441** (0.730) | 0.173 → **0.044** (0.552) | 0.191 → **0.113** (0.428) |

8 格里 7 格改善。§4.2 点名要盯的 `run, forward`
从 0.467 掉到 **0.110**（语料 0.002），"方向对了但步态仍在混模态"这一条
大幅收敛 —— 这一栏的功劳大概率在全注意力那一 arm
（[§2.2](temporal_window_full_attention.md) 的直接靶子），不在关键词化。
唯一变差的是 `run, left`（0.506 → 0.876），该格语料本身就散（0.591，n=3），
误差棒宽，不单独下结论。

#### 既有质量指标未退化（§4 末行的那条要求）

`eval_checkpoint.py` 的 17 项电池，v3 / v4 同任务同种子逐项对比
（`outputs/eval_checkpoint/merged_locomotion_v4_fullattn/model000200000/eval_report.html`）：
**16/17 项持平或改善**，均值 0.8868 → **0.8971**，四个分量全部上升
（jerk +0.004、snap +0.012、spectral +0.014、bone_length +0.014）。
两份 `cond.npy` 逐字节相同（MD5 一致），不是 cond 的账。

唯一的回归是 `NewSkeleton/task1`（dragon + `--action_label fly --loop`，域外骨架）：
0.788 → 0.533，`jerk_norm` 0.725 → **0.166**。拆分探针显示是**三者叠加**才炸
（`fly` 不带 `--loop` → jerk 0.995；`walk --loop` → 0.720；`fly --loop` → 0.166），
语料内的 6 个 loop 任务全部正常。**这一项归到全注意力那一 arm 的 §6 判读**，
详见 [temporal_window_full_attention.md §6.1](temporal_window_full_attention.md)。

#### 分流判定（§8 步骤 7）

落在**第一支**，但带一条观察项：cfg 2 上四方向都进了个位数、逼近语料地板 → **收工**；
`right` 带着 R0 `left` 那个"整组一致欠转、随 scale 恶化"的签名，只是小一档。

**暂不启用 §3.5 的硬输入**。它的判据是"仍然系统性欠转**且**随 scale 不降"——
现在 `left` 已完全修复、`right` 在可用区间内是 8.2°，
再为一个词改 index layout 并重训一次，性价比不成立。
若后续人工判读仍能看见 `right` 的偏侧，届时**只给 `right` 一位**
（§3.5 已经预留了"退到只给一个词一位"的写法，只是当时预期的是 `left`）。

**§3.3 的 FiLM 备选方案作废本轮**：曲线形状不是表达力签名（见上面第 3 条）。

---

## 5. 词表小修（随 §2 一起做）

### 5.1 `strafe` 从 `turn` 移出

[`motion_labels.py` `_VOCAB_SURFACE_FORMS["turn"]`](../data_loaders/truebones/truebones_utils/motion_labels.py)
里删掉 `"strafe", "strafes", "strafing"`，把 `strafe` 加进 `ACTION_VOCAB`。
横移是纯平移、不改朝向，和 turn 是两回事。`circle` / `bank` 留在 `turn`
（它们确实改朝向）。

**必须在迁移脚本之前做** —— 否则 27 个横移 clip 会被写成 `…, turn, …`。

### 5.2 `fast` / `shuffle` 加进 `ACTION_VOCAB`

迁移时把来源里的高速动作命名统一为规范 token `fast`，并保留 `shuffle`，
得到 `run, fast` / `walk, shuffle`。解决 §2.2 的塌缩，同时避免近义 token 分裂训练质量。

### 5.3 `glide` / `crawl` 的跨模态误命中

迁移脚本跑完后人工过一遍 swim 开头的 label（只有 KI_Human 系几条）。
关键词化之后散文匹配消失，这条规则不需要长期维护。

---

## 6. 本轮明确不做

| 项 | 内容 | 不做的理由 |
|---|---|---|
| S9 | 把行进方向作为模型条件通道（保留 root XZ，或补 body-local 方向向量） | 大量 clip 本身就是原地动作，没有 root motion 可保留；方向只能来自人工标注（§2.4） |
| S10 | 收紧 `_sample_loop_tile_count` 的平铺上限 | 问题 2 已由 `loop_phase_length` 补传解决（b30fefe），没必要再动数据增强 |
| S11 | `temporal_window ≥ num_frames`，或加全局时间 token | 本轮不做。**后来由 [temporal_window_full_attention.md](temporal_window_full_attention.md) 单独做掉了**：`--temporal_window` 连同整条 mask 通路已删除，时间自注意力现在是全注意力，R1 的 ckpt 已含这一项（d697d5f） |
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
5. 补 §4 的评测（几何自动判定 + 人工盲评兜底 + 相位方差脚本），
   **先跑 R0 基线** —— ✅ 已完成，结果见 §4.2（并据此把主指标换成连续角度误差）
6. 训 **R1**，按 §4 评测 + CFG scale 扫描，逐格对比 §4.2 的角度表
   —— ✅ 已完成（`merged_locomotion_v4_fullattn`），结果见 **§4.3**
7. 按曲线形状分流 —— ✅ **落在第一支，收工**（判定与理由见 §4.3 末）：
   - **角度误差随 scale 正常下降，且四个方向都逼近语料地板 → 收工** ← **R1 在这一支**
     （cfg 2：3.6 / 5.5 / 3.4 / 8.2°，地板 0.4 / 0.1 / 0.6 / 0.9°）
   - `left`（或 L/R）单独不达标 → §3.5 的硬输入兜底 ~~← **R0 目前指向这一支**~~
     —— R0 指向它的那笔账是拼法不匹配造成的，R1 上自己好了；
     `right` 留作观察项，不重训
   - 各方向一起早早停止下降 + 伪影增加 → 启用 §3.3 的 FiLM 备选方案，训 **R2**
     —— 不成立，**本轮作废**
