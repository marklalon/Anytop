# action_group / action_label 重构方案

> ⚠️ **部分已被 [action_label_keyword_refactor.md](action_label_keyword_refactor.md) 取代
> （2026-09-01 实施）**。仍然有效的是 `action_group` 的三分切分、每组一个模型、
> `action_labels.jsonl` 作为唯一真源。**已失效**的是本文关于 `action_label` 的两处描述：
> label 不再是短语描述而是**受控关键词**（`"walk, forward"`），
> `ACTION_VOCAB_CORE` / `ACTION_VOCAB_DETAIL` / `GROUP_MULTIHOT_MASK` /
> multi-hot 通路**已全部删除**，条件只走 frozen-T5。读本文时请按这两条折算。
>
> 状态：**代码改造已完成（§7 步骤 1-6），待重训三组**
> 已产出：三个数据集的 `action_labels.jsonl`（zoo 1106 / zoo_upgrade 265 / unitybundles 2657）；
> 受控词表、`GROUP_MULTIHOT_MASK`、`load_action_labels` 在
> [`motion_labels.py`](../data_loaders/truebones/truebones_utils/motion_labels.py)；
> `action_tags.jsonl` 及其全部代码路径已删除（2026-08-30，见 §9）。
> 待办：**重训三组**（条件维度变了，旧 checkpoint 不兼容且加载时会显式报错）。
> 目标：把原来一个 `action_tags` 字段承担的两份职责拆开 ——
> **`action_group`** 负责训练集切分（分 3 个模型训练），**`action_label`** 负责推理时的
> text-to-motion 条件控制。

---

## 0. 现状与问题

`action_tags.jsonl` 里的 14 个封闭 tag（+`unknown`）目前**同时**干三件事：

| 职责 | 入口 | 现状 |
|---|---|---|
| 训练集切分 | `--action_tags getup,death,fall,rest,jump,turn,gethurt`（[train.bat](../train.bat)） | 手动列出某一组的全部 tag |
| 模型条件 | `--action_tag_cond` → 15 维 multihot → MLP → 加到 timestep token（[anytop.py](../model/anytop.py)） | 粒度只到 14 类 |
| 推理路由 | `resolve_anytop_group()`（[anytop_service.py](../../server/anytop_service.py)） | tag → group 展开表 |

三个问题：

1. **语义粒度不够**：14 个类无法支撑 text-to-motion，用户说不了 "with mouth open"。
2. **切分靠手写 tag 列表**：容易漏、容易和 service 的 `ANYTOP_ACTION_GROUPS` 走偏。
3. **强制互斥分类**：标注时必须在 `idle` / `emote` 之间二选一，而很多 clip 本来就两者都是。

---

## 1. 新的数据契约

`action_tags.jsonl` → **`action_labels.jsonl`**（改名，避免文件名与内容语义不符）：

```json
{"clip": "Alligator_BigMouth_1.npy", "action_group": "stationary", "action_label": "idle, stands still opening and closing its mouth"}
{"clip": "Horse_RunJump_1.npy",      "action_group": "locomotion", "action_label": "run, gallops forward and leaps over an obstacle"}
{"clip": "Bear_Run_1.npy",           "action_group": "locomotion", "action_label": ""}
```

| 字段 | 类型 | 约束 |
|---|---|---|
| `clip` | str | 带 `.npy` 的裸文件名，与现在一致 |
| `action_group` | str | **必填**，封闭三值：`locomotion` / `stationary` / `transition` |
| `action_label` | str | 可为**空串** = 无条件（走 null embedding）。非空时须命中 >=1 个受控词 |

### 1.1 action_group

三组定义与 [anytop_service.py](../../server/anytop_service.py) 现有注释一致，**单值、互斥**：

- `locomotion` —— 持续性位移（含原地跑 / 原地飞：判据是**动作语义**，不是 root 位移）
- `stationary` —— 原地 / 交互行为
- `transition` —— 姿态转换（完整的转换过程）

作用是把分布差异大的动作分开训练，3 组各训一个模型。

### 1.2 action_label

定位是**精简版 prompt**，服务于推理时的 text-to-motion：

- 首要目标：稳定响应 `idle / attack / run / walk / fly / swim / die / ...` 等粗粒度条件；
- 次要目标：尽力响应 `with mouth open` 这类细粒度条件（不保证准确）。

**格式不是硬约束。** 硬约束只有两条：

1. 文本中至少命中一个受控词（§2.1）；
2. 长度 <= 约 15 词。

`<粗粒度动词>, <细节短句>` 只是推荐的写作惯例，不是必须。因为派生 multihot 是**全文正则命中**
（§2.4），动词出现在句子任何位置都算数 —— `"stands still and growls occasionally"` 里句尾的
`growls` 照样命中 `emote`，不需要提到句首。

**多个粗粒度动词也不必并列在前面**写成 `<动词1>, <动词2>, <细节>`。自然语言怎么顺怎么写，
让 multihot 去命中即可；粗粒度串由训练期自动合成（§2.6），不依赖人工书写的前缀。

---

## 2. 关键设计决策

### 2.1 受控词汇表，不是互斥标签集（已确认）

**问题**：封闭词表下有些 clip 无法判断该标 `idle` 还是 `emote`。

**结论：这类歧义不需要解决。**

1. 它**不影响分组** —— `idle` / `emote` / `attack` / `interact` 都映射到 `stationary`，
   选哪个训练集切分完全一样。同组歧义占绝大多数，直接忽略。
2. 它**不影响文本条件** —— label 喂给模型的是 T5 embedding 而不是 one-hot。
   `"idle, stands still and growls occasionally"` 里 idle 和 growl 都在，
   查 `idle` 和查 `growl` 都能召回。**没有"二选一"这个动作。**

因此校验规则是「文本中**至少命中一个**受控词（含同义词）」，命中多个允许且鼓励；
而不是「首词必须属于词表」。

> **不要求必须命中 core 词。** 迁移实测有 27 条 clip（rearing / sniffing / burrowing /
> climbing 等）没有对应的粗粒度动作，硬塞一个 core 词只会把错误的词写进 multihot。
> 这些 clip 的 multihot 全零 —— 那是个**有定义的状态**（映射到 projection 的 bias，
> 与被 CFG 丢弃的 null 态不同），T5 文本通路仍然携带完整语义。
>
> 注意全零现在有**两个**来源：这里的「label 本来就没有 core 词」，以及 §2.4.1 的
> 「命中的 core 词在该组被降级」。两者目前合流到同一个状态，量小可接受，见 §2.4.1 的副作用一节。

词表的角色 = **受控词汇表 / 召回锚点**，只保证一件事：用户输入 `run` 时，
训练集里确实有一批 label 含 `run`。

> **标注原则（不对称）：宁可多写，不要漏写。**
> 多写一个词的代价是轻微语义稀释；漏写的代价是这条 clip 对该查询永久不可召回。

### 2.2 group 由「人工 + AI」标注，不用物理判据（已否决物理判据）

曾考虑用根关节 XZ 净位移自动判 `locomotion`。**不可行**：数据集中大量动作是
**原地跑 / 原地飞，没有 root motion**（同一个坑已在 caption 方向判定里踩过 ——
根速度路线因 in-place clips 被否决，方向改从 clip 名标签取）。

所以 group 走：**LLM 从 caption + clip 名给建议值 -> 人工复核**。

> 可选的**质检信号**（不作为判据，只用来给复核清单排序）：首尾帧姿态距离（去根平移）
> 与 pipeline 已有的 `is_loop` / `loop_full_cycle` 元数据。首尾姿态差大且非闭环的 clip
> 大概率是 `transition`；与标注值冲突的条目优先人工看。这部分不依赖 root motion。

### 2.3 label 由 LLM 压缩 caption 生成（已确认）

zoo 的 [motion_captions.jsonl](../../dataset/truebones_processed/motion_captions.jsonl) 有 1137 条整句
caption，与 `action_tags.jsonl` 的 clip key **一一对应**（去掉 `.npy` 后 overlap = 1137/1137，
两边各 1137 条，无单边条目）。

用已有的 vLLM / Qwen 通道（[server/llm_utils.py](../../server/llm_utils.py)）把 caption 压成
`<动词>, <细节短句>`。纯规则版只能把细节退化成 clip 名，拿不到 "with mouth open" 这类信息，
而那正是次要目标要的。

### 2.4 派生 multihot：要，但设频次门槛（已确认，附"类型过多"问题的解答）

**问题：类型数量过多会不会影响训练？**

分两层看：

- **参数量**：`Linear(V -> 256)`，V 从 15 涨到 40 也只是 4k -> 10k 参数，可忽略。
  **维度本身不是问题。**
- **真正的约束是每个词的样本数**：全库 1402 条 clip。一个词只出现 5 次，模型学不出可靠响应，
  还会挤占容量、制造噪声。经验下限约 **每词 >= 20~25 条**，对应词表上限约 40~50 词。

**实测**（统计 1137 条真实 label 的命中次数，非估计值。2026-08-21 按清理后的 1137 条
重算，与迁移当天的旧数字有出入，见 §8.4）：

| core 词 | 次数 | core 词 | 次数 | core 词 | 次数 |
|---|---|---|---|---|---|
| idle | 298 | die | 93 | shake | 46 |
| attack | 256 | fall | 86 | jump | 41 |
| walk | 138 | bite | 53 | hurt | 30 |
| fly | 123 | getup | 53 | look | 28 |
| turn | 117 | roar | 51 | eat | 22 |
| run | 94 | rest | 47 | swim | 19 |
| scratch | 18 | | | | |

最低频三个是 `scratch` 18 / `swim` 19 / `eat` 22。`eat`、`swim` 虽然接近下限但是用户一定会
输入的**粗粒度模式词**，保留；`scratch` 18 略低于 20，因为是有稳定 T5 语义的独立行为，一并保留，
实际下限记为 **>= 18**。

> **这一层门槛是全库口径，不够用。** 三组各训一个模型，真正决定可学性的是**组内**样本量，
> 见 §2.4.1。

每条 label 命中的 core 词数：0 个 26 条 / 1 个 703 条 / 2 个 317 条 / 3 个 88 条 / 4 个 3 条 ——
多命中是常态，正是 2.1 说的「不用二选一」。

**结论 —— 两层设计：**

| 层 | 内容 | 词数 |
|---|---|---|
| **multihot 硬通路** | `ACTION_VOCAB_CORE`，从 label 文本正则命中自动派生 | **19** |
| **T5 文本软通路** | label 全文，长尾词、细节短语全部保留 | 不限（`ACTION_VOCAB_DETAIL` 另有 30 词） |

长尾词**不进 multihot 但仍在 label 文本里**，语义不丢，只是不给硬通路。
multihot 是**自动派生的，不增加任何标注负担**，且天然支持多命中 ——
`"idle, growls"` 就是 idle + emote 双热，正好对应"无法判断"的真实情况：它本来就是两者。

CFG 时 multihot 与 T5 emb **同时**丢弃（共用一个 drop 掩码），保持 uncond 分支一致。

### 2.4.1 per-group multihot mask（组内门槛，已确认）

§2.4 的 ">= 20 条" 是**全库 1402 条**口径，而 §1.1 规定三组各训一个模型 ——
全局看健康的词切到组内可能只剩个位数。组内实测（**命中 clip 数 / 涉及物种数**）：

| core 词 | locomotion (307) | stationary (600) | transition (230) |
|---|---:|---:|---:|
| idle | – | 279/69 | 19/13 |
| walk | 135/59 | – | 3/3 |
| run | 86/50 | – | 8/5 |
| fly | 48/13 | 46/14 | 29/10 |
| swim | 18/5 | – | 1/1 |
| jump | 13/9 | 5/4 | 23/19 |
| turn | 43/17 | 57/31 | 17/13 |
| attack | 15/6 | 225/61 | 16/13 |
| bite | 7/2 | 43/20 | 3/3 |
| roar | 5/3 | 41/20 | 5/4 |
| eat | – | 22/15 | – |
| die | – | 5/5 | 88/47 |
| fall | – | – | 86/47 |
| hurt | 7/6 | 11/5 | 12/12 |
| getup | – | 1/1 | 52/35 |
| rest | – | 26/15 | 21/16 |
| look | 1/1 | 27/14 | – |
| shake | – | 35/20 | 11/7 |
| scratch | – | 18/14 | – |

**门槛：组内 clip 数 < 10 或 物种数 < 5 -> 该组降级。**

物种维度不是可选项。AnyTop 是跨骨架条件模型，失效模式是「词绑定到骨架」而不是「词学不会」：
locomotion 的 `bite` 7 条**全部来自 Raptor2 + Trex 两个物种**，纯计数门槛拦不住；反过来
locomotion 的 `jump` 13 条覆盖 9 个物种，数量相近但安全得多。locomotion 的 `swim`
18 条 / 5 物种正好卡在线上，保留 —— 它是用户一定会输入的粗粒度模式词。

**降级 = 不进 multihot，只进 T5；不是从词表删除。** 为什么这样能减轻过拟合：

- multihot 槽是 [anytop.py](../model/anytop.py) 那个 `Linear(V -> D)` 里
  **从零训练**的一整列 D 维向量，用 5 条样本去拟合它就是记忆；
- T5 是**冻结的预训练**编码器，`roar` 的表示预训练时就落在 `growl` / `scream` 附近，
  不需要从这 5 条 clip 里学出来；
- T5 读的是整句，稀有词的梯度混在 `run` / `attack` 这些有充分支撑的词里流；
  multihot 槽是孤立无歧义的开关，记忆抓手强得多。

> **能治的和不能治的。** 降级削弱的是「词」的记忆通道，不改变「动作」本身样本少这件事 ——
> locomotion 的 bite 仍然只有 7 条兽脚类样本，用户输入 bite 仍然大概率拿到 Trex 的动作，
> 变的只是绑定的锐利程度。真正的解是补数据或推理端不暴露该词。

> **已冻结（2026-08-30）。** 上表是迁移当天 zoo 1137 条的口径，保留为历史记录。
> 实际提交的 `GROUP_MULTIHOT_MASK` 按同一规则（组内 clip >= 10 **且** 物种 >= 5）在
> **全部 4028 条**（zoo 1106 + zoo_upgrade 265 + unitybundles 2657，
> locomotion 1029 / stationary 2196 / transition 803）上重算，物种维度用
> `motion_metadata.json` 的 `object_type` 而非 clip 名前缀（unitybundles 的
> `FEP_MagmaDemon_...` 前缀是美术包名不是物种，按前缀切会把物种数严重低估）。
> 结果：locomotion 8 槽 / stationary 14 槽 / transition 12 槽，逐词数字见 §9.1。
> 副作用（multihot 全零的 clip）现为 **142/4028 = 3.5%**，仍在可接受区间；
> 若后续扩大，按本节末尾的方案给 multihot 加一位独立的「已降级」指示位。

**实现：保持 19 槽全局布局不变，按 group 置零，不要做三套词表。**
三套 `ACTION_VOCAB_CORE` = 三套索引布局 = 结构不兼容的 checkpoint，且
`action_multihot_words` / `coarse_label_from_words` 及全部消费者都要加 group 参数。
恒零列拿不到梯度，无害。

| group | 有效槽 | 降级（组内不足） | 恒零（组内 0 条） |
|---|---|---|---|
| locomotion | **7** | bite, roar, hurt, look | idle, eat, die, fall, getup, rest, shake, scratch |
| stationary | **12** | jump, die, getup | walk, run, swim, fall |
| transition | **11** | walk, run, swim, bite, roar | eat, look, scratch |

**mask 必须是冻结常量**，不能在 import 时从数据集现算 —— 否则加 clip 会静默改变布局语义。
样本量后续会增加，届时按同一规则重算并显式提交新的 mask。

**副作用：清掉陈旧动作后，降级会让 6 条 clip 的 multihot 变全零**（label 命中的 core 词在本组全被降级）：
stationary 4 条（`Alligator_DieLoop_1` / `Horse_FeetUp_1` / `Horse_InAir_1` /
`Horse_Jumping_1`，只命中 die / getup / jump），transition 2 条（`Horse_RunToStop_1` /
`Scorpion-2_RunToAttrack_1`，只命中 run）；locomotion 零副作用。这 6 条会和 §2.1 那 26 条「没有 core 词」的 clip
合流到同一个「无 tag」状态。目前量小可接受；若数据扩充后这个数变大，需要给 multihot 加一位
独立的「已降级」指示位，而不是让两种语义共用全零。

**降级不解决的两件事**（另行处理）：组内不均衡（stationary 83% 的 clip 命中 idle 或 attack，
locomotion 70% 命中 walk 或 run）；transition 组只有 230 条的绝对量问题。

### 2.5 空 label = 无条件（已确认）

`zoo_upgrade` 的 265 条暂无 caption，label 留空。空 label **必须走 null embedding 分支**，
不能把空串丢给 T5 编码 —— 否则等于教模型「空文本 -> 任意动作」，污染 CFG 的 uncond 分支。

后续给 upgrade 补了 caption 再回填即可。~~也可用 `--backfill-from-clipname` 从 clip 名
生成粗 label 作为过渡~~ **已移除（2026-08-30，见 §9.5）**：clip 名回退整体删除，
`action_labels.jsonl` 与 `species_tags.jsonl` 同契约 —— 必须存在、逐 clip 覆盖，
缺失直接 fast-fail。

### 2.6 训练期粗粒度串增强（由 multihot 反向合成，不切字符串）

训练时以概率 p 把 label 换成**由派生 multihot 反查词表、按词表固定顺序拼成的粗粒度串**，
其余概率用 label 全文：

```
全文版： "stands still and growls occasionally, tail flicking"
合成版： "idle, growl"        <- 由 multihot 反查生成，非人工书写
```

> **注意：不要用"截断 label 前缀"实现。** 那隐含假设粗粒度动词写在句首，而 §1.2 明确不强制
> 这个格式（动词可以散落在句中，也可以有多个），字符串截断没有确定的切点。反向合成没有这个问题。

三个作用：

- 模型同时学会响应粗查询和细查询（对应首要 / 次要两个目标）；
- 多动词天然成立，且顺序由词表固定 —— 不会出现 `"idle, growl"` 与 `"growl, idle"`
  两种写法污染训练分布；
- **训练分布与推理查询分布对齐** —— 前端用户实际打的就是 `idle` / `idle, growl` 这类短查询，
  这个分布模型训练时真见过，而不是只见过完整长句；
- 顺带消除"到底该写多细"的标注压力 —— 写细不会伤害粗粒度响应。

**两条实现约束：**

1. **合成串走的是完整词表，不受 §2.4.1 的 mask 影响。** 合成串是喂给 T5 的**字符串**，
   不是 multihot 向量。被降级的词恰恰最需要出现在短查询训练分布里 —— 用 mask 后的词去合成
   等于让降级词永远学不到短查询响应，与降级的初衷相反。所以合成读的是
   `vocab_words_in(label)` 的结果，不是 masked multihot。
2. **detail-only label 回退到 detail 词。** §2.1 那 26 条没有 core 词的 clip，
   若按「只从 core 词合成」会得到**空串**，而空串按 §2.5 等于 null 条件 ——
   模型会对「唯一入口是 detail 词」的那些动作恰好训练在"无条件"上，
   `sneak` / `rear` / `sniff` / `dig` 这类用户照样会打的短查询就永远学不到。
   `coarse_label_from_words` 在无 core 命中时回退到 detail 词（`'sneak'` / `'rear'`），
   代价为零：它是 T5 字符串，不占任何 multihot 槽位。

### 2.7 推理端：group 由请求显式指定

**不做文本反推路由。** `action_group` 是训练时人工标注的事实，推理时由调用方直接指定 ——
前端本来就有这三个选项，用户选了什么客户端最清楚。请求未带 `action_group` 时直接报错并列出
三个合法值（fail-fast，与仓库其它地方一致），不猜。

> 曾实现过一版从 label 文本反推 group（首个受控词 + 终止事件越位，实测 95.0%；TF-IDF 分类器
> 95.5%）。**已删除**：它在解决一个不该存在的问题 —— 剩余 5% 的错判集中在 `rest`
> （33 条里 transition 18 / stationary 15，接近五五开），而那本质上是**用户没表达的信息**，
> 任何算法都只能赌。为迁就它还一度要往词表里加 `liedown` 这种纯为路由服务的词条。
> 让用户显式指定，这些复杂度整体消失。

label 里的受控词只服务于**条件通路**（派生 multihot + §2.6 的粗粒度串合成），与选哪个模型无关。

#### 2.7.1 group 绑定在 checkpoint 上（2026-08-31）

`--action_group` 是**训练期参数**，而且是**必填的三选一**（`locomotion` / `stationary` /
`transition`）：没有 `all`，也不接受逗号列表 —— 它既切分数据集，又决定模型见过的
multihot mask，所以 group 是**权重的属性**，不是每次请求的选项。传错的后果不是报错
而是静默劣化：点亮一批该 checkpoint 训练时恒为 0 的列。因此四处各加一道闸：

* **训练必填**：[`parser_util.add_data_options(training=True)`](../utils/parser_util.py)
  只在训练解析器里注册 `--action_group`，`required=True` + `choices` 为三个 group。
  「训练在全部 clip 上」这个选项本身被取消 —— 每组训自己的模型是这套设计的前提。
* **训练落盘**：`args.json` 本来就带 `action_group`（`vars(args)` 全量落盘）；
  新增 [`train_anytop.assert_resume_keeps_action_group`](../train/train_anytop.py)
  拒绝「续训时换 group」—— `args.json` 每次启动都会重写，否则续训会一边拿另一组的
  clip 喂旧权重，一边把推理所信任的那条契约悄悄改掉。换组 = 换 `--save_dir`。
* **推理无此参数**：生成侧的解析器**根本不注册** `--action_group`（外部传了直接
  argparse 报错），group 由
  [`parser_util.apply_checkpoint_action_group`](../utils/parser_util.py)
  从 checkpoint 自己的 `args.json` 读出来写进 `args.action_group`。
  要换 group 就换 checkpoint，没有第二条路。
* **服务端配置**：`_assert_checkpoint_declares_group` 在加载时校验每个 group 槽位配到的
  checkpoint 确实是那一组训练的，`PCVG_ANYTOP_MODEL_PATH_*` 配错在启动期就报错，
  而不是让 stationary 的请求被 locomotion 的模型静默接管 —— 路由日志还会跟着一起撒谎。

请求仍然必须显式带 `action_group`（§2.7 不变），但它现在**只用于选 checkpoint**：
`build_anytop_args` 不再把它转成命令行参数下发。

**旧 checkpoint**（`action_group` 为空 / `'all'`，早于本次改动）：无条件生成照常，
但 `--action_label` 会 fail-fast —— 已经没有任何入口能补上那个 mask 了，只能重训。
服务端把这类 checkpoint 装进某个 group 槽位时打 warning，不阻止启动。

> V2P 侧的 `--action_group`（`train_video2pose.py` / `video2pose_dataset.py`）不在此列：
> 那里没有 action 条件通路，group 纯粹是数据集过滤器，`''` = 不过滤仍然合法。
> `resolve_requested_action_group` 因此保持宽松，收紧发生在 AnyTop 的 CLI 上。

这一阶段曾保留可选的查询归一化；后续闭集词表契约已删除 surface-form 翻译。
当前只接受精确 token，来源里的高速动作命名已在标注迁移时统一为 `fast`。

---

## 3. 跨组 clip 的处理

现有多标签数据里，**zoo 有 71 条 clip 跨组**（upgrade 零跨组）：

```
Raptor2_RunJumpBite_1   [locomotion, jump, attack]   三组都沾
BrownBear_Charge_1      [locomotion, attack]
Deer_WalkCall_1         [locomotion, emote]
BrownBear_Limping_1     [locomotion, gethurt]
Horse_RunJump_1         [locomotion, jump]
```

现在这些 clip 会**同时进两个训练集**；改成单值 group 后每条只能进一个。

迁移时的**默认优先级：`transition` > `locomotion` > `stationary`**（含姿态转换事件的优先归
transition —— 该组样本最少、分布最独特，宁可多喂），然后这 71 条**全部进人工复核清单**。

选定优先级后的 group 分布（对比 `locomotion`-first）：

| group | locomotion-first | **transition-first（采用）** | upgrade |
|---|---|---|---|
| stationary | 603 | 603 | 138 |
| locomotion | 317 | **288** | 76 |
| transition | 260 | **289** | 51 |

两套优先级下归属不同的共 **29 条**，且**集中在两类**，复核时优先看：

- **`turn` + 位移（13 条）**：`Jaws_SharkSwimLeft/Right`、`Pteranodon_ArcLeft/Right`、
  `Bird_CircleLand`、`Parrot_CircleFly`、`Pigeon_Circle`、`Scorpion-2_StrafeLeft/Right`、
  `Raptor2_SpinRun` —— 持续位移中的转向，本质是 locomotion 变体，大概率要改回 locomotion。
- **`gethurt` + 位移（6 条）**：`Dog_LimpLoop`、`Spider_Injuredwalk`、`BrownBear_Limping`、
  `Dog-2_WalkLimp`、`Scorpion-2_LimpAlive`、`Spider_RetreatInjured` —— 跛行是持续步态的风格
  变体，同上。

其余（`Raptor2_RunJump*` 系列、`Horse_RunJump`、`Trex_IdleAttackToRun*`）归 transition 是对的。

### 3.1 `rest` / `turn` / `gethurt` 三个 tag 的组语义本身跨组

优先级只能给初值，因为**这三个 tag 各自同时装着「一次性转换」和「持续状态」两种东西**：

| tag | 条数 | 一次性 -> `transition` | 持续 / 风格 -> 其它组 |
|---|---|---|---|
| `rest` | 51 | 躺下去的过程（`Bear_LayDown` / `Camel_LayDown` / `Buffalo_SleepUp`） | 躺着的循环（`Crocodile_SleepLoop` / `Buffalo_GroundLoop` / `Jaguar_Sit`）-> `stationary` |
| `turn` | 47 | 原地 180 度转身（`Bear_Turn180` / `SabreToothTiger_LeftTurn`） | 飞行/游动中的转向（`Jaws_SharkSwim180` / `Pteranodon_ArcLeft`）-> `locomotion` |
| `gethurt` | 42 | 受击反应（`Bear_HitLying`） | 跛行步态（`Dog_LimpLoop` / `Spider_Injuredwalk`）-> `locomotion` |

另有个例：`Monkey_B02Idle_1` 被标 `turn` -> 路由到 transition，但它明显是 idle。

**这三个 tag 的 clip（共 140 条）不做整体硬映射，全部逐条按实际动作判**
（LLM 建议 + 人工确认），一并进复核清单。

---

## 4. 代码改动清单

| 文件 | 改动 |
|---|---|
| [motion_labels.py](../data_loaders/truebones/truebones_utils/motion_labels.py) | `ACTION_TAGS`(15) -> `ACTION_GROUPS`(3) + `CONTROLLED_VOCAB` + `VOCAB_ALIASES` + `MULTIHOT_VOCAB`（频次门槛子集）；**新增冻结常量 `GROUP_MULTIHOT_MASK` + 访问器 `group_multihot_mask(group)`（§2.4.1）**；`load_action_tags` -> `load_action_labels`（校验 group 合法 + label 命中）；`_FALLBACK_ACTION_RULES` 改为 clip 名 -> 粗动词的回退规则（2026-08-30 已删除，见 §9.5） |
| [param_utils.py](../data_loaders/truebones/truebones_utils/param_utils.py) | `ACTION_TAGS_FILE` -> `ACTION_LABELS_FILE = "action_labels.jsonl"` |
| [dataset.py](../data_loaders/truebones/data/dataset.py) | tag 集合求交 -> group 单值相等过滤；`__getitem__` 带出 `action_group` / `action_label` / label emb |
| [tensors.py](../data_loaders/tensors.py) | multihot 拼装改为：[B,512] label emb + [B,V] 派生 multihot + [B] valid mask。**派生后按训练组的 `group_multihot_mask()` 逐元素相乘**（§2.4.1）—— 训练与推理必须用同一个 mask，否则推理时会点亮训练中恒零的槽 |
| [anytop.py](../model/anytop.py) | `action_tag_projection`(15->D) -> `action_label_projection`(512->D) + `action_multihot_projection`(V->D)；加性通路与 `action_tag_null_emb` / CFG 逻辑原样保留；空 label 直接走 null |
| [parser_util.py](../utils/parser_util.py) | `--action_tags` -> `--action_group`（训练过滤，单值）；新增 `--action_label`（推理）；`--action_tag_cond` -> `--action_label_cond`；新增 `--action_label_truncate_prob`（§2.6） |
| [anytop_service.py](../../server/anytop_service.py) | 删除 tag 展开表与 `resolve_anytop_group()`；请求直接带 `action_group`，缺失或非法则报错列出三个合法值 |
| [reference_bank.py](../eval/motion_quality/reference_bank.py) / scorer / `eval_tasks.json` | 过滤键更换。**注意用受控词而非 group 过滤参考先验**，否则先验从「attack 的参考」放宽到「整个 stationary 组」，打分会变松 |
| [train.bat](../train.bat)、[multi_dataset_training.md](./multi_dataset_training.md)、README | 参数与训练契约描述 |

### 4.1 label 的 T5 embedding 怎么进训练

- **离线预计算 sidecar**：label 字符串去重后 mean-pool 成 512 维（<= 1402 条约 3MB），
  dataset 查表即可，**训练进程不需要常驻 T5**。
- **推理端**：service 已常驻 T5 conditioner（三组共享），直接编码用户 prompt 得到同一空间的向量。
- 查表未命中直接报错（fail-fast，与现有 `load_action_tags` 风格一致）。
- 先用 **mean-pool + 加性 token**（与 species FiLM / joint-name 同构，改动最小），
  不要一上来就上 token 序列 + cross-attention。

---

## 5. 迁移流程

**zoo**（迁移当天 1180 条，清理后 1137 条，见 §8.4）
1. `action_group`：按优先级映射给初值（`rest` 除外，逐条判），LLM 读 caption + clip 名给建议值；
2. `action_label`：LLM 压缩 caption 为 `<动词>, <细节短句>`，动词受受控词表约束；
3. 产出 `action_labels.jsonl` + `action_labels_review.jsonl`。

**zoo_upgrade**（265 条）
1. `action_group`：由旧 tag 直接映射（零跨组，无歧义）；
2. `action_label`：留空（~~`--backfill-from-clipname` 可选~~ 回退已删除，见 §9.5）。

**复核清单** `action_labels_review.jsonl` 收录：
- 71 条跨组 clip（其中 29 条对优先级敏感，§3 已列出高风险的两类）；
- 全部 `rest` / `turn` / `gethurt` tag 的 clip（§3.1 的已知雷区，共 140 条）；
- LLM 给的 group 与旧 tag 映射不一致的；
- label 超长 / 未命中任何受控词 / 动词与 group 冲突的；
- （可选）首尾姿态距离与标注 group 相悖的。

---

## 6. 影响与成本

- **必须三组全部重训** —— 条件维度 15 -> 512 (+V)，纯 arch 改动。
- **`cond.npy` 不需要重新生成**，数据集不需要重新预处理，只换 sidecar + 新增 label emb 缓存。
- 旧 checkpoint 的 `args.json` 带 `action_tag_cond`，加载路径需要**明确的弃用报错**，不要静默忽略。
- 71 条跨组 clip 的训练归属会变，`locomotion` / `transition` 两组样本数小幅变动。

分布变化（zoo，迁移当天口径；upgrade 零跨组，138 / 76 / 51 不变）：

| group | 迁移前（跨组 clip 重复计入两组） | 迁移后（单值，transition-first） |
|---|---|---|
| stationary | 603 + 42 | 603 |
| locomotion | 258 + 59 | 288 |
| transition | 248 + 41 | 289 |

迁移后总数从 1251（含重复）降到 1180（每条 clip 恰好进一组）；2026-08-21 清理掉 43 条
源文件已不存在的 clip 后为 1137（locomotion 307 / stationary 600 / transition 230，见 §8.4）。

旧 tag 频次（zoo + upgrade 合计 1402 条）：

```
locomotion 310  attack 306  emote 232  idle 228  death 101  fly 62  rest 50
getup 49  turn 45  gethurt 41  jump 36  interact 36  swim 11  fall 4
```

---

## 7. 执行顺序

1. ~~定受控词表 + 同义词合并表 + multihot 频次门槛~~ **已完成** ->
   `ACTION_VOCAB_CORE`(19) / `ACTION_VOCAB_DETAIL`(30) / `_VOCAB_SURFACE_FORMS` /
   `CORE_WORD_GROUP`（2026-08-30 已删除，见 §9.5）在
   [motion_labels.py](../data_loaders/truebones/truebones_utils/motion_labels.py)，
   配套 `vocab_words_in` / `action_multihot_words` / `coarse_label_from_words`。
2. ~~写迁移脚本，跑 LLM，产出两个 `action_labels.jsonl` + 复核清单~~ **已完成**
   （见 §8）。
3. ~~人工过复核清单~~ **已完成**（复核结果已落回三个数据集的 `action_labels.jsonl`）。
4. ~~冻结 `GROUP_MULTIHOT_MASK`（§2.4.1）~~ **已完成** —— 见 §9.1。
5. ~~代码改造 + 单测~~ **已完成**（§9）。单测覆盖 schema 校验、group 过滤、空 label 走 null、
   multihot 派生 + mask 生效、合成串不受 mask 影响、detail-only 回退非空；
   旧 checkpoint 不做兼容：`args.json` 带 `action_tag_cond` 时直接报错退出。
6. ~~预计算 label embedding sidecar~~ **已完成** ——
   `tools/build_action_label_embeddings.py`，三个数据集各一份 `action_label_embs.npy`。
7. **三组重训** <- 当前卡在这里。

---

## 8. 迁移执行记录（2026-08-20）

产出：

| 文件 | 条数 | 说明 |
|---|---|---|
| `zoo/truebones_processed/action_labels.jsonl` | 1180 -> 1137 | 全部带 label（清理后，见 §8.4） |
| `zoo/truebones_processed/action_labels_review.jsonl` | 224 | 待人工复核 |
| `zoo_upgrade/clean_processed/action_labels.jsonl` | 265 | label 全空 |
| `zoo_upgrade/clean_processed/action_labels_review.jsonl` | 32 | 待人工复核 |

LLM 后的 group 分布（zoo）：stationary 602 / locomotion 302 / transition 276
（规则种子预测是 603 / 288 / 289，LLM 把约 30 条从 transition 挪到了 locomotion，
基本都是 §3 预判的「`turn`/`gethurt` + 位移」那两类 —— 方向与人工预期一致）。

质量核查：退化 label 0 条、未命中受控词 0 条、超长 0 条、schema 校验失败 0 条；
1045/1180 是唯一 label（清理后 1022/1137）。

### 8.1 执行中踩到的三个坑

1. **JSON schema 约束是必需的，不是优化。** 最初裸 prompt 跑 1180 条有 **363 条**解析失败
   （重复 key `"label": "attack": "bite, ..."`、冒号错位、截断）。同一条 clip 顺序重试却
   正常 —— vLLM 批处理让贪心解码在并发下非确定，`temperature=0` 挡不住。改用 OpenAI
   `response_format` / `json_schema`（vLLM 原生支持）后失败归零。为此给
   [`server/llm_utils.call_llm`](../../server/llm_utils.py) 加了通用的 `extra_payload` 参数。
   注意 schema 只约束**形状**不约束**内容**，仍有 7 条吐出 `"label"` 这种占位符，需要单独
   检测（`is_degenerate`）并重跑。

2. **规范词必须出现在自己的 surface form 表里。** `getup` 的表里只写了 `get up` / `gets up` /
   `rise`，漏了 `getup` 本身，于是模型老实按规范词写的 `"getup, lifts head..."` 一个都匹配不上。
   已加断言，任何规范词不自匹配就在 import 时炸。

3. **`in place` / `stationary` 不能做 `idle` 的别名。** 这个数据集绝大多数是原地动画，
   caption/label 里 "while stationary"、"runs in place" 是高频填充语，会把 `idle` 槽误点亮
   一大片。已从别名表移除并注释原因。

4. **规范词的边界坑不止 `getup` 一个。** `gethurt` 同理 —— `hurt` 的词边界匹配被前面的
   `t` 挡住，写成一个词就命中不了。定完词表要拿**所有历史标签字符串**过一遍匹配。

5. **`collapse` 属于 `fall` 不属于 `die`。** 最初把它放在 `die` 下，102 条 `die` 命中里有
   9 条是**只**靠 "collapses" 命中的，而这 9 条全是摔倒 / 躺下（`fall, collapses onto its
   side`），没有一条是死亡 —— 9% 的槽位污染。改挂到 `fall` 后 die 93 / fall 98。
   **教训**：别名表定完要按「只靠这一个别名命中」的条件抽查，否则这类污染完全静默。

### 8.2 已知数据陷阱

zoo 与 zoo_upgrade 有 **15 个同名 clip**（`Bear_Stand_1` / `Horse_Idle_1` / `Ostrich_Run_1` …），
但它们是不同数据集里的不同动作。`motion_captions.jsonl` 按裸文件名索引，所以
**caption 只能 join 给它真正所属的那个数据集**，迁移脚本的 `--captions` 也只传给 zoo。
后续给 upgrade 补 caption 时同样要注意。

### 8.3 受控词表复查（2026-08-20）

按 group 统计 core 词频次时发现一批**假阳性**，修完 zoo 的 1180 条里有 188 条命中集合变化
（只减不增，无 clip 掉到零命中，校验器全量 0 failure）。

**1. 跨词的最长匹配从来没生效（结构性）。** `_VOCAB_MATCHERS` 上方注释写着
「longest-first 让 `"stands still"` 压过 `"stand"`」，但 longest-first 只在**单个词自己的
form 列表内**排序，跨词之间没有任何优先级 —— `"stands still"` 同时点亮 `idle` 和 detail 词
`stand`，`"breathes fire"` 同时点亮 `spit` 和 `breathe`。`vocab_words_in` 改为收集所有
match span，**某词的匹配若严格落在另一个词更长的匹配内部就抑制**。等长匹配双方保留，
`"grazes"` 仍然 `eat + graze`、`"flap"` 仍然 `fly + flap`，core/detail 共现不受影响。
`stand` 227 -> 86。

**2. 三个 form 撤下**（口径同 §8.1 第 5 条：按「只靠这一个 form 命中」抽查）：

| 词 | 撤下 | 证据 |
|---|---|---|
| `look` | `alert` | 18 条只靠它命中的全是姿态形容（`stands alert` / `low alert posture` / `ears alert`），**零真阳性**。48 -> 30 |
| `rest` | 裸 `lie` / `lies` / `lying`（保留 `lie down` 等短语） | 25 条只靠它命中的里约 18 条是 `die, lies motionless`，6 条是 `getup, rises from lying to standing`（离开的状态），无一在休息。72 -> 48 |
| `getup` | 裸 `rise` / `rises` / `rising`，换成 `rises to stand` / `rise back up` 等 | 裸词表示任意向上运动（`rises upward undulating fins` / `rise onto hind legs` / `chest rising and falling`），在游泳、立起、呼吸上误触发多于真正起身。57 -> 50 |

**3. 剩余 4 条改的是 label 不是词表**（词表层面无法区分，且都是描述本身有问题）：

| clip | 改后 |
|---|---|
| `Dog-2_IdleBreathe_1` | `idle, stands still breathing slowly with head level`（原描述里 "chest rising and falling" 误触发 `fall`，且细节无价值） |
| `Crocodile_Bounce_1` | `hurt, die, takes a hit and collapses to the ground`（实为受击死亡，原 caption 的 "bouncing" 描述错误） |
| `Scorpion-2_LimpAlive_1` | `getup, idle, slowly rises to stand with tail swaying`（原写「从 lying 恢复」，描述**离开的状态**必然点亮那个状态的槽；改为描述**去向**） |
| `Camel_Restless_1` | `attack, head lowered, shifts weight and lifts legs`（实为攻击，与 rest / pacing 无关） |

> **教训（§8.1 第 5 条的推广）**：surface form 表定完，除了按「只靠这一别名命中」抽查，
> 还要检查**跨词的包含关系**。另外 label 写作上，「描述离开的那个状态」这个句式
> （`recovery from lying down` / `rises from sleeping`）必然点亮那个状态的槽位，
> 应当改写成描述去向。

`action_labels.jsonl` 现已纳入 git（zoo 与 zoo_upgrade 两份都是），改动直接走 diff 复核。

### 8.4 clip 清理：1180 -> 1137（2026-08-21）

`--overwrite` 全量重跑预处理后 clip 数掉到 1137（重跑前磁盘上实际是 1179）。**不是这次跑失败**：
这 43 条的源动画文件在 raw 目录里已经不存在了。它们是 2026-06-14 / 06-30 从 FBX 源产出的遗留
npy —— 07-01 raw 数据整体转成 GLB（`tools/dataset_cleanup/convert_fbx_2_glb.py` 把 GLB 写在
FBX 旁边，FBX 事后删除）时这批文件没能转出来，而后续几次预处理都走**增量**（按源文件路径去重，
不删旧产物），孤儿 npy 就一直留在 `motions/` 里被计数。`--overwrite` 且无 `--filter` 会整体清空
`motions/` / `bvhs/` 重建，只剩当前 raw 推得出来的 1137 条。

丢失分布（43 条 / 14 个物种）：Dog-2 14、Gazelle 8、PolarBearB 6、Raptor2 4、SabreToothTiger 2，
Buffalo / Camel / Comodoa / Dog / Roach / Skunk / Stego / Tricera / Tyranno 各 1。其中 12 条是
各物种的 `*Fall*`，另有若干是同一动作的第二个源文件（`Gazelle_Run_2` / `PolarBearB_Walk_2` 等）。
43 条里 36 条能在 `TrueboneZ-OO.csv` 官方清单里找到同名 BVH 且帧数吻合 —— 要补回来就按清单重转
缺的源文件放回物种目录，然后**不带** `--overwrite` 走增量。

同步清掉的 clip 键 sidecar（各 -43，清理后均对 1137 条 clip 全覆盖，无缺口）：

| 文件 | 条数 |
|---|---|
| `zoo/truebones_processed/action_tags.jsonl` | 1180 -> 1137 |
| `zoo/truebones_processed/action_labels.jsonl` | 1180 -> 1137 |
| `dataset/truebones_processed/motion_captions.jsonl`（V2P 侧） | 1180 -> 1137 |

本文 §2.3 / §2.4 / §2.4.1 / §4.1 / §6 的统计口径已按 1137 条重算；§5、§6 的分布变化表、
§8 / §8.1 / §8.3 保留迁移当天（2026-08-20）的数字，标注为历史记录。

**两个遗留问题：**

- ~~`preprocess_and_validate.py --rm` 删 clip 时只同步 `action_tags.jsonl`~~
  **已修（2026-08-30）**：`--rm` 现在同步 `action_labels.jsonl`。
  `motion_captions.jsonl` 仍不同步 —— 它在 V2P 侧的 `dataset/truebones_processed/`，
  不在 `--rm` 操作的数据集目录里。
- V2P 侧 `dataset/truebones_processed/` 的 `glb/`(1182) 与 `glb_pose/`(1180) 仍含这 43 条，
  两个数据集目前对不齐。


---

## 9. 代码改造执行记录（2026-08-30）

§7 步骤 3-6 全部完成。`action_tags.jsonl`（及其 15 个封闭 tag、`ACTION_TAGS` 常量、
`load_action_tags` / `infer_action_tags_from_clip_name`、service 的 tag→group 展开表）
**从仓库中删除**，不保留兼容路径。三个数据集的 `action_tags.jsonl` 一并删除。

### 9.1 冻结的 `GROUP_MULTIHOT_MASK`

按 §2.4.1 的规则（组内 clip 数 >= 10 **且** 组内物种数 >= 5）在全部 4028 条 clip
（locomotion 1029 / stationary 2196 / transition 803）上重算。**物种数取
`motion_metadata.json` 的 `object_type`，不是 clip 名前缀** —— unitybundles 的
`FEP_MagmaDemon_Attack01_1.npy` 前缀 `FEP` 是美术包名，按前缀切会把物种数压到个位数。

命中 clip 数 / 物种数（`keep` = 保留槽位，`drop` = 组内不足降级，`—` = 组内 0 条）：

| core 词 | locomotion (1029) | stationary (2196) | transition (803) |
|---|---:|---:|---:|
| idle | — | 874/251 keep | 53/37 keep |
| walk | 396/175 keep | — | 3/3 drop |
| run | 344/193 keep | 7/6 drop | 10/6 keep |
| fly | 218/78 keep | 321/75 keep | 77/34 keep |
| swim | 59/11 keep | 4/3 drop | 5/4 drop |
| jump | 30/27 keep | 12/9 keep | 66/49 keep |
| turn | 110/34 keep | 147/91 keep | 224/103 keep |
| attack | 18/13 keep | 884/233 keep | 31/25 keep |
| bite | 2/1 drop | 91/56 keep | 3/3 drop |
| roar | 2/2 drop | 54/31 keep | 5/4 drop |
| eat | — | 39/29 keep | — |
| die | — | 5/5 drop | 305/211 keep |
| fall | 15/10 keep | 11/9 keep | 311/212 keep |
| hurt | 7/6 drop | 215/164 keep | 16/16 keep |
| getup | — | 4/4 drop | 111/71 keep |
| rest | 1/1 drop | 55/34 keep | 52/32 keep |
| look | 1/1 drop | 63/35 keep | 1/1 drop |
| shake | — | 178/133 keep | 13/9 keep |
| scratch | — | 20/16 keep | — |

有效槽位：**locomotion 8 / stationary 14 / transition 12**（19 槽全局布局不变）。
副作用（masked 后 multihot 全零的 clip）：zoo 26 + zoo_upgrade 13 + unitybundles 103
= **142 / 4028 = 3.5%**。

### 9.2 词表的一处增补

`crouch`（surface forms: crouch/crouches/crouching/squat/squats/squatting/hunker...）
加入 `ACTION_VOCAB_DETAIL`。原因：全库 201 处 label 文本出现 crouch/squat，其中
`KI_Human_Crouch01Start_1` 是**唯一一条**命中不了任何受控词的 label（会被
`load_action_labels` 判为非法）。detail 词不占 multihot 槽位，加它对 checkpoint 布局零影响。
`ACTION_VOCAB_DETAIL` 30 -> 31。

### 9.3 与方案的两处偏离

1. **flag 名 `--action_label_coarse_prob`，不是 §4 表里的 `--action_label_truncate_prob`。**
   §2.6 明确要求「不要用截断 label 前缀实现」，一个叫 `truncate` 的 flag 恰好诱导那种读法。
   实现是 `coarse_label_from_words(vocab_words_in(label))` 反向合成，不切字符串。
2. **`t5_out_dim` 是 768，不是 §4.1 写的 512。** 512 是方案里的示意值；实际维度取
   `cond.npy` 里 `joints_names_embs` 的宽度（`t5-base` = 768），label projection 的
   输入维直接用模型的 `t5_out_dim`，两边永远同源。

### 9.4 落地的接口（与 §4 清单对照）

| 位置 | 结果 |
|---|---|
| `motion_labels.py` | `ACTION_TAGS` 删除；新增 `GROUP_MULTIHOT_MASK` / `group_multihot_mask()` / `CORE_WORD_GROUP` / `action_multihot_vector()` / `normalize_action_group` / `normalize_action_label`；`load_action_tags` -> `load_action_labels`（校验 group 合法 + 非空 label 命中受控词 + <=15 词）；`infer_action_tags_from_clip_name` -> `infer_action_label_from_clip_name`（返回 `(group, label)`；2026-08-30 已删除，见 §9.5） |
| `param_utils.py` | `ACTION_TAGS_FILE` -> `ACTION_LABELS_FILE`；新增 `ACTION_LABEL_EMBEDDINGS_FILE = "action_label_embs.npy"`；`parse_action_tags` -> `parse_action_words` |
| `dataset.py` | tag 求交 -> `filter_motion_names_by_action_group` 单值相等；`resolve_requested_action_group` 拒绝逗号列表（那是 stale 的 `--action_tags` 写法）；`load_action_label_embeddings` + `_resolve_action_label_condition`（§2.6 粗粒度增强 + emb 查表） |
| `tensors.py` | collate 产出 `action_group` / `action_label` / `action_multihot`（**逐行按自己那条的 group mask**）/ `action_label_emb` / `action_label_valid` |
| `anytop.py` | `action_tag_projection`(15->D) -> `action_label_projection`(t5_out_dim->D) **+** `action_multihot_projection`(19->D)，两路相加；共用一个 drop 掩码；空 label 走 `action_label_null_emb` |
| `parser_util.py` | `--action_tags` -> `--action_group`（单值 choices）；`--action_tag_cond` -> `--action_label_cond`；新增 `--action_label`（推理）/ `--action_words`（打分先验）/ `--action_label_coarse_prob`；`args.json` 带 `action_tag_cond` 时 `assert_action_conditioning_not_deprecated` 直接退出 |
| `anytop_service.py` / `serve.py` / `anytop_client.py` | 删除 `ANYTOP_ACTION_GROUPS` 展开表与 `resolve_anytop_group`；请求直接带 `action_group`（+ 可选 `action_label`），缺失或非法即报错列出三个合法值 |
| `reference_bank.py` / `scorer.py` | 打分先验的过滤键改为**受控词**（`action_words`）而非 group；`eval_checkpoint._SCORE_ACTION_TAGS = "locomotion"` -> `_SCORE_ACTION_WORDS = "walk,run"`（`locomotion` 已不是受控词） |
| `eval_tasks.json` | 那里的 `--action_tags locomotion` 走的是**模型条件**通路（不是打分先验），所以译成 `--action_group locomotion --action_label walk`；与旧行为一致，checkpoint 没开对应 flag 时仍然 fail-fast（2026-08-31 起生成侧已无 `--action_group`，该行只剩 `--action_label walk`，见 §2.7.1） |
| V2P 侧（`video2pose_dataset.py` / `train_video2pose.py` / `inference/video2pose.py`） | `--action_tags` -> `--action_group`，共用 `resolve_requested_action_group` |
| `tools/build_action_label_embeddings.py` | **新增**。把 label 全文 + 其合成粗粒度串一起编码进 `action_label_embs.npy`（label 文本为 key）。zoo 1123 串 / zoo_upgrade 230 / unitybundles 1944，均 768 维 |

`action_label_embs.npy` 是**派生产物**（在 `.gitignore` 的 `dataset` 之下），改了
`action_labels.jsonl` 就要重跑；`--action_label_cond` 打开而 sidecar 缺失会直接报错，
不会静默退化成无条件训练。

### 9.5 删除 clip 名回退，`action_labels.jsonl` 改为必须存在（2026-08-30）

`infer_action_label_from_clip_name` 及其规则表（`_FALLBACK_LABEL_RULES` /
`_GETUP_UP_CONTEXT` / `_FALLBACK_DETAIL_GROUP`）、`CORE_WORD_GROUP`（只服务于回退
的种子 group）、`regenerate_dataset_artifacts.py` 的 `_ensure_action_labels_fallback`
自动补写 **全部删除**。理由：

- 回退产物是单动词粗 label，与手写 label 的信息量差距太大，混进
  `action_labels.jsonl` 后难以区分「人写的」和「猜的」；
- 自动补写让 sidecar 悄悄增长，diff 复核（§8.3「改动直接走 diff 复核」）失去意义；
- 缺失条目的正确处置是**停下来补标注**，不是猜一个合法值继续跑。

新契约与 `species_tags.jsonl` 对齐：`action_labels.jsonl` **必须存在**且覆盖
`motions/` 下每个 clip —— `load_action_labels` 缺文件直接 `FileNotFoundError`，
`load_motion_metadata` 缺条目直接 exit；`regenerate_dataset_artifacts.py` 在重算前
显式检查文件存在并在报错里写清补法（不再自动创建文件）。

受影响文件：`motion_labels.py`（删回退段 + `CORE_WORD_GROUP`）、
`tools/regenerate_dataset_artifacts.py`（删 backfill 与 REVIEW 报告，加 fast-fail）、
`tests/test_action_label_fallback.py`（删除）、
`tests/test_multi_token_species_names.py`（删两个依赖回退的测试）。

### 9.6 推理端 CFG：`--action_label_cfg_scale`（2026-09-01）

训练侧的 `--action_label_cfg_drop_prob`（默认 0.2）一直只写了一半的合同 —— 它按样本
硬丢条件（T5 与 multihot **共用一个 drop 掩码**，见 §2.4/`anytop.py`），把权重训出一个
真正的 uncond 模态；但采样时从来没人用过它，`--action_label` 给出的永远是**未放大的**
条件预测。现在补上另一半：

`model/cfg_sampler.py::ClassifierFreeActionModel` 每个扩散步跑两次去噪 ——
一次带 label，一次把 `y['action_label_active']` 全置 False（`_resolve_action_label_active`
里显式掩码优先于训练期的 Bernoulli，`_build_action_label_token` 再把该行路由到
`action_label_null_emb`）—— 然后外推：

    out = out_uncond + s * (out_cond - out_uncond)

AnyTop 预测 x0 而非 eps，所以外推发生在 x0 上，代数完全一样，`s = 1` 退化为原来的条件预测。
**只有 action 条件被引导**：两次 forward 之间除了那个掩码逐位相同，species FiLM /
canonical frame / loop / playspeed / 骨架图在差分里自行抵消，不会被一起放大。

两次 forward 是**顺序**跑的，不是拼成一个 2B batch：`y` 里除张量外还带着逐样本的
python 列表（parents、joint 名、metadata），batch 化要手工复制每一项，而这里两路都要
走完整模型，拼 batch 并不省注意力计算（与 controlnet 时期「uncond 会整段跳过 reference
cross-attn」的情形不同）。

| 位置 | 结果 |
|---|---|
| `model/cfg_sampler.py` | **新增**。`ClassifierFreeActionModel(model, scale)`；属性透传到内层模型（`generate.py` 读 `model.feature_len`，`unwrap_anytop_model` 沿 `.model` 下钻） |
| `parser_util.py` | 新增 `--action_label_cfg_scale`（generate 组，默认 1.0 = 关闭；典型 1.5~3） |
| `sample/generate.py` | `_wrap_action_label_cfg()`：scale=1.0 原样返回模型（保持每步 1 次 forward）；否则要求 (a) 有 `--action_label`，(b) checkpoint 的 `action_label_cfg_drop_prob > 0`（该值是 model 组参数，由 `parse_and_load_from_model` 从 `args.json` 还原，所以读到的是**权重实际的训练值**），(c) scale >= 0，任一不满足直接 exit。单物种与 `--object_type all` 两条采样路径都已接上 |
| `tests/test_action_label_cfg_sampler.py` | **新增** 12 个用例：外推公式（含 scale=0 = 纯 uncond）、uncond 掩码形状/取值、不改调用方的 `y`、非法 scale、属性透传，以及与真实 AnyTop 两次显式 forward 的逐值对齐 |

服务端同步接上（逐请求可调）：

| 位置 | 结果 |
|---|---|
| `server/anytop_service.py` | `build_anytop_args(action_label_cfg_scale=1.0)`：1.0 不上命令行（保持每步 1 次 forward）；否则校验 **有限且 >= 0**、**label 非空**，再在 `generate_args` 之后校验 checkpoint 的 `action_label_cfg_drop_prob > 0`——全部抛 `ValueError` 而不是让 `generate.py` 的 `sys.exit` 冒上来（`SystemExit` 是 `BaseException`，WS handler 的 `except Exception` 接不住）。`collect_gen_kwargs` 同步带上该字段 |
| `server/serve.py` | 请求体新增 `action_label_cfg_scale`（缺省 1.0；显式判 `None`，**不写 `or 1.0`**，否则 0.0 会被悄悄改成 1.0） |
| `client/anytop_client.py` | 新增 `--action-label-cfg-scale`（默认 None = 不发该字段） |

代价是**采样时间翻倍**（每步两次 forward），所以默认关闭。
