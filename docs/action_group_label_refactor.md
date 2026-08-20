# action_group / action_label 重构方案

> 状态：**词表与数据迁移已完成（§7 步骤 1-2），代码改造未开始**
> 已产出：两个数据集的 `action_labels.jsonl` + 复核清单；受控词表已落在
> [`motion_labels.py`](../data_loaders/truebones/truebones_utils/motion_labels.py)。
> 待办：人工过复核清单 -> 代码改造 -> 重训三组。
> 目标：把现在一个 `action_tags` 字段承担的两份职责拆开 ——
> **`action_group`** 负责训练集切分（分 3 个模型训练），**`action_label`** 负责推理时的
> text-to-motion 条件控制。

---

## 0. 现状与问题

`action_tags.jsonl` 里的 14 个封闭 tag（+`unknown`）目前**同时**干三件事：

| 职责 | 入口 | 现状 |
|---|---|---|
| 训练集切分 | `--action_tags getup,death,fall,rest,jump,turn,gethurt`（[train.bat](../train.bat)） | 手动列出某一组的全部 tag |
| 模型条件 | `--action_tag_cond` → 15 维 multihot → MLP → 加到 timestep token（[anytop.py:136-159](../model/anytop.py#L136-L159)） | 粒度只到 14 类 |
| 推理路由 | `resolve_anytop_group()`（[anytop_service.py:62-125](../../server/anytop_service.py#L62-L125)） | tag → group 展开表 |

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

zoo 的 [motion_captions.jsonl](../../dataset/truebones_processed/motion_captions.jsonl) 有 1180 条整句
caption，与 `action_tags.jsonl` 的 clip key **一一对应**（去掉 `.npy` 后 overlap = 1180/1180，
两边各 1180 条，无单边条目）。

用已有的 vLLM / Qwen 通道（[server/llm_utils.py](../../server/llm_utils.py)）把 caption 压成
`<动词>, <细节短句>`。纯规则版只能把细节退化成 clip 名，拿不到 "with mouth open" 这类信息，
而那正是次要目标要的。

### 2.4 派生 multihot：要，但设频次门槛（已确认，附"类型过多"问题的解答）

**问题：类型数量过多会不会影响训练？**

分两层看：

- **参数量**：`Linear(V -> 256)`，V 从 15 涨到 40 也只是 4k -> 10k 参数，可忽略。
  **维度本身不是问题。**
- **真正的约束是每个词的样本数**：全库 1445 条 clip。一个词只出现 5 次，模型学不出可靠响应，
  还会挤占容量、制造噪声。经验下限约 **每词 >= 20~25 条**，对应词表上限约 40~50 词。

**实测**（迁移完成后，统计 1180 条真实 label 的命中次数，非估计值）：

| core 词 | 次数 | core 词 | 次数 | core 词 | 次数 |
|---|---|---|---|---|---|
| idle | 308 | fall | 98 | roar | 52 |
| attack | 257 | die | 93 | shake | 50 |
| walk | 136 | rest | 72 | look | 49 |
| fly | 124 | getup | 57 | jump | 44 |
| turn | 124 | bite | 56 | hurt | 28 |
| run | 116 | swim | 21 | eat | 21 |
| scratch | 18 | | | | |

最低频三个是 `scratch` 18 / `eat` 21 / `swim` 21。`eat`、`swim` 虽然接近下限但是用户一定会
输入的**粗粒度模式词**，保留；`scratch` 18 略低于 20，因为是有稳定 T5 语义的独立行为，一并保留，
实际下限记为 **>= 18**。

每条 label 命中的 core 词数：0 个 27 条 / 1 个 723 条 / 2 个 335 条 / 3 个 89 条 / 4 个 6 条 ——
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

### 2.5 空 label = 无条件（已确认）

`zoo_upgrade` 的 265 条暂无 caption，label 留空。空 label **必须走 null embedding 分支**，
不能把空串丢给 T5 编码 —— 否则等于教模型「空文本 -> 任意动作」，污染 CFG 的 uncond 分支。

后续给 upgrade 补了 caption 再回填即可（也可用 `--backfill-from-clipname` 从 clip 名
生成粗 label 作为过渡）。

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

推理端唯一保留的文本处理是可选的**查询归一化**：把 `sprint` / `gallop` 经 surface form 表
归到 `run`，纯受控词短查询按词表顺序重排，与 §2.6 的合成串对齐。

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
| [motion_labels.py](../data_loaders/truebones/truebones_utils/motion_labels.py) | `ACTION_TAGS`(15) -> `ACTION_GROUPS`(3) + `CONTROLLED_VOCAB` + `VOCAB_ALIASES` + `MULTIHOT_VOCAB`（频次门槛子集）；`load_action_tags` -> `load_action_labels`（校验 group 合法 + label 命中）；`_FALLBACK_ACTION_RULES` 改为 clip 名 -> 粗动词的回退规则 |
| [param_utils.py:53](../data_loaders/truebones/truebones_utils/param_utils.py#L53) | `ACTION_TAGS_FILE` -> `ACTION_LABELS_FILE = "action_labels.jsonl"` |
| [dataset.py:72-95,427-478](../data_loaders/truebones/data/dataset.py#L72-L95) | tag 集合求交 -> group 单值相等过滤；`__getitem__` 带出 `action_group` / `action_label` / label emb |
| [tensors.py:109-113](../data_loaders/tensors.py#L109-L113) | multihot 拼装改为：[B,512] label emb + [B,V] 派生 multihot + [B] valid mask |
| [anytop.py:136-159,368-420](../model/anytop.py#L136-L159) | `action_tag_projection`(15->D) -> `action_label_projection`(512->D) + `action_multihot_projection`(V->D)；加性通路与 `action_tag_null_emb` / CFG 逻辑原样保留；空 label 直接走 null |
| [parser_util.py:145-165](../utils/parser_util.py#L145-L165) | `--action_tags` -> `--action_group`（训练过滤，单值）；新增 `--action_label`（推理）；`--action_tag_cond` -> `--action_label_cond`；新增 `--action_label_truncate_prob`（§2.6） |
| [anytop_service.py:62-125](../../server/anytop_service.py#L62-L125) | 删除 tag 展开表与 `resolve_anytop_group()`；请求直接带 `action_group`，缺失或非法则报错列出三个合法值 |
| [reference_bank.py](../eval/motion_quality/reference_bank.py) / scorer / `eval_tasks.json` | 过滤键更换。**注意用受控词而非 group 过滤参考先验**，否则先验从「attack 的参考」放宽到「整个 stationary 组」，打分会变松 |
| [train.bat](../train.bat)、[multi_dataset_training.md](./multi_dataset_training.md)、README | 参数与训练契约描述 |

### 4.1 label 的 T5 embedding 怎么进训练

- **离线预计算 sidecar**：label 字符串去重后 mean-pool 成 512 维（<= 1445 条约 3MB），
  dataset 查表即可，**训练进程不需要常驻 T5**。
- **推理端**：service 已常驻 T5 conditioner（三组共享），直接编码用户 prompt 得到同一空间的向量。
- 查表未命中直接报错（fail-fast，与现有 `load_action_tags` 风格一致）。
- 先用 **mean-pool + 加性 token**（与 species FiLM / joint-name 同构，改动最小），
  不要一上来就上 token 序列 + cross-attention。

---

## 5. 迁移流程

新增 `tools/migrate_action_tags_to_labels.py`：

**zoo**（1180 条）
1. `action_group`：按优先级映射给初值（`rest` 除外，逐条判），LLM 读 caption + clip 名给建议值；
2. `action_label`：LLM 压缩 caption 为 `<动词>, <细节短句>`，动词受受控词表约束；
3. 产出 `action_labels.jsonl` + `action_labels_review.jsonl`。

**zoo_upgrade**（265 条）
1. `action_group`：由旧 tag 直接映射（零跨组，无歧义）；
2. `action_label`：留空（`--backfill-from-clipname` 可选）。

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

分布变化（zoo；upgrade 零跨组，138 / 76 / 51 不变）：

| group | 迁移前（跨组 clip 重复计入两组） | 迁移后（单值，transition-first） |
|---|---|---|
| stationary | 603 + 42 | 603 |
| locomotion | 258 + 59 | 288 |
| transition | 248 + 41 | 289 |

迁移后总数从 1251（含重复）降到 1180（每条 clip 恰好进一组）。

旧 tag 频次（zoo + upgrade 合计 1445 条）：

```
locomotion 318  attack 311  emote 240  idle 234  death 101  fly 62  rest 51
getup 49  turn 47  gethurt 42  jump 38  interact 36  fall 15  swim 13
```

---

## 7. 执行顺序

1. ~~定受控词表 + 同义词合并表 + multihot 频次门槛~~ **已完成** ->
   `ACTION_VOCAB_CORE`(19) / `ACTION_VOCAB_DETAIL`(30) / `_VOCAB_SURFACE_FORMS` /
   `CORE_WORD_GROUP` 在 [motion_labels.py](../data_loaders/truebones/truebones_utils/motion_labels.py)，
   配套 `vocab_words_in` / `action_multihot_words` / `coarse_label_from_words`。
2. ~~写迁移脚本，跑 LLM，产出两个 `action_labels.jsonl` + 复核清单~~ **已完成**
   （见 §8）。
3. **人工过复核清单** <- 当前卡在这里，zoo 224 条 + upgrade 32 条。
4. 代码改造 + 单测（schema 校验、group 过滤、空 label 走 null、multihot 派生）。
   旧 checkpoint **不做兼容**：`args.json` 带 `action_tag_cond` 时直接报错退出。
5. 预计算 label embedding sidecar；
6. 三组重训。

---

## 8. 迁移执行记录（2026-08-20）

工具：[`tools/migrate_action_tags_to_labels.py`](../../tools/migrate_action_tags_to_labels.py)
（可续跑，`--limit` / `--dry-run` 可先试水）

```bash
# zoo：LLM 从 caption 生成 group + label
python tools/migrate_action_tags_to_labels.py     --dataset Anytop/dataset/truebones/zoo/truebones_processed     --captions dataset/truebones_processed/motion_captions.jsonl --workers 12

# zoo_upgrade：无 caption，规则 group + 空 label
python tools/migrate_action_tags_to_labels.py     --dataset Anytop/dataset/truebones/zoo_upgrade/clean_processed
```

产出：

| 文件 | 条数 | 说明 |
|---|---|---|
| `zoo/truebones_processed/action_labels.jsonl` | 1180 | 全部带 label |
| `zoo/truebones_processed/action_labels_review.jsonl` | 224 | 待人工复核 |
| `zoo_upgrade/clean_processed/action_labels.jsonl` | 265 | label 全空 |
| `zoo_upgrade/clean_processed/action_labels_review.jsonl` | 32 | 待人工复核 |

LLM 后的 group 分布（zoo）：stationary 602 / locomotion 302 / transition 276
（规则种子预测是 603 / 288 / 289，LLM 把约 30 条从 transition 挪到了 locomotion，
基本都是 §3 预判的「`turn`/`gethurt` + 位移」那两类 —— 方向与人工预期一致）。

质量核查：退化 label 0 条、未命中受控词 0 条、超长 0 条、schema 校验失败 0 条；
1045/1180 是唯一 label。

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
