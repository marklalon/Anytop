# action_group 与 action_label

本文描述 AnyTop 中动作条件的**当前实现**：数据里的两个字段、词表与拼写规则、模型如何消费标签、训练与推理时的行为，以及标注工具链。
只写现状；词表规模、槽源秩、指纹、checkpoint 版本等会随代码变动的数值**不在此固化**，以代码为准（见 §10）。

---

## 1. 总览

每个 clip 在 `action_labels.jsonl` 里有两个动作字段，职责完全分开：

| 字段 | 取值 | 作用 |
|---|---|---|
| `action_group` | `locomotion` / `stationary` / `transition` 三选一 | **训练集切分**：决定 clip 属于哪一部分语料。不进入模型条件 |
| `action_label` | 受控词表里的关键词，逗号分隔，可为空 | **text-to-motion 条件**：推理时的 prompt，训练时的监督 |

```
action_labels.jsonl ──► dataset loader ──► word / slot / mask ids ──► AnyTop
   (clip, group, label,        (只发词级 id，               │  按槽聚合冻结词向量 → 3 个通道
    is_loop)                     不预拼向量)                 │  → Linear 投影 → 加到 timestep token
                                                            └─ 可选 AdaLN：再对每层调制
```

关键事实：

- 标签是**受控关键词**，不是自然语言；推理时**不跑 T5**，词向量随 checkpoint 一起存放。
- `action_group` 只影响"用哪部分语料训练"和"服务端选哪个 checkpoint"，**标签本身已经携带 group 信息**（首个主词决定动作类别），所以条件里没有 group token。
- 一个标签词组合只有一种合法拼写；拼写不规范在加载 sidecar 时直接失败。

---

## 2. 数据契约：`action_labels.jsonl`

每个数据集目录下一份，每行一个 JSON：

```json
{"clip": "Horse_RunJump_1", "action_group": "transition", "action_label": "run, forward, jump", "is_loop": false}
{"clip": "Bear_Run_1",      "action_group": "locomotion", "action_label": "", "is_loop": true}
```

| 字段 | 约束 |
|---|---|
| `clip` | 不带 `.npy` 的 clip 名（带了也会被归一化成同一个 key） |
| `action_group` | 必填，三个合法值之一 |
| `action_label` | 可为空（= 无条件）；非空必须通过 §4 的解析与规范拼写校验 |
| `is_loop` | 布尔，预处理**必需**；缺失直接退出。由 `tools/prefill_loop_flags.py` 提议，人工核验 |

行级额外字段（`reviewed`、`pending_delete`、`autofill`）只给审核工具用，`load_action_labels` 不读。例外：预处理会读 `pending_delete`——标了它的 clip 不构建、也不要求 `is_loop`；已在 `motions/` 里的不重建也不删除，真正移除仍由审核页的「清理」完成。

**sidecar 是唯一真源。**

- 必须存在，且覆盖 `motions/` 下每个 clip；缺文件抛 `FileNotFoundError`，缺条目或缺 `is_loop` 直接退出。没有任何"按 clip 名猜一个"的回退。
- `motion_metadata.json` 不保存这些字段；加载时由 sidecar join 进来，写出时剥掉，避免第二份副本与 sidecar 分叉。
- 空 label 走**可学习的 null embedding**，从不把空串交给 T5 编码（否则会污染 CFG 的无条件分支）。

---

## 3. action_group

### 3.1 三个 group 的定义

单值、互斥，按**动作语义**而不是物理量判定：

| group | 含义 |
|---|---|
| `locomotion` | 持续性位移（原地跑 / 原地飞也算：判据是语义，不是 root 位移） |
| `stationary` | 原地 / 交互行为 |
| `transition` | 姿态转换（完整的转换过程） |

不能用 root 位移自动判：大量 clip 是原地动画，且预处理会对 locomotion clip 做 root XZ 去趋势（见 §8.3）。group 由 LLM 建议、人工核验得出。

### 3.2 训练：`--action_group`

`--action_group` 是**训练必填**参数，取值 `locomotion | stationary | transition | all`，不接受列表。

- **`all`**（`train_all.bat` 的做法）：一个模型覆盖整个语料，**不带任何 group 条件**。采样默认对 clip 均匀，因此 group 配比就是语料自身的配比；稀有尾部靠 §6.4 的采样规则补。
- **单组**：只用该组 clip 训练，得到该组专属模型。

值写入 checkpoint 的 `args.json`。续训时不允许改变它（`assert_resume_keeps_action_group`）：`args.json` 每次启动都会重写，改了会让旧权重吃另一组的数据，同时悄悄改掉推理所信任的契约。换组 = 换 `--save_dir`。

### 3.3 推理：group 属于权重

生成侧**没有** `--action_group`。`apply_checkpoint_action_group` 从 checkpoint 的 `args.json` 读出：

- 单组 checkpoint → 该组名；
- `all` checkpoint（或没有记录）→ 空串，所有消费者读作"任意 group"。

空串只会影响**片长先验**（§7.4）：它按 `group|label` 建表，空 group 时跨 group 匹配同一个 label。

### 3.4 服务端

`server/anytop_service.py` 按请求里显式给出的 `action_group` 选 checkpoint 槽位（环境变量 `PCVG_ANYTOP_MODEL_PATH_{LOCOMOTION,STATIONARY,TRANSITION}`）：

- 缺失或非法 → 报错并列出三个合法值，**不做文本反推**（"没表达的信息"任何算法都只能赌）；
- `action_group` 只用来选槽位，不会转发给 `generate`；
- 加载时 `_assert_checkpoint_declares_group` 校验每个槽位配的 checkpoint 确实是该组训的，配错在启动期就报错，而不是让某组请求被另一个模型静默接管。

### 3.5 其他使用 group 的地方

| 位置 | 用法 |
|---|---|
| 同 group 内的头词顺序一致性 | 见 §4.4 |
| 片长先验的 key | `group|label` |
| `--reference_action_group` | 覆盖"参考 clip 是否走 locomotion 的 root 策略"（默认由文件名判断） |
| V2P 的数据过滤 | 纯数据集过滤，没有 action 条件通路 |
| 评测参考先验 | **不用 group**，用头词集合（§9） |

---

## 4. action_label：词表与拼写

### 4.1 词表

词表与全部规则在 [`motion_labels.py`](../data_loaders/truebones/truebones_utils/motion_labels.py)，三类词：

| 类 | 常量 | 说明 |
|---|---|---|
| 主词 | `HEAD_VOCAB` | 标签"关于什么"：一个身体状态（`idle` `run` `rear`…）或一个整段事件（`die` `getup` `land` `draw` `sheathe` `stop`…） |
| 修饰词 | `MODIFIER_VOCAB` | 其余动作词：执行方式（`hover`；`glide` 翅膀不动的滑翔。步态快慢由 `--num_frames` 决定，物种整体的运动风格是 species_tags 的 variant，都不写成修饰词）、叠加的次要动作（`bite` `eat` `look` `shake`）、打击类型（`swat` `slash` `stab`…）、情绪手势、活动/物件、持械（`bow` `gun` `hammer` `shield` `weapon`） |
| 方向词 | `DIRECTION_VOCAB` | `forward backward left right up down` |

要点：

- **一个词只能属于一类**。词序在 `MODIFIER_VOCAB` 里就是规范拼写顺序（分块只管排序，不加权）。
- **方向词一律用裸形**（`left`，不用 `leftward` / `sideways`）。
- **持械只有一个修饰词**：器具本身就是动作时写对应词（`bow` / `gun` / `hammer` / `shield`），其余手持武器写 `weapon` 兜底，两者不同时出现。标签不描述手数，也没有手部槽；非武器道具不改标签。
- `crouch` / `dead` / `sit` / `sleep` / `hover` 是**修饰词**：它们只限定姿态或悬停状态，不是标签"关于"的东西。悬停待机写 `idle, hover`。
- 词表按**准入规则**维护：一个词要留下，必须在固定物种、group、基础动作后，语料里仍有可归因于它的变化。

### 4.2 格式契约（`parse_action_label`）

- 每段必须是词表里的**精确 token**；没有同义词翻译。未知词、重复词、空段全部硬失败。
- 总词数 ≤ `ACTION_LABEL_MAX_WORDS`（8）。
- 非空标签含 **1～2 个主词**（`ACTION_LABEL_MAX_HEADS`）。
- `weapon` 不与具体器具词同现。
- 空标签合法，解析为 `[]`，表示无条件。
- 只写一部分内容是合法的：不写方向 = 方向的边缘分布（见 §6.2）。

### 4.3 规范拼写（`canonical_action_label`）

一个词组合只有一种拼写，否则同一个条件的训练质量会被分散到多个向量上：

1. **主词保持书写顺序，不重排**（首个主词的权重高于后续主词，重排会改变条件）。
2. 方向词紧跟 `turn`（若有），否则紧跟最后一个主词；平面方向在前，竖直方向（`up` / `down`）最后，至多一个竖直词。
3. 其余修饰词按 `CONTROLLED_VOCAB` 顺序。

sidecar 里的 label 必须**已经是**规范拼写，否则加载失败。推理时用户输入若可识别但顺序不对，会被改写成规范拼写并打印提示（只会挪动非主词）。

### 4.4 主词顺序即条件

首个主词权重 1.5、后续主词权重 1.0（§5.2），因此 `attack, jump` 与 `jump, attack` 是**两个不同的条件**。为了让语料对同一类 clip 只做一次"关于什么"的决定：

- **同一 group 内，同一个词集只允许一种主词顺序**（`_validate_head_order_consistency`，加载时校验）；
- 跨 group 可以不同（各 group 是各自的 checkpoint 语料）。

主词顺序**不编码方向**。方向性事件直接写进词表：`draw` / `sheathe` / `stop` / `kneel` / `laydown` / `sitdown`…

---

## 5. 条件表示：槽通道

### 5.1 三个槽，每槽一个通道

契约在 [`action_label_conditioning_contract.py`](../data_loaders/truebones/truebones_utils/action_label_conditioning_contract.py)（无 torch 依赖，数据构建、loader、模型、checkpoint 加载共用同一份定义）。

| 槽 | 成员 | 向量来源 |
|---|---|---|
| head | 标签的**全部**主词（1～2 个） | 冻结 T5 |
| direction | 方向词 | **合成正交码**（不经过 T5） |
| modifier | 其余修饰词（不含任何主词） | 冻结 T5 |

- 三个槽**划分**标签的词：一个词只喂一个通道。缺席的槽是零行，**不重新归一化**其它槽。
- 槽内聚合 = 成员词向量的**加权**均值，再 L2 归一化。
- 因此标签里加修饰词，head / direction 两个通道**逐位不变**（轴保留是恒等式，不是超参）。direction / modifier 槽是集合，词序不进条件。

### 5.2 head 槽权重

`HEAD_SLOT_PRIMARY_WEIGHT = 1.5`：head 槽的首词权重 1.5，其余所有词权重 1.0。L2 归一化后只有比值进入模型。

- r = 1（等权）是**唯一不允许**的取值：此时 `a, b` 与 `b, a` 的 head 通道逐位相同，词序彻底消失；任何 r ≠ 1 都保持两者可分。
- 这是**契约常量而非旗标**：推理必须复现同一个值，指纹负责拒绝用别的值训出的 checkpoint。
- 接受的代价：双主词标签的 head 通道与纯首词不再逐位相等；共享第二主词的标签会互相靠近一些。

### 5.3 方向轴用合成正交码

`DIRECTION_VOCAB` 的词向量是 `SYNTHETIC_CODE_VOCAB` 的**单位正交行**（每个词占一根坐标轴），不是 T5 编码。原因：词表封闭，没有"未见词"需要借邻居；而 T5 在这条轴上的几何是反的——一个词最近的邻居是它的反义词（left/right、forward/backward、up/down），恰是最需要分开的几对。

由此：direction 通道只在前几根轴上非零，投影层只读这个前缀（`slot_channel_widths`），其余列不会成为拿不到梯度的死权重。

### 5.4 词向量表（sidecar）

- `dataset/action_word_embeddings.npy`，由 `tools/build_action_label_embeddings.py` 生成：每个词一个向量，**按词索引**，不按标签字符串。只依赖词表，所以改标签不会使它过期。
- T5 词的编码文本由 `vocab_t5_text` 给出：多数词用自身，少数裸词会落进错误的语义邻域（`shake` 是奶昔、`land` 是地形、`rear` 是后侧），在 `_VOCAB_T5_TEXT` 里改写成能表达动作的短语（`land` → "touching down"）。改写必须保持"展开后的 T5 文本互不相同"。
- 编码设置（池化 / EOS / 后处理 `center_l2`）由契约固定，不是选项；对 T5 编码词做居中，合成码行不参与居中（否则破坏正交性）。
- T5 型号必须与 `cond.npy` 的 `joints_names_embs` 一致，否则维度对不上，建模型时报错。

### 5.5 loader 与模型的分工

- loader **只发词级 id**：`word_ids` / `slot_ids` / `word_mask`（定长），不预拼向量。
- 槽拼装发生在模型侧，用 checkpoint 内的冻结词表（persistent buffer）完成。**推理不读任何 sidecar**。
- 这是有意的：把表示锁进数据通路，以后换表示就得重建数据。

### 5.6 注入模型

```
channels   = 三个槽通道（direction 只取合成码前缀）按槽顺序拼接
action_emb = action_label_projection(channels)             # Linear → GELU → Linear
token      = where(active & valid, action_emb, action_label_null_emb)
timestep_emb += token
```

- `active`：CFG 的硬丢弃掩码（训练时按概率采样，推理时可强制 False）。
- `valid`：该行是否真有标签（空标签 = 无条件）。无标签的行与被丢弃的行走同一条 null 路径，所以推理时不给 `--action_label` 自动落在无条件模态。
- **AdaLN（有 `--action_label_cond` 即有，不是开关）**：另用一个零初始化的头，把 action token 变成**每层** temporal 与 feed-forward 分支输入的逐通道 scale / shift。加性 token 保留；AdaLN 补的是加性路径表达不了的乘性增益——两个标签共享同一槽（向量近乎平行）时，仍能产生不同动作。一个头服务所有层；`--action_adaln_bottleneck` 控制其隐层宽度。
- 有参考动作编码器时，参考生效的行一律旁路 action label（训练与推理相同）；encode 推理与 `--action_label` 互斥。

### 5.7 建模型时的硬门

只对模型**改不回来**的性质设硬门（`model/anytop.py`）：

1. **每槽源满秩**：不同成员集合的归一化和不可能相同，且存在线性读出能判定每个词是否在槽内，覆盖解析器允许的**全部**合法组合（含上限 8 词），不只是语料见过的。
2. **`latent_dim` ≥ 各槽源总秩**：否则投影层在可达条件空间上不可能单射。

各向异性指标（p95 / 最近邻中位 / 有效秩）可被第一层 Linear 重标度，不设门。当前秩值见 `slot_source_rank_report`。

### 5.8 两层指纹

checkpoint 的 load / resume / bind 受两层指纹把关，不兼容的 checkpoint 直接被拒，不会静默读成"质量回退"：

| 指纹 | 描述 | 变化来源 |
|---|---|---|
| `embedding_fingerprint` | 冻结词向量本身（含词表哈希与向量内容哈希） | 改 T5 文本 / 模型 / EOS / 后处理 / 合成码，或词表被换成同形状的另一张 |
| `conditioning_contract_fingerprint` | 运行时语义：词表、parser 版本、槽布局、规范化规则 | 改词表、槽规则、权重、parser 版本 |

此外 `CKPT_VERSION`（`parser_util.py`）覆盖所有会改变训练或推理语义的改动。

---

## 6. 训练期行为

### 6.1 CFG 硬丢弃

`--action_label_cfg_drop_prob`（默认 0.2）：按样本整条丢弃标签，换成 null embedding，训练出真正的无条件模态，推理 CFG 才有东西可引导。

### 6.2 方向槽 dropout

`--direction_slot_drop_prob`（默认 0.15）：按样本只抹掉方向词，其余词不动。

- 目的：让**空方向槽 = 任一方向（边缘分布）**。否则语料里大多数单侧攻击不写方向，模型会把"空槽"学成"这个物种攻击的样子"，裸 `attack, swat` 就会出双臂混合或固定一侧。
- 纯 mask 运算，与 CFG 抽样同写法，不影响 compile；推理永不 drop。
- 与 CFG 丢弃叠加：带方向词的行贡献显式方向监督的概率是 `(1 − cfg_drop) × (1 − p)`；左右可辨性不够就降这个值。
- 标签里至少有一个主词，因此 dropout 不会让标签变空、不会翻转 `valid`。

### 6.3 修饰词槽 dropout

`--modifier_slot_drop_prob`（默认 0.15）：同构，只抹修饰词，让空修饰槽表示"修饰词的边缘分布"（裸 `attack` 抽到某种攻击，而不是语料里少数不带修饰的 clip）。与方向 dropout 逐行独立。

### 6.4 采样：稀有主词补偿

默认对 clip 均匀采样，物种不做均衡。两条规则都以 label 的**首个主词**为键：

- `--rare_head_word_floor N` / `--rare_head_word_max_boost B`：clip 数少于 N 的主词按 N 计，每个 clip 被抽到的倍数为 `N / count`，上限 B。一条规则覆盖整个尾部，新增稀有词无需登记。
- `--head_word_weights word=w,...`：叠加在上面的显式倍数。给 `attack` / `idle` 这类大词加倍数只是在从别的词那里抢曝光。

改这两项会改变训练分布，应开新 run 而不是续训。

### 6.5 is_loop 与 loop 相关增强

- `is_loop` 来自 sidecar（§2），与标签一起进入条件。
- **phase-free** vs **phase-anchored**（`loop_is_phase_free`）：只有每个主词都在 `PHASE_FREE_LOOP_HEADS`（步态、翅拍、呼吸、持续姿态：`walk run swim fly idle crawl climb roll rest fall`）且每个修饰词都在 `PHASE_FREE_LOOP_MODIFIERS`（步态 / 速度 / 保持的姿态 / 手持物）的 loop，才做**循环滚动与平铺**增强。其余 loop 是一个"离开准备姿态再回来"的事件（打击、跳、吼、`idle, look`…），frame 0 就是准备姿态，窗口里正好一次事件，滚动会打乱事件时序。白名单制：新词默认 anchored（少做一点增强，好过打乱时序）。
- `--loop_cond_prob`：loop clip 被**告知**自己是 loop 的概率。失败时仍做物理上的 loop 增强，但窗口按开放 clip 重采样、`is_loop=False` 输入，故意引入标签噪声，防止 `is_loop` 被模型当作内容键。
- loop 相关损失权重（`--lambda_loop_wrap`）见 `train_all.bat`。

---

## 7. 推理

### 7.1 `--action_label`

```
python sample/generate.py ... --action_label "run, forward" --num_frames 40
```

- 只接受受控 token；未知 token 是硬错误并列出合法词（词向量在 checkpoint 里，推理不跑 T5，自由文本无处可去）。
- 可识别但不规范的写法被改写为规范拼写并提示；**主词顺序按用户给的保留**，它就是条件。
- 不写方向 = 方向边缘分布。标签只描述**动作**，不描述角色拿着什么：`attack, slash` 同时覆盖持械与徒手；器具本身就是动作时才写成修饰词。
- 空 = 无条件（null embedding）。checkpoint 必须以 `--action_label_cond` 训练，否则带 label 直接退出。

### 7.2 CFG：`--action_label_cfg_scale`

`ClassifierFreeActionModel`（`model/cfg_sampler.py`）在高噪声半程（`t ≥ T/2`）每步顺序跑两次前向——一次带标签，一次把 `action_label_active` 强制为 False——在 x0 上外推：

```
out = out_uncond + s * (out_cond - out_uncond)
```

- 默认 1.0 = 关闭，每步 1 次前向；>1 时只有高噪声半程做两次前向，细节步（`t < T/2`）scale 固定为 1、单次前向（`CFG_SWITCH` / `CFG_LATE_SCALE`）。
- 只有 action 条件被引导，两次前向之间除这个掩码外逐位相同（species FiLM、canonical frame、loop 等在差分里抵消）。
- 要求：给了 `--action_label`，且 checkpoint 的 `action_label_cfg_drop_prob > 0`（否则没有无条件模态可引导），scale ≥ 0；服务端同样校验（并用 `ValueError` 而非 `sys.exit`）。
- **实践建议**（来自对 checkpoint 的实测，不是超参调优）：语料内的标签用 2；模型没见过的词组合用 1（标签越没被拟合，外推方向越不可靠）；≥ 4 会明显劣化，不建议。

### 7.3 `--loop auto`

默认 `auto`：有 `--reference_motion` 时跟随它自己的 loop 判定；否则有 `--action_label` 时，标签 **phase-free → on，anchored → off**；都没有 → off。显式 `on` / `off` 覆盖。

### 7.4 `--num_frames` 自动取值

省略时按优先级：reference 的原生长度 → 有 `--action_label` 时取"带该标签的训练 clip 长度中位数"（目标物种自己的 clip；没有就取最相似物种的）→ checkpoint 的原生窗口。这样输入模型的 `resample_speed` 条件落在训练分布内。先验表存在 `cond.npy` 的 `clip_length_prior`，key 为 `group|label`；表由 `tools/regenerate_dataset_artifacts.py` 烘焙，缺失时静默退回默认值。

---

## 8. 标注规则与工具链

### 8.1 谁来标

- `action_group`、`action_label`：**LLM 给草稿，人工核验**；`is_loop`：工具提议，人工核验。
- 动作标签**只依据 clip 名与动作内容**，不使用数值型运动特征做标注。
- 已 `reviewed: true` 的行是人看过 GIF 之后签字的真值，所有自动工具默认跳过它（要重判要显式 `--include-reviewed`，用于规则变化导致旧签字过期时）。

### 8.2 方向词标注规则

- **有单一主导侧向或平面朝向 → 必须写方向词；对称 / 交替 / 旋转 / 原地 → 不写。**
- stationary 只补 `left` / `right`（按定义无净位移，不补 `forward` / `backward`）。
- **本身不指向任何方向的动作一律不写平面方向词**（`NO_PLANAR_DIRECTION_WORDS`）：`hurt` `getup` `idle` `rest` `stop` `draw` `sheathe` `bite`。理由：受击和身体状态不是"朝哪发力"；`draw` / `sheathe` 的侧向是刀鞘位置；`bite` 由头部（中心关节）发起，侧向能量读到的是身体转向。匹配**不限于头词**（`idle, right, look` → `idle, look`），已 reviewed 的行也适用。镜像的两条 clip 因此可以拥有**相同**标签（`Trex_BiteLeft` / `Trex_BiteRight` 都是 `attack, bite`）。
- **竖直轴不在豁免内**：`up` / `down` 说的是动作真的向上 / 向下，`idle, up, aim, bow` 保留。
- `hover` 是原地悬停，没有水平朝向要写（`NO_HEADING_WORDS`）。`dive` / `fall` / `up` / `down` 在竖直轴上，不要求平面朝向（`VERTICAL_WORDS`）。
- **jump 的方向**：方向槽有 dropout，所以裸 `jump` 只能表示"任意跳"。原地竖直跳写 `jump, up`；前跳写 `jump, forward`；行进中的跳写 `run, forward, jump` 这类。
- **die 的方向是身体倒向哪边**，一次死亡只写一个词。死在空中（标签带 `hover` / `fall` / `fly`）、起始离地、质心不下降、原地瘫倒的，都不写朝向。

### 8.3 为什么方向判定不能用 root 位移

预处理对每条 locomotion clip 做 root XZ 去趋势，净位移为 0，只剩周期内的 surge / sway；clip 名也靠不住（`MB_Unka_DeathLeft` 实际向右倒）。可用的证据是内容：

- locomotion 朝向：**支撑脚相对 root 的水平速度取反**（身体被钉住后，着地脚必以 −行进速度滑动）。支撑脚用"离该关节自己的地板很近"判定，不能用 contact 通道（contact 恰好把这里要测的滑动筛掉了）。规范帧里 +Z 为前、+X 为左。
- stationary / transition 侧向：把位置拆成镜像对称 / 反对称两半，对称就留空，反对称再比较左右肢速度能量。
- jump：腾空帧的 root 水平位移。
- die：关节质心水平位移（clip 首 1/10 对末 1/10，以体长为单位）。

这些测量只在**方向齐全的人形等物种**上成立（四足和飞龙的 `left` clip 是转弯，不是横移）；每项判据在写入前都先在已标行上标定，达不到门槛只出清单。

### 8.4 工具

| 工具 | 作用 |
|---|---|
| `dataset/review/llm_annotate.py` | 用本地 VLM 对渲染 GIF 出 `action_group` / `action_label` 草稿（`actions`）、物种标签草稿（`species`），`apply` 才写回 sidecar；group 视为冻结输入；镜像对（仅 Left/Right 之差）放在一次调用里回答；头词顺序跟随该 group 内已有拼写 |
| `tools/prefill_loop_flags.py` | 提议 `is_loop`；只填没有该键的行；`reviewed` 行不动 |
| `tools/prefill_direction_words.py` | 只补**空**方向槽，永不改写已有方向词；默认 `--dry-run`，`--apply` 才写；`turn` 行不在范围内（其方向是偏航，不是肢体侧或行进方向） |
| `tools/action_label_sidecar.py` | 行级读写 helper（保留所有 key 与换行风格，tmp + `os.replace`）；`autofill_action_label` 写入时同时置 `reviewed: false` 与 `autofill: true` |
| `dataset/review/serve.py` + `index.html` | 人工核验界面；按 `reviewed` 过滤，自动补标的行带"自动补标"角标；手改标签会清掉该标记，只点 OK 保留 |
| `tools/build_action_label_embeddings.py` | 生成词向量表（§5.4） |
| `tools/count_action_label_words.py` | 按槽统计词频 |

`autofill` 只是一个布尔：表示"这条是工具写的、还没人核验"。具体判据数值只对写它的那次运行有意义，不写进长期数据。

### 8.5 新数据集的顺序

原始 GLB 入目录 → 标注每个 clip 的 group 与 label → `prefill_loop_flags.py` → 人工核验 → `preprocess_and_validate.py`（读 sidecar，从不写它）。完整流程见 `dataset/readme.txt` 与 `anytop-new-dataset` skill。

---

## 9. 评测口径

- **参考先验按主词集合选**，不按 group：`reference_prior_words(label)` 取标签里的 `HEAD_VOCAB` 词（词表序）。方向词与修饰词不参与选取——否则 `walk, forward` 与 `run, forward` 会选出同一批参考。用 group 会把"attack 的参考"放宽成"整个 stationary"，打分变松。
- **用什么 label 生成，就用什么 label 打分**：先验词由生成时的 `--action_label` 派生；`eval_checkpoint` 的 task 带 `--action_label` 时用它，否则必须给 `eval_label` 说明参考动作（两者同时给会报错）。先验词用 `parse_action_label` 解析，打错字当场失败，不会静默缩小参考集。
- **方向遵循用连续角度误差**，并对着"语料地板"读（同一估计器量该物种真实 clip 的误差，通常 1° 以内）：个位数度才算贴轴，两位数就是模型没听指令。top-1（吸附到最近象限）对系统性欠转是结构性盲的，只作粗读；`left` 与 `right` 必须分开报，不能平均成一个 LR 轴。
- **没有对旧表示的消融基线**，所以只做绝对读数，不声称相对任何旧表示的提升。

---

## 10. 改动后需要做什么

| 改动 | 需要 |
|---|---|
| 词表 / 槽规则 / `HEAD_SLOT_PRIMARY_WEIGHT` / parser 版本 | 重建词向量表（`build_action_label_embeddings.py`），**重训**；旧 checkpoint 被两层指纹拒绝 |
| `_VOCAB_T5_TEXT`、T5 型号、池化 / 后处理、合成码 | 重建词向量表，重训 |
| 改 sidecar 里的标签 | 重训；`preprocess_and_validate.py --regenerate-side-artifacts` 刷新依赖标签的侧产物（如 `clip_length_prior`）；不需要重新预处理 motion。词向量表按词索引，**不会**因此过期 |
| 改 `NO_PLANAR_DIRECTION_WORDS` 等标注规则 | 先更新 sidecar，再按上一行处理；预填工具要加 `--include-reviewed` 才会重判旧签字 |
| 只改训练期开关（各 drop 概率、采样规则） | 开新 run，不要续训 |
| 只改预填 / 审核工具行为，不改标签 | 无 |

---

## 11. 关键代码位置

| 内容 | 位置 |
|---|---|
| 词表、解析、规范拼写、sidecar 加载与校验、phase-free 白名单 | [`motion_labels.py`](../data_loaders/truebones/truebones_utils/motion_labels.py) |
| 槽分配、权重、合成码、指纹、秩报告 | [`action_label_conditioning_contract.py`](../data_loaders/truebones/truebones_utils/action_label_conditioning_contract.py) |
| 槽拼装、dropout、null、投影 | [`model/anytop.py`](../model/anytop.py)（`_action_condition` / `_build_action_label_token`） |
| AdaLN 头 | [`model/motion_transformer.py`](../model/motion_transformer.py) |
| 推理 CFG | [`model/cfg_sampler.py`](../model/cfg_sampler.py) |
| 采样权重（稀有词） | [`dataset.py`](../data_loaders/truebones/data/dataset.py)（`rare_head_word_boost` / `parse_head_word_weights`） |
| 标签解析与改写、loop auto | [`sample/conditioning.py`](../sample/conditioning.py)、[`sample/generate.py`](../sample/generate.py) |
| 训练 / 推理的 group 读写 | [`parser_util.py`](../utils/parser_util.py)（`apply_checkpoint_action_group`）、[`train_anytop.py`](../train/train_anytop.py)（`assert_resume_keeps_action_group`） |
| 服务端路由 | [`server/anytop_service.py`](../../server/anytop_service.py) |
| 片长先验 | [`utils/clip_length_prior.py`](../utils/clip_length_prior.py) |
| 评测参考先验 | [`eval/motion_quality/reference_bank.py`](../eval/motion_quality/reference_bank.py) |
| 当前训练配置示例 | [`train_all.bat`](../train_all.bat) |
