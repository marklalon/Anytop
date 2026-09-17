# Action label 词级条件方案

> 状态：**已落地并已训练评测（2026-09-07）。** 数据契约、定稿表示（每个角色槽一个
> 条件通道）、word-keyed sidecar、loader、模型、checkpoint 与测试在 2026-09-06
> 全部切换完成；当前 checkpoint payload 为 v3（§14）。`merged_locomotion_v5_pwp`
> 已训到 200k 步并完成评测，结果见 **§12**。
> **需要重训**：v1 checkpoint 一律拒绝加载。
>
> 两条要先说清楚的：
>
> 1. **没有做消融，本文也不声称任何「相对旧表示的增益」。** §12 全部是绝对读数 ——
>    对着该物种自己真实 clip 的测量地板读，不对着另一个 checkpoint 读。
>    这不是省事：唯一现成的对照 `merged_locomotion_v4_fullattn` 连 `cond.npy`
>    和语料都不同（§12.1），拿它逐格比会得出无法归因的数字。
> 2. **主词顺序不编码方向。** transition 与另外两组走同一套槽池化，head 槽是
>    词的集合；语料里的 A→B 对已改写成单个事件词或「主词 + 修饰主词」（§2.2、§14）。
>    v5 是 `--action_group locomotion` 的 checkpoint，§12 的读数与这条无关。
>
> 推理端默认：语料内标签 `--action_label_cfg_scale 2`，
> 未见过的词组合用 **1**（§12.4）。
>
> **2026-09-11 增补：hands 轴。** `weapon` + `1hand`/`2hand` 换成互斥的
> `hand0`/`hand1`/`hand2`，并占第四个槽通道；§1、§2.1、§5、§6 已按四槽改写，
> 缘由、标注迁移与实测见 **§13**。`CKPT_VERSION` 7→8，**需要重训**；§12 的读数
> 是旧标注上的历史记录。
>
> **2026-09-17 增补：去掉 transition 的方向机制。** 曾经只在 transition 组开启的
> 顺序角色（第二主词过固定带符号置换 `R_B`、`role_ids`、`order_head_mask`）整体删除；
> 主词顺序在任何 group 都不进条件，同一 group 内同一词集只允许一种主词顺序。
> 语料一并改写，词表补 `draw` / `sheathe` / `stop` / `kneel`。parser contract 2→3，
> `CKPT_VERSION` 13，**需要重训**。缘由、改写范围与预检见 **§14**。

## 1. 当前状态

三套数据共有 3745 个动作（2026-09-17）：

| 数据源 | 动作数 |
|---|---:|
| zoo | 938 |
| zoo_upgrade | 246 |
| unitybundles | 2561 |

- `CONTROLLED_VOCAB`：107 个 token，与语料实际用词完全一致；
- 不同 `action_label`：383 个；按 group 展开为 384 个条件点；
- motion、metadata、action label 一一对应；
- 三套数据分别执行 `audit_action_labels.py --action-group all --strict`，R1/R3/R4/R5 均为 0；
- stationary、locomotion、transition 分别训练独立 checkpoint。

切换前的运行路径是：

```text
完整 action_label 字符串
  → 整串 T5 masked mean
  → 每数据源一份 label-keyed action_label_embs.npy
  → dataset 按完整字符串取 768 维向量
  → action_label_projection
  → 加入 timestep embedding
```

现在的运行路径是：

```text
action_label
  → parse_action_label + action_label_slots（唯一实现，loader 与 generate 共用）
  → loader 发 word_ids / slot_ids / word_mask（定长 8）
  → 模型用 checkpoint 内的冻结词表拼四个槽通道（张量镜像 assemble_slot_channels）
  → action_label_projection: Linear(4 * t5_out_dim, latent_dim)
  → 加入 timestep embedding
```

`CKPT_VERSION` 为 13（§14），checkpoint payload 的 `metadata.checkpoint_version` 为 3。
推理不读任何 sidecar：词表是 persistent buffer，随权重一起出入 checkpoint。

## 2. 已完成的数据契约

### 2.1 标签格式

- 标签由受控 token 组成，以逗号分隔；**总词数最多 8 个**，未知词、重复词、空段和第 9 个词均硬失败；
- 每条非空标签必须包含 1～2 个 `HEAD_VOCAB` 主词；
- 主词保持书写顺序（主词在前，修饰它的第二主词在后）；这个顺序**不编码任何语义**，
  同一 group 内同一词集只允许一种顺序（跨文件由预检的键唯一硬门兜底）；
- 方向词紧跟 `turn`，否则紧跟最后一个主词；
- 其他修饰词按 `CONTROLLED_VOCAB` 顺序排列；
- `HANDS_VOCAB`（`hand0`/`hand1`/`hand2`）**最多一个**，多于一个硬失败；它收尾整个词表，
  所以按序排列后天然落在最后（`attack, bow, hand2`）；不写 = 未指定（§13）；
- 空标签表示无条件分支，必须走 `action_label_null_emb`，不能编码空文本。

`HEAD_VOCAB` 只决定一个词进 head 槽，别的什么都不决定。

### 2.2 主词顺序不编码方向

三个 group 走同一套规则：head 槽是主词的**集合**（`assemble_slot_channels` 对本槽词向量取均值），
`land, fly` 与 `fly, land` 是同一个条件。transition 不再有任何特殊分支 —— 没有顺序 gate、
没有角色 id、没有第二主词的变换。一个 A→B 的转换用**一个事件词**写：`draw`（拔出）、
`sheathe`（收回）、`stop`（跑停）、`getup`、`land`、`kneel`、`laydown`……
方向写在词里，不写在词序里。

当前 764 条 transition 的结构为：

| 结构 | 数量 |
|---|---:|
| 单个事件词，如 `die`、`getup`、`draw` | 439 |
| `turn` 家族 | 203 |
| 单主词 + 修饰词 | 93 |
| 双主词（主词 + 修饰主词） | 29 |

最后一行是有意保留的「主词 + 修饰主词」，不是残余的 A→B：`jump, land`（13）、
`land, fly`（6）、`getup, crouch`（3）、`land, run`（2）、`turn, crawl`（2）、`die, fall`（2）、
`draw, crawl`（1）。第二个主词说明第一个主词发生的姿态/语境（从飞行中落地、起身到蹲姿、
在爬行中拔枪），与 stationary 里 `attack, rear`、locomotion 里 `walk, crouch` 的用法完全一致。

同一词集在同一 group 内只允许一种主词顺序（`_validate_head_order_consistency`，现在对
transition 同样生效）；两种顺序既然是同一个条件，两种拼写就只是把同一条件的训练质量
劈成两个字符串。跨 group 允许不同顺序（各 group 是各自的 checkpoint）。

槽分配的唯一实现位于
[`action_label_conditioning_contract.py`](../data_loaders/truebones/truebones_utils/action_label_conditioning_contract.py)，
`action_label_slots(tokens)` 不再接收 `action_group`。

## 3. 要解决的问题

### 3.1 完整字符串条件不利于组合泛化

当前每个完整标签只有一个离线 T5 向量。模型见过 `run, forward` 和 `walk, weapon`，并不能直接复用
其中的词向量组成未见组合。标签变化还会使整份 label-keyed sidecar 失效。

### 3.2 长标签中的控制轴被稀释

在完整字符串或均匀逐词平均中，一个词的相对份额随标签长度下降。给标签增加装备、动作细节后，
基础动作和方向的差异明显减弱。语料里 ≥3 词的 clip 占 651/3802（17.1%）。

### 3.3 transition 方向由词表承担，不由词序承担

逐词求和或平均对词序可交换，所以「`idle, attack` = 拔剑、`attack, idle` = 收剑」这种靠词序
表达的方向在槽池化下天然丢失。曾经为此给 transition 的第二主词加过一个固定带符号置换
（`R_B`），2026-09-17 整体撤销（§14）：方向改为写进词表（`draw` / `sheathe` / `stop` …），
一个只在一个 group 生效的词序机制就没有东西可编码了，而且它让同一个标签字符串在不同
group 的 checkpoint 里意味着不同的条件。

## 4. 已否决的表示：单向量加权平均

```python
label_emb = sum(weight[token] * word_emb[token]) / sum(weight[token])
```

否决的不是某一组权重，而是整个族。令 ρ = 细节词在池化向量里的能量占比：轴保留
≈ 1/(1+ρ) 要求 ρ 小，而「只差一个修饰词的标签要分得开」要求 ρ 大。一个池化向量只有一个 ρ，
两者不可能同时满足；四档权重只是在同一条 frontier 上滑动：

| 候选 | pairwise p95 | 最近邻中位 | 有效秩 | 长轴保留 |
|---|---:|---:|---:|---:|
| 当前整串 T5 | 0.798 | 0.878 | 18.34 | — |
| 中心化 + 均匀权重 | 0.633 | 0.841 | 14.58 | 0.392 |
| 中心化 + medium | 0.751 | 0.901 | 10.71 | 0.703 |
| 中心化 + 强语义权重 | 0.969 | 0.990 | 7.73 | 1.012 |

上表是一次性的历史记录，不是每次预检都重算的东西：否决理由是结构性的（一个池化向量只有一个 ρ），不靠这几个数字的余量支撑。`ACTION_WORD_WEIGHT_PRIOR` 与评测器里那 32 个 `word/...` 候选已于 2026-09-06 一并删除；要复算就从 git 历史取回那一版工具（语料 SHA-256 见预检文档 §1）。

## 5. 定稿表示：每个槽一个条件通道

```text
head       = L2( mean(  本槽词向量 ) )
direction  = L2( mean(  本槽词向量 ) )
modifier   = L2( mean(  本槽词向量 ) )
hands      = L2( 本槽唯一成员的词向量 )        （2026-09-11 起，§13）
缺席的槽   = 零行（present 标记为 False），不重新归一化其他槽
条件输入   = 四个通道按固定顺序拼接
```

槽划分：`head` = `HEAD_VOCAB` 成员，`direction` = `DIRECTION_VOCAB` 成员，
`hands` = `HANDS_VOCAB` 成员（至多一个），`modifier` = 其余全部词。唯一实现是
[`assemble_slot_channels`](../data_loaders/truebones/truebones_utils/action_label_conditioning_contract.py)。

它为什么能同时满足两边：

- **轴保留是恒等式，不是超参**。head / direction 通道的输入只含本槽的词，标签从 2 词长到
  8 词，这两个通道逐位不变（实测最大漂移 0.0）。不存在需要在 0.15 和 1.0 之间权衡的常数。
- **可分性由拼接给出**。各通道各自进入 `action_label_projection` 第一层的一个分块
  （对拼接做一次 Linear 恒等于对每块做一次 Linear 再求和），所以每通道的相对尺度是**可学的**，
  离线不需要、也不应该替模型定一个预算。
- **注入性**。`(group, {word_id})` 在 384 个条件点上唯一，碰撞是构造上不可能。
- **可按 token 审计**。一个词只影响它所在的槽。

槽内均值不会在当前词表上造成信息碰撞。定稿词向量的槽源秩为：head 36/36、
direction 6/6、modifier 62/62、hands 3/3，总秩 107。源向量线性无关意味着不同成员集合的
归一化和不可能相同，并且存在一个线性 readout 能判断每个词是否在槽内；这个证明覆盖解析器允许的
**全部非空槽组合**，包括总词数上限 8，而不是只覆盖语料中见过的组合。

各槽是互不重叠的拼接块，总可达子空间秩为 107，小于默认 `latent_dim=256`，所以第一层 Linear
可以在整个可达空间上保持单射。另做的数值诊断穷举完整 head（36 单词 + 630 无序对 = 666 种）
和 direction（63 种）配置，以及当前语料上限的 ≤3 词 modifier 配置（39773 种）；它用于观察
最近邻和数值间隔，不承担全域正确性证明。K-token 的信息优势因此只剩布局和时间局部化能力，见 §5.1。

### 5.1 为什么不是 K-token

K-token（保留每个词的独立 token）在信息上严格更强，但实测下：

- 槽源满秩保证全部合法组合的成员可由一层线性读出；数值诊断的最小 readout 间隔也为正；
- 学习注意力池化在初始时接近均匀，反而把「轴保留」从恒等式退化成训练目标。

K-token 唯一不可替代的能力是**时间局部化**（让第 40 帧去看第二个端点）。当前条件是加在
timestep embedding 上的单个向量，对所有帧恒定；若将来确实需要逐帧按词条件化，再上 per-layer
cross-attention。为此 loader 的输出保持词级（§7），换表示时只动模型消费端。

## 6. 训练前 geometry gate

只对模型**改不回来**的性质设硬门；p95 / 最近邻中位数 / 有效秩是各向异性指标。四槽完整
可达子空间秩为 107，紧随其后的 256 维 `nn.Linear` 足以在这个子空间上保持单射并重标度，
因此这些指标只报告、不阻断。硬门与实测结果（2026-09-17 去掉方向机制后重跑）：

| 硬门 | 判据 | 定稿表示实测 |
|---|---|---|
| 碰撞 | 余弦 ≥ 0.999999 的不同标签对 = 0 | 0 |
| 最坏近邻 | 不高于 baseline 同项 | 0.9525（baseline 0.9873） |
| 通道漂移 | 追加修饰词后 head/direction 逐元素变化 = 0 | 0.0 |
| 词表秩 | 满仿射秩 | raw 107 / 中心化 106 |
| 槽源秩 | head / direction / modifier / hands 源分别满秩 | 36 / 6 / 62 / 3，全部满秩 |
| 投影宽度 | `latent_dim` ≥ 各槽总可达秩 | 256 ≥ 107 |
| 键唯一 | `(group, {word})` 唯一 | 384/384 |

评测工具为
[`evaluate_action_label_geometry.py`](../tools/evaluate_action_label_geometry.py)，完整指标、
阈值与选择规则见
[`action_label_geometry_preflight.md`](action_label_geometry_preflight.md)。
`--skip-exhaustive` 只跳过最近邻/readout 数值诊断；全合法输入域的注入性仍由槽源秩硬门认证。

## 7. word-keyed sidecar 与运行时装配

sidecar 改为全局 word-keyed 格式：

```text
dataset/action_word_embeddings.npy
  schema_version = 3
  keying = "word"
  ordered_vocab = CONTROLLED_VOCAB
  embeddings = float32[107, 768]     # eos=keep, masked mean, center_l2
  embedding_contract                 # 含 word_table_sha256
  embedding_fingerprint
```

装配规则：

1. 训练入口 bootstrap 只读取和验证一次 sidecar；
2. 同一个不可变 conditioning bundle 同时传给 loader 和 model；
3. **loader 只输出 `word_ids`、`slot_ids`、`word_mask`**，不输出拼好的向量。
   槽拼装在模型侧，用 checkpoint 内的冻结词表完成；
4. model 将冻结词向量保存为 persistent buffer，并用张量镜像
   `assemble_slot_channels` 的语义（同一套 `slot_ids`，不另写一份规则）；
5. 推理只读取 checkpoint 内的词向量，不从数据目录重新选择 sidecar。

第 3 条是有意的：离线预拼 `label → bundle` 查表会把表示锁进数据通路，以后换表示要重建数据；
发词级 id 则只需改模型消费端，`embedding_fingerprint` 不受影响。

模型侧的改动很小：

- `nn.Linear(t5_out_dim, latent_dim)` → `nn.Linear(4 * t5_out_dim, latent_dim)`；
- 构模时校验实际 `latent_dim` 不小于 sidecar/预检记录的槽源总秩（当前 107）；
- 缺席槽为零行，且在场的槽恒为单位范数，所以 `slot_mask` 对模型是冗余信息，只用于断言；
- CFG、`action_label_null_emb`、`action_label_valid`、`cfg_sampler`、生成入口都不变 ——
  null 仍然整束替换。

## 8. 指纹与 checkpoint 契约

### 8.1 两层指纹

`embedding_fingerprint` 只描述冻结词向量：

- 有序 token→T5 文本表；
- T5 名称和模型材料 hash；
- tokenizer 类与版本；
- pooling、EOS 策略和向量后处理（定稿：masked mean / keep / center_l2）；
- embedding shape、dtype 和 sidecar schema；
- **词表本身的 `word_table_sha256`**（`<f4` 连续字节的 SHA-256）。

最后一条是 schema 3 才加的。在此之前上面的字段全部只描述"这张表应该怎么造出来"，
没有一条会随向量改变，因此把 `embeddings` 换成同形状的另一张表，
`embedding_fingerprint` 分毫不动 —— sidecar 加载、resume 比对、checkpoint bind
三道基于该指纹的闸门会一起放行。现在 `build_action_conditioning_bundle` 和
`validate_loaded_action_conditioning` 都会把实际向量重新哈希后与该字段核对。

`conditioning_contract_fingerprint` 描述运行时语义：

- `embedding_fingerprint`；
- 有序词表、`HEAD_VOCAB`、parser/canonicalization 版本；
- slot 字段、槽名、最大词数、最大主词数；
- 最终表示布局（`slot_channel_representation()`）及其全部参数。

修改槽规则不会使纯 embedding sidecar 失效；修改 T5 文本、EOS 或向量后处理会使两层都失效。
当前定稿值见预检文档 §1。

### 8.2 checkpoint v3

不用 `get_extra_state()`。元数据放在 checkpoint 顶层，避免与当前 EMA buffer 同步逻辑冲突：

```python
{
    "model": model_state_dict,
    "model_avg": ema_state_dict_or_none,
    "metadata": {
        "checkpoint_version": 3,
        "action_conditioning": {
            "embedding_contract": {...},
            "embedding_fingerprint": "...",
            "conditioning_contract": {...},
            "conditioning_contract_fingerprint": "...",
        },
    },
}
```

加载分两步，两步查的不是同一批材料：

1. **先纯 metadata**：`validate_action_conditioning_metadata` /
   `assert_bundle_matches_metadata`，在任何权重落地之前拒掉跨词表、跨契约的 resume；
2. **再逐份 bind**：`load_model` 之后才调用 `bind_checkpoint_action_conditioning`，
   因为它认证的是 buffer 里的词表，而那个 buffer 是 persistent 的，
   只有在 `load_state_dict` 覆盖之后才是 checkpoint 自己的值。`model_avg` 有独立的
   buffer，也要单独 bind。

顺序颠倒过来（先 bind 再 load）认证的是本 run 启动时的材料，随后被 checkpoint 静默覆盖。
一个训练 run 内两层 fingerprint 必须固定；跨词表或跨契约 resume 直接拒绝，
除非使用单独的显式迁移工具。

## 9. 实施顺序

1. ~~构建全局 word-keyed sidecar~~ **已完成**。`build_action_label_embeddings.py` 改成词表
   模式，直接调用几何预检的编码器 helper（`_encode_both_eos_policies` / `_postprocess_atoms`），
   所以出厂的向量就是被评测过的那批。实测复现出预检文档 §1 的两个指纹：
   `0f0a698c…` / `47314397…`（schema 2 时为 `2e017b7a…` / `dfd6ac0e…`）；
2. ~~训练入口装配统一 bundle~~ **已完成**：`train_anytop.bootstrap_action_conditioning`
   读一次、校验一次，同一个 `ActionConditioningBundle` 同时进 loader 和模型；
3. ~~dataset/collate 输出词级 id 与 mask~~ **已完成**。padding 到契约上限 8 而不是 batch
   最长（定长张量，不会因 batch 触发重编译），`slot_ids` 的 padding 值是 `SLOT_PAD_ID = -1`；
4. ~~模型侧镜像槽拼装~~ **已完成**：`AnyTop._assemble_action_slot_channels`，与 numpy 契约
   逐位相等（float64 下 8.3e-17）；第一层 Linear 宽度为 `4 * t5_out_dim`；
5. ~~checkpoint v3 和两层指纹校验~~ **已完成**：`build_checkpoint_payload` /
   `load_checkpoint_weights` / `bind_checkpoint_action_conditioning`；
6. ~~单元测试、EMA/resume round-trip 和跨契约拒绝测试~~ **已完成**：
   [`test_action_label_word_conditioning.py`](../tests/test_action_label_word_conditioning.py)；
7. **改做了别的（2026-09-07）**：原计划是「固定 held-out 切分后训练消融」，
   即再训一版旧表示当对照。**消融没有做，也不会补做** —— 理由见 §12.1：
   唯一现成的对照 `merged_locomotion_v4_fullattn` 的 `cond.npy` 与语料都已不同，
   本来就不是对照组；而重训一版匹配基线的代价没人愿意付。
   替代品是**绝对读数**：held-out `(物种, 标签)` 组合直接对着该物种自己的语料地板量，
   见 §12.5。这换不来「相对旧表示提升了多少」，只能回答「够不够用」；
8. ~~重标定 `action_label_cfg_scale`~~ **已完成（2026-09-07）**：定在 **2**，
   语料内标签；held-out 组合用 **1**。判据与曲线见 §12.4。

## 10. 验收

实现测试至少覆盖：

- 主词顺序在任何 group 都不进条件（`land, fly` 与 `fly, land` 通道逐位相等），
  同一 group 内两种主词顺序在加载时硬失败，跨 group 允许；
- loader 只发 `word_ids` / `slot_ids` / `word_mask` 三个字段，契约里没有任何 role 字段；
- action label 总词数恰好 8 个时接受，第 9 个词硬失败；
- head/direction 通道不随修饰词数量变化（等式断言，不是阈值）；
- 缺席槽为零行且不改变其他槽；
- padding 与所有 mask 完全生效；
- loader/model 使用同一 ordered vocab、同一 `slot_ids` 与同一 fingerprint；
- 实际训练 `latent_dim` 小于槽源总秩时启动失败；
- 普通模型、EMA、保存、加载和 resume 往返一致；
- 外部 sidecar、代码词表或 checkpoint metadata 不一致时硬失败；
- 推理不依赖外部 sidecar。

训练后以生成动作验收（**逐条结论见 §12.8**，这里只留原始条目）：

- held-out `(species, action)` 组合不低于当前 baseline；
- mode、direction 和长标签控制能力不退化；
- 原有动作质量指标不退化。

第 1 条的比较对象不存在，所以它按绝对读数改判 —— §12.8 写明是"改判"，不要当成通过。

## 11. 当前已落地文件

- 数据词表、parser 和 canonicalization：
  [`motion_labels.py`](../data_loaders/truebones/truebones_utils/motion_labels.py)
- 槽划分、槽拼装、槽源秩证书、bundle 与双指纹契约：
  [`action_label_conditioning_contract.py`](../data_loaders/truebones/truebones_utils/action_label_conditioning_contract.py)
- 几何预检（硬门 + 穷举扫描）：
  [`evaluate_action_label_geometry.py`](../tools/evaluate_action_label_geometry.py)。
  `_slot_source_rank_report` 现在是契约模块 `slot_source_rank_report` 的薄包装，
  因为建模时要跑同一份证书
- 词表 sidecar 构建：[`build_action_label_embeddings.py`](../tools/build_action_label_embeddings.py)
  → `dataset/action_word_embeddings.npy`（全局一份，词索引；改标签不会让它过期）
- loader：[`dataset.py`](../data_loaders/truebones/data/dataset.py)
  （`load_action_conditioning` / `_apply_action_label_condition`）
  与 [`tensors.py`](../data_loaders/tensors.py)（`_build_action_slot_batch`）
- 模型：[`anytop.py`](../model/anytop.py)（`_init_action_conditioning`、
  `_assemble_action_slot_channels`、`validate_loaded_action_conditioning`）
- checkpoint 契约：[`model_util.py`](../utils/model_util.py)
  （`build_checkpoint_payload` / `load_checkpoint_weights` /
  `bind_checkpoint_action_conditioning`），写入在
  [`training_loop.py`](../train/training_loop.py)，装配在
  [`train_anytop.py`](../train/train_anytop.py)
- 推理：[`generate.py`](../sample/generate.py)（`_resolve_action_condition` 只发词级 id，
  不再跑 T5、不再读 sidecar）
- 测试：[`test_action_label_conditioning_contract.py`](../tests/test_action_label_conditioning_contract.py)（纯契约）、
  [`test_action_label_word_conditioning.py`](../tests/test_action_label_word_conditioning.py)（端到端）、
  [`action_label_test_utils.py`](../tests/action_label_test_utils.py)（共用夹具）

---

## 12. 训练后评测：`merged_locomotion_v5_pwp`（2026-09-07）

**被测对象**：`save/merged_locomotion_v5_pwp/model000200000.pt`，200k 步，取 EMA 权重，
`--action_group locomotion`、`--action_label_cfg_drop_prob 0.2`、`latent_dim 256`。
checkpoint 自带的元数据与预检文档 §1 逐字相同：`checkpoint_version 2`、
`embedding_contract` schema 3、`word_table_sha256 e62b012e…`、
`embedding_fingerprint 0f0a698c…`、`conditioning_contract_fingerprint 47314397…`。
**训练出来的确实是本文描述的那份契约**，这一条不靠信任，靠比对。

原始输出全部在 `outputs/direction_following/PWP*`。

### 12.1 为什么没有基线，以及这让哪些结论失效

`merged_locomotion_v4_fullattn` 是唯一现成的候选对照，**但它不是对照组**：

| | v4_fullattn | v5_pwp |
|---|---|---|
| `cond.npy` MD5 | `b686e0f6…`（40941972 B） | `dfb916a7…`（40881606 B） |
| `action_labels.jsonl`（zoo / unitybundles） | `B823237D…` / `982DE2BE…` | `C2EA57A8…` / `A498D525…` |
| 条件表示 | 整串 T5 | 每角色槽一通道 |

两次训练之间除了本文这项改动，还夹着 prop socket 关节删除、scale 归一化两处修正、
预处理 1:1 改名、以及 R1–R6 locomotion 重标 —— 骨架条件和语料**都换过**。
逐格比 v4 得到的差值无法归因到任何一项。

所以本节全部改成**绝对读数**，参照系是同一个估计器量**该物种自己真实 clip** 的结果
（"语料地板"）。这个换法有代价，说清楚：

- **能回答**：控制力够不够用、长标签会不会稀释方向轴、未见组合能不能走、
  质量有没有塌、CFG 该定多少；
- **不能回答**：相对整串 T5 表示提升了多少。**本文任何地方都不声称这个数。**
  §3 那三条动机仍然只有离线几何证据，不因 §12 而升级成生成侧的因果证据。

### 12.2 方向遵循：绝对角度误差

方向遵循 sweep，KI_Human × {`walk`, `run`} × 四方向 ×
cfg{1, 1.5, 2, 2.5, 3, 4, 6} × 16 条 = **896 clip**，seed 10，DDPM 100 步。
角度误差自动校准在 KI_Human 上两轴各 100%（FB 7/7、LR 10/10），地板 0.6°。

`mean|e|`，度，括号是同格 `bias`（正 = 偏向角色左侧）。语料地板一行是同一个估计器
量 KI_Human 真实 clip 的结果：

| cfg | forward | backward | left | right | 四向均值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.1 (−0.7) | 1.1 (−0.1) | 3.9 (−1.7) | 8.3 (+8.3) | 3.60 |
| 1.5 | 1.1 (−0.9) | 1.0 (−0.8) | 1.9 (−0.1) | 6.1 (+6.1) | 2.53 |
| **2** | 1.2 (−1.0) | 1.1 (−1.1) | **1.7** (+0.6) | **4.9** (+4.8) | **2.23** |
| 2.5 | 1.3 (−1.3) | 1.2 (−1.0) | 2.0 (+1.3) | 4.8 (+3.9) | 2.33 |
| 3 | 1.4 (−1.4) | 1.5 (−0.8) | 2.5 (+2.0) | 5.1 (+3.2) | 2.63 |
| 4 | 1.8 (−1.8) | 2.5 (+0.6) | 3.8 (+3.8) | 7.4 (+1.1) | 3.88 |
| 6 | 2.2 (−2.2) | 29.9 (+28.8) | 7.0 (+7.0) | 14.0 (−5.1) | 13.28 |
| **语料地板** | **0.4** (n=4) | **0.1** (n=3) | **0.6** (n=5) | **0.9** (n=5) | 0.5 |

四条读数：

1. **前后向已经贴地板，而且不需要 CFG**。cfg 1 上 forward 1.1°、backward 1.1°，
   地板是 0.4° / 0.1°。加 CFG 在这两栏**只会变差**（cfg 4 起 backward 崩）。
2. **左右向需要 CFG，收益在 cfg 2 用完**。left 3.9 → 1.7、right 8.3 → 4.9。
3. **`right` 是唯一站得住的残留问题，而且 CFG 治不了它**，见 §12.3。
4. cfg 6 是明确的破坏区：backward 29.9°、p90 75.4°，top-1 在 FB 轴掉到 82.8%
   （其余档位四方向 top-1 全部 100%，`mixed` / `unclear` 全 0）。
   top-1 在这里再次证明没有分辨率：cfg 1~4 全是 100%，而实际误差差了 1.7 倍。

### 12.3 `right` 的残留：账在 `run, right` 一个标签上

按标签拆开（`compose` 子命令，`score` 会把 walk/run 并进同一个 `right` 格）：

| cfg | `walk, right` | `run, right` |
|---:|---:|---:|
| 1 | 7.4 (+7.4) | 9.2 (+9.2) |
| 1.5 | 3.6 (+3.5) | 8.6 (+8.6) |
| 2 | **1.6** (+1.4) | 8.1 (+8.1) |
| 2.5 | 1.7 (−0.1) | 8.0 (+8.0) |
| 3 | 2.1 (−1.6) | 8.0 (+8.0) |
| 4 | 6.3 (−6.3) | 8.5 (+8.5) |
| 6 | 19.0 (−19.0) | 8.9 (+8.9) |

`walk, right` 被 CFG 修好了（7.4 → 1.6，地板 0.9）。**`run, right` 在七个档位上是一条
直线**：8.0~9.2°，`|bias| ≈ mean|e|`，整组一致欠转、采样平均不掉、加 scale 不动。

这正好命中
[action_label_keyword_refactor.md §3.5](action_label_keyword_refactor.md) 留的启用判据
——「仍然系统性欠转**且**随 scale 不降」。R1 当时把 `right` 记成 8.2° 的观察项并缓办，
理由是"在可用区间内是个位数"；现在按标签拆开可以看到那 8° 不是 walk/run 共同的偏置，
而是 `run, right` 一个标签独占的。**是否为它上 §3.5 的硬输入位仍是待决项**，本文不替
这个决定下结论：量级仍是个位数，而代价是改 index layout 加一次重训。

### 12.4 `action_label_cfg_scale` 重标定

R1 那轮的 cfg 上限是人眼判的（"cfg>3 动作明显变形"）。这轮两边都量：同一批 896 条
clip 用 `eval/motion_quality` 的评分器打分（与 `eval_checkpoint.py` 同一个评分器、
同一个参考先验 `walk,run`）。

| cfg | 四向角度均值 (°) | 质量中位数 | bone_length | jerk | snap | spectral |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.60 | **0.6725** | 0.906 | 0.583 | 0.918 | 0.334 |
| 1.5 | 2.53 | 0.6433 | 0.892 | 0.548 | 0.905 | 0.327 |
| **2** | **2.23** | 0.6465 | 0.887 | 0.526 | 0.896 | 0.314 |
| 2.5 | 2.33 | 0.6360 | 0.886 | 0.495 | 0.881 | 0.296 |
| 3 | 2.63 | 0.6089 | 0.873 | 0.477 | 0.864 | 0.285 |
| 4 | 3.88 | 0.5904 | 0.828 | 0.440 | 0.836 | 0.286 |
| 6 | 13.28 | 0.5283 | 0.757 | 0.351 | 0.753 | 0.243 |

这一列质量分只在本表内部横向比较有意义：它是 60 帧 KI_Human 片段对 `walk,run`
先验的分数，和 §12.6 那套 17 项电池的 0.89 不是同一个量纲，别混着读。

**质量随 cfg 单调下降，角度误差在 cfg 2 取极小**，所以判据是干净的：
在角度误差不劣于最优的那些档里取最小的 cfg。

- **cfg 2 = 语料内标签的默认值**。角度最优（2.23°），质量比 cfg 1 只掉 3.9%；
  cfg 1.5 质量一样但角度差 0.3°，被 cfg 2 支配；cfg 2.5 以上两边都更差。
- **cfg 1 在只要前后向时是更好的选择**：forward/backward 在 cfg 1 上就已经是
  1.1° / 1.1°，加 CFG 反而变差，还省一半采样时间（CFG 每步跑两次 forward）。
- **held-out 组合用 cfg 1**，见 §12.5 —— 未见过的标签在 cfg 上升时一致变差。
- **cfg ≥ 4 不要用**：backward 在 cfg 4 开始崩，cfg 6 是 29.9°。

与 R1 的口径差别值得记一笔：R1 从 cfg 1 到 cfg 2 换来 15.0° → 5.2°，
这轮只有 3.60° → 2.23°。**CFG 的边际作用小了很多**，因为 cfg 1 的起点已经好得多
（R1 cfg 1 是 3.7/6.6/11.8/15.5，这轮是 1.1/1.1/3.9/8.3）。这句只是记录两次实测的
数值，不构成归因 —— 中间隔着 §12.1 那一堆改动。

代码里的默认值**保持 1.0 不变**：`--action_label_cfg_scale` 需要
`--action_label` 且要求 checkpoint 训练时 `action_label_cfg_drop_prob > 0`，
把默认值改成 2 会让没有 CFG 训练的 checkpoint 直接报错。
`utils/parser_util.py` 的帮助文本已改成记这里的实测值。

### 12.5 长标签与 held-out 组合

#### 标签长度阶梯 —— §3.2 的靶子

LH_Hero，全部 `forward`，地板 **0.17°**（FB 轴 17/17 校准通过）。前四行是该物种语料
里真实存在的标签，从 2 词长到 5 词：

| 标签（词数） | cfg 1 | cfg 2 | cfg 3 | cfg 2 的速度 | 语料速度 |
|---|---:|---:|---:|---:|---:|
| `run, forward` (2) | 0.8 | 0.5 | 0.5 | 0.1152 | 0.0887 |
| `run, forward, fast` (3) | 0.4 | 0.4 | 0.9 | 0.1328 | 0.1105 |
| `run, forward, weapon, 1hand` (4) | 0.7 | 0.3 | 0.8 | 0.0875 | 0.0887 |
| `run, forward, fast, weapon, 1hand` (5) | 0.5 | 0.6 | 0.9 | 0.1000 | 0.1105 |
| `walk, forward` (2) | 0.8 | 0.8 | 1.1 | 0.0225 | 0.0182 |
| `walk, forward, carry` (3) | 0.8 | 0.3 | 0.4 | 0.0260 | 0.0184 |
| `walk, forward, weapon, 1hand` (4) | 0.8 | 1.0 | 1.1 | 0.0180 | 0.0182 |

**标签从 2 词涨到 5 词，方向误差没有任何趋势**：0.5 / 0.4 / 0.3 / 0.6，全部压在地板
（0.17°）上方一度以内。§3.2 说的"控制轴随标签长度被稀释"在生成侧看不到。

这是本轮唯一一条能不靠基线就直接检验 §3 动机的读数，因为它的预言是"**不变**"，
而不是"比某个对照更好" —— 不变可以对着自己的短标签行读。
配套的模型侧测量见 §12.7。

模式控制同时也在：`walk` 0.0225 对 `run` 0.1152 = **5.1×**，语料是 4.9×。
`fast` 也确实落地：`run` 0.1152 → 0.1328（+15%），语料同一对是 +25%。

#### held-out `(物种, 标签)`

三个 sweep、四组提示，每组的标签在语料里对该物种（或对所有物种）都不存在。
`compose` 会自动把没有对应真实 clip 的行标成 `held-out`：

| 探针 | 为什么是 held-out | cfg 1 | cfg 2 | cfg 3 |
|---|---|---:|---:|---:|
| KI_Soldier `walk, {四向}`（不带武器） | 该物种全部 locomotion clip 都是 `weapon, 2hand` | **3.8** | 4.5 | 5.2 |
| KI_Human `walk, {四向}, weapon, 2hand` | 该物种没有任何一条带武器的 clip | **3.5** | 5.8 | 6.7 |
| LH_Hero `walk, forward, fast` | `walk` 与 `fast` 在**整个语料里从未同现** | **2.0** | 2.1 | 2.5 |
| LH_Hero `run, crouch, forward` | `crouch` 只在 KI_Human 的 `walk` 上出现过 | 0.9 | **0.4** | 1.1 |

（前两行是四个方向的 `mean|e|` 平均；分方向的表在
`PWP_heldout_{soldier,human}_compose.txt`。）

三条读数：

1. **未见组合能走**。KI_Soldier 从没有过不带武器的 walk，四个方向在 cfg 1 上是
   0.9 / 3.2 / 4.3 / 6.8°，地板 0.21°。这是 §3.1 组合泛化那条主张能拿到的最直接
   证据，虽然它证明的是"够用"而不是"比整串 T5 好"。
2. **held-out 组合的最优 cfg 是 1，不是 2**，而且趋势一致（四组里三组随 cfg 单调
   变差，第四组 `run, crouch, forward` 在 cfg 2 取极小）。解释是直接的：CFG 把预测
   往「条件 − 无条件」的方向外推，标签越是模型没拟合过的，那个方向就越不可靠，
   放大它就是放大误差。**这条是实践结论：提示里出现语料没见过的词组合时，
   把 cfg 调回 1。**
3. **修饰词的迁移是不对称的**。`crouch` 从没跟 `run` 同现过，迁过去有效果
   （速度 0.1152 → 0.0900，慢 22%，方向 0.4°）；`fast` 从没跟 `walk` 同现过，
   迁过去**没有任何效果**（0.0225 → 0.0221，方向也退到 2.1°）。
   槽表示保证了 `fast` 的向量能到达 modifier 通道，但**到不了不代表用得上** ——
   这条边界值得记下来，它不是 §5 那个可分性证明能覆盖的东西。

### 12.6 质量没有塌

`eval/eval_checkpoint.py` 的 17 项电池（未开 action-label CFG）：

```
Overall score  median=0.8926  p25=0.8537  p75=0.9459  (n=136 clips)
```

136 条 clip 的分量中位数：bone_length **0.930**、jerk **0.964**、snap **0.879**、
spectral_flatness **0.861**。最弱的两格是 `NewSkeleton/task1` 0.6111
（域外 dragon + `fly --loop`，jerk 0.174）与 `InpaintFrames/task1` 0.7214
（spectral 0.143）—— 两者在 v4 上分别是 0.4795 与 0.7866，都不是本轮新出现的。

次指标相位离散度（`phase`，左右脚接触相位的 circvar，0 = 完全一致）在
cfg 1~2 上大部分格子比语料还紧：`walk, forward` 0.007~0.026（语料 0.117）、
`walk, left` ≈0.00~0.016（语料 0.630）、`run, backward` 0.017~0.154（语料 0.000）。
**唯一散的是 `run, forward`**：0.73 / 0.83 / 0.76（cfg 1 / 1.5 / 2），语料 0.002。
R1 文档把这一格记成 0.110 并点名要盯 —— 这轮它明显更差。语料 n 只有 4 条、
circvar 会偏低，但差了两个数量级不是 n 的问题，**列为观察项**。
（同样地，v4 那个 0.110 不能当基线用，理由见 §12.1。）

### 12.7 checkpoint 自身的读数（不采样，不需要基线）

把 checkpoint 里的冻结词表与 `action_label_projection` 直接读出来算：

**三个槽通道都被用上了**。第一层 `Linear(3*768, 256)` 按槽切成三块：

| 槽 | Frobenius | 谱范数 |
|---|---:|---:|
| head | 17.222 | 7.281 |
| direction | 14.579 | 6.940 |
| modifier | 21.549 | 9.011 |

没有一块塌向零 —— §5 那句"每通道的相对尺度是可学的"，学出来的结果是三块同量级，
modifier 略大（它要承载 65 个词，head 32 个、direction 6 个）。

**轴保留在通道层是恒等式，在 token 层是 84%~88%**。`||W_block @ channel||` 随标签
加长逐位不变（`walk, forward` 的 head 恒为 4.473、direction 恒为 2.470，不管后面
挂几个修饰词），这是 §5 的恒等式，实测复现。但 `action_label_projection` 后面还有
GELU 和第二层 Linear，**恒等式不会自动穿过非线性**，所以真正加到 timestep embedding
上的那个 token 变了多少是一个需要测的量：

| 只改一个槽 | 裸标签 | `+fast, weapon, 2hand` | 保留 |
|---|---:|---:|---:|
| direction: forward → backward | 3.3573 | 2.9619 | 88.2% |
| direction: forward → left | 3.8611 | 3.2388 | 83.9% |
| direction: left → right | 4.3593 | 3.6823 | 84.5% |
| head: walk → run | 4.1003 | 3.2339 | 78.9% |

即：给标签追加三个修饰词，方向轴在 token 上仍保住 **84%~88%** 的幅度，
head 轴保住 79%。
作为尺度，`||token(walk, forward)|| = 3.10`、`||null_emb|| = 0.46`、
两者之差 3.11 —— CFG 有一个量级正常的方向可外推。

### 12.8 §10 验收对照

训练后那四条验收（§10 末尾）的实际状态：

| 验收项 | 状态 |
|---|---|
| held-out `(species, action)` 不低于当前 baseline | **改判为绝对读数**（§12.5）：能走，四向个位数；"不低于 baseline"这一问因为没有 baseline 而不成立 |
| mode / direction / 长标签控制不退化 | 直接量了绝对值：mode 5.1×（语料 4.9×）、direction 贴地板、长标签无趋势（§12.2 / §12.5） |
| 原有动作质量指标不退化 | 电池中位数 0.8926，四个分量都健康（§12.6） |

### 12.9 本轮明确没有回答的

- **相对旧表示的增益** —— 没有基线，见 §12.1；
- **`run, right` 要不要上 §3.5 的硬输入位** —— 判据成立了（§12.3），
  性价比没算，留给下一轮决定；
- **`run, forward` 的相位离散** —— 现象记录在 §12.6，没有定位。

---

## 13. hands 轴：`weapon` + `1hand`/`2hand` → `hand0`/`hand1`/`hand2`（2026-09-11）

### 13.1 问题

推理时不写 `weapon` 也会生成持械姿势，而且**没有办法要一个空手的动作**。根因有两层：

1. **词表没有"空手"这个值。** 旧轴只有 `weapon`（+ `1hand`/`2hand`）一个方向，缺席 =
   未指定 = 该物种所有 clip 的边缘分布。对一个大半 clip 持械的物种，"idle" 的边缘分布就是持械。
2. **标注本身不一致。** `weapon` 只写在 combat idle / combat locomotion 上；同一角色的
   attack / hurt / die / block 一律没写（`KI_Warrior_Attack02` 是双手剑 slash，标签
   `attack, slash`；`LH_Hero_THSwordIdle` = `idle`，而 `THSwordRun` =
   `run, forward, weapon, 1hand`；`IAC_Caveman_Attack1Weapon` = `attack, right, punch`）。
   所以"没有 weapon 的 clip"本来就是持械与空手的混合，不是空手集合。

另外一个类别值要写两个 token（`weapon, 2hand`），既麻烦，又让两个 token 的共享语义成为
整张表最坏的近邻对（预检文档 §5）。

### 13.2 定稿

```text
HANDS_VOCAB = ("hand0", "hand1", "hand2")      # motion_labels.py，收尾 CONTROLLED_VOCAB
hand0  双手空着
hand1  一只手拿着东西（剑、火把、麻袋），另一只空着
hand2  两只手都拿着东西：双手握持（大剑、步枪、拉开的弓、箱子）或一手一件（剑 + 盾）
```

- **语义是手部占用数，不是武器类别**（用户定稿）。剑 + 盾 = hand2，与步枪相同；匕首和火把都是
  hand1；弓提在手边 hand1，拉开 hand2。走路时哪条胳膊在摆，由这个数决定。
- 互斥，**至多一个**；`parse_action_label` 与 `action_label_slots` 都硬失败。
- **不写 = 未指定**（边缘分布），与方向轴同一规则。`hand0` 是一个肯定陈述，不是默认值。
- 动作用什么器具（`bow` / `gun` / `hammer` / `shield`）留在 block I，与 hands 词并列：
  `attack, bow, hand2`。它们描述动作种类（拉弓、射击、敲打、盾击），不是手部状态。
- **标注范围（用户定稿）：只补出现过持械 clip 的物种**（unitybundles 34 个，见 13.4），
  这些物种的每条 clip 都给定 hand0/1/2，手部状态在 clip 内变化的（拔剑/收剑/举起/放下）留空。
  其余物种（含全部 zoo / zoo_upgrade）不写。

### 13.3 为什么是第四个槽通道，而不是 modifier 槽

hands 词一旦补齐，会出现在有手物种**几乎每一条** clip 上。放进 modifier 槽，
`attack, slash, hand2` 的 modifier 通道 = L2(mean(slash, hand2))：`slash` 的份额减半，
而且 `attack, slash` 带不带 hand 词读数不同——这正是 §3.2 反对的稀释。单独一个通道：

- head / direction / modifier 三个通道在加不加 hand 词时**逐位相同**
  （`test_hands_channel_is_the_token_vector_and_leaves_the_modifier_channel_alone`）；
- 该轴至多一个成员，所以通道就是该 token 的向量本身，模型只需分开三个点；
- 词表变了 checkpoint 本来就要重训，多一个通道零额外成本。

`ACTION_LABEL_SLOTS = ("head", "direction", "modifier", "hands")`，
`action_label_projection` 第一层 `Linear(4 * 768, latent_dim)`；
`ACTION_LABEL_PARSER_CONTRACT_VERSION` 1→2，`CKPT_VERSION` 7→8，v7 checkpoint 拒绝加载。

### 13.4 T5 文本与几何

三个 token 的 T5 文本是裸计数：`hand0` = "empty hands"，`hand1` = "one hand"，
`hand2` = "both hands"。按预检的口径（同形状探针、词表均值中心化）实测：

| 候选对 | 余弦 |
|---|---:|
| "one hand" / "both hands"（采用） | 0.527 |
| "empty hands" / "one hand"、"empty hands" / "both hands"（采用） | 0.244 / 0.343 |
| "one-handed weapon" / "two-handed weapon" | 0.868 |
| "holding in one hand" / "holding in both hands" | 0.804 |
| "weapon in one hand" / "weapon in both hands"（旧表） | 0.787 |

共享锚短语（weapon / holding）是碰撞来源，与预检 §5 的结论一致；因为 hands 是独立通道，
三个点线性可分即可，0.53 足够。迁移后重跑预检：GO，`slot/eos_keep/center_l2`，
最坏近邻 0.9504（旧 0.9559），hands 槽 3 个配置最坏对 0.5245、成员 readout 间隔 +1.000，
槽源秩 64 / 6 / 62 / 3 = 135，键唯一 473/473。新 sidecar
`embedding_fingerprint = 202b0219…`，`conditioning_contract_fingerprint = 70c8ea6a…`。

### 13.5 标注迁移

迁移一次性完成（一次性工具 `tools/dataset_cleanup/migrate_hand_state_labels.py` 已删除）。
每条 clip 的判定来源分三档：

| 来源 | 含义 | 条数 |
|---|---|---:|
| `tag` | 旧标签已带计数（`1hand`/`2hand`），直接沿用；策略表与之冲突时打印并以旧标签为准 | 120 |
| `name` | clip 名写明了道具状态（Weapon / 2HLong / Shield / Rifle / Torch / Axe …） | 155 |
| `impl` | 器具词决定：拉弓必然双手 | 14 |
| `pack` | **物种级默认**（源包惯例：RTS 单位武器不离手、KI 士兵包全程步枪…），无人逐条核过 | 412 |

结果：unitybundles 685 行改动，hand0 173 / hand1 263 / hand2 249，17 条留空
（拔剑、收剑、举起、放下、取放物），旧标签与策略无冲突。**`pack` 档 412 条需要过一遍**
（工具已删，如需修正直接改 `action_labels.jsonl`）。评审 GIF 只渲染骨架、
不带道具，raw GLB 里也没有道具 mesh，所以这一档只能靠对源包的了解。

同包里显然持械但从没打过旧标签、因此不在范围内的物种（`TTR_LightInfantry`、
`TTR_LightCavalry`、`TNR_CavalryMage`、`MLH_Footman`、`MLH_Horseman`、`MLH_Mage`、
`KI_Slinger`、`MLS_DemonHunter`…）没有动；要纳入，直接在 `action_labels.jsonl`
补对应物种的 hand 状态（工具已删）。

顺手修的一处旧缺陷：`Dog-2_Swimturn` / `SwimTurn2`（zoo，locomotion）拼的是
`turn, left, swim`，而 unitybundles 同 group 的 `MB_TigerDrago_SwimL` 拼 `swim, turn, left`。
主词序一致性校验是**按文件**跑的，跨库不查，预检的键唯一硬门才把它抓出来（迁移前就是 NO_GO）。
已改成 `swim, turn, left/right`。

### 13.6 推理

```text
--action_label "idle, hand0"          空手 idle
--action_label "walk, forward, hand2" 双手持物前进
--action_label "attack, bow, hand2"   拉弓
--action_label "idle"                 手部状态未指定（该物种的边缘分布）
```

### 13.7 待办

- **重训**（词表、槽布局；现为 CKPT v13，§14）；`merged_locomotion_v7` 及之前的 checkpoint 全部拒绝。
- `pack` 档 395 条人工过一遍（13.5）。
- 是否把同包未标物种纳入范围（13.5）。

---

## 14. 去掉 transition 的方向机制（2026-09-17）

### 14.1 改了什么

条件侧：

- `action_label_slots(tokens)` 不再接收 `action_group`，只返回 `word_ids` / `word_mask` /
  `slot_ids`；`role_ids`、`order_head_mask`、`ROLE_NONE` / `ROLE_HEAD_1`、
  `action_order_enabled()` 全部删除；
- `assemble_slot_channels(word_vectors, slots)` 与模型的张量镜像不再有第二主词分支：
  head 槽就是本槽词向量的均值，三个 group 同一条规则；
- `R_B` 的推导、校验、提交哈希（`role_b_material()` / `validate_role_b_payload()` /
  `ROLE_B_*`）删除，模型的 `action_role_b_perm` / `action_role_b_sign` buffer 删除；
- 槽源秩证书的 head 源只剩 36 个状态词本身（不再并入 `R_B` 像），总秩 143 → 107；
- conditioning contract 去掉 `role_gate` / `role_b_material_sha256` / `role_ids` 三个字段，
  `slot_fields` 变为三项，`representation.kind` 改名 `slot_channels`；
  `ACTION_LABEL_PARSER_CONTRACT_VERSION` 2 → 3，`CKPT_VERSION` 12 → 13，
  checkpoint payload version 2 → 3；
- 预检去掉「反向 transition 中位数 ≤ 0.50」硬门与 `rev50` 列，head 配置枚举从有序对
  改为无序对，键唯一判据从 `(group, {(word, role)})` 改为 `(group, {word})`；
  报告 `schema_version` 3。

数据侧：

- `_validate_head_order_consistency` 不再豁免 transition：同一 group 内同一词集
  两种主词顺序在加载时硬失败；
- 词表补 `draw`（"drawing a weapon"）、`sheathe`、`stop`（"run to stop"）、`kneel`，
  四个都进 `HEAD_VOCAB`；
- 语料里所有靠词序表达方向的 A→B 对改写：`idle, attack` → `draw`、`attack, idle` →
  `sheathe`、`run, idle` → `stop`、`idle, crouch`（蹲下）→ `kneel`、`crouch, idle` →
  `getup, crouch`、`Bear_StandToHind` / `HindToStand` 等反向对拆成不同词集；
  `MU04_*_IdleToSpinAttack` / `SpinAttackToIdle`、`RU01_GripperRobot_*Grip*` 共 15 条
  没有对应事件词的 clip 从 unitybundles 移除；
- 保留的 29 条双主词 transition 是「主词 + 修饰主词」（§2.2 表），与另外两组用法一致。

词表结构（同日）：

- `STATE_VOCAB` 改名 **`HEAD_VOCAB`**：这个集合唯一的作用就是"进 head 槽、计入
  `ACTION_LABEL_MAX_HEADS`、保持书写顺序"，周边一律叫 head（`SLOT_HEAD`、`head_words_in`、
  页面里的 `heads`）；"state" 是为 transition 端点服务的旧判据，而现在 transition 的主力
  主词恰是事件词（draw / sheathe / stop / getup / land），不是 state。contract payload 的
  `state_vocab` 键随之改为 `head_vocab`；review 页 `label_contract.head_vocab`；
- `ACTION_VOCAB = HEAD_VOCAB + MODIFIER_VOCAB`：主词抽成开头的一个 block（内部字母序，
  因为主词位置不参与任何排序），后面只剩修饰词 block，"head 是 action 的子集"从 assert
  变成结构。**修饰词之间的相对顺序原样保留**，所以 3745 条标签的 canonical 拼写零变化，
  三个 jsonl 一行未动；
- word id 变了 → sidecar 用 `build_action_label_embeddings.py --force` 在 CUDA 上重建
  （词向量集合不变，只是行序），`embedding_fingerprint ef548fd7…`、
  `conditioning_contract_fingerprint 52785435…`。

### 14.2 为什么

`R_B` 从建起来到删掉都没有生成侧证据（§12 的 v5 是 locomotion checkpoint，`role_ids`
恒为 `NONE`）。留着它的代价是三条：

1. 同一个标签字符串在 transition 与 stationary 的 checkpoint 里是**不同的条件向量**
   （`idle, attack` 在前者过 `R_B`），标签失去了「跨 group 同义」这条最基本的性质；
2. 方向只能靠标注者记住「逗号左边是起点」，而语料里 58% 的 transition 本来就是
   自反事件词（`die` / `getup` / `spawn`），同一个逗号在两套语义之间来回切换；
3. 它只覆盖「两个端点都在 `HEAD_VOCAB` 里」的转换；`getup` 早已合并了
   dead / sleep / lie / sit / crawl 五种起点，说明事件词才是语料实际在用的表达。

把方向写进词表后，词序机制没有东西可编码，删掉它让 transition 与另外两组共享
同一条件语义、同一校验规则。

### 14.3 预检

`tools/evaluate_action_label_geometry.py`（CUDA，2026-09-17，词表重排之后）：**GO**，
`slot/eos_keep/center_l2`，3745 行 / 383 个标签，最坏近邻 0.9521（baseline 0.9873），
通道漂移 0.0，词表秩 107 / 中心化 106，槽源秩 36 / 6 / 62 / 3 = 107，键唯一；
两个指纹与重建后的 sidecar 逐字相同（`ef548fd7…` / `52785435…`）。
数值诊断：head 666 配置最坏对 0.8920、成员 readout 间隔 +0.771（`putdown`）；
direction 63 配置 0.9580 / +0.559；modifier 39773 配置 0.9690 / +0.627；hands 3 配置
0.5238 / +1.000。

### 14.4 待办

- **重训**：`CKPT_VERSION` 13 拒绝一切旧 checkpoint；conditioning contract 指纹
  本身也拒绝（`slot_fields` 与 `representation` 都变了）。
- transition group 的第一个 checkpoint 训出来之后，验收「`draw` / `sheathe` / `stop`
  生成的动作朝向正确的端点」—— 这是 §10 撤掉的那条验收的替代形式。
