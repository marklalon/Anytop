# Action label 词级条件方案

> 状态：**已落地（2026-09-06）。** 数据契约、定稿表示（每个角色槽一个条件通道）、
> word-keyed sidecar、loader、模型、checkpoint v2 与测试全部切换完成；
> 剩下的是要真训练才能做的两步：held-out 消融（§9.7）和
> `action_label_cfg_scale` 重标定（§9.8）。**需要重训**：v1 checkpoint 一律拒绝加载。

## 1. 当前状态

三套数据共有 3802 个动作：

| 数据源 | 动作数 |
|---|---:|
| zoo | 970 |
| zoo_upgrade | 247 |
| unitybundles | 2585 |

- `CONTROLLED_VOCAB`：103 个 token，与语料实际用词完全一致；
- 不同 `action_label`：398 个；按 group 展开为 408 个条件点；
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
  → loader 发 word_ids / role_ids / slot_ids / word_mask / order_head_mask（定长 8）
  → 模型用 checkpoint 内的冻结词表拼三个槽通道（张量镜像 assemble_slot_channels）
  → action_label_projection: Linear(3 * t5_out_dim, latent_dim)
  → 加入 timestep embedding
```

`CKPT_VERSION` 为 2，checkpoint payload 的 `metadata.checkpoint_version` 也是 2。
推理不读任何 sidecar：词表和 `R_B` 是 persistent buffer，随权重一起出入 checkpoint。

## 2. 已完成的数据契约

### 2.1 标签格式

- 标签由受控 token 组成，以逗号分隔；**总词数最多 8 个**，未知词、重复词、空段和第 9 个词均硬失败；
- 每条非空标签必须包含 1～2 个 `STATE_VOCAB` 候选主词；
- 主词保持书写顺序；该顺序在 transition 中表示时间方向；
- 方向词紧跟 `turn`，否则紧跟最后一个主词；
- 其他修饰词按 `CONTROLLED_VOCAB` 顺序排列；
- 空标签表示无条件分支，必须走 `action_label_null_emb`，不能编码空文本。

`STATE_VOCAB` 只用于识别候选主词，不等于「自动启用顺序编码」。

### 2.2 transition 顺序 gate

只有真正的双状态 transition 才启用顺序角色：

```python
heads = [token for token in tokens if token in STATE_VOCAB]
order_enabled = (
    action_group == "transition"
    and len(heads) == 2
    and heads[0] != "turn"
)
```

当前 755 条 transition 的结构为：

| 结构 | 数量 | 顺序编码 |
|---|---:|---|
| 单词事件，如 `die`、`getup` | 438 | 关闭 |
| `turn` 家族 | 223 | 关闭 |
| 单状态 + 修饰词 | 22 | 关闭 |
| 双状态 A→B | 72 | 开启 |

启用时：

- 第一个端点角色为 `NONE`；
- 第二个端点角色为 `HEAD_1`，应用固定正交变换 `R_B`；
- 两个端点由 `order_head_mask` 明确标出；
- stationary、locomotion，以及未通过 gate 的 transition 不应用角色变换。

角色分配的唯一实现位于
[`action_label_conditioning_contract.py`](../data_loaders/truebones/truebones_utils/action_label_conditioning_contract.py)。
`R_B` 不是仓库里的数组文件：它由同一模块的 `role_b_material()` 从固定 namespace 即时推导
（纯 stdlib、整数运算、约 1 ms 并带缓存），并对照代码里提交的 `ROLE_B_MATERIAL_SHA256` 校验，
见 §7.1。

## 3. 要解决的问题

### 3.1 完整字符串条件不利于组合泛化

当前每个完整标签只有一个离线 T5 向量。模型见过 `run, forward` 和 `walk, weapon`，并不能直接复用
其中的词向量组成未见组合。标签变化还会使整份 label-keyed sidecar 失效。

### 3.2 长标签中的控制轴被稀释

在完整字符串或均匀逐词平均中，一个词的相对份额随标签长度下降。给标签增加装备、动作细节后，
基础动作和方向的差异明显减弱。语料里 ≥3 词的 clip 占 650/3802（17.1%）。

### 3.3 transition 方向需要显式位置角色

逐词求和或平均对词序可交换。没有位置角色时，`idle, attack` 与 `attack, idle` 会得到完全相同的
条件。当前 5 组双向 transition 共 36 个 clip，会因此成为「同条件、反动作」的冲突样本。

## 4. 已否决的表示：单向量加权平均

```python
label_emb = sum(weight[token] * word_emb[token]) / sum(weight[token])
```

否决的不是某一组权重，而是整个族。令 ρ = 细节词在池化向量里的能量占比：轴保留
≈ 1/(1+ρ) 要求 ρ 小，而「只差一个修饰词的标签要分得开」要求 ρ 大。一个池化向量只有一个 ρ，
两者不可能同时满足；四档权重只是在同一条 frontier 上滑动：

| 候选 | pairwise p95 | 最近邻中位 | 有效秩 | 长轴保留 | 反向对 |
|---|---:|---:|---:|---:|---:|
| 当前整串 T5 | 0.798 | 0.878 | 18.34 | — | 0.806 |
| 中心化 + 均匀权重 + `R_B` | 0.633 | 0.841 | 14.58 | 0.392 | 0.015 |
| 中心化 + medium + `R_B` | 0.751 | 0.901 | 10.71 | 0.703 | 0.015 |
| 中心化 + 强语义权重 + `R_B` | 0.969 | 0.990 | 7.73 | 1.012 | 0.015 |

`R_B` 本身有效并保留：它把 5 对真实反向 transition 的余弦中位数从 0.806 降到 0.015。
上表是一次性的历史记录，不是每次预检都重算的东西：否决理由是结构性的（一个池化向量只有一个 ρ），不靠这几个数字的余量支撑。`ACTION_WORD_WEIGHT_PRIOR` 与评测器里那 32 个 `word/...` 候选已于 2026-09-06 一并删除；要复算就从 git 历史取回那一版工具（语料 SHA-256 见预检文档 §1）。

## 5. 定稿表示：每个角色槽一个条件通道

```text
head       = L2( mean(  本槽词向量，HEAD_1 词先过 R_B ) )
direction  = L2( mean(  本槽词向量 ) )
modifier   = L2( mean(  本槽词向量 ) )
缺席的槽   = 零行（present 标记为 False），不重新归一化其他槽
条件输入   = 三个通道按固定顺序拼接
```

槽划分：`head` = `STATE_VOCAB` 成员，`direction` = `DIRECTION_VOCAB` 成员，
`modifier` = 其余全部词。唯一实现是
[`assemble_slot_channels`](../data_loaders/truebones/truebones_utils/action_label_conditioning_contract.py)。

它为什么能同时满足两边：

- **轴保留是恒等式，不是超参**。head / direction 通道的输入只含本槽的词，标签从 2 词长到
  8 词，这两个通道逐位不变（实测最大漂移 0.0）。不存在需要在 0.15 和 1.0 之间权衡的常数。
- **可分性由拼接给出**。三个通道各自进入 `action_label_projection` 第一层的一个分块
  （对拼接做一次 Linear 恒等于对每块做一次 Linear 再求和），所以每通道的相对尺度是**可学的**，
  离线不需要、也不应该替模型定一个预算。
- **注入性**。`(group, {(word_id, role_id)})` 在 408 个条件点上唯一，碰撞是构造上不可能。
- **可按 token 审计**。一个词只影响它所在的槽。

槽内均值不会在当前词表上造成信息碰撞。定稿词向量的槽源秩为：head 64/64（32 个原始状态词
+ 32 个 `R_B` 角色源）、direction 6/6、modifier 65/65。源向量线性无关意味着不同成员集合的
归一化和不可能相同，并且存在一个线性 readout 能判断每个词是否在槽内；这个证明覆盖解析器允许的
**全部非空槽组合**，包括总词数上限 8，而不是只覆盖语料中见过的组合。

三个槽是互不重叠的拼接块，总可达子空间秩为 135，小于默认 `latent_dim=256`，所以第一层 Linear
可以在整个可达空间上保持单射。另做的数值诊断穷举完整 head（1024 种）和 direction（63 种）
配置，以及当前语料上限的 ≤3 词 modifier 配置（45825 种）；它用于观察最近邻和数值间隔，不承担
全域正确性证明。K-token 的信息优势因此只剩布局和时间局部化能力，见 §5.1。

### 5.1 为什么不是 K-token

K-token（保留每个词的独立 token）在信息上严格更强，但实测下：

- 槽源满秩保证全部合法组合的成员可由一层线性读出；数值诊断的最小 readout 间隔也为正；
- 头序反转的最坏情况完全由两个词本身的相似度决定（`laydown↔putdown` 原始 atom 余弦 0.594；
  单通道+`R_B` 0.601、拆两个通道 0.592、拆通道+`R_B` 0.592），换布局无效，K-token 同样无效；
- 学习注意力池化在初始时接近均匀，反而把「轴保留」从恒等式退化成训练目标。

K-token 唯一不可替代的能力是**时间局部化**（让第 40 帧去看第二个端点）。当前条件是加在
timestep embedding 上的单个向量，对所有帧恒定；若将来确实需要逐帧按词条件化，再上 per-layer
cross-attention。为此 loader 的输出保持词级（§7），换表示时只动模型消费端。

## 6. 训练前 geometry gate

只对模型**改不回来**的性质设硬门；p95 / 最近邻中位数 / 有效秩是各向异性指标。三槽完整
可达子空间秩为 135，紧随其后的 256 维 `nn.Linear` 足以在这个子空间上保持单射并重标度，
因此这些指标只报告、不阻断。硬门与实测结果：

| 硬门 | 判据 | 定稿表示实测 |
|---|---|---|
| 碰撞 | 余弦 ≥ 0.999999 的不同标签对 = 0 | 0 |
| 最坏近邻 | 不高于 baseline 同项 | 0.9559（baseline 0.9900） |
| 反向 transition | 中位数 ≤ 0.50 | 0.015 |
| 通道漂移 | 追加修饰词后 head/direction 逐元素变化 = 0 | 0.0 |
| 词表秩 | 满仿射秩 | raw 103 / 中心化 102 |
| 槽源秩 | head / direction / modifier 源分别满秩 | 64 / 6 / 65，全部满秩 |
| 投影宽度 | `latent_dim` ≥ 三槽总可达秩 | 256 ≥ 135 |
| 键唯一 | `(group, {(word, role)})` 唯一 | 408/408 |

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
  embeddings = float32[103, 768]     # eos=keep, masked mean, center_l2
  embedding_contract                 # 含 word_table_sha256
  embedding_fingerprint
```

装配规则：

1. 训练入口 bootstrap 只读取和验证一次 sidecar，并调用一次 `role_b_material()` 取得 `R_B`；
2. 同一个不可变 conditioning bundle 同时传给 loader 和 model；
3. **loader 只输出 `word_ids`、`role_ids`、`word_mask`、`order_head_mask`、`slot_ids`**，
   不输出拼好的向量。槽拼装在模型侧，用 checkpoint 内的冻结词表完成；
4. model 将冻结词向量和 `R_B` 保存为 persistent buffer，并用张量镜像
   `assemble_slot_channels` 的语义（同一套 `slot_ids`，不另写一份规则）；
5. 推理只读取 checkpoint 内的词向量，不从数据目录重新选择 sidecar。

第 3 条是有意的：离线预拼 `label → bundle` 查表会把表示锁进数据通路，以后换表示要重建数据；
发词级 id 则只需改模型消费端，`embedding_fingerprint` 不受影响。

### 7.1 `R_B` 按需推导，不落地成文件

`R_B` 是 768 维带符号置换，由 `ROLE_B_NAMESPACE` 推出：下标按 `SHA-256(namespace/perm/i)`
排序，符号取 `SHA-256(namespace/sign/i)` 首字节奇偶。全过程只用 stdlib 和整数，没有浮点、没有
RNG、没有字典序依赖，SHA-256 摘要两两不同因此 `sorted` 的结果与平台和 Python 版本无关；
实测一次 0.9 ms，且 `functools.lru_cache` 之后为零。

冻结 `R_B` 的不是那份文件，而是代码里提交的常量：

```python
ROLE_B_NAMESPACE      = "anytop/action-label/role-b/v1/t5-base/768"
ROLE_B_MATERIAL_SHA256 = "0204f95ca92d163554ed17bc8ff22ee2858ae128c2691b4893cc0f9c958c4c2b"
```

`role_b_material()` 每次推导后都比对这个哈希，namespace、维度或推导规则被改动会当场硬失败，
而不是悄悄换掉 `ROLE_HEAD_1` 的含义 —— 这正是原先「提交材料值」想要的性质，只是把 13 KB
的数组换成一行 64 字符。同一个哈希继续通过 `role_b_material_sha256` 进入
`conditioning_contract_fingerprint`，所以跨材料 resume 仍在 checkpoint 加载时被拒绝，
预检文档 §1 记录的 `conditioning_contract_fingerprint` 也不因删文件而改变。

三条边界：

- **checkpoint 内的 `perm`/`sign` 仍是 persistent buffer**（§7 第 4 条不变）。旧 checkpoint 的
  权威材料是它自己存的那份，不是当前代码推出来的那份；
- 非本地推导来的材料（checkpoint buffer、外部导出副本）走 `validate_role_b_payload()`，
  结构与哈希检查一条都不少；
- 推导规则只允许存在一份实现，即 `role_b_material()`；外部导出副本（旧 artifact、
  从 checkpoint 抽出的 perm/sign）一律经 `validate_role_b_payload()` 复核，
  不引入任何第二份推导代码。

模型侧的改动很小：

- `nn.Linear(t5_out_dim, latent_dim)` → `nn.Linear(3 * t5_out_dim, latent_dim)`；
- 构模时校验实际 `latent_dim` 不小于 sidecar/预检记录的槽源总秩（当前 135）；
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
- 有序词表、`STATE_VOCAB`、parser/canonicalization 版本；
- slot 字段、槽名、最大词数、transition gate；
- `R_B` 材料 hash（`ROLE_B_MATERIAL_SHA256`，由 `role_b_material()` 现算并自校验）；
- 最终表示布局（`slot_channel_representation()`）及其全部参数。

修改角色规则不会使纯 embedding sidecar 失效；修改 T5 文本、EOS 或向量后处理会使两层都失效。
当前定稿值见预检文档 §1。

### 8.2 checkpoint v2

不用 `get_extra_state()`。元数据放在 checkpoint 顶层，避免与当前 EMA buffer 同步逻辑冲突：

```python
{
    "model": model_state_dict,
    "model_avg": ema_state_dict_or_none,
    "metadata": {
        "checkpoint_version": 2,
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
   因为它认证的是 buffer 里的词表和 `R_B`，而那两个 buffer 是 persistent 的，
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
   逐位相等（float64 下 8.3e-17）；第一层 Linear 宽度为 `3 * t5_out_dim`；
5. ~~checkpoint v2 和两层指纹校验~~ **已完成**：`build_checkpoint_payload` /
   `load_checkpoint_weights` / `bind_checkpoint_action_conditioning`；
6. ~~单元测试、EMA/resume round-trip 和跨契约拒绝测试~~ **已完成**：
   [`test_action_label_word_conditioning.py`](../tests/test_action_label_word_conditioning.py)；
7. **未做（需要训练）**：固定 held-out `(species, action)` 切分后训练消融；
8. **未做（需要训练）**：重标定 `action_label_cfg_scale`。

## 10. 验收

实现测试至少覆盖：

- 当前 72 条 transition 开启顺序角色，其余 683 条关闭；
- action label 总词数恰好 8 个时接受，第 9 个词硬失败；
- `turn` 开头和单主词 transition 不应用 `R_B`；
- `role_b_material()` 可重复推导、命中提交哈希，且被篡改的材料（符号翻转、非置换）硬失败；
- 反向 transition 条件不碰撞；
- head/direction 通道不随修饰词数量变化（等式断言，不是阈值）；
- 缺席槽为零行且不改变其他槽；
- padding 与所有 mask 完全生效；
- loader/model 使用同一 ordered vocab、同一 `slot_ids` 与同一 fingerprint；
- 实际训练 `latent_dim` 小于槽源总秩时启动失败；
- 普通模型、EMA、保存、加载和 resume 往返一致；
- 外部 sidecar、代码词表或 checkpoint metadata 不一致时硬失败；
- 推理不依赖外部 sidecar。

训练后以生成动作验收：

- transition 末帧更接近第二端点；
- held-out `(species, action)` 组合不低于当前 baseline；
- mode、direction 和长标签控制能力不退化；
- 原有动作质量指标不退化。

## 11. 当前已落地文件

- 数据词表、parser 和 canonicalization：
  [`motion_labels.py`](../data_loaders/truebones/truebones_utils/motion_labels.py)
- 角色、槽划分、槽拼装、槽源秩证书、bundle 与双指纹契约：
  [`action_label_conditioning_contract.py`](../data_loaders/truebones/truebones_utils/action_label_conditioning_contract.py)
- 固定 `R_B`：`role_b_material()` / `ROLE_B_MATERIAL_SHA256`，与角色契约同文件（无数据文件；
  外部副本复核走 `validate_role_b_payload()`）
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
