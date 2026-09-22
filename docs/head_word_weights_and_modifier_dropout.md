# 采样：逐 clip 均匀 + 稀有首词下限；条件：修饰词槽 dropout

> 状态：**已实施**（2026-09-21），`merged_all_v23` 起生效。
> 触发：`merged_all_v22` 生成的 attack 明显差于单组 `merged_stationary_v20`，而 attack 是语料里
> clip 最多的首词；样本少得多的 transition 组反而没问题。
> 影响：`--balanced` / `--balanced_group_floor` 删除；新增 `--rare_head_word_floor` /
> `--rare_head_word_max_boost`（稀有首词的采样下限）、`--head_word_weights`（显式逐词乘数，
> 暂不配置）、`--modifier_slot_drop_prob`；`--direction_slot_drop_prob` 默认 0.3 → 0.15。
> 不 bump `CKPT_VERSION`、不改契约、不需要 cond regen；采样分布变了，**换 `RUN_NAME` 重训**。

## 0. 一句话

首词 sqrt 平衡（`--balanced`）把 attack 每条 clip 的曝光砍了一半，去换稀有首词的抬升；attack
之所以"样本最多"是因为它是 110 个标签的总和，按首词分组恰好惩罚了最异质的那个组。现在改回
逐 clip 均匀采样，只给语料尾部的首词一个采样下限（clip 数不足 N 的首词按 N 条计，逐 clip 倍率封顶）；
同时给修饰词槽加训练侧 dropout，让裸 `attack` 成为"任一种 attack"的边缘分布而不是 11 条无修饰词
clip 的查表。

## 1. v22 的问题在哪

按 v22 的规则（首词分组、sqrt、无 floor）在全量 3635 条上重算：

| head | clips | 均匀采样 | v22 balanced | 每 clip 曝光倍数 | 400k×20 每 clip 曝光 | 该首词下的不同标签数 |
|---|---:|---:|---:|---:|---:|---:|
| attack | 891 | 24.5% | **12.4%** | **0.51** | **~1116** | **110** |
| idle | 732 | 20.1% | 11.3% | 0.56 | ~1232 | 62 |
| die | 293 | 8.1% | 7.1% | 0.88 | ~1947 | 17 |
| turn | 200 | 5.5% | 5.9% | 1.07 | ~2356 | 6 |
| getup | 60 | 1.7% | 3.2% | 1.95 | ~4302 | 7 |

对比 `merged_stationary_v20`（均匀采样、200k×24）：attack 每 clip ≈ 2494 次曝光。v22 跑了两倍
步数，attack 每条 clip 看到的次数反而只有 v20 的 45%。

落到标签层面：clip 数相近的 `attack, bite`（85 条）与 `turn, left`（98 条），v22 下后者拿到
2.4 倍的采样质量（1.22% vs 2.98%）；`getup`（49 条）比 `attack, right, swat`（24 条）多 8 倍。
sqrt 规则的前提"首词 = 这条 clip 是关于什么"对 transition 成立（turn 6 种标签、spawn 1 种），
对 attack 完全不成立（110 种，49 种只有 ≤2 条）。transition 既被抬权又本身单模态，所以它好
而 attack 差。

自动 eval（jerk / snap / bone-length 平滑度）看不出 attack 退化（0.917 vs 0.900），但 idle
0.892→0.726、rest 0.838→0.587，整个 stationary 侧都在 v22 变差了。没有 A/B（v20 是 version 14
无法采样），合并语料本身、latent 384 / AdaLN、is_loop token 也都是 v20→v22 一起变的因素。

## 2. 采样：均匀 + 稀有首词下限（`--rare_head_word_floor`）

规则（[`TruebonesSampler`](../data_loaders/truebones/data/dataset.py)）：

```
n(首词)       = 该首词在当前训练子集里的 clip 数
boost(首词)   = 1                                   若 n ≥ floor
              = min(floor / n, max_boost)          若 n < floor
weight(clip)  = boost[首词] × head_word_weights.get(首词, 1)
按池内归一化；pointer 之前的条目权重 0；<unlabeled> 永远是 1
```

- **下限是一条规则，不是一张表。** clip 数不足 `floor` 的首词按 `floor` 条计，即整词的采样
  质量至少等于一个 `floor` 条 clip 的首词；逐 clip 倍率 `floor / n` 封顶在 `max_boost`。
  新增一个稀有首词自动被覆盖，不用往列表里补（之前按词手写 15 个乘数，容易漏）。
- 均匀采样下每条 clip 曝光 = 8M / 3635 ≈ **2200 次**，attack 回到 v20 的量级；下限只动尾部，
  attack / idle 等 ≥ floor 的首词一律 ×1。
- 没有任何自动平衡规则，也没有物种平衡。总预算固定，下限只是把预算从所有人那里等比例搬
  一点给尾部。
- `--head_word_weights word=w,...` 保留为**显式逐词乘数**，乘在 boost 之上，给"某个词就是要
  手调"的情况用；列出的词必须是 `HEAD_VOCAB` 且当前子集里有 clip，否则启动硬失败。
  `train_all.bat` 当前**不传**（no-op）。
- 两者都不传时走普通 `RandomSampler`，完全不读标签。

`train_all.bat` 当前取值：`--rare_head_word_floor 20 --rare_head_word_max_boost 4`。在全量
3635 条上实际得到（训练日志 rank 0 一行）：

```
[sampler] head-word weights over 3635 clips, floor 20 (max boost 4): crawl 1x4 0.03%->0.11%,
  sheathe 1x4 0.03%->0.11%, kneel 2x4 0.06%->0.21%, fall 2x4 0.06%->0.21%, lift 2x4 0.06%->0.21%,
  rear 2x4 0.06%->0.21%, stop 5x4 0.14%->0.53%, putdown 6x3.33 0.17%->0.53%, draw 7x2.86 0.19%->0.53%,
  pickup 7x2.86 0.19%->0.53%, sitdown 7x2.86 0.19%->0.53%, takeoff 9x2.22 0.25%->0.53%,
  roll 9x2.22 0.25%->0.53%, burrow 9x2.22 0.25%->0.53%, laydown 16x1.25 0.44%->0.53%
```

即 ≤5 条 ×4（封顶），6–19 条抬到 20 条当量（0.53%），≥20 条不动。总质量偏移 135 个 clip
当量 / 3635 ≈ 3.7%，attack 24.5% → 23.6%。旧 sqrt+floor 10 给尾部的每 clip 倍率是 ~9.4×，这里
最高 4×——尾部的问题主要是数据不够，抬得再高只是背单条 clip。

## 3. 条件：修饰词槽 dropout（`--modifier_slot_drop_prob`）

schema 自己声明"只写一部分词 = 边缘分布"，但之前只有 direction 槽靠 dropout 兑现了这一点。
modifier 槽没有 dropout，模型学到的是"attack + 空 modifier = 那 11 条无修饰词 clip"（5 条机器人
BlastAttack、2 条蝎子、1 条法师……），推理时打裸 `attack` 得到的是这个杂烩。

实现与 direction 完全同构（[`model/anytop.py`](../model/anytop.py) `_drop_slot_words`）：训练时按行
以 `modifier_slot_drop_prob` 抹掉 `SLOT_MODIFIER` 的词，纯 mask 运算，eval / 推理不丢；两个槽的
抽样互相独立。每行显式修饰词监督的比例 = (1 − cfg_drop) × (1 − p) = 0.8 × 0.85 = 68%。

| 槽 | ∅ 的含义 | 训练时 dropout |
|---|---|---|
| direction | 边缘（任一方向） | `--direction_slot_drop_prob` 0.15（默认由 0.3 改为 0.15，与脚本取值一致） |
| modifier | 边缘（任一种该首词的动作） | `--modifier_slot_drop_prob` 0.15 |
| hands | **空手**（内容默认） | 无——空 = 空手，不是 unspecified |

已知副作用：`idle` 的边缘会混进少量 `idle, roar / sleep / eat`（56×0.15 ≈ 8 条当量 vs 226 条
真 idle），可接受；`attack` 边缘混进 cast / projectile 正是目的。第二个主词（`attack, jump`）
不在 modifier 槽里，不会被丢。

## 4. 代码位置

| 位置 | 作用 |
|---|---|
| [`data_loaders/truebones/data/dataset.py`](../data_loaders/truebones/data/dataset.py) | `rare_head_word_boost`、`parse_rare_head_word_floor`、`parse_head_word_weights`、`clip_action_head_word`、`TruebonesSampler` |
| [`model/anytop.py`](../model/anytop.py) | `_drop_slot_words` / `_drop_modifier_slot` |
| [`utils/parser_util.py`](../utils/parser_util.py) | `--rare_head_word_floor`、`--rare_head_word_max_boost`、`--head_word_weights`、`--modifier_slot_drop_prob` |
| [`tests/test_head_word_weights_sampler.py`](../tests/test_head_word_weights_sampler.py) | 下限规则（floor/cap、≥floor 不动、unlabeled 不动、与显式乘数相乘）、首词归键、CLI 校验 |
| [`tests/test_action_label_word_conditioning.py`](../tests/test_action_label_word_conditioning.py) | modifier dropout 只清 modifier 通道、eval 不丢、两槽独立 |

`docs/action_balanced_sampling.md` 已作废删除。
