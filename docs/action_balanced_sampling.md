# 按动作首词平衡采样（`--balanced`）

## 0. 一句话

`--balanced` 的分组键从**物种**改成 **`action_label` 的首个主词**：采样质量先按
`sqrt(该首词的 clip 数)` 分给各首词组，再在组内逐条均分。AnyTop 原来的按物种分组
**已删除**，没有开关可以切回去；sqrt 质量规则、组内均分规则、aux 池预算都沿用原样。

## 1. 为什么换轴

语料在动作轴上的倾斜远大于任何其它轴（`outputs/_action_firstword_counts.txt`，
全量 3635 clip）：

```
attack 891   idle 732   walk 369   run 340   die 293 ...   stop 5   crawl 1   sheathe 1
```

逐条均匀抽样等于把这个比例原样交给模型：`stop` / `sheathe` 在一个 epoch 里出现的
次数比 `attack` 少两到三个数量级，于是推理时给定这些首词，模型输出的东西更像
`idle`——**条件里最该起作用的那个词反而最不起作用**。

物种平衡解决的是另一个问题（拓扑公平），它不会动这个比例：一个物种的 200 条
`attack` 和 1 条 `stop` 在物种内部仍然是 200:1。

## 2. 规则

对每个池（本组 clip / aux clip 各自一池，见
[aux_group_and_head_word_augmentation.md](aux_group_and_head_word_augmentation.md) §4.2）：

```
组 = parse_action_label(label) 的第一个 HEAD_VOCAB 词
组质量 ∝ sqrt(组内 clip 数)，在池内归一化
组内质量 = 组质量 / 组内 clip 数（逐条均分）
```

- **多主词标签算在首词名下**：`"attack, jump, charge"` 进 `attack` 组。首词也正是
  head 通道里被加权在前的那个词（`HEAD_SLOT_PRIMARY_WEIGHT`），两处对"这条 clip
  是关于什么的"的判断保持一致。
- **空标签**单独成组 `<unlabeled>`（当前语料为 0 条）。标签解析失败会**硬失败**并
  报出 clip 名，不会静默归到某个组里。
- **不再平衡物种**：同一首词内部每条 clip 等概率，所以一个物种在该动作上的份额 =
  它贡献的 clip 比例。这是刻意的——换掉的就是"物种公平"这个轴。
- 组的遍历顺序固定（`HEAD_VOCAB` 顺序），权重逐位可复现。
- 计数只统计**已经过 split / `--action_group` 过滤后的 `name_list`**，所以每个
  训练组看到的是它自己的分布（见下表）。
- `--balanced` 不开时一切照旧：整池一组、逐条均分，**完全不读标签**（坏标签不会让
  非平衡的 run 挂掉），`aux_group_mass` 仍按原样生效。

## 3. 实测（三个训练组，全量 sidecar）

`x/clip` = 该组每条 clip 被抽到的概率相对逐条均匀采样的倍数。

### locomotion（937 clip / 7 首词）

| head | clips | uniform | balanced | x/clip |
|---|---:|---:|---:|---:|
| walk | 369 | 39.4% | 28.7% | 0.73 |
| run | 340 | 36.3% | 27.5% | 0.76 |
| fly | 102 | 10.9% | 15.1% | 1.39 |
| swim | 59 | 6.3% | 11.5% | 1.82 |
| hover | 57 | 6.1% | 11.3% | 1.85 |
| roll | 9 | 1.0% | 4.5% | 4.66 |
| crawl | 1 | 0.1% | 1.5% | 13.99 |

### stationary（1924 clip / 9 首词）

| head | clips | uniform | balanced | x/clip |
|---|---:|---:|---:|---:|
| attack | 891 | 46.3% | 32.8% | 0.71 |
| idle | 732 | 38.0% | 29.8% | 0.78 |
| hurt | 214 | 11.1% | 16.1% | 1.45 |
| work | 41 | 2.1% | 7.0% | 3.30 |
| rest | 30 | 1.6% | 6.0% | 3.86 |
| pickup | 7 | 0.4% | 2.9% | 8.00 |
| putdown | 6 | 0.3% | 2.7% | 8.64 |
| lift | 2 | 0.1% | 1.6% | 14.96 |
| draw | 1 | 0.1% | 1.1% | 21.16 |

### transition（774 clip / 16 首词）

| head | clips | uniform | balanced | x/clip |
|---|---:|---:|---:|---:|
| die | 293 | 37.9% | 20.6% | 0.55 |
| turn | 200 | 25.8% | 17.0% | 0.66 |
| jump | 71 | 9.2% | 10.2% | 1.11 |
| spawn | 70 | 9.0% | 10.1% | 1.12 |
| getup | 60 | 7.8% | 9.3% | 1.20 |
| land | 21 | 2.7% | 5.5% | 2.04 |
| laydown | 16 | 2.1% | 4.8% | 2.33 |
| takeoff | 9 | 1.2% | 3.6% | 3.11 |
| burrow | 9 | 1.2% | 3.6% | 3.11 |
| sitdown | 7 | 0.9% | 3.2% | 3.53 |
| draw | 6 | 0.8% | 3.0% | 3.81 |
| stop | 5 | 0.6% | 2.7% | 4.17 |
| rear | 2 | 0.3% | 1.7% | 6.60 |
| kneel | 2 | 0.3% | 1.7% | 6.60 |
| fall | 2 | 0.3% | 1.7% | 6.60 |
| sheathe | 1 | 0.1% | 1.2% | 9.33 |

sqrt 而不是全平衡：全平衡会把 `stationary` 的那 1 条 `draw` 抬到 11.1% 的采样质量
（每 9 步就重复它一次），sqrt 给 1.1%。同一条 clip 被反复抽到仍然是这个模式的主要
风险，上表最后两三行就是要盯的地方——如果这些首词开始过拟合到单条 clip 的具体姿态，
应当先补数据，其次才是调分组指数。

## 4. 用法与影响

```
train/train_anytop.py ... --balanced     # 按 action_label 首词分组
```

启用后训练日志里会打印一行实际分布：

```
[sampler] --balanced by action head word (9 groups): attack 891->32.8%, idle 732->29.8%, ...
```

- **数据分布变了 = 需要重训**，不要在旧 run 上 `--auto_resume` 直接续：同一个
  `RUN_NAME` 续训会让前后两段看到不同的采样分布。换新的 `RUN_NAME`。
- cond.npy / sidecar **不需要重新生成**，这纯粹是采样器侧的改动。
- 三个 `train_*.bat` 目前都没有传 `--balanced`，所以在脚本里加上这一行之前，改动是
  不生效的。

## 5. 代码位置

| 位置 | 作用 |
|---|---|
| [`data_loaders/truebones/data/dataset.py`](../data_loaders/truebones/data/dataset.py) | `clip_action_head_word()`、`TruebonesSampler` 的分组与质量计算 |
| [`utils/parser_util.py`](../utils/parser_util.py) | `--balanced` |
| [`tests/test_action_balanced_sampler.py`](../tests/test_action_balanced_sampler.py) | sqrt 质量、首词归组、物种不参与平衡、aux 预算不变 |
