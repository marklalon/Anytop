# 强化 action label 跨物种迁移：canonical 坐标系改造

> 状态：**已实施（2026-08-31）。§5 的 1/2/3/5/6 全部完成，cond 已重生成并通过 §6.1 硬判据；
> 只差步骤 4 的重训。§4.3 的 rest 几何清理做成了预处理的一步（幂等），不是一次性补丁——见 §8 实施记录。**
> 前身是 `action_cond_film_and_energy_removal.md`（**已删除**，内容拆进本文与
> [global_energy_removal.md](global_energy_removal.md)）：那份方案假设"species 通道压制 action
> 通道"，探针 A/B/C 三条预测**全部落空**（Dog 与 Buffalo 的 action 敏感度几乎相等；
> species/action 话语权比在 v1@200k 上反而降到 0.27–0.86；跨物种 action 方向余弦 0.30–0.89），
> 结论是 **action 通道没有被压制**，action FiLM 失去证据支撑。该文的分支决策表把下一步指向
> "几何/归一化通路"，本文就是那条支线的排查结果。
>
> 改动范围：**方案 D**（§2）= `canonical_features.py` + `regenerate_dataset_artifacts.py` +
> `merge_dataset_cond.py`，**cond-only，不动模型**；**杠杆 2**（§3）= `anytop.py` +
> `tensors.py`。13 维形状、两个 key、`feature_space` 版本号**全部不变**。
> **需要 cond 重生成 + 全量重训。**
> 与 [global_energy_removal.md](global_energy_removal.md) 互相独立，两者都要重训，可合并做。
>
> 测量基线：`save/merged_locomotion_v2/cond.npy`（260 物种）+ 三个 processed 数据集
> （truebones/zoo 1106、zoo_upgrade 265、unitybundles 2657 clip）。**全部是纯数据侧测量，
> 无模型参与、无采样、无渲染。**

---

## 1. 诊断

### 1.1 canonical 空间是 7 个不同的仿射坐标系，而模型不知道自己在哪一个里

`canonical_feature_mean` / `canonical_feature_std` 是**逐 object_subset** 的 13 维向量
（quadruped / biped / multiped / serpentine / aquatic / winged / drifting），
在 [canonical_features.py:228](../data_loaders/truebones/truebones_utils/canonical_features.py#L228)
`_apply_global_stats` 里作为 `(x−mean)/std` 应用。

全仓库 grep：这两个 key 只出现在
[tensors.py:181-192](../data_loaders/tensors.py#L181) 的 collate（喂给训练期 aux-loss 解码），
在 [anytop.py](../model/anytop.py) 和 [motion_transformer.py](../model/motion_transformer.py) 里
**零引用**。模型唯一能反推"当前在哪个坐标系"的线索是 `species_emb`——即
`species_tags` 的 T5 均值池化（[physics_joint_annotation.py:846](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L846)
`build_species_embedding_text`），首 tag 才是 object_subset，被稀释在 3 个 token 的 mean-pool 里，
而且 `species_cfg_drop_prob=0.15` 会在 timestep FiLM 路径上把它丢掉。

**永远不该 drop 掉定义输出坐标系的信息。**

坐标系之间的增益差（`std_B / std_A`，1.00 = 同一个系）：

| A → B | pos_x | pos_y | pos_z | vel_x | vel_y | vel_z |
|---|---|---|---|---|---|---|
| quadruped → winged | 1.55 | **1.95** | 0.95 | 2.05 | 2.00 | 1.59 |
| biped → quadruped | 0.55 | 0.88 | 1.26 | 0.72 | 0.69 | 0.68 |
| aquatic → quadruped | 0.68 | **0.26** | 0.64 | 0.32 | 0.34 | 0.72 |
| drifting → quadruped | 0.79 | **0.33** | 0.72 | 0.25 | 0.15 | 0.25 |

`pos_y` 在每一张表里都是最差的一列。这并非偶然：
`clamp_vertical_trajectory`（[animation_utils.py:703](../data_loaders/truebones/truebones_utils/animation_utils.py#L703)）
本身就是**逐 object_subset 的垂直方向非线性变换**——winged 把正向超出
`[0.3, 0.5]×body_length` 的部分压缩回带内，aquatic 再对负向做一次同样的事，
其余 subset 只保留 `−0.5` 的绝对地板。所有非 quadruped 的 swim 演示，
垂直分量都经过了一次 quadruped clip 从未经历的压缩。

### 1.2 换错坐标系的代价：骨长 RMS 误差 10–62%

固定骨架、固定 canonical 张量，**只换那 13 维归一化向量**再解码，量骨长相对自身 rest 的
RMS 偏差（%）。每格 12 个 clip 取中位数：

```
clips of      | quadruped     biped    winged  multiped  drifting serpentin   aquatic
quadruped     |      6.5*      28.0      23.8       9.8      39.9      18.0      61.6
biped         |      15.0      0.9*      20.5      19.2      58.6      26.6      86.3
winged        |      13.2      11.5      1.7*      16.0      20.0      20.4      38.0
multiped      |       2.2      13.9      16.6      0.0*      26.7       5.0      42.7
drifting      |       7.3       9.4      10.2       5.5      0.0*       6.0       4.3
serpentine    |      16.0      30.9      50.8      15.5      98.6      3.3*     140.9
aquatic       |      10.4      19.4      14.0      10.9      13.1      11.6     11.9*
```

对 swim 这条线具体读：locomotion 组的 59 个 swim clip 里 winged 占 25、biped 14、aquatic 11、
quadruped 只有 9。这三个 subset 的 canonical 模式解码到 quadruped 系，分别是
**13.2 / 15.0 / 10.4 %**。

### 1.3 Ground truth 是刚性的，所以上面这些误差全部归因于归一化

全库 252 物种（每物种取 clip）扫描 position 通道自身的骨长偏差中位数：

| subset | vs rest（%） | vs 自身首帧（%） |
|---|---|---|
| biped | 0.2 | 0.0 |
| quadruped | 1.3 | 0.8 |
| winged | 0.3 | 0.0 |
| drifting / multiped / serpentine / aquatic | 0.0 | 0.0 |

**训练目标本身是刚性的**（0.0–1.3%）。§1.2 里 10–62% 的偏差不含任何标注噪声成分。

### 1.4 v3 collapse 了关节轴，但没 collapse xyz / 6d 轴——相对上游 AnyTop 的回归

`edb3cf3`（"replace per-species mean/std with canonical feature space (v3)"）删掉的旧
`get_mean_std` 里明确写着：

```python
Std[1:, :3]   = Std[1:, :3].mean()    # pos block  -> 一个标量
Std[1:, 3:9]  = Std[1:, 3:9].mean()   # rot block  -> 一个标量
Std[1:, 9:12] = Std[1:, 9:12].mean()  # vel block  -> 一个标量
```

v3 的 13 维向量只在**关节维**做了 pool，块内保留逐通道 std。实测块内 max/min：

| subset | pos (x, y, z) | pos 比值 | rot6d | vel |
|---|---|---|---|---|
| quadruped | 0.468 / 0.720 / 0.824 | 1.76 | 2.98 | 1.57 |
| biped | 0.846 / 0.820 / 0.656 | 1.29 | 1.56 | 1.62 |
| winged | 0.727 / 1.406 / 0.785 | 1.93 | 1.85 | 1.53 |
| multiped | 0.470 / 0.740 / 0.978 | 2.08 | 1.70 | 1.94 |
| drifting | 0.596 / 2.156 / 1.152 | **3.62** | 1.56 | 2.63 |
| serpentine | 0.535 / 0.558 / 1.140 | 2.13 | 4.00 | 2.21 |
| aquatic | 0.687 / 2.810 / 1.294 | **4.09** | 1.55 | 2.24 |

`l_simple` 在 canonical 空间计算，所以 `1/std` 就是逐通道的 loss 权重：
**垂直方向的物理误差被系统性少罚 1.3–4.1 倍。**
[gaussian_diffusion.py:481](../diffusion/gaussian_diffusion.py#L481) 的注释还写着
"`l_simple` weights every joint/axis uniformly in standardized space"——
关节维成立，轴维已经不成立了，实施时一并更正。

垂直少罚正是历史上 no_mean_std 那次回归的签名（"crouch + bone-inflate"）。

---

## 2. 方案 D：position gain 全局共享

### 2.1 归零是结构性的，不是调参调出来的

解码顺序是 `x·std + mean` → `×L` → `+rest`
（[canonical_features.py:334](../data_loaders/truebones/truebones_utils/canonical_features.py#L334)）。
`mean` 是逐通道的，在**每个关节、每一帧上完全相同**，所以 mean 错配 =
整副骨架被平移一个常量向量 = **刚性平移，骨长恒等不变**。
能造成形变的只有 `std`（增益）。

于是：**只要 position 块的增益跨 subset 共享，坐标系错配就不可能产生任何形变**，
最坏情况退化为整体位置偏移。骨长只由 position 通道决定
（导出路径 [features.py:1007](../data_loaders/truebones/truebones_utils/features.py#L1007)
`recover_animation_from_motion_np`：先按 rotation 做 FK，再用 RIC position 通道覆盖关节摆放），
所以 rot / vel 的增益**可以**继续保持逐 subset。

### 2.2 四个变体实测对照（同一批 clip，同一套换系流程）

| 方案 | 换系后骨长误差 中位 / p90 / max | 代价 |
|---|---|---|
| **A. 现状** | 16.3 / 57.8 / 140.9 % | — |
| **B.** 块内 collapse 成标量（恢复 v3 前不变量） | 10.5 / 41.0 / 91.7 % | 无 |
| **C.** 逐 subset mean + **全局共享**块标量 std | **1.7 / 11.9 / 11.9 %（= 底噪，增量为 0）** | pos/rot/vel 全部落在 0.58–2.89× 单位方差 |
| **D.** 只共享 **position** 块 gain，rot/vel 保持逐 subset | **1.7 / 11.9 / 11.9 %（增量为 0）** | 只有 pos 受影响；rot/vel 严格单位方差 |

C 与 D 的矩阵是**逐行拉平**的（每个 off-diagonal 格子精确等于对角线），
这就是 §2.1 那条论证的实证。**D 用最小代价拿到 C 的全部收益。**

单独做 B 只能把误差降 ~35%，因为增益差仍在：残差-from-rest 被整体缩放不是刚性变换。
**B 不是一个可以停下来的中间态，它是 D 的一部分。**

### 2.3 实施

统计量的产出点只有两处，都在 `finalize_lnorm_stats` 之后、
`set_canonical_global_stats` 之前插一步：

- [regenerate_dataset_artifacts.py:225](../tools/regenerate_dataset_artifacts.py#L225)
  （`_compute_canonical_stats_per_object_subset`，[167](../tools/regenerate_dataset_artifacts.py#L167)）
- [merge_dataset_cond.py:180](../tools/merge_dataset_cond.py#L180)

新增一个纯函数放在
[canonical_features.py](../data_loaders/truebones/truebones_utils/canonical_features.py)
（`finalize_lnorm_stats` 旁边，[396](../data_loaders/truebones/truebones_utils/canonical_features.py#L396)）：

```python
def collapse_stat_blocks(subset_stats):
    """把每个 subset 的 std 在块内塌成标量；position 块再跨 subset 共享成一个常量。

    subset_stats: {subset: (mean13, std13)} -> 同形状。mean 不动（错配只造成刚性平移）。
    """
```

- 块定义与 `_POS_SLICE` / `_ROT_SLICE` / `_VEL_SLICE` 复用同一组常量
  （[canonical_features.py:60-63](../data_loaders/truebones/truebones_utils/canonical_features.py#L60)）。
- 共享 position gain 取各 subset 块标量的**几何平均**（乘性量取几何平均，
  避免被 aquatic 的 2.81 拉走）。
- `contact`（index 12）不属于任何块，保持逐 subset；aquatic / serpentine 的 contact 恒为 0，
  会继续走 `_STD_FLOOR`（[57](../data_loaders/truebones/truebones_utils/canonical_features.py#L57)）
  floor 到 1.0，行为不变。
- `set_canonical_global_stats` / `get_canonical_global_stats` / 编解码 / collate /
  `_view_stat_*` **一律不改**：13 维形状、两个 key、`feature_space=canonical_motion_v3`
  全部不变，编解码契约不动。

**不需要**改 `_inherit_canonical_stats_from_dataset`
（[dataset_pipeline.py:907](../data_loaders/truebones/truebones_utils/dataset_pipeline.py#L907)）——见 §2.5。

### 2.4 代价：扩散 SNR，不是先验

实测（真实 clip，量的是模型实际看到的空间）：

| subset | A 现状 pos / rot / vel | D pos / rot / vel |
|---|---|---|
| quadruped | 0.98 / 1.02 / 1.04 | **0.73** / 1.09 / 1.07 |
| biped | 0.94 / 0.95 / 0.88 | 0.78 / 0.95 / 0.92 |
| winged | 1.02 / 1.05 / 1.14 | 1.17 / 1.06 / 1.17 |
| multiped | 0.92 / 1.00 / 1.12 | **0.71** / 1.01 / 1.19 |
| drifting | 1.04 / 1.01 / 0.88 | **1.81** / 1.02 / 0.98 |
| serpentine | 1.01 / 1.00 / 0.79 | 0.91 / 1.15 / 0.77 |
| aquatic | 1.05 / 1.10 / 0.94 | **2.00** / 1.11 / 0.92 |

position 通道的单位方差从 1.14× 散布放宽到 **2.8×**（0.71–2.00）；
**rot / vel 完全不变，mean 精确为零**（D 不动 mean）。

与历史上 no_mean_std 那次翻车对照：那次是 rot 6d 带 ~1.9 的 DC 而 motion-std 只有 0.27
（SNR 差 ~3 倍）、position 过度缩放且各向异性（std 1.4–2.8、pos_z DC +1.2）。
**D 没有任何 DC 问题，rot/vel 标定分毫不动，只有 pos 的方差散布变宽**——量级小一个档。

这个代价的本质，是把"这个体型动多大"这条知识从**预处理查找表**挪进了**受条件约束的模型**。
对留出物种而言这是改善：查找表只能按 bucket 查，模型可以从骨架 / species 描述符插值。

### 2.5 为什么 D 不是重新引入"动作先验"

当年 Direction 2（**逐物种**统计）被否掉，理由是它要求新物种自带动作统计。
D 是沿同一条轴**往回走**，不是往前走：

**（a）新物种在推理时需要什么，一个字都没改。**
[dataset_pipeline.py:907](../data_loaders/truebones/truebones_utils/dataset_pipeline.py#L907)
`_inherit_canonical_stats_from_dataset` 要的是：① `species_tags.jsonl` 里有 object_subset 标签；
② checkpoint 的 cond.npy 里存在**任意一个同 subset 的兄弟物种**。
它**不碰新物种自己的 clip**（rest-pose-only 的 `process_new_skeleton` 路径就是为此写的）。
D 只改 `regenerate_dataset_artifacts` 往表里写什么数。

**（b）表里的动作先验含量下降了。**

| | 现状 A | 方案 D |
|---|---|---|
| mean（逐 subset，动作导出） | 13 | 13 |
| std（逐 subset，动作导出） | 13 | 3（rot / vel / contact 各一标量） |
| **每 subset 合计** | **26** | **16** |
| 全局共享常量 | 0 | 1（position gain） |

**（c）"新体型无法生成"这个失败模式反而变软。**
今天 [dataset_pipeline.py:1003-1032](../data_loaders/truebones/truebones_utils/dataset_pipeline.py#L1003)
是**硬 fast-fail**：找不到同 subset donor 就 `ValueError`，理由是跨 subset 借统计量 = OOD。
一个全新体型（比如第一个真正的水生四足）今天直接跑不起来。
D 之后这条禁令的依据消失大半——跨 subset 借的代价被拆成两块，且都不含形变：
借 **mean** = 刚性平移（§2.1 已证），借 **rot/vel gain** = 幅度偏差。
**实施 D 时应同步把这条 fast-fail 降级为 warning**，这正是"新物种能不能生成"的场景。

---

## 3. 杠杆 2：把坐标系喂给模型

D 之后每个 subset 的坐标系只剩 `mean13 + rot_gain + vel_gain + contact_gain`，
position gain 已是全局常量——要喂的东西比现在少，且不再包含任何能造成形变的量。

- 数据侧：这两个 key 已经被 collate 到 `y` 里
  （[tensors.py:188](../data_loaders/tensors.py#L188)，`[B, 13]` 逐样本），**无需新增管线**。
- 模型侧：在 [anytop.py](../model/anytop.py) 里加一个小 projection，
  照 `_apply_species_film`（[anytop.py:707](../model/anytop.py#L707)）的形式接进 `timesteps_emb`。
  **零初始化**，起点恒等。
- **不要给它 CFG drop。** 它不是一个语义条件，是输出坐标系的定义；
  丢掉它等于让模型去猜自己在往哪个空间里写数。这也是现状的病根之一（§1.1）。

**D 与杠杆 2 的关系**：

- 只做 D → 形变结构性归零，pos 标定松到 2.8×。
- 只做杠杆 2 → 标定完全保住，形变从"结构性"降级为"可学"。但 swim 只在 7 个 subset 里的
  2 个有 ≥3 物种（§4.2），"可学"的监督信号非常薄。
- **D + 杠杆 2** → 两头都要，而且 D 之后要喂的东西反而更少。推荐这条。

---

## 4. 其余杠杆（独立于 D，可单独做）

### 4.1 `lambda_bone` 在 v1 / v2 都是 0.0

两个 `args.json` 均为 `lambda_bone = 0.0`。它是唯一直接约束这条几何通路的 loss
（`bone_length_consistency_loss`，[gaussian_diffusion.py:475](../diffusion/gaussian_diffusion.py#L475)），
而它一直关着。D 与它互补：D 消掉系统性剪切，bone loss 兜住残差。
D 之后重训时应当同时打开（它本身是 target-relative 的，不会与真实的骨长形变对抗）。

> **实际决定（2026-08-31）：v3 首训不开 `--lambda_bone`。** 先让 D + 杠杆 2 单独跑，
> 免得两个变量一起动、分不清残差是谁的。它仍然是下一个候选杠杆——§6.3 的 bone-length drift
> 量出来之后再决定要不要 0.1–0.3。

### 4.2 `GROUP_MULTIHOT_MASK` 的门槛要加 object_subset 轴

现规则 `clips >= 10 AND species >= 5`
（[motion_labels.py:116](../data_loaders/truebones/truebones_utils/motion_labels.py#L116)）
没有体型轴。locomotion 组实测（"subs" = 有 ≥3 个物种携带该词的 subset 数）：

| 词 | clips | 物种 | **subs** | 最集中的 subset（clips/species） |
|---|---|---|---|---|
| walk | 380 | 168 | **5** | winged 3.9 |
| run | 326 | 172 | **5** | winged 3.4 |
| eat / retreat | 53 / 52 | 31 / 30 | **5** | biped 2.3 |
| turn | 57 | 16 | 3 | biped 5.0 |
| jump | 16 | 14 | 3 | winged 1.7 |
| **fly** | 189 | 70 | **2** | winged 4.0 |
| **swim** | 59 | 11 | **2** | **winged 12.5** / biped 7.0 |
| **crawl** | 31 | 14 | **2** | biped 5.0 |

`swim` 全局看（59 clips / 11 物种）轻松过线，但它只在 quadruped 和 aquatic 上有 ≥3 物种，
且 winged 上是 **12.5 clips/species**——这个比值就是记忆化指纹（walk/run 是 1.7–3.9）。
建议补一条：**在该 group 内，至少 3 个 object_subset 各有 ≥3 个物种**。
`fly` 同病（77 物种但 quadruped 仅 1 clip）。

对应地，数据补齐（retarget 增广）应当**按缺失的 subset 定向**，不是按物种数——
把 9 个四足 swim clip 迁到 15–20 个四足骨架上，目的是把 swim 的 `subs` 从 2 抬到 4–5。

### 4.3 rest 姿势与 clip 的骨长对不上：48 / 252 物种

用同一套骨长度量扫全库，>10% RMS 的有 48 个物种。按"clip 自身是否刚性"可以干净地二分：

| 判定 | 数量 | 典型（括号内是 clip 自身的非刚性度） |
|---|---|---|
| **rest 对不上 clip**（clip 自身刚性） | ~30 | Pirrana 200.9%(0.0)、Dog 37.1%(5.0)、Dog-2 37.3%(3.9)、Horse 33.5%(0.1)、Deer_Buck 30.3%(1.9)、Buffalo 11.2%(0.9) |
| **clip 自身非刚性** | ~18 | HermitCrab 44.6%(44.6)、MU06_Archangel 25.9%(25.9)、Tukan 25.6%(25.6) |

逐骨看第一类的元凶：Dog / Dog-2 是 `Bip01_Ponytail3Nub`、`_Bip01_Head11_TungeControler`、
`Bip01_Head2_Eyeleds`、`Bip01_Head1_Jaw` 这类 **nub / locator / 表情控制骨**，
在动画里长度塌成 0（比值 x0.00）而 rest 里带非零 offset；Pirrana 是 `locator`（x9.98）。

范围有限（不是身体骨），但这些关节**是模型必须预测的 token**，而 position 通道的原点就是
rest——等于给每个物种塞了一批必须死记的 per-joint DC 常量。
注意 Dog 是仅有的 3 个提供 swim 的四足之一，Buffalo 也在表内。

> 已知的例外：13 个 unitybundles 物种（MLH_* / MLS_* / RMW_*）的偏差来自
> `rest_pos_ric_hml` 仍把手持道具停在 ±3，而每个 clip 里道具都在手上——
> 这是已记录的行为（scale 统计已经过滤，rest 几何没有），不是新问题。

### 4.4 contact 是 aquatic / serpentine 的死通道

这两个 subset 的 contact 恒为 0，`_STD_FLOOR` 把 std floor 到 1.0，
于是 canonical 空间里该通道恒等于 0。**11 个 aquatic swim clip 在 contact 通道上不携带任何信息。**
模型只能从 winged / biped 的 swim clip 学"游泳时脚不着地"。
这不是 bug（floor 的行为是对的），但在评估 swim 迁移时要记得这一条：
四足游泳需要的 `contact ≈ (0−0.219)/0.413 = −0.53` 这个目标，
只有 39 个 clip（winged 25 + biped 14）真正演示过。

### 4.5 `L` 的性质，作为背景

`_length_scale_from_rest`（[canonical_features.py:92](../data_loaders/truebones/truebones_utils/canonical_features.py#L92)）
是 rest 关节位置的 **RMS 展布**，它作用在已经被 `scale_factor` 归一到
`HML_REF_MAX_SPAN` 的骨架上。所以 `L` 实际衡量的是"紧凑度"，且因为是关节维的**无权 RMS**，
它对**绑定的关节密度**敏感。实测跨库 12.9×（0.031 `MU04_Pollen` → 0.394 `Cobra`），
quadruped 内部 2.6×。

但按 subset 量真实 clip 在 canonical 空间的幅度，逐物种 pos-rms 的 p10/p50/p90 是
0.53 / 0.81 / 1.24（quadruped），**没有失控**。
所以 `L` 记在这里是**背景，不是待办**——除非 D 之后仍有残留的跨物种幅度问题，
才回来动它（那时它是单变量）。

---

## 5. 实施顺序

> 执行状态见 [§8 实施记录](#8-实施记录20260831)：1/2/3/5/6 已完成（2 通过硬判据），
> 4（重训）未做。

1. **§2.3 的 `collapse_stat_blocks`** + 两处调用点 + fast-fail 降级（§2.5c）+
   [gaussian_diffusion.py:481](../diffusion/gaussian_diffusion.py#L481) 注释更正。
2. `regenerate_dataset_artifacts` 重生成 cond.npy。**用 §1.2 的换系脚本自检**：
   新 cond 的 off-diagonal 应当逐行拉平到对角线（中位 ~1.7%）。这一步不需要模型，
   是一个可以在重训前就确认改动生效的硬判据。
3. **§3 的坐标系条件化**（零初始化、无 CFG drop）。
4. 重训。（原计划同时打开 `--lambda_bone`；**首训改为不开**，见 §4.1 的决定。）若与
   [global_energy_removal.md](global_energy_removal.md) 合并做，判据不冲突。
5. **§4.2 的 mask 规则**与数据补齐——这条独立于重训，可以并行推进。
   （mask 规则已落地，数据补齐仍未做——它是采数据，不是改代码。）
6. §4.3 的 rest 几何清理——独立，但它会改变 cond，做了就要跟着重生成一次。
   （已做成 `regenerate_dataset_artifacts` 的一步，跑在统计量之前，所以"跟着重生成一次"
   这条不再需要——同一趟里就一致了。见 §8.5。）

---

## 6. 验证

### 6.1 重训前（无模型）

- §1.2 的换系矩阵拉平（步骤 2 自检）。
- §2.4 的逐 subset 真实幅度落在预期区间（pos 0.71–2.00，rot/vel 不变）。

### 6.2 需要模型侧确认的一件事

上面证明了"坐标系错配**足以**造成 10–15% 骨长误差"，但没有证明 v2 推理时**确实**发生了泄漏。
最便宜的验证是写一个**探针 E**（A/B/C 探针脚本 `tools/probe_action_cond.py` 已随诊断结论删除；
E 量的是 x₀ 幅度而非条件敏感度，届时单独写一个即可）：
单次前向预测 x₀，量 (Dog, walk) / (Dog, swim) / (Buffalo, walk) / (Buffalo, swim)
四组输出在 position 块上的逐通道 std。
若 swim 相对 walk 的幅度比接近 winged/quadruped 的增益比（1.55–1.95），就是坐标系泄漏的指纹。
与 A/B/C 同构——单次前向，几乎免费。

另一个旁证：导出侧已有
`recover_animation_from_motion_np(..., rigid_bone=True)`
（[features.py:1007](../data_loaders/truebones/truebones_utils/features.py#L1007)），纯 FK、骨长绝对刚性。
它不修根因（姿态还是错的，只是刚性地错），但如果 Buffalo swim 在 `rigid_bone` 下从
"完全变形"变成"姿势不对但身体完整"，就旁证了形变确实是从 position 通道进来的。

### 6.3 重训后

几何侧用 [eval/motion_quality/scorer.py](../eval/motion_quality/scorer.py) 的 bone-length drift，
与 v2（若同时删了 energy，则走 energy-off 口子）对照。需要采样出 NPY，
但不需要 BVH 导出、不需要渲染、不需要人看。

---

## 7. 期望要放对

**没有任何归一化改动能凭空造出只有 3 个四足演示过的动作。**
D + 杠杆 2 拆掉的是"坐标系错配导致的形变"这一层，Buffalo 的 swim 仍然只能靠
从 Dog / Dog-2 / SabreToothTiger 的 9 个 clip 迁移。若这 9 个 clip 本身不足以定义四足泳姿，
结果会从"完全变形"改善为"可辨认但不对"，**不会直接变成好动作**。
真正的补齐是数据（§4.2 末）。本方案的画像是"移除结构性形变"，不是"能力解锁"。

同样，**loss 不能作为判据**：特征空间变了（position 通道的尺度改了），
`l_simple` 的绝对值在两个空间之间没有可比性。判据是 §6 的三组。

---

## 8. 实施记录（2026-08-31）

除重训（§5 步骤 4）外，§5 的每一步都已落地。下面记录改了什么、量出了什么，
以及与本文原始估计不一致的地方。

### 8.1 §5 步骤 1 — `collapse_stat_blocks` 与两处调用点

- [canonical_features.py](../data_loaders/truebones/truebones_utils/canonical_features.py)
  新增 `collapse_stat_blocks(subset_stats)` 与私有 `_block_std_scalar`：块内取
  **算术平均**（复刻 v3 前 `get_mean_std` 的 `.mean()`），position 块标量再跨 subset 取
  **几何平均**。`mean` 一个字节不动；`contact` 不属于任何块，逐 subset 原样保留（aquatic /
  serpentine 的 0 继续走 `_STD_FLOOR` → 1.0）。退化块（整块常量）返回 `None` 并原样跳过。
- 两处调用点都在 `finalize_lnorm_stats` 之后、`set_canonical_global_stats` 之前，如 §2.3 所写：
  `regenerate_dataset_artifacts._compute_canonical_stats_per_object_subset` 与
  `merge_dataset_cond._recompute_canonical_stats`。
- §2.5c 的 fast-fail 降级：`dataset_pipeline._merge_object_into_cond` 里"找不到同 subset donor"
  由 `ValueError` 改为 **warning + 跨 subset 借用**（warning 里点名 donor 与它的 subset）。
  "cond 里没有任何物种带统计量"仍然是硬错误——那时根本没有空间可落。
  按 §2.3 的判断，`_inherit_canonical_stats_from_dataset` **没有改**。
- `gaussian_diffusion.bone_length_consistency_loss` 的注释按 §1.4 更正：uniform 现在关节维、
  轴维都成立，原句只在关节维成立的历史背景写进注释里。
- 13 维形状、两个 key、`feature_space=canonical_motion_v3` 全部未变，编解码契约未动。
- 测试：`tests/test_canonical_features.py` 新增 3 例（块拉平 + 共享 gain 的数值定义、
  **换系后骨长逐比特相等**、退化块容错）。

### 8.2 §5 步骤 2 — cond 重生成 + §6.1 硬判据

三个 processed 数据集各跑一次 `regenerate_dataset_artifacts`，再跑
`merge_dataset_cond` → `dataset/merged/cond.npy`（260 物种 / 4028 clip）。
先只带本节改动跑了一遍，逐 key 比对旧 cond：**只有 `canonical_feature_std` 变了**，
其余字段与 `motion_metadata.json` / `metadata.txt` 逐字节相同，重生成是幂等的。
（最终这一版还带上了 §8.5 的 rest re-seat，所以 5 个物种的 `rest_pose` / `offsets` /
`rest_pos_ric_hml` 也变了；下面表里"改动后"的数字是最终版。`motions/*.npy` 一个字节没动——
存的是 physical 特征，rest 相减在加载时做。）
`validate_anytop_dataset --datasets` 三个数据集全 PASS。

自检脚本是新增的 [tools/check_canonical_frame_swap.py](../tools/check_canonical_frame_swap.py)
（§1.2 的换系流程 + §2.4 的幅度表，纯数据侧，无模型）。同一批 clip、同一套流程，
改动前后：

```
改动前（每格 12 clip 取中位数，%）
                   aquatic       biped    drifting    multiped   quadruped  serpentine      winged
aquatic               0.0*        7.9         3.5         6.3         5.7         7.1         5.6
biped                72.0         0.7*       47.1        13.9         9.2        20.2        16.0
drifting             10.6        11.6         0.6*        9.0         8.8        10.3         9.6
multiped             82.9        15.6        52.6         0.6*        3.9         8.4        25.7
quadruped            68.1        18.0        42.7         5.5         1.6*       12.6        23.4
serpentine          126.7        29.5        86.1        11.1        12.6         1.8*       49.0
winged               50.9        14.6        28.7        18.6        14.7        25.3         2.7*
→ off-diagonal 相对自身对角线的超出量：中位 +12.62 / p90 +64.99 / max +124.84 个百分点

改动后
aquatic               0.0*        0.0         0.0         0.0         0.0         0.0         0.0
biped                 0.7         0.7*        0.7         0.7         0.7         0.7         0.7
drifting              0.4         0.4         0.4*        0.4         0.5         0.4         0.4
multiped              0.6         0.6         0.6         0.6*        0.6         0.6         0.6
quadruped             1.6         1.6         1.6         1.6         1.6*        1.6         1.6
serpentine            1.8         1.8         1.8         1.8         1.8         1.8*        1.8
winged                2.7         2.7         2.7         2.7         2.7         2.7         2.7*
→ 超出量：中位 +0.00 / p90 +0.01 / max +0.03 个百分点
```

**逐行完全拉平，判据通过。** 共享 position gain = **0.9221**（七个 subset 同值且各向同性）。
对角线上剩下的 0.0–2.7% 是 clip 相对自身 rest 的固有非刚性，即 §4.3 那件独立的事，
本改动不碰它。

§2.4 的代价（12 clip/subset 采样，1.0 = 标定目标）：

| subset | 改动前 pos / rot / vel | 改动后 pos / rot / vel |
|---|---|---|
| aquatic | 0.96 / 1.05 / 0.96 | **1.42** / 1.03 / 0.85 |
| biped | 0.72 / 0.85 / 0.72 | 0.60 / 0.83 / 0.75 |
| drifting | 1.05 / 1.01 / 1.31 | **1.78** / 1.00 / 1.56 |
| multiped | 0.98 / 1.08 / 1.07 | 0.78 / 1.08 / 1.03 |
| quadruped | 0.77 / 1.24 / 1.16 | **0.57** / 1.20 / 1.17 |
| serpentine | 0.97 / 1.20 / 1.29 | 0.82 / 1.18 / 1.36 |
| winged | 0.92 / 1.12 / 1.63 | 0.98 / 1.11 / 1.62 |

pos 散布 0.57–1.78（≈3.1×），rot / vel 实质不变——与 §2.4 的预测（0.71–2.00）同量级。
注意 §2.2 的 D 行写"rot/vel 严格单位方差"，而 §2.3 的函数定义要求**每个块都塌成标量**；
按 §2.3（规范性的那一条）实施，所以 rot/vel 也做了块内拉平，表里它们因此有 ±0.02 的位移，
这是块内各向异性被消掉的结果，不是回归。

### 8.3 §5 步骤 3 — 杠杆 2：把坐标系喂给模型

[anytop.py](../model/anytop.py) 新增 `canonical_frame_projection`：把
`[canonical_feature_mean ‖ canonical_feature_std]`（26 维，已由 collate 逐样本放进 `y`，
数据侧零改动）过一个 `Linear(26,256) → GELU → Linear(256,256)`，**末层零初始化**，
与 playspeed / loop / action 走同一条加性通路加到 timestep token 上。

- **无条件接入，没有 CLI 开关。** 最初落地时是 `--canonical_frame_cond`（默认关），
  后来固化为常开并把 flag 删掉：`canonical_motion_v3` 的每一份 cond.npy 都带这两个 key，
  loader 缺了它们直接 `RuntimeError`（[dataset.py:1196](../data_loaders/truebones/data/dataset.py#L1196)），
  所以"关"唯一的含义就是"对输出坐标系瞎写"——正是本文要修的那个缺陷。留着一个只有错误取值的
  开关，等于把缺陷保留成一个配置项。`args.json` 里因此也不会再出现 `canonical_frame_cond`。
- **没有 CFG drop、没有 keep mask**，`train()` 与 `eval()` 下逐比特相同——它是输出空间的定义，
  不是可以引导的语义条件（§1.1 的病根之一就是它被 drop 掉）。
- 参数 +72,704（16.03M → 16.10M，+0.45%）。
- 端到端验证（真 merged cond → 真 loader → 真 batch）：把未开启版本的 state_dict 装进开启版本，
  两者输出 `max|Δ| = 0.0`（起点精确恒等，只多出 4 个新 key）；把末层权重扰动后
  `max|Δ| = 1.13`（通路真的活着）。
- **生成路径当时漏了。** 上面这条只走了训练 loader；`sample/generate.py` 的
  `create_condition` 自己拼 extras dict，原本刻意不带这两个 key（注释写的是"生成侧解码用整份
  cond entry，y 不需要全局统计量"）——那在 forward 不读它们时成立，现在 forward 每步都读，
  于是 `generate.py` 的两个调用点和走它的 `server/anytop_service.py` 全都会 `ValueError`。
  已修：`create_condition` 从 cond entry 取这两个 key 放进 extras（collate 本来就会逐样本堆叠）。
  注意这**不是常开引入的**：只要用 `--canonical_frame_cond` 训出 v3，旧的 flag 版一样在生成侧炸。
  测试：[test_generate_inpainting.py](../tests/test_generate_inpainting.py) 补了 y 里两个 key 的
  形状断言，以及"cond 缺统计量必须响亮失败"的负例。
- **旧 checkpoint**：`save/` 里没有一个是这条通路之后训的，装载会在
  `load_model` 报 `Missing keys: canonical_frame_projection.*`。它们本来就跨不过这次的
  特征空间变更（且除三个史前 run 外全都先被 `assert_global_energy_not_deprecated` 拦下），
  所以没有为此加豁免——v3 之后训的 checkpoint 一律带这四个 key。
- 手工构造 `y` 的调用方（测试里的 `_make_y`）必须补上这两个 key，forward 每次都读。
- 测试：新增 [tests/test_canonical_frame_cond.py](../tests/test_canonical_frame_cond.py)（8 例）。

### 8.4 §5 步骤 5 — `GROUP_MULTIHOT_MASK` 的 object_subset 轴

规则按 §4.2 落成 `clips >= 10 AND species >= 5 AND subs >= 3`，
`subs` = 携带该词的物种数 ≥3 的 object_subset 个数。重算工具：
[tools/action_multihot_mask_report.py](../tools/action_multihot_mask_report.py)
（三个组全量，4028 clip 对得上三个数据集的 npy 总数）。

**§4.2 表里的 locomotion 数字与实测有出入，实测为准。** 差异来自本文那次测量没有走仓库自己的
`vocab_words_in`：那个函数会丢弃"整个匹配都落在另一个更长匹配里"的词，所以
`retreat` 里的 `eat` 不该点亮 `eat`——而 §4.2 表里 eat 53 / retreat 52 几乎相等，正是漏掉这条
去重的签名。实测（locomotion）：walk 395/175/subs 5、run 344/193/5、fly 218/78/4、
swim 59/11/**2**、turn 88/26/4、jump 30/27/3、crawl 39/20/3、retreat 91/36/5、
fall 15/10/**1**、eat **0**。swim 的结论不变（subs=2，最密的 aquatic 上 14.8 clip/物种，
而 walk/run 是 5.1–5.6）。

新规则关掉 13 个槽位：locomotion 的 `swim` `fall`；stationary 的 `jump` `roar` `fall` `rest`；
transition 的 `run` `fly` `hurt` `rest` `shake` `crouch` `rear`。
`roar` 因此在三个组里全被关掉，两处拿它举例的测试改用 `bite`（stationary 留、locomotion 关）。
**数据补齐（把 9 个四足 swim clip retarget 到 15–20 副四足骨架）不在本次范围内**——它是采数据，
不是改代码；工具已经能在补完后一条命令重算并给出 diff。

### 8.5 §5 步骤 6 — §4.3 rest 几何：做成了预处理的一步

**不是一次性打补丁，而是 pipeline 的一步。** 新增
[rest_geometry.py](../data_loaders/truebones/truebones_utils/rest_geometry.py)，
挂进 `regenerate_dataset_artifacts`（在 canonical 统计量之前），而
`preprocess_and_validate` 本来就会调用它——所以**每次预处理都会自动带上**，数据集更新后
不需要记得再跑一遍任何东西。`--no-rest-reseat` 是复现旧构建用的逃生口。

这也顺带取消了 §5 步骤 6 那句"做了就要跟着重生成一次"：re-seat 与统计量在同一趟里完成，
出来的 rest 和归一化表天然一致。

**改的是 rest 骨头的长度，方向不动。** 长度是这处分歧里唯一旋转不变的量——bone vector 每一帧
都随父关节的朝向转，不存在"clip 里的那个 offset"可以照抄，但"clip 里的那个长度"只有一个。
被动画塌到父关节上的 nub（ratio 0）在 rest 里也塌下去，这本来就是每一帧的实际情况。
`offsets` 与 `rest_pose[:,0:3]` 一起改（实测 `rest_pos_ric_hml == FK(offsets)`，误差 1e-6），
所以两份 rest 表示始终一致。

**四条判据**（`reseat_candidates`）：与 rest 差 >10%、clip 内部刚性 ≤2%、**叶关节**、
**不是道具 socket**。

- *叶关节* 是承重的那条：只有叶关节能只改自己的 `offsets` 而不需要给整棵子树补偿位移。
- *道具 socket* 用的是仓库自己已有的
  [`find_prop_socket_joints`](../data_loaders/truebones/truebones_utils/animation_utils.py#L1046)
  ——它在 260 个物种上标定过，正好命中 14 个 unitybundles 物种的 Bow / Arrow / Sword / Shield
  而在 truebones 上一个都不碰。这些 socket **本来就已经被排除在 scale 统计之外**；把它们的 rest
  挪到手上会改变 rest span，而 rest span 就是 `_length_scale_from_rest` 拿来除每个物种的 `L`
  （§4.5 明确把 L 列为"背景，不是待办"）。同一个判据、同一个理由，不需要新写一条规则，
  也不需要人工挑名单。这正是 §4.3 末尾"已知的例外"那一段要的行为。

一处度量修正：rest 里存在长度 1e-6 ~ 1e-22 的塌陷骨（Tukan `N_ALL`、Crow `Pelvis`、
Dog `EyesBlue`），除以它会把正常 clip 变成 350 万 % 的"误差"。因此按骨架尺度设下限
（`1e-3 × rest span`），Tukan 由此从 3.5M% 回到 140%，与备忘里的量级一致。同一个下限也用来
判断 **clip 长度**够不够长到能谈"相对离散度"——否则一根被稳稳塌在 0 上的 nub 会因为
除以 1e-9 而显示成无穷大的离散度，被判成"非刚性"而正好漏掉。

**实跑结果：13 个关节 / 5 个物种**（全库 4028 clip，不是抽样）：

| 物种 | 关节 | rest → clip |
|---|---|---|
| truebones/zoo/Horse | `Bip01_R_Toe0Nub`、`Bip01_Xtra02/03/04Nub`、`Bip01_R_Finger0Nub` | x0.45–0.88 |
| truebones/zoo_upgrade/Deer_Buck | `SM_Deer_Horns_01..04` | → 0（x0.00） |
| truebones/zoo_upgrade/Rabbit2 | `EarLeft02` / `EarRight02` | x0.45 |
| truebones/zoo_upgrade/serpent_man | `head_tongue_03` | x0.51 |
| unitybundles/MU01_FlowerPotMonster | `RigPetal SE7` | — |

Dog / Dog-2 的 `Bip01_Ponytail3Nub`、`Bip01_Head2_Eyeleds` 在只看 8 个 clip 时是候选，
**看全部 clip 就不是了**——它们在别的 clip 里确实被动画过，所以那个 rest 是合法的中立位，
判据把它们放过了。这是"证据多了结论更保守"，不是漏检。

**幂等性实测**：第二遍跑 `regenerate_dataset_artifacts`，三个数据集都报
"rest geometry already agrees with the clips; nothing re-seated"。
`tools/audit_rest_vs_clip_geometry.py`（只读，与 pipeline 共用同一套判据）现在报
**0 re-seatable**，剩下的 40 个 >10% 物种全是判据主动放过的（内部关节 / 真的被动画 / 道具）。
改完 §6.1 硬判据重跑仍然逐行拉平（超出量 max +0.03 个百分点），三个数据集
`validate_anytop_dataset` 全 PASS。

测试：新增 [tests/test_rest_geometry.py](../tests/test_rest_geometry.py)（7 例：选择判据、
被动画的骨不动、内部关节不动、长度对齐 + FK 一致性、幂等、道具 socket 排除、退化输入容错）。

### 8.6 §5 步骤 4 与 §6.2 —— 未做

- **重训未做**（本次范围外）。`train.bat` 已经备好：`RUN_NAME` 抬到
  `merged_locomotion_v3`（特征空间变了，`--auto_resume` 绝不能把 v2 续进来）。
  坐标系条件不需要写在命令行里（§8.3 已固化为常开），`--lambda_bone` **本轮不开**（§4.1）。
- **§6.2 的探针 E 跑不了**：`save/merged_locomotion_v2/args.json` 里带着
  `global_energy_cond`，`parser_util.assert_global_energy_not_deprecated` 会直接
  `SystemExit`——那个 checkpoint 已经无法加载，不是本次改动造成的。
  探针 E 要等 v3 训出来（那时它量的是"改完之后还漏不漏"，与原设计的"改之前漏不漏"不是同一个问题）。
- §6.3 的重训后几何对照同样要等 v3。
