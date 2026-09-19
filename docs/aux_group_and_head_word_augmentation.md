# 跨组训练补充（aux_action_groups）与主词增广（head-word augmentation）

> 状态：**§9 步骤 1–8 已实施，待重训**（2026-09-19，分支 `train/v21`）。
> 三份 `action_labels.jsonl` 已完成一次性迁移（`.bak` 保留）；
> **本轮落规则 J + F，规则 L 已屏蔽**（用户 2026-09-19，见 §7）：aux 池 **68** 条
> （J 的 44 条 `jump` + F 的 24 条 `fall`），全部取值 `["transition"]`；
> stationary / locomotion 两组的 aux 池因此都是空的。
> `train_transition.bat` 已改名 `merged_transition_v21` 并带上四个新旗标
> （`--aux_group_mass` 随 aux 池大小重算，0.25 → **0.08**，算式见 §4.2）。
> 用户自行启动 transition 组重训，另两组暂不重训。
> 触发：`merged_transition_v20` 对四足 `--action_label jump` 的外推明显差于 `merged_stationary_v18`；
> 补 `--action_label_cfg_scale 2 --num_frames 30` 后**依然无效**，排除采样超参，坐实是监督塌陷
> 影响：`MOTION_METADATA_SCHEMA_VERSION` 7 → **8**；`CKPT_VERSION` **不变（16）**、
> `ACTION_LABEL_PARSER_CONTRACT_VERSION` **不变（5）**；两个特性默认关闭，state_dict 与条件张量契约均不变。
> **需重训**（本轮只重训 transition，见 §9），**不需** cond regen，**不需**重新预处理 motion。
> 相关：[`action_label_per_word_pooling.md`](action_label_per_word_pooling.md)、
> [`action_group_label_refactor.md`](action_group_label_refactor.md)、
> [`multi_dataset_training.md`](multi_dataset_training.md)

---

## 0. 一句话与决定性数字

一个 clip 的 **首组**是它的身份（决定它属于哪个模型的分布、eval、长度先验），
**非首组**只是别的组的**训练补充样本**；同理一个 label 的**首主词**是它的身份（决定 head 槽、
推理语义），**非首主词**只是 head 通道的**训练补充**。两条都是**纯训练期增广**，
推理契约一字不动。

在当前语料上，transition 模型能在 **head 槽**拿到 `jump` 向量的 clip 数：

| 配置 | head 槽含 jump 的 clip | 四足物种数 | 含 Buffalo 自己 |
|---|---|---|---|
| `merged_stationary_v18`（CKPT 12，旧主词平均池化，**当年好使的那个**） | 89 | 20 | ✅ |
| `merged_transition_v20`（CKPT 14 起，现状） | **54** | **5** | ❌ |
| 只做机制 A（跨组补充） | 54 | 5 | ❌ |
| 只做机制 B（主词增广） | 69 | 9 | ❌ |
| **A + B** | **113** | **23** | ✅ |

**两条机制互不替代，单独任何一条都不解决本问题**：A 把 Buffalo 的扑击腾空搬进了
transition，但那条 label 的首词仍是 `attack`，head 通道拿不到 jump；B 能把 head 通道
补上，但补的是各组**自己池子里**的 clip —— 对 transition 只多 15 条（`land, jump`），
Buffalo 那条在 stationary 里够不着。合起来才**超过** v18 的监督水平（89 / 20）。

> **摘掉规则 L、加上规则 F 之后这张表一个数字都不变**（2026-09-19 在真实语料上重测：
> 113 条 / 23 个四足物种 / 含 Buffalo = 本组原生 head=jump 的 54 条 + 本组可提升的 15 条
> + aux 里带 `jump` 的 44 条）。规则 F 的 24 条 `fall` clip 走的是 null 分支，不进这张表。
> 原因见 §6.4：L 带进来的 918 条 `walk` / `run` / `fly` 在 `aug` 模式下**全部**走 null 分支，
> 对 head 槽的 jump 监督贡献恰好为 **0**。L 买到的只有"通用步态 / 振翅动力学"这一条
> null 分支先验，代价是把 aux 池撑到 962 条并顶替掉 CFG 无条件分支 —— 本轮不做这笔交易（§7.4）。

> 同一口径下"只做 B"对 **stationary** 模型是 33 条 / 16 个四足物种 / 含 Buffalo ——
> 这正是 §2 的要害：单改槽规则（无论是机制 B 还是退回平均池化）只能救 stationary，
> 救不了 transition，因为**组的边界才是这里的硬墙**。

Buffalo 在 transition 训练集里也从"3 条倒地/起身"变成：

```
OWN  die                    Buffalo_Die
OWN  die                    Buffalo_Shot
OWN  getup                  Buffalo_GetUp
AUX  attack, jump, charge   Buffalo_Jump      (primary=stationary)  -> head=jump
```

（规则 L 若一并上，这里还会多 `Buffalo_WalkLoop` / `Buffalo_RunLoop` 两条 locomotion aux，
但它们只进 null 分支，不改变"Buffalo 在 transition 里终于见过一次腾空"这件事。）

---

## 1. 根因复盘（三条独立改动叠加）

统计口径：按每个模型**自己的槽规则**数"head 槽拿到 jump 向量"的 clip。
`merged_*_v18` 是 CKPT 12（标签里**所有**主词池化进 head 槽），`v20` 是 CKPT 14 起
（[`label_slot_ids`](../data_loaders/truebones/truebones_utils/action_label_conditioning_contract.py)：
**只有第一个**主词进 head 槽，其余进 modifier）。

1. **`b789585`（换组）**：57 条 head=jump 从 stationary 搬到 transition。这一步是**集中**，不是稀释。
2. **`98d26d1`（首词进 head 槽，CKPT 14）——主因**：把 `jump` 从所有"jump 不写在第一位"的
   clip 的 head 通道里**整体删除**，共 **59 条**：`attack, jump, *`(31)、`land, jump`(15)、
   `run, jump`、`swim, jump`、`roll, jump`。这 31 条 `attack, jump, *` 正是 Bear / BrownBear /
   Buffalo / Coyote / Fox / Gazelle / Goat / Horse / Leapord / Lion / Lynx / Raindeer /
   SandMouse … **18 个四足物种的扑击腾空**。
3. **同 commit 的 `jump, land` → `land, jump`**：额外降级 5 条四足落地跳
   （`Camel_Buck` `Deer_ComeDown` `Horse_JumpLand` `IAC_Mammoth_Land` `IAC_Sabertooth_Land`）。
4. 另有 5 条被改判 locomotion 的 `fall`，`jump` 一词整条消失
   （`Horse_InAir`、`IAC_Mammoth_Fall/JumpUp`、`IAC_Sabertooth_Fall/JumpUp`）。
   后两条 clip 名叫 `JumpUp` 却标 `fall`，**列入人工复核**（按既定规则 clip 名不能当判据，
   这里只作为"需要看视频"的提示）。
   —— 规则 F（§7.2）已把这 5 条带进 transition 的 aux 池，但只走 null 分支：
   它们补的是滞空动力学，**不**顶替人工复核，`jump` 一词仍然不在它们的 label 里。

全库带 `jump` 的 clip 共 **113** 条 / 四足 **35** 条 / **23** 个四足物种；
transition 模型的 head 通道现在只够到其中 **54 / 5 / 5**。

**可验伪的推论**：`merged_stationary_v20` 问 `jump` 应当**比 v18 更差**
（stationary 现在 head=jump 是 0 条）。这是本方案根因判断的对照实验。

---

## 2. 为什么不回退到"主词平均池化"

用户提问：是否退回 CKPT 12 的 head 槽平均？**不建议**，三条理由：

1. **解决不了本问题。** 那 33 条 `attack, jump, *` 留在 **stationary** 组。回退只让
   *stationary* 模型的 head 通道重新拿到 jump，*transition* 模型一条也拿不到 ——
   而 `jump` 现在的身份组是 transition。**组的边界才是这里的硬墙，槽规则不是。**
2. **会重新引入 CKPT 14 要修的缺陷。** 3714 条里 358 条是双主词，第二主词几乎全是
   姿态/媒介限定（hover / rear / crouch）；平均池化让首词在 head 通道只剩 cos 0.71，
   且 `idle, hover` 与 `attack, hover` 因共享 hover 而靠近。这是实测过的。
3. **代价不对称。** 回退要改 `ACTION_LABEL_PARSER_CONTRACT_VERSION`、重建 sidecar、
   槽源秩从 139 退回 107（`latent_dim` 下限随之变），且三组全部重训 —— 换来的是一个
   只覆盖一半问题的修复。

机制 B（§5）用**训练期增广**拿到"jump 也能进 head 槽"的监督，而推理时
`attack, hover` ≠ `hover, attack` 的几何**完全保留**。这是回退的严格上位替代。

---

## 3. 为什么不是 `--action_group all`

已有实现与实测（`merged_all_v3` @210k vs `merged_transition_v17` @70k，等样本数）：
transition 上 **不更好** —— pos_err 0.130 vs 0.115（v17 在 101 桶里赢 76 桶），
root 末端误差更差，turn 滑步 0.675 vs 0.540；只在 jerk 和多样性上更好。
用户复评结论一致：**更丰富但更容易互相干扰、动作分布塌陷**。

本方案与 `--action_group all` 的**五点结构差异**，正是"不塌陷"的依据：

| | `--action_group all` | 本方案（aux） |
|---|---|---|
| 覆盖面 | 全量三组无条件合并 | **按规则选入的子集**，逐条可审 |
| 采样质量 | 组权重 1:1:1，本组只占 1/3 | **本组恒占 `1 - aux_group_mass`（本轮 95%）** |
| 模型数 | 1 个 | **仍是 3 个**，每组风格一致性由主质量保证 |
| 组标记 | 加了 `--action_group_cond` 嵌入 | **不加**（§6.4 用 `--aux_label_mode` 处理外来条件，不靠标记隔离） |
| 切分 | 在合并集上 | **按首组算，aux 只进 train**（§4.3） |

---

## 4. 机制 A：`aux_action_groups`（首组身份 + 非首组补充）

### 4.1 数据格式

`action_labels.jsonl` 每行新增可选键，缺省 `[]`：

```jsonc
{"clip": "Buffalo_Jump", "action_group": "stationary",
 "action_label": "attack, jump, charge", "is_loop": false,
 "aux_action_groups": ["transition"]}
```

- `action_group`（首组）**语义一字不变**：身份、eval 归属、`clip_length_prior` 的键、
  `validate_anytop_dataset` 的判据、review 页面的分组，全部只看它。
- `aux_action_groups` ⊆ `ACTION_GROUPS`，**不得包含首组**，不得重复。
  `MOTION_METADATA_SCHEMA_VERSION` 7 → 8；`load_motion_metadata` 原样 join 进 entry。
- **一次性迁移写入，不做常驻脚本**（用户 2026-09-19）：实施时按 §7 的规则表直接改写
  三份 `action_labels.jsonl`（保留 `.bak`），改完即弃，不留 `tools/prefill_*` 工具。
  代价是这批取值**不能靠重跑脚本复现**，所以 §7 的规则表就是它唯一的出处记录；
  日后新增的 clip 需要人工决定 `aux_action_groups`（或再写一次性脚本），
  `validate_anytop_dataset` 的字段校验仍然兜底格式，但兜不住"漏标"。

### 4.2 采样质量预算 `--aux_group_mass`（防塌陷的核心旋钮）

**aux 不是按条加权，而是按总质量配额**，这样 aux 池有多少条都不会改变本组的支配地位：

```
m = --aux_group_mass     # 默认 0.0 = 特性关闭
本组 clip 合计拿 (1 - m) 的采样质量
aux  clip 合计拿   m    的采样质量
两池内部各自按现有规则再分：--balanced 时 sqrt(物种 clip 数) 再组内均分，否则逐条均分
```

- `m = 0` 时 aux clip **直接不进 `name_list`**（而不是权重置零）：否则它们仍会影响
  `sampler_index_joint_counts()` 与 `JointBucketBatchSampler` 的桶分布、以及数据集统计。
- 守卫的判据要分清两种"空"：
  `m > 0` 且**整个语料里没有任何一行带 `aux_action_groups` 键** → **硬失败**
  （这是 sidecar 未重建，与仓库其它守卫同一风格）；
  而**本组**的 aux 池恰好为空（只上规则 J 时 **locomotion 与 stationary 都是这样**）→
  只打印一行提示，`m` 退化为空操作。
  **用"语料里有没有这个键"判 sidecar 是否过期，不要用"本组有没有命中"**，
  否则 locomotion 会被自己的合法空池误杀。
- **`m` 要跟着 aux 池的大小走，不能照抄。** 质量配额是"整池 m"，所以逐条权重是
  `m / N_aux` 对 `(1-m) / N_own`；池越小，同一个 `m` 的逐条过采样倍数越高：

  | 规则 | N_own | N_aux | m | 每条 aux / 每条 own | head=jump 占抽样 |
  |---|---|---|---|---|---|
  | J + L（原方案） | 758 | 962 | 0.25 | 0.26× | ≈ 1.1%（44/962 可提升） |
  | J only | 758 | 44 | 0.25 | **5.7×** | **25%** |
  | J only | 758 | 44 | 0.05 | 0.91× | ≈ 5% |
  | **J + F（本轮采用）** | 758 | **68** | **0.08** | **0.97×** | **5.18%** |

  aux 池里**可提升的只有带 `jump` 的 44 条**（规则 F 的 24 条 `fall` 走 null 分支，§6.4），
  而质量配额是整池均分，所以 head 通道被 jump 占据的比例是 `m × 44 / N_aux` ——
  **换规则集就必须重算 `m`，不能照抄**。`m = 0.25` 会把它顶到 25%，远超"当年好使的"
  v18 的自然比例 4.5%（89 / 1997），且让那几十条 clip 各被重复 5.7 倍 ——
  这正是 §8.0 守卫 1（die / getup / turn 不得退化）最怕的形状。
  **取 `m = 0.08`**：逐条与本组持平（68 / 826 = 0.082，实测 0.97×），
  jump 监督维持在 5.18%（与 J-only + `m=0.05` 同量级，加 F 没有稀释它），
  fall 的 null 分支占 2.82%；200k 步 × batch 24 下每条 aux 被抽到约 5.6k 次。

### 4.3 split 安全（**最重要的实现约束**）

[`load_motion_names_for_split_with_action_group`](../data_loaders/truebones/data/dataset.py)
是**先按组过滤、再在过滤后的物种集合上算 split** 的：它把过滤结果里出现的 `object_type`
定长洗牌后按 `_compute_filtered_split_counts(len(object_types_list))` 切片，并把
`train/val/test.txt` 写回 `data_root`。

如果 aux clip 参与 split 计算，会让 transition 组里出现的物种集合变化 →
**val/test 物种被重新洗牌** → 所有 v18/v20 实验失去可比性，更糟的是**留出物种可能泄漏进 train**。

约束（实现时必须满足）：

1. split **只用首组成员**计算，逻辑与写盘行为保持现状不动。
2. aux clip **只并入 `train` 集**，且**只对首组 split 已判为 train 的物种**生效；
   某物种在该组被判为 val/test 的，它的 aux clip **丢弃**，不得泄漏。
3. 不变量测试：同一 cond、同一组，`--aux_group_mass 0` 与 `>0` 两次运行产出的
   `train.txt` / `val.txt` / `test.txt` **必须逐字节相同**。

### 4.4 采样器（现状会静默失效）

v18/v20 都是 `--balanced False`，此时
[`get_dataset_loader`](../data_loaders/get_data.py) 的 `use_weighted_sampler` 为假，
走的是 `RandomSampler` —— **根本没有加权通路**，`aux_group_mass` 会被静默忽略。

改法：`use_weighted_sampler = balanced or has_aux_clips`；`TruebonesSampler` 增加
`aux_mask` 与 `aux_group_mass`，先按 §4.2 分两池质量，池内再走现有 sqrt/均分逻辑。
`JointBucketBatchSampler` 包在外层，不受影响。

### 4.5 只看首组的消费者（实现时逐一确认不被污染）

| 位置 | 用途 | 要求 |
|---|---|---|
| `utils/clip_length_prior.py` | `--num_frames auto` 的长度先验，按 `(group, label)` 建键 | **只用首组** |
| `utils/validate_anytop_dataset.py` | 组合法性校验 | 首组校验不变；新增 aux 字段校验 |
| `tools/regenerate_dataset_artifacts.py` | 侧写产物 | 首组建键；aux 字段透传 |
| `dataset/review/serve.py` | 标注页分组 | 首组分组；aux 作为标记显示 |
| `eval/motion_quality/reference_bank.py` | 参考库 | **本就按主词而非组建键，天然不受影响**（见其 L56 注释） |
| `_validate_head_order_consistency` | 词集在组内只许一种拼法 | **仍按首组**；aux clip 带自身 label 原样入池，不产生新约束 |

---

## 5. 机制 B：主词增广 `--head_aug_words` / `--head_aug_prob`

### 5.1 规则：一次 `slot_ids` 交换，落在模型里

模型只消费 `word_ids` / `slot_ids` / `word_mask`；
[`assemble_slot_channels`](../data_loaders/truebones/truebones_utils/action_label_conditioning_contract.py)
按槽取均值，**书写顺序除了已经编码进 slot_ids 的部分之外不到达模型**（见其 docstring
与 `slot_channel_representation`）。所以"把第二主词提为首主词"就是**交换两个 token 的 slot_id**，
不需要改拼写、不需要重新 parse、不需要动 sidecar：

```
attack, jump, charge   slot_ids = [HEAD, MODIFIER, MODIFIER]
                  --(p)-->      [MODIFIER, HEAD, MODIFIER]     # head 通道变成 jump
```

落点与 [`_drop_direction_slot`](../model/anytop.py) 并列，同样是**训练期纯张量掩码操作**，
compile / cudagraph 友好，逐样本零 Python 开销：

```python
# head_aug_word: (V,) bool buffer, 由 --head_aug_words 构造
promotable = head_aug_word[word_ids] & (slot_ids == SLOT_MODIFIER) & word_mask
do = promotable.any(dim=1) & (torch.rand(B, device=...) < self.head_aug_prob)
slot_ids = torch.where(do[:, None] & promotable,          SLOT_HEAD,
            torch.where(do[:, None] & (slot_ids == SLOT_HEAD), SLOT_MODIFIER, slot_ids))
```

**必须用 `word_mask` 门控**：`tensors.py` 用 `SLOT_PAD_ID` 填 `slot_ids`，却用 **0** 填
`word_ids`，而词表 index 0 是 `attack` —— 不门控的话每个 padding 位都会被当成 `attack` 命中。

`ACTION_LABEL_MAX_HEADS = 2`，所以一行最多一个可提升的主词，不存在多重提升的歧义。
eval / inference 永不增广（与 direction drop 同一条 `self.training` 判据）。

### 5.2 为什么要词表而不是全局开

第二主词在语料里绝大多数是**姿态/媒介限定**（hover / rear / crouch）。无差别提升会把
`hover` 训成"带攻击的悬停"，正好抵消 CKPT 14 的收益。所以默认**空词表 = 关闭**，
只对"第二主词其实是该 clip 主导身体事件"的词显式开启。

- 首批建议：`--head_aug_words jump`。
- `--head_aug_prob` 建议 0.25（与 `--action_label_cfg_drop_prob 0.3` 同量级，
  是正则而非改写：原拼写仍占多数）。

### 5.3 边界

- **只作用于训练**：`action_labels.jsonl` 不落任何增广痕迹，
  `_validate_head_order_consistency`、审计工具、sidecar 全都看不到它。
- 与 direction dropout、`action_label_cfg_drop_prob` 正交：
  顺序上**先做主词交换，再做 direction drop，最后 CFG 硬丢**
  （CFG 丢掉整行时交换与否都不可见，顺序无所谓，但固定下来便于复现）。
- `ACTION_LABEL_PARSER_CONTRACT_VERSION` **不变**：交换后的 `slot_ids` 仍是契约 5
  能产出的合法配置（等价于语料里本来就存在的 `jump, land` 一类拼法）。
- 槽源秩不变：`word_slots()` 早已声明主词可达 head 与 modifier 两个槽，
  预检的秩报告本来就按可达槽取源行。

---

## 6. aux clip 的条件处理 `--aux_label_mode`（A 与 B 的耦合点）

> 起因：未触发主词增广时，从 stationary 带进 transition 的 `attack, jump, charge`
> （首词 `attack`）对 transition 模型是零贡献还是负贡献？

### 6.1 实测：三个组的原生首词**完全不相交**

| 组 | 原生首词（该组 label 的第一个主词） |
|---|---|
| locomotion | walk, run, fly, swim, hover, crawl, fall, roll |
| stationary | attack, idle, hurt, dance, work, rest, pickup, putdown, lift, draw |
| transition | die, turn, spawn, getup, jump, land, laydown, takeoff, burrow, sitdown, draw, stop, rear, kneel |

这是分组定义本身的结果，不是巧合。规则 J+L 下：

```
transition 的 aux 池 962 条 -> 962 条（100%）带来本组原生不存在的首词
   walk 369 | run 340 | fly 102 | swim 59 | hover 57 | attack 33 | crawl 1 | roll 1
stationary 的 aux 池 928 条 -> 928 条（100%）同上
```

**本轮上规则 J + F，池子变成 68 条，首词分布也随之收窄**（2026-09-19 实测）：

```
transition 的 aux 池 68 条 -> 68 条（100%）带来本组原生不存在的首词
   attack 33 | fall 22 | run 5 | swim 5 | hurt 2 | roll 1
stationary / locomotion 的 aux 池：0 条
```

其中 **J 的 44 条**在 modifier 槽里带 `jump`，`aug` 模式下全部被提升为 head=jump；
**F 的 24 条**（首词 `fall` 22 + `hurt, fall` 2）不含 `--head_aug_words` 里的词，
全部退化为 null 分支（§6.4）。

> `fall` 本身是 **locomotion** 的原生首词，不是 transition 的 —— 所以把 fall clip
> 原样带 label 进来，训的仍是一块 transition 推理时不会被查询的 head 区域。
> 这正是把它留在 null 分支的理由，也是 `--head_aug_words` 里**没有** `fall` 的理由。

**原样带 label 的 aux clip，其条件路径系统性地在训练一片"本组推理时永远不会被查询"的
head 区域。**

### 6.2 实测：`attack` 不会泄漏到 `jump`（也不会泄漏到任何 transition 首词）

按既定 T5 探针方法（同形状探针 + 表内均值中心化），在冻结词表
`dataset/action_word_embeddings.npy`（104 词 / 768 维）上测：
**null = 全部词对 |cos| 的 p95 = 0.190**（中位数 0.066）。

```
cos(jump, attack) = +0.102      < null
attack 对 transition 全部 14 个原生首词：最大 |cos| = 0.164 (rear)，全部 < null
attack 的最近邻：gun .20, retreat .16, fast .15, push .14, look .13, die .12
jump   的最近邻：dive .29, hover .21, fast .18, throw .17, smash .16, whip .15
```

结论：**既没有正泄漏也没有负干扰** —— `attack` 占的是 head 通道里一块与本组查询近乎正交的区域。

### 6.3 所以答案是：条件路径 ≈ 零，但 aux clip 整体**不是**零贡献

未触发增广时，一条 aux clip 的梯度分三路：

| 路径 | 与首词有关？ | 贡献 |
|---|---|---|
| **CFG null 分支**（`--action_label_cfg_drop_prob 0.3` ⇒ 30% 的抽样） | 否 | **主要收益**：ε(x, species, ∅) 学到 Buffalo 这副骨架会腾空、会走 |
| **骨架/物种通路**（joint-name emb、结构通道、cross-limb、FK 方向损失） | 否 | 实打实的身体动力学监督 |
| **head 通道条件路径**（剩下 70%，head=`attack`） | 是 | ≈ 零（§6.2）：既不帮 `jump`，也不伤它，只是没人会去查 |

唯一真实的负向路径**不在**"attack 干扰 jump"上，而是 **CFG 基准漂移**：
Buffalo 的 null 分支从"三种倒地"变成"倒地/起身/腾空/走/跑"，同一个 `--action_label_cfg_scale`
的含义随之改变 —— **v20 与 v21 的 cfg 值不可直接比较**。cfg 的选取留给人工，不列为验收项。
方向上这是好事 —— 原先 CFG 是从"躺着"往一个模型不认识的方向推，现在是从"一般的水牛运动"
往 jump 推。

### 6.4 结论：加旋钮 `--aux_label_mode`，默认 `aug`

既然条件路径 ≈ 零而其余两路才是收益，就不该让 aux clip 白带一个无人查询的首词进来：

| 模式 | 行为 | 适用 |
|---|---|---|
| `label` | 原样用 aux clip 的 label | 仅作逃生舱；本次不训对照 run（§8.1），**没有实测支撑** |
| `null` | 该 clip **永远**走 CFG null 分支，动作条件完全不参与 | 纯物种先验补充 |
| **`aug`（默认）** | 首词可按 `--head_aug_words` 提升 → **100% 提升**（而非本组 clip 的 `--head_aug_prob`）；不可提升 → 退化为 `null` | 目标配置 |

`aug` 模式下：

- `attack, jump, charge` → **100% head=jump**，modifier={attack, charge}。
  推理问裸 `jump` 时 modifier 为空槽（零行），这是 transition 组大量存在的常见状态
  （`die` / `getup` / `jump` 单词标签），支撑充分。
- `walk, forward`（规则 L 会带进来的 918 条）→ `walk` 不在 `--head_aug_words` 里 →
  走 null 分支，**只贡献 Buffalo 的走路动力学，不往 transition 的 head 通道塞一个 `walk`**。
  这让体量最大的规则 L 从"最危险"变成"最安全"的那一条。
  **本轮规则 L 已屏蔽（§7.4），走这条退化分支的是规则 F 的 24 条 `fall` clip**：
  `fall` / `hurt` 都不在 `--head_aug_words` 里 → 全部进 null 分支，
  **只补腾空/坠落的身体动力学与物种先验，不往 transition 的 head 通道塞一个 `fall`**。
  规则 F 与规则 L 在这一点上是同一个机制，只是体量从 918 降到 24 ——
  CFG 无条件分支被 aux 主导的风险随之从"半个池子"降到 2.8% 的抽样质量（§4.2）。
  （`aug` 与 `label` 仍然不同：`label` 下 J 的 44 条只按 `--head_aug_prob` 的 25% 概率
  提升，另外 75% 会把 `attack` / `run` / `swim` 写进 transition 的 head 通道，
  F 的 24 条则会把 `fall` / `hurt` 写进去。）

被否决的第四种模式：把 head 槽置为缺席（零行）而保留 direction / modifier。
`parse_action_label` 要求每条 label 至少一个主词，训练语料里 head=∅ 从不出现；
引入一个全新的条件状态，收益不抵风险。

---

## 7. aux 规则表与实测影响

> 规则**只在实施时跑一次**，结果直接写进三份 `action_labels.jsonl`，不留常驻脚本；
> 本节因此是这批 `aux_action_groups` 取值的**唯一出处记录**。人工可逐条推翻。
> 下列数字在 `train/v21` 当前语料实测。

### 7.1 规则 J（jump 桥接）—— **本轮采用**

label 含 `jump` 且首组 ≠ transition → aux `["transition"]`。
覆盖 **44 条**（首组 stationary 33 + 首组 locomotion 11）。

### 7.2 规则 F（坠落/滞空桥接）—— **本轮采用**

label 含 `fall` 且首组 ≠ transition → aux `["transition"]`（用户 2026-09-19 追加）。
覆盖 **24 条**：首组 locomotion 22 条（首词就是 `fall`，含 `fall, dead` / `fall, hand1`
/ `fall, hand2` 等写法）+ 首组 stationary 2 条（`hurt, fall`）。
**与规则 J 零重叠** —— 这 24 条 label 里一个 `jump` 都没有
（`IAC_Mammoth_JumpUp`、`KI_Human_Jump01MidAir` 这些 clip **名字**里有 Jump，
但按既定规则 clip 名不作判据，label 写的是 `fall`，见 §1 第 4 条与 §10）。

11 个物种：Horse / IAC_Mammoth / IAC_Sabertooth（四足）、IAC_Caveman / IAC_Cavewoman /
KI_Archer / KI_CasterMage / KI_Human / KI_Soldier（双足）、MB_TigerDrago / MB_Unka（飞行）。

**它买到的是什么、不是什么**（实测，别搞混）：

- **不是** head 槽的监督。`fall` 不在 `--head_aug_words` 里，24 条全部走 null 分支，
  §0 那张表一个数字都不变（§6.4）。`fall` 本就是 locomotion 的原生首词，
  往 transition 的 head 通道里写 `fall` 只会训一块没人查询的区域。
- **不是** 新物种覆盖。这 11 个物种在 transition 里**本来就**有 own 的
  `jump` / `land` / `takeoff` 或 J 带进来的 aux，一个新增物种都没有。
- **是** 滞空/坠落身体动力学的样本量，以及 CFG 无条件分支里"这副骨架离地之后怎么动"
  的先验密度 —— 与 `land` / `jump` 直接相邻的那一段。

### 7.3 合计（本轮落盘）

| 组 | 本组 | +aux(J) | +aux(F) | 池合计 | 本组质量占比(m=0.08) |
|---|---|---|---|---|---|
| locomotion | 959 | 0 | 0 | 959 | 100%（aux 池空，`m` 为 no-op） |
| stationary | 1997 | 0 | 0 | 1997 | 100%（同上） |
| transition | 758 | **44** | **24** | **826** | 92% |

sidecar 里带 `aux_action_groups` 的行共 **68** 行，取值全部 `["transition"]`。

### 7.4 规则 L（步态/飞行动力学桥接）—— **本轮屏蔽，保留备选**

首组 = locomotion 且首主词 ∈ {walk, run, fly, swim, hover, crawl} → aux
{stationary, transition}。命中 **928 条 locomotion clip**，其中 10 条同时也命中 J
（`run, jump` / `swim, jump`），所以**只由 L 带进来的是 918 条** —— 本文其余各节说的
"L 的 918 条"都是这个口径。（J 的 11 条 locomotion clip 里有 1 条 L 够不着：
`roll, jump, forward`，`roll` 不在 L 的词集里。）
对应用户"行走姿态、翅膀煽动的动力学是通用的"。它若上，池子是
locomotion 959 / stationary 2925 / transition 1720。

**2026-09-19 用户决定：本轮只实施规则 J，规则 L 屏蔽。** 支撑这个取舍的实测：

- **L 对决定性指标的贡献是 0。** §0 的 113 条 / 23 个四足物种 / 含 Buffalo，
  不上 L 时一字不差 —— 因为 L 的 918 条首词是 `walk`/`run`/`fly`，
  不在 `--head_aug_words` 里，`aug` 模式下全部走 null 分支，head 槽拿不到任何 jump。
- **L 的收益只有一条 null 分支先验**（"这副骨架会走会跑会振翅"），
  而代价是把 aux 池从 68 撑到 986：`--aux_group_mass` 是整池配额，
  这 918 条会**顶替**掉 44 条真正带 jump 的 clip 的抽样质量（同样的 m 下，
  每条 J clip 的权重被稀释 14 倍），同时让 transition 的 CFG 无条件分支被 aux 主导。
  规则 F 是同一个机制的**小剂量版本**：24 条、2.8% 质量，可控得多（§7.2）。

- 原方案里"L 与 J 同场上"的理由是"J 单独对 Buffalo 只多一条 aux clip"。
  这条理由成立，但**那一条恰好就是要的那一条**（`Buffalo_Jump`，`attack, jump, charge`
  → head=jump），而 §8.0 的主目标问的就是 Buffalo 的腾空。

代价（记录在案）：本轮拿不到"通用步态动力学"这条先验（规则 F 只补了坠落那一段，
没补行走/振翅）。若 v21 的 jump 落地姿态仍不像四足、或非四足物种的腾空明显差，
**L 是第一个要补上的东西**，补的时候必须同步把 `--aux_group_mass` 调回 0.25 量级
（按 §4.2 的逐条权重表重算），否则 44 条 J clip 会被 918 条 L clip 挤掉。

好处：**归因仍然干净**。若 transition 的 die / getup / turn 出现稀释，
现在只可能来自 68 条 aux 或 `--head_aug_prob`，回退顺序是
**先降 `--aux_group_mass`（0.08 → 0.03），仍不行再把 `--head_aug_prob` 降到 0.1**。

---

## 8. 验收

### 8.0 目标与守卫

**主目标**：`merged_transition_v21` 对 `--object_type Buffalo --action_label jump`
的腾空高度、离地帧数、落地姿态，达到或超过 `merged_stationary_v18` 的
`Basic/task3/Buffalo_3.bvh`。采样长度同口径 `--num_frames 30`；cfg 人工选取
（§6.3：基准漂移使两版的 `--action_label_cfg_scale` 不同义，不要直接照搬 v18 的 2）。
扩展到 `eval_tasks_transition.json` 里四足无 jump 样本的物种（Buffalo / MB_Unka / RedDragon 新骨架）。

**不退化守卫**（必须同时看，否则就是拿塌陷换外推）：

1. transition 自身的 die / getup / turn 桶 vs GT：pos_err、root 末端误差、turn 滑步
   —— 对照 `merged_transition_v20`，不得变差。
2. stationary 的 attack 桶：机制 B 会让本组的 `attack, jump, *` 有 `--head_aug_prob`
   比例的样本不以 attack 为首词，需确认 `--action_label attack` 未被稀释。
3. `train/val/test.txt` 三组逐字节不变（§4.3 不变量）。

### 8.1 只训一个 run

**决定（用户 2026-09-19）：不做消融对比，只训一个 run（`merged_*_v21`），直接看效果。**

数据侧：**启用规则 J + F，规则 L 屏蔽**（§7）。训练侧：

```
--aux_group_mass 0.08 --aux_label_mode aug --head_aug_words jump --head_aug_prob 0.25
```

三组都要重训，同一套旗标。注意**只上规则 J + F 时，locomotion 与 stationary 两组的
aux 池都是空的**（两条规则都只往 transition 送），`--aux_group_mass` 对它们退化为空操作（§4.2 的守卫
必须按"语料里有没有这个键"判，而不是按"本组有没有命中"，否则这两组会被误杀）；
但它们仍会吃到机制 B 对本组 `attack, jump, *` / `run, jump` / `swim, jump` /
`roll, jump` 的提升。

代价是：若 v21 没修好，无法从训练结果本身分辨是三个部件（aux 质量预算 /
`aux_label_mode` / 主词增广）里哪一个没起作用。**§8.2 的探针把这份诊断能力绝大部分
挪到了同一个 run 的采样上和训练前的单元测试上，零额外训练成本** —— 只有在探针也指不明时，
才需要回头补训练侧的对照。

### 8.2 零额外训练成本的探针（全部在 `merged_transition_v21` 这一个 checkpoint 上做）

| 探针 | 怎么做 | 回答什么 |
|---|---|---|
| **训练前数据断言** | 单元测试（§9 步骤 8），不是 run | `Buffalo_Jump` 确实进了 transition 的 **train** 集，且其 `slot_ids` 在 `aug` 模式下被提升为 head=`jump`。**这一条拦住绝大多数"配置没生效"的失败。** |
| **空标签 = 直接看 null 分支** | `--object_type Buffalo`，**不传** `--action_label`（`_resolve_action_condition` 返回 `None` → 走学到的 null 嵌入） | aux 有没有把 Buffalo 的物种先验从"三种倒地"救回来 —— **不用再单训一个不带主词增广的 run，免费问到。** |
| **head 通道 vs modifier 通道** | 同一物种对比 `jump` / `jump, attack` / `attack, jump` | 腾空是谁带来的。若 `attack, jump`（在 transition 是 OOD 首词）反而更像扑击，说明主词提升没真正生效 |
| **有原生样本的对照物种** | `--object_type Horse --action_label jump`（Horse 在 transition 有原生 head=jump 的 `Horse_RunJump`） | Horse 好而 Buffalo 差 ⇒ 跨物种迁移不足；两者都好 ⇒ 成了 |

### 8.3 失败时的排查顺序

1. **空标签探针仍出"倒地"** ⇒ aux clip 根本没进训练集。查 §4.3（物种被判 val/test 导致
   aux 被丢弃）和 §4.4（`--balanced False` 下采样器静默失效）—— 这两个坑都会让
   `--aux_group_mass` 无声无息地等于 0。
2. **空标签探针已正常（会走会跑），但 `jump` 仍差** ⇒ 物种先验修好了、head 通道没接上。
   查主词提升是否生效（数据断言应已拦住），再考虑把 `--head_aug_prob` 提到 0.5。
3. **两者都正常但 `jump` 偏"扑击"而非中性跳跃** ⇒ 不是 bug，是语料事实（§10 第一条）。
4. **transition 自身的 die / getup 退化** ⇒ 降 `--aux_group_mass`（0.08 → 0.03），
   仍不行再把 `--head_aug_prob` 降到 0.1；不要改 `--aux_label_mode`（§10 第四条）。
   规则 L 不在场，所以归因只在 68 条 aux 与主词增广之间二选一（§7.4）。
   若退化集中在 `die`，先看规则 F 的 2 条 `hurt, fall` 与 transition 自己的
   `die, fall` 是否把 null 分支往"坠地"带偏。

---

## 9. 实施步骤

1. `motion_labels.py`：`MOTION_METADATA_SCHEMA_VERSION` 7→8；`load_action_labels` 解析
   `aux_action_groups`；新增 `_validate_aux_action_groups`（子集、不含首组、不重复）。
2. `dataset.py`：`load_motion_names_for_split_with_action_group` 按 §4.3 拆成
   "首组算 split" + "aux 只并 train"；`Truebones` 记录 `aux_mask`。
3. `TruebonesSampler` + `get_data.py`：§4.4 的两池质量预算与 `use_weighted_sampler` 条件。
4. `parser_util.py`：`--aux_group_mass`（默认 0.0）、`--aux_label_mode`
   （`label` / `null` / `aug`，默认 `aug`）、`--head_aug_words`（默认空）、
   `--head_aug_prob`（默认 0.0）；四者进 args.json。
5. `model/anytop.py`：`head_aug_word` (V,) bool buffer + `_promote_head_slot()`，
   紧挨 `_drop_direction_slot` 放，按 §5.1 的顺序调用。**数据侧不动一行。**
   collate 需多带一个 `y['is_aux']` (B,) bool，供 §6.4 三模式分流：
   `aug` = 对 aux 行把提升概率钉成 1.0、不可提升的行并入 `action_label_active=False`；
   `null` = 直接并入 `action_label_active=False`。两者都是既有掩码通路，不加新分支结构。
6. **一次性迁移** `action_labels.jsonl` ×3：按 §7.1 的**规则 J**与 §7.2 的**规则 F**
   写入 `aux_action_groups`（规则 L 屏蔽，不写），先出 dry-run 清单交人工过目，
   确认后落盘并留 `.bak`。脚本用完即弃，不进 `tools/`。
   预期命中数按 §7.3 的表核对（带键的行 **68** 条 = J 44 + F 24，取值全是
   `["transition"]`；transition 池 +68 / stationary +0 / locomotion +0），
   对不上就是规则实现有偏，不要直接落盘。
7. `utils/validate_anytop_dataset.py`：新字段校验 + split 不变量断言。
8. 测试：`tests/test_aux_group_membership.py`（split 不变量、val/test 不泄漏、质量预算、
   §4.2 的两种"空"分别硬失败/只告警）、`tests/test_head_word_augmentation.py`（p=1 时 head 通道等于
   `jump` 的词向量、p=0 完全旁路、`eval_mode` 下永不触发、padding 位不被误命中、
   词表外的第二主词不动）。
9. 三组重训（§8.1 的单一配置），按 §8 验收。
   —— **2026-09-19 决定只重训 transition 一组**（`merged_transition_v21`）。
   stationary / locomotion 保持 v20：规则 L 屏蔽后这两组的 aux 池都是空的，
   重训只会吃到机制 B 的主词提升（stationary 33 条 `attack, jump, *`、
   locomotion 3 条 `run/swim/roll, jump`），要等 transition 验证通过再动。
   **代价**：§8.0 守卫 2（stationary 的 attack 桶是否被稀释）本轮无法验证，
   因为 stationary 模型没有重训；该守卫顺延到 stationary 重训时。

**不需要**：cond regen、重新预处理 motion、CKPT_VERSION 提升、sidecar 重建、词表重建。

### 9.1 步骤 1–8 落地实测（2026-09-19）

> 下面这一段是**规则 J + L 同上**时的实测，保留作为 L 的基线记录；
> 规则 L 屏蔽后的实测见 §9.2。

- `action_labels.jsonl` ×3 迁移后 aux 池：**transition 962 / stationary 928 / locomotion 0**，
  与当时 §7 的 J+L 表逐项吻合。
- 真实 `dataset/merged/cond.npy` 上构建 transition 训练集：池 **758 → 1720**，
  aux 962，`use_weighted_sampler=True`；采样质量实测 **own 0.750000 / aux 0.250000**。
- Buffalo：own = Die / Shot / GetUp，aux = **Jump / WalkLoop / RunLoop** —— 即 §0 预期。
- **§4.3 不变量已在真实语料上验证**：`--aux_group_mass 0.25` 与 `0` 两次运行产出的
  三份 `train.txt` md5 完全相同。（注意 manifest 不在 git 跟踪内，`git status` 证明不了这件事。）
- 全量测试 **973 passed**，新增 `tests/test_aux_group_membership.py`(13) 与
  `tests/test_head_word_augmentation.py`(11)。

**实施期间发现、值得记住的两件事**：

1. `DEFAULT_SPLIT_RATIOS = {train 1.0, val 0, test 0}` —— 现行配置下**物种数 ≥4 时
   val/test 恒为空**，只有 2–3 个物种才会留出。所以 §4.3 的「留出物种不得泄漏」在当前
   语料上是**防御性代码**，测试用 3 物种夹具专门覆盖它，以防日后比例被改回去。
2. `_build_action_slot_batch` 用 `SLOT_PAD_ID` 填 slot 却用 **0** 填 word id，而词表
   index 0 是 `attack` —— 主词提升的查表**必须用 `word_mask` 门控**，否则每个 padding
   位都会命中。已写成测试。

### 9.2 摘掉规则 L、加上规则 F 后的实测（2026-09-19，本轮生效配置）

两步就地重算三份 sidecar，都没有回滚 `.bak`：

1. **摘 L**：J∩L 的 10 行 `["stationary","transition"]` → `["transition"]`，
   纯 L 的 918 行整条删掉 `aux_action_groups` 键，J 的其余 34 行原样保留。
2. **加 F**：24 行 `fall` clip 插入 `"aux_action_groups": ["transition"]`
   （zoo 1 行 + unitybundles 23 行；`clean_processed` 一条都没有）。

改后 sidecar 与 `.bak` 相比，除这 68 行的 `aux_action_groups` 外**逐行一字不差**
（`git diff` 68 增 68 删）。实测：

- 带 `aux_action_groups` 的行：**68** = J 44（stationary 33 + locomotion 11）
  + F 24（locomotion 22 + stationary 2），取值全部 `["transition"]`，两条规则零重叠；
  `load_action_labels` 三份全部解析通过，`validate_anytop_dataset --datasets` 三份全 PASS。
- 真实 `dataset/merged/cond.npy` 上的训练池：
  **transition 758 + 68 = 826**（`use_weighted_sampler=True`，采样质量实测
  **own 0.920000 / aux 0.080000**，逐条 aux/own = **0.97×**）/
  **stationary 1997 + 0** / **locomotion 959 + 0**。
  后两组打印 §4.2 的 no-op 提示、`use_weighted_sampler` 回落为 `False`
  （即与 v20 完全同一条 `RandomSampler` 通路），**不是**硬失败 —— 键在语料里存在。
- 抽样质量的去向：head=jump **5.18%**、fall 的 null 分支 **2.82%**、本组 92%。
- §4.3 不变量在新语料上重验：`--aux_group_mass 0` 与 `>0` 两次构建产出的
  三份 `train.txt`（以及 `val.txt` / `test.txt`）md5 完全相同。
- §0 的决定性指标重测：transition 能把 jump 送进 head 槽的 clip
  **113 条 / 23 个四足物种 / 含 Buffalo** = 本组原生 54 + 本组可提升 15 + aux 44
  —— **与 J+L 完全相同**；规则 F 对这张表贡献 0（全走 null）。
- aux 池 68 条的首词：`attack` 33 / `fall` 22 / `run` 5 / `swim` 5 / `hurt` 2 / `roll` 1。
- Buffalo：own = Die / Shot / GetUp，aux = **Jump**（`attack, jump, charge` → head=jump）。
- `train_transition.bat` 的 `--aux_group_mass` 0.25 → 0.05 → **0.08**（§4.2 的逐条权重表）。
- 全量测试 **973 passed**（含 `tests/test_aux_group_membership.py` 13 +
  `tests/test_head_word_augmentation.py` 11）。

---

## 10. 已知未决

- **`--head_aug_words jump` 的语义副作用**：训练后 `--action_label jump` 在四足上可能偏向
  "扑击式腾空"而非中性跳跃，因为语料里四足的腾空绝大多数就是扑击。
  这是语料事实，不是 bug；若要中性跳跃需要补语料，或在推理时靠 modifier 槽区分。
- **规则 L 本轮未上场**（§7.4），所以它的组间干扰仍然是未实测的。
  重新打开它之前要记住两件事：它对 head 槽的 jump 监督贡献是 **0**；
  它会把 aux 池从 68 撑到 986，同一个 `--aux_group_mass` 下把每条 J clip 的权重稀释 14 倍。
  补它的时机是"v21 的腾空动力学不像四足"，而不是"jump 出不来"。
- **规则 F 的 24 条 `fall` 只走 null 分支，等于赌"无条件分支里的滞空先验能被 CFG 推出来"。**
  若 v21 的 `jump` 腾空够高但**下落段**仍然僵硬，可以考虑把 `fall` 也加进
  `--head_aug_words`，让 transition 模型能被 `--action_label fall` 直接查询。
  代价有两条，都不小：它会打破 §6.1 "三组原生首词互不相交"这条正在被依赖的性质
  （`fall` 是 locomotion 的原生首词），并且会顺带让 stationary 的 `hurt, fall`
  与 transition 自己的 `die, fall` 有 25% 的样本不以 `hurt` / `die` 为首词。
  **本轮不做**，记在这里以免日后重新论证。
- **§6.2 的正交性只测了词向量，没测训练后的模型。** 冻结 T5 表里 `attack` 与
  `jump` 不相关，不等于模型内部的 head 投影之后仍不相关（投影是学出来的，可能把两者
  拉近）。验证它需要一个 `aux_label_mode=label` 的对照 run，**本次不训**（§8.1）。
  所以"外来首词零干扰"始终是**基于几何的推断，不是实测结论** —— 在默认的 `aug` 模式下
  这条推断其实用不上（外来首词要么被提升成 `jump`，要么整条走 null，根本不进 head 通道），
  它只在有人把 `--aux_label_mode` 改成 `label` 时才成为前提。
- **`aug` 模式下 aux 的 null 分支占比本轮是 24 / 68**：规则 L 屏蔽后剩规则 F 的
  `fall` clip 走这条路（§6.4）。它只占 2.82% 的抽样质量，远不到 J+L 时"无条件分支被
  aux 主导"的程度，但也不再是零。要留意的是 `--aux_group_mass` 现在按
  `m × 44 / 68` 换算成 head 通道里 jump 的占比 —— **改规则集就要重算 m**（§4.2）。
  若 `die` / `getup` 出现"动得太多"的退化，第一个旋钮仍是降 `--aux_group_mass`。
- **`clip_length_prior` 是否也该看 aux**：当前方案是不看（保持可预测）。
  若 `--num_frames auto` 在 transition 的 `jump` 上继续给出偏短的 22 帧，可作为后续开关。
- **§1.4 的 5 条 `fall` 改判**（尤其 `IAC_Mammoth_JumpUp` / `IAC_Sabertooth_JumpUp`）
  需要人工看视频定夺，与本方案独立 —— 若它们确有起跳，改回 transition 的 `jump, up`
  可再加 2 个四足物种。规则 F 让这 5 条至少进了 transition 的训练集（null 分支），
  但**没有**改变这件事：它们的 head 通道仍然一个 `jump` 都没有。
