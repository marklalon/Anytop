# 一个模型训练全部 action group（`--action_group all`）

2026-09-20，分支 `train/v22`。

## 1. 结论

三个 group 合并成**一个** checkpoint，**不加** group 条件。依据是一个可以直接验证的语料性质：
`action_label` 的**首个主词**在全部 3635 条 clip 上唯一决定 action group，没有例外。
group token 因此不携带 label 之外的任何信息。

同时落地的两件事：

- **aux group 机制整体删除**。它存在的唯一理由是"每组各训一个模型"下把一条 clip 借给另一组的
  训练池；全语料训练时那条 clip 本来就在语料里，机制退化成恒等。
- **`--action_label_adaln`**：给 action 条件加一路乘性通道，见
  [conditional_modulation_upgrade.md](conditional_modulation_upgrade.md) §3。

这不是第一次尝试。2026-09-14 的 `feat/all_group` 用的是"合并 + group token + 组质量 1:1:1"，
在 transition 上与 `merged_transition_v17` 同 sample 数持平但**更不忠实**（见 §5）。本轮改的是
条件通路和容量，不是再调一次组质量。

## 2. 语料性质：label 决定 group

全量 3635 行 `action_labels.jsonl`（三个数据集）实测：

| 口径 | 结果 |
|---|---|
| 首词 → group | **唯一，零例外** |
| 完整 label 串 → group | 唯一（`draw` 曾有一条 stationary 误标，2026-09-20 改回 transition） |
| 387 个不同 label 的 4 通道条件向量，最近邻跨组 | 16 个 label / 301 条 clip（8.3%） |

第三行是模型实际看到的东西，也是本方案唯一真正的风险面。最近的几对：

```
0.836  'turn, left'[transition]     <-> 'run, turn, left'[locomotion]    (98 clips)
0.836  'turn, right'[transition]    <-> 'run, turn, right'[locomotion]   (96 clips)
0.810  'jump, forward'[transition]  <-> 'swim, jump, forward'[locomotion]
0.783  'hover, turn, left'[loco]    <-> 'turn, left'[transition]
```

对照：**同组**最近邻 cos 的中位数是 0.854。所以对 turn / jump 这一族，跨组边界并不比组内边界宽。

两条成因，都是结构性的，不是标注问题：

1. 契约 6 的加权主词池化（[action_label_per_word_pooling.md](action_label_per_word_pooling.md) §6）
   把每个主词都放进 head 槽，266 条双主词 clip 的 head 通道因此是跨组混合
   （`run, turn` 的 head 通道对 `turn` 是 cos 0.61）。
2. 条件表示是**四个单位向量的拼接**。两个 label 若共享 direction 槽、只差一个主词，
   拼接后的 cos 有下界 `(cos_head + 1) / 2 >= 0.5`，与主词权重 `r` 无关。
   实测 `r` 从 1.5 提到 4.0，`turn, left` / `run, turn, left` 只从 0.84 降到 0.72。

**这不作为缺陷处理。** 这两个动作在动作空间里本来就相邻，硬把它们分到两个模型里才是问题——
让它们在同一个模型里互相泛化正是合并的目的。`r` 保持 1.5，不因为这个数字去调它。
需要盯的是**物理伪影**而不是语义邻接：原地 turn 的足部滑步（`feat/all_group` 上
turn skate 从 0.540 涨到 0.675）要按伪影验收，不能因为"turn 本来就像 run-turn"而豁免。

## 3. 采样质量：`--balanced` 取代组质量参数

`feat/all_group` 有一个 `--action_group_weights`，用来避免 stationary 压倒其余两组。本轮**不要**
这个参数：`--balanced` 现在按 label 首词分组（[action_balanced_sampling.md](action_balanced_sampling.md)），
而首词就是 group 的函数，所以组间质量是首词平衡的副产品。

| 采样方式 | locomotion | stationary | transition |
|---|---:|---:|---:|
| 逐 clip 均匀 | 25.8% | 52.9% | 21.3% |
| `--balanced`（首词 sqrt） | 27.9% | **37.6%** | 34.5% |

已经接近 1:1:1，再加一层组质量只会和首词质量互相抵消。**`--balanced` 对 `--action_group all`
是必需的**，不是可选项——三个 `train_*.bat` 历史上都没传过它。

## 4. 删除 aux group

删掉的面：

- `motion_labels`：`AUX_ACTION_GROUPS_KEY`、`_validate_aux_action_groups`、`aux_key_present_in`，
  以及 `load_action_labels` / `load_motion_metadata` 的 join。常量降级为
  `_RETIRED_AUX_ACTION_GROUPS_KEY`，只用于把旧 `motion_metadata.json` 里的残留键剥掉。
- `dataset.py`：`aux_action_groups_of`、`require_aux_group_sidecars_migrated`、
  `filter_motion_names_by_aux_action_group`、`load_aux_motion_names_for_train`、
  `load_aux_motion_names_per_source`、`MotionDataset.aux_mask` / `aux_motion_names` /
  `aux_group_mass`，以及 `TruebonesSampler` 的双池预算（现在只有一个池）。
- `tensors.py`：`y['is_aux']` 通道。
- `--aux_group_mass`、review UI 的 aux 筛选与"移除 aux group"菜单、
  `tests/test_aux_group_membership.py`。
- 三个 `action_labels.jsonl` 里 68 行的 `aux_action_groups` 键（只删键，label 与 group 未动）。

`load_action_labels` 忽略未知键，所以即使不清理 sidecar 也不会报错；清掉是为了不留死字段。
`MOTION_METADATA_SCHEMA_VERSION` **不动**（它只是 review payload 的戳），不触发任何 regen。

`docs/aux_group_and_head_word_augmentation.md` 已作废。

## 5. 与历史结果的关系

`merged_all_v3`（合并 + group token，2026-09-15 对比 `merged_transition_v17`，同 sample 数）：

- transition 的 `l_simple` 追平，但 101 个 in-dist bucket 上 pos_err 0.130 对 0.115（v17 在 76/101 上更好）；
- turn skate 0.675 对 0.540（GT 0.439）——最大的单项回归，正好落在 §2 的混淆族上；
- jerk / GT 1.08 对 1.24、diversity +20%，即**更平滑、更多样、更不忠实**。

那次读数是"cfg=1 下物种先验盖过了弱的加性 label"。本轮的两处改动正对着这一点：乘性通路
（§6）和容量（§7）。**没有**保留 group token 去补 margin——按 §2，它补的是 margin 而不是信息，
而 margin 应该由条件通路自己给出。

## 6. 条件通路

`--action_label_adaln`，设计与验收见
[conditional_modulation_upgrade.md](conditional_modulation_upgrade.md) §3。要点：
零初始化、只调制 temporal 与 FFN 的**分支输入**（残差不动）、由加性路径同一个 `action_repr`
驱动所以 CFG 的无条件分支自动走 null。

## 7. 容量

语料从约 1/3 变成全部，模式数随之。`latent_dim` 256 → **384**，`ff_size` 保持 2048。

| 配置 | 参数量 | it/s（220 步，无 `--compile`） |
|---|---:|---:|
| 256 | 16.97M（+adaln 19.14M） | 5.01 |
| **384** | 28.79M（+adaln **33.67M**） | **3.89** |

+70% 参数只换来 +29% 步时——这一步部分是 kernel launch 绑定而不是 FLOP 绑定
（见 [fp16_vs_bf16_precision.md](fp16_vs_bf16_precision.md) 附近的 185→151ms 那次优化）。
显存不是约束。

## 8. 推理侧

checkpoint 的 `args.json` 记 `action_group: all`；`apply_checkpoint_action_group` 把它读成**空**
group，下游一律理解为"任意 group"：

- `sample/conditioning.py` 原本对"没有 group 却给了 label"硬退出，现在放行——label 自己带着 group。
- `utils/clip_length_prior.py` 的两个 matcher 在 group 为空时只比 label 部分
  （`_group_matches`）。表里的 key 仍是 `group|label`，所以**不需要重新 bake cond.npy**。
- 三个 `eval/eval_tasks_*.json` 指向同一个 `RUN_NAME`。
- resume 守卫把 `all` 当成一个独立语料：`all` ↔ 单组互相 resume 都会被拒。

## 9. CKPT_VERSION

**不 bump**（仍是 18）。`--action_label_adaln` 默认关，关闭时不创建任何参数，state_dict 与之前
逐键相同；`--action_group all` 与 `--balanced` 改的是语料和采样分布，不改张量含义。

代价是**没有守卫会提醒你前后两段分布不同**，所以必须换 `RUN_NAME`（`merged_all_v22`），
不要在旧 run 上 `--auto_resume`。

## 10. 现存对照基线的限制

只有 `merged_transition_v21a` 记的是 `version=18`；`merged_locomotion_v20` / `merged_stationary_v20`
都停在 `version=14`，已经不能再推理。所以 v22 的严格 A/B **只有 transition 有同契约基线**，
另外两组只能看 eval battery 的绝对分数。要补上，需要各重训一个同契约的单组模型。

## 11. 复现

```
Anytop\train_all.bat
```
