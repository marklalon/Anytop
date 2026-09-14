# 单模型训练全部 action group

> 状态：**代码已落地（2026-09-14），未训练。** 训练入口是
> [train_all_groups.bat](../train_all_groups.bat)（`merged_all_v1`），对照基线是正在训练的
> `merged_locomotion_v15`（[train.bat](../train.bat)）。
>
> **不需要重生成 cond，也没有 bump `CKPT_VERSION`。** 所有新开关默认关闭；关闭时
> state_dict、训练和推理语义都与改动前逐位一致，v15 等单组 checkpoint 仍能 resume 和采样。

## 1. 为什么现在可以再试

之前三组分别训练，是因为早期把所有动作放进一个模型时出现了近似单峰塌陷。
那次实验（`save/truebones_zoo_all_no_pior_v1`，2026-06-30）和现在有三处本质不同：

| | 当时 | 现在 |
|---|---|---|
| 数据 | 只有 zoo | zoo + zoo_upgrade + unitybundles，3781 条 |
| 动作条件 | 14 类 multihot `action_tags` | 受控词表，每个角色槽一个 T5 通道，transition 带 `R_B` 顺序角色 |
| 训练步数 | 盘上只到 30k | 单组 checkpoint 训到 200k |

当时 `idle` 一个 tag 就有 298 条，`attack` 有 256 条，并且跨三组。同一个 (species, tag)
底下是几种不兼容的动作，x0 落在它们中间，这和后来在 locomotion_v4 上查出的
「label 桶内离散度」是同一个机制。

现在的实测（2026-09-14，全部 3781 条标注）：

- 模型实际能看到的条件（词 + 角色）共 **461** 种；
- 跨组撞到同一条件的只有 5 种，人数少的一组合计 **9 条 clip（0.24%）**；
  再算上 `is_loop` 是 4 种、7 条：`die`（stationary 2 条）、`jump`（transition 2 条）、
  `swim, turn, left/right`（locomotion 与 transition 各 2 条）、`idle, block`（transition 1 条）；
- `idle, crouch` 这种同一字符串同时出现在 stationary 和 transition 的情况，
  已被 transition 的 `R_B` 区分开；
- `audit_action_labels.py --action-group all`：R1 有 8 个桶超阈值（6 个物种），R3/R4/R5 均为 0。

所以合并后 p(x | species, label, is_loop) 基本不会比分组时更多峰。剩下的风险是：

1. **采样比例。** 按 clip 均匀采样时 stationary 占 56%，locomotion 每步看到的样本只剩约 1/4；
2. **CFG 的无条件分支变宽。** 无条件模式从「组内的平均」变成「全库的平均」，
   `cfg_scale > 1` 时外推量变大；
3. **加性条件注入的容量。** 所有条件 token 都加到 timestep embedding 上，每层作为同一个 bias
   加给全部 (帧, 关节) token；要区分的模式多了约 3 倍。

§2 处理前两条，第三条是 §5 的备选方案。

## 2. 设计

### 2.1 训练

| 开关 | 值 | 作用 |
|---|---|---|
| `--action_group all` | | 不按组过滤 clip。R_B 角色只看每条 clip 自己的 group，这部分本来就是逐 clip 的 |
| `--action_group_weights` | `1,1,1`（默认） | 各组的采样质量，顺序为 locomotion,stationary,transition |
| `--action_group_cond` | 开 | 组 embedding，只能配合 `all` 使用 |
| `--action_group_cfg_drop_prob` | 0.15 | 与 `--action_label_cfg_drop_prob` 各自独立抽样 |

**采样器**（`TruebonesSampler`）分两层，各自可选：

- **组**：组 g 的质量是 `w_g / Σw`，只在有 clip 的组之间归一；
- **物种**（`--balanced`）：组内按 sqrt(clip 数) 分给物种；不开时组内按 clip 均匀。

不带组权重、开 `--balanced` 时整个子集是一个质量为 1 的分区，权重与旧实现逐位相同。
等权时每组的组内分布和单组训练完全一样。

**group token**：`nn.Embedding(len(ACTION_GROUPS) + 1, latent_dim)`，最后一行是 null，
全部行零初始化，加到 timestep embedding 上（排在 action label token 之后）。
- collate 在 host 端生成 `y['action_group_id']`（ACTION_GROUPS 下标，没有 group 时为 -1），
  和 group 字符串一起发出，保证每个 batch 的 key 集合不变；
- 训练时以 `action_group_cfg_drop_prob` 的概率换成 null；`y['action_group_active']` 可以显式覆盖；
- 这个 embedding 不做 weight decay，理由和 `unreliable_embedding` 一样：它是零初始化的加性通路，
  decay 会把它拉回「通路关闭」的状态；
- 刻意不把 group 写进 label 文本或 T5 通道。

group drop 和 label drop 相互独立，因此模型四种组合都会见到：(g, l)、(g, ∅)、(∅, l)、(∅, ∅)。

**日志**：`all` 训练会额外记录 `l_simple_locomotion` / `l_simple_stationary` / `l_simple_transition`，
加权方式与总的 `l_simple` 相同。

**步数与学习率**：等权时每组拿到 1/3 的样本，所以合并模型的第 3S 步对应单组训练的第 S 步：
600k 对 200k，150k 对 50k。`StepLR` 同理从每 10k 步衰减一次改为每 30k 步，
使「学习率 vs 该组已见样本数」的关系和单组训练一致。batch 16、lr 等其余超参都沿用 v15。

启动时，`validate_action_group_options` 会在动 save_dir **之前**检查以下错误组合：
- `--action_group_cond` 配单组；
- `--action_group_weights` 配单组；
- 权重个数不对、为负数、非有限值，或全部为 0。

resume 时，`all` 被当作一个独立的语料：`all` ↔ 单组之间不能互相续训。

### 2.2 推理

`sample/generate.py` 新增 `--action_group`。`parser_util.apply_checkpoint_action_group` 按下表处理：

| checkpoint 记录 | 不给 `--action_group` | `--action_group` = 自己的组 | `--action_group` = 其他组 |
|---|---|---|---|
| 单组 | 以该组运行（与以前相同） | 以该组运行 | **报错** |
| `all` | 不带组（全库） | 以请求的组运行 | 以请求的组运行 |

`args.checkpoint_action_group` 记录 checkpoint 训练时用的组。

`_resolve_action_condition` 的行为：

- **带 label**：必须有 group。`all` checkpoint 不给 `--action_group` 会报错，
  因为 transition 第二个主词的 `R_B` 角色取决于 group。
- **只有 group、没有 label**：模型开了 `--action_group_cond` 时，返回
  `{'action_group': g, 'action_label': '', 'action_slots': None}`，只设置 group token，
  label 通路走 null。没开 group cond 的 `all` checkpoint 会打印 warning 并按无条件生成。
- **CFG**：`ClassifierFreeActionModel` 的无条件分支只把 `action_label_active` 设为 False，
  group token 保持不变，所以外推基准是 **ε(g, ∅)**，也就是该组自己的无条件模式。
  这让 locomotion 上标定的 `cfg_scale 2` 大体还能沿用（仍需重扫，见 §3）。
  `--action_label_cfg_scale ≠ 1` 仍然必须带 label。

### 2.3 服务端

- 一个组的槽位（`PCVG_ANYTOP_MODEL_PATH_<GROUP>`）可以指向单组 checkpoint，也可以指向 `all` checkpoint；
- 几个槽位指向同一个文件时只加载一次，由这几个组共用；
- 请求里的 `action_group` 会转发给 generate（以前是故意不转发的）。单组 checkpoint 只接受自己的组，
  启动时的 `_assert_checkpoint_declares_group` 保证路由不会配错。

### 2.4 评估

- [eval_tasks.json](../eval/eval_tasks.json) 里每个任务都显式写了 `--action_group`；
  新增 `Stationary`、`Transition` 两类任务，其中包括一个只给 group 不给 label 的任务；
- 单组 checkpoint 会**跳过**其他组的任务：不写进报告，也不算失败。因此同一套任务能同时评
  v15 和 `merged_all_v1`；
- 任务可以用 `--action_words` 指定打分用的参考先验，不写时仍是 `walk,run`；
- 注意：已有任务的参数变了（加了 `--action_group locomotion`），参数校验和随之改变，
  旧 checkpoint 的增量评估会把这些任务重新生成一次。

## 3. 对比协议

**基线**：`merged_locomotion_v15`，用当前代码、同一份数据训练。stationary 和 transition
在当前架构下没有单组基线（用户决定不另训），这两组只看绝对指标。

**样本数对齐**：`merged_all_v1` 的第 3S 步对 v15 的第 S 步。两边都每 5k 步存一次 checkpoint，
所以对应的 step 都能直接取到。

**CFG**：两边都扫 `cfg_scale ∈ {1, 2, 3, 4}`，比较各自的最优值，而不是在同一个 scale 上比。

| 看什么 | 塌陷时的表现 |
|---|---|
| 生成动作与数据的关节速度 RMS 之比，按 (species, label) 算 | 被平均化：明显小于 1 |
| 同一条件下多个 seed 两两之间的距离，除以数据桶内的离散度 | 模式塌陷：趋近 0 |
| 用 action separability 看跨组串扰 | locomotion 标签生成的动作落进 stationary 的簇 |
| locomotion 专项：同手同脚比例（gait_metric）、loop 首尾闭合、root 漂移 | 比 v15 差 |
| transition 专项：起止姿态距离、A→B 方向是否正确 | die/getup 没有真正完成转换 |
| `l_simple_<group>` 与 v15 在对应 step 的对比 | 明显偏高 |
| 同一 checkpoint 上 (g, l) 与 (∅, l) 的差异 | 衡量模型对 group token 的依赖程度 |

**止损**：跑到 150k 步时和 v15 的 50k 做一次 locomotion 对比。如果明显落后就停，直接进 §5。

## 4. 与单组 checkpoint 的兼容性

- 新模型参数全部挂在 `action_group_cond` 开关之后，默认关闭，所以单组模型的 state_dict 没有新 key；
- collate 多发一个 `action_group_id`，没开 group cond 的模型不会读它；
- 单组 checkpoint 在不带 `--action_group` 时的推理结果与改动前相同；
- 因此不 bump `CKPT_VERSION`。bump 的话，正在训练的 v15 一旦需要 resume 就会被拒。

## 5. 出现塌陷症状时的备选方案（尚未实施）

按改动从小到大：

1. 调组权重，比如给 locomotion 加重；
2. 条件注入改成 AdaLN-Zero：用条件向量对每层 norm 的输出做 scale + shift（zero-init），
   替换现在给所有 token 加同一个 bias 的方式。DiT 的对比实验里这种方式明显强于加性注入；
3. latent 从 256 提到 384。

不要把 action token 做成和 species 耦合的函数，这会破坏零样本迁移依赖的可分离性
（见 `project_action_label_spread_not_uniqueness` 的讨论）。

## 6. 改动文件

| 文件 | 改动 |
|---|---|
| `utils/parser_util.py` | 训练 `--action_group` 增加 `all`；新增 `--action_group_weights`、`--action_group_cond`、`--action_group_cfg_drop_prob`，以及生成用的 `--action_group`；`apply_checkpoint_action_group` 按 §2.2 的表重写 |
| `train/train_anytop.py` | `validate_action_group_options`；resume 时把 `all` 当作独立语料 |
| `data_loaders/truebones/data/dataset.py` | `resolve_action_group_weights`；`TruebonesSampler.compute_weights` 支持按组分层 |
| `data_loaders/get_data.py` | 透传 `action_group_weights` |
| `data_loaders/tensors.py` | collate 生成 `action_group_id` |
| `model/anytop.py` | `action_group_embedding`、`_build_action_group_token` |
| `model/cfg_sampler.py` | 文档：无条件分支保留 group token |
| `utils/model_util.py` | 透传两个模型参数 |
| `train/training_loop.py` | `l_simple_<group>`；group embedding 不做 weight decay |
| `sample/generate.py` | `_resolve_action_condition` 支持只给 group 的请求和 `all` checkpoint |
| `eval/eval_checkpoint.py`、`eval/eval_tasks.json` | 按组跳过任务；`--action_words`；新增 stationary/transition 任务 |
| `server/anytop_service.py` | 接受 `all` checkpoint，同一文件只加载一次，转发 `action_group` |
| `train_all_groups.bat` | `merged_all_v1` 的训练配置 |
| `tests/test_unified_action_group.py`、`tests/test_action_group_checkpoint_binding.py` | 覆盖以上改动 |
