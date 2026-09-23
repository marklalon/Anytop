# action_label：方向槽 dropout

> 状态：**已实施**（2026-09-18；方案本体见下，实际落地与方案的出入记在 §6）
> 触发：裸 `attack, swat` 出双臂混合而非单侧
> 影响：`ACTION_LABEL_PARSER_CONTRACT_VERSION` 4 → 5、`CKPT_VERSION` 15 → **16**，**需从头重训**，不需 cond regen
> 相关：[`action_label_per_word_pooling.md`](action_label_per_word_pooling.md)、[`tools/audit_action_labels.py`](../tools/audit_action_labels.py)
> 后续：2026-09-21 modifier 槽也加了同构的 dropout（`--modifier_slot_drop_prob`），∅ 同样是边缘；
> 见 [`head_word_weights_and_modifier_dropout.md`](head_word_weights_and_modifier_dropout.md) §3。

## 0. 先固定语义（本节改动依赖它）

| 槽 | ∅ 的含义 | 由谁定义 | 训练时 dropout |
|---|---|---|---|
| direction | **未指定 = 边际**（任一方向） | 训练时随机抹掉方向词 | 是 |

推理侧：裸 `attack, swat` = 任一侧。

标注规则：

- **方向**：内容有单一主导侧向或平面朝向 → 写方向词；对称、交替、旋转、原地 → ∅。
  stationary 只补 `left` / `right`（stationary 按定义无净位移，`forward` / `backward` 不补）。
  **例外：本身不指向任何方向的动作一律不写平面方向词**（2026-09-19，见 §7）——
  `hurt` / `getup` / `idle` / `rest` / `stop` / `draw` / `sheathe` / `headbutt` / `bite`。
  竖直词 `up` / `down` 不在豁免内（`idle, up, aim, bow` 保留）。

不变量（补标脚本遵守）：

- **已有的方向词是人工核验过的真值，没有错标，只可能漏标。** 脚本只填 ∅ 行，永不改写已有词；
  标定把已有词当真值来校准判据的符号和阈值。
- **判定只看内容。** clip 名是最弱的证据，只写进 `evidence` 供人核对，不参与判定，也不用来推断「疑似误标」。

## 1. 方向槽

### 1.1 模型：方向槽 dropout

- 位置：[`model/anytop.py`](../model/anytop.py) `_action_condition`，取到 `word_ids / slot_ids / word_mask`
  之后、调 `_assemble_action_slot_channels` 之前：

  ```python
  if self.training and self.direction_slot_drop_prob > 0.0:
      drop = torch.rand(batch_size, device=device) < self.direction_slot_drop_prob
      word_mask = word_mask & ~(drop[:, None] & (slot_ids == SLOT_DIRECTION))
  ```

  纯 mask 运算，和现有 `_resolve_action_label_active` 的 CFG 抽样同一写法，不影响 compile。
  没有方向词的样本天然不受影响。
- 参数：`--direction_slot_drop_prob`，放在 [`parser_util.py`](../utils/parser_util.py) `--action_label_cfg_drop_prob`
  同组，写进 `args.json`；推理侧永远不 drop。默认 **0.15**（2026-09-21 由 0.3 改为 0.15，与
  `train_all.bat` 一致；下面"0.3 的来由"记录当时的推导）。
- **监督预算（0.3 的来由）**：`--action_label_cfg_drop_prob`（0.2）是按样本丢掉**整条**标签，方向 dropout 只在
  留下来的行上再抹方向词，所以带方向词的行实际只有 `(1 − 0.2) × (1 − p)` 的 batch 贡献显式方向监督：
  p = 0.5 → **40%**、p = 0.3 → 56%、p = 0.2 → 64%。验收项「left 与 right 可辨」直接吃这个比例，故默认取 0.3；
  仍偏弱就降到 0.2，或让两种 drop 在同一行互斥（同一行不同时丢）。
- `action_label_valid` 不受影响：方向词不可能是标签里唯一的词（`parse_action_label` 硬要求至少一个 HEAD 词，已验证）。
- 与 `--action_label_cfg_drop_prob`（0.2）独立叠加；训练路径没有 L/R 镜像增广，无需考虑顺序。
- 测试：drop 后 direction 通道为零行、其余三通道逐位不变；`eval()` 模式无 drop。
- 不改词表、不动 fingerprint；重训由 §2 的契约变更一起触发。

### 1.2 数据：`tools/prefill_direction_words.py`（方向漏标自动补标）

判定**只用内容**；clip 名只写进 `evidence` 字段供人核对，不参与判定。

**A. stationary / transition 的 `left` / `right`（主体工作量）**

- 对 `cond.is_symmetric` 的物种算自镜像距离 `asym = ‖A − mirror(A)‖ / ‖A‖`
  （`symmetry_partner_indices` 交换 + x 取反；Jaws 的 L/R 对镜像后距离 0.0015，度量已验证）。
  - `asym < 0.15` → 对称，保持 ∅，不动。
  - `asym ≥ 0.5` 且无方向词 → 判侧：按 `cond.joint_side_labels` 把关节分 left / right，算两侧肢体速度能量比 `r`；
    `|r − 0.5| ≥ 0.2` 写侧词，否则（交替连击、旋转、居中）留 ∅ 但进复核清单。
  - `0.15–0.5` → 不改标签，进复核清单。
- **标定门槛**：先在已标行上标定符号（label 里的 `left` 指的是哪一侧的肢体在动），一致率 ≥ 90% 才允许写。
  以前用的速度加权 x 均值只有 70%，不能用。不达标就只出清单。
  - 标定集 = stationary 的 105 条 + transition 里**非 `turn`** 的 6 条。
  - transition 的 200 条 `turn, left/right` 已经全部带方向词（偏航方向，不是肢体侧），
    不在脚本范围内，也不进肢体能量的标定集。
- 首批必处理：8 对内容镜像对（zoo_upgrade Elephant AtkL/R、MB_Unka ×3、MU04_ScorpionKing Left/RightStrike、
  TTR_LightInfantry PunchA/B、RU01_Metal/TankerRobot Left/RightProjectile）。

**B. locomotion 平面朝向**

- 缺朝向的 locomotion 行很少（R4 报 16 条量级：`run, jump` / `swim, jump` / `swim` / `fly, roll` 等）。
- **根位移不可用**：预处理对每一条 locomotion clip 做根 XZ detrend
  （`dataset_pipeline`：`flatten_root_travel = is_locomotion or (is_transition and is_loop)`），
  净位移为 0，`.npy` 和 `bvhs/` 里只剩周期内的 surge / sway。实测 651 条 locomotion 行净位移中位数 0、最大 0.2 残差，
  不能拿它判方向。
- **唯一判据 = 支撑脚法**：`contact_joints` 触地帧相对根的 XZ 速度取反的均值 = 行进方向（相对朝向）。
- 标定：以已标的 `walk, forward` / `backward` / `left` / `right` 行做混淆矩阵，准确率 < 95% 不自动写。
- `swim` / `fly` / `hover` 无接触 → 留 ∅ 进清单。

**C. transition 的 `jump`**

原则：方向槽有 dropout 后，∅ 只能是「边际」，不能再承担任何具体内容。「原地竖直跳 = 裸 `jump`」违反这一点——
裸 `jump` 会同时收到竖直跳和被 dropout 抹掉 `forward` 的前跳，推理时既分不清也没法显式要竖直跳。
竖直跳有明确方向（`up`），和 `fly, up` 同性质，R4 的 `VERTICAL_WORDS` 本来就把 up / down 当合法朝向。

| 内容 | 拼法 | 现状 |
|---|---|---|
| 原地竖直跳（transition） | `jump, up` | 已有 2–3 条；37 条无方向的 `jump[, happy]` 按内容改 `up` 或 `forward` |
| 前跳（transition） | `jump, forward` | 已有 ~10 条 |
| 行进中的跳（locomotion） | `run, forward, jump` / `swim, forward, jump` | 10 条待补 |
| 裸 `jump` | 只由 dropout 产生 | 推理时 = 任意跳 |
| `hover` | 保持 ∅ | 原地持续悬停，内容上无朝向；留在 `NO_HEADING_WORDS` |
| `fall` / `dive` / `land` | 不变 | 竖直方向由词本身承担，无 up / forward 歧义 |

判据：起跳前 / 落地后触地帧的支撑脚水平速度（有 → `forward`，无 → `up`）+ 根竖直位移（确认是跳）；
判不了的进清单。非 loop 的 transition 不走根 XZ 压平（见上面的 `flatten_root_travel` 条件），
其根 XZ 净位移只作交叉核对，不单独作判据。

**D. 审计工具衔接**

- R4：`NO_HEADING_WORDS` 去掉 `jump`（[`audit_action_labels.py`](../tools/audit_action_labels.py)；这个常量同时是
  `--r4-exempt-words` 的默认值，CLI 默认跟着变），`run / swim / roll + jump` 必须有朝向。
  R4 只查 locomotion（`check_r4` 开头的 `if clip["group"] != "locomotion": continue`），所以这只影响 10 条
  locomotion 行；transition 的 `jump` 行不靠 R4，靠补标脚本 + 复核。
- R3：**不扩展**名字解析。`clip_side` 只认尾缀是有意的（`LeftFoot` / `FlyLeftWing` 是部位名，前缀匹配就是误报），
  `PunchA / PunchB` 也不是侧向标记。这 8 对由 §A 的内容镜像检测直接给出，不需要 R3 再找一遍。
  （**这份忽略名单已删，见 §8**。）

**写回（两个脚本共用）**

- 就地按行重写，沿用 `fill_missing_loop_flags` 的方式（读 bytes、保留换行风格、只重写变化的行、
  tmp + `os.replace`）。抽成共用的行级重写 helper，别复制一份。
  （**出处已移**：共用的 `read_action_label_rows` / `rewrite_action_label_rows` / `autofill_action_label`
  现在在 `tools/action_label_sidecar.py`，不在 `motion_labels`。）
- 改 `action_label`；写 `"reviewed": false`（现在 3714 行全是 true；review UI 按 truthy 判断，false 等价未审，
  且比删 key 多保留「曾审过、被自动改动」的信息；serve.py 自己取消审核是删 key，两种拼法并存无碍）。
- 加 `"autofill"` 标记。`load_action_labels` 只读 clip / action_group / action_label / is_loop，
  行级额外字段已被容忍（`pending_delete` 就是先例）。
  （原方案记 `{"slot", "from", "evidence"}`，**已收缩成 `true`，见 §8**。）
- 默认 `--dry-run` 输出清单（CSV + 复用 `audit_report_html.py` 出带 GIF 的页面），`--apply` 才写。
  （**页面已删，见 §8**。）

## 3. 执行顺序与验收

1. 脚本 `--dry-run`，看标定准确率，定阈值。
2. `prefill_direction_words --apply` → 1.1 代码。
3. 预检：`audit_action_labels.py --cond-path dataset/merged/cond.npy --action-group all`（R3 / R4 / R5）、tests。
4. 人工：review UI 过滤 `reviewed=false` 行逐条看 GIF。
5. 同步 `relabel_actions_llm.py` 的 prompt（方向规则「单一主导侧向 → 写方向词」），
   `dataset/readme.txt` 的预填那一步（现在只有 `prefill_loop_flags.py`）加 prefill 工具；
   `action_label_per_word_pooling.md` §6 记决策。
6. 重训（契约变了，无法 resume）。

验收：

- 裸 `attack, swat` 出单侧而非双臂混合；`attack, left, swat` 与 `right` 可辨。

## 4. 影响面速查

| 项 | 数量（2026-09-18 工作树） |
|---|---|
| 标签行 | 3714（unitybundles 2534 / zoo 937 / zoo_upgrade 243），全部 `reviewed: true` |
| stationary 有 left/right | 105；transition 有 left/right 206（其中 `turn` 200） |
| locomotion 无方向词 | 95（含 hover / fall / dive 等豁免；`Dragon_WyvernIdle` 转 stationary 后少一条） |
| transition 无方向 `jump*` | 37；`jump, up` 2；`jump, forward` 10 |
| locomotion `run, jump` / `swim, jump` | 5 / 5 |

## 5. 对原始草案的审核修正

原草案整体成立：dropout 位置与写法、∅ 语义、内容优先于名字、根位移不可用 → 支撑脚法、先标定再写、
写回方式都对。以下是核对代码和数据后补正的：

1. **把「只补不改」写成不变量**（§0）。已有方向词是人工核验过的真值；脚本只填 ∅ 行，标定把已有词当真值。
2. **标定集排除 `turn`**。原草案「已标 102 行」实为 stationary 105 + transition 非 turn 6；transition 另有
   200 条 `turn, left/right`，已全部带方向词，是偏航方向不是肢体侧，不进肢体能量标定。
3. **R3 不扩展前缀 / A-B 匹配**。`clip_side` 只认尾缀是防部位名误报的既定设计；8 对镜像由内容检测给出。
4. **审计生成产物不算手改触点**：`audit_action_labels.html` 重跑自动重写，从原草案的触点列表里删掉。
5. **写回 helper 的出处**是 `fill_missing_loop_flags`；抽共用而不是复制。
   （后续调整：行级读写搬到 `tools/action_label_sidecar.py`；`fill_missing_loop_flags` 本身搬回
   `prefill_loop_flags.py`——`--rejudge` 与「`reviewed` 不被覆盖」是那个工具的策略；
   `motion_labels` 只留键位规则 `set_loop_flag`，review UI 也走它。「抽共用而不是复制」这条不变。）
6. 补充：stationary 只补 left / right 的边界；根 XZ detrend 的代码出处与实测数字（§1.2 B）；
   LLM 标注 prompt 与 readme 要同步，否则下一批数据又按旧规则标。
7. **行号引用改符号名**（本次复核）。原草案写了 `motion_labels.py` / `parser_util.py` / `relabel_actions_llm.py` /
   `audit_action_labels.py` 的行号，一改就过期；改成 `_VOCAB_T5_TEXT`、`check_r4`、
   `NO_HEADING_WORDS`、`--action_label` 帮助文本、`"load"` 词元组与 transition prompt 这样的符号引用。
8. **方向 dropout 与 CFG drop 的监督稀释**（§1.1）：两者叠加后带方向词的行只有 `(1 − 0.2) × (1 − p)` 的 batch
   贡献显式方向监督，p = 0.5 时只剩 40%，故默认从 0.5 改为 0.3。
9. **审计工具不认 `pending_delete`**：`tools/audit_action_labels.py`（R1 / R3 / R4 / R5）不过滤 pending 行，
   而 `relabel_actions_llm.py` 会跳过。74 条待删 clip 仍会被 R1 的桶离散度当作正式数据报出来 —— 要么给审计加
   同样的过滤，要么先 retire 再审计。


## 6. 落地记录（2026-09-18）

代码与数据都已改完，**只差重训和人工过 GIF**。与方案的出入，按发现顺序：

### 6.1 标定结果与实际写入

三条判据都在已标行上标定过，门槛写在工具里（`--side-gate` / `--heading-gate` / `--jump-gate`）：

| 判据 | 标定 | 门槛 | 结果 |
|---|---|---|---|
| 侧向（自镜像分解 + 左右肢体能量比） | 60/63 = 95.2%（62 条判不了） | 90% | 开，写 183 stationary + 44 transition |
| locomotion 朝向（支撑脚相对速度） | 455/467 = 97.4%（197 条判不了） | 95% | 开，写 5 条（`run, jump` 类） |
| jump 方向（腾空根位移） | 13/13 = 100%（2 条判不了） | 90% | 开，35 条 `jump` → `jump, up`，其余按位移写平面词 |

- **方向**共写 232 行（unitybundles 177 / zoo 43 / zoo_upgrade 12），804 行进复核清单
  （`dataset/review/prefill_direction_words.{csv,html}`）。当时 `autofill` 里用 `earlier` 叠着记
  （**记法已作废，见 §8**）。

### 6.2 与方案不同的地方

1. **`turn` 行整体退出方向工具**（不只退出标定集）。方案只说 transition 的 200 条 `turn` 不进肢体能量
   标定集；实际上 locomotion 的 `walk/run, turn, left` 也一样——那个 left 是偏航方向，支撑脚法量的是
   行进方向，两者在转弯 clip 上本就不同（实测把 turn 行算进去，朝向标定从 97.4% 掉到 90.1%）。
   所有带 `turn` 的行现在既不参与标定也不接受补标。
2. **距离阈值按同类最近邻中位数自动定**（不是固定常数），可用 `--distance-max` 覆盖。
3. **R1/R3/R4/R5 现在过滤 `pending_delete`**（§5.10 的两个选项里取了「给审计加同样的过滤」）。
4. **审计新增「失效 ignore 条目」告警**：`action_label_audit_ignore.jsonl` 的 11 条旧拼写已按新标签重写，
   剩下 14 条（clip 已退役 / 标签改到认不出）每次跑审计都会列出来，不再静默烂掉。
   （**告警连同忽略名单已删，见 §8**。）
5. **review UI 加了 `自动补` 角标**：鼠标悬停显示补了哪个槽、原标签和判据；在页面上手改标签会删掉
   `autofill` 记录（工具的提议已被推翻），只点 OK 核验则保留（提议被确认）。

### 6.3 复核之后还剩什么

- 方向 804 行的复核清单（一张 html 页面并排放着 GIF 和候选值）。
- 审计仍报的 29 条：R1 15（多数是 KI_Villager 的 `work, hammer` 桶，补标前就在）、R3 1
  （Elephant AtkL/R 这对镜像，能量比判不出主导侧）、R4 11（`swim` / `fly, roll` 无触地帧）、R5 2
  （`die, fall`）。这些工具都判不了，等人看。
- 重训（契约变了，旧 checkpoint 会被指纹拒绝）。

## 7. 不指向方向的动作：平面方向词豁免（2026-09-19）

补标跑完之后定的规则：**有一类动作根本没有平面方向可写**，它们的标注一律不带
`left` / `right` / `forward` / `backward`——**已经人工 reviewed 过的行也一样改**。

| 词 | 为什么没有平面方向 |
|---|---|
| `hurt` / `getup` / `idle` / `rest` / `stop` | 受击反应或身体状态，不是朝哪儿发力；身体偏向哪边是附带的 |
| `draw` / `sheathe` | 那个侧向是刀鞘的位置，不是动作的方向 |
| `headbutt` / `bite` | 用头部发起，头是 center 关节；侧能量读到的是身体转向，不是这一击 |

常量：`tools/audit_action_labels.py` 的 **`NO_PLANAR_DIRECTION_WORDS`**（`takes_planar_direction()`），
词必须在 `ACTION_VOCAB` 里，否则 import 就报错（豁免一个不存在的词等于什么都没豁免）。
**匹配位置不限于头词**：`idle, right, look` → `idle, look`、`attack, left, bite` → `attack, bite`。
LLM 标注侧的 `reset`（回到中立起始姿势，`relabel_actions_llm.py` 的 transition 词表里有、
`ACTION_VOCAB` 里没有）是同一类，落到 sidecar 时写作 `rest`，两边都已豁免。

**竖直轴不在豁免内**：`up` / `down` 说的是动作真的向上 / 向下，和平面方向不是一回事——
`idle, up, aim, bow`（朝上瞄）、`idle, up, look` 原样保留，transition 的跳仍然按腾空位移判 `jump, up`。

三处落地（数据、工具、审计，缺一就会漂回去）：

1. **数据**：三个 sidecar 共 **93 行**去掉平面方向词。其中 54 行是这次 `prefill_direction_words --apply`
   自己补的（记录连同 `autofill` 一起删掉，`reviewed` 恢复成补标前的值），另外 39 行是人工标的真值
   （`Trex_BiteLeft/Right`、`Dog_LookLeft/Right`、`hurt, left/right` 等，`reviewed` 不动——
   剩下的词仍然是人核过的，只是少了一个规则不再允许的词）。
2. **补标工具**：`prefill_direction_words.py` 的 `kind_of()` 直接把这些行判为「不在范围内」，
   既不判也不进标定集；transition 的跳仍然留在范围内，只是平面那一半的结论改成进清单。
3. **审计**：R3 的「两边各自要带侧词」对这些行不成立——**两边拼成同一个标签就是镜像**
   （`Trex_BiteLeft` 和 `Trex_BiteRight` 都是 `attack, bite`），控制台按对数报出来；
   R4 的朝向要求同样豁免，且**不受 `--r4-exempt-words` 控制**（那个开关只替换 `hover` 那张表，
   这条是语料级规则，不是旋钮）。`not_mirror` / `crossed` 两项照查不误。
   （`crossed` 与「两边各自要带侧词」这两项已于 §9 删除：它们从 clip 名字读方向。）

`tests/test_direction_exemption.py` 把三处都钉住了，其中一条直接读真实 sidecar：
任何一次改标只要把平面方向词写回这些动作上，测试就红。

**副作用**：镜像对合并成同一个标签后（`Dog_LookLeft` / `LookRight` 现在同为 `idle, look`），
原本会被 R1 的桶离散度当成双峰桶报出来。R1 已在 §8 删除，这条不再存在。

## 8. 简化（2026-09-19）

前面几节留下的三样东西都删了：它们要么在判「可能有问题」，要么把一次性的判据写进了长期数据。

**审计只剩三条确定性文本规则**：`tools/audit_action_labels.py` 现在是 R3 / R4 / R5。
删掉的是 **R1（标签桶离散度）**——它按物种中位成对距离判桶内离散度，要解码每条 clip 的 npy、
要 `--ratio-threshold` / `--min-distance` / `--frames` 三个阈值、要 `--r1-exempt-labels` 白名单，
报出来的是「这个桶看着不像一个动作」而不是「这条标签写错了」；§6.3 里它报的 15 条正是这种。
连带删掉：距离度量（`load_trajectory` / `scale_blocks` / `pairwise_distances`）、
feature-space 警告、`--labels` 覆盖输入，以及 `audit_report_html.py` 的 R1 面板。

**忽略名单整套删除**：`dataset/review/action_label_audit_ignore.jsonl`、`--ignore` /
`--no-default-ignore`、§6.2.4 的「失效 ignore 条目」告警、报告页「无需修改」生成 ignore 行的那段。
76 条里 51 条是 R1 的桶，随 R1 一起失去意义；剩下 25 条 clip 条目多数是 §7 的豁免已经从规则上
盖掉的镜像对（bite / roar / death / hurt / getup）。三条规则都是确定性的，报出来就是真错，
不需要一份「看过了，不用改」的名单来压。

**prefill 不再出页面**：`prefill_direction_words.py` 的 `--html` 与
`dataset/review/prefill_*.html` 都删了，`prefill_common.write_html` 一并删除。补标写出来的行本来就是
`reviewed:false`，`dataset/review/serve.py` + `index.html` 已经按这个过滤，GIF 在那儿看就行，
不需要第二套页面。判据看控制台汇总和 `--report` 的 CSV。

**`autofill` 收缩成一个布尔**：`{"slot", "from", "evidence"}` 和 `earlier` 叠加链都去掉了，
现在就是 `"autofill": true`（§1「写回」与 §6.1 的那两条记法作废，`action_label_sidecar.autofill_action_label`
的签名同步简化为 `(entry, new_label)`）。理由：sidecar 是长期数据、要逐行给人读，
而 `left_share 0.253` 这种数只对写它的那一次运行有意义；语义只需要「这条是工具写的、还没人核」，
`reviewed:false` 加一个标记就够。已有的 219 行记录已就地改写。review UI 的角标同步改成「自动补标」，
不再展开判据；手改标签仍然删掉这个标记，只点 OK 核验则保留。

**几何预检删除**：`tools/evaluate_action_label_geometry.py` 与
`docs/action_label_geometry_preflight.md` 都删了。预检当时要回答的是「词级槽表示能不能用」，
那个结论已经落进契约和代码：槽源满秩 + `latent_dim` 宽度这两条硬门在
`model/anytop.py` 建模型时就断言，词表指纹在 load / resume 时 gate，
`tests/test_action_label_conditioning_contract.py` 钉住秩的具体值。
工具里唯一还有人用的是 T5 编码那几个 helper（`_resolve_t5_dir` / `_sha256_files` /
masked-mean pooling / `_postprocess_atoms`），已经搬进 `tools/build_action_label_embeddings.py`
——那是它们唯一的调用者。


## 9. 死亡的朝向，与「不从 clip 名字读方向」（2026-09-19）

起因：`action_labels.jsonl` 里还有 254 行裸 `die`，而人工标过的 41 行死亡里
`forward` / `backward` 占 30 行。补标工具一条都没补上——不是阈值太严，是**量错了东西**。

### 9.1 死亡的方向不是「侧」，是身体倒向哪边

`prefill_direction_words.kind_of()` 原来把所有非 locomotion、非 jump、非 turn、非豁免的行
都交给 SIDE 测量（左右肢速度能量份额）。对一次死亡这是两重错：

* SIDE **只能吐出 `left` / `right`**，`forward` / `backward` 它根本拼不出来。所以 30 行
  `die, forward` / `die, backward` 连标定都进不去（`if present and not sides: continue` 静默跳过），
  剩下 11 行 `die, left/right` 反过来还在用尸体四肢的抽动给 SIDE 的阈值背书。
* 一次死亡**不是用某条肢体发出的**。倒地时左右肢的能量差是余波，不是朝向。

新增第四种测量 **TOPPLE**，只管带 `die` 的行（`BODY_TRAVEL_WORDS`，匹配位置不限于头词）：

    关节质心的水平位移，clip 前 1/10 段对后 1/10 段，单位是体长；取占优轴，写一个词。

用质心而不是 root：四足原地倒下时 root 几乎不动（`BrownBear_Twitching` root 走 0.00、
质心走 0.60）。前后 1/10 取均值而不是取端点，免得一帧抖动定朝向。

**一次死亡只写一个词**——41 行真值里没有一行带两个方向词，所以 TOPPLE 不像 HEADING 那样
出对角双词。标定结果：**41/41 一致（100%）**，门槛 `--topple-gate` 0.90。

### 9.1.1 「没有明确朝向」比「走得不够远」更重要

第一版只有位移下限，结果把一批**根本不是倒地**的死亡硬标了朝向。关键在于：
**这不是阈值高低的问题**——`MLS_BattleOwl_Die` 水平飘了 2.21 体长，比 41 行真值的中位数
（1.93）还大，任何位移阈值都拦不住它。一次死亡有朝向的前提是**身体朝那个方向倒到地上**。
四种情况说明它没倒，全部判 `keep`（空槽就是对的，不进 review 清单）：

| 判据 | 常量 / 值 | 依据（41 行真值的边界） |
| --- | --- | --- |
| 标签自己写着死在空中：`hover` / `fall` / `fly` | `AIRBORNE_DEATH_WORDS` | 真值里**没有一行**带这三个词。飞行生物死了是往下掉，水平那一截是飞行残余动量 + 下落弧线，不是它「倒向」哪边 |
| 起始离地（前 1/10 段最低关节高出本 clip 地板多少） | `TOPPLE_START_CLEARANCE_MAX` = 1.00 体长 | 真值最高的一条正好是 1.00（`LH_Hero_FlyDie`，一次人工标了朝向的飞行死亡），次高 0.68。门槛卡在真值上界之上，只挡比它更高的（`MU01_Bird_Die` = 1.05）。余量很薄是有意的：这根轴分不开这两条 clip，分开它们的是人的判断，这里让着人 |
| 质心下降量 | `TOPPLE_MIN_DROP` = 0.35 体长 | 真值最小 0.36。不下降就没倒：`Cobra_Death` 走了 0.80 只降 0.21，是滑不是倒；`Pirrana_DeadFloat` / `Jaws_SharkDeadLoop` 的质心反而**上升**（死鱼浮起来），负值自然被挡 |
| 水平位移 | `TOPPLE_MIN_TRAVEL` = 0.40 体长 | 真值最小 0.46。原地瘫倒，哪边都不指 |

再加一条 `TOPPLE_TIE_SHARE` = 0.85：偏轴比（弱轴 / 强轴）超过它算没有占优轴，判 `review`。
故意放得松，因为人在 0.81 上照样只写占优轴（`KI_Warrior_Death01A` = `left`）。

第一版把 `hover` 的豁免推翻了（理由是「飘也是真的在动」），那是错的：`NO_HEADING_WORDS`
关于 `hover` 的论证本来就成立，只是它当时只挂在 R4 上。这里把它补回来。
**注意一个联动**：R5 要求 `die, fall` 写成 `die`（死亡本来就会倒）。真按 R5 改了，
`MB_TigerDrago_FlyDeath` 就失去 `fall` 这个空中标记，而它的起始离地只有 0.12、下降 1.04，
会重新被判出一个朝向——那一条改完之后要人工看 GIF 定朝向，别让工具补。

### 9.1.2 落地

`--apply` 写了 **154 行**（truebones 51 / zoo_upgrade 10 / unitybundles 93），全是 `die`；
`backward` 59、`forward` 41、`right` 33、`left` 21。剩下 100 行仍然空：
**95 行 `keep`**（12 行标签写着死在空中 = 10 `hover` + 2 `fall`、48 行起始离地超限、
26 行不下降或反向上升、9 行原地瘫倒）、**5 行 `review`**（接近 45° 的对角，
进 `dataset/review/serve.py` 看 GIF）。
写出来的行都是 `reviewed:false` + `autofill`。

### 9.2 审计不再从 clip 名字读方向

R3 之前有两项判据是拿 clip 名字当方向证据的，两项在本语料里都是错的：

* **`crossed`**（「名字带 Left 的那条写着 right」）。`MB_Unka_DeathLeft` **确实**向角色右边倒，
  标签 `die, right` 是对的，**说谎的是名字**。这一项报了 4 条，4 条全是对的标签
  （`MB_Unka_Death` / `MB_Unka_DeathDramatic` / `MB_TigerDrago_Fly` / `Scorpion-2_Strafe`）。
* **「两边各自要带侧词」**。两条同名 clip 到底是不是互为镜像的两条 take、各自又朝哪边，
  是**动作**的事实。`prefill_direction_words.py` 在那里量（位置上的镜像检测 + 侧能量），
  R3 手里只有名字，两个问题一个都答不了。

现在 R3 只剩一条：**名字配成的左右一对，两条标签必须互为镜像**。这条判据是**对称的**——
它只说这一对自己和自己不一致，不说哪一边错、也不说哪个侧词该落在哪个名字上。
名字继续用来**配对**（配对不是方向结论），这一点没变。

连带删掉：`candidate_left` / `candidate_right` / `candidate_basis`（唯一的来源就是名字），
报告页上那颗「候选（来自 clip 名字，未确认）」按钮和它的样式、`_clip_payload` 的 `suggest` 字段，
以及 `R3_PROBLEMS` 里 `crossed` / `no_side_word` 两句。控制台改成按对数报「两边拼成同一个标签」
（同时给出其中有几对是指向性动作）并指向 `prefill_direction_words.py`，不再当违规。

结果：全语料 R3 从 **6 条**（其中 4 条是假的）降到 **0 条**。R4 6 条、R5 2 条不动。

### 9.3 未做的部分

**jump 的朝向不在这次范围内**（用户人工标）。已知情况记在这里：`jump` 头词的 51 行里
只剩 2 行没有方向词——`KI_Archer_CombatJump01`、`KI_Soldier_CombatJump01Rifle`——
两条都掉进 `JUMP_UP_MAX`(0.05) 与 `JUMP_PLANAR_MIN`(0.15) 之间的死区（腾空水平位移
0.061 / 0.095 体长）。量过一轮：改成「腾空水平位移 / 起跳高度」的比值能把第一条判成 `up`
（0.015，而 `up` 真值全部 ≤0.014），但第二条 0.031 和人工标成 `jump, forward` 的
`MU01_Chick_JumpForward`（0.033，且 net 位移同为 0.00）在**任何**已有测量上都分不开，
所以工具补不了它。

### 9.4 下游

154 行标签变了 ⇒ **需要重新生成 action_label 侧产物**
（`preprocess_and_validate.py --regenerate-side-artifacts`，会重写 cond 里的 action_label 向量
和词表指纹）**并重训**。不需要重新预处理 motion。


## 10. prefill 默认跳过已核验的行（2026-09-19）

`reviewed: true` 是人看完 GIF 之后签的字，所以那一行的空槽是**判断**，不是漏标。
之前补词工具每次跑都把这些行重新量一遍、重新列进清单（也会在 `--apply` 时写进去），
复核过的东西又回到清单里。`tools/prefill_loop_flags.py` 早就是「reviewed 行不动」
（连 `--rejudge` 都保留它们），这次把补词工具对齐。

- **默认**：`reviewed: true` 的行整行跳过——不判、不进控制台计数、不进 `--report` 的 CSV、不写。
  `--include-reviewed` 才连它们一起判，`--skip-reviewed` 是显式写出默认值。
- **reviewed 行仍然是证据**，只是不再是写入目标：标定集、`prefill_direction_words` 的镜像伙伴
  与「两条镜像取不能同侧」互检都照旧。丢弃发生在所有互检
  **之后**、计数/CSV/写入之前（`prefill_common.drop_reviewed`），所以判据本身一个字没变。
- **写时再查一次**：`prefill_common.apply_proposals(include_reviewed=False)` 按磁盘上的行再读一遍
  `reviewed`，和「槽已被人填上」那条 SKIP 并列，所以干跑之后才被签字的行同样不写。
- 什么时候该加 `--include-reviewed`：规则变了让旧签字过期的时候（新豁免、换了测量的轴、词表改动）。
- 实测（`--action-group stationary`，truebones zoo cond）：119 行（60 keep + 59 review）现在直接跳过；
  加 `--include-reviewed` 的输出与改动前逐条相同。
- 下游：只改工具行为，不动任何标签，**不需要重新生成侧产物、不需要重训**。
