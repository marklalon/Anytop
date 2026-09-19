# action_label：方向槽 dropout 与 hands 槽「空手默认」

> 状态：**已实施**（2026-09-18；方案本体见下，实际落地与方案的出入记在 §6）
> 触发：裸 `attack, swat` 出双臂混合而非单侧；裸 `idle` 在持械物种上出持械姿势
> 影响：`ACTION_LABEL_PARSER_CONTRACT_VERSION` 4 → 5、`CKPT_VERSION` 15 → **16**，**需从头重训**，不需 cond regen
> 相关：[`action_label_per_word_pooling.md`](action_label_per_word_pooling.md)、[`tools/audit_action_labels.py`](../tools/audit_action_labels.py)

## 0. 先固定语义（两条改动都依赖它）

| 槽 | ∅ 的含义 | 由谁定义 | 训练时 dropout |
|---|---|---|---|
| direction | **未指定 = 边际**（任一方向） | 训练时随机抹掉方向词 | 是 |
| hands | **空手**（内容默认） | 标注：持物必写 `hand1` / `hand2` | 否 |

推理侧：裸 `attack, swat` = 任一侧、空手；要持械必须写 `hand1` / `hand2`。

标注规则：

- **方向**：内容有单一主导侧向或平面朝向 → 写方向词；对称、交替、旋转、原地 → ∅。
  stationary 只补 `left` / `right`（stationary 按定义无净位移，`forward` / `backward` 不补）。
  **例外：本身不指向任何方向的动作一律不写平面方向词**（2026-09-19，见 §7）——
  `hurt` / `getup` / `idle` / `rest` / `stop` / `draw` / `sheathe` / `headbutt` / `bite`。
  竖直词 `up` / `down` 不在豁免内（`idle, up, aim, bow` 保留）。
- **hands**：有手物种凡持物 → 写 hand 词；无手物种 → ∅。

不变量（两个补标脚本共同遵守）：

- **已有的方向词和 hand 词是人工核验过的真值，没有错标，只可能漏标。** 脚本只填 ∅ 行，永不改写已有词；
  标定把已有词当真值来校准判据的符号和阈值。
- **判定只看内容。** clip 名是最弱的证据，只写进 `evidence` 供人核对，不参与判定，也不用来推断「疑似误标」。

已知项：**全程持械的 14 个物种**（MLH_Knight、RMW_Orc、RMW_Skeleton、TNR_Cavalry / CavalrySpear /
Mage / Spearman、TTR_Crossbowman / HeavyCavalry / Mage / MountedKnight / MountedMage / Spearman / Swordman）
没有一条 ∅ 行，裸 `idle` 是零样本外推。接受，验收时单独记录。

与现状的差异：`motion_labels.py` 的 hands 注释块目前写的是相反的语义（∅ = 边际、`hand0` 是显式陈述），
改动落地时注释块整段重写，不能只删一个 token。

> 被放弃的对称方案：保留 `hand0` 并给 hands 槽也加 dropout（∅ = 边际、`hand0` = 显式空手）。
> 没选它的原因是推理只用裸标签，用户要的默认就是空手，边际在 hands 轴上没有用处。

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
  同组，写进 `args.json`；推理侧永远不 drop。默认 **0.3**（推理只用 ∅，边际条件比单侧条件更重要；可调）。
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
| 原地竖直跳（transition） | `jump, up` | 已有 2–3 条；37 条无方向的 `jump[, hand*/happy]` 按内容改 `up` 或 `forward` |
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
- `dataset/review/action_label_audit_ignore.jsonl` 里 11 条 hands 迁移后失效的旧拼写更新。
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

## 2. hands 槽

### 2.1 词表与契约

- `HANDS_VOCAB = ("hand1", "hand2")`（[`motion_labels.py`](../data_loaders/truebones/truebones_utils/motion_labels.py)），
  `_VOCAB_T5_TEXT` 去掉 `"hand0": "empty hands"`，hands 注释块按 §0 重写；「至多一个」校验不变。
- `ACTION_LABEL_PARSER_CONTRACT_VERSION` 4 → 5、`CKPT_VERSION` +1（见文首）；
  `tools/build_action_label_embeddings.py --force` 重建词表 sidecar（CUDA 上算指纹）；
  hands 槽秩 3 → 2，**总槽源秩 138 → 137**（swipe 合并已把 139 降到 138）；`latent_dim` 下限 137。
- **rank 触点不止两个 pin**：`test_action_label_conditioning_contract.py` 里 `report["total_rank"] == 138 → 137`、
  每槽 rank 字典 `"hands": 3 → 2`、负向探针 `slot_source_rank_report(table, latent_dim=137)` 要下移到 **136**
  （137 变合法后那条断言会失效）；`test_action_label_word_conditioning.py` 的 `total_rank` pin 同步
  （同文件的 `latent_dim=total_rank - 1` 是派生的，自动跟）；`tests/action_label_test_utils.py` 的
  `TEST_LATENT_DIM` 注释 `3 hands = 138` → `2 hands = 137`（140 这个宽度仍够）。
- 其它触点：`parser_util.py` 的 `--action_label` 帮助文本（现在写着 `hand0` = 空手）、
  `dataset/review/relabel_actions_llm.py` 的 `"load"` 词元组与 transition prompt（两处都含 `hand0`）、
  docs `action_label_per_word_pooling.md`。
  `dataset/review/audit_action_labels.html` 是生成产物（下次跑审计自动重写），不用手改。
- tests 里除 rank 之外还有：`test_action_label_cond.py` 的 `HANDS_VOCAB` pin、`vocab_t5_text('hand0')` 断言、
  含 `hand0` 的 round-trip 列表，以及「两个 hands 词」报错用例 —— `idle, hand0, hand1` 要改成 `idle, hand1, hand2`
  （`hand0` 删词后它不再是「互斥轴写两遍」而是未知词，报错文案不同）；
  `test_action_label_conditioning_contract.py` 的 `hand0` 槽位用例、`test_action_label_word_conditioning.py` 的标签列表。
  `eval/eval_tasks_*.json` 当前没有 `hand0` 任务，不用改。
- 现有 49 行 `hand0`：删词即可，语义不变，`reviewed` 不动。
- 不加 hands dropout。

### 2.2 `tools/prefill_hand_words.py`（`hand1` / `hand2` 漏标检查）

GIF 不含道具网格，内容上只能看持物姿势，所以走参考集 + k-NN：

- 参考集：同物种已标 `hand1` / `hand2` 行，加现有 `hand0` 行作「空手」参考——**必须在 2.1 删词之前跑**。
  同骨架物种（KI ×7、TNR ×5、TTR ×8、TTR 骑兵 ×5）共享参考集。
- 描述子：上肢静态姿势——双手相对胸部位置、肘角、双手间距，取 clip 时间中位数。k = 3 投票 + 距离阈；
  投票不一致或超阈 → 只列清单。
- 标定：已标行留一法，出 hand1 / hand2 / 空手三类混淆矩阵，只自动写准确率 ≥ 90% 的类。
- 物种默认层（表驱动，全部 `reviewed=false`）：Unity 122 个无 hand 词物种里 29 个 tag 为 Biped，其中 7 个是有手人形，
  需按包知识给默认——KI_Human、KI_Performer、KI_Slinger、MLH_Footman、MLH_Mage、MLS_DemonHunter、
  TTR_LightInfantry（后者空手，实际不用动）；MLS_Druid 也是人形，顺手确认。其余是无手生物 / 机器人，∅ 正确。
- 逐行重点：20 个物种内混标——LH_Hero 107 行 ∅（含 THSwordCastSpell）、KI_Villager 24、KI_CasterMage 26、
  KI_Archer 11、TNR_Worker 7……已有的 hand 词都是核验过的真值，这些 ∅ 行只可能是漏标，脚本只补不改。
- 名字里的 Sword / Spear / Bow / Axe / Pickaxe / Hammer / Gun / Staff / Shield / Sack / Torch 只进 evidence。

## 3. 执行顺序与验收

1. 两个脚本 `--dry-run`，看标定准确率，定阈值。
2. `prefill_hand_words --apply` → 2.1 词表改动 + 49 行删 `hand0` → `prefill_direction_words --apply` → 1.1 代码。
3. 预检：`audit_action_labels.py --cond-path dataset/merged/cond.npy --action-group all`（R3 / R4 / R5）、tests。
4. 人工：review UI 过滤 `reviewed=false` 行逐条看 GIF。
5. 同步 `relabel_actions_llm.py` 的 prompt（方向规则「单一主导侧向 → 写方向词」、hands 规则「持物必写」），
   `dataset/readme.txt` 的预填那一步（现在只有 `prefill_loop_flags.py`）加两个 prefill 工具；
   `action_label_per_word_pooling.md` §6 记决策。
6. 重训（契约变了，无法 resume）。

验收：

- 裸 `attack, swat` 出单侧而非双臂混合；`attack, left, swat` 与 `right` 可辨。
- 裸 `idle` 空手；`idle, hand2` 持械。
- 全程持械 14 物种的裸 `idle` 单独记录效果（零样本项）。

## 4. 影响面速查

| 项 | 数量（2026-09-18 工作树） |
|---|---|
| 标签行 | 3714（unitybundles 2534 / zoo 937 / zoo_upgrade 243），全部 `reviewed: true` |
| `hand0` / `hand1` / `hand2` | 49 / 223 / 245 |
| stationary 有 left/right | 105；transition 有 left/right 206（其中 `turn` 200） |
| locomotion 无方向词 | 95（含 hover / fall / dive 等豁免；`Dragon_WyvernIdle` 转 stationary 后少一条） |
| transition 无方向 `jump*` | 37；`jump, up` 2；`jump, forward` 10 |
| locomotion `run, jump` / `swim, jump` | 5 / 5 |
| Unity 物种 hand 词覆盖 | 无 122 / 全有 14 / 混合 20 |

> **dance 已确认全部退役**（2026-09-18，与本次 `pending_delete` 一起落地）：unitybundles 的 68 条 `*_Dance*`
> （KI_Performer 34 + LH_Hero 34）全部标 `pending_delete`，zoo 另有 5 条（4 条重复的 Pteranodon hit/death +
> `Spider_Riser`）。退役后 `dance` 在三个数据集里**一条不剩**，而 `sway` / `fullbody` / `footwork` / `armwork`
> 只出现在 dance 行上，同样归零 —— 词表与 embedding fingerprint 都不变（不会报错、也不会拒绝 resume），
> 但这 5 个词从此只有 T5 文本嵌入、没有动作样本，推理时是纯零样本外推。与「14 个全程持械物种的裸 `idle`」
> 同性质，验收时一并单独记录。

## 5. 对原始草案的审核修正

原草案整体成立：dropout 位置与写法、∅ 语义分工、内容优先于名字、根位移不可用 → 支撑脚法、先标定再写、
写回方式都对。以下是核对代码和数据后补正的：

1. **把「只补不改」写成不变量**（§0）。已有方向词 / hand 词是人工核验过的真值；脚本只填 ∅ 行，标定把已有词当真值。
2. **标定集排除 `turn`**。原草案「已标 102 行」实为 stationary 105 + transition 非 turn 6；transition 另有
   200 条 `turn, left/right`，已全部带方向词，是偏航方向不是肢体侧，不进肢体能量标定。
3. **槽源秩 139 → 138 是旧数**。swipe 合并（09-17）已到 138，本次 hands 3 → 2 是 138 → 137，tests 的 pin 要改
   （细目见 §2.1 与 §5.10）。
4. **R3 不扩展前缀 / A-B 匹配**。`clip_side` 只认尾缀是防部位名误报的既定设计；8 对镜像由内容检测给出。
5. **`eval_tasks_stationary.json` 没有 `hand0`**，从触点列表删掉；补上 `_VOCAB_T5_TEXT["hand0"]`。
   （原草案还列了 `audit_action_labels.html` —— 那是审计生成产物，重跑自动重写，不算手改触点。）
6. **写回 helper 的出处**是 `fill_missing_loop_flags`；抽共用而不是复制。
   （后续调整：行级读写搬到 `tools/action_label_sidecar.py`；`fill_missing_loop_flags` 本身搬回
   `prefill_loop_flags.py`——`--rejudge` 与「`reviewed` 不被覆盖」是那个工具的策略；
   `motion_labels` 只留键位规则 `set_loop_flag`，review UI 也走它。「抽共用而不是复制」这条不变。）
7. **hands 语义翻转要连注释块一起改**：`motion_labels.py` 现在的注释论证的是相反语义。
8. 补充：stationary 只补 left / right 的边界；根 XZ detrend 的代码出处与实测数字（§1.2 B）；
   LLM 标注 prompt 与 readme 要同步，否则下一批数据又按旧规则标。
9. **行号引用改符号名**（本次复核）。原草案写了 `motion_labels.py` / `parser_util.py` / `relabel_actions_llm.py` /
   `audit_action_labels.py` 的行号，一改就过期；改成 `HANDS_VOCAB`、`_VOCAB_T5_TEXT`、`check_r4`、
   `NO_HEADING_WORDS`、`--action_label` 帮助文本、`"load"` 词元组与 transition prompt 这样的符号引用。
10. **hands 3 → 2 的触点不止两个 pin**（§2.1 已展开）：每槽 rank 字典、负向探针下移到 136、
    `TEST_LATENT_DIM` 注释，以及 `test_action_label_cond.py` 里含 `hand0` 的四处（pin / `vocab_t5_text` /
    round-trip 列表 / 「两个 hands 词」报错用例）。
11. **方向 dropout 与 CFG drop 的监督稀释**（§1.1）：两者叠加后带方向词的行只有 `(1 − 0.2) × (1 − p)` 的 batch
    贡献显式方向监督，p = 0.5 时只剩 40%，故默认从 0.5 改为 0.3。
12. **dance 全部退役**（§4）：`dance` / `sway` / `fullbody` / `footwork` / `armwork` 五个词条失去全部训练样本，
    验收时与全程持械物种的裸 `idle` 一起单独记录。
13. **审计工具不认 `pending_delete`**：`tools/audit_action_labels.py`（R1 / R3 / R4 / R5）不过滤 pending 行，
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
| hands（同物种上肢姿势 k-NN，留一法） | hand1 96.6% / hand2 97.8% | 90% | 开，14 条 k-NN + 32 条物种默认 |

- **方向**共写 232 行（unitybundles 177 / zoo 43 / zoo_upgrade 12），804 行进复核清单
  （`dataset/review/prefill_direction_words.{csv,html}`）。
- **hands** 共写 46 行（`dataset/review/prefill_hand_words.{csv,html}`），165 行进清单；随后
  `--drop-hand0` 删掉 49 行的 `hand0`。
- 6 行两个槽都补了，当时 `autofill` 里用 `earlier` 叠着记（**记法已作废，见 §8**）。

### 6.2 与方案不同的地方

1. **`turn` 行整体退出方向工具**（不只退出标定集）。方案只说 transition 的 200 条 `turn` 不进肢体能量
   标定集；实际上 locomotion 的 `walk/run, turn, left` 也一样——那个 left 是偏航方向，支撑脚法量的是
   行进方向，两者在转弯 clip 上本就不同（实测把 turn 行算进去，朝向标定从 97.4% 掉到 90.1%）。
   所有带 `turn` 的行现在既不参与标定也不接受补标。
2. **hands k-NN 只用同物种参考，不用同骨架**。方案说 KI / TNR / TTR 各自共享参考集；实测跨物种匹配的是
   「放松的手臂姿势」而不是持物状态（村民的 idle 匹配到士兵举枪的 death），hand1 精度只有 89%。改成
   只跟本物种的已标行比之后，两类都过 96%。代价是参考不足 3 条的物种不判（留给物种默认表）。
3. **物种默认表只填 4 个物种**，不是方案列的 7 个：Mini Legion 的武器是烘进网格的，GIF 里看得见——
   MLH_Footman（长矛+盾 hand2）、MLH_Mage（法杖 hand1）、MLS_DemonHunter（双刀 hand2）、
   MLS_Druid（法杖 hand1）。KI_Human / KI_Performer 是空手（Basic Motions / Dance），
   KI_Slinger 只在投掷瞬间持物、按 clip 判，TTR_LightInfantry 已有 hand 词走 k-NN。
4. **距离阈值按同类最近邻中位数自动定**（不是固定常数），可用 `--distance-max` 覆盖。
5. **R1/R3/R4/R5 现在过滤 `pending_delete`**（§5.13 的两个选项里取了「给审计加同样的过滤」）。
6. **审计新增「失效 ignore 条目」告警**：`action_label_audit_ignore.jsonl` 的 11 条旧拼写已按新标签重写，
   剩下 14 条（clip 已退役 / 标签改到认不出）每次跑审计都会列出来，不再静默烂掉。
   （**告警连同忽略名单已删，见 §8**。）
7. **review UI 加了 `自动补` 角标**：鼠标悬停显示补了哪个槽、原标签和判据；在页面上手改标签会删掉
   `autofill` 记录（工具的提议已被推翻），只点 OK 核验则保留（提议被确认）。
8. **全程持械物种从 14 个变成 23 个**（补标之后），验收时的零样本项按新表记，见
   [`action_label_per_word_pooling.md`](action_label_per_word_pooling.md) §8。

### 6.3 复核之后还剩什么

- 方向 804 行 + hands 165 行的复核清单（两张 html 页面并排放着 GIF 和候选值）。
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
   自己补的（记录连同 `autofill` 一起删掉，`reviewed` 恢复成补标前的值；`MLH_Mage_GetHit` 还叠着一条
   hands 补标记录，那条保留、`reviewed` 仍为 false），另外 39 行是人工标的真值
   （`Trex_BiteLeft/Right`、`Dog_LookLeft/Right`、`hurt, left/right` 等，`reviewed` 不动——
   剩下的词仍然是人核过的，只是少了一个规则不再允许的词）。
2. **补标工具**：`prefill_direction_words.py` 的 `kind_of()` 直接把这些行判为「不在范围内」，
   既不判也不进标定集；transition 的跳仍然留在范围内，只是平面那一半的结论改成进清单。
3. **审计**：R3 的「两边各自要带侧词」对这些行不成立——**两边拼成同一个标签就是镜像**
   （`Trex_BiteLeft` 和 `Trex_BiteRight` 都是 `attack, bite`），控制台按对数报出来；
   R4 的朝向要求同样豁免，且**不受 `--r4-exempt-words` 控制**（那个开关只替换 `hover` 那张表，
   这条是语料级规则，不是旋钮）。`not_mirror` / `crossed` 两项照查不误。

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
`--no-default-ignore`、§6.2.6 的「失效 ignore 条目」告警、报告页「无需修改」生成 ignore 行的那段。
76 条里 51 条是 R1 的桶，随 R1 一起失去意义；剩下 25 条 clip 条目多数是 §7 的豁免已经从规则上
盖掉的镜像对（bite / roar / death / hurt / getup）。三条规则都是确定性的，报出来就是真错，
不需要一份「看过了，不用改」的名单来压。

**prefill 不再出页面**：`prefill_hand_words.py` / `prefill_direction_words.py` 的 `--html` 与两张
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
