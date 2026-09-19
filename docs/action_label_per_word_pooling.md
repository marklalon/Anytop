# Action label 词级条件方案

> **状态**：已落地并训练评测。核心不变量见下；具体数值（槽源秩、指纹、词表规模、checkpoint 版本）
> 以契约模块（`action_label_conditioning_contract.py`）为准，本文不固化，避免随代码变动而失效。
>
> **三条长期结论，先说清楚**：
> 1. **没有做消融，本文不声称任何「相对旧表示的增益」。** 评测全是绝对读数——对着该物种自己真实
>    clip 的测量地板读，不对着另一个 checkpoint 读（唯一现成对照组的 cond 和语料都不同，逐格比无法归因）。
> 2. **主词顺序进入条件**：head 通道只装标签的**第一个**主词（单成员、未稀释）；第二个主词按位置
>    路由到 modifier 通道，所以 `attack, hover` 与 `hover, attack` 是两个条件。主词顺序不编码**方向**——
>    方向写进词表（`draw` / `sheathe` / `stop` / `kneel`…）。
> 3. **改条件语义就要重训**：每次改动都会让旧 checkpoint 被两层指纹拒绝加载，不会静默读成「质量回退」。
>    推理端默认：语料内标签 CFG scale 2，未见过的词组合用 1。

## 1. 设计目标

- **组合泛化**：整串 T5 向量无法复用已见词去拼未见组合；改标签还会使整份 label-keyed sidecar 失效。
- **控制轴稀释**：均匀逐词平均里，一个词的份额随标签长度下降；加装备 / 动作细节后，基础动作和方向的
  差异被冲淡。
- **transition 方向**：靠词序表达方向（`idle, attack` vs `attack, idle`）在池化下天然丢失；曾为此给
  transition 第二主词加固定带符号置换，后整体撤销，方向改由词表承担（§6）。

## 2. 定稿表示：每个角色槽一个独立条件通道

四个槽，各占一个条件通道；缺席槽为零行（不重新归一化其它槽）；四通道按固定顺序拼接进 timestep 条件。

| 槽 | 成员 | 说明 |
|---|---|---|
| head | 标签的**第一个**主词（恰好一个成员） | 通道 = 该词向量本身，未稀释 |
| direction | 方向词（至多一个垂直词） | 槽内均值 |
| modifier | 其余修饰词 + **后续主词**（第二个及以后的主词） | 槽内均值 |
| hands | `hand1` / `hand2`（至多一个） | 通道 = 该词向量本身；不写 = 零行 = 空手 |

槽内聚合 = 成员词向量均值 + L2 归一化。**槽内是集合**（词序在槽内不进条件）；槽**间**的分配由位置决定
（哪个主词排第一）。

**为什么这个表示同时满足「轴保留」和「可分性」**（被选中的原因）：
- **轴保留是恒等式，不是超参**：head / direction 通道只含本槽的词，标签从 2 词加到 8 词，这两个通道
  逐位不变。不存在需要在「保留」和「可分」之间权衡的常数。
- **可分性由拼接给出**：各通道各自进入第一层线性投影的一个分块，每通道的相对尺度是**可学的**，
  离线不替模型定预算。
- **按 token 可审计**：一个词只影响它所在的槽。

**为什么不是 K-token**（保留每个词独立 token）：信息上更强，但槽源满秩已保证全部合法组合可由一层线性
读出；而学习注意力池化初始接近均匀，会把「轴保留」从恒等式退化成训练目标。K-token 唯一不可替代的是
**时间局部化**（让第 40 帧去看第二个端点）；当前条件是加在 timestep 上的单向量、对所有帧恒定，若将来要
逐帧按词条件化，再上 per-layer cross-attention。为此 loader 保持词级输出，换表示只动模型消费端。

## 3. 标签格式契约

- 受控词表，逗号分隔；**总词数 ≤ 8**，未知词 / 重复词 / 空段 / 第 9 个词均硬失败。
- 每条非空标签含 **1～2 个主词**（`HEAD_VOCAB`）；**首词进 head 槽，后续主词进 modifier 槽**。
- **主词顺序 = 条件**：同一 group 内同一词集只允许一种主词顺序（两种顺序是两个条件，语料必须对同一类
  clip 只决定一次它「关于什么」）。跨 group 允许不同顺序（各 group 是各自的 checkpoint）。
- 方向词紧跟 `turn`（否则紧跟最后一个主词），其余修饰词按词表序。
- **hands 轴**：`hand1` / `hand2` 互斥、至多一个；**不写 = 空手**（内容默认，不是边缘分布）——
  持物必须写。语义是手部**占用数**，不是武器类别：剑 + 盾 = hand2（与步枪
  相同），匕首 / 火把都是 hand1；动作用什么器具（`bow` / `gun` / `hammer` / `shield`）留在 modifier。
- **direction 轴**：**不写 = 任一方向**（边缘分布）——与 hands 轴相反。这个语义由训练侧
  `--direction_slot_drop_prob` 随机抹掉方向词来教；内容上有单一主导侧向 / 朝向就必须写。
- 空标签 = 无条件分支，走 null embedding，不编码空文本。

**为什么 crouch / dead / sit / sleep 是修饰词而非主词**：语料里它们只出现在次位（限定姿态，不是标签
「关于」的东西）。其余次位主词（hover / turn / jump / rear / roll…）每一个也都当过首词，无法按词降级，
只能按位置路由（§2、§6）。

## 4. 数据契约与 sidecar（概念）

- **word-keyed 全局词表**：一张全局冻结词表（每 token 一个向量），改标签不会使它过期；词 id 是词表里的
  位置，所以词表顺序变了必须重建词表（重建后旧 sidecar 会被拒，fail-fast，不会静默用错向量）。
- **loader 只发词级 id**（word / slot / mask，定长），不预拼向量；槽拼装发生在模型侧，用 checkpoint 内
  的冻结词表完成。这是有意的：把表示锁进数据通路会让以后换表示要重建数据。
- **推理不读任何 sidecar**：词表是 persistent buffer，随权重一起出入 checkpoint。
- **两层指纹 gate load / resume / checkpoint-bind**：
  - `embedding_fingerprint` 只描述冻结词向量，含**词表本身的哈希**——这是唯一一条会随向量改变而失败的
    字段，防止「同形状的另一张表」被静默当成同一张表；
  - `conditioning_contract_fingerprint` 描述运行时语义（词表、parser 版本、槽布局）。
  - 改槽规则只失效后者；改 T5 文本 / EOS / 后处理两者都失效。当前具体值在 checkpoint 与词表 sidecar 里。

## 5. 槽通道可分性（原则）

只对模型**改不回来**的性质设硬门；各向异性指标（p95 / 最近邻中位 / 有效秩）可被第一层 Linear 重标度，不当门。

- **硬门**（建模型时在 `model/anytop.py` 里断言）：每槽源满秩；`latent_dim` ≥ 各槽源总秩。
- **为什么这些是硬门**：槽源满秩意味着不同成员集合的归一化和不可能相同、且存在线性 readout 能判断每个词
  是否在槽内，覆盖解析器允许的**全部**非空槽组合（含上限 8 词），不只覆盖语料见过的组合。
- 当前满秩值见 `slot_source_rank_report`（`action_label_conditioning_contract.py`）的返回，本文不固化。

## 6. 决策历史（只留「为什么」和不变量，去掉实现细节）

### hands 轴：`weapon` + `1hand/2hand` → `hand1 / hand2`（2026-09-11 起，含当时的 `hand0`）

旧轴没有「空手」这个值，且标注不一致（combat idle 写 weapon、attack / hurt / die 不写），「没写 weapon 的
clip」其实是持械与空手的混合。当时的解法是补一个显式的 `hand0`（「空手」）；2026-09-18 改成让空槽本身
承担这个语义后 `hand0` 被删除（见下一条），但「旧轴分不开空手与漏标」这个动机不变。
改成互斥占用数 + 独立第四槽：单独成槽是因为 hands 词会出现在有手物种几乎
每条 clip 上，塞进 modifier 会让每个真修饰词的份额减半（`attack, slash, hand2` 里 slash 被稀释），且
「带不带 hand 词」读数不同。独立槽下其它三通道逐位相同，模型只需分开两个点和一个零行（空手）。

### 两根轴的 ∅ 语义分道：direction = 边缘，hands = 空手（2026-09-18）

两根轴原本共用一条规则「不写 = 未指定」，但推理只用裸标签，两边想要的默认不是同一个东西：

- **direction**：裸 `attack, swat` 出双臂混合，因为语料里绝大多数单侧攻击也不写方向词，模型学到
  「空方向槽 = 这个物种攻击的样子」而不是「任一侧」。要让空槽真的是边缘分布，得让模型见到**同一条 clip
  在带方向词和不带方向词两种条件下**，于是加训练侧 dropout：`--direction_slot_drop_prob`（默认 0.3）
  按样本抹掉方向词、其余词不动。与 `--action_label_cfg_drop_prob`（整条标签丢掉，训练脚本里是 0.3）
  叠加后，带方向词的行有 (1−0.3)×(1−0.3) = 49% 的 batch 贡献显式方向监督；左右可辨性不够就降这个值。
  推理永不 drop。
- **hands**：裸 `idle` 在持械物种上出持械姿势。这里想要的默认恰恰**不是**边缘——用户要的就是空手。所以
  反过来：不给 hands 加 dropout，把 ∅ 直接定义成「空手」，`hand0`（显式空手）随之删除，词表 106 → 105、
  槽源秩 138 → 137。被否决的对称方案是「保留 hand0 + 给 hands 也加 dropout」：边缘在 hands 轴上没有用处。

两边都要求标注侧配合，且**只补不改**（已有的方向词 / hand 词是人工核验过的真值）：
`tools/prefill_direction_words.py` 用自镜像分解 + 左右肢体能量比（侧向）、支撑脚相对速度（locomotion 朝向）、
腾空根位移（transition 的 `jump, up` vs `jump, forward`）；`tools/prefill_hand_words.py` 用同物种已标行的
上肢姿势做 k-NN，加一张「模型自带武器」的物种默认表。两者都先在已标行上标定，达不到门槛就只出清单。
`jump` 也因此从 R4 的 `NO_HEADING_WORDS` 里去掉：∅ 只能是边缘，竖直跳要自己写 `up`。

### 去掉 transition 的方向机制

曾只在 transition 组给第二主词过固定带符号置换。代价是同一个标签字符串在 transition 与 stationary 的
checkpoint 里是**不同的条件向量**，标签失去「跨 group 同义」；方向只能靠标注者记住「逗号左边是起点」，而
语料过半 transition 是自反事件词（`die` / `getup` / `spawn`）。方向写进词表（`draw` / `sheathe` / `stop` /
`kneel`…）后，词序机制没有东西可编码，删除它让 transition 与另外两组共享同一条件语义、同一校验。

### 首词进 head 槽，后续主词进 modifier 槽

旧表示把一条标签的所有主词取均值放进 head 槽：语料里约 1/10 的标签是双主词，第二个主词几乎都是姿态 / 媒介
限定（hover = 空中、rear = 直立…），不是标签「关于」的东西；均值让首词在 head 通道被稀释，且 `idle, hover`
与 `attack, hover` 因共享 hover 而靠近。改为位置路由后，head 通道永远是首词的未稀释向量。
**有意接受的两项代价**：
1. **修饰词稀释转移**：hover 现在和打击类型共用 modifier 均值（`attack, hover` 桶内靠 bite / slash 区分）；
   次位主词只在少数 clip 上出现，不再单开第五槽。
2. **权重共享丢失**：hover 作首词和作次位是投影第一层的两个块，「在空中」要学两遍；对当首词很少的
   rear / roll / crawl 几乎没损失，对 hover / jump / turn 两边各够多。

## 7. 评测口径（重点）

**没有基线 → 全部绝对读数**，参照系是同一估计器量**该物种自己真实 clip** 的结果（「语料地板」）。能回答
「控制力够不够 / 长标签会不会稀释 / 未见组合能不能走 / 质量有没有塌 / CFG 定多少」；**不能回答**「相对整串
T5 提升了多少」——本文任何地方都不声称这个数。

关键读数（对语料地板）：
- **方向遵循**：前后向在 cfg 1 就已贴地板，加 CFG 反而变差；左右向需要 CFG，收益在 cfg 2 用完。
- **长标签**：标签从 2 词涨到 5 词，方向误差无趋势，全部压在地板上方一度以内——「控制轴随长度被稀释」在
  生成侧看不到。
- **未见组合**：能走（四向个位数）；修饰词迁移不对称（能到通道 ≠ 用得上）。
- **质量**：质量电池中位数健康，最弱两格是历史遗留，不是本轮新出现。

**CFG scale 结论**（实践结论，不是超参调优）：
- **语料内标签用 cfg 2**：角度误差在 cfg 2 取极小，质量只比 cfg 1 掉约 4%。
- **held-out 组合用 cfg 1**：标签越是模型没拟合过的，CFG 外推「条件 − 无条件」的方向越不可靠，放大它就是
  放大误差。
- **cfg ≥ 4 不要用**：backward 从 cfg 4 开始崩。
- 代码默认值保持 1.0（改 2 会让没有 CFG 训练的 checkpoint 直接报错）；帮助文本记录这里的实测值。

**本轮明确没回答的**：相对旧表示的增益（无基线）；`run, right` 是否上硬输入位（判据成立、性价比未算）；
`run, forward` 的相位离散（现象已记录、未定位）。

## 8. 待办

- 改条件语义的每次改动都要求重训；旧 checkpoint 被两层指纹拒绝（不需要 regen cond）。
- 训出 transition checkpoint 后，验收 `draw` / `sheathe` / `stop` 生成的动作朝向正确端点。
- 训出后验收裸 `attack, swat` 出单侧而非双臂混合，`attack, left, swat` 与 `right` 可辨；裸 `idle` 空手、
  `idle, hand2` 持械。**23 个物种全程持械**（MLH_Footman / Mage、MLS_DemonHunter / Druid、RMW_Orc /
  Skeleton、TNR_Cavalry / CavalryMage / CavalrySpear / Infantry / Mage / Spearman、TTR_Crossbowman /
  Halberdier / HeavyCavalry / HeavySwordman / LightCavalry / LightInfantry / Mage / MountedKnight /
  MountedMage / Spearman / Swordman），没有一条空 hands 行，它们的裸 `idle` 是零样本外推，单独记录。
  `dance` / `sway` / `fullbody` / `footwork` / `armwork` 五个词退役后同样没有训练样本，一并单独记录。
- 训出后验收 `attack, hover` vs `idle, hover`（head 通道现在完全分开）、`attack, hover, bite` 的打击类型
  是否仍可辨（modifier 稀释是否真的可接受）。