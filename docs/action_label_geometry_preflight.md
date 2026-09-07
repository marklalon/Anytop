# action_label 条件几何预检

> 日期：2026-09-06  
> 结论：**GO —— 定稿表示为每个角色槽一个条件通道 `slot/eos_keep/center_l2`。**  
> 工具：[`tools/evaluate_action_label_geometry.py`](../tools/evaluate_action_label_geometry.py)  
> 固定角色变换：由 `role_b_material()` 按需推导，见
> [`action_label_conditioning_contract.py`](../data_loaders/truebones/truebones_utils/action_label_conditioning_contract.py)

## 1. 可复现输入

- 预检当日的三套语料：3802 行、398 个不同 label 串；按 checkpoint/group 展开为
  408 个条件点；
- `action_labels.jsonl` SHA-256：zoo
  `B823237D939333FB42BE4C9DA1D919F5D813AA0418AB0EBFBC86730A9B4DD2B1`，zoo_upgrade
  `9A4185A0A4F6D7A4E6D69709A1E618AD6EB873A6ED926D53CA5925D6EC16FC40`，unitybundles
  `982DE2BED668132422704B515083C9F4EB4B622220BC2CD2EC16D70003940B8E`；

> **语料在预检之后、开训之前动过（2026-09-06 20:35）。** 提交 5203808
> （burrow 由 stationary 改判 transition）与 3b576dc（Anaconda_Spin、
> MB_Unka_TurnRight 改标）改写了 zoo 与 unitybundles 两份 jsonl，
> 因此 `merged_locomotion_v5_pwp` 训练用的是下面这份，不是上面那份：
> zoo `C2EA57A887F7554E64D019C137733243AE0747EAA310818CF0CBF081BB36FA57`，
> unitybundles `A498D525171998E52EAB27F909407244452BC1ED950C200BF7D015B3A1D74C2F`
> （zoo_upgrade 未变）。规模变成 **3802 行 / 396 个 label / 404 个条件点**，
> transition 由 755 条增至 760 条。
> 2026-09-07 在这份语料上重跑 `--device cuda --skip-exhaustive`：**硬门逐项 GO，
> 数值与 §5 记录一字不差**（worstNN 0.9559、rev50 0.015、drift 0.0e+00、
> 词表秩 102、槽源秩 135、键唯一 True，反向 transition 仍是 5 对），
> 两个指纹也仍是 `0f0a698c…` / `47314397…` —— 指纹只描述词表与运行时语义，
> 本来就不随语料变。原始输出：`outputs/direction_following/PWP_geometry_recheck{,_cuda}.log`。
- 当前真实双向 transition：5 对；
- 编码器：本地 `t5-base`，768 维，`transformers==5.5.4`；
- T5 本地材料 SHA-256：
  `c8f93c55a1eb74684d18984edfe5017c94a153962811c5f543fb2e926aca5fdc`；
- `R_B` 材料 SHA-256（namespace `anytop/action-label/role-b/v1/t5-base/768` 现算，
  对照代码常量 `ROLE_B_MATERIAL_SHA256` 校验）：
  `0204f95ca92d163554ed17bc8ff22ee2858ae128c2691b4893cc0f9c958c4c2b`；
- 词表内容 SHA-256（`word_table_sha256`，`embedding_contract` schema 3 起收录）：
  `e62b012e418883a5d30a6cf7c02674894e0fbd5bda5450de4b4e0674ac7b00e4`；
- 定稿指纹：`embedding_fingerprint`
  `0f0a698cb78fc9ce8359f435b7a24e9e9e115b8c9f2d25aa507345f2be70feb1`，
  `conditioning_contract_fingerprint`
  `47314397cc943d749cc7db3996a689d2ced2451ef5ef29aabc5e526f3be29fc3`。
  （schema 2 时这两个值是 `2e017b7a…` / `dfd6ac0e…`；向量未变，变的只是
  `embedding_contract` 多了 `word_table_sha256` 一项。）

复算命令：

```powershell
& '..\.venv\Scripts\python.exe' tools/evaluate_action_label_geometry.py
& '..\.venv\Scripts\python.exe' tools/evaluate_action_label_geometry.py --device cpu
```

数值组合诊断（`--skip-exhaustive` 可跳过）与工具其余部分共用 `--device`，float64 全程，
CPU/GPU 结果一致到 ~1e-12。全合法输入域的注入性由槽源秩硬门证明，不依赖组合穷举。

> **两条命令不再等价（schema 3 起）。** 上面第二条是 `--device cpu`，它只保证
> *诊断数值*与 GPU 一致；**指纹不一致**。`word_table_sha256` 哈希的是词向量本身的
> 字节，而 T5 前向在 CPU 与 CUDA 上末位不同，所以 `--device cpu` 复算出的是
> `embedding_fingerprint 246d45f4…` / `conditioning_contract_fingerprint e7ad99ee…`，
> 与出厂 sidecar 和 checkpoint 里的 `0f0a698c…` / `47314397…` 不同（2026-09-07 实测）。
> **出厂词表是 CUDA 那份**：要核对指纹必须用第一条命令（默认 `--device cuda`）；
> 拿 CPU 复算结果去比对会得到一个假的"契约不符"。硬门判定本身不受影响，两边全部 GO。

脚本只读语料与本地 T5，不加载扩散模型、不读取 checkpoint，也不把未训练的几何冒充生成质量。
统计按 `action_group` 分开计算，因为三个 group 使用不同 checkpoint。

## 2. 为什么上一版 gate 不可满足

上一版要求同时满足「长标签控制轴保留 ≥ 0.75」和「有效秩保留 ≥ 0.75 ／ 最近邻不劣化」。
对**任何**单向量固定池化，令 ρ = 细节词能量 ÷ 核心词能量：

- `retention ≈ 1/(1+ρ)` —— 轴保留要求 ρ ≤ 1/3；
- 只差一个修饰词的两个标签，其距离正比于 ρ —— 秩与最近邻要求 ρ 大。

一个池化向量只有一个 ρ，所以可行域是空的。当时工具里的四档权重就是这条 frontier 的采样，
它们只是在同一条曲线上滑动（下表 `center` 一列，权重从弱到强）：

| 被否决的单向量候选 | p95 | 最近邻中位 | 有效秩 | 反向对 | retention |
|---|---:|---:|---:|---:|---:|
| 当前整串 T5（baseline） | 0.798 | 0.878 | 18.34 | 0.806 | — |
| 中心化 + 均匀权重 | 0.633 | 0.841 | 14.58 | 0.015 | 0.392 |
| 中心化 + mild | 0.677 | 0.866 | 12.65 | 0.015 | 0.526 |
| 中心化 + medium | 0.751 | 0.901 | 10.71 | 0.015 | 0.703 |
| 中心化 + 原 0.15 强先验 | 0.969 | 0.990 | 7.73 | 0.015 | 1.012 |

这 32 个候选**不再**由工具重算：否决是结构性的，不是余量问题，所以上表就是最终记录。`ACTION_WORD_WEIGHT_PRIOR`、`_fixed_weight`、`_axis_retention` 与旧的 `GO_NO_GO` 门已于 2026-09-06 从代码里删除；这一版数字用 §1 的语料与编码器输入、加上 git 历史里那一版 `evaluate_action_label_geometry.py` 即可复算。

## 3. 新 gate：只对不可逆的性质设硬门

条件路径上紧跟表示的是 `action_label_projection` 的第一层 `nn.Linear`。它可以重标度任意
子空间，因此 p95、最近邻中位数、有效秩这类**各向异性**指标是模型自己能改回来的，不能当硬门；
把它们当硬门正是上一版不可满足的原因。硬门只留下模型改不回来的事：

| 硬门 | 判据 |
|---|---|
| 碰撞 | 同一 checkpoint 内不同标签余弦 ≥ 0.999999 的对数为 0 |
| 最坏近邻 | 每个 checkpoint 的最近邻余弦最大值不高于 baseline 的同项 |
| 反向 transition | 5 对真实反向对的余弦中位数 ≤ 0.50 |
| 通道漂移 | 追加修饰词后 head/direction 通道的最大逐元素变化 = 0 |
| 词表秩 | 冻结词表满仿射秩（raw 103，中心化后 102） |
| 槽源秩 | head 的 64 个普通/角色源、direction 的 6 个源、modifier 的 65 个源分别满秩 |
| 投影宽度 | `latent_dim` 不小于三个槽的总可达秩 |
| 键唯一 | `(group, {(word_id, role_id)})` 在语料上唯一 |

其余指标（p95 / 最近邻中位 / 有效秩）改为**只报告**，与 baseline 对比以便发现回归。

选择规则：先过硬门，再按最坏近邻排序；差异小于 `SELECTION_TOLERANCE = 0.005` 视为无证据，
落回固定顺序（`eos_keep` 优先，postprocess 按 `center_l2 > center > l2 > raw`），
以免第四位小数的抖动决定一次训练的契约指纹。

## 4. 结果

### 4.1 硬门

| 变体 | 最坏近邻 | 反向对中位 | 通道漂移 | 词表秩 | 槽源总秩 | 键唯一 | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| baseline 整串 T5 | 0.9900 | 0.806 | — | — | — | — | 参考 |
| **slot/eos_keep/center_l2（选中）** | **0.9559** | **0.015** | **0.0** | 102（raw 103） | 135 | 是 | **GO** |
| slot/eos_keep/center | 0.9618 | 0.015 | 0.0 | 102 | 135 | 是 | GO |
| slot/eos_keep/raw | 0.9672 | 0.328 | 0.0 | 103 | 135 | 是 | GO |
| slot/eos_keep/l2 | 0.9742 | 0.330 | 0.0 | 103 | 135 | 是 | GO |

八个 slot 变体（两种 EOS × 四种后处理）全部通过硬门；EOS 策略至此不再是悬置项，
按选择规则定为 `keep`。

定稿表示的槽源秩为 head 64/64、direction 6/6、modifier 65/65。不同槽占拼接后的不同块，
所以总可达秩是 135；默认 `latent_dim=256`，不存在 `2304→256` 投影造成的信息瓶颈。
源向量满秩还直接证明：总词数不超过 8 时，任意两个不同槽成员集合都不会在归一化均值后碰撞，
每个成员也都可由线性 readout 判断。该结论覆盖解析器允许的全部组合。

### 4.2 只报告的几何（相对 baseline）

| group | Δp95 | Δ最近邻中位 | 有效秩比 |
|---|---:|---:|---:|
| locomotion | −0.167 | −0.068 | 0.735 |
| stationary | −0.186 | +0.002 | 0.625 |
| transition | −0.231 | −0.222 | 0.491 |

有效秩比低于上一版的 0.75 阈值，但**可分性三项全面优于 baseline**。原因在逐通道有效秩里
看得见：信息没有消失，而是分到了方差不等的三个通道上，参与率因此下降。

| 通道 | locomotion | stationary | transition |
|---|---:|---:|---:|
| head | 4.99 | 3.57 | 15.46 |
| direction | 3.53 | 3.04 | 2.67 |
| modifier | 5.95 | **20.17** | 3.08 |
| baseline 整串 | 16.23 | 19.75 | 19.48 |

stationary 的 modifier 通道单独就超过 baseline 整串的有效秩。

### 4.3 槽组合数值诊断

| 槽 | 枚举配置数 | 最坏配置对余弦 | 最小成员 readout 间隔 |
|---|---:|---:|---:|
| head | 1024（32 单头 + 992 有序对，含 `R_B`） | 0.8109 | +0.940（land） |
| direction | 63（全部非空子集） | 0.9578 | +0.560（right） |
| modifier | 45825（≤3 词子集） | 0.9690（`catch,kick` vs `bite,catch,kick`） | +0.626（bite） |

head 和 direction 的合法域全部穷举；modifier 只穷举到当前语料上限 3，用于观察数值邻近程度，
不再声称覆盖总词数上限 8 下的全部组合。全域不碰撞与成员可读出由 §4.1 的槽源满秩证明。

## 5. `_VOCAB_T5_TEXT`：1hand / 2hand 改写

`weapon, 1hand` vs `weapon, 2hand` 曾是全语料最坏的近邻对。根因在词表文本：
`"weapon in one hand"` / `"weapon in both hands"` 共享 weapon 语义，两个 atom 余弦 0.784，
是整张表最近的一对。整串编码时代这没有代价；词级条件下，两个 token 共享的内容正是它们的标签
分不开的部分。语料里 1hand/2hand 从不单独出现（47/47 个标签都带 weapon 类词），weapon 语义
已经由同一标签里的另一个 token 承载，1hand/2hand 需要携带的只是**数量**。

改为 `"one hand"` / `"both hands"` 后：

| 指标 | 改前 | 改后 |
|---|---:|---:|
| cos(1hand, 2hand) | 0.784 | **0.526** |
| locomotion 最坏近邻 | 0.9775 | **0.9420** |
| stationary 最坏近邻 | 0.9685 | **0.9389** |
| transition 最坏近邻 | 0.9560 | 0.9559 |

改后该对不再是任何 group 的最坏近邻。另测的 9 组候选中，凡是保留 weapon 锚的（如
`"one-handed weapon"`、`"wielding a sidearm"`）下游最坏近邻反而**更差**（0.976～0.987）：
锚得越牢，两个 atom 的共享分量越大。`"single handed"` / `"double handed"` 在抽象义污染上最差
（−0.250），与词表注释里记录的旧测量一致。

改动只影响词表指纹：当前 sidecar 是 label-keyed 且 `vocab_t5_text_applied: False`，
所以已训练的 checkpoint 与既有向量都不受影响，只是下次运行
`tools/build_action_label_embeddings.py` 会把旧 sidecar 判为 stale 并重建（重建结果逐位相同）。
这也是改这张表的最后一个免费窗口 —— word-keyed sidecar 一旦落地，再改就要重建向量并重训。

## 6. 对实现的约束

- loader 只发 `word_ids / role_ids / word_mask / order_head_mask / slot_ids`，槽拼装在模型侧
  用 checkpoint 内的冻结词表完成；离线不预拼 label→bundle 查表，否则表示被锁进数据通路。
- 槽拼装的唯一实现是
  [`assemble_slot_channels`](../data_loaders/truebones/truebones_utils/action_label_conditioning_contract.py)，
  模型侧用张量镜像同一套 `slot_ids`，不另写一份。
- 学习后的通道缩放、CFG 与生成动作质量不属于这份训练前预检；它们仍必须进入实现后的消融与
  held-out 生成验收。
