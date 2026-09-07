# 关节条件重构：结构化骨架通道 + joint name 同义增广

> 状态：**设计稿（2026-09-07），未实现、未训练。** 本文所有数字都是
> **离线可复现的语料测量**，跑在 `dataset/merged/cond.npy`（260 物种 / 9699 关节）
> 和 `merged_locomotion_v5_pwp` step 200k 的生成结果上。
>
> 三条要先说清楚的：
>
> 1. **没有任何训练侧证据。** 本文不声称任何 loss / 生成质量的增益。§3–§5 的论证
>    全部是"这个信号现在不存在 / 被稀释到什么程度"，不是"补上以后会好多少"。
>    验收指标见 §8，要跑完才有话说。
> 2. **上一版方案里的 `d_contact`（到最近 contact 关节的图距）已被实测否掉**，
>    原因见 §3.1；本文的结构描述子是重新设计的，不要拿旧提法对照。
> 3. **"复用 `rest_pose` 的 10 个死维度"这个落点也被否掉了**，原因见 §3.4 ——
>    `rest_pose` 在时间轴上只占 1 帧，而 joint name 加在全部帧上，两者影响力差 60 倍。
>
> 前置事实见 [[project-bare-leg-token-reverses-knee]]：同一副骨架只改腿骨命名，
> 膝盖屈曲从 +39.4°/100% 同号翻成 −17.9°/81% 反号。本文是那次事故的结构性收尾。

---

## 1. 为什么要动

### 1.1 关节名是模型里唯一的 per-joint 身份信号

`model/anytop.py:1151` 的 `x = x + text_embedding(name_emb)` 是**唯一**告诉模型
"这个 token 是什么关节"的东西。这不是元数据，它就是骨架上的位置编码。

已经付出过代价：`LeftShin` → `LeftLeg` 这一个改名，在 `offsets` / `parents` /
`rest_pose` / `tpose_rest_rotations` / `graph_dist` / `joint_relations` **全部逐字节相同**
的前提下，把两条腿的膝盖都翻了向。

### 1.2 T5 mean-pool 对长短语的稀释（本文新增的实测）

和 action label 走过的路一样（见 `action_label_per_word_pooling.md`），
整串短语过 T5 再 mean-pool，短语越长，单个词对最终向量的贡献越小。

**同形状探针**：只取词数相同、且恰好只有一个位置不同的文本对，均值中心化后算余弦。
（探针方法论见 [[feedback-t5-geometry-probe-methodology]]：不做同形状 + 中心化会测到
长度和 EOS 的伪影。）语料内 2797 个不同文本，全部这样的对：

| 词数 | 对数 | 平均余弦 | 中位数 |
|---:|---:|---:|---:|
| 1 | 465 | **0.345** | 0.344 |
| 2 | 1796 | 0.450 | 0.434 |
| 3 | 590 | 0.714 | 0.726 |
| 4 | 160 | 0.799 | 0.815 |
| 5 | 295 | 0.762 | 0.783 |
| 6 | 1351 | 0.802 | 0.870 |
| 7 | 2161 | 0.697 | 0.687 |
| 8 | 696 | 0.818 | 0.858 |
| 9 | 189 | 0.882 | 0.914 |
| 10 | 113 | 0.886 | 0.912 |
| 11 | 701 | 0.913 | 0.925 |
| 12 | 718 | 0.915 | 0.934 |
| 13 | 430 | 0.935 | 0.948 |
| 14 | 24 | **0.952** | 0.958 |

余弦越高 = 那个不同的词越无关紧要。**14 个词的文本里换掉一整个词，向量只动 0.048。**
这不是"信号变弱"，这是信号基本没了。

而当前的词数分布是重的：

```
 0 词:  256      6 词: 1603      11 词:  461
 1 词:  793      7 词: 1564      12 词:  418
 2 词: 1599      8 词:  934      13 词:  249
 3 词:  822      9 词:  301      14 词:   20
 4 词:  301     10 词:  134
 5 词:  244
```

**9699 个关节里 5684 个（58.6%）是 6 词以上，1583 个（16.3%）是 9 词以上。**
平均 5.78 词。

### 1.3 62% 的词根本不需要 T5

把 54572 个词 token 按性质拆开：

| 类别 | 词数 | 占比 |
|---|---:|---:|
| 结构性（`Segment` `First..Tenth` `Of` `N` `ChainStart/Middle/End` `Instance` `Contact` `EndEffector`） | 33855 | **62.0%** |
| 侧别（`Left` / `Right`） | 6599 | 12.1% |
| 语义（真正需要开放词表的身体部位词） | 14118 | 25.9% |

结构性的那 62% **完全可以从 `parents` 推出来**——模型手里已经有 `parents`。
它们被写进文本，唯一的效果就是把剩下 26% 真正需要 T5 的部分稀释掉。

这就是本文的中心论点：**当前设计把一个可以精确计算的量，编码成自然语言，
再用一个会做 mean-pool 的编码器去压缩，最后压垮了旁边那个真正需要它的量。**

---

## 2. 现状盘点：四个"结构通道"实际装了什么

在动手之前先确认现有通道为什么补不上这个洞。以下全部是 `merged/cond.npy` 实测。

### 2.1 `rest_pose`：13 维里只有 3 维是活的

`[pos(3), rot6d(6), vel(3), contact(1)]`。在 **260/260 个物种**上：

```
rest_pose[:,3:9]  (rot6d)   == 单位阵    260/260
rest_pose[:,9:12] (vel)     恒等于 0     260/260
rest_pose[:,12]   (contact) 恒等于 0     260/260
```

每个关节实际只提供 T-pose 的 3 个坐标。**且 `contact` 维恒为 0** ——
"这个关节是着地点"从来不是条件；contact 只存在于被去噪的 `x` 里（每帧二值），
推理时从噪声开始。它唯一的先验入口是名字文本里那个 `Contact` 词。

而且这 3 个坐标**不是跨物种可比的**：

```
y_min  mean +0.102  std 0.244  range [-0.513, +1.835]    落在 [-0.02,0.02] 的: 133/260
span   mean +0.764  std 0.344  range [+0.034, +1.680]    落在 [0.8,1.2]  的:  87/260
```

地面不在 0，身高不是 1。模型要自己从 3 个原始坐标里推出地面和体尺。

### 2.2 `graph_dist`：无向、对称、5 跳饱和

`dataset_pipeline.py:64-101`，`MAX_PATH_LEN = 5`，且 `topo_rel[i,j] = topo_rel[j,i]` ——
**无方向**。53 关节人形上的直方图：

```
dist: 0→53  1→104  2→150  3→196  4→242  5→2064
```

**2064 / 2809 = 73.5% 的关节对落在同一个 "far" bucket。** 小腿↔手臂、小腿↔头、
小腿↔尾巴，模型看到同一个 token。

### 2.3 `joint_relations`：只区分 1 跳

6 个码 `self/parent/child/sibling/no_relation/end_effector`，同一副 rig：

```
no_relation: 2600 (92.6%)   child: 52   parent: 52   sibling: 52   self: 40   end_effector: 13
```

方向只在 1 跳里有。`end_effector` 写在对角线上（判据是"无子节点"，不依赖名字）——
这是现有通道里**唯一**一个真正的、名字无关的 per-joint 结构标记。

### 2.4 `kinematic_chains`：根本没进模型

`_load_physical_motion` 返回了它，但 `data_loaders/tensors.py` 的 `item` dict 里没有这个 key。
同样没进模型的还有 `contact_joints`、`symmetry_partner_indices`、`face_joints`。
所谓"链结构"目前只是离线量。

### 2.5 小结

| 通道 | 形式 | 分辨率 | 名字无关？ |
|---|---|---|---|
| `rest_pose` | per-token 内容（**但只在 1 帧上**，见 §3.4） | 3 个未归一化坐标 | ✅ |
| `graph_dist` | 成对 attention bias | 无向，5 跳饱和，73.5% 同 bucket | ✅ |
| `joint_relations` | 成对 attention bias | 1 跳，92.6% `no_relation` | ✅ |
| `kinematic_chains` | — | 不进模型 | — |
| `joints_names_embs` | **per-token 内容，全部帧** | 高，但被 §1.2 稀释 | ❌ |

**成对关系 ≠ token 内容。** `graph_dist`/`joint_relations` 只进 attention logits
（`topology_key_emb`/`query_emb`，`model/motion_transformer.py:944`），从不进 token 的内容向量。
注意力只能在已有内容上重新加权，改不了"我是什么"。

有个说明问题的细节：翻转案例里模型其实**手里有全部拼图**——小腿的父节点名字是
`Left Thigh`（正确）、子节点是 `Left Foot Contact`（正确），`joint_relations` 也标了谁 parent 谁 child。
"夹在 thigh 和 foot 之间 ⇒ 我是小腿"原则上一跳注意力就能推。推不出来，是因为这要求模型去
**覆盖**自己 token 里从第 0 层就写死的 `Left Leg ≈ Left Arm`。

---

## 3. 方案 S：结构化骨架通道

### 3.1 设计约束：为什么 `d_contact` 失败

上一版提过"到最近 contact 关节的图距"，**实测否掉**：绝对跳数随骨架大小变化。
人的前臂离脚 9 跳，四足的前臂离前掌 1 跳——同一个解剖角色拿到完全不同的值。
在正常的 A 骨架上误报 18/53。

由此定下三条硬约束：

1. **尺度自由**：不许出现绝对跳数、绝对长度、绝对坐标。所有量要么归一化到 [0,1]，
   要么除以骨架自身的尺度。
2. **链内相对**：位置信息必须相对于**该关节自己所在的那条链**，不是相对于全局。
3. **名字无关**：只能读 `parents`、`rest_pose[:, 0:3]`、`contact_joints`。
   读到任何 `joints_names` 就是设计错误。

### 3.2 链的定义

不能直接用 `cond['kinematic_chains']` ——它是 root-to-tip 的贪心合并路径，
人形的第 3 条链是 `Hips→Spine→Spine1→Spine2→RightShoulder→...→RightHandPinky3`，
把躯干、手臂、一根手指串成一条，链内深度没有意义。

改用**无分叉段（branch-free run）**：从一个分叉点（或 root）出发，
沿着"每个内部节点恰好 1 个子节点"一直走到叶子或下一个分叉点。

人形：`Hips` 是分叉点 → `LeftThigh→LeftShin→LeftFoot→LeftToeBase` 是一条长度 4 的 run。

### 3.3 12 维描述子

| # | 名称 | 定义 | 为什么 |
|---:|---|---|---|
| 1 | `depth_norm` | `(k+1)/L`，k 是在 run 内的序号，L 是 run 长度 | 取代文本里的 `Segment Second Of 3` |
| 2 | `run_len` | L | 同上，给 `depth_norm` 定标 |
| 3 | `run_ends_contact` | run 的末端（或其子节点）是不是 contact 关节 | **取代失败的 `d_contact`**；人形的腿 run = 1、臂 run = 0，四足前后肢都 = 1（这是对的） |
| 4 | `is_contact` | 该关节本身是不是 contact | 取代文本里的 `Contact`；现在**完全不是条件**（§2.1） |
| 5 | `is_leaf` | 无子节点 | 取代 `EndEffector`；与 `joint_relations` 码 5 冗余，保留是为了让它进 token 内容 |
| 6 | `height_n` | `(y - y_min) / (y_max - y_min)` | 地面参考 + 体高归一化的高度。**这是分开大腿/小腿的那一维** |
| 7 | `lateral_signed` | `x / (y_max-y_min)`，**带符号** | 侧别 + 侧向展开；符号位见 [[project-skeleton-renamer-lr-orient-fixes]] |
| 8 | `fore_aft_n` | `(z - z_mean) / (y_max-y_min)` | 前后位置，分开鸟爪的多根趾 |
| 9 | `attach_h_n` | 该 run 起点的 `height_n` | 该肢体挂在躯干什么高度上，分前肢/后肢 |
| 10 | `subtree_n` | 子树关节数 / 总关节数 | 分支重要性 |
| 11 | `sib_rank` | 同父兄弟按 (z, x) 排序后的归一化名次 | 取代文本里的 `Instance Second Of 4` |
| 12 | `sib_n` | 兄弟数 | 给 `sib_rank` 定标 |

全部只读 `parents` / `rest_pose[:,0:3]` / `contact_joints`。

### 3.4 落点：**不要**复用 `rest_pose` 的死维度

上一轮我提过把这些塞进 `rest_pose` 那 10 个恒为常数的维度。**这是错的**，理由是时间轴：

```python
tpos_embedded = ...                                  # [1,   B, J, d]  ← rest_pose 只有 1 帧
x_embedded    = ...                                  # [T,   B, J, d]
x = torch.cat([tpos_embedded, x_embedded], dim=0)    # [T+1, B, J, d]
x = x + joints_embedded_names[None, ...]             # ← name 加在全部 T+1 帧上
```

`rest_pose` 是 T+1 帧里的 **1 帧**；joint name 加在**每一帧**上。60 帧生成时是 1:61 的
呈现频次差。把结构信息塞进 rest_pose，等于让它以 1/60 的强度去和名字竞争。

**定稿落点：与 name 并列的、逐帧相加的独立通道。**

```python
# InputProcess.__init__
self.struct_embedding = nn.Sequential(
    nn.Linear(STRUCT_DIM, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim)
)
# InputProcess.forward，紧跟在 name 那一行后面
x = x + joints_embedded_names[None, ...]
x = x + self.struct_embedding(joint_struct)[None, ...]
```

三个必须写死的点：

- **不许 zero-init 最后一层。** 直觉上零初始化"从恒等开始、平滑接入"很诱人，
  但 [[project-skeleton-renamer-s8-known-cond]] 上一次就是这么踩出一个**死通道**的：
  梯度进不去，通道永远是 0。用正常 init。
- **`joint_name_drop_prob` 不许作用在这个通道上。** 整个机制的意义就是"名字被抹掉时
  还剩什么"。两者同时被抹 = C1 白做。
- **padding 行必须显式清零**，不能靠"全零 = padding"来检测——语料里有
  **256 个真实关节**本来就是全零 name 行（`locator2` / `C_ctrl` / `Bip01` / `Saddle`）。

### 3.5 实测验证

**(a) 名字无关性。** A（`LeftShin`）与 B（`LeftLeg`）两副 cond 的 12 维描述子
**逐关节完全相同**。也就是说这个通道存在时，那次翻转在条件层面就不可能发生。

**(b) 跨物种对齐。** 同一解剖角色在不同物种上的读数：

| 物种 | 关节 | `depth_norm` | `run_len` | `run_ends_contact` | `height_n` | `attach_h_n` |
|---|---|---:|---:|---:|---:|---:|
| new/Human (A) | LeftThigh | 0.250 | 4 | 1.0 | 0.590 | 0.590 |
| new/Human (A) | **LeftShin** | **0.500** | 4 | 1.0 | **0.325** | 0.590 |
| unitybundles/KI_Human | B-thigh.L | 0.250 | 4 | 1.0 | 0.583 | 0.583 |
| unitybundles/KI_Human | **B-shin.L** | **0.500** | 4 | 1.0 | **0.297** | 0.583 |
| zoo_upgrade/Hen | LeftUpLeg | 0.250 | 4 | 1.0 | 0.426 | 0.426 |
| zoo_upgrade/Hen | **LeftLeg** | **0.500** | 4 | 1.0 | **0.184** | 0.426 |
| zoo/Rat | LeftUpLeg | 0.333 | 3 | 1.0 | 0.584 | 0.584 |
| zoo/Rat | **LeftLeg** | **0.667** | 3 | 1.0 | 0.542 | 0.584 |
| zoo/Fox | Bip01_L_Thigh | 0.167 | 6 | 1.0 | 0.649 | 0.649 |
| zoo/Fox | **Bip01_L_Calf** | **0.333** | 6 | 1.0 | **0.386** | 0.649 |
| new/Human (A) | LeftArm | 0.500 | 4 | **0.0** | 0.917 | 0.924 |

大腿/小腿在 `depth_norm` 上永远是 run 内的第 1/第 2 节，在 `height_n` 上永远是高/低；
`run_ends_contact` 干净地分开腿（1.0）和臂（0.0）——这正是 `d_contact` 想做而做不到的事。

**(c) 表达力。** 见 §4.2，它要和文本瘦身一起量。

---

## 4. 方案 T：文本瘦身

### 4.1 规则

从 `build_joint_embedding_texts` 的输出里剥掉这些词，交给 §3 的通道承载：

```
Segment  First..Twelfth  Of  <数字>
ChainStart  ChainMiddle  ChainEnd  ChainEarly  ChainLate
Instance  Contact  EndEffector
```

保留：`Left` / `Right` + 身体部位词。`Left Hand Thumb Segment Third Of 3 ChainEnd EndEffector`
→ `Left Hand Thumb`。

**注意这不是删逻辑**：`_build_chain_relative_joint_tokens` 和 `_sibling_instance_tokens`
照常运行，只是产物改成写进结构通道，不写进文本。当前的撞名消歧仍然由它们保证。

### 4.2 实测：瘦身丢了什么、结构通道补回多少

核心问题是——剥掉这些词以后，同一副骨架里会有多少关节变得无法区分？结构通道能补回几个？

`merged` 语料，同一骨架内两边都有名字的关节对共 **212860** 对：

| | 对数 |
|---|---:|
| 现状就撞名（文本完全相同） | **0** |
| 剥掉结构词后新增的撞名 | 11783 |
| ├ 12 维结构描述子能分开（L2 ≥ 0.02） | **11781（99.98%）** |
| └ 仍然分不开 | **2** |

剩下的 2 对是 `zoo/Monkey` 的 `Bip01_L_Finger21` / `Bip01_L_Finger11` 及其右侧镜像——
T-pose 里几何重合的手指。

> 中间结论值得记一笔：先用 9 维（`abs` 侧向、无兄弟名次）时残留 **318 对**，
> 几乎全是鸟爪的 `Toe0/1/2/3`、蝎子体节、猴子手指——纯粹靠 `Instance N Of M` 区分的兄弟。
> 加上 `lateral_signed` / `fore_aft_n` / `sib_rank` / `sib_n` 四维后降到 2 对。
> **`sib_rank` 不是可选项**，它是 `Instance` 那组词的唯一替代。

### 4.3 瘦身带来的去稀释

| | 现状 | 瘦身后 |
|---|---:|---:|
| 平均词数 | 5.78 | **1.95** |
| 最长词数 | 14 | **4** |
| 6 词以上的关节 | 5684 (58.6%) | **0** |
| 不同文本数 | 2797 | 676 |
| 按分布加权的"换一个词的余弦"（§1.2 曲线） | 0.697 | **0.475** |

语义词对最终向量的贡献大致翻倍。**这是本方案的主要收益**，而代价是 212860 对里的 2 对。

### 4.4 S 和 T 必须同时上

单独上 T = 直接损失 11783 对区分度。单独上 S = 稀释问题原样保留，而且新通道要和
一个仍然臃肿的文本通道抢容量。**两者是一个改动，不是两个。**

---

## 5. 方案 A：joint name 同义增广

### 5.1 现在的同义表是"塌陷"，是推理期的单点故障

`_EMBED_TEXT_SYNONYM_TOKENS` 有 62 条映射、17 个"多拼写 → 一个规范词"的组：

```
Calf     <- tibia, fibula, clf, shin
Thigh    <- femur, thi
Hand     <- carpal, carpus, metacarpal, metacarpus, palm, hnd
Foot     <- cannon, tarsal, metatarsal, metatarsus, feet, fot
Clavicle <- scapula, collarbone, clv, clav, scap
UpperArm <- humerus, humer, uar
Forearm  <- radius, ulna, far
```

它在**预处理期**把所有拼写塌陷成一个。后果：

- 模型训练时**只见过** `Calf`，从没见过 `Shin` / `Tibia` / `LowerLeg`；
- 于是模型的全部泛化责任被推给了这张手写表；
- 表一旦漏一个拼写，模型拿到的就是一个没见过的 token，且**没有任何回退**。

`LeftLeg` 事故就是这个模式的完整演示：表里没有"裸 `leg` 在 thigh 之下时 = Calf"这条，
于是 `Left Leg` 落到 T5 空间里 `Left Arm` 附近，模型照着鸡和老鼠的腿去生成。
（该规则已在 `_bare_leg_means_calf` 补上，但那是补了**一个洞**，不是补了这个模式。）

### 5.2 增广：把塌陷从硬规则变成软先验

保留预处理期的塌陷（保证确定性和 cond 可复现），但在**训练时**按概率把规范词
**再展开**回它的某个已知拼写：

```
预处理（不变）:  LeftShin / LeftTibia / LeftLowerLeg  ──►  "Left Calf"      写进 cond
训练时增广    :  "Left Calf"  ──采样──►  {Left Calf, Left Shin, Left Tibia, Left LowerLeg}
推理          :  仍然用规范拼写，行为不变
```

模型于是学到"这四个拼写指同一个关节"，而不是"只有 `Calf` 是合法输入"。
一副没见过的、把小腿写成 `Shin` 的 rig，即使词表漏了，也落在训练见过的邻域里。

### 5.3 实现

**离线**（一次，`tools/build_joint_name_synonym_embs.py`）：

瘦身后只有 **676 个不同文本**，语义词表 **171 个词**，其中 top-60 覆盖 95.1% 的语义词出现次数。
所以可以把**全部变体预先烘成表**，训练时零 T5 调用：

```
joint_name_synonym_embs.npy
  { canonical_text -> float32[K, 768] }      # K = 该文本的拼写变体数，含规范拼写本身
```

**训练时**（dataset 侧，per (sample, joint) 采样一个 k）：与 C1 的整关节 drop 组合成
一条链：`同义采样 → 整关节 drop（替换成 unknown_joint_name） → 逐元素 dropout → FiLM`。
顺序不能变，理由和 C1 一样（见 `InputProcess.forward` 的注释）：FiLM 必须看到
**替换之后**的行。

**推理时**：不变，只用规范拼写。所以**不改 checkpoint 契约、不改 cond schema、
不需要 regen**——sidecar 只在训练侧读。

### 5.4 范围控制（这是风险最大的一节）

同义组是**人工断言**的，断错就是往训练里注标签噪声。约束：

- **只展开已经在 `_EMBED_TEXT_SYNONYM_TOKENS` 里、已经在生产中用了的 17 组。**
  这些拼写本来就已经被塌陷成同一个词了——增广只是不再隐藏这个事实，不引入新断言。
- 扩充新组要单独 review，且必须过一遍 §8 的留一验证。
- **不做形态学增广**（复数、大小写、加连字符）。`Wings` vs `Wing`、`Fangs` vs `Fang`
  在语料里是不同关节，不能当同义。

---

## 6. 与已落地部分的关系

| 已落地 | 与本文的关系 |
|---|---|
| `_bare_leg_means_calf`（词表补洞，推理侧） | 补的是一个洞。§5 补的是"靠手写表兜底"这个模式 |
| `joint_name_support_report.json`（OOD 预检） | 正交；它是护栏，S/T/A 是把护栏后面的东西变结实 |
| `--joint_name_drop_prob` / `--joint_name_drop_all_prob`（C1） | **强依赖**。C1 的意思是"名字没了要能退回结构"，而现在退回去只有 §2 那点东西。S 是 C1 的落点 |

C1 现在已经在 `train.bat` 里开着（0.15 / 0.05）但还没重训。**建议 S/T/A 和 C1 一起进同一轮训练**，
不要先单独训一个 C1——C1 单独训的效果上限就是 §2 那张表。

---

## 7. 实施顺序

| 步 | 内容 | 需要 regen？ | 需要重训？ |
|---:|---|---|---|
| 1 | `build_joint_struct_features(parents, rest_pose, contact_joints)`，纯函数 + 单测 | 否 | 否 |
| 2 | dataset 构造期按物种算一次并缓存；`generate.py` 走同一个函数 | **否**（可从 cond 现场算） | 否 |
| 3 | collate 发 `joint_struct`；`InputProcess.struct_embedding` | 否 | 是 |
| 4 | `build_joint_embedding_texts` 加 `--slim` 输出结构词分离的两份产物 | **是**（cond 里的 `joints_names_embs` 变了） | 是 |
| 5 | `build_joint_name_synonym_embs.py` + dataset 侧采样 | 否（sidecar） | 是 |
| 6 | 一轮训练，S+T+A+C1 全开 | — | — |

第 1–3 步**不需要 regen**是个重要性质：12 维描述子完全可以从现有 cond 的
`parents` / `rest_pose` / `contact_joints` 现场算出来。只有第 4 步（文本瘦身）动到
`joints_names_embs`，才需要 regen。

`cond` 的 key 集合：`joint_struct` 如果决定要 bake 进 cond，必须**所有物种都有**，
不能有的有有的没有——见 [[project-cond-key-set-must-be-stable-for-compile]]。
更省事的做法是干脆不 bake，走第 2 步的现场计算。

---

## 8. 验收指标

**离线（改完就能跑，不用训练）**

1. `build_joint_struct_features` 在 260 物种上对 A/B 两副 cond 输出**逐位相同** ——
   这是"名字无关"的定义，不过就是实现错了。
2. 同骨架内撞名：瘦身后新增 11783 对，结构描述子未解决 ≤ 5 对。
3. 词数：平均 ≤ 2.0，最长 ≤ 4，6 词以上为 0。
4. §1.2 的同形状探针在瘦身后的文本上重跑，加权余弦 ≤ 0.50。
5. 同义增广的留一验证：把每个同义组里的一个拼写留出来，确认它的 T5 向量
   与组内其它拼写的均值中心化余弦 > 该拼写与**其它组**的最大余弦。
   （不过就是同义组断错了。）

**训练后**

6. **回归测试就是这次的 bug**：用 B 骨架（`LeftUpLeg`/`LeftLeg`）生成 walk，
   左右膝有符号屈曲角必须与 A 同号，`frac_positive ≥ 0.95`。
   测法见 §9 的脚本；A 的地板是 +39.4°/100%。
7. 语料内物种的生成质量不劣化——对着各物种自己的真实 clip 读绝对指标，
   **不**对着另一个 checkpoint 读（理由同 `action_label_per_word_pooling.md` §12.1：
   `cond.npy` 和语料都变了，逐格比会得出无法归因的数字）。
8. 名字盲测：推理时把 `joints_names_embs` 整体换成 `unknown_joint_name`，
   生成结果应当仍然是一个站立行走的骨架（而不是塌掉）。这是 S 是否真的接住了 C1 的判据。

---

## 9. 复现

本文所有数字来自 `dataset/merged/cond.npy` 与
`outputs/merged_locomotion_v5_pwp/generate_step000200000_human{,_b}/`。

- §1.2 同形状探针、§1.3 词性拆分、§4.2/§4.3 瘦身统计：只读 `cond.npy` 的
  `joints_names_embs` + `joints_names_embs_meta.embedding_texts`，不需要 T5。
- §3.5 描述子表：`parents` / `rest_pose[:,0:3]` / `contact_joints`。
- §8.6 的膝盖角：解析生成的 BVH 做 FK，取 `cross(thigh_dir, shin_dir)` 在
  内外侧轴上的投影符号。A 的读数 +39.44°/100%、B 的读数 −17.92°/19%。

---

## 10. 明确没做 / 没验证的

- **没有任何训练侧证据。** §3–§5 全部是"这个信号不存在 / 被稀释成什么样"的离线论证。
- **没有消融。** S、T、A、C1 四项打算一起进同一轮训练，所以这一轮出来也无法归因到单项。
  要归因需要额外的 4 次训练，成本自负。
- **`sib_rank` 的排序键（先 z 后 x）是拍的**，只验证了它能把 §4.2 的 318 对降到 2 对，
  没有验证它在别的骨架上是否稳定。旋转对称的骨架（海星、蜘蛛）上这个排序可能抖。
- **同义组只覆盖已有的 17 组**，没有为长尾的 171 个语义词补新组。
- `height_n` / `attach_h_n` 依赖 `rest_pose` 的 Y 轴就是重力轴。语料里
  `orientation_quat` 已经把骨架转到规范朝向，但这个假设没有单独验证过。
