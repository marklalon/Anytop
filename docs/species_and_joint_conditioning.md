# species 描述符与 joint 嵌入

本文描述 AnyTop 输入层里**每个关节 token 的身份信息**如何构成：关节名嵌入（joint-name embedding）、物种描述符（species code）对它的 FiLM 调制，以及和它们并列的结构通道（structural channel）。只写当前实现；会随代码变动的数值（词表规模、schema 版本、秩）不在此固化，以代码为准（§8）。

---

## 1. 总览

输入层 `InputProcess`（[anytop.py](../model/anytop.py)）为每个 `(帧, 关节)` 生成一个 latent token，由四部分相加：

```
token(frame, joint) =  运动投影(该帧该关节的特征)          # 帧 0 为 rest pose 的投影
                     + text_embedding( 关节名 ⊕ species 调制 )   # 本文 §3 §4
                     + struct_embedding( 关节结构特征 )          # 本文 §5
                     + 帧位置编码
```

三条与"这是什么关节 / 这是什么物种"有关的信号，各管一件事：

| 信号 | 回答的问题 | 来源 | 随什么变 |
|---|---|---|---|
| 关节名嵌入 | 这个关节是**身体的哪个部位**（语义） | 冻结 T5，烘焙在 `cond.npy` | 骨架 |
| 结构通道 | 这个关节在**骨架的什么位置**（几何 / 拓扑） | 由 `parents` 与 rest 位置现场计算 | 骨架 |
| species 描述符 | 这个物种**怎么动**（体型类别 × 变体） | body plan 取冻结 T5（描述符表），variant 取正交码 | 物种 |

species code **只**通过“对关节名嵌入做 FiLM”进入模型，不进入 timestep 条件，decoder 里也没有 species AdaLN 头。timestep 条件只承载 diffusion 步、`resample_speed`、action label 等（见 [action_group_and_label.md](action_group_and_label.md) 与 [anytop_model_architecture.md](anytop_model_architecture.md)）。

---

## 2. species 描述符

### 2.1 它是什么

一个物种的描述符是 `[body-plan]` 或 `[body-plan, variant]`（见 [dataset_tags.py](../data_loaders/truebones/truebones_utils/dataset_tags.py)）：

- **body plan**（必填）：封闭词表 `SPECIES_BODY_PLANS`（Quadruped / Biped / Multiped / Serpentine / Aquatic / Winged / Drifting）。
- **variant**（可选）：封闭词表 `SPECIES_VARIANTS`（Clanking / Fluttering / Waddling）。它表示整个物种都有、且帧数调不出来的运动风格：机械的僵硬节拍、短腿左右摇摆、快速浅幅振翅。按 body plan 默认方式运动的物种不写 variant。

步态快慢不进描述符，也不进 action_label：由 `--num_frames`（循环周期）决定；要指定某种具体步态，用 reference motion。翅膀状态里只有 `glide`（滑翔，翅膀不动）留作逐 clip 的 action_label 修饰词；拍翅是 `fly` 的默认，不写词。

variant 的取舍按物种而定，不按 clip 中有没有表现出来：一个物种有任一 clip 呈现该风格，整个物种就带这个 variant。同一物种只能有一个 variant。

人工维护在每个数据集的 `species_tags.jsonl`：

```json
{"species": "Cat", "species_tags": ["Quadruped"]}
{"species": "RU01_TechRobot", "species_tags": ["Biped", "Clanking"]}
```

超过两个槽的行视为过期 sidecar，直接拒绝，不会被静默编码成另一段文本。species 识别 LLM 只给 body plan，variant 由人工在 review 页面设置。`species_tags.jsonl` 必须覆盖训练用到的每个物种，没有回退。

### 2.2 `species_code` 从哪来：描述符表

合法描述符集合是有限的（每个 body plan 一条，再加每个 body plan × variant 一条），所以**全部描述符被一次性编码成一张表**，物种向量不存在 `cond.npy` 里：

- `cond.npy` 只带每个物种烘焙进去的 `species_tags`；
- 训练启动时构建（或扩充）仓库级的 `dataset/species_descriptor_embs.npy`——这是**唯一**为物种跑 T5 的地方——把正在训练的 cond 绑定上对应的行，并把这张表拷到 `save_dir`，与 checkpoint、`cond.npy` 放在一起；
- 生成与 `--species_tags`（命令行和 AnyTop 服务一样）**只读 checkpoint 自带的那份表**；表里没有的组合直接拒绝，而不是现场编码一个权重从没见过的向量。

行向量是 T5Conditioner 对描述符文本的 masked token mean，维度与关节名嵌入相同。绑定时每个 cond 条目得到两个向量：

- `species_emb`：整条描述符文本的表行（video2pose 读它）；
- `species_code`：AnyTop 读的条件，`[T5(body plan) ‖ variant one-hot]`（`species_descriptor_table.species_code`）。body plan 取表里**不带 variant** 的那一行；variant 不过 T5，每个 variant 占一根单位轴，没有 variant 时全零。T5 下 `Winged` 与 `Winged Fluttering` 几乎重合，variant 只是 body plan 方向里一个很小的差；改成正交码后，它是与 body plan 部分和其他 variant 都严格正交的独立方向。宽度见 `dataset_tags.species_code_dim`。

多个物种共用同一个描述符（同 body-plan 与 variant），所以描述符的**不同取值数远少于物种数**。这是 species 通路天花板的来源。

### 2.3 推理时改风格：`--species_tags`

`--species_tags Quadruped` 用描述符表算出的 species code **替换**目标物种自己的 `species_code`，从而重新"着装"：例如让 Winged 的龙按四足方式运动；`--species_tags Biped,Clanking` 让人形按机械方式走。

- variant 的训练样本只来自少数几个 rig 家族（Clanking 主要是 RU01 机器人，Waddling 主要是 MU01 / TTR），覆盖到其他骨架时效果可能偏弱，必要时加 `--species_cfg_scale`；
- 相似度（skeleton_similarity / reference_bank / clip_length_prior）只比较 body plan，不看 variant；

- 要求 checkpoint 以 `--species_cond` 训练；
- 组合必须在 checkpoint 的描述符表里；
- 与 `--object_type all` 不兼容；
- 骨架不在 checkpoint 的 `cond.npy` 里的新骨架（`process_new_skeleton`）必须显式给 `--species_tags`，没有静默回退到某个默认物种。

---

## 3. 关节名嵌入

### 3.1 文本如何生成

每个关节的名字被转换成一句**"侧别 + 身体部位"**的短文本（`build_joint_embedding_texts`，[joint_embedding_text.py](../data_loaders/truebones/truebones_utils/joint_embedding_text.py)），例如 `Left Upper Arm`、`Right Finger`。步骤概述：

1. 名字先经规范化（`canonicalize_joint_name`）：拆分粘连词、去掉物种前缀 / 绑定前缀（如 `Bip01`、`npc`）、折叠同义词与左右/前后/上下缩写码（`Lf` → Front、`TL` → Upper 等，只在上下文能区分时才读码）；
2. 侧别来自几何（`joint_side_labels`），不是名字；侧别词放句首（`Right Finger`），保持 T5 见过的名词短语构造；
3. **结构派生信息不进文本**：链上位置、同级编号、触地 / 末端标记归结构通道。因此同级重复部件（蜈蚣的腿对、蝙蝠的翼指）**共享同一段文本**，区分它们是 `sib_rank` / `fore_aft_n` / `lateral_signed` 的工作；
4. **无意义名字得到空文本**：辅助 / 占位节点（helper）、非解剖标记、仅由绑定戳构成的名字、孤立出现在肢体之外的肢体词，一律输出 `''`。

空文本编码为**全零行**，与 padding 行相同。模型因此只需学一个"名字未知"状态，同时覆盖：真的无名、被训练期随机隐藏、推理时遇到训练没见过的名字。

### 3.2 编码与存放

- 用冻结 T5（默认 `t5-base`，宽度 `t5_out_dim`）对每段文本做 masked token mean，**烘焙进 `cond.npy` 的 `joints_names_embs`**；训练和生成都读它，运行时不跑 T5。
- 同一 T5 型号下，同一段文本的向量与来自哪个 cond 无关，所以可跨 cond 复用（`embedding_cache_cond`）。
- `JOINT_NAME_EMBEDDING_SCHEMA_VERSION`（文本规则版本）记录在 checkpoint 的 `args.json`，不匹配则拒绝加载：张量形状能对上，含义已经变了。

### 3.3 推理遇到未见名字：置空

为新骨架烘焙 `cond.npy` 时（`blank_unseen_joint_names`），参考 cond（checkpoint 自带）里**没出现过**的名字文本一律换成空文本（全零行），原文保存在 `joints_names_embs_meta['blanked_unseen_texts']`。理由：一段训练从没见过的文本会被编码到 T5 空间里模型毫无先验的位置，而"全零 = 名字未知"是训练过的状态，模型此时靠结构通道和 rest pose 兜底。这条路径不需要 T5。

### 3.4 训练期对名字的两种 dropout

| 参数 | 粒度 | 作用 |
|---|---|---|
| `--joint_name_drop_prob` | **整个关节**的名字行清零 | 训练模型在"不知道这是什么关节"时回退到 rest pose、graph 距离、关系矩阵与结构通道；防止某个罕见 token 单点翻转整条肢体的运动先验。无 1/(1−p) 缩放（被隐藏的行不是一个被稀疏化的向量） |
| `--dropout_prob` | 名字向量**逐元素** | 常规正则，不改变关节身份 |

整关节 drop 抽样在 `AnyTop._sample_joint_name_drop`，写成纯张量比较（无 `.any()` 分支），不破坏 compile / cudagraph；评估与推理从不 drop。

---

## 4. species 对关节名的 per-joint FiLM

开关：`--species_cond`（`train_all.bat` 开启）。

### 4.1 结构

```text
joints_clean  = 关节名嵌入（已做整关节 drop，尚未做逐元素 dropout）      [B, J, D_t5]
species_b     = species_code 广播到每个关节                             [B, J, D_t5 + n_variant]

gamma_residual, beta = MLP( concat[joints_clean, species_b] )
                       Linear(D_t5 + D_code → bottleneck) → GELU → Linear(bottleneck → 2·D_t5)

modulated = (1 + gamma_residual) · dropout(joints_clean) + beta
name_vec  = text_embedding( modulated )                                  # Linear: D_t5 → latent_dim
```

- **concat 而非求和**：每个关节的 gamma / beta 是 `species × joint` 交互的函数。名字投影是一个没有非线性的 `Linear`，若改成 `joint + W(species)` 相加，结果恒等于 `E·joint + 一个只依赖物种的常量`——对所有关节完全相同，零关节分化。concat 之下，纯加性行为仍可由 `gamma_residual = 0, beta = W(s)` 平凡表示。
- **MLP 吃"逐元素 dropout 之前"的名字，调制作用在"dropout 之后"的名字上**：逐元素 dropout 训练时置零并按 1/(1−p) 缩放、推理时不作用；若让 MLP 吃它的输出，gamma / beta 本身就会产生 train / eval 分布错位。整关节 drop 则**必须**对 MLP 可见，否则 FiLM 会把刚被隐藏的名字原样还给模型。
- **隐层宽度** `--species_film_bottleneck`（默认 128，`train_all.bat` 为 64；0 表示取 `latent_dim`）：描述符的不同取值远少于物种数，宽隐层只是浪费。
- **零初始化**：输出层权重与偏置为零，起点是 identity（gamma = 1, beta = 0）；模型只需学"偏离"。
- MLP 在 fp32 下运行（`run_in_fp32`）。

### 4.2 CFG：整条调制可被 bypass

- 训练：`--species_cfg_drop_prob`（默认 0.15，`train_all.bat` 为 0.2）按样本抽样；被丢弃的样本**整条绕过 FiLM**，关节名嵌入原样进入 `text_embedding`（不是只把 species 输入置零——那样仍会保留一条由关节名驱动的 learned 条件路径）。
- 带参考动作的行：目标物种 FiLM **一律 bypass**，训练和推理都是。参考提供运动内容，不再需要物种风格。
- 推理 `--species_active`：显式开关。`--reference_motion` 为已编码的参考时默认 False，且不能与该参数同时给出。
- 推理 `--species_cfg_scale`（默认 1.0 = 关）：高噪声半程（`t ≥ T/2`）每步额外一次 `species_active=False` 的前向（细节步 scale 固定为 1，不额外前向），按 `uncond + s·(cond − uncond)` 组合。这里的"无条件"**只移除显式 species 调制**，关节名、骨架、结构通道、action label 等全部保留。
  - 前向次数：只开 species CFG 为 2 次/步；与 action CFG 同时开为 4 次/步；
  - 要求 checkpoint 用 `--species_cond` 训练且 `species_cfg_drop_prob > 0`；带参考动作时 scale 必须为 1。

注意：per-joint FiLM 是 species 的**唯一**通道，bypass 即完全无 species。无条件分支里关节名、拓扑、rest pose 仍在，所以"无条件"不是"不知道是什么骨架"，只是"不知道步态风格"。

### 4.3 species 通路的作用边界

关节名、拓扑边关系、rest pose、骨长已经让模型看到相当完整的骨架信息；body plan 在很大程度上是这些信息的**有损摘要**，因此 species 的作用是**风格偏置**（步态节奏、能量、跨关节协调），而不是骨架识别。描述符碰撞（同描述符的物种拿到同一个向量）进一步限制了它能区分的程度——但 FiLM 因关节名不同，仍能区分共用同一描述符的两个物种，加性做法做不到。评估 species 效果时应挑**描述符互不相同**的物种做对照。

variant 在数据里是按物种打的，与骨架高度相关（例如所有 Fluttering 物种都是昆虫）。模型可以从 tag 学到它，也可以从骨架学到它，二者预测力相同；species CFG 放大的是 tag 那一部分。

---

## 5. 结构通道

关节名只应当携带"身体部位"语义。"这个关节在骨架中的位置"由**纯几何 / 拓扑**的结构特征给出，只读 `parents` 与物理 rest 位置，**不读任何关节名、不读触地信号**（触地是启发式猜测，把它作为条件等于把它的错误交给模型）。因此重命名骨架不会改变它。

[joint_struct_features.py](../data_loaders/truebones/truebones_utils/joint_struct_features.py) 的 `JOINT_STRUCT_FEATURE_NAMES` 给出每个关节的特征：沿无分叉链的位置（`depth_norm` / `run_len_inv`）、是否叶子、归一化高度、带符号的横向偏移（左 ≠ 右）、前后偏移、该链起点的高度（肢体挂在哪）、子树规模、同级排序与同级数量。

- 训练与生成**必须**调用同一个 `build_joint_struct_features`，从 cond 条目现场计算，不写进 `cond.npy`，否则两边会对同一骨架给出不同条件。
- 经 `struct_embedding`（MLP）投影后加到 token 上；**常规初始化，不是零初始化**——它要从第一步就携带信号。
- padding 行在投影之后重新清零（MLP 有偏置，零输入不再是零输出）。
- 名字 dropout 的概率**不作用于**它：它的存在意义就是在名字缺失时仍然保留关节身份。
- `JOINT_STRUCT_FEATURE_SCHEMA_VERSION` 记录在 `args.json`，不匹配拒绝加载。

---

## 6. 一个关节 token 的完整装配

训练一步里，对每个 `(样本, 关节)`：

1. 取 `joints_names_embs` 行；按 `joint_name_drop_prob` 决定整行是否清零。
2. 复制一份作 FiLM 输入（`joints_clean`）；另一份做逐元素 dropout。
3. 若 `species_cond`：用 `concat[joints_clean, species_code]` 算 gamma / beta，调制 dropout 后的名字；若该样本 `species_active = False`（CFG 丢弃或带参考），则取未调制的名字。
4. `text_embedding` 把名字向量投到 `latent_dim`，加到该关节的运动 token 上。
5. 加结构通道（padding 行清零后）与帧位置编码。

padding 行与"无名字"行同为零向量；是否为真实关节由 attention mask 对角线给出，**不能**靠名字是否全零判断（语料中有真实关节被刻意置空，如定位器、道具挂点）。

---

## 7. 常见问题与取舍

- **为什么 species 不进 timestep 条件？** 全局条件与逐关节条件的输入语义不同：timestep 只能拿到纯物种向量，逐关节需要 `species × joint` 交互；一个头也无法对两个对象做独立的 drop 决策。species 只走逐关节 FiLM，timestep 条件只承载时间、速度和动作。
- **为什么名字置空优于猜一个名字？** 置空是训练过的状态；猜一个会落在 T5 空间里没有先验的位置。
- **为什么 FiLM 零初始化？** 起点等于"没有 species"，不会在训练早期把随机扰动注入关节名；代价是它要从头学出所需的调制，早期曲线不能用来判断它有没有用。
- **用 `--species_tags` 的外推风险**：它驱动的是非线性 MLP 的乘性 gamma，直接缩放关节名语义；描述符表限定了只能用训练见过的离散组合，这是该风险被约束住的原因。但 body plan × variant 的组合在训练里不一定出现过（如 Winged + 鸟类骨架 + Fluttering），这种用法本身就是外推。

---

## 8. 改动后需要做什么

| 改动 | 需要 |
|---|---|
| 改 `species_tags.jsonl` | 重新烘焙 `cond.npy`（`tools/regenerate_dataset_artifacts.py`）；描述符表会在下次训练启动时自动扩充；已有 checkpoint 的行为随之改变，需重训才能受益 |
| 新增 body-plan | 重建描述符表并重训（表是封闭词表，词表外没有向量） |
| 改关节名文本规则 / T5 型号 | 重新烘焙 `joints_names_embs`（`JOINT_NAME_EMBEDDING_SCHEMA_VERSION` 变）并重训 |
| 改结构特征 | `JOINT_STRUCT_FEATURE_SCHEMA_VERSION` 变，重训，不需重新烘焙 cond |
| 改 FiLM 结构 / bottleneck / 注入位置、variant 编码 | 重训（`CKPT_VERSION` 覆盖会改变语义的改动） |
| 只改 drop 概率、`--species_cfg_scale` | drop 概率需新 run；`--species_cfg_scale` 仅推理，无需重训 |

---

## 9. 关键代码位置

| 内容 | 位置 |
|---|---|
| 输入装配、FiLM、整关节 drop、species 激活逻辑 | [`model/anytop.py`](../model/anytop.py)（`InputProcess`、`_resolve_species_active`、`_sample_joint_name_drop`） |
| 描述符词表、`species_tags.jsonl` 读取与校验 | [`dataset_tags.py`](../data_loaders/truebones/truebones_utils/dataset_tags.py) |
| 描述符表（构建、绑定、查表）、species code | [`species_descriptor_table.py`](../data_loaders/truebones/truebones_utils/species_descriptor_table.py) |
| 关节名文本生成与 T5 烘焙、未见名字置空 | [`joint_embedding_text.py`](../data_loaders/truebones/truebones_utils/joint_embedding_text.py)、[`joint_name_canonical.py`](../data_loaders/truebones/truebones_utils/joint_name_canonical.py) |
| 结构特征 | [`joint_struct_features.py`](../data_loaders/truebones/truebones_utils/joint_struct_features.py) |
| `--species_tags` 覆盖、species CFG 包装 | [`sample/conditioning.py`](../sample/conditioning.py) |
| 新骨架的名字与描述符处理 | [`new_dataset_joint_name_checklist.md`](new_dataset_joint_name_checklist.md) |
| 训练配置示例 | [`train_all.bat`](../train_all.bat) |
