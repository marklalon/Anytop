# 新数据集接入备忘录：骨骼名的检查与词表更新

> 适用场景：往 `datasets.jsonl` 里加一个新数据集，里面有一批**没见过的骨骼命名习惯**。
> 本文只管**骨骼名**这一条线；数据集注册、cond 合并、多源训练见 [multi_dataset_training.md](multi_dataset_training.md)。
> 旧文 [t5_conditioner_joint_name_preprocessing.md](t5_conditioner_joint_name_preprocessing.md) 描述的是已被取代的
> `T5Conditioner.tokenize()` 实现，只作历史背景读，**不要照着改**。
>
> 本文只给常量名和函数名，不给行号——代码在动，行号会过期。搜索名字即可定位；
> 除非特别说明，A/B/C 三层的常量都在 `data_loaders/truebones/truebones_utils/physics_joint_annotation.py` 里。

---

## 0. 先建立心智模型：三层名字，模型只看第三层

```
raw_name              "Bip01_L_ForeArm" / "LfLegAnkle" / "GorillaJaw"
   │  _canonicalize_joint_name()  + _disambiguate_duplicate_canonical_names()
   ▼
canonical_joint_names "Left Forearm" / "Lf Leg Ankle" / "Gorilla Jaw"     ← 导出 BVH 骨名、renamer 词表
   │  _refine_joint_embedding_name() + build_joint_embedding_texts()
   ▼
embedding_text        "Left Forearm"  / "Left Front Leg Ankle Contact"    ← T5 编码，模型唯一看到的东西
```

三条推论，接新数据集时反复用得上：

1. **不存在 OOV。** T5 是开放词表，再古怪的骨骼名都能编码。所以"新数据集有一堆没见过的名字"本身**不是**问题，
   问题只会是「编码到了错误的邻域」。
2. **改第三层最安全。** 只影响 `joints_names_embs`，不动 BVH 骨名、不动 renamer bank、不动导出文件。
   除非你确实想改导出的骨名，否则**默认改 refine 层，不要改 `_canonicalize_joint_name`**。
3. **第二层有个反直觉的坑**：`_disambiguate_duplicate_canonical_names` 为了保证名字唯一，
   会把原始名里的区分 token **重新贴回去**。所以在第二层剥离一个词（比如物种名），
   往往剥不掉——它会以后缀形式回来，你只是白白付出了 BVH 名变更 + renamer bank 重建的代价。

还有一条**第四层**，容易被忘：对称配对用的是 `_joint_signature`，它读的是**第二层的拼写**，
不经过第三层的任何同义词/缩写映射。所以在 B 层把两个拼法折叠到一起，**不会**让它们配成左右对；
拼写腐蚀导致的配对失败要单独在 `_SIGNATURE_SPELLING_TOKENS` 里修。

---

## 1. 落地流程（按顺序做）

### 1.1 前置：sidecar 必须先写

```bash
# species_tags.jsonl 没有 fallback，缺一个物种就 SystemExit
# {"species": "Cat", "species_tags": ["Quadruped", "Small", "Stalking"]}
$EDITOR dataset/<新数据集>/species_tags.jsonl

# 只有蛇/鱼这类没有可用肢体对的物种才需要
# {"species": "Pirrana", "chain_forward_joints": [10, 3, 4]}
$EDITOR dataset/<新数据集>/chain_forward_joints.jsonl

# action_labels.jsonl 同样没有 fallback（2026-08-30 起，见
# action_group_label_refactor.md §9.5）：预处理阶段宽容（carry-forward 读），
# 但训练 / 产物重建（regenerate_dataset_artifacts.py）时缺文件或缺 clip 条目
# 会直接 fast-fail。每个 clip 一行，group 三选一：
# {"clip": "Cat_Walk_1.npy", "action_group": "locomotion", "action_label": "walk, ..."}
$EDITOR dataset/<新数据集>/action_labels.jsonl
```

`chain_forward_joints` 的下标绑定的是**塌陷后**的骨架顺序，不是原始 GLB 顺序——
参考 `dataset_tags.py` 里该常量的说明，以及踩过的坑：Pirrana 曾因为用了塌陷前的下标而整体朝向反了。

### 1.2 跑预处理

```bash
cd Anytop
python preprocess_and_validate.py --filter "<新物种名>"
```

只想刷新骨骼名相关的元数据（不重导出动作，秒级）：

```bash
python preprocess_and_validate.py --re-encode-joint-names-only
```

### 1.3 主检查点：逐物种读 `joint_name_inspection/<species>.json`

这是**唯一一个把三层名字并排摊开**的产物，新数据集每个物种都应该人工扫一遍：

```json
{
  "index": 2,  "raw_name": "NPC_LLeg1",
  "canonical_name": "Left Leg 1",  "canonical_bvh_name": "LeftLeg1",
  "embedding_text": "Left Leg Segment First Of 2 ChainStart",
  "is_anatomical": true, "side": "left", "is_contact": false, "is_end_effector": false
}
```

看四件事：`embedding_text` 是不是人话；`side` 有没有该左右却是 `center`；
`is_contact` / `is_end_effector` 在末端关节上有没有点亮；`is_anatomical` 有没有误杀真解剖。

顺手做一次**新鲜度对拍**：用当前代码 live 重算一遍 `build_joint_embedding_texts`，和 JSON 里的
`embedding_text` 逐条比对。全等说明产物和代码同步；不等说明有人改了词表却没重跑（或忘了 bump）。

### 1.4 词表 diff 审计（最高性价比的一步）

把模型实际看到的每个词列出来，**和既有数据集的词表做差集**，只看新数据集独有的词。
按「出现次数从少到多」截断长尾是不够的：系统性的整类问题（一整套没识别的肢位码、一个拼错的
词被 13 个物种共用）出现次数并不低，长尾截断会把它们全部漏掉，而它们恰恰是危害最大的。

自足脚本，直接存文件跑（**live 重算**，不必等 cond 重新生成）：

```python
# audit_joint_name_tokens.py  —— 在 Anytop/ 下运行
# 用法: python audit_joint_name_tokens.py <新数据集cond> [<已有数据集cond> ...]
import sys, copy, re
from collections import Counter, defaultdict
sys.path.insert(0, '.')
from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.animation_utils import refresh_joint_metadata_in_object_cond
from data_loaders.truebones.truebones_utils.physics_joint_annotation import build_joint_embedding_texts

DERIVED = {'Left','Right','Segment','Of','Instance','Contact','EndEffector','ChainStart','ChainEnd',
           'ChainEarly','ChainMiddle','ChainLate','First','Second','Third','Fourth','Fifth','Sixth',
           'Seventh','Eighth','Ninth','Tenth','HeadFeature'}
INDEX_RE = re.compile(r'^Index\d+$')

def vocab(path):
    uses, species, examples, blank = Counter(), defaultdict(set), defaultdict(set), []
    for key, entry in load_cond(path).items():
        sp = key.split('/')[-1]
        names = [str(n) for n in entry['canonical_bvh_joint_names']]
        oc = copy.deepcopy(dict(entry)); refresh_joint_metadata_in_object_cond(oc)
        for i, text in enumerate(build_joint_embedding_texts(oc)):
            if not text:
                blank.append(f'{sp}:{names[i]}'); continue
            for token in text.split():
                if token in DERIVED or token.isdigit() or INDEX_RE.match(token): continue
                uses[token] += 1; species[token].add(sp); examples[token].add(f'{sp}:{names[i]}')
    return uses, species, examples, blank

new_uses, new_species, new_examples, blank = vocab(sys.argv[1])
old = Counter()
for path in sys.argv[2:]:
    old.update(vocab(path)[0])
novel = {t: n for t, n in new_uses.items() if t not in old}
print(f'{len(new_uses)} tokens, {len(novel)} NOVEL vs the existing corpus, {len(blank)} blanked joints\n')
for token, n in sorted(novel.items(), key=lambda kv: -kv[1]):
    print(f'  {n:4d} uses /{len(new_species[token]):3d} species  {token:16s} {sorted(new_examples[token])[:2]}')
print('\n--- blanked (zero embedding) ---')
for entry in blank[:60]:
    print(f'  {entry}')
```

判读只有四种结论：**真解剖（不动）/ 同义词、缩写或拼写错误（→ B6）/ rig 垃圾（→ B4）/ 证据不足（不动，写进注释）**。

一次**脏**的审计长这样（治理前的真实输出，现已全部处理）：

```
   26 uses / 13 species  Reg        ['MLH_Archer:UpperRegRight']    → "Leg" 被全局 L→R 替换改坏 → B6+B7
   20 uses / 10 species  Rower      ['MLH_Archer:RowerArmRight']    → 同上，"Lower"                → B6+B7
   73 uses / 15 species  Container  ['TNR_Cavalry:LeftHandContainer'] → 装备挂点                   → B4
   11 uses /  3 species  Fl         ['PC_PolygonalWolf:FlLeg1']     → 肢位码没识别                 → B5+C1
    1 uses /  1 species  Ponitail   ['Comodoa:Ponitail']            → ponytail 拼错，漏掉屏蔽      → B4
    2 uses /  1 species  Tounge     ['Kappa_gorilla:KappaTounge01'] → Tongue 拼错                  → B6
```

一次**干净**的审计长这样（同样是真实输出，治理后。剩下的全是该留的）：

```
   1  Shell      ['HermitCrab:Shell']                  → 真解剖（寄居蟹壳）        → 不动
   2  Gill       ['Pirrana:GillLeft']                  → 真解剖                    → 不动
   2  Crest      ['Boar:SpineCrest0101']               → 查树后否决：是背脊不是颈脊 → 不动
   2  Stomach    ['Alligator:Stomach']                 → 查树后否决：三个 rig 三种含义 → 不动
   2  Ant        ['spider_tarantula:LeftAnt00']        → 查树后否决：是触角不是蚂蚁 → 不动
  10  Ctr        ['spiderling:LeftLegCtr1']            → 第四对腿的位置码，和同 rig
                                                          的 LegMid 并存，映射成
                                                          center 反而会被 B4 吃掉  → 不动
   7  Ne/Nw/Se/Sw ['FlowerPot:PetalNe1']               → 花瓣的罗盘方位码，不是解剖 → 不动
   1  Belleh     ['Deer_Buck:Belleh']                  → 证据不足（belly？）        → 不动
```

**"证据不足就不动"是硬规矩**：留一个无意义 token 的代价，远小于把一个错误解剖词塞进 T5 空间。

### 1.5 唯一性与家族分布

```bash
# canonical 名撞车报告（预处理会自动写并在控制台告警）
cat dataset/<新数据集>/joint_name_collision_report.json

# 解剖家族分布：看「家族数 / 关节数」的比例，和现有数据集同量级即可；
# 比例暴涨说明有一整类名字没被归并
python tools/family_keys_stats.py           # 在仓库根运行
```

### 1.6 对称与朝向校验

先看**没有任何对称对**的物种：真无肢体的（蠕虫、火焰、球体）是正常的，
而一个四足兽出现在这张表里，说明它的左右写法没被 `detect_joint_side` 认出来（→ C1）。
一个自足的判据：把每个 `center` 关节的名字里的 `l`/`r` 字符逐位翻转，
如果翻转后的名字在同一副骨架里存在，那它几乎肯定是一对左右。

```bash
python utils/validate_anytop_dataset.py --datasets dataset/datasets.jsonl
```

朝向出问题时的两个已知信号：控制台出现
`no named left-right joint pairs found; estimated the lateral axis from rest-pose mirror symmetry`
（说明左右对没认出来 → 查 C1），或者 T-pose 朝向偏离最近轴超过阈值。

⚠️ 如果数据集的 `ignore_warnings.txt` 里有 `!skip-orientation-detection`，
**朝向那一路检查整体不生效**：rest-pose 的朝向修正走恒等，验证器也跳过 recovered-facing 比对。
此时 `resolve_face_joints` 选错关节（挑中披风、头发、武器）不会报任何错，
但它选的关节仍然写进 cond——哪天去掉这个标志就会立刻算错朝向。新数据集带这个标志时，
要单独扫一眼每个物种的 `face_joint_names` 是不是肢体/肩胯类的名字。

---

## 2. 六个红旗 → 该改哪张表

| 症状 | 怎么发现 | 改哪里 |
|---|---|---|
| 某关节的 `side` 是 `center`，但名字/几何明显有左右 | inspection JSON；§1.6 的翻转判据 | `detect_joint_side` §3-C1 |
| `embedding_text` 里出现物种名（`Gorilla Jaw`） | §1.4 审计里冒出物种词 | `_EMBED_TEXT_CREATURE_TOKENS` §3-B3（**但先读 §5 的例外**） |
| 同一块骨头在不同物种拼法不同（`Ulna` vs `Forearm`） | §1.4 的 NOVEL 列表 + §1.5 家族比例暴涨 | `_EMBED_TEXT_SYNONYM_TOKENS` §3-B6 |
| 道具/控制骨混进来（`Saddle`/`Ctrl`/`Bone02`/`Shield`） | §1.4 审计；inspection JSON 的 `is_anatomical` | `_EMBED_TEXT_NON_ANATOMICAL_TOKENS` §3-B4 |
| 全小写粘连名整块变成一个 OOV token（`smallfrontarm`） | §1.4 审计里出现长怪词 | `_COMPOUND_*_TOKENS` §3-A5 |
| 左右两侧拼法不同导致配不成对（`Lower_Arm_L` / `Rower_Arm_R`） | §1.6 的对称对清单少了整条肢 | `_SIGNATURE_SPELLING_TOKENS` §3-C2 |
| T-pose 朝向反了/侧躺 | 验证器告警；`resolve_face_joints` 选错关节 | `_FACE_JOINT_*` / `_FORWARD_*` §3-C3 |

---

## 3. 词表清单（按"改了会波及什么"分组）

### A 层 · 名字规范化 —— 改这里会动 BVH 骨名、导出文件、renamer 词表

| # | 常量 | 作用 |
|---|---|---|
| A1 | `_CANONICAL_NAME_PREFIXES` | 剥掉 `Bip01`/`NPC`/`BN`/`Rig` 这类 rig 前缀。新数据集有自己的前缀就加这里 |
| A2 | `_CANONICAL_NAME_SUFFIXES` | 同上，尾缀（`SHJnt`） |
| A3 | `_JAPANESE_NAME_REPLACEMENTS` | 罗马音 → 英文（`momo`→Thigh）。**全局生效**，只放绝不歧义的词 |
| A4 | `_JAPANESE_GATED_REPLACEMENTS` + `_JAPANESE_EVIDENCE_TOKENS` | 危险的短词（`o`=尾、`te`=手），仅在骨架被判定为日式命名时才启用。判定门槛是 3 个不同 evidence token |
| A5 | `_COMPOUND_MODIFIER_TOKENS` / `_COMPOUND_ANATOMY_TOKENS` / `_COMPOUND_SPLIT_PROTECTED_TOKENS` | 全小写粘连名的 DP 切分词表。加新解剖词进 ANATOMY；**凡是能被切开但不该切的真词必须进 PROTECTED**（`eyebrow` 会被切成 `eye`+`brow`）。注意有长度下限，`lwing` 这类 5 字符的粘连名根本进不了切分 |

### B 层 · Embedding text —— 只影响模型输入，**新数据集的首选改动层**

| # | 常量 | 作用 |
|---|---|---|
| B1 | `_EMBED_TEXT_SKIP_TOKENS` | 链位置废词（`base`/`tip`/`end`）。**注意注释里的警告：`front`/`back`/`rear`/`mid` 故意不在这里**，它们是前肢与后肢唯一的区分 |
| B2 | `_EMBED_TEXT_SIDE_TOKENS` | 名字自带的左右词，一律丢弃，侧别统一由几何标签重新贴到句首 |
| B3 | `_EMBED_TEXT_CREATURE_TOKENS` | 物种名。新物种名 + 变体 rig 里出现的其他生物名要加，**但坐骑型 rig 是例外，见 §5** |
| B4 | `_EMBED_TEXT_NON_ANATOMICAL_TOKENS` | rig 脚手架 / 道具 / 马具 / 武器 / 护具 / 挂点。整条名字只剩这些词时该关节被**置零 embedding** |
| B5 | `_EMBED_TEXT_LIMB_CODE_TOKENS` / `_EMBED_TEXT_QUADRANT_LIMB_CODE_TOKENS` | 四足肢位码 `lf/rf/lb/rb` → `Front`/`Back`（左右交给几何）；后者是**halves 互换**的写法 `fl/fr/bl/br` 以及六足中腿 `lm/rm`，因为拼法有歧义（`MouthBL` 是嘴的左下角），只在同名里还有 `arm`/`leg` 时才解码 |
| B6 | `_EMBED_TEXT_SYNONYM_TOKENS` | 解剖同义词 + rig 缩写 + 拼写错误，全部折叠到语料已有的词。**长尾治理的主力表** |
| B7 | `_EMBED_TEXT_TOKEN_PAIR_MERGES` | 相邻两词合成一词（`upper leg`→Thigh、`horse link`→Ankle）。单词映射解决不了时用这个。**注意执行顺序：pair merge 跑在 B6 单词映射之前**，所以被拼错的词要么两条都写（`('rower','reg')` 和 `rower`/`reg` 各自），要么就落不到 merge 上 |
| B8 | `_EMBED_TEXT_HEAD_FEATURE_TOKENS` | 头部附属物额外追加一个 `HeadFeature` 类别词，让它们在 T5 空间里彼此靠近 |

### C 层 · 元数据推断 —— **改错了不会报错，只会静默变差**

| # | 位置 | 作用 |
|---|---|---|
| C1 | `detect_joint_side` 的 marker 元组 | 左右识别。新数据集用了新的侧别写法（`_L_`/`Lft`/`L01`/`Lwing`…）必须加。显式 `Left`/`Right` 优先于肢位码；歧义的肢位码要跟 B5 一样加 `arm`/`leg` 门控 |
| C2 | `_joint_signature` / `_signature_tokens` / `_LIMB_CODE_SIGNATURE_TOKENS` / `_SIGNATURE_SPELLING_TOKENS` | 对称配对签名。剥掉侧别、**但保留前后半码**——否则前肢会和后肢配成一对。签名是**拼写键**，不吃 B6 同义词：左右拼法被改坏（`Lower`/`Rower`）时在 `_SIGNATURE_SPELLING_TOKENS` 里做纯拼写修复，不要把整张同义词表塞进来（会重排所有现有 rig 的分组） |
| C3 | `_FACE_JOINT_*` / `_FORWARD_REFERENCE_PRIORITIES` / `_BODY_AXIS_*`（在 `face_orientation.py`） | 朝向解算挑哪些关节。新物种的髋/肩/鼻子叫了别的名字，朝向就会算错；道具骨（披风/头发/武器）要进 exclude |
| C4 | `_CONTACT_JOINT_*` / `_CONTACT_CHAIN_*` | 触地关节判定（脚/爪/掌） |
| C5 | `_END_EFFECTOR_*` | 末端执行器判定 |

### D 层 · 度量 —— 不影响模型，但影响 renamer 的评测与 S6 prior

文件：`skeleton_renamer/sr_common.py`

| # | 常量 | 作用 |
|---|---|---|
| D1 | `_FAMILY_CREATURE_TOKENS` | 与 B3 **手工保持同步**（该模块刻意只依赖 numpy，不能 import Anytop） |
| D2 | `_FAMILY_LIMB_CODE_TOKENS` / `_FAMILY_QUADRANT_LIMB_CODE_TOKENS` | 与 B5 同步（含门控）：肢位码 → `Front`/`Back`/`Mid`，**映射而不是删除** |
| D3 | `_FAMILY_POSITION_TOKENS` | 方位词归一（`hind`/`rear`/`back` → `Back`）并提到 key 最前，让码形 rig 与词形 rig 落到同一个 key。**`fore` 和 `mid` 故意不在这里**：`ForeArm` 是前臂不是前肢的臂；`LegMid` 在蜘蛛是「第二对腿」、在 serpent_man 是「腿的中段（小腿）」，同一拼法两个意思，`family_key` 只看得到名字，分不开。代价是「码形/词形落同一 key」这条只对 Front/Back 成立（`LmLeg1`→MidLeg 而 `LegMid1`→Leg），这个方向是安全的一侧——欠分裂看得见（一个过大的通用桶），错误合并看不见 |
| D4 | `_FAMILY_QUALIFIER_TOKENS` | 装饰限定词（`jiggle`/`twist`/`low`）。方位词归 D3 管，不放这里 |
| D5 | `_TERMINAL_NAME_RE` | `Nub`/`End` 末端标记正则。新数据集用别的末端后缀就得加。判定末端一律用 `terminal_mask`（名字后缀 **且** 是叶子），不要用裸的 `is_terminal_name` |

---

## 4. 改词表的七条规矩（都是踩出来的）

1. **先查树，再映射。** 任何一条同义词都要先看它在骨架里的**实际父子链**，不要查字典。
   - `HorseLink` 不是马专用骨：33 个物种（猫、狮子、鸡）都有，恒为 `Thigh→Calf→HorseLink→Foot`，即踝关节。按物种名剥掉会得到 `Link`，更糟。
   - `Ant00`（spider_tarantula）挂在 `Head01` 下，是**触角**缩写不是蚂蚁。
   - `Spline01..06`（Anaconda）是 `Hips→…→Neck`，是**真脊椎**，不是 IK 控制器。
   - `Rower`/`Reg` 不是新词：同一副骨架里左边写 `Lower_Arm_L`、右边写 `Rower_Arm_R`，
     是作者镜像后对复制出来的名字做了全局 L→R 替换，把单词本身也改坏了。查树确认
     `UpperArm→RowerArm→Hand`、`Hips→UpperReg→RowerReg→Foot` 才敢映射。
   - 反面：`ball`（Bear 同时用于脚掌和手掌）、`belly`/`stomach`（三个 rig 三种含义）、`crest`（Boar 的是**背**脊不是颈脊）、
     `Crown`（一个 rig 挂在 Head 下、另一个挂在 UpperBody 下）——查完树之后全部放弃映射。

2. **剥离必须有空回退。** 剥完不能让名字变空。现有实现靠 `_refine_joint_embedding_name` 结尾的
   `deduped_tokens or canonical_name.split()` 兜底：一个真叫 `Dragon` 的关节会保留 `Dragon`。

3. **置零只在"整条名字都是 marker"时才触发，所以别让方位词落单。** 往 B4 加词之前先看这个词在
   语料里是怎么被修饰的：`CapeBack01`/`FrontSkirt` 把 `cape`/`skirt` 拿掉之后只剩一个 `Back`/`Front`，
   那就成了"这是生物的背/前面"——比留着 `Cape Back` 更糟。这类布料附件按头发的先例处理：**不进 B4**。
   硬装备（武器、盾、护甲、背包、挂点）没有这个问题，照常置零。

4. **不要塌掉前后肢，也不要为了统一它而误伤 `fore`。** 前者是写进注释的历史 bug（Crocodile 前后腿
   曾撞成同一句），后者是 `family_key` 真踩过的坑：把方位词提到 key 最前时，`ForeArm`（前臂，
   radius/ulna）会被并进 `FrontArm`（前肢的臂）。源模块用 `('fore','arm') → 'Forearm'` 挡它，
   D3 用「`fore` 不列入方位词」挡它。涉及 `front/back/rear/hind/fore` 和肢位码的任何
   改动，都要**逐条打印被移动的 key**再决定，别只看总数。

5. **短码要门控，不要全局。** `fl/fr/bl/br/lm/rm` 这类两字母码在别的 rig 里可能是别的意思
   （`MouthBL` = 嘴的左下角）。加进 B5 和 `detect_joint_side` 时都要求同名里还有 `arm`/`leg`，
   门控条件两边必须一致，否则 embedding text 说它是后腿、side 说它是 center。

6. **默认改 B 层。** 只有当你确实要改导出的 BVH 骨名时才动 A 层，并且要记得 A 层的剥离会被
   `_disambiguate_duplicate_canonical_names` 部分抵消（见 §0-3）。

7. **加了词表要补测试。** 参考 `tests/test_joint_embedding_texts.py`、`tests/test_symmetry_metadata.py`。
   改完至少跑一遍**三个数据集的全量对拍**（live 重算 vs `joint_name_inspection/` 里存的旧值），
   把「新增置零」「文本改写」「side 变化」「对称对增减」四个数字都打出来：
   置零列表里出现任何解剖词，或者改写后的文本只剩方位词，就是回归。

---

## 5. 已知缺口（下一个数据集很可能踩到）

- **`face_orientation.py` 有自己的一份 `_canonicalize_joint_name` 和 `_joint_signature`。**
  两份实现并存且行为故意不同——那一份丢弃单字符 token 含孤立数字。
  改任何一份都要同步检查另一份（`_LIMB_CODE_SIGNATURE_TOKENS` 在两边各有一份，必须一起改）。
- **`_EMBED_TEXT_CREATURE_TOKENS` 是手工列表，不是从 `species_tags.jsonl` 派生的。**
  加新物种时**必须手工补**，没有任何机制会提醒你。
  （没做成自动派生，是因为变体 rig 里会出现数据集中并不存在的生物名，比如 antilope 的 Quilin/Moose。）
- **坐骑型 rig 是 B3 的反例，别照着表格加。** `MLH_Horseman`/`MLS_Dryad` 是**一副骨架上两个生物**：
  `horse_*` 是坐骑、`man_*` 是骑手。把 `horse`/`man` 加进物种表，会把骑手的手臂和马的前腿
  塌成同一句——和 Kappa 那个案例（同一个生物被反复贴species名）正好相反。这两个词要**明确保留**。
- **缩写表按物种硬编码**（`thi`/`clf`/`nek`/`spn` 来自 SabreToothTiger，`Clav`/`Scap` 来自 Deer_Buck）。
  新 rig 有自己的缩写体系时只能继续往 B6 加。
- **`ant` / `horse` / `jaws` 被刻意排除在物种表之外**（分别撞 antenna、HorseLink、jaw）。
  再加物种名时要检查是否和解剖词撞车。
- **装备挂点会被判成 end effector。** `LeftHandContainer` 这类挂在手下的空节点是叶子，
  `_END_EFFECTOR_*` 认它。B4 置零只去掉名字里的 marker 词，不会取消这个标志。

---

## 6. 收尾：什么时候必须 bump / 重算 / 重训

| 改了什么 | bump `JOINT_NAME_EMBEDDING_SCHEMA_VERSION` | 重跑预处理 | 重训 | renamer bank 重建 |
|---|---|---|---|---|
| B 层任意词表 | ✅ | `--re-encode-joint-names-only` 足够 | ✅ | — |
| A 层任意词表 | ✅ | 全量（BVH 骨名会变） | ✅ | ✅ |
| C 层 side / 对称签名 | ✅ | `--re-encode-joint-names-only` 足够 | ✅ | — |
| C 层 contact / end-effector | ✅ | **全量** | ✅ | — |
| C 层 朝向（face/forward） | ✅ | 全量 | ✅ | — |
| D 层（`family_key`） | — | — | — | ✅（喂 S6 hierarchy prior） |

⚠️ **contact 那一行是唯一不能走增量的**：接触状态是**逐帧烘进 motion `.npy` 特征**的，
而 `--re-encode-joint-names-only`（内部走 `tools/regenerate_dataset_artifacts.py`）只重写 cond、
不重写 motions。改了接触判定却只跑增量，cond 和动作张量会**静默失配**。

版本号常量 `JOINT_NAME_EMBEDDING_SCHEMA_VERSION` 在 `physics_joint_annotation.py`。
忘了 bump 的话不会报错，只会静默用旧 embedding 训练；bump 之后加载旧 cond 会打印
`uses joint-name embedding schema N; current code expects N+1` 告警——
这个告警是**正常的中间状态**，重跑完预处理就消失。
