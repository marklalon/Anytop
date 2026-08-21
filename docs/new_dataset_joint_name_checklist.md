# 新数据集接入备忘录：骨骼名的检查与词表更新

> 适用场景：往 `datasets.jsonl` 里加一个新数据集，里面有一批**没见过的骨骼命名习惯**。
> 本文只管**骨骼名**这一条线；数据集注册、cond 合并、多源训练见 [multi_dataset_training.md](multi_dataset_training.md)。
> 旧文 [t5_conditioner_joint_name_preprocessing.md](t5_conditioner_joint_name_preprocessing.md) 描述的是已被取代的
> `T5Conditioner.tokenize()` 实现，只作历史背景读，**不要照着改**。

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
```

`chain_forward_joints` 的下标绑定的是**塌陷后**的骨架顺序，不是原始 GLB 顺序——
参考 [`dataset_tags.py:69`](../data_loaders/truebones/truebones_utils/dataset_tags.py#L69) 的说明，
以及踩过的坑：Pirrana 曾因为用了塌陷前的下标而整体朝向反了。

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

### 1.4 稀有 token 审计（10 秒，最高性价比）

把模型实际看到的每个词按**从罕见到常见**列出来。凡是你不认识的词，就是一条待处理线索。
下面这段是自足的，直接存成文件跑（**live 重算**，所以不必等 cond 重新生成）：

```python
# audit_joint_name_tokens.py  —— 在 Anytop/ 下运行
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
counts, examples, blank = Counter(), defaultdict(set), []
for path in sys.argv[1:]:
    for species, entry in load_cond(path).items():
        names = [str(n) for n in entry['canonical_bvh_joint_names']]
        oc = copy.deepcopy(dict(entry)); refresh_joint_metadata_in_object_cond(oc)
        for i, text in enumerate(build_joint_embedding_texts(oc)):
            if not text:
                blank.append((species, names[i])); continue
            for token in text.split():
                if token in DERIVED or token.isdigit() or INDEX_RE.match(token): continue
                counts[token] += 1
                examples[token].add(f"{species.split('/')[-1]}:{names[i]}")
print(f"{len(counts)} distinct anatomy tokens, {sum(counts.values())} uses, {len(blank)} blanked joints\n")
for token, n in sorted(counts.items(), key=lambda kv: (kv[1], kv[0])):
    if n > 6: break                      # 只看长尾
    print(f"  {n:4d}  {token:20s} {sorted(examples[token])[:2]}")
print("\n--- blanked (zero embedding) ---")
for species, name in blank[:40]:
    print(f"  {species.split('/')[-1]:18s} {name}")
```

```bash
python audit_joint_name_tokens.py dataset/<新数据集>/cond.npy
```

判读只有四种结论：**真解剖（不动）/ 同义词、缩写或拼写错误（→ B6）/ rig 垃圾（→ B4）/ 证据不足（不动，写进注释）**。

一次**脏**的审计长这样（这些是当前两个数据集在治理前的真实输出，现已全部处理）：

```
   1  Ponitail   ['Comodoa:Ponitail']                  → ponytail 拼错，漏掉屏蔽    → B4
   1  Lftb       ['SabreToothTiger:LeftTwistBoneLftb'] → 纯 rig 后缀码             → B4
   1  Pelv       ['SabreToothTiger:Pelv']              → Pelvis 缩写               → B6
   2  Clav       ['Deer_Buck:LeftClav']                → Clavicle 缩写             → B6
   2  Shin       ['Hyena:LeftShin']                    → Calf 同义词               → B6
   2  Tounge     ['Kappa_gorilla:KappaTounge01']       → Tongue 拼错               → B6
```

一次**干净**的审计长这样（同样是真实输出，治理后。剩下的全是该留的）：

```
   1  Shell      ['HermitCrab:Shell']                  → 真解剖（寄居蟹壳）        → 不动
   2  Gill       ['Pirrana:GillLeft']                  → 真解剖                    → 不动
   2  Pectoral   ['Pirrana:PectoralFinLeft']           → 真解剖                    → 不动
   2  Crest      ['Boar:SpineCrest0101']               → 查树后否决：是背脊不是颈脊 → 不动
   2  Stomach    ['Alligator:Stomach']                 → 查树后否决：三个 rig 三种含义 → 不动
   2  Ant        ['spider_tarantula:LeftAnt00']        → 查树后否决：是触角不是蚂蚁 → 不动
   2  Lt         ['SabreToothTiger:LeftToe0Lt00']      → 证据不足：lt/rt 在别的 rig
                                                          极可能是 left/right，映射
                                                          成 Toe 会注入假解剖      → 不动
   1  Belleh     ['Deer_Buck:Belleh']                  → 证据不足（belly？）        → 不动
```

**"证据不足就不动"是硬规矩**：留一个无意义 token 的代价，远小于把一个错误解剖词塞进 T5 空间。

### 1.5 唯一性与家族分布

```bash
# canonical 名撞车报告（预处理会自动写并在控制台告警）
cat dataset/<新数据集>/joint_name_collision_report.json

# 解剖家族分布：长尾家族数应该和现有数据集同量级，暴涨说明有一整类名字没被归并
python tools/family_keys_stats.py           # 在仓库根运行
```

### 1.6 朝向与整体校验

```bash
python utils/validate_anytop_dataset.py --datasets dataset/datasets.jsonl
```

朝向出问题时的两个已知信号：控制台出现
`no named left-right joint pairs found; estimated the lateral axis from rest-pose mirror symmetry`
（说明左右对没认出来 → 查 §3-C），或者 T-pose 朝向偏离最近轴超过阈值。

---

## 2. 六个红旗 → 该改哪张表

| 症状 | 怎么发现 | 改哪里 |
|---|---|---|
| 某关节的 `side` 是 `center`，但名字/几何明显有左右 | inspection JSON；`joint_side_labels` 统计 | `detect_joint_side` §3-C1 |
| `embedding_text` 里出现物种名（`Gorilla Jaw`） | §1.4 审计里冒出物种词 | `_EMBED_TEXT_CREATURE_TOKENS` §3-B3 |
| 同一块骨头在不同物种拼法不同（`Ulna` vs `Forearm`） | §1.4 审计的长尾 + §1.5 家族数暴涨 | `_EMBED_TEXT_SYNONYM_TOKENS` §3-B6 |
| 道具/控制骨混进来（`Saddle`/`Ctrl`/`Bone02`） | §1.4 审计；inspection JSON 的 `is_anatomical` | `_EMBED_TEXT_NON_ANATOMICAL_TOKENS` §3-B4 |
| 全小写粘连名整块变成一个 OOV token（`smallfrontarm`） | §1.4 审计里出现长怪词 | `_COMPOUND_*_TOKENS` §3-A5 |
| T-pose 朝向反了/侧躺 | 验证器告警；`resolve_face_joints` 选错关节 | `_FACE_JOINT_*` / `_FORWARD_*` §3-C3 |

---

## 3. 词表清单（按"改了会波及什么"分组）

### A 层 · 名字规范化 —— 改这里会动 BVH 骨名、导出文件、renamer 词表

文件：[`physics_joint_annotation.py`](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py)

| # | 常量 | 行 | 作用 |
|---|---|---|---|
| A1 | `_CANONICAL_NAME_PREFIXES` | [166](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L166) | 剥掉 `Bip01`/`NPC`/`BN` 这类 rig 前缀。新数据集有自己的前缀就加这里 |
| A2 | `_CANONICAL_NAME_SUFFIXES` | [177](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L177) | 同上，尾缀（`SHJnt`） |
| A3 | `_JAPANESE_NAME_REPLACEMENTS` | [180](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L180) | 罗马音 → 英文（`momo`→Thigh）。**全局生效**，只放绝不歧义的词 |
| A4 | `_JAPANESE_GATED_REPLACEMENTS` + `_JAPANESE_EVIDENCE_TOKENS` | [218](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L218) / [207](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L207) | 危险的短词（`o`=尾、`te`=手），仅在骨架被判定为日式命名时才启用。判定门槛是 3 个不同 evidence token |
| A5 | `_COMPOUND_MODIFIER_TOKENS` / `_COMPOUND_ANATOMY_TOKENS` / `_COMPOUND_SPLIT_PROTECTED_TOKENS` | [458](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L458) / [426](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L463) / [436](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L473) | 全小写粘连名的 DP 切分词表。加新解剖词进 ANATOMY；**凡是能被切开但不该切的真词必须进 PROTECTED**（`eyebrow` 会被切成 `eye`+`brow`） |

### B 层 · Embedding text —— 只影响模型输入，**新数据集的首选改动层**

| # | 常量 | 行 | 作用 |
|---|---|---|---|
| B1 | `_EMBED_TEXT_SKIP_TOKENS` | [230](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L230) | 链位置废词（`base`/`tip`/`end`）。**注意注释里的警告：`front`/`back`/`rear`/`mid` 故意不在这里**，它们是前肢与后肢唯一的区分 |
| B2 | `_EMBED_TEXT_SIDE_TOKENS` | [314](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L314) | 名字自带的左右词，一律丢弃，侧别统一由几何标签重新贴到句首 |
| B3 | `_EMBED_TEXT_CREATURE_TOKENS` | [348](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L348) | 物种名。**新数据集必查**：新物种名 + 变体 rig 里出现的其他生物名都要加 |
| B4 | `_EMBED_TEXT_NON_ANATOMICAL_TOKENS` | [260](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L260) | rig 脚手架 / 道具 / 马具。整条名字只剩这些词时该关节被**置零 embedding** |
| B5 | `_EMBED_TEXT_LIMB_CODE_TOKENS` | [365](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L365) | 四足肢位码 `lf/rf/lb/rb` → `Front`/`Back`（左右交给几何） |
| B6 | `_EMBED_TEXT_SYNONYM_TOKENS` | [385](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L385) | 解剖同义词 + rig 缩写 + 拼写错误，全部折叠到语料已有的词。**长尾治理的主力表** |
| B7 | `_EMBED_TEXT_TOKEN_PAIR_MERGES` | [711](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L711) | 相邻两词合成一词（`upper leg`→Thigh、`horse link`→Ankle）。单词映射解决不了时用这个 |
| B8 | `_EMBED_TEXT_HEAD_FEATURE_TOKENS` | [318](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L318) | 头部附属物额外追加一个 `HeadFeature` 类别词，让它们在 T5 空间里彼此靠近 |

### C 层 · 元数据推断 —— **改错了不会报错，只会静默变差**

| # | 位置 | 行 | 作用 |
|---|---|---|---|
| C1 | `detect_joint_side` 的 marker 元组 | [1393](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L1393) | 左右识别。新数据集用了新的侧别写法（`_L_`/`Lft`/`L01`…）必须加。显式 `Left`/`Right` 优先于肢位码 |
| C2 | `_joint_signature` / `_LIMB_CODE_SIGNATURE_TOKENS` | [1045](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L1045) / [1033](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L1033) | 对称配对签名。剥掉侧别、**但保留前后半码**——否则前肢会和后肢配成一对 |
| C3 | `_FACE_JOINT_*` / `_FORWARD_REFERENCE_PRIORITIES` / `_BODY_AXIS_*` | [face_orientation.py:34-73](../data_loaders/truebones/truebones_utils/face_orientation.py#L34) | 朝向解算挑哪些关节。新物种的髋/肩/鼻子叫了别的名字，朝向就会算错 |
| C4 | `_CONTACT_JOINT_*` / `_CONTACT_CHAIN_*` | [86](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L86) / [125](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L125) / [143](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L143) | 触地关节判定（脚/爪/掌） |
| C5 | `_END_EFFECTOR_*` | [12](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L12) / [53](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L53) | 末端执行器判定 |

### D 层 · 度量 —— 不影响模型，但影响 renamer 的评测与 S6 prior

文件：[`skeleton_renamer/sr_common.py`](../../skeleton_renamer/sr_common.py)

| # | 常量 | 行 | 作用 |
|---|---|---|---|
| D1 | `_FAMILY_CREATURE_TOKENS` | [876](../../skeleton_renamer/sr_common.py#L876) | 与 B3 **手工保持同步**（该模块刻意只依赖 numpy，不能 import Anytop） |
| D2 | `_FAMILY_LIMB_CODE_TOKENS` | [892](../../skeleton_renamer/sr_common.py#L892) | 与 B5 同步：`lf/rf/lb/rb` → `Front`/`Back`，**映射而不是删除** |
| D3 | `_FAMILY_POSITION_TOKENS` | [901](../../skeleton_renamer/sr_common.py#L901) | 方位词归一（`hind`/`rear`/`back` → `Back`）并提到 key 最前，让码形 rig 与词形 rig 落到同一个 key。**`fore` 故意不在这里**：`ForeArm` 是前臂不是前肢的臂 |
| D4 | `_FAMILY_QUALIFIER_TOKENS` | [903](../../skeleton_renamer/sr_common.py#L903) | 装饰限定词（`jiggle`/`twist`/`low`）。方位词归 D3 管，不放这里 |
| D5 | `_TERMINAL_NAME_RE` | [705](../../skeleton_renamer/sr_common.py#L705) | `Nub`/`End` 末端标记正则。新数据集用别的末端后缀就得加。判定末端一律用 `terminal_mask`（名字后缀 **且** 是叶子），不要用裸的 `is_terminal_name` |

---

## 4. 改词表的六条规矩（都是踩出来的）

1. **先查树，再映射。** 任何一条同义词都要先看它在骨架里的**实际父子链**，不要查字典。
   - `HorseLink` 不是马专用骨：33 个物种（猫、狮子、鸡）都有，恒为 `Thigh→Calf→HorseLink→Foot`，即踝关节。按物种名剥掉会得到 `Link`，更糟。
   - `Ant00`（spider_tarantula）挂在 `Head01` 下，是**触角**缩写不是蚂蚁。
   - `Spline01..06`（Anaconda）是 `Hips→…→Neck`，是**真脊椎**，不是 IK 控制器。
   - 反面：`ball`（Bear 同时用于脚掌和手掌）、`belly`/`stomach`（三个 rig 三种含义）、`crest`（Boar 的是**背**脊不是颈脊）——查完树之后全部放弃映射。

2. **剥离必须有空回退。** 剥完不能让名字变空。现有实现靠 `_refine_joint_embedding_name` 结尾的
   `deduped_tokens or canonical_name.split()` 兜底：一个真叫 `Dragon` 的关节会保留 `Dragon`。

3. **不要塌掉前后肢，也不要为了统一它而误伤 `fore`。** 前者是写进注释的历史 bug（Crocodile 前后腿
   曾撞成同一句），后者是 `family_key` 真踩过的坑：把方位词提到 key 最前时，`ForeArm`（前臂，
   radius/ulna）会被并进 `FrontArm`（前肢的臂）。源模块用 `('fore','arm') → 'Forearm'` 挡它，
   D3 用「`fore` 不列入方位词」挡它。涉及 `front/back/rear/hind/fore` 和 `lf/rf/lb/rb` 的任何
   改动，都要**逐条打印被移动的 key**再决定，别只看总数。

4. **默认改 B 层。** 只有当你确实要改导出的 BVH 骨名时才动 A 层，并且要记得 A 层的剥离会被
   `_disambiguate_duplicate_canonical_names` 部分抵消（见 §0-3）。

5. **置零 ≠ 删除。** 进了 B4 的关节拿到零向量，意思是"这个名字不携带任何跨物种信息"。
   它的拓扑和几何照样进模型。对 `Bone02` 这种纯占位名，置零比编码一个随机方向更诚实。

6. **加了词表要补测试。** 参考 [`tests/test_joint_embedding_texts.py`](../tests/test_joint_embedding_texts.py)、
   [`tests/test_symmetry_metadata.py`](../tests/test_symmetry_metadata.py)。

---

## 5. 已知缺口（下一个数据集很可能踩到）

- **`face_orientation.py` 有自己的一份 `_canonicalize_joint_name` 和 `_joint_signature`**
  （[L77](../data_loaders/truebones/truebones_utils/face_orientation.py#L77) / [L125](../data_loaders/truebones/truebones_utils/face_orientation.py#L125)）。
  `_joint_signature` 的 `lb/rb` 缺口已补（与 `physics_joint_annotation` 同样保留前后半码），
  但**两份实现仍然并存且行为故意不同**——这一份丢弃单字符 token 含孤立数字。
  改任何一份都要同步检查另一份。
- **`_EMBED_TEXT_CREATURE_TOKENS` 是手工列表，不是从 `species_tags.jsonl` 派生的。**
  加新物种时**必须手工补**，没有任何机制会提醒你。
  （没做成自动派生，是因为变体 rig 里会出现数据集中并不存在的生物名，比如 antilope 的 Quilin/Moose。）
- **缩写表按物种硬编码**（`thi`/`clf`/`nek`/`spn` 来自 SabreToothTiger，`Clav`/`Scap` 来自 Deer_Buck）。
  新 rig 有自己的缩写体系时只能继续往 B6 加。
- **`ant` / `horse` / `jaws` 被刻意排除在物种表之外**（分别撞 antenna、HorseLink、jaw）。
  再加物种名时要检查是否和解剖词撞车。

---

## 6. 收尾：什么时候必须 bump / 重算 / 重训

| 改了什么 | bump `JOINT_NAME_EMBEDDING_SCHEMA_VERSION` | 重跑预处理 | 重训 | renamer bank 重建 |
|---|---|---|---|---|
| B 层任意词表 | ✅ | `--re-encode-joint-names-only` 足够 | ✅ | — |
| A 层任意词表 | ✅ | 全量（BVH 骨名会变） | ✅ | ✅ |
| C 层（side / contact / 朝向） | ✅ | 全量 | ✅ | — |
| D 层（`family_key`） | — | — | — | ✅（喂 S6 hierarchy prior） |

版本号在 [`physics_joint_annotation.py:475`](../data_loaders/truebones/truebones_utils/physics_joint_annotation.py#L512)。
忘了 bump 的话不会报错，只会静默用旧 embedding 训练；bump 之后加载旧 cond 会打印
`uses joint-name embedding schema N; current code expects N+1` 告警。
