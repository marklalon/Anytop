# 关节部位标注与辅助预测头

> 状态：方案，未实施。
> 核心思路：每个关节的「部位 + 接触」由人工核验的标注提供。它只作为训练的辅助预测目标和 loss 的权重来源，不再作为条件喂给模型。模型学会从名字、几何和运动推断部位，生成时把预测结果写进输出目录的 `joint_parts.jsonl`。
> 数据来源唯一：训练时从 `joint_parts.jsonl` 读，推理时从生成目录的 `joint_parts.jsonl` 读。两处格式相同，npy 本身不带元数据。cond.npy 不再保存任何接触或部位信息。
> 影响：需要重新生成 cond；模型侧做 joint_struct schema bump 和 CKPT bump，从头重训；生成的 npy 格式不变。

## 1. 背景与依据

### 1.1 现状

- 接触信息有两个来源，都是启发式推断：`physics_joint_annotation.infer_contact_joints`（先几何，后名字）和 `_infer_end_effector_joints`。推断在预处理时运行，结果写入 cond 的 `contact_joints / contact_joint_names / contact_joint_source / end_effector_joints / end_effector_names`。
- 接触信息通过 `joint_struct` 的三个通道 `run_ends_contact / is_contact / contact_known`（[joint_struct_features.py](../data_loaders/truebones/truebones_utils/joint_struct_features.py)）作为**结构条件**进入 `InputProcess.struct_embedding`；`--topology_cond` 池也会读到它们。
- 「部位」在代码里目前没有显式表示，只隐式地由 T5 名字嵌入携带。

### 1.2 探针实验结论（merged_all_v41，2026-10-04）

| 特征来源 | 名字 15% 置零（与训练一致） | 名字全部置零 |
|---|---|---|
| 原始输入，最好的非线性 probe | 0.778 | 0.778 |
| 中间层，线性 probe | 0.82 / 0.83 | 0.79 / 0.785 |
| 中间层，MLP probe | 0.81 / 0.83 | 0.80 / 0.815 |
| 名字保留（上限） | 0.95 | 0.95 |

由此得出的设计约束：

1. **提升空间只出现在名字缺失时。** 部位 loss 只加在被置零名字的关节上。否则模型会去学「名字→部位」这条捷径，而这条捷径本来就有。
2. **部位信息集中在浅层，越深越淡。** 预测头接在中间层。
3. **soft（毛发、衣物）最弱（0.5–0.7）。** 人工核验要重点覆盖。
4. **探针证明不了动作质量会变好。** 第 8 节用新旧对比来验收。

### 1.3 为什么把接触从条件改为目标

- 推断出来的接触是**有噪声的条件**：模型照单全收，错标的脚会被当成支撑点来学。
- 生成时遇到新骨架，只能再跑一遍同一个启发式；训练和推理的噪声分布虽然一致，但都是错的。
- 改为目标以后，标注错误只影响辅助 loss；模型对新骨架的接触判断可以直接输出，供下游使用。

## 2. 标签定义

### 2.1 部位类别

每个关节恰好属于一个类别。ID 顺序固定，改动时要 bump `JOINT_PART_SCHEMA_VERSION`。

| id | 名称 | 含义 / 判定 |
|---|---|---|
| 0 | `trunk` | 骨盆、脊柱、胸腔；蛇、鱼、蠕虫的主体段 |
| 1 | `neck` | 颈椎链 |
| 2 | `head` | 头及其附属：下颌、眼、耳、舌、角、喙、触须根 |
| 3 | `arm` | 前肢从肩/锁骨到腕之前 |
| 4 | `hand` | 腕及以下：掌、指、爪 |
| 5 | `leg` | 后肢从髋到踝之前；多足动物步行足的近端段 |
| 6 | `foot` | 踝及以下：跖、趾、蹄、爪尖 |
| 7 | `wing` | 整条翼链，包括翼指和受驱动的羽毛骨 |
| 8 | `tail` | 尾链 |
| 9 | `fin` | 鳍 |
| 10 | `soft` | 被动附属物：毛发、鬃、衣物、披风、饰物、被动触手、植物叶片 |
| 255 | `helper` | 非解剖节点：包装根、locator、IK target。显示，但不进 loss |

未标注的关节在训练里取 ignore（−1）。`helper` 是一个确定的判断，ignore 表示「还不知道」，两者不同。

### 2.2 接触位（独立于部位）

`contact ∈ {0, 1}`：正常站立或行走时会着地的关节。语义沿用现有 `contact_joints`，即**整条脚链**（踝、趾、趾尖）。四足动物的前肢 `hand` 链同样可以是 contact。

### 2.3 歧义判定规则（写进 UI 的帮助面板）

- **按功能判定，不按骨骼同源。** 蝙蝠的「手指」是 `wing`；章鱼主动划水的腕是 `arm`；水母被动拖曳的触手是 `soft`。
- 海龟、海豹的鳍肢：主动划水的大鳍肢归 `arm/leg`，小的稳定鳍归 `fin`。
- 尾巴末端受尾链驱动、自身无主动运动的饰物（毛团、尾羽），归 `soft`。
- 爪长在 hand 上的归 `hand`，长在 foot 上的归 `foot`。

## 3. 数据：存储、预填工具、读取路径

### 3.1 唯一数据源

| 场景 | 读哪里 | 读取函数 |
|---|---|---|
| 预处理、训练 loader、数据集 clip 的还原和评估 | `<processed>/joint_parts.jsonl` | `joint_parts.load_joint_parts(sidecar_dir, species, cond_entry)` → 按关节名绑定到 cond 骨架 |
| 推理生成的动作（导出、还原、评估、后处理） | 生成输出目录下的 `joint_parts.jsonl` | 同一个函数，`sidecar_dir` = npy 所在目录 |

- **cond.npy 不保存任何接触或部位字段。** 删除 `contact_joints / contact_joint_names / contact_joint_source / end_effector_joints / end_effector_names`，也不新增 `joint_parts`。改了标注不需要重新生成 cond，两边也不会不一致。
- 训练 loader 在内存中把绑定好的数组挂到样本上，供 leaf_drop 同步删除、bone_length_aug 读取，**永远不回写 cond.npy**。
- 两种场景用同一个读取函数和同一种格式，只是目录不同。函数放在 `data_loaders/truebones/truebones_utils/joint_parts.py`，不依赖 `motion_edit`、`sample` 或 `tools`。

### 3.2 sidecar 格式

```json
{"species": "Alligator",
 "skeleton_sig": "sha1(joints_names + parents)",
 "reviewed": false,
 "joints": {"koshi":   {"part": "trunk", "contact": 0, "src": "name", "why": "Hips"},
            "sippo01": {"part": "tail",  "contact": 0, "src": "name", "why": "Tail 01"}}}
```

- **按关节名索引，不按下标。** 物种的关节集合一变（剪裁、prop socket 删除、leaf 清理），下标就会悄悄错位，`chain_forward_joints.jsonl` 在 Pirrana 上就出过这个问题。
- `skeleton_sig` 和当前 cond 不一致时，这一行判为 **stale**：UI 列出新增和消失的关节，要求重新确认；预处理和训练对 stale 行直接报错。
- `src ∈ {name, inherit, geometry, contact_rule, manual}`，`why` 是一句依据，用于 UI 悬浮提示。人工改过的关节记为 `src = manual`，重新预填时不会被覆盖。

### 3.3 预填工具 `tools/prefill_joint_parts.py`

这是每接入一个新数据集都要跑的流水线步骤，所以放在 `tools/`。分类逻辑放在新模块 `data_loaders/truebones/truebones_utils/joint_parts.py`，工具、UI 的「重新预填」按钮和 precheck 调用的是同一个函数。整个模块**只依赖 `data_loaders` 自身**，不引用、也不读取 motion_edit 的代码或产物。

```
python tools/prefill_joint_parts.py --dataset truebones/zoo [--filter A,B] [--dry-run]
```

默认只补两类关节：没有行的物种，以及已有行里 `src != manual` 且未 reviewed 的关节。

分四遍做，前一遍的结果优先：

1. **名字。** 输入是 `build_joint_embedding_texts` 的 slim 文本加 canonical 名。部位关键词表写在 `joint_parts.py` 里，用 `joint_name_matches_keywords` 做**词前缀匹配**（`ear` 不能命中 `rear`）。`joint_name_is_helper_node` 命中的关节归 `helper`。
2. **沿树继承。** 名字为空或未命中的关节（`Bone_03`、`Leaf`、`Petal`）继承最近的已标祖先。然后细化：`arm` 链上腕及以下改为 `hand`，`leg` 链上踝及以下改为 `foot`。
3. **几何兜底**，只处理前两遍都没定下来的关节。使用 `joint_struct` 的量（`height_n / lateral_signed / fore_aft_n / attach_h_n / is_leaf`）和物种的 `species_tags`：
   - 沿前后轴延伸最远的两条主链：前端为 neck→head，后端为 tail；
   - 侧向分支，末端接近地面：`leg`→`foot`；
   - 挂点高、横向展开大，且物种标记为飞行：`wing`；
   - 物种标记为游泳或漂浮，且分支很短：`fin`；
   - 其余短叶链：`soft`，UI 中标为低置信度。
4. **接触。** 把现有 `infer_contact_joints`（及其私有辅助函数）从 `physics_joint_annotation.py` 原样搬进 `joint_parts.py`，作为接触预填，初值与今天 cond 里的 `contact_joints` 完全一致。

工具输出一份报告：每个物种各类别的计数、几何兜底的关节数。报告按兜底数从多到少排序，作为人工核验的顺序。

### 3.4 工作量

4 个数据集共 331 个物种、约 1.3 万个关节。人工主要看三类：几何兜底的关节、`soft`、接触。按每个物种 1–2 分钟估计，全部核验一遍需要 6–10 小时。

## 4. 核验 UI（`dataset/review/`）

### 4.1 页面与路由

新增独立页面 `dataset/review/parts.html`，和 `index.html` 互相加一个顶栏链接，共用 `serve.py` 的数据集发现逻辑。`discover_datasets` 增加 `joint_parts` 路径；数据集只要有 `cond.npy` 就列出来，不要求有 `action_labels.jsonl`。

**前端库自带**：three.module、OrbitControls 和**同版本**的 BVHLoader 放在 `dataset/review/vendor/`，不指向 `motion_edit/ui/vendor/`。视口的绘制代码在 `parts.html` 里独立编写。

[serve.py](../dataset/review/serve.py) 新增：

| 路由 | 作用 |
|---|---|
| `GET /parts` | 返回 `parts.html` |
| `GET /vendor/*` | 返回 `dataset/review/vendor/` 下的文件 |
| `GET /api/parts/species?ds=` | 物种列表：关节数、状态（`missing / auto / reviewed / stale`）、几何兜底数、各类别计数 |
| `GET /api/parts/tpose.bvh?ds=&species=` | 单帧 T-pose BVH，由 cond 生成，缓存到 `<processed>/bvh_tpose/<species>.bvh`；cond 的 mtime 变化后重建 |
| `GET /api/parts/skeleton?ds=&species=` | 每个关节：index、raw 名、canonical 名、embedding 文本、side、mirror twin、part、contact、src、why |
| `POST /api/parts/update` | 修改若干关节的 part / contact，或设置 `reviewed`。整文件原子重写，带 mtime 冲突检测，做法同 `LabelStore` |
| `POST /api/parts/prefill` | 对一个物种重跑预填（`manual` 关节不动），返回 diff，确认后写入 |

把 cond 写成 BVH 的那段函数，从 `tools/sample_tpose_bvh.py` 提取到 `data_loaders/truebones/truebones_utils/` 下，由工具和 serve.py 共用；serve.py 不 import `tools/`。

T-pose **只从 cond 生成，不读原始 GLB**。cond 里的骨架才是训练时的那一副：已剪裁、已去 prop socket、已统一朝向和尺度。BVH 的关节顺序就是 cond 的顺序，前端用序号对应 index，不靠名字匹配。

### 4.2 布局与交互

```
┌ 顶栏：数据集 ▾ │ 状态筛选 ▾ │ 搜索物种 │ 进度 123/331 │ 动作标签 ↗ ┐
├ 左栏：物种列表 ┬ 中：3D 视口 ──────────────┬ 右栏：关节表 ──────┤
│ ● Alligator  ✓ │  关节球按部位着色          │ # 名字 文本 部位 接触 │
│ ○ Ant     (3)  │  骨段按子关节部位着色      │ 每行可下拉 / 勾选    │
│ ⚠ Bat  stale   │  接触关节外圈白环          │ 来源 why 悬浮提示    │
│                │  hover 显示名字/部位/why  │                      │
├────────────────┴── 色板：1 trunk 2 neck … 0 soft H helper  C 接触 ──┤
└ [重新预填] [左→右镜像] [撤销] ………………… [确认并下一个 ⏎] ┘
```

- **视口**：中键旋转，Shift+中键平移，滚轮缩放；正视、侧视、俯视三个快捷视角。关节用 `InstancedMesh` 球体，骨段用线，按 bbox 自动取景。
- **配色**：每个部位一个固定色相，明暗两种主题下都要有足够对比度。`src = geometry` 的关节画成半透明，`helper` 用灰色小点，`contact` 用白色外环（不靠颜色区分，色盲也能看出来）。
- **选择**：单击选一个关节；Shift+单击选整棵子树；Ctrl+单击切换单个关节；拖框多选。视口和右栏表格双向联动。
- **标注**：数字键或色板设部位，`C` 切换接触；「左→右镜像」按 mirror twin / side 配对复制；支持 `Ctrl+Z` 撤销。
- **确认**：`⏎` 写入 `reviewed: true` 并跳到下一个未核验物种。改动即时保存（防抖 500 ms）。
- **第二阶段（可选）**：从 `bvhs/` 选一个该物种的行走类 clip 播放，下方实时画出接触关节的高度曲线。用来核对接触比只看 T-pose 可靠，但第一版不做。

## 5. 预处理与读取方改动

### 5.1 删除推断，以及 cond 里的字段

| 位置 | 改动 |
|---|---|
| `physics_joint_annotation.infer_contact_joints` 及私有辅助函数、`_infer_end_effector_joints` | 前者搬进 `joint_parts.py`，只作预填用；后者删除 |
| `features.get_common_features_from_rest_pose`：调用推断、`TPoseFeatures.foot_indices / contact_joint_source` | 删除推断。`foot_indices` 由调用方从 sidecar 解析后传入 |
| `features.tpose_features_from_cond` | 不再读 `cond['contact_joints']`，改为接收 `contact_joints` 参数 |
| `dataset_pipeline.py`、`joint_name_canonical.py` 写入 `end_effector_* / contact_*` | 删除，不写入任何替代字段 |
| `regenerate_dataset_artifacts._recompute_contact_joints` | 删除；改为一个只读检查：每个物种在 sidecar 里都有非 stale 的行 |
| [skeleton_metadata.py](../data_loaders/skeleton_metadata.py) | 除自身外没有引用，整文件删除 |
| `joint_embedding_text._bare_leg_means_calf` 的 `end_effector_joints` 参数 | 删除（函数里本来就有「无子节点」判断）。删完对全体 embedding 文本跑 diff，有任何变化就 bump 名字 schema |
| `leaf_drop` | `_INDEX_LIST_KEYS` 删掉 `contact_* / end_effector_*`；改为同步删除 loader 挂上来的 `joint_parts / joint_contact` 数组，`_protected_joints` 改读这两个数组 |
| `build_joint_name_inspection_rows` | `is_contact / is_end_effector` 改为从 sidecar 读 `part / contact` |
| `validate_anytop_dataset.py:134`、`precheck_dataset.py` | 改为校验 sidecar：覆盖全部物种、无 stale、取值合法、contact ⊂ 非 helper；未 reviewed 给 WARN |

### 5.2 其余读取方改走唯一来源

| 读取方 | 场景 | 来源 |
|---|---|---|
| `bone_length_aug`（肢体分组、高度补偿） | 训练 loader | loader 挂上来的数组（来自 sidecar） |
| `eval/motion_quality/scorer.py`（limb mask） | 数据集 clip / 生成结果 | 数据集 sidecar / 生成目录 sidecar |
| `utils/npy_restore.py`（`_trunk_fields`） | 数据集 clip / 生成结果 | 数据集 sidecar / 生成目录 sidecar；`ctx.contact_joints` 由调用方传入，不再读 cond_entry |
| `utils/exporter._ground_root_on_lowest_contacts` | 生成结果导出 | 生成目录 sidecar；**外部 rig** 用 `joint_parts.prefill_contacts` 启发式 |
| `utils/retarget_pipeline.py`（`bake_foot_floor_offset`、源骨架接触） | 目标是数据集物种时读 sidecar；源骨架或外部 rig 用启发式 | 同左 |
| 预处理中需要接触的步骤 | 构建数据集 | sidecar；**缺少行就直接报错**，提示先运行 `prefill_joint_parts.py` |

`process_new_skeleton`（推理时的新骨架）不需要标注：部位和接触由模型预测，写进生成目录的 `joint_parts.jsonl`（第 6.4 节）。

## 6. 模型改动

### 6.1 条件侧：删除接触通道

- `JOINT_STRUCT_FEATURE_NAMES` 去掉 `run_ends_contact / is_contact / contact_known`，`JOINT_STRUCT_FEATURE_SCHEMA_VERSION` 从 1 改为 2。`build_joint_struct_features` 不再读接触，模块 docstring 同步修改。
- `struct_embedding` 的输入维度跟随 `JOINT_STRUCT_DIM`，`--topology_cond` 池自动跟随。

### 6.2 辅助预测头

**数据管道**（`data_loaders/tensors.py`、loader）：每个样本新增三个键，并且**始终存在**（compile 要求键集合稳定，所以没有标注时填全 −1 / 全 False，不能是 None）：

- `y['joint_part_target']`：`int64[J_max]`，`helper`、未标注和 padding 为 −1；
- `y['joint_contact_target']`：`float[J_max]`；
- `y['joint_contact_valid']`：`bool[J_max]`。

**名字置零 mask 外置**：现在 `_drop_joint_names` 在 `InputProcess` 内部抽样。改为在 `AnyTop.forward` 入口抽样得到 `y['joint_name_drop']`（`bool[B, J]`），`InputProcess` 只负责应用。抽样分布不变；采样时 mask 全为 False，键依然存在。

**预测头**（`model/anytop.py`）：

```
h = 第 k 层 decoder 输出   [T+1, B, J, d]      k = --part_head_layer，默认中间层
z = concat(h[0], mean_t(h[1:], 按有效帧加权))   [B, J, 2d]   # 0 帧是 rest token
part_logits   = MLP_part(LN(z))      [B, J, C]   # d→d/2→C，GELU
contact_logit = MLP_contact(LN(z))   [B, J]
```

- 两个头合计约 0.2M 参数，fp32 计算。decoder 加一个可选参数 `return_layer=k`，前向时把该层输出顺手保存下来，不额外做一次前向。
- `AnyTop.forward(..., return_aux: bool = False)`：默认只返回 x0，采样路径不用改；训练时传 `True`，返回 `(x0, part_logits, contact_logit)`。`return_aux` 是 Python 常量，只会多出一张 compile 图。

**loss**（`gaussian_diffusion.training_losses`）：

```
part_mask  = joint_name_drop & (part_target >= 0)
L_part     = CE(part_logits, part_target)[part_mask]，按类别逆频率开方加权
L_contact  = BCE(contact_logit, contact_target)[contact_valid]
loss      += lambda_part * L_part + lambda_contact * L_contact
```

- 部位 loss 只看被置零名字的关节（约束 1）；接触 loss 看全部关节，因为接触不能从名字读出。
- 类别权重在启动时由训练集统计一次，写进 args.json。
- 不随 t 加权：rest token 不加噪，探针里 t=0 和 t=50 只差约 1 个点。

### 6.3 版本与兼容

- joint_struct schema 从 1 改为 2，加上新参数，一起做 CKPT bump。旧 checkpoint 直接拒绝，不做 warm start。
- `args.json` 记录 `joint_part_schema_version`，resume 和生成时校验。
- cond 需要重新生成（删除字段）；motions 不需要重新预处理。

### 6.4 推理输出：生成目录下的 joint_parts.jsonl

生成的 `.npy` **格式不变**，仍然是裸特征数组（`sample/export.py:73`）。部位信息和数据集一样放在旁边的 sidecar 里：生成时在输出目录写一个 `joint_parts.jsonl`，格式与 3.2 相同，每个骨架一行，同一骨架的所有 npy 共用这一行。

- 读取方已经能从文件名加 cond 解析出物种（`infer_object_type_from_filename` / `resolve_species_key`），拿到物种名后，就到 npy 所在目录的 `joint_parts.jsonl` 里查这一行。数据集 clip 用的是同一个函数 `joint_parts.load_joint_parts(sidecar_dir, species, cond_entry)`，只是 `sidecar_dir` 不同。
- 行的来源：
  - **已知物种**（数据集 sidecar 里有这一行）：原样复制人工标注，`source = annotation`；
  - **新骨架**（`process_new_skeleton`）：用模型预测，`source = model`。
- 行里每个关节额外记录 `part_prob` 和 `contact_prob`。已知物种也记录，用来诊断辅助头和人工标注的差距。3.2 的解析器会忽略这两个额外字段。
- 模型预测的取法：采样最后 10% 去噪步的 logits 取平均（这些步传 `return_aux=True`，CFG 只取 cond 分支），再对本次运行里该骨架的**所有样本**求平均，然后取 argmax，接触以 0.5 为阈值。部位是骨架的属性，不是单条动作的属性，所以每个骨架只存一行。
- 写入规则：输出目录里已经有 `skeleton_sig` 相同的行时，**保留不动**，保证同一目录下先后导出的动作看到的部位一致；sig 不同（骨架已经变了）就覆盖，并打印警告。
- 需要接触的读取方（`write_feature_bvh`、`tools/restore_glb_from_npy.py`、eval）找不到这一行时直接报错，提示用新代码重新生成，**不退回启发式**。

motion_edit 如何消费这个块，留给它自己重构，不在本方案范围内。

## 7. 辅助目标能否反过来提升动作质量

辅助头本身只通过**共享主干的梯度**起作用：它逼迫中间层编码部位信息，但不保证这些信息被用来生成更好的动作。下面讨论两种更直接的回路，本方案都不采用（7.2 默认关闭，可选）。

### 7.1 预测结果再注入（不推荐）

做法：把第 k 层的部位概率经过一个线性层，加回 k+1 层之后的残差流，相当于模型自己产出一个条件再喂给自己。

不推荐的原因：探针显示，部位信息在中间层已经是**线性可读**的。把它压成 softmax 再加回去，不会带来新信息，只是把同一信息换了个形式。用 ground truth 做 teacher forcing 会产生 exposure bias：训练看的是干净标签，推理看的是预测，结果等于把「标注作为条件」从后门请回来。如果用的是预测值，又基本是冗余的。跨去噪步的 self-conditioning 同理，还要多付约 1.5 倍训练开销。

### 7.2 部位感知的 loss 权重（可选，默认关）

`l_simple` 按部位加权，例如把 `soft` 的权重降到 0.5。被动附属物的运动本质上是噪声，不该和躯干抢容量。这只是调权重，不加新信息；等第 8 节的结果出来，再看要不要开。

## 8. 验收：新旧对比

只比两个版本：

- **旧**：现有 merged_all_v41 checkpoint（300k 步，EMA），接触在 cond 里，没有辅助头。
- **新**：本方案全部改动：去掉接触条件，加辅助头。训练配置和 v41 相同，训练到同样的步数。

评估在同一组物种和动作标签上进行，推理精度设置相同（不同精度下的 eval 分数不可比）。两边计算指标时使用**同一份 sidecar 标注**，保证足部指标的口径一致。

| 场景 | 指标 |
|---|---|
| 正常名字 | `eval/motion_quality` 的足部滑动、穿地，`fk_angle_deg`，`l_simple` |
| 名字全空（模拟 `process_new_skeleton` 遇到未见名字） | 同上，另加新版辅助头的部位平衡准确率和接触 AUROC |

通过条件：

- 新版在名字全空场景下的动作质量优于旧版；
- 新版在正常名字场景下不退化；
- 辅助头在名字全空时的部位准确率 ≥ 0.85（探针基线约 0.79）。

只比两个版本，所以无法区分各项改动各自的贡献。如果新版退化，再针对性地补做消融；第一个怀疑对象是「去掉接触条件」。

## 9. 实施顺序

1. `joint_parts.py`（类别表、按关节名绑定、读取函数、启发式迁移）、`tools/prefill_joint_parts.py` 和单测；对 4 个数据集跑 dry-run，查看报告。
2. `serve.py` 路由、`parts.html`、`dataset/review/vendor/`，然后人工核验全部 331 个物种。
3. 预处理和读取方改造（第 5 节）、cond 重新生成、precheck 和 validate。
4. 模型和辅助 loss（第 6 节）和生成目录 sidecar 输出（6.4）；跑单测，确认 `return_aux=False` 时与改动前逐位一致，确认 compile 图的数量。
5. 训练新版，按第 8 节与 v41 对比。

第 2 步的人工核验可以和第 3、4 步并行；第 5 步必须等第 2 步全部 reviewed 后再开训。
