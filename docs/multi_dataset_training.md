# 多数据集训练方案（cond.npy 驱动）

> 状态：**P0~P5 已实施**（P6 全量重预处理+重训未做，需要 GPU 长跑）
> 目标：让 AnyTop 既能用单个数据集训练，也能用多个数据集合并训练，**差别仅在于训练时传入哪个 `cond.npy`**。

## 已落地的入口

```bash
# 合并成一个训练 cond（不复制任何 motion 文件，默认重算统计量）
python tools/merge_dataset_cond.py --datasets dataset/datasets.jsonl \
    --out dataset/merged/cond.npy

# 训练：单数据集 / 合并只差 --cond_path
python train/train_anytop.py --cond_path dataset/truebones/zoo/truebones_processed/cond.npy ...
python train/train_anytop.py --cond_path dataset/merged/cond.npy ...

# 推理：不传 --cond_path 时自动读 checkpoint 同目录的 cond.npy 快照
python -m sample.generate --model_path save/<run>/model.pt --object_type zoo_upgrade/Horse

# 打分 / 校验：目录 = 单源，.jsonl = 清单多源
python eval/evaluate_motion_quality.py --dataset_root dataset/datasets.jsonl ...
python utils/validate_anytop_dataset.py --datasets dataset/datasets.jsonl
```

新增模块：[`dataset_sources.py`](../data_loaders/truebones/truebones_utils/dataset_sources.py)（namespace / 规范键 / 文件名 token / 清单）、
[`cond_schema.py`](../data_loaders/truebones/truebones_utils/cond_schema.py)（v4 读写与就地升级）。

> 旧版 cond.npy 在**加载时自动升级到 v4**（namespace 由目录推断），所以未迁移的数据集也能直接跑；
> 要让磁盘上的文件落盘 v4，直接重跑预处理管线即可（v4 打戳已包含在写出步骤里）。

---

## 0. 两个契约

| 契约 | 组成 | 消费者 |
|---|---|---|
| **训练契约** | `cond.npy` + 它所引用的各数据集目录下的 `motions/`、`motion_metadata.json`、`action_tags.jsonl`、`species_tags.jsonl`、`train/val/test.txt` | `train_anytop.py`、`Truebones` dataset |
| **推理契约** | `cond.npy` **单文件**（自足，不依赖任何数据集目录） | `generate.py`、`server/`、`process_new_skeleton`、跨骨架 retarget |

`cond.npy` 是唯一入口：训练时它告诉加载器「有哪些物种、动作数据在哪」；推理时它自带全部条件（含烘焙的 species tags）。

**数据集清单 `datasets.jsonl` 只有两类消费者**：离线合并工具、`eval / --score`。训练与推理都不读清单。

---

## 1. 数据集清单格式

默认位置 `dataset/datasets.jsonl`（不强制，路径由命令行显式传入）：

```jsonl
{"namespace": "truebones/zoo",         "path": "dataset/truebones/zoo/truebones_processed"}
{"namespace": "truebones/zoo_upgrade", "path": "dataset/truebones/zoo_upgrade/clean_processed"}
```

| 字段 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `namespace` | 是 | — | 唯一，正则 `[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*`；不允许某个 namespace 是另一个的路径前缀（否则后缀解析有歧义） |
| `path` | 是 | — | **Anytop-root-relative** POSIX 路径（工作目录即 Anytop）或绝对路径，用现有的 [`param_utils._resolve_project_path`](../data_loaders/truebones/truebones_utils/param_utils.py#L11) 解析 |
| `enabled` | 否 | `true` | |
| `species_include` / `species_exclude` | 否 | — | 裸名列表，方便临时裁剪 |

**行序 = 裸名解析优先级**（重名时取第一个）。

---

## 2. Namespace 与物种键

### 2.1 规范键

cond dict 的 key 一律为 `<namespace>/<species>`：

```
truebones/zoo/Horse
truebones/zoo_upgrade/Horse
```

单数据集的 cond 也带 namespace（预处理时写入），因此**合并 = 纯并集**，格式完全一致。

### 2.2 解析规则

`resolve_species_key(cond_dict, user_input) -> canonical_key`，按顺序尝试：

1. 精确匹配规范键 —— `truebones/zoo/Horse`
2. 后缀匹配 —— `zoo/Horse`、`zoo_upgrade/Horse`；**唯一命中才接受**，多命中则报错并列出候选
3. 裸名 —— `Horse` → 按 cond 插入顺序取第一个命中
4. 以上三轮先做大小写敏感，全部落空后再做一轮大小写不敏感

第 4 步解决实测存在的 `Rhino` ↔ `rhino`、`Scorpion` ↔ `scorpion` 大小写冲突：大小写敏感轮先命中精确的那个，避免误匹配。

> 实施澄清：第 2 步**只对含 `/` 的选择器生效**。否则裸名 `Horse` 会同时后缀命中两个 Horse 而报歧义，
> 第 3 步（取第一个源）永远走不到，与「单数据集用法不变」冲突。
> 另外解析器还接受 §2.3 的文件名 token（`Horse@truebones_zoo_upgrade`），
> 这样从生成文件名反解出的物种和用户输入走的是同一套规则（`eval_checkpoint --score` 依赖这点）。

### 2.3 文件名 token

规范键含 `/`，不能直接进文件名。`species_file_token(cond_dict, key)`：

- 该物种裸名在整个 cond 中唯一 → 返回裸名 `Horse`
- 否则 → `Horse@truebones_zoo_upgrade`（`/` 换 `_`）

影响点：

- [sample/generate.py:1602](../sample/generate.py#L1602) `npy_name = f'{object_type}_{i}.npy'` 必须改用 token
- [sample/generate.py:422](../sample/generate.py#L422) / [:498](../sample/generate.py#L498) retarget 中间产物命名同上
- `utils/misc.infer_object_type_from_filename` 的 `valid_types` 匹配表要用 **token → 规范键** 构建，才能把 `truebones_zoo_upgrade_Horse_Idle_1.npy` 反解回规范键

> 效果：单数据集下所有裸名唯一，输出文件名与今天完全一致，不破坏现有使用习惯。

---

## 3. cond.npy 格式（schema v4）

原则：**不引入非物种的顶层 key**（大量代码假设 `for k in cond_dict` 都是物种），所有新信息放进每条 entry。

新增字段：

| 字段 | 类型 | 训练用 | 推理用 | 说明 |
|---|---|---|---|---|
| `cond_schema_version` | int | 是 | 是 | `4`，加载时校验 |
| `dataset_namespace` | str | 是 | — | `"truebones/zoo"` |
| `dataset_root` | str \| None | 是 | — | **Anytop-root-relative** POSIX 路径（同 §1 的 `path` 约定）；`None` = 「cond.npy 自身所在目录」（单数据集 cond 用这个，天然可移植） |
| `species_name` | str | 是 | 是 | 裸名。用于 join `motion_metadata.json` 的 `object_type`、`species_tags.jsonl` 的 key、以及 `motions/{species_name}_*.npy` 前缀 |
| `species_tags` | tuple[str, ...] | — | 是 | 推理用烘焙副本；**训练不读它**，训练读各源的 `species_tags.jsonl` |

变更字段：

- `object_type`：由裸名改为**规范键**（与 dict key 保持一致）；裸名移到 `species_name`

保持不变：`parents / offsets / rest_pose / joints_names_embs / species_emb / canonical_feature_mean·std / …`

> 已确认两个现有数据集的 `feature_space=canonical_motion_v3`、`joints_names_embs_meta.schema_version=8`、`t5_name=t5-base` 完全一致，可直接合并。

---

## 4. 合并工具 `tools/merge_dataset_cond.py`

```
python tools/merge_dataset_cond.py \
    --datasets dataset/datasets.jsonl \
    --out dataset/merged/cond.npy \
    [--no-recompute-stats] \
    [--dry-run]
```

**只产出一个 `cond.npy`，不复制任何 motion 文件。**

### 4.1 流程

1. 读清单 → `DatasetSource(namespace, root, include/exclude)` 列表
2. 逐源加载 `cond.npy`，**跨源一致性校验**（任一不符则 fast-fail）：
   - `feature_space` / `physical_feature_space` 一致
   - `joints_names_embs_meta` 的 `schema_version` / `t5_name` / `embedding_dim` 一致
   - `species_emb_meta.t5_name` 一致
   - `len(parents) <= MAX_JOINTS`（**zoo 的 Dragon 已顶到 100 上限**，新数据集越界必须立刻报错）
   - 该源 `species_tags.jsonl` 覆盖其 cond 中全部物种
3. re-key 成 `<namespace>/<species>`，写入 §3 的新字段（`species_tags` 从该源 `species_tags.jsonl` 读入烘焙）
4. 冲突检测：`namespace` 重复 → 报错；规范键重复 → 报错（正常不会发生）
5. **统计量重算**（默认开，见 §4.2）
6. 输出报告：每源物种数 / clip 数、裸名冲突表、每 object_subset 统计量 before→after
7. 原子写（temp + rename）

### 4.2 归一化统计量重算（关键）

`canonical_feature_mean/std` 是 **per-object_subset** 的常量，两个数据集各自独立统计，实测差异明显：

```
quadruped   zoo:         mean[0]=0.0098  std[0]=0.4691
            zoo_upgrade: mean[0]=0.0573  std[0]=0.4013
```

不重算不会崩（每个物种自洽反标准化），但同一个 subset 桶里会存在两套标准化空间，削弱 AnyTop 赖以泛化的跨物种共享。

合并工具是唯一同时看得到所有源 motions 的地方，因此在这里重算：

- 遍历所有源的 `motions/*.npy`，按物种的 object_subset 分桶
- 复用 [canonical_features.accumulate_lnorm_stats](../data_loaders/truebones/truebones_utils/canonical_features.py) / `finalize_lnorm_stats` 增量累积（与 [regenerate_dataset_artifacts._compute_canonical_stats_per_object_subset](../tools/regenerate_dataset_artifacts.py#L244) 同一套逻辑）
- 结果写回**合并 cond 的所有 entry**；各源自己的 `cond.npy` 不动

`--no-recompute-stats` 保留为逃生口（快速验证管线用）。

---

## 5. 训练侧改造

### 5.1 命令行

- `train_anytop.py` 新增 `--cond_path`，**必填**（与 `generate.py` 同名）
- 单数据集训练：`--cond_path dataset/truebones/zoo/truebones_processed/cond.npy`
- 合并训练：`--cond_path dataset/merged/cond.npy`
- `--objects_subset all` 的含义 = 当前 cond.npy 里的全部物种

### 5.2 `get_opt` / opt

```python
get_opt(device, cond_path)  # cond_path 必填
opt.cond_file = cond_path
opt.sources   = tuple[DatasetSource]   # 由 cond entry 的 dataset_root 去重派生
```

删除单值 `opt.data_root` / `opt.motion_dir`（当前 [get_opt.py:31-33](../data_loaders/truebones/truebones_utils/get_opt.py#L31-L33)），改由 `opt.sources` 驱动枚举。
`dataset_root` 为 `None` 的 entry 归到「cond.npy 所在目录」这个源。

### 5.3 `Truebones` / `MotionDataset`

改造点（[data_loaders/truebones/data/dataset.py](../data_loaders/truebones/data/dataset.py)）：

1. **clip id 复合化**。实测两个数据集有 **15 个同名 clip**（`Horse_Idle_1.npy`、`Bear_Trot_1.npy` …），裸文件名不能再当 key。
   `data_dict` / `name_list` 的 key 改为 `f"{namespace}/{filename}"`，例如 `truebones/zoo/Horse_Idle_1.npy`。
   entry 内保留 `motion_path`（绝对路径）、`object_type`（规范键）、`source`。
2. **去掉文件名前缀匹配**。[:589](../data_loaders/truebones/data/dataset.py#L589) 与 [:936](../data_loaders/truebones/data/dataset.py#L936) 的 `name.startswith(f'{object_type}_')` 在合并后会让 `Horse_Idle_1.npy` 同时归属两个 Horse。改为：枚举时按 `source` + `species_name` 前缀，归属时直接读 `data_dict[name]['object_type']`。
3. **split 按源各自划分再取并集**。AnyTop 的 split 是「按物种整体留出」，若在并集上全局重算，zoo 现有的 val/test 留出物种会全部改变、历史实验不可比。逐源调用 `ensure_split_manifests(source.root, source.motion_dir)`，结果并集后转成复合 clip id。
   > 注意：只要传了 `--action_tags`（`train.bat` 传了），[dataset.py:425-478](../data_loaders/truebones/data/dataset.py#L425-L478) 会无视 `train.txt` 现算 split 并覆写该文件 —— 这条路径同样按源独立执行。
4. `cache/motion_lengths.npy` 保持**每源一份**，key 仍是裸文件名（源内唯一）。
5. `motion_metadata.json` / `action_tags.jsonl` 按源分别加载，join 时用裸文件名。

### 5.4 采样权重

qualified 之后 `truebones/zoo/Horse` 与 `truebones/zoo_upgrade/Horse` 是两个独立物种，`TruebonesSampler` 各给一份 `sqrt(clip数)` 质量 —— 这符合「两个不同骨架就是两个物种」的语义（实测 zoo Horse 79 关节 / upgrade Horse 39 关节，`scale_factor` 差近一倍）。

**不引入任何按数据集的权重调节**：采样权重只由每个物种在当前训练集中的 clip 数决定，与它来自哪个数据集无关。`TruebonesSampler` 的现有 `sqrt(clip数)` 逻辑不变，只是物种身份从裸名换成规范键。

### 5.5 `dataset_tags`

[dataset_tags.py](../data_loaders/truebones/truebones_utils/dataset_tags.py) 目前是「单目录、进程级单例」。改造：

- `configure(sources=[...])` —— 逐源读各自的 `species_tags.jsonl` / `chain_forward_joints.jsonl`，合并成一个按**规范键**索引的快照
- 跨源同名不再抛 duplicate（源内仍抛）
- `object_subsets` / `species_for()` 返回规范键
- `chain_forward_joints` 的关节索引是**该数据集塌缩顺序专属**的，必须按规范键存取，绝不能跨源按裸名共享
- `worker_initargs()` 改为多源形式
- `using_dataset_dir()` 保留（单源 = N=1），预处理链路不受影响

### 5.6 训练启动

`run_training()` 中：

```python
shutil.copy2(args.cond_path, os.path.join(save_dir, 'cond.npy'))
```

始终执行（覆盖），并把原始 `cond_path` 记进 `args.json`。这样 `save_dir` 自带完整推理契约。

**不改**：预处理链路（`preprocess_and_validate.py` / `dataset_pipeline.py` / `process_new_skeleton.py`）保持单数据集语义。

---

## 6. 推理侧改造

### 6.1 自足

`save_dir/cond.npy` 即完整推理契约。已核实纯生成路径（`--object_type X`，含 `--reference_motion` 指向用户自己的文件）**不读数据集任何其它文件**：模型条件、BVH 导出的 `canonical_bvh_joint_names`/offsets/parents、`species_emb`、`--object_type all` 的物种列表，全部来自 cond entry。

### 6.2 `dataset_tags` 的 cond 回退

新增 `configure_from_cond(cond_dict)`：从每条 entry 的 `species_tags` 字段构建快照。让以下路径在**没有数据集目录**时也能跑：

- [server/anytop_service.py:554-590](../../server/anytop_service.py#L554) `_registered_species_tags` / `_species_tags_registered`
- [utils/skeleton_similarity.py:235](../utils/skeleton_similarity.py#L235) 跨物种 retarget 的 lineage-tag 折扣
- `process_new_skeleton` 判定 object_subset

### 6.3 消除硬编码的「默认数据集」回退

| 位置 | 现状 | 改为 |
|---|---|---|
| [generate.py:266-279](../sample/generate.py#L266-L279) `_load_default_cond_cache` | 回读 `DEFAULT_DATASET_DIR/cond.npy` 找 retarget 源物种 | 回读 **checkpoint 同目录的 `cond.npy`** |
| [dataset_pipeline.py:902-949](../data_loaders/truebones/truebones_utils/dataset_pipeline.py#L902) `_inherit_canonical_stats_from_dataset` | 从 `DEFAULT_DATASET_DIR/cond.npy` 继承 subset 统计量 | 参数改为直接收 cond.npy 路径，默认 checkpoint 同目录 |

### 6.4 输出命名

`_export_motion` 及 retarget 中间产物改用 §2.3 的 `species_file_token`。

---

## 7. `eval` / `--score` 改造

保持「既能单个数据集、也能清单」：

- `eval/evaluate_motion_quality.py` 的 `--dataset_root` 按扩展名判定 —— 目录 = 单源（当前行为不变），`.jsonl` = 清单多源
- `DistributionMotionQualityScorer(dataset_root=...)` 内部统一转成 `sources` 列表
- [offline_reference_dataset.py](../data_loaders/truebones/offline_reference_dataset.py) 提供多源版本；`_matches_object_subset` 的文件名前缀匹配改为读 `motion_metadata`
- [reference_bank.py](../eval/motion_quality/reference_bank.py) 参考池 = 各源并集，clip id 复合化，缓存 key 含全部源路径
- `--eval_during_training` 直接复用训练 cond 派生的 `opt.sources`，不需额外传参

> 参考分布必须从真实 clip 现算，这是**唯一无法用 cond 快照替代**的数据集依赖。

---

## 8. 其它改造点

| 位置 | 改动 |
|---|---|
| [utils/retarget_cache.py:26-29](../utils/retarget_cache.py#L26-L29) | `_CACHE_DIR` 从 `dataset/truebones/zoo/truebones_processed/cache/retarget/` 移到 `<Anytop>/cache/retarget/`。缓存 key 是 prompt+messages 的 SHA-256（含骨骼名与长度），与物种名无关，迁移无冲突风险；可保留旧目录做一次性只读回退 |
| `param_utils.DEFAULT_DATASET_DIR` | 保留，仅作预处理/工具的默认值，不再被训练/推理引用 |
| `tools/check_bone_length_drift.py`、`tools/restore_glb_from_npy.py`、`tools/simulate_corrupted_motion.py`、`tools/visualize_joint_name_embeddings.py`、`tools/extract_action_categories.py` | 硬编码 cond 路径 → 统一 `--cond-path`（默认 checkpoint 同目录 cond.npy） |
| [utils/validate_anytop_dataset.py](../utils/validate_anytop_dataset.py) | 新增 `--datasets`，逐源循环校验（单源行为不变） |
| `utils/auto_retarget.py` | `auto_retarget_pipeline` / `rank_donors` 已移除，donor 读 `motions/` 的依赖随之消失 —— 无需改动 |
| `data_bridge/restore_glb_from_anytop.py` | 离线工具，读 `tpose_reference_paths.jsonl` + 原始 mesh，仍按单数据集目录运行，不改 |
| `train.bat` | 增加 `--cond_path` |

---

## 9. 实施阶段

| 阶段 | 内容 | 产出 |
|---|---|---|
| **P0** ✅ | cond schema v4：预处理写入 `dataset_namespace` / `species_name` / `species_tags`，key 改规范键 | 两个数据集的 cond.npy 已是 v4 |
| **P1** ✅ | `resolve_species_key` + `species_file_token` + `infer_object_type_from_filename` 反解 + `dataset_tags.configure_from_cond` | 推理自足；单数据集回归 |
| **P2** ✅ | `tools/merge_dataset_cond.py`（含统计量重算） | `dataset/merged/cond.npy`，104 物种 |
| **P3** ✅ | 训练侧多源加载（复合 clip id、去前缀匹配、逐源 split、`dataset_tags.configure(sources)`） | 合并训练集 1402 clip 可加载 |
| **P4** ✅ | `--cond_path` 必填 + `save_dir` cond 快照 + 推理侧默认回退改造 | 端到端自足 |
| **P5** ✅ | `eval / --score` 清单支持 | 多源打分 |
| **P6** ⬜ | 全量重预处理 + 统计量重算 + 重训 | 新 checkpoint |

P0–P1 是纯格式与解析层，可独立验证；P3 风险最高（改动 dataset.py 核心索引）。

**P0 的落地方式与原设计略有不同**：预处理链路内部仍用裸名（`不改预处理链路` 的要求），
v4 打戳集中在**唯一的落盘点** `_save_cond_with_tpose_sidecar` / `regenerate_dataset_artifacts` 的写出，
读侧一律走 `cond_schema.load_cond`（旧文件在内存里自动升级）。
这样 `_merge_object_into_cond`、增量预处理、`process_new_skeleton` 都不需要理解 namespace。

---

## 10. 回归验证清单

**单数据集模式必须与改造前逐项等价：**

验证方式：在 HEAD 的 git worktree 里用改造前的代码跑同一份数据（cond 临时降级回裸名键），
导出 `name_list` / `pointer` / `length_arr` / sampler 权重与新代码逐项比对。

- [x] `train.txt` / `val.txt` / `test.txt` 内容完全一致（三份 diff 为空）
- [x] `MotionDataset.name_list` 顺序、`pointer`、`length_arr` 一致（268 clip 全等）
- [x] `TruebonesSampler` 权重向量逐元素一致（maxdiff = 0.0）
- [x] 生成输出文件名与今天一致（`Horse_0.npy`）
- [x] `--object_type Horse` / `--objects_subset quadruped` 行为不变
- [x] 现有测试迁移后全绿（293 passed / 4 skipped）

**合并模式：**

- [x] 15 个同名 clip 全部被独立加载，无覆盖、无丢失
- [x] 10 组重名物种各自持有正确的 parents/offsets
- [x] `Horse` 裸名解析到清单首个源；`zoo_upgrade/Horse` 精确命中
- [x] 生成的 BVH 关节数与对应源一致（zoo Horse 79 / upgrade Horse 39）
- [x] 每 object_subset 的统计量在合并 cond 中唯一

---

## 11. 决策记录

以下五点已确认，实施时按此执行，无遗留待议项。

| # | 决策 | 影响 |
|---|---|---|
| D1 | **不做按数据集的采样权重调节** | 清单无 `weight` 字段，cond 无 `dataset_weight` 字段；`TruebonesSampler` 的 `sqrt(clip数)` 逻辑原样保留（§5.4） |
| D2 | **cond key 一律为规范键**，单数据集也不例外（`Horse` → `truebones/zoo/Horse`） | CLI 与输出文件名靠裸名解析 + 文件名 token 保持不变；但**任何直接写 `cond['Horse']` 的外部脚本会失效**，需随 P0 一并排查改造（§2.1 / §2.3） |
| D3 | **`dataset_root` 存 Anytop-root-relative 路径，不提供 override 逃生口** | 换机器训练需要 Anytop 目录下相同的数据集结构；合并 cond 不做跨机器路径重映射（§3） |
| D4 | **统计量重算默认开启** | `tools/merge_dataset_cond.py` 默认遍历全部源 clip 重算 per-object_subset 统计量（实测 ~1400 clip，约数分钟）；`--no-recompute-stats` 仅作快速验证管线的逃生口（§4.2） |
| D5 | **namespace 不得是另一个 namespace 的路径前缀** | 清单加载时校验：同时出现 `truebones` 与 `truebones/zoo` 直接报错，保证 §2.2 的后缀匹配无歧义 |

### D2 的排查范围

P0 阶段需要把所有「按裸名直接索引 cond」的写法改为走 `resolve_species_key`：

- `sample/generate.py` 的 `_lookup_object_type_case_insensitive` / `_build_retarget_cond_dict`
- `eval/motion_quality/reference_bank.py` 的 `_resolve_lookup_key`
- `tools/check_bone_length_drift.py`、`tools/restore_glb_from_npy.py`、`tools/simulate_corrupted_motion.py`、`tools/visualize_joint_name_embeddings.py`
- `tests/` 中以 `cond['Horse']` / `cond['Dragon']` 等字面量取 entry 的用例
- `data_bridge/restore_glb_from_anytop.py` 中按 `object_type` 查 `tpose_reference_paths.jsonl` 的一段（sidecar 的 key 仍是裸名，需用 `species_name` 去 join）
