# 删除 global_energy

> 状态：**代码已删除（2026-08-31），待重训验证**。本文从
> `action_cond_film_and_energy_removal.md`（**已删除**）拆出——原文的 action FiLM
> 部分已被探针证伪并废弃，但删除 global_energy 的理由是**结构性的、独立成立**，与 action
> 条件通路的结论无关，所以单独立档。
> **cond.npy 无需重生成**（global_energy 是运行时计算的，不进 cond）。**需要全量重训。**
> 与 [canonical_frame_and_label_transfer.md](canonical_frame_and_label_transfer.md) 互相独立，
> 但两者都要重训，建议合并到同一次训练里做（各自的判据不冲突，见 §4）。
> 实施记录见 §6；剩余步骤只有"重训 + 与 v2（energy-off）对采样质量"。

---

## 0. 为什么删

### 0.1 它是目标泄漏

`global_energy` 由 x₀ 的速度／旋转差分**确定性算出**
（[dataset.py:200](../data_loaders/truebones/data/dataset.py#L200)、
[gaussian_diffusion.py:1587](../diffusion/gaussian_diffusion.py#L1587)）。
它不是一个用户意图，是答案的一个函数。训练期把它喂进去，模型学到的是
"给定这段动作的能量，重建这段动作"，而不是"给定这个标签，生成一段动作"。

### 0.2 它在结构上碾压 action 通路

| | action_label | global_energy |
|---|---|---|
| 注入形式 | 加性 token，加到 `timesteps_emb`（[anytop.py:766](../model/anytop.py#L766)） | **乘性 FiLM，逐层**（[motion_transformer.py:1207](../model/motion_transformer.py#L1207)） |
| 与目标的关系 | 多对一语义描述，噪声大 | **由 x₀ 确定性算出**（目标泄漏） |
| CFG drop | 0.2 | 0.3 |

推理端**没有任何 CFG**：`*_cfg_drop_prob` 只在训练期造 null 路径，采样端没有
`(1+w)ε_cond − w·ε_uncond` 外推，也没有 `ClassifierFreeSampleModel` 包装，模型裸传给
`p_sample_loop`（[generate.py:802](../sample/generate.py#L802)）。
所以 action 通道没有任何放大机制，energy FiLM 却是全强度逐层作用。

> 注：探针 A/B/C（2026-08-31）已证伪"species 通道压制 action 通道"这个假设，
> 但那组探针是在 **energy 全程关闭**下跑的，对本节的 energy↔action 不对称**不构成反证**。

### 0.3 `norm_ref` 是最硬的伤

```python
modulated = self._apply_global_energy_cond(self.norm_ref(x), global_energy_condition)
x = torch.where(active, modulated, x)     # 注意是替换，不是 x + ...
```

- energy 激活时残差流**每层被 LayerNorm 重新归一化**再乘 `(1+tanh γ)`。加性 action token 的
  条件强度体现在它在残差流里的**幅度**上——过一次 `norm_ref`，幅度被抹平，只剩方向。8 层做 8 次。
- γ 虽零初始化，但 `norm_ref` 在 FiLM 外面，所以 γ=0 时 `modulated = norm_ref(x) ≠ x`。
  **这不是恒等起点，是硬架构分叉**：`energy_active` 的样本走"带 LayerNorm"的子网，
  dropped 的走"不带"的，两条子网共享权重。drop=0.3 即 30/70 分裂。
- `norm_ref` 是**无条件创建**的（[motion_transformer.py:1076](../model/motion_transformer.py#L1076)），
  删掉 energy 后它变成死权重，应一并 `del`（同 `self_attn` / `multihead_attn` 的处理）。

### 0.4 CLI 允许采样边缘分布的方框

训练分布是很窄的 `p(energy | action, species)`（idle 与 gallop 的能量几乎不重叠），
而 `--global_energy` 是对**全局** running mean/std 的 z-score
（[generate.py:257](../sample/generate.py#L257)），CLI 允许在边缘分布的方框里任选。
`idle + energy=2.0` 训练中从未出现，却是一个合法的命令行。

---

## 1. 删除清单（约 187 处引用，排除 `outputs/`）

| 文件 | 引用数 | 要点 |
|---|---|---|
| [anytop.py](../model/anytop.py) | 74 | `global_energy_projection`（[117](../model/anytop.py#L117)）、3 个 running-stats buffer（[122-124](../model/anytop.py#L122)）、`_update_global_energy_running_stats`（[222](../model/anytop.py#L222)）／`_coerce_global_energy_condition`（[265](../model/anytop.py#L265)）／`_build_global_energy_token`（[300](../model/anytop.py#L300)）、`GlobalEnergyExtractor`（[950](../model/anytop.py#L950)） |
| [motion_transformer.py](../model/motion_transformer.py) | 33 | `global_energy_film`（[1072](../model/motion_transformer.py#L1072)）、`_apply_global_energy_cond`（[1132](../model/motion_transformer.py#L1132)）、forward 里的应用块（[1205-1210](../model/motion_transformer.py#L1205)）+ **`del self.norm_ref`**（[1076](../model/motion_transformer.py#L1076)） |
| [generate.py](../sample/generate.py) | 30 | `--global_energy`、`resolve_global_energy_condition`（[257](../sample/generate.py#L257)）、`_compute_global_energy_from_reference`（[277](../sample/generate.py#L277)）、参考动作自动提取 |
| [gaussian_diffusion.py](../diffusion/gaussian_diffusion.py) | 21 | `_build_global_energy_conditioning`（[1587](../diffusion/gaussian_diffusion.py#L1587)）整个方法 + 其调用 |
| [tensors.py](../data_loaders/tensors.py) | 7 | collate 里的 key |
| [model_util.py](../utils/model_util.py) | 5 | `model_supports_global_energy_conditioning`（[64](../utils/model_util.py#L64)）、两个 arg 透传（[110-111](../utils/model_util.py#L110)） |
| [parser_util.py](../utils/parser_util.py) | 3 | 两个训练 flag + 一个采样 flag |
| [dataset.py](../data_loaders/truebones/data/dataset.py) | 3 | `_compute_global_energy_condition_np` 及其写入 |
| tests | 11 | `test_dataset_loop.py`(6) / `test_native_loop.py`(5) |

### 唯一真正丢的功能

带参考动作时从参考中自动提取能量（`_compute_global_energy_from_reference`,
[generate.py:277](../sample/generate.py#L277)）。实施前需确认该路径当前是否在用；
若在用，等价替代是直接用参考动作做 `skip_timesteps` 初始化，那条路径已经存在。

---

## 2. 必须加的守卫

仿 `assert_action_conditioning_not_deprecated`（[parser_util.py:35](../utils/parser_util.py#L35)），
`args.json` 里出现 `global_energy_cond` / `global_energy_cfg_drop_prob` 就 `SystemExit`。

否则旧 checkpoint 会**静默加载**：`model_supports_global_energy_conditioning` 返回 False，
条件被悄悄关掉，表现为"质量退化"而不是"不兼容"。这个仓库已有先例——
`action_tag_cond` 那次就是靠这条守卫才没有变成一次误诊。

---

## 3. 与旧 checkpoint 对照的口子

旧模型的 energy hard-null 路径是训练过的：`global_energy_active=False` 时完全 bypass
`norm_ref` + FiLM，按设计注释是 "byte-identical to a `global_energy_cond=False` model"
（[motion_transformer.py:1210](../model/motion_transformer.py#L1210)）。

所以**可以把 v2 强制切到 energy-off 状态**再与新模型比较，这个比较成立。
所有与旧 checkpoint 的对照都必须走这个口子，否则比的是两个不同的子网。

---

## 4. 期望要放对：loss 不能作为判据

energy 是从 x₀ 直接算出的目标泄漏。删掉它 `l_simple` **必然上升**——
**loss 变差恰恰是这个改动正确的预期结果。用训练 loss 对比会得出完全相反的结论。**

权重空间对比同样无意义：架构已变（移除 `global_energy_projection` / `global_energy_film` /
`norm_ref`），两个 state dict 没有共享参数空间。

判据只能是**采样质量**：用 [eval/motion_quality/scorer.py](../eval/motion_quality/scorer.py)
的 `spectral_flatness` / `jerk_norm` / `snap_norm` + bone-length drift 对 reference bank 打分，
与 v2（energy-off，§3）对照。需要采样出 NPY，但不需要 BVH 导出、不需要渲染、不需要人看。

---

## 5. 实施顺序

1. 确认 `_compute_global_energy_from_reference` 是否仍有调用方（§1 末）。
2. 一次删干净，含 `del self.norm_ref` 和 §2 的守卫。
3. 重训。若与
   [canonical_frame_and_label_transfer.md](canonical_frame_and_label_transfer.md)
   合并做，两个改动的判据互不冲突：本方案看采样质量，那边看骨长漂移与跨 subset 迁移，
   都不看 loss 绝对值。
4. 与 v2（energy-off）对照。

---

## 6. 实施记录（2026-08-31）

### 已删除

| 文件 | 删掉了什么 |
|---|---|
| `model/anytop.py` | `global_energy_cond` / `global_energy_cfg_drop_prob` / `global_energy_stats_momentum` 三个属性、`global_energy_projection`、3 个 running-stats buffer、`_update_global_energy_running_stats` / `_coerce_global_energy_condition` / `_build_global_energy_token` / `_coerce_energy_active` 四个方法、forward 里的构造块与两个下传 kwarg、`GlobalEnergyExtractor` 整个类（−380 行） |
| `model/motion_transformer.py` | `GraphMotionDecoderLayer.__init__` 的 `global_energy_cond` 形参、`global_energy_film`、**`self.norm_ref`（§0.3）**、`_apply_global_energy_cond`、layer forward 的应用块、两级 forward 签名里的 `global_energy_condition` / `global_energy_active`（−63 行） |
| `diffusion/gaussian_diffusion.py` | `GlobalEnergyExtractor` import、`_build_global_energy_conditioning` 整个方法及 `training_losses` 里的调用（−59 行） |
| `sample/generate.py` | `--global_energy` 解析链：`resolve_global_energy_condition`、`_compute_global_energy_from_reference`、参考动作自动提取块、`_generate_all_species` 的形参与写入、两处 `model_kwargs['y']['global_energy_cond']`；**连带删掉只为能量提取存在的 `physical_energy_features` / `reference_physical_motion` 管线**（−86 行） |
| `data_loaders/tensors.py` | collate 里的 `global_energy_cond` 分支 + 两处 key 列表 |
| `data_loaders/truebones/data/dataset.py` | `_compute_global_energy_condition_np` 及其调用与 metadata 写入 |
| `utils/model_util.py` | `model_supports_global_energy_conditioning`、`get_gmdm_args` 的两个透传 |
| `utils/parser_util.py` | 两个训练 flag + 一个采样 flag |
| `train/training_loop.py` | spike-capture 的 `global_energy_cond` 记录 + 两处注释 |
| `tests/` | `test_dataset_loop.py` 的两个能量探针测试、`test_native_loop.py` 的 `global_energy_cond` collate 测试与 fixture 形参、`test_generate_inpainting.py` 的 `physical_energy_features` 实参 |
| `train.bat` / `eval/eval_tasks.json` / `client/anytop_client.py` / `server/anytop_service.py` / `server/serve.py` | CLI/服务层的 flag 透传（服务端会把 payload 里的 `global_energy` 转成 `--global_energy`，不删就会给 generate.py 传未知参数） |

### §1 末的确认

`_compute_global_energy_from_reference` **有一个调用方**（`generate.py` 参考动作分支），
所以那条"唯一真正丢的功能"确实在用，现已删除。等价替代按原文：直接用参考动作做
`skip_timesteps` 初始化，该路径本来就在。

`generate.py` 里为它服务的 `physical_energy_features` / `reference_physical_motion`
管线（`_prepare_img2img_reference_bundle` 的必填 kwarg + 快照 + 返回值）删掉后无人读取，
一并移除。

### §2 守卫

`utils/parser_util.py:assert_global_energy_not_deprecated`，由 `extract_args` 调用。
判据是 **key 是否出现**，不看值——因为 `norm_ref` 是**无条件创建**的（§0.3），
`global_energy_cond: false` 的 checkpoint 同样带着 `norm_ref.{weight,bias}`，
同样不是权重兼容的。已对 `save/merged_locomotion_v2` 与 `save/quadropeds_final_v1`
的 args.json 实测触发 SystemExit。

### 验证

- 全量 pytest：386 passed, 15 subtests passed（含 `test_action_label_cond` /
  `test_species_cond_hybrid` / `test_diffusion_loss_precision` 三个真正跑 AnyTop forward 的套件）
- `test_dataset_loop.py` 独立回归脚本：all regression checks passed
- 构建 AnyTop 后 `named_parameters() + named_buffers()` 中 `energy` / `norm_ref` 命中数为 0

### 剩余

§5 的第 3、4 步：重训，然后与 v2（按 §3 强制切到 energy-off）对采样质量。
**判据按 §4：不看 loss。**
