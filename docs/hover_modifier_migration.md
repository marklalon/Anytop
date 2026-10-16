# hover 修饰词迁移（2026-10-16）

`hover` 表示身体保持悬停的状态，归入 `MODIFIER_VOCAB`，不再占用 head 通道。
主词仍表达正在做的动作；飞行移动用 `fly`，悬停待机用 `idle, hover`。
这也适用于幽灵和漂浮角色，无须引入飞行语义。

| 原标注 | 新标注 |
|---|---|
| `hover` | `idle, hover` |
| `hover, look` | `idle, hover, look` |
| `hover, turn, left` | `turn, left, hover` |
| `hover, rear` | `rear, hover` |
| `attack, hover` | `attack, hover` |
| `attack, hover, right, swat` | `attack, right, hover, swat` |

迁移规则：含 hover 且移除其主词身份后没有主词的标签补上 idle；已有其他主词则保留。
所有含 hover 的标签按统一规范排列方向和修饰词。其余字段保持原值，包括 action_group、
is_loop、reviewed；这是标签表示迁移，无须重新判断动作或循环。review/llm 下的历史草稿不改写。

四份正式标注共 4,981 行，其中 256 行含 hover，121 行文字变化：

| 数据集 | 原 hover 首位 | 另需调整方向词序 | 修改行数 |
|---|---:|---:|---:|
| integrate_20260924/processed | 48 | 6 | 54 |
| truebones/zoo/truebones_processed | 13 | 0 | 13 |
| truebones/zoo_upgrade/clean_processed | 0 | 0 | 0 |
| unitybundles/processed | 46 | 8 | 54 |

hover 从 `PHASE_FREE_LOOP_HEADS` 移到 `PHASE_FREE_LOOP_MODIFIERS`：`idle, hover`
仍允许循环相位平移和拼接；`attack, hover`、`idle, hover, look` 仍有事件相位锚点。
方向豁免和空中死亡规则仍按词匹配 hover，不受槽位变化影响。

解析契约版本升到 8；总词数不变（92），head/modifier/direction 的源秩分别为 32/54/6。
词表位置变化，必须用 `python tools/build_action_label_embeddings.py` 重建全局
`dataset/action_word_embeddings.npy`。词槽语义变化要求重训，旧 checkpoint 不兼容新条件契约。
评测任务中的裸 hover 同步替换为 idle, hover。需要通过
`python tools/regenerate_dataset_artifacts.py` 重建 cond.npy，更新其中按完整标签存储的
clip_length_prior；旧表中的 hover 无法匹配新的 idle, hover。动作特征无需因本次标签迁移重新预处理。

验收重点：同一动作添加 hover 后 head 通道保持不变、hover 仅进入 modifier；
纯 hover 不再是合法完整标签；多主词权重与词序测试改用 jump，仍验证原有组合能力。
模型质量需用重训后的悬停待机、悬停攻击和飞行移动对照评估。

本地验证：4,981 行全部通过解析与加载，含 hover 的标签均为规范词序；对比 Git HEAD，
仅 121 行的 action_label 有变化，所有其他字段保持一致。重建后的词表可正常加载且槽源满秩。
相关测试 253 项及 27 个子测试通过。完整 test_dataset_loop.py 最初有 11 项因加载后的
data_dict 不包含 truebones/zoo/Ostrich_Run.npy 失败。后续确认 Ostrich 的 Winged/Flapping
标签使其不属于测试指定的 biped 子集；将该循环夹具改为直接选择 Ostrich 后，
整个测试文件 30 项全部通过，无须重跑动作预处理。
