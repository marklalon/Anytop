# 训练样本过滤器

`--sample_filter configs/sample_filters/tiny.json` 用一个 JSON 文件组合训练样本。
`train_tiny.bat` 已迁移到该配置，保留原来的两个 namespace、Winged、locomotion 范围。

## 规则

- `include` 必填，是非空规则列表；任意一条命中即可加入。`"include": [{}]` 选择全部候选。
- `exclude` 可省略；任意一条命中即排除，排除始终优先。
- 一条规则的所有字段必须同时满足。普通列表字段匹配任意一个值。
- 并集按 clip ID 去重，不因重复命中提高采样权重。
- 只在当前 cond 和 `--train_split` 中选择，不能通过精确 clip 绕过 split。

| 字段 | 语义 |
| --- | --- |
| `namespaces` | 精确 namespace；`truebones/zoo` 不含 `truebones/zoo_upgrade` |
| `namespace_prefixes` | namespace 本身及其 `/` 分隔的子 namespace |
| `object_types` | 精确规范 key，例如 `truebones/zoo/Bat`，不接受模糊物种简称 |
| `species_tags_any` | 至少具有一个所列标签 |
| `species_tags_all` | 具有全部所列标签，允许其它标签 |
| `species_tags_none` | 不具有任何所列标签 |
| `species_tags_exact` | 标签集合相等，忽略顺序 |
| `action_groups` | `locomotion`、`stationary`、`transition` 中任意一个，不写 `all` |
| `heads_any / heads_all / heads_none` | 动作标签的全部主词中，包含任一个／全部／一个都不含 |
| `clips` | 精确 `<namespace>/<filename>.npy` |
| `clip_globs` | 显式通配符匹配完整 clip ID，支持 `*`、`?`、`[...]` |

所有字段值均为非空字符串列表。species_tags 忽略大小写，其余值区分大小写。
动作主词复用现有词表和标签解析，不读取文件名猜测动作，不做子串匹配；多主词标签的任一主词都可被 `heads_any` 命中。
`tags_all`、`species` 等旧方案名称不是别名，会报未知字段。

## 常用组合

当前 tiny 去掉 walk/run，再加 unitybundles 的 shake：

```json
{
  "include": [
    {
      "namespaces": ["truebones/zoo", "integrate_20260924"],
      "species_tags_all": ["Winged"],
      "action_groups": ["locomotion"],
      "heads_none": ["walk", "run"]
    },
    {
      "namespaces": ["unitybundles"],
      "heads_any": ["shake"]
    }
  ],
  "exclude": []
}
```

只训练 fly：把第一条的 `heads_none` 换成 `"heads_any": ["fly"]`，并去掉第二条。
仅包含 Winged、不能有其它标签：使用 `"species_tags_exact": ["Winged"]`。
有 Winged 但不能有 Fluttering：同一条规则中写 `species_tags_all: ["Winged"]` 和 `species_tags_none: ["Fluttering"]`。

精确增加 clip：在 include 新增 `{"clips": ["<namespace>/<filename>.npy"]}`。
精确删除 clip：在 exclude 新增同样的规则。若只想删除某个数据集中的 run，可在 exclude 的同一条规则中同时写 namespaces 和 heads_any。

## 预览

在 Anytop 目录执行：

```bat
..\.venv\Scripts\python.exe tools/preview_sample_filter.py ^
  --cond_path dataset/cond_all.npy ^
  --sample_filter configs/sample_filters/tiny.json ^
  --output "%TEMP%\tiny-selection.json"
```

预览不初始化模型，不写训练缓存。输出每条规则命中数、候选/命中/长度不合格/最终样本数，以及按数据集、object_type、动作组、主词的分布。
`--output` 保存包含最终 clip ID、标签、长度的完整 JSON 清单。
`--split`、`--min_length`、`--sample_limit`、`--num_frames`、`--seed` 应与训练一致。
先排除短于 min_length 的 clip；sample_limit 优先从长度不小于 num_frames 的池中选取，不足时退回全部合格 clip，再按 seed 随机限量。训练和预览使用同一实现。

未知字段、动作词、namespace、object_type、species tag 或不存在的精确 clip 直接报错。
通配符和合法组合零命中会警告；最终无可训练样本报错。选中文件损坏会报错，不会静默丢弃。
精确 clip 存在于其它 split 时，只验证其存在，不把它移入当前 split。

## 训练、验证和续训

- 训练入口已删除 `--dataset_namespaces`、`--objects_subset`、`--action_group`。数据来源、物种和动作组合统一在过滤文件中配置。
- `--sample_filter` 可省略，等价于 `{"include": [{}]}`，默认选择当前 cond 和 train_split 中的全部合格样本。
- 最终训练池只有一个动作组时，将该组写入 checkpoint args.json；跨组时写 `all`，同时记录 `effective_action_groups`。这些字段仅记录训练语料，推理不读取它们，长度先验统一跨组匹配标签。采样权重仍由原有机制管理。
- 验证集不继承训练过滤文件或训练池的动作组。可单独指定 `--val_sample_filter`；未指定时使用 val split 的全部合格样本。
- 保存 `sample_filter.json`（规范化规则）与 `sample_selection.json`（规则、split、最终 clip ID / object_type / 标签 / 长度及指纹）。该指纹不对 motion 文件内容做完整哈希。
- 包括默认全量训练在内，每次训练均保存样本快照。续训比较 checkpoint 所在目录的快照；规则或最终样本元数据变化时拒绝续训。旧 run 没有样本快照，迁移请使用新的 run_name，例如 `train_tiny.bat tiny_filter_v1`。

参考配对和稀有主词采样统计都基于筛选后的实际训练池。
