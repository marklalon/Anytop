"""``skeleton_profiles_report.md``: what a person has to look at after a build."""

from __future__ import annotations

SECTIONS = (
    ("decode_failures", "解码失败的 clip"),
    ("override_notes", "覆盖文件问题"),
    ("rare_contacts", "几乎不着地的接触关节（locomotion 帧中接触占比过低）"),
    ("symmetry_axis", "左右镜像轴明显不一致（≥ 30°，可能是骨架左右定义的问题）"),
    ("name_dof", "名字与自由度冲突（名字像膝/肘，但不是 hinge）"),
)


def _chain_line(row: dict) -> str:
    joints = row["joints"]
    span = joints[0] if len(joints) == 1 else f"{joints[0]} → {joints[-1]}"
    notes = [f"{row['role']}，{len(joints)} 个关节"]
    if row["fit_passed"]:
        notes.append("拟合通过：" + "、".join(row["fit_passed"]))
    if row["passive"]:
        notes.append("passive：" + "、".join(row["passive"]))
    return f"{span}（{'；'.join(notes)}）"


def render_report(namespace: str, findings: dict[str, dict]) -> str:
    lines = [f"# Skeleton Profile 报告：{namespace}", "",
             "\"不稳定\"和\"低置信度\"只给数量：它们反映该物种的 clip 少、动作族单一，"
             "已经体现在关节的 confidence 里，逐关节的原因见 skeleton_profiles.json 中关节的 unstable 字段。"
             "左右一致性只在 locomotion clip 上检查；表中\"左右不一致\"是 明显/全部，"
             "下面只列出镜像轴差 ≥ 30° 的，其余见 skeleton_profiles.json 的 findings。", ""]
    lines.append("| 物种 | clip 数 | 候选部位 / passive 关节 | 左右不一致 | 名字冲突 | 不稳定 | 低置信度 | 其他 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for species in sorted(findings):
        f = findings[species]
        other = len(f.get("decode_failures", [])) + len(f.get("override_notes", [])) + len(f.get("rare_contacts", []))
        if f.get("fallback"):
            other_text = f"{other}（无动作，回退）"
        else:
            other_text = str(other)
        lines.append(
            f"| {species} | {f.get('clips', 0)} | {len(f.get('passive_chains', []))} / "
            f"{sum(len(c['passive']) for c in f.get('passive_chains', []))} | "
            f"{len(f.get('symmetry_axis', []))}/{len(f.get('symmetry', []))} | {len(f.get('name_dof', []))} | "
            f"{len(f.get('unstable', []))} | {len(f.get('low_confidence', []))} | {other_text} |")
    lines.append("")
    for species in sorted(findings):
        f = findings[species]
        blocks = []
        for key, title in SECTIONS:
            items = f.get(key) or []
            if not items:
                continue
            blocks.append(f"**{title}**")
            blocks.append("")
            for item in items:
                blocks.append(f"- {item}")
            blocks.append("")
        chains = f.get("passive_chains") or []
        if chains:
            blocks.append("**次级运动候选**（每行一个挂在身体上的候选子树；名字是毛发、耳、衣物一类的连同子树默认 passive，尾巴归 tail_weight；"
                          "增删写进 passive_overrides.json 的 add / remove，或在微调 UI 上点选；"
                          "\"拟合通过\"表示数据里有被身体带动的运动，其余用默认弹簧参数）")
            blocks.append("")
            for row in chains:
                blocks.append(f"- {_chain_line(row)}")
            blocks.append("")
        if f.get("fallback"):
            blocks.insert(0, "该物种没有动作数据，Profile 由同 species_tags 物种按规范关节名回退得到（confidence = 0）。\n")
        if not blocks:
            continue
        lines.append(f"## {species}")
        lines.append("")
        lines.extend(blocks)
    return "\n".join(lines).rstrip() + "\n"
