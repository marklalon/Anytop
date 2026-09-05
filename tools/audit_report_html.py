#!/usr/bin/env python3
"""Static HTML review page for :mod:`audit_action_labels` findings.

A finding is only half an answer.  Every rule in the audit reports that two
clips disagree and none of them says which one is wrong -- that is settled by
watching the review GIF.  The text report prints the GIF paths and leaves the
operator to open them one at a time; this page puts the GIFs, the current
labels and an editable field for the corrected one side by side.

The page is one self-contained file with NO server behind it: the findings are
inlined as JSON and every GIF is referenced by an absolute ``file://`` URI, so
it opens straight off disk.  That is also why the GIFs are mounted lazily --
one run can put a few hundred animated GIFs on the page, and decoding them all
at once is what kills the tab (the same trick ``dataset/review/index.html``
uses).

What the operator does here is DECIDE, not apply.  Nothing on this page writes
to ``action_labels.jsonl``.  "复制修复指令" puts one block of text on the
clipboard naming, per sidecar file, the clip, its old label and the new one,
plus the evidence the operator saw; pasting that into an LLM with file access
is what performs the edit.  Every label typed here is checked with the same
rules ``_validate_action_label_entry`` enforces -- controlled vocabulary, word
and head counts, canonical spelling -- so what reaches the clipboard is
something the trainer will accept.  The vocabulary tables are baked into the
page at build time, which means a regenerated report follows a vocabulary
change and an old one on disk keeps documenting the rules it was written under.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from hashlib import sha1
from pathlib import Path

_ANYTOP_DIR = Path(__file__).resolve().parent.parent
for _candidate in (str(_ANYTOP_DIR), str(_ANYTOP_DIR.parent)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    ACTION_GROUPS,
    ACTION_LABEL_MAX_HEADS,
    ACTION_LABEL_MAX_WORDS,
    CONTROLLED_VOCAB,
    DIRECTION_VOCAB,
    STATE_VOCAB,
)

# Written next to the other review front-end so both live in one place. It is a
# generated file: every run overwrites it, and the operator's verdicts survive
# in localStorage rather than in the HTML.
DEFAULT_REPORT_PATH = _ANYTOP_DIR / "dataset" / "review" / "audit_action_labels.html"

# One short line per rule, shown in the page's legend. The rule docstrings in
# audit_action_labels.py are the authority; these say what the operator has to
# decide, which is the part the docstrings do not spell out.
RULE_INFO = {
    "R1": {
        "name": "标签桶离散度",
        "text": "同一 (物种, action_group, 标签) 桶里，最远的一对动作不得超过该物种自己的"
                "中位成对距离。共享标签本身不是缺陷 —— 要判断的是这个标签能不能同时描述"
                "桶里的每一条 clip；不能，就把描述不到的那条改掉。",
    },
    "R3": {
        "name": "镜像一致性",
        "text": "只靠名字配上的左右两条 clip，标签必须互为镜像、各自带自己的方向词，而且"
                "不能左右写反。名字是关于动作最弱的证据 —— 页面给出的候选值是从名字读的，"
                "改哪一边看 GIF 决定。",
    },
    "R4": {
        "name": "方向拼写",
        "text": "每条 locomotion 标签都要写出朝向：平面方向 (forward/backward/left/right)，"
                "或者一个垂直动作 (up/down/dive/fall)。全语料里一条普通前进的走路被写成"
                "'walk' 87 次、'walk, forward' 30 次，不统一就等于噪声。",
    },
    "R5": {
        "name": "步态词冲突",
        "text": "同一条标签里两个词把同一根轴写了两遍，或者其中一个词在另一个词旁边不携带"
                "任何信息。删掉多余的那个，或换成词表里合并后的写法。",
    },
}


# The R3 sentences, in the page's language. Keyed on the finding's
# ``problem_codes`` rather than on its prose so a reworded rule falls back to
# the English sentence instead of quietly losing the translation.
R3_PROBLEMS = {
    "crossed": "左右写反了：名字带 Left 的那条标签写着 right，带 Right 的那条写着 left"
               "（这样的一对自己和自己是自洽的，只有显式检查才抓得到）",
    "not_mirror": "两条标签不是彼此的镜像",
    "no_side_word": "有一侧没写方向词，L/R 这根轴就丢了",
}


def file_uri(path) -> str:
    """``file:///D:/...`` for *path*, or ``''`` when there is no path.

    ``Path.as_uri`` percent-encodes for us, which matters: clip names carry
    ``#`` and spaces, and an unencoded ``#`` would truncate the URL at the
    fragment and silently show a broken image.
    """
    text = str(path or "").strip()
    if not text:
        return ""
    return Path(text).resolve().as_uri()


def _clip_payload(record, role="") -> dict:
    """One clip card's data. *role* marks its part in the finding (e.g. ``L``)."""
    return {
        "clip": record["clip"],
        "species": record["species"],
        "group": record["group"],
        "label": record["label"],
        "gif": file_uri(record.get("gif_path", "")),
        "bvh": record.get("bvhview", ""),
        "labels_path": record.get("labels_path", ""),
        "role": role,
        "suggest": record.get("suggest", ""),
    }


def _lookup(index, species, clip, fallback=None):
    """The collected record for ``(species, clip)``.

    A finding names its clips by file name, and file names are only unique
    *within* a source -- two datasets may both hold ``Dog_Walk.npy``. Keying the
    index by species as well is what keeps a finding from picking up the other
    dataset's clip (and its GIF).
    """
    record = index.get((species, clip))
    if record is not None:
        return record
    # A finding whose clip is not in the index would otherwise drop out of the
    # page entirely; show it without a GIF rather than lose the finding.
    stub = {"clip": clip, "species": species, "group": "", "label": "",
            "gif_path": "", "labels_path": ""}
    stub.update(fallback or {})
    return stub


def _case_key(rule, species, subject) -> str:
    """Stable identity of a finding across runs, for the localStorage verdicts.

    Deliberately built from what the finding is ABOUT (rule, species, and the
    label / mirror base / clip it concerns) rather than its position in the
    list: a re-run with different thresholds reorders the findings, and an
    index-based key would hand every verdict to the wrong case.
    """
    return "%s|%s|%s" % (rule, species, subject)


def build_cases(findings, clips) -> list[dict]:
    """Turn the audit's per-rule findings into uniform review cases.

    A case is one panel on the page: a headline, the problems, the metrics that
    produced it, and the clips it concerns with their GIFs. The rules disagree
    about what a finding looks like (a bucket, a mirror pair, a single clip), so
    the normalisation happens here and the page only ever sees a case.
    """
    index = {(item["species"], item["clip"]): item for item in clips}
    cases = []
    for finding in findings:
        rule = finding["rule"]
        species = finding.get("species", "")
        metrics = []
        note_hint = ""

        if rule == "R1":
            label = finding["label"]
            worst = list(finding.get("worst_pair", []))
            members = []
            for name in finding.get("clips", []):
                record = _lookup(index, species, name,
                                 {"label": label,
                                  "group": finding.get("action_group", "")})
                members.append(_clip_payload(record, "最远一对" if name in worst else ""))
            # The distance is the reason the pair is on the page; printing it
            # keeps the operator from re-deriving how bad "bad" was.
            metrics = [
                {"k": "ratio", "v": "%.2f×" % finding["ratio"]},
                {"k": "最远距离", "v": "%.2f" % finding["max_distance"]},
                {"k": "种内中位", "v": "%.2f" % finding["species_spread"]},
            ]
            subject = label or "(空标签)"
            headline = ("标签 %s 覆盖了 %d 条动作，其中最远的一对相距种内中位距的 %.2f 倍"
                        % (json.dumps(label, ensure_ascii=False), len(members),
                           finding["ratio"]))
            problems = []
            if len(worst) == 2:
                problems.append("最远的一对：%s ↔ %s" % (worst[0], worst[1]))
            problems.append("这个标签描述不了的那条，改成它自己的标签；两条都对就标"
                            "「无需修改」。")
            note_hint = "例如：两条其实是同一个动作的两次录制，保持不变"

        elif rule == "R3":
            subject = finding.get("base", "")
            members = []
            for side, role in (("left", "L"), ("right", "R")):
                entry = finding[side]
                record = _lookup(index, species, entry["clip"],
                                 {"label": entry["label"],
                                  "group": finding.get("action_group", ""),
                                  "gif_path": entry.get("gif", "")})
                record = dict(record)
                candidate = finding.get("candidate_%s" % side, "")
                if candidate and candidate != entry["label"]:
                    record["suggest"] = candidate
                members.append(_clip_payload(record, role))
            headline = "左右镜像对 %s 的两条标签不一致" % subject
            codes = finding.get("problem_codes") or []
            raw = list(finding.get("problems", []))
            problems = [R3_PROBLEMS.get(code, text) for code, text
                        in zip(codes, raw)] or raw
            problems.append("候选值是从 clip 名字读的，没有确认过 —— 以 GIF 为准。")
            note_hint = "例如：名字反了，动作本身是对的"

        elif rule == "R4":
            subject = finding["clip"]
            record = _lookup(index, species, finding["clip"],
                             {"label": finding["label"], "group": "locomotion",
                              "gif_path": finding.get("gif", "")})
            members = [_clip_payload(record)]
            headline = "locomotion 标签 %s 没有写朝向" % json.dumps(
                finding["label"], ensure_ascii=False)
            problems = [finding.get("problem", ""),
                        "补一个平面方向 (forward/backward/left/right)，或者一个垂直动作"
                        " (up/down/dive/fall)。原地不移动的话，它可能根本不属于"
                        " locomotion。"]
            note_hint = "例如：原地转身，应该归到 transition"

        elif rule == "R5":
            subject = finding["clip"]
            record = _lookup(index, species, finding["clip"],
                             {"label": finding["label"]})
            members = [_clip_payload(record)]
            words = finding.get("words") or []
            headline = ("标签 %s 里 %s 这两个词把同一根轴写了两遍"
                        % (json.dumps(finding["label"], ensure_ascii=False),
                           " 和 ".join("'%s'" % word for word in words))
                        if words else
                        "标签 %s 里有互相冲突的词"
                        % json.dumps(finding["label"], ensure_ascii=False))
            problems = [finding.get("advice", "") or finding.get("problem", ""),
                        "删掉多余的那个词，或者换成词表里合并后的写法。"]
            note_hint = ""

        else:  # a rule added to the audit but not taught to this page yet
            subject = finding.get("clip", json.dumps(finding, sort_keys=True)[:40])
            record = _lookup(index, species, finding.get("clip", ""),
                             {"label": finding.get("label", "")})
            members = [_clip_payload(record)]
            headline = finding.get("problem", "见下方原始数据")
            problems = [json.dumps(finding, ensure_ascii=False, sort_keys=True)]

        cases.append({
            "id": _case_key(rule, species, subject),
            "rule": rule,
            "species": species,
            "group": finding.get("action_group", ""),
            "subject": subject,
            "headline": headline,
            "problems": [text for text in problems if text],
            "metrics": metrics,
            "clips": members,
            "note_hint": note_hint,
        })
    return cases


def write_html_report(path, findings, clips, meta) -> Path:
    """Render the review page to *path* and return it."""
    cases = build_cases(findings, clips)
    payload = {
        "meta": dict(meta, generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "rule_info": RULE_INFO,
        "cases": cases,
        "vocab": {
            # The page mirrors canonical_action_label exactly: head words keep
            # the order they were typed (that order is a transition's time
            # order); directions bind after turn when present, or after the
            # complete head sequence otherwise, before the sorted modifiers.
            "order": {word: index for index, word in enumerate(CONTROLLED_VOCAB)},
            "heads": list(STATE_VOCAB),
            "directions": list(DIRECTION_VOCAB),
            "groups": list(ACTION_GROUPS),
            "max_words": ACTION_LABEL_MAX_WORDS,
            "max_heads": ACTION_LABEL_MAX_HEADS,
        },
        # Verdicts are stored per report identity, so re-running the same audit
        # keeps them and a different scope starts clean.
        "store_key": "auditActionLabels." + sha1(
            ("%s|%s" % (meta.get("cond_path", ""), meta.get("action_group", "")))
            .encode("utf-8")).hexdigest()[:10],
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # The blob sits in a <script type="application/json"> block, where the one
    # thing that can end it early is a literal "</script>".
    data = data.replace("<", "\\u003c")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE.replace("/*__REPORT_DATA__*/", data),
                    encoding="utf-8")
    return path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>action_labels 审计核验</title>
<style>
  :root {
    --bg: #303030;
    --card: #3a3a3a;
    --card-edge: #4a4a4a;
    --panel: #343434;
    --text: #e6e6e6;
    --dim: #9a9a9a;
    --accent: #6fa8ff;
    --ok: #4caf50;
    --warn: #e0704a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 13px/1.45 "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
  }

  header {
    position: sticky;
    top: 0;
    z-index: 10;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    background: #262626;
    border-bottom: 1px solid #1c1c1c;
    box-shadow: 0 2px 8px rgba(0,0,0,.4);
  }
  .spacer { flex: 1; }
  header h1 { font-size: 14px; margin: 0; font-weight: 600; }
  label.f { color: var(--dim); font-size: 12px; }
  select, input[type=search], input[type=text], button, textarea {
    background: #3d3d3d;
    color: var(--text);
    border: 1px solid var(--card-edge);
    border-radius: 4px;
    padding: 4px 8px;
    font: inherit;
  }
  input[type=search] { width: 190px; }
  button { cursor: pointer; }
  button:hover:not(:disabled) { border-color: var(--accent); }
  button:disabled { opacity: .4; cursor: default; }
  button.primary { border-color: var(--accent); color: #cfe2ff; }
  button.primary:hover:not(:disabled) { background: #33445e; }
  .counts { color: var(--dim); font-size: 12px; }
  .counts b { color: var(--text); font-weight: 600; }

  .meta {
    padding: 8px 16px;
    background: #2b2b2b;
    border-bottom: 1px solid #1c1c1c;
    color: var(--dim);
    font-size: 12px;
  }
  .meta code {
    background: #00000055; border-radius: 3px; padding: 1px 5px;
    font-family: Consolas, "Courier New", monospace;
    color: #cfcfcf; user-select: all;
  }
  .meta .legend { margin-top: 6px; }
  .meta summary { cursor: pointer; color: var(--accent); }
  .meta dl { margin: 6px 0 0; display: grid; grid-template-columns: auto 1fr; gap: 4px 10px; }
  .meta dt { color: var(--text); font-weight: 600; }
  .meta dd { margin: 0; }

  main { padding: 14px 16px 80px; display: flex; flex-direction: column; gap: 12px; }

  .case {
    background: var(--panel);
    border: 1px solid var(--card-edge);
    border-left: 4px solid var(--card-edge);
    border-radius: 5px;
    padding: 10px 12px 12px;
  }
  .case[hidden] { display: none; }
  .case.done { border-left-color: var(--accent); }
  .case.skip { border-left-color: var(--ok); opacity: .72; }

  .case-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  .badge {
    flex: none;
    padding: 1px 7px;
    border-radius: 3px;
    background: #4a3a2f;
    color: #ffb79c;
    font-weight: 600;
    letter-spacing: .5px;
  }
  .badge.R1 { background: #3a3f52; color: #a9c2ff; }
  .badge.R3 { background: #4a3a2f; color: #ffb79c; }
  .badge.R4 { background: #33413a; color: #9fd8b4; }
  .badge.R5 { background: #4a2f3d; color: #ffa9c8; }
  .species { color: var(--accent); font-weight: 600; }
  .subject { color: var(--dim); }
  .metrics { color: var(--dim); font-size: 12px; }
  .metrics b { color: var(--text); font-weight: 600; }
  .state-tag { font-size: 12px; color: var(--dim); }
  .state-tag.done { color: var(--accent); }
  .state-tag.skip { color: var(--ok); }

  .headline { margin: 6px 0 0; }
  .problems { margin: 4px 0 0; padding-left: 18px; color: var(--dim); }
  .problems li { margin: 1px 0; }

  .cards { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; }
  .card {
    width: 236px;
    background: var(--card);
    border: 1px solid var(--card-edge);
    border-radius: 5px;
    overflow: hidden;
  }
  .card.edited { border-color: var(--accent); }
  .thumb { position: relative; width: 234px; height: 234px; background: #202020; }
  .thumb img { display: block; width: 234px; height: 234px; }
  .thumb.failed::after, .thumb.nogif::after {
    content: 'GIF 加载失败 · 点击重试';
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    color: var(--warn); font-size: 12px; text-align: center; padding: 0 10px;
  }
  .thumb.nogif::after { content: '没有 review GIF'; color: var(--dim); }
  .thumb.failed { cursor: pointer; }
  .clip {
    position: absolute; top: 0; left: 0; right: 0;
    padding: 3px 6px;
    background: rgba(0,0,0,.62);
    color: #fff; font-size: 11px; text-decoration: none;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  a.clip:hover { color: var(--accent); text-decoration: underline; }
  a.clip:visited { color: #fff; }
  .role {
    position: absolute; top: 22px; right: 0;
    padding: 2px 6px;
    background: rgba(180,85,47,.92);
    color: #fff; font-size: 10px;
  }
  .cur {
    position: absolute; bottom: 0; left: 0; right: 0;
    padding: 2px 6px;
    background: rgba(0,0,0,.62);
    color: var(--dim); font-size: 11px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .cur b { color: #ddd; font-weight: 600; }

  .edit { padding: 6px; display: flex; flex-direction: column; gap: 5px; }
  .edit .line { display: flex; gap: 5px; }
  .edit input[type=text] { flex: 1; min-width: 0; }
  .edit select { flex: none; width: 96px; }
  .edit .reset { flex: none; padding: 3px 7px; }
  .status { font-size: 11px; color: var(--dim); min-height: 15px; word-break: break-word; }
  .status.bad { color: var(--warn); }
  .status.good { color: #7fd08a; }
  .suggest {
    font-size: 11px; color: var(--accent);
    background: none; border: none; padding: 0; text-align: left; cursor: pointer;
  }
  .suggest:hover { text-decoration: underline; }

  .note { margin-top: 8px; display: flex; align-items: center; gap: 6px; }
  .note input { flex: 1; min-width: 0; }
  .note label { color: var(--dim); font-size: 12px; flex: none; }

  .empty { padding: 60px; text-align: center; color: var(--dim); }

  #notice {
    position: sticky; top: 0; z-index: 9;
    padding: 10px 16px;
    background: #33413a;
    border-bottom: 1px solid #1c1c1c;
    line-height: 1.6;
  }
  #notice.bad { background: #45302a; }
  #notice[hidden] { display: none; }
  #notice button { margin-left: 10px; }

  dialog {
    width: min(900px, 92vw);
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--card-edge);
    border-radius: 6px;
    padding: 14px;
  }
  dialog::backdrop { background: rgba(0,0,0,.6); }
  dialog textarea {
    width: 100%; height: 58vh; resize: vertical;
    font-family: Consolas, "Courier New", monospace; font-size: 12px;
    background: #2a2a2a;
  }
  dialog .bar { display: flex; gap: 8px; align-items: center; margin-top: 10px; }
</style>
</head>
<body>

<header>
  <h1>action_labels 审计核验</h1>

  <label class="f" for="rule">规则</label>
  <select id="rule"><option value="all">all</option></select>

  <label class="f" for="species">物种</label>
  <select id="species"><option value="all">all</option></select>

  <label class="f" for="state">状态</label>
  <select id="state">
    <option value="all">all</option>
    <option value="todo">待处理</option>
    <option value="done">已改</option>
    <option value="skip">无需修改</option>
  </select>

  <input type="search" id="q" placeholder="搜索 clip / 标签 / 物种">

  <span class="counts" id="counts"></span>
  <span class="spacer"></span>

  <button id="copy" class="primary" title="把所有已确认的修改整理成一段话复制走，直接粘给 LLM 让它写回 jsonl">复制修复指令</button>
  <button id="preview" title="先看看会复制什么">预览</button>
  <button id="clear" title="清空本页所有人工标记（改过的标签、无需修改、备注）">清空标记</button>
</header>

<div id="notice" hidden></div>
<div class="meta" id="meta"></div>
<main id="list"></main>

<dialog id="dlg">
  <div>下面这段文字就是「复制修复指令」会放进剪贴板的内容：</div>
  <textarea id="dlgText" readonly></textarea>
  <div class="bar">
    <button id="dlgCopy" class="primary">复制</button>
    <button id="dlgClose">关闭</button>
    <span class="counts" id="dlgHint"></span>
  </div>
</dialog>

<script type="application/json" id="report-data">/*__REPORT_DATA__*/</script>
<script>
const DATA = JSON.parse(document.getElementById('report-data').textContent);
const CASES = DATA.cases;
const VOCAB = DATA.vocab;
const META = DATA.meta;
const $ = (id) => document.getElementById(id);
const NUL = '\u0000';
const clipKey = (c) => c.species + NUL + c.clip;

// ── verdict store ───────────────────────────────────────────────────────────
// Everything the operator decides lives here and in localStorage: the HTML is
// regenerated by every audit run, so state kept in the DOM would be thrown away
// the next time the tool is run.
const BLANK = { labels: {}, groups: {}, skip: {}, notes: {} };
let S = Object.assign({}, BLANK);
try {
  const raw = localStorage.getItem(DATA.store_key);
  if (raw) S = Object.assign({}, BLANK, JSON.parse(raw));
} catch (err) { /* private window / disabled storage: run without persistence */ }

function persist() {
  try { localStorage.setItem(DATA.store_key, JSON.stringify(S)); }
  catch (err) { /* nothing to do -- the page still works for this session */ }
}

// ── label canonicalisation (mirrors motion_labels.canonical_action_label) ───
// The page validates with the trainer's own rules so a label that reaches the
// clipboard is one _validate_action_label_entry would accept. Anything else
// would be a fix that fails on the next preprocessing run, hours later.
function canonical(text) {
  const raw = String(text == null ? '' : text).toLowerCase()
    .split(/[,，、;；]+/)
    .map((piece) => piece.trim().replace(/\s+/g, ' '))
    .filter(Boolean);
  const tokens = [];
  for (const word of raw) if (!tokens.includes(word)) tokens.push(word);
  if (!tokens.length) return { ok: false, empty: true, value: '', error: '' };
  const unknown = tokens.filter((w) => !(w in VOCAB.order));
  if (unknown.length) {
    return { ok: false, value: '', error: '不在词表里：' + unknown.join(' / ') };
  }
  if (tokens.length > VOCAB.max_words) {
    return { ok: false, value: '',
             error: tokens.length + ' 个词，超过上限 ' + VOCAB.max_words };
  }
  const heads = tokens.filter((w) => VOCAB.heads.includes(w));
  if (!heads.length) {
    return { ok: false, value: '', error: '缺少 head 词（idle / walk / attack …）' };
  }
  if (heads.length > VOCAB.max_heads) {
    return { ok: false, value: '',
             error: heads.length + ' 个 head 词，超过上限 ' + VOCAB.max_heads };
  }
  const directions = tokens.filter((w) => VOCAB.directions.includes(w))
    .sort((a, b) => VOCAB.order[a] - VOCAB.order[b]);
  const mods = tokens.filter((w) => !VOCAB.heads.includes(w) &&
                                    !VOCAB.directions.includes(w))
    .sort((a, b) => VOCAB.order[a] - VOCAB.order[b]);
  const directionAnchor = heads.includes('turn') ? 'turn' : heads[heads.length - 1];
  const canonical = [];
  for (const head of heads) {
    canonical.push(head);
    if (head === directionAnchor) canonical.push(...directions);
  }
  return { ok: true, value: canonical.concat(mods).join(', '), error: '' };
}

// ── per-clip edit state ─────────────────────────────────────────────────────
// A clip can appear in more than one case (an R1 bucket member that also trips
// R4); the edit belongs to the CLIP, so every card showing it stays in sync.
const cardsByClip = new Map();

function currentOf(clip) {
  const key = clipKey(clip);
  const label = S.labels[key];
  return {
    label: label === undefined ? '' : label,
    group: S.groups[key] === undefined ? clip.group : S.groups[key],
  };
}

function editOf(clip) {
  const key = clipKey(clip);
  const typed = S.labels[key];
  const group = S.groups[key] === undefined ? clip.group : S.groups[key];
  const groupChanged = group !== clip.group;
  if (typed === undefined || !String(typed).trim()) {
    return groupChanged
      ? { key: key, clip: clip, label: clip.label, group: group,
          changed: true, valid: true, error: '' }
      : null;
  }
  const parsed = canonical(typed);
  if (!parsed.ok) {
    return { key: key, clip: clip, label: '', group: group,
             changed: true, valid: false, error: parsed.error || '标签为空' };
  }
  if (parsed.value === clip.label && !groupChanged) return null;
  return { key: key, clip: clip, label: parsed.value, group: group,
           changed: true, valid: true, error: '' };
}

const caseState = (c) => {
  if (c.clips.some((clip) => editOf(clip))) return 'done';
  if (S.skip[c.id]) return 'skip';
  return 'todo';
};

// ── lazy GIF mounting ───────────────────────────────────────────────────────
// A run can put several hundred animated GIFs on one page. Only the cards near
// the viewport carry a src; filtering a case out hides it, which the observer
// reports as "not intersecting", so its frames are freed too.
const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const img = e.target;
    if (e.isIntersecting) {
      img.dataset.live = '1';
      if (!img.getAttribute('src')) img.src = img.dataset.src;
    } else {
      delete img.dataset.live;
      img.removeAttribute('src');
    }
  }
}, { rootMargin: '600px 0px', threshold: 0 });

// ── rendering ───────────────────────────────────────────────────────────────
function metaBlock() {
  const m = META;
  const dl = [
    ['cond', m.cond_path],
    ['范围', m.action_group + '（' + m.clip_count + ' clips / ' + m.species_count + ' 物种）'],
    // The R1 thresholds only explain findings R1 produced; printing them for a
    // run that did not ask for R1 reads as a setting that did something.
    ['规则', (m.rules || []).join(', ') +
             ((m.rules || []).includes('R1')
               ? '　R1 阈值 ratio>' + m.thresholds.ratio + '、最小距离 ' +
                 m.thresholds.min_distance + '、' + m.thresholds.frames + ' 帧'
               : '')],
    ['生成', m.generated + '　' + (m.command || '')],
  ];
  if ((m.rules || []).includes('R1')) {
    dl.push(['R1 豁免', (m.r1_exempt && m.r1_exempt.length)
             ? m.r1_exempt.join(' / ') + ' —— 这类标签桶直接跳过'
             : '无']);
  }
  if (m.ignore_path && (m.ignored_buckets || m.ignored_clips)) {
    const parts = [];
    if (m.ignored_buckets) parts.push(m.ignored_buckets + ' 个桶');
    if (m.ignored_clips) parts.push(m.ignored_clips + ' 条 clip');
    dl.push(['已确认忽略', m.ignore_path + '（' + parts.join(' · ') + '）']);
  }
  const rows = dl.map(([k, v]) =>
    '<dt>' + esc(k) + '</dt><dd><code>' + esc(v) + '</code></dd>').join('');
  const legend = Object.keys(DATA.rule_info).map((rule) => {
    const info = DATA.rule_info[rule];
    return '<dt>' + rule + ' ' + esc(info.name) + '</dt><dd>' + esc(info.text) + '</dd>';
  }).join('');
  $('meta').innerHTML =
    '<dl>' + rows + '</dl>' +
    '<div class="legend"><details><summary>规则说明 · 怎么用这一页</summary><dl>' +
    legend +
    '<dt>用法</dt><dd>看 GIF，确认这条标签描述得对不对。要改就在输入框里写新标签' +
    '（页面按受控词表校验并规范化）；两条都没问题就点「无需修改」。' +
    '全部过完点「复制修复指令」，把剪贴板里的内容粘给 LLM，让它写回 action_labels.jsonl。' +
    '所有标记存在浏览器本地，重新生成本页不会丢。</dd>' +
    '</dl></details></div>';
}

function esc(text) {
  return String(text == null ? '' : text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function card(clip) {
  const el = document.createElement('div');
  el.className = 'card';

  const thumb = document.createElement('div');
  thumb.className = 'thumb' + (clip.gif ? '' : ' nogif');
  if (clip.gif) {
    const img = document.createElement('img');
    img.width = 234; img.height = 234; img.decoding = 'async'; img.alt = '';
    img.dataset.src = clip.gif;
    img.onload = () => thumb.classList.remove('failed');
    img.onerror = () => {
      if (!img.getAttribute('src')) return;   // unmounted on purpose
      img.removeAttribute('src');
      thumb.classList.add('failed');
    };
    thumb.onclick = () => {
      if (!thumb.classList.contains('failed')) return;
      thumb.classList.remove('failed');
      img.src = img.dataset.src + '?r=' + Date.now();
    };
    io.observe(img);
    thumb.appendChild(img);
  }

  const name = document.createElement(clip.bvh ? 'a' : 'div');
  name.className = 'clip';
  if (clip.bvh) { name.href = clip.bvh; name.rel = 'noopener'; }
  name.textContent = clip.clip.replace(/\.npy$/i, '');
  name.title = clip.bvh ? '在 BVH 查看器中打开 ' + clip.clip : clip.clip;
  thumb.appendChild(name);

  if (clip.role) {
    const role = document.createElement('div');
    role.className = 'role';
    role.textContent = clip.role;
    thumb.appendChild(role);
  }

  const cur = document.createElement('div');
  cur.className = 'cur';
  cur.innerHTML = esc(clip.group) + ' · <b>' + esc(clip.label || '(空)') + '</b>';
  cur.title = '当前 action_group / action_label';
  thumb.appendChild(cur);

  const edit = document.createElement('div');
  edit.className = 'edit';
  const line = document.createElement('div');
  line.className = 'line';

  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = clip.label || '(空标签)';
  input.title = '新的 action_label；留空表示不改。保存前会按受控词表规范化';
  input.value = currentOf(clip).label;

  const group = document.createElement('select');
  group.title = 'action_group';
  for (const name of VOCAB.groups) {
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = name;
    group.appendChild(opt);
  }
  if (clip.group && !VOCAB.groups.includes(clip.group)) {
    const opt = document.createElement('option');
    opt.value = clip.group; opt.textContent = clip.group;
    group.appendChild(opt);
  }
  group.value = currentOf(clip).group;

  const reset = document.createElement('button');
  reset.className = 'reset';
  reset.type = 'button';
  reset.textContent = '↺';
  reset.title = '撤销这条 clip 的修改';

  line.append(input, group, reset);
  const status = document.createElement('div');
  status.className = 'status';
  edit.append(line, status);

  if (clip.suggest) {
    const suggest = document.createElement('button');
    suggest.type = 'button';
    suggest.className = 'suggest';
    suggest.textContent = '候选（来自 clip 名字，未确认）：' + clip.suggest;
    suggest.title = '点击填入输入框';
    suggest.onclick = () => { input.value = clip.suggest; commit(clip, input, group); };
    edit.appendChild(suggest);
  }

  input.oninput = () => commit(clip, input, group);
  group.onchange = () => commit(clip, input, group);
  reset.onclick = () => {
    const key = clipKey(clip);
    delete S.labels[key];
    delete S.groups[key];
    persist();
    syncClip(clip);
    refreshAll();
  };

  el.append(thumb, edit);
  const entry = { el: el, input: input, group: group, status: status, clip: clip };
  const key = clipKey(clip);
  if (!cardsByClip.has(key)) cardsByClip.set(key, []);
  cardsByClip.get(key).push(entry);
  paint(entry);
  return el;
}

function commit(clip, input, group) {
  const key = clipKey(clip);
  const typed = input.value;
  if (String(typed).trim()) S.labels[key] = typed; else delete S.labels[key];
  if (group.value !== clip.group) S.groups[key] = group.value;
  else delete S.groups[key];
  // Touching a case is a verdict of its own: it can no longer be "无需修改".
  for (const c of CASES) {
    if (S.skip[c.id] && c.clips.some((other) => clipKey(other) === key)) {
      delete S.skip[c.id];
    }
  }
  persist();
  syncClip(clip);
  refreshAll();
}

function syncClip(clip) {
  for (const entry of cardsByClip.get(clipKey(clip)) || []) {
    const state = currentOf(clip);
    if (entry.input.value !== state.label) entry.input.value = state.label;
    if (entry.group.value !== state.group) entry.group.value = state.group;
    paint(entry);
  }
}

function paint(entry) {
  const edit = editOf(entry.clip);
  entry.el.classList.toggle('edited', !!edit);
  if (!edit) {
    entry.status.className = 'status';
    entry.status.textContent = '';
    return;
  }
  if (!edit.valid) {
    entry.status.className = 'status bad';
    entry.status.textContent = '✗ ' + edit.error;
    return;
  }
  entry.status.className = 'status good';
  const bits = [];
  if (edit.label !== entry.clip.label) bits.push('写回 "' + edit.label + '"');
  if (edit.group !== entry.clip.group) bits.push('分组 → ' + edit.group);
  entry.status.textContent = '✓ ' + bits.join('，');
}

function caseNode(c) {
  const el = document.createElement('section');
  el.className = 'case';
  el.dataset.id = c.id;

  const head = document.createElement('div');
  head.className = 'case-head';
  const badge = document.createElement('span');
  badge.className = 'badge ' + c.rule;
  badge.textContent = c.rule;
  badge.title = (DATA.rule_info[c.rule] || {}).name || '';
  const species = document.createElement('span');
  species.className = 'species';
  species.textContent = c.species;
  const subject = document.createElement('span');
  subject.className = 'subject';
  subject.textContent = c.subject;
  head.append(badge, species, subject);

  if (c.metrics.length) {
    const metrics = document.createElement('span');
    metrics.className = 'metrics';
    metrics.innerHTML = c.metrics
      .map((m) => esc(m.k) + ' <b>' + esc(m.v) + '</b>').join(' · ');
    head.appendChild(metrics);
  }

  const spacer = document.createElement('span');
  spacer.className = 'spacer';
  const tag = document.createElement('span');
  tag.className = 'state-tag';
  const skip = document.createElement('button');
  skip.type = 'button';
  skip.textContent = '无需修改';
  skip.title = '看过 GIF，标签是对的';
  skip.onclick = () => {
    if (S.skip[c.id]) delete S.skip[c.id]; else S.skip[c.id] = true;
    persist();
    refreshAll();
  };
  head.append(spacer, tag, skip);

  const headline = document.createElement('div');
  headline.className = 'headline';
  headline.textContent = c.headline;

  const problems = document.createElement('ul');
  problems.className = 'problems';
  for (const text of c.problems) {
    const li = document.createElement('li');
    li.textContent = text;
    problems.appendChild(li);
  }

  const cards = document.createElement('div');
  cards.className = 'cards';
  for (const clip of c.clips) cards.appendChild(card(clip));

  const note = document.createElement('div');
  note.className = 'note';
  const noteLabel = document.createElement('label');
  noteLabel.textContent = '备注';
  const noteInput = document.createElement('input');
  noteInput.type = 'text';
  noteInput.placeholder = c.note_hint ||
    '可选，会一起复制给 LLM（用来说明标签改不了的情况）';
  noteInput.value = S.notes[c.id] || '';
  noteInput.oninput = () => {
    if (noteInput.value.trim()) S.notes[c.id] = noteInput.value;
    else delete S.notes[c.id];
    persist();
  };
  note.append(noteLabel, noteInput);

  el.append(head, headline, problems, cards, note);
  el._tag = tag;
  el._skip = skip;
  return el;
}

function paintCase(el, c) {
  const state = caseState(c);
  el.classList.toggle('done', state === 'done');
  el.classList.toggle('skip', state === 'skip');
  el._tag.className = 'state-tag ' + (state === 'todo' ? '' : state);
  el._tag.textContent = state === 'done' ? '已改'
                      : state === 'skip' ? '无需修改' : '待处理';
  el._skip.textContent = S.skip[c.id] ? '撤销「无需修改」' : '无需修改';
  el._skip.disabled = state === 'done';
  el._skip.title = state === 'done'
    ? '这条已经有修改了，先撤销修改再标「无需修改」' : '看过 GIF，标签是对的';
}

const nodes = [];

function build() {
  metaBlock();
  const list = $('list');
  if (!CASES.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = '没有发现问题 —— 这一轮审计全部通过。';
    list.appendChild(empty);
  }
  for (const c of CASES) {
    const el = caseNode(c);
    list.appendChild(el);
    nodes.push({ el: el, case: c });
  }
  const rules = [...new Set(CASES.map((c) => c.rule))].sort();
  $('rule').innerHTML = '<option value="all">all</option>' + rules.map((r) =>
    '<option value="' + r + '">' + r + ' · ' +
    esc((DATA.rule_info[r] || {}).name || '') + '</option>').join('');
  const species = [...new Set(CASES.map((c) => c.species))].sort();
  $('species').innerHTML = '<option value="all">all</option>' +
    species.map((s) => '<option value="' + esc(s) + '">' + esc(s) + '</option>').join('');
  refreshAll();
}

function refreshAll() {
  const rule = $('rule').value;
  const species = $('species').value;
  const state = $('state').value;
  const q = $('q').value.trim().toLowerCase();
  let shown = 0;
  const tally = { todo: 0, done: 0, skip: 0 };
  for (const node of nodes) {
    const c = node.case;
    paintCase(node.el, c);
    const s = caseState(c);
    tally[s] += 1;
    let visible = true;
    if (rule !== 'all' && c.rule !== rule) visible = false;
    if (species !== 'all' && c.species !== species) visible = false;
    if (state !== 'all' && s !== state) visible = false;
    if (visible && q) {
      const hay = (c.species + ' ' + c.subject + ' ' + c.headline + ' ' +
        c.clips.map((clip) => clip.clip + ' ' + clip.label).join(' ')).toLowerCase();
      visible = hay.includes(q);
    }
    node.el.hidden = !visible;
    if (visible) shown += 1;
  }
  const edits = collectEdits();
  $('counts').innerHTML =
    '待处理 <b>' + tally.todo + '</b> · 已改 <b>' + tally.done + '</b> · 无需修改 <b>' +
    tally.skip + '</b> / ' + CASES.length +
    ' &nbsp;·&nbsp; 当前显示 <b>' + shown + '</b>' +
    ' &nbsp;·&nbsp; 待写回 <b>' + edits.valid.length + '</b> 条 clip' +
    (edits.invalid.length ? ' <span style="color:var(--warn)">（' +
      edits.invalid.length + ' 条标签无效）</span>' : '');
  $('copy').disabled = edits.valid.length === 0 && tally.skip === 0;
  $('preview').disabled = $('copy').disabled;
}

// ── the clipboard payload ───────────────────────────────────────────────────
// One edit per CLIP (not per case): a clip named by two findings must reach the
// sidecar once, with one label.
function collectEdits() {
  const seen = new Map();
  const invalid = [];
  for (const c of CASES) {
    for (const clip of c.clips) {
      const edit = editOf(clip);
      if (!edit) continue;
      if (!edit.valid) {
        if (!invalid.some((item) => item.key === edit.key)) invalid.push(edit);
        continue;
      }
      const existing = seen.get(edit.key);
      if (existing) { existing.why.push(whyOf(c)); continue; }
      edit.why = [whyOf(c)];
      seen.set(edit.key, edit);
    }
  }
  return { valid: [...seen.values()], invalid: invalid };
}

function whyOf(c) {
  const note = S.notes[c.id];
  return c.rule + ' ' + ((DATA.rule_info[c.rule] || {}).name || '') + '：' +
    c.headline + (note ? '（人工备注：' + note + '）' : '');
}

function buildPrompt() {
  const { valid, invalid } = collectEdits();
  const skipped = CASES.filter((c) => caseState(c) === 'skip');
  const byFile = new Map();
  for (const edit of valid) {
    const file = edit.clip.labels_path || '(未知路径)';
    if (!byFile.has(file)) byFile.set(file, []);
    byFile.get(file).push(edit);
  }

  const out = [];
  out.push('# action_labels 修复请求');
  out.push('');
  out.push('下面的修改来自 Anytop/tools/audit_action_labels.py 的审计报告，' +
           '我已经逐条看过 review GIF 确认过了。请帮我把它们写回对应的 ' +
           'action_labels.jsonl。');
  out.push('');
  out.push('## 怎么改');
  out.push('1. 每个 action_labels.jsonl 是 JSONL：一行一个对象，形如 ' +
           '{"clip": "<名字>.npy", "action_group": "...", "action_label": "..."}，' +
           '有的行还带 reviewed / pending_delete 等字段。');
  out.push('2. 按 clip 精确匹配。只改下面列出的 action_label / action_group，' +
           '其余字段、行的顺序、UTF-8 编码和原有换行符都不要动。');
  out.push('3. 匹配不到的 clip 直接报告给我，不要新增行、不要猜。');
  out.push('4. action_label 已经按受控词表规范化过（小写、", " 分隔、head 词在前并保持' +
           '时间顺序、修饰词按词表顺序、最多 ' + VOCAB.max_words + ' 个词 / ' +
           VOCAB.max_heads + ' 个 head 词），照抄即可，不要再改写。');
  out.push('5. 改完在 Anytop/ 目录下跑一遍复核：');
  out.push('   python tools/audit_action_labels.py --action-group ' +
           (META.action_group || 'all'));
  out.push('');
  out.push('## 修改清单（' + valid.length + ' 条 clip，' + byFile.size + ' 个文件）');
  if (!valid.length) out.push('（无）');
  for (const [file, edits] of byFile) {
    out.push('');
    out.push('### ' + file);
    for (const edit of edits) {
      out.push('- ' + edit.clip.clip + '　[' + edit.clip.species + ']');
      if (edit.label !== edit.clip.label) {
        out.push('    action_label: "' + edit.clip.label + '" -> "' + edit.label + '"');
      }
      if (edit.group !== edit.clip.group) {
        out.push('    action_group: "' + edit.clip.group + '" -> "' + edit.group + '"');
      }
      for (const why of edit.why) out.push('    依据: ' + why);
    }
  }
  out.push('');
  out.push('## 机器可读版本（每行一个 JSON，字段就是要写成的值）');
  out.push('```jsonl');
  for (const edit of valid) {
    out.push(JSON.stringify({
      labels_path: edit.clip.labels_path,
      clip: edit.clip.clip,
      action_group: edit.group,
      action_label: edit.label,
    }));
  }
  out.push('```');

  // "无需修改" on ANY rule means the case is confirmed fine, and all of them
  // go to the SAME ignore list -- nothing distinguishes R1 from the others. R1
  // entries are bucket keys (species/group/label); R3/R4/R5 are suppressed per
  // clip, so a confirmed mirror pair emits one line for each half.
  const ignoreLines = [];
  for (const c of skipped) {
    if (c.rule === 'R1') {
      ignoreLines.push(JSON.stringify({
        species: c.species,
        action_group: c.group,
        action_label: c.subject,
      }));
    } else {
      const seen = new Set();
      for (const clip of c.clips) {
        if (!clip.clip) continue;
        const key = clip.species + '\u0000' + clip.clip;
        if (seen.has(key)) continue;
        seen.add(key);
        ignoreLines.push(JSON.stringify({ species: clip.species, clip: clip.clip }));
      }
    }
  }
  if (ignoreLines.length) {
    out.push('');
    out.push('## 已核验「无需修改」—— 追加到忽略名单（R1 桶 + 单条 clip，不区分规则）');
    out.push('');
    out.push('下面这些我看过 GIF，标签是对的，不用改。把每行原样追加到：');
    out.push('   ' + (META.ignore_path || '(报告 meta 的忽略文件)'));
    out.push('（文件不存在就新建；追加，不要覆盖已有行；已存在的行可跳过。' +
             '下次跑审计会自动读这个文件，这些桶 / clip 就不再报。）');
    out.push('```jsonl');
    for (const line of ignoreLines) out.push(line);
    out.push('```');
  }
  if (invalid.length) {
    out.push('');
    out.push('## 注意：以下 clip 我填的新标签没通过词表校验，没有包含在上面的清单里');
    for (const edit of invalid) {
      out.push('- ' + edit.clip.clip + '：' + edit.error);
    }
  }
  out.push('');
  out.push('（报告：' + META.cond_path + '　范围 ' + META.action_group +
           '　生成于 ' + META.generated + '）');
  return { text: out.join('\n'), valid: valid, invalid: invalid, skipped: skipped };
}

function notice(text, bad) {
  const el = $('notice');
  el.className = bad ? 'bad' : '';
  el.textContent = text;
  const close = document.createElement('button');
  close.type = 'button';
  close.textContent = '知道了';
  close.onclick = () => { el.hidden = true; };
  el.appendChild(close);
  el.hidden = false;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    // A file:// page without clipboard permission still has the old path.
    const area = document.createElement('textarea');
    area.value = text;
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (e2) { ok = false; }
    area.remove();
    return ok;
  }
}

$('copy').onclick = async () => {
  const built = buildPrompt();
  const ok = await copyText(built.text);
  if (ok) {
    notice('已复制 ' + built.valid.length + ' 条修改' +
      (built.skipped.length ? '（另附 ' + built.skipped.length + ' 条「无需修改」）' : '') +
      (built.invalid.length ? '；' + built.invalid.length + ' 条标签无效被跳过' : '') +
      ' —— 直接粘给 LLM 即可。', built.invalid.length > 0);
  } else {
    openPreview(built, '浏览器拒绝了自动复制，请在下面手动全选复制。');
  }
};

function openPreview(built, hint) {
  $('dlgText').value = built.text;
  $('dlgHint').textContent = hint || (built.valid.length + ' 条修改 · ' +
    built.skipped.length + ' 条无需修改' +
    (built.invalid.length ? ' · ' + built.invalid.length + ' 条无效' : ''));
  $('dlg').showModal();
  $('dlgText').focus();
  $('dlgText').select();
}

$('preview').onclick = () => openPreview(buildPrompt(), '');
$('dlgCopy').onclick = async () => {
  const ok = await copyText($('dlgText').value);
  $('dlgHint').textContent = ok ? '已复制。' : '复制失败，请手动 Ctrl+C。';
};
$('dlgClose').onclick = () => $('dlg').close();

$('clear').onclick = () => {
  if (!confirm('清空本页所有人工标记（改过的标签、无需修改、备注）？')) return;
  S = { labels: {}, groups: {}, skip: {}, notes: {} };
  persist();
  for (const entries of cardsByClip.values()) {
    for (const entry of entries) {
      entry.input.value = '';
      entry.group.value = entry.clip.group;
      paint(entry);
    }
  }
  for (const node of nodes) {
    const note = node.el.querySelector('.note input');
    if (note) note.value = '';
  }
  refreshAll();
};

for (const id of ['rule', 'species', 'state']) $(id).onchange = refreshAll;
$('q').oninput = refreshAll;

build();
</script>
</body>
</html>
"""
