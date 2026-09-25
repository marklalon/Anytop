#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draft ``species_tags.jsonl`` / ``action_labels.jsonl`` with the vision LLM.

Step 3 of ``../readme.txt``: the review GIFs ``render_gifs.py`` writes are
handed to the local VLM (``http://127.0.0.1:8066/v1``) and its answers become
DRAFTS; nothing reaches a dataset's sidecars until ``apply``.  The datasets are
the ones ``../datasets.jsonl`` lists, the same manifest ``serve.py`` and
``render_gifs.py`` read::

    llm_annotate.py actions --dataset unitybundles --filter "RMW_*" --dry-run
    llm_annotate.py actions --dataset unitybundles --clip "Fly*" --include-reviewed
    llm_annotate.py species --dataset unitybundles --filter "MB_*"
    llm_annotate.py apply   --dataset unitybundles            # actions drafts
    llm_annotate.py apply   --dataset unitybundles --what species

Drafts go to ``<processed>/review/llm/`` -- a contract file (the sidecar's own
row shape) plus a ``*.review.jsonl`` carrying what a human needs to judge it:
the family the clip was answered with, every warning the local validator raised
and whether the row wants a look.  Both are appended as work finishes, so an
interrupted run resumes where it stopped.

**actions.**  ``action_group`` is FROZEN input read from the row itself: it was
decided by hand and decides where the clip trains, so the model is told the
group and never asked for it.  Rows marked ``"reviewed": true`` are the ground
truth -- they are skipped unless ``--include-reviewed``, and they are where the
few-shot examples come from.  Clips whose names differ only by a trailing
Left/Right are one FAMILY and are answered in one call, which is what keeps a
mirror pair a mirror; a reviewed sibling is shown as fixed and anchors the
other side.  Labels are exact controlled-vocabulary tokens
(``motion_labels.CONTROLLED_VOCAB``) and are spelled here with
``canonical_action_label``; head order follows the dataset's existing spelling
of the same word set, since head order IS the condition.

**species.**  Two closed tags (``dataset_tags``: body plan x locomotion).  The
evidence is the rig out of ``cond.npy`` (joint tree, leg chains, contact
joints) plus one clip GIF of the species, preferably a locomotion clip.

**apply.**  Writes the action drafts into ``action_labels.jsonl`` through
``tools.action_label_sidecar`` -- only rows that are not reviewed (unless
``--include-reviewed``), each marked ``"reviewed": false`` + ``autofill`` so
``serve.py`` lists it for the human pass -- and refuses a draft whose head order
contradicts the dataset's spelling of the same word set.  ``--what species``
rewrites ``species_tags.jsonl``; a changed species tag needs a cond regen.

**Video.**  The server does not sample a GIF on its own frame times: what
reaches the model is ``min(frames sent / 2, ~1 per second of container time)``.
:func:`gif_as_video` re-times the GIF to 2 fps and writes every frame twice, so
the surviving half is one copy of each rendered frame.  The price is playback
far slower than the action, which the prompt has to say: speed words therefore
come from the clip name only.
"""
from __future__ import annotations

import argparse
import base64
import fnmatch
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageSequence

HERE = Path(__file__).resolve().parent            # .../Anytop/dataset/review
DATASET_ROOT = HERE.parent                        # .../Anytop/dataset
ANYTOP_DIR = DATASET_ROOT.parent                  # .../Anytop
for _candidate in (str(ANYTOP_DIR), str(ANYTOP_DIR.parent)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from data_loaders.truebones.truebones_utils.cond_schema import load_cond  # noqa: E402
from data_loaders.truebones.truebones_utils.dataset_tags import (  # noqa: E402
    CANONICAL_OBJECT_SUBSETS,
    SPECIES_LOCOMOTIONS,
    SPECIES_TAGS_FILE,
    check_species_tags,
    normalize_species_tag,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    ACTION_GROUPS,
    ACTION_LABEL_MAX_HEADS,
    ACTION_LABEL_MAX_WORDS,
    ActionLabelError,
    CONTROLLED_VOCAB,
    DIRECTION_VOCAB,
    HEAD_VOCAB,
    LOOP_FLAG_KEY,
    MODIFIER_VOCAB,
    canonical_action_label,
    clip_key,
    head_words_in,
    parse_action_label,
)
from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    ACTION_LABELS_FILE,
    MOTION_METADATA_FILE,
)
from tools.action_label_sidecar import (  # noqa: E402
    autofill_action_label,
    read_action_label_rows,
    rewrite_action_label_rows,
)
from tools.audit_action_labels import (  # noqa: E402
    NO_HEADING_WORDS,
    NO_PLANAR_DIRECTION_WORDS,
    PLANAR_DIRECTIONS,
    VERTICAL_WORDS,
    mirror_label,
    takes_planar_direction,
)

# ── paths ────────────────────────────────────────────────────────────────────
DATASETS_FILE = DATASET_ROOT / "datasets.jsonl"
DRAFT_SUBDIR = Path("review") / "llm"
GIF_SUBDIR = Path("review") / "gif"
COND_FILE = "cond.npy"

# ── llm ──────────────────────────────────────────────────────────────────────
ENDPOINT = os.environ.get("PCVG_LLM_ENDPOINT", "http://127.0.0.1:8066/v1")
TIMEOUT = 300.0
RETRIES = 3
MAX_TOKENS = 512
# The local server corrupts JSON under higher concurrency.
MAX_WORKERS = 4
FRAME_MS = 500          # 2 fps of container time: the 1 fps term admits half ...
FRAME_REPEAT = 2        # ... and that half is one copy of each rendered frame
SHOTS = 12              # reviewed examples retrieved per call

# How many clips one call may label at once. Past this the answer gets long
# enough for the guided decoder to stall, so a bigger family is split back into
# single clips and mirror consistency falls back to the audit.
MAX_FAMILY = 4

_PRINT_LOCK = threading.Lock()


def log(message):
    with _PRINT_LOCK:
        print(message, flush=True)


# ═══════════════════════════ vocabulary view ════════════════════════════════
# MODIFIER_VOCAB is ordered in role blocks (motion_labels.py comments them);
# the prompt shows them as blocks, so the split is read back out of the tuple at
# these leaders. A leader that moved is a loud failure, not a wrong prompt.
MODIFIER_BLOCKS = (
    ("trot", "how the head is executed (gait, speed, wing state)"),
    ("bite", "secondary action layered on the head"),
    ("spin", "how a strike is delivered"),
    ("happy", "affect / social gesture"),
    ("aim", "activity and object handling"),
    ("bow", "the implement the action is performed with"),
)

# The media a body travels in. A label names at most one; turn, jump, fall,
# roll and hover ride on one.
TRAVEL_MODES = ("walk", "run", "fly", "swim", "crawl")

# Word pairs that name one axis twice, or where one adds nothing beside the
# other. A pair is only checked while both words exist in the vocabulary.
_CONFLICTS = (
    (("die", "fall"), "a death already falls -- write 'die' alone"),
    (("die", "idle"), "'idle' is a live motionless stance and contradicts a death"),
    (("walk", "run"), "the gait between them is 'walk, trot'"),
    (("slow", "fast"), "the speed axis takes one word, not both"),
)

# Speed is name-only evidence: the 2 fps playback cannot show it.
_NAME_SPEED = {"slow": ("slow", "slowly"),
               "fast": ("fast", "sprint", "sprinting", "dash", "quick")}


def _check_vocab():
    """Fail at startup when a word this tool is built around left the vocabulary."""
    missing = [w for w in TRAVEL_MODES if w not in HEAD_VOCAB]
    missing += [w for w in _NAME_SPEED if w not in MODIFIER_VOCAB]
    missing += [leader for leader, _ in MODIFIER_BLOCKS if leader not in MODIFIER_VOCAB]
    if missing:
        raise SystemExit("motion_labels.py no longer has %s; llm_annotate.py has to "
                         "follow the vocabulary" % ", ".join(missing))
    edges = [MODIFIER_VOCAB.index(leader) for leader, _ in MODIFIER_BLOCKS]
    if edges != sorted(edges) or edges[0] != 0:
        raise SystemExit("MODIFIER_BLOCKS leaders are out of order against "
                         "MODIFIER_VOCAB: %s" % list(zip(MODIFIER_BLOCKS, edges)))


def modifier_blocks():
    """``[(title, words), ...]`` -- MODIFIER_VOCAB cut at the block leaders."""
    edges = [MODIFIER_VOCAB.index(leader) for leader, _ in MODIFIER_BLOCKS]
    bounds = edges + [len(MODIFIER_VOCAB)]
    return [(MODIFIER_BLOCKS[i][1], MODIFIER_VOCAB[bounds[i]:bounds[i + 1]])
            for i in range(len(MODIFIER_BLOCKS))]


def vocab_block_text():
    lines = ["  HEAD -- what the clip is about; 1 or %d, the first one is the main one:"
             % ACTION_LABEL_MAX_HEADS,
             "    " + ", ".join(HEAD_VOCAB)]
    for title, words in modifier_blocks():
        lines.append("  MODIFIER, %s -- zero or more:" % title)
        lines.append("    " + ", ".join(words))
    lines.append("  DIRECTION -- planar words, then at most one vertical:")
    lines.append("    " + ", ".join(DIRECTION_VOCAB))
    return "\n".join(lines)


# ═══════════════════════════ datasets ═══════════════════════════════════════
def read_jsonl(path):
    rows = []
    path = Path(path)
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def load_datasets(pattern, manifest=DATASETS_FILE):
    """Datasets from the manifest whose namespace (or its last part) matches."""
    out = []
    for entry in read_jsonl(manifest):
        ns = entry["namespace"]
        short = ns.split("/")[-1]
        if pattern not in ("all", "*") and not (fnmatch.fnmatch(ns, pattern)
                                                or fnmatch.fnmatch(short, pattern)):
            continue
        processed = (ANYTOP_DIR / entry["path"]).resolve()
        out.append({"ns": ns, "short": short, "processed": processed,
                    "gif_dir": processed / GIF_SUBDIR})
    return out


def draft_dir(ds, out_root):
    if out_root:
        return Path(out_root) / ds["ns"].replace("/", "__")
    return ds["processed"] / DRAFT_SUBDIR


def load_clips(ds):
    """Every sidecar row of *ds* as a clip record, species resolved.

    Species comes from ``motion_metadata.json`` (``object_type``), the only
    record of which species a clip belongs to; a clip it does not know falls
    back to the longest cond species its name starts with.
    """
    meta_path = ds["processed"] / MOTION_METADATA_FILE
    motions = {}
    if meta_path.is_file():
        motions = json.loads(meta_path.read_text(encoding="utf-8")).get("motions", {})
    rows = read_action_label_rows(ds["processed"])
    known_species = sorted({str(m.get("object_type")) for m in motions.values()
                            if m.get("object_type")}, key=len, reverse=True)
    clips = []
    for row in rows:
        clip = clip_key(row["clip"])
        meta = motions.get(clip + ".npy") or {}
        species = meta.get("object_type")
        if not species:
            species = next((s for s in known_species if clip.startswith(s + "_")),
                           clip.rsplit("_", 1)[0])
        action = clip[len(species) + 1:] if clip.startswith(species + "_") else clip
        frame_range = meta.get("source_frame_range") or [0, 0]
        gif = ds["gif_dir"] / (clip + ".gif")
        clips.append({
            "ns": ds["ns"], "clip": clip, "species": species, "action": action,
            "group": str(row.get("action_group") or "").strip().lower(),
            "label": str(row.get("action_label") or ""),
            "reviewed": bool(row.get("reviewed")),
            "pending_delete": bool(row.get("pending_delete")),
            "is_loop": row.get(LOOP_FLAG_KEY),
            "gif": gif if gif.is_file() else None,
            "src_frames": max(0, int(frame_range[-1]) - int(frame_range[0])),
        })
    return clips


# ═══════════════════════════ names ══════════════════════════════════════════
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+")


def decamel(text):
    """'PolygonalOneEyedBat' -> 'polygonal one eyed bat' (prompt-facing name)."""
    parts = []
    for chunk in re.split(r"[^A-Za-z0-9]+", str(text)):
        parts.extend(_CAMEL_RE.findall(chunk))
    return " ".join(p.lower() for p in parts if p)


def readable_species(species):
    """'MU01_Ghost' -> 'ghost': the pack code carries no species information."""
    code, _, rest = str(species).partition("_")
    if rest and re.fullmatch(r"[A-Z]{2,4}[0-9]{0,2}", code):
        return decamel(rest)
    return decamel(species)


# A trailing side marker, matched case-sensitively so "Upright" does not read as
# the right-hand version of "Up". Same rule as tools/audit_action_labels.py.
_SIDE_WORD_RE = re.compile(r"(Left|Right)$")
_SIDE_LETTER_RE = re.compile(r"(?<=[a-z])([LR])$")


def clip_side(action):
    """``'L'``, ``'R'`` or ``None`` for the side this clip's name names."""
    match = _SIDE_WORD_RE.search(action)
    if match:
        return "L" if match.group(1) == "Left" else "R"
    match = _SIDE_LETTER_RE.search(action)
    return match.group(1) if match else None


def clip_family(action):
    """The action name with its trailing side marker removed -- the family key."""
    stripped = _SIDE_WORD_RE.sub("", action)
    if stripped != action:
        return stripped
    return _SIDE_LETTER_RE.sub("", action)


# Direction words a clip NAME carries. Used for NOTES only: the name is the
# weakest evidence there is and a note is for a human to look at, never a
# forced retry.
_NAME_PLANAR = {
    "forward": ("forward", "forwards", "fwd", "fw"),
    "backward": ("backward", "backwards", "back", "reverse", "retreat"),
    "left": ("left",),
    "right": ("right",),
}
_NAME_VERTICAL = {
    "up": ("up", "ascend", "ascending", "rise", "rising"),
    "down": ("down", "descend", "descending", "drop"),
}
# "...Up" compounds where Up is not a direction at all.
_UP_COMPOUNDS = ("backup", "getup", "standup", "wakeup", "pickup", "setup",
                 "warmup", "levelup", "powerup", "lineup", "closeup")


def name_direction_words(action):
    tokens = set(decamel(action).split())
    found = {word for table in (_NAME_PLANAR, _NAME_VERTICAL)
             for word, forms in table.items() if tokens & set(forms)}
    if "up" in found and any(c in action.lower() for c in _UP_COMPOUNDS):
        found.discard("up")
    side = clip_side(action)
    if side:
        found.add("left" if side == "L" else "right")
    return found


def name_speed_words(action):
    tokens = set(decamel(action).split())
    return {word for word, forms in _NAME_SPEED.items() if tokens & set(forms)}


# The prop a character carries is spelled as a suffix on the clip name, and the
# motion underneath is the same one: IAC_Caveman ships Land beside LandWeapon.
_VARIANT_SUFFIX_RE = re.compile(
    r"(weapon|shield|sword|spear|axe|bow|staff|rifle|crossbow|longbow|torch"
    r"|dagger|hammer|gun|pistol)$", re.IGNORECASE)


def variant_base(action, known):
    """``(base action, prop)`` when *action* is a prop variant of one in *known*.

    Only a suffix that leaves an action actually present counts -- "Bow" ends
    "Elbow" too.
    """
    match = _VARIANT_SUFFIX_RE.search(action)
    if not match or match.start() == 0:
        return None, None
    base = action[:match.start()]
    return (base, match.group(0)) if base in known else (None, None)


_AIRBORNE_NAME_RE = re.compile(r"(fly|flight|swim|hover|float|glide|dive|jump|fall|air)",
                               re.IGNORECASE)
_AIRBORNE_WORDS = frozenset(w for w in ("fly", "swim", "jump", "fall", "glide", "dive",
                                        "land", "takeoff", "roll", "hover")
                            if w in CONTROLLED_VOCAB)


# ═══════════════════════════ transport ══════════════════════════════════════
def discover_model(endpoint):
    with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=30) as resp:
        models = json.loads(resp.read().decode("utf-8")).get("data") or []
    if not models:
        raise SystemExit("no model served at %s" % endpoint)
    return models[0]["id"]


def gif_as_video(path):
    """``(base64 GIF, frame count)`` re-timed so the server's sampler keeps every frame."""
    src = Image.open(path)
    frames = []
    for frame in ImageSequence.Iterator(src):
        rgba = frame.convert("RGBA")
        flat = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        flat.alpha_composite(rgba)
        rgb = flat.convert("RGB")
        frames.append(rgb)
        for rep in range(1, FRAME_REPEAT):
            twin = rgb.copy()
            # one corner pixel, so Pillow does not merge the twin away
            twin.putpixel((rep % twin.size[0], 0), (0, 0, 0))
            frames.append(twin)
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=FRAME_MS, loop=0)
    return base64.b64encode(buf.getvalue()).decode("ascii"), len(frames) // FRAME_REPEAT


def gif_frame_count(path):
    try:
        with Image.open(path) as im:
            return int(getattr(im, "n_frames", 1))
    except OSError:
        return 0


_JSON_PAIR_RE = re.compile(
    r'"([A-Za-z_][A-Za-z0-9_.\- ]*)"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|(-?\d+))')


def _salvage_json(text):
    """Best-effort ``{key: value}`` from a truncated flat JSON object."""
    out = {}
    for key, string, number in _JSON_PAIR_RE.findall(text or ""):
        out[key] = int(number) if number else string.replace('\\"', '"')
    return out


def call_llm(args, system, text, gif_path, schema, schema_name, essential=None):
    """One structured chat completion with the clip as a video; returns the object.

    ``response_format`` is a json_schema, so enums are enforced by the decoder.
    *essential* is the subset of keys a salvaged (truncated) reply has to carry:
    the decoder occasionally stalls inside free text and pads to max_tokens, and
    the fields already emitted are still good.
    """
    content = [{"type": "text", "text": text}]
    if gif_path is not None:
        video, _ = gif_as_video(gif_path)
        content.append({"type": "video_url",
                        "video_url": {"url": "data:video/gif;base64," + video}})
    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": schema_name, "schema": schema}},
    }
    body = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(args.retries):
        request = urllib.request.Request(
            args.endpoint.rstrip("/") + "/chat/completions",
            data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as resp:
                reply = json.loads(resp.read().decode("utf-8"))
            choice = reply["choices"][0]
            raw = choice["message"].get("content")
            if raw is None:
                raise RuntimeError("null content (finish_reason=%s)"
                                   % choice.get("finish_reason", "?"))
            try:
                return json.loads(raw)
            except ValueError:
                salvaged = _salvage_json(raw)
                if all(key in salvaged for key in (essential or schema.get("required", ()))):
                    return salvaged
                raise RuntimeError(
                    "unparseable content (finish_reason=%s, %d chars): %r"
                    % (choice.get("finish_reason", "?"), len(raw), raw[:160]))
        except (urllib.error.URLError, ValueError, KeyError, RuntimeError,
                TimeoutError) as exc:
            last = exc
            if attempt + 1 < args.retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("llm call failed after %d attempts: %s" % (args.retries, last))


# ═══════════════════════════ draft files ════════════════════════════════════
class Sink:
    """Append-as-you-go JSONL pair (contract file + review file), resumable."""

    def __init__(self, main_path, review_path, key_field):
        self.main_path, self.review_path, self.key_field = (
            Path(main_path), Path(review_path), key_field)
        self.lock = threading.Lock()
        self.done = {row.get(key_field) for row in read_jsonl(self.main_path)}

    def write(self, main_rows, review_rows):
        """Append one unit of work whole: a family is written together or not at all."""
        with self.lock:
            self.main_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.main_path, "a", encoding="utf-8") as fh:
                for row in main_rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    self.done.add(row[self.key_field])
            with open(self.review_path, "a", encoding="utf-8") as fh:
                for row in review_rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def invalidate(self, keys):
        """Remove old contract rows for work that ``--overwrite`` will redo.

        Invalidation happens before the LLM calls start.  A rejected result or a
        failed request therefore cannot leave an older, apparently valid draft
        behind for ``apply`` to consume.  Review rows are history and remain.
        """
        keys = {str(key) for key in keys}
        if not keys:
            return 0
        with self.lock:
            rows = read_jsonl(self.main_path)
            kept = [row for row in rows if str(row.get(self.key_field, "")) not in keys]
            removed = len(rows) - len(kept)
            self.done.difference_update(keys)
            if not removed:
                return 0
            self.main_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.main_path.with_name(self.main_path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                for row in kept:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp, self.main_path)
            return removed

    def sort_in_place(self):
        """Rewrite both files key-sorted, last write wins (what makes --overwrite replace)."""
        for path in (self.main_path, self.review_path):
            rows = read_jsonl(path)
            if not rows:
                continue
            latest = {str(row.get(self.key_field, "")): row for row in rows}
            with open(path, "w", encoding="utf-8") as fh:
                for key in sorted(latest):
                    fh.write(json.dumps(latest[key], ensure_ascii=False) + "\n")


def run_pool(args, jobs, worker, name_of):
    """Run *worker* over *jobs*; one failing job never ends the run."""
    results, failures = [], []
    counter = {"n": 0}

    def wrapped(job):
        try:
            out = worker(job)
        except Exception as exc:                        # noqa: BLE001
            log("  [FAIL] %s: %s" % (name_of(job), exc))
            failures.append((name_of(job), str(exc)))
            return None
        with _PRINT_LOCK:
            counter["n"] += 1
            if counter["n"] % 25 == 0:
                print("  ... %d/%d" % (counter["n"], len(jobs)), flush=True)
        return out

    if args.workers <= 1:
        outs = [wrapped(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            outs = list(pool.map(wrapped, jobs))
    results = [out for out in outs if out]
    return results, failures


# ═══════════════════════════ species tags ═══════════════════════════════════
_LEG_RE = re.compile(r"(leg|thigh|femur|tibia|calf|shin|foot|feet|toe|paw|hoof|"
                     r"ankle|knee|hock)", re.IGNORECASE)
_WING_RE = re.compile(r"(wing|pinion|rotor|propeller)", re.IGNORECASE)
_ARM_RE = re.compile(r"(arm|hand|finger|thumb|shoulder|clavicle)", re.IGNORECASE)
_TAIL_RE = re.compile(r"tail", re.IGNORECASE)
_HEAD_RE = re.compile(r"(head|skull|jaw|neck)", re.IGNORECASE)
_TREE_MAX_LINES = 140


def rig_facts(entry):
    """Anatomy the picture cannot be trusted for, read from one cond.npy species.

    A "chain root" is a joint whose name matches a limb pattern while its parent
    does not -- one hit per limb. Contact joints are the preprocessor's own
    foot detection and count legs independently of how the rig names them.
    """
    names = [str(n) for n in entry.get("joints_names") or []]
    raw_parents = entry.get("parents")
    parents = ([int(p) for p in raw_parents] if raw_parents is not None
               else [-1] * len(names))

    def ancestors(j):
        seen = set()
        while 0 <= parents[j] < len(names) and parents[j] not in seen:
            j = parents[j]
            seen.add(j)
            yield j

    def chain_roots(hit):
        """Joints that are hits with no hit above them -- one per limb, even
        when an unmatched joint (a horse's ``HorseLink``) sits mid-chain."""
        return [names[j] for j in range(len(names))
                if hit(j) and not any(hit(a) for a in ancestors(j))]

    contact_set = {str(n) for n in entry.get("contact_joint_names") or []}

    children = {}
    for j, p in enumerate(parents):
        children.setdefault(p, []).append(j)
    lines = []
    truncated = False

    def walk(index, depth):
        nonlocal truncated
        if len(lines) >= _TREE_MAX_LINES:
            truncated = True
            return
        lines.append("  " * depth + names[index])
        for child in children.get(index, []):
            walk(child, depth + 1)

    for root in children.get(-1, []):
        walk(root, 0)
    if truncated:
        lines.append("  ... (%d joints total, tree truncated)" % len(names))
    return {
        "n_joints": len(names),
        "tree": lines,
        # Sorted joint names are the rig signature: two species with the same
        # set are the ones the joint-name embedding cannot tell apart either.
        "signature": tuple(sorted(names)),
        "legs": chain_roots(lambda j: bool(_LEG_RE.search(names[j]))),
        "wings": chain_roots(lambda j: bool(_WING_RE.search(names[j]))),
        "arms": chain_roots(lambda j: bool(_ARM_RE.search(names[j]))),
        # One per foot: the preprocessor marks every joint of a foot.
        "contacts": chain_roots(lambda j: names[j] in contact_set),
        "has_tail": any(_TAIL_RE.search(n) for n in names),
        "has_head": any(_HEAD_RE.search(n) for n in names),
    }


SPECIES_SYSTEM = (
    "You tag rigged 3D characters for a cross-skeleton motion-generation "
    "dataset. You are given one clip of the character moving and the actual "
    "joint tree of its skeleton. Answer only with the requested JSON."
)

SPECIES_RULES = """\
TASK
Produce two tags describing HOW THIS CHARACTER MOVES: body plan, locomotion.
They are joined into one short text that conditions the model on the species, and
tag 1 also decides which training subset the species is grouped into. Judge the
creature, not the picture: how it is lit and how big it is drawn carry no
information. BOTH TAGS ARE CLOSED SETS: answer with exactly one listed word each,
spelled as listed. Any other word is rejected.

TAG 1 -- body_plan, closed set: {subsets}
  Decide by PRIMARY locomotion, then by leg count:
    - travels through the AIR as its main way of moving -- flies, hovers,
      levitates -- and HAS WINGS or rotors                   -> winged
      (this wins over leg count: a dragon with four legs and wings is winged)
    - swims as its main way of moving                       -> aquatic
    - otherwise count the LEG CHAINS IN THE RIG below:
        2 -> biped     4 -> quadruped     5 or more -> multiped
    - NO LEG CHAINS AT ALL: ask WHAT PROPELS THE BODY, not how high it rides.
        the body itself DEFORMS to push against the ground -- a travelling wave
        running down the spine, peristalsis                  -> serpentine
        the body does NOT deform to propel itself: it is carried along as one
        piece, with at most a bob, a sway or a spin          -> drifting
        (a ghost, a flame, a wingless golem, a hovering robot, a slime, a rolling
        shell, a robed figure whose hem slides over the floor. Hovering or
        skimming the ground both count; nothing pushes off anything.)
      A rig with leg chains that NEVER bear weight -- they hang or stay folded --
      counts as no leg chains here: a leg that never touches down is decoration.
  THE RIG WINS OVER THE PICTURE for leg count. Cloth, tentacles, decoration and
  weapons look like limbs and are not; a rigged leg chain is. Report the number
  you settled on in leg_count (0 when it has none).

TAG 2 -- locomotion, closed set: {locomotions}
  EXACTLY ONE of these words -- no modifier, no role, no style, no invented
  gerund. It names this species' SIGNATURE way of travelling, decided from WHAT
  THE CREATURE IS: its name, its build, its anatomy. When none fits perfectly,
  pick the nearest one.
  IT IS NOT THE GAIT IN THE CLIP. The clip is one arbitrary sample and is usually
  a walk cycle. A horse is Galloping even when the clip shows it walking; a bear
  is Lumbering even in a slow plod; a cat is Stalking even when it trots past.

THE CLIP
One clip of this character, "{clip}" -- the only picture you get. Use it to see
the body and to rule out an obvious error about how the body travels at all (it
floats, hovers, rolls or drags itself rather than stepping on its legs). Its NAME
counts as evidence: a take the artist called FlyForward belongs to a species that
travels through the air. It does NOT name the gait. It is rendered on a 1x1 grid
at ground level; the character may float above the grid or sink into it, and
GROUND CONTACT IS NOT EVIDENCE about whether it flies. Playback is far slower than
the real action, so speed on screen means nothing.

CHARACTER
  id: {species}          name reads as: "{label}"

RIG (parsed from the actual skeleton -- authoritative for anatomy)
  joints: {n_joints}
  leg chains ({n_legs}): {legs}
  foot contact joints ({n_contacts}): {contacts}
  wing/rotor chains ({n_wings}): {wings}
  arm chains ({n_arms}): {arms}
  head: {has_head}   tail: {has_tail}
  joint tree:
{tree}
"""

SPECIES_SCHEMA_KEYS = ("body_plan", "leg_count", "locomotion", "why", "confidence")
# The keys the tags are built from; decoded first, so a stall inside `why`
# never strands them.
SPECIES_TAG_KEYS = ("body_plan", "leg_count", "locomotion")


def species_schema():
    return {
        "type": "object",
        "properties": {
            "body_plan": {"type": "string", "enum": list(CANONICAL_OBJECT_SUBSETS)},
            "leg_count": {"type": "integer"},
            "locomotion": {"type": "string", "enum": list(SPECIES_LOCOMOTIONS)},
            "why": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": list(SPECIES_SCHEMA_KEYS),
        "additionalProperties": False,
    }


def validate_species(reply):
    """``(tags, legs, problems)`` -- tags in sidecar spelling."""
    problems = []
    tags = [normalize_species_tag(reply.get("body_plan") or ""),
            normalize_species_tag(reply.get("locomotion") or "")]
    try:
        check_species_tags(tags, "reply")
    except ValueError as exc:
        problems.append(str(exc))
    try:
        legs = int(reply.get("leg_count"))
    except (TypeError, ValueError):
        legs = -1
        problems.append("leg_count is not an integer")
    if not problems:
        body = tags[0].lower()
        if body == "biped" and legs != 2:
            problems.append("biped with leg_count=%d" % legs)
        if body == "quadruped" and legs != 4:
            problems.append("quadruped with leg_count=%d" % legs)
        if body == "multiped" and legs < 5:
            problems.append("multiped with leg_count=%d" % legs)
    return tags, legs, problems


# Clip names that name a way of travelling, ranked: a species ships several and
# "Walk" answers "how does it move" better than "TurnLeft".
_LOCOMOTION_PRIORITY = (
    "walk", "run", "gallop", "trot", "fly", "swim", "crawl", "slither", "roll",
    "hover", "glide", "float", "drift", "march", "sprint", "dash", "jog", "move",
    "patrol", "climb", "jump", "hop", "leap", "step", "strafe", "turn",
)
_LOCOMOTION_NAME_RE = re.compile("(%s)" % "|".join(_LOCOMOTION_PRIORITY), re.IGNORECASE)
_DIRECTION_RE = re.compile(r"(backward|forward|back|left|right|strafe|reverse|up|down)",
                           re.IGNORECASE)


def _locomotion_rank(action):
    """The plainest forward gait first: kind of travel, then off-axis words, then length."""
    low = action.lower()
    off_axis = sum(1 for w in _DIRECTION_RE.findall(low) if w.lower() != "forward")
    for rank, word in enumerate(_LOCOMOTION_PRIORITY):
        if word in low:
            return (rank, off_axis, len(action), action)
    return (len(_LOCOMOTION_PRIORITY), off_axis, len(action), action)


def pick_species_clip(clips):
    """``(clip record, how)`` -- the clip whose GIF shows the species."""
    with_gif = [c for c in clips if c["gif"] is not None]
    if not with_gif:
        return None, "none"
    loco = sorted((c for c in with_gif if c["group"] == "locomotion"),
                  key=lambda c: _locomotion_rank(c["action"]))
    if loco:
        return loco[0], "locomotion group"
    named = sorted((c for c in with_gif if _LOCOMOTION_NAME_RE.search(c["action"])),
                   key=lambda c: _locomotion_rank(c["action"]))
    if named:
        return named[0], "clip name"
    return sorted(with_gif, key=lambda c: (len(c["action"]), c["action"]))[0], "any clip"


def annotate_species(args, job, sink):
    species, facts, chosen = job["species"], job["facts"], job["clip"]
    prompt = SPECIES_RULES.format(
        subsets=", ".join(CANONICAL_OBJECT_SUBSETS),
        locomotions=", ".join(SPECIES_LOCOMOTIONS),
        clip=chosen["action"], species=species, label=readable_species(species),
        n_joints=facts["n_joints"],
        n_legs=len(facts["legs"]), legs=", ".join(facts["legs"]) or "(none found)",
        n_contacts=len(facts["contacts"]),
        contacts=", ".join(facts["contacts"]) or "(none found)",
        n_wings=len(facts["wings"]), wings=", ".join(facts["wings"]) or "(none found)",
        n_arms=len(facts["arms"]), arms=", ".join(facts["arms"]) or "(none found)",
        has_head="yes" if facts["has_head"] else "no",
        has_tail="yes" if facts["has_tail"] else "no",
        tree="\n".join("    " + line for line in facts["tree"]),
    )
    if args.dry_run:
        log("=" * 78 + "\n" + prompt + "\nvideo: %s" % chosen["gif"])
        return None

    schema = species_schema()
    reply = call_llm(args, SPECIES_SYSTEM, prompt, chosen["gif"], schema,
                     "species_tags", essential=SPECIES_TAG_KEYS)
    tags, legs, problems = validate_species(reply)
    if problems:                                       # one repair attempt, then flag
        repair = prompt + ("\n\nYour previous answer was rejected:\n  - %s\n"
                           "Answer again, fixing exactly those points."
                           % "\n  - ".join(problems))
        try:
            reply = call_llm(args, SPECIES_SYSTEM, repair, chosen["gif"], schema,
                             "species_tags", essential=SPECIES_TAG_KEYS)
            tags, legs, problems = validate_species(reply)
        except RuntimeError as exc:
            problems.append("repair call failed: %s" % exc)

    warnings = list(problems)
    if legs not in (len(facts["legs"]), len(facts["contacts"])):
        warnings.append("rig has %d leg chains and %d feet, model said %d"
                        % (len(facts["legs"]), len(facts["contacts"]), legs))
    review = {
        "species": species, "species_tags": tags, "previous_tags": job["previous"],
        "leg_count": legs, "rig_leg_chains": facts["legs"],
        "rig_contacts": facts["contacts"], "rig_wing_chains": facts["wings"],
        "rig_joints": facts["n_joints"], "clip": chosen["clip"],
        "clip_from": job["clip_from"], "rig_siblings": job["siblings"],
        "why": str(reply.get("why") or "").strip(),
        "confidence": reply.get("confidence"),
        # No confidence means the reply was salvaged from a stalled decode.
        "needs_review": bool(warnings) or reply.get("confidence") in (None, "low"),
        "warnings": warnings,
    }
    main = {"species": species, "species_tags": tags}
    sink.write([main] if not problems else [], [review])
    log("  %-28s %-11s %-12s %-6s %s%s" % (
        species, tags[0], tags[1], reply.get("confidence"), job["clip_from"],
        "  [REVIEW]" if review["needs_review"] else ""))
    return review


def cmd_species(args):
    todo_total, results_all = 0, []
    for ds in load_datasets(args.dataset):
        cond_path = ds["processed"] / COND_FILE
        if not cond_path.is_file():
            print("[SKIP] %s: no %s -- the rig comes from it (preprocess first)"
                  % (ds["ns"], COND_FILE))
            continue
        cond = load_cond(cond_path)
        clips = load_clips(ds)
        by_species = {}
        for clip in clips:
            by_species.setdefault(clip["species"], []).append(clip)
        previous = {row.get("species"): row.get("species_tags")
                    for row in read_jsonl(ds["processed"] / SPECIES_TAGS_FILE)}
        out = draft_dir(ds, args.out_root)
        sink = Sink(out / SPECIES_TAGS_FILE, out / "species_tags.review.jsonl", "species")

        # Planned over ALL species, never the filtered subset: rig siblings are
        # a property of the whole library.
        facts = {}
        for entry in cond.values():
            name = str(entry.get("species_name") or entry.get("object_type"))
            facts[name] = rig_facts(entry)
        by_rig = {}
        for name, fact in facts.items():
            by_rig.setdefault(fact["signature"], []).append(name)

        wanted = None
        if args.species_from:
            wanted = {line.split("#", 1)[0].strip() for line in
                      Path(args.species_from).read_text(encoding="utf-8").splitlines()}
            wanted.discard("")
        jobs = []
        for name in sorted(facts):
            if not fnmatch.fnmatch(name, args.filter):
                continue
            if wanted is not None and name not in wanted:
                continue
            if not args.overwrite and name in sink.done:
                continue
            chosen, how = pick_species_clip(by_species.get(name, []))
            if chosen is None:
                print("  [WARN] %s/%s: no review GIF -- run render_gifs.py" % (ds["ns"], name))
                continue
            jobs.append({"species": name, "facts": facts[name], "clip": chosen,
                         "clip_from": how, "previous": previous.get(name),
                         "siblings": sorted(s for s in by_rig[facts[name]["signature"]]
                                            if s != name)})
        if args.limit:
            jobs = jobs[:args.limit]
        if args.overwrite and not args.dry_run:
            sink.invalidate(job["species"] for job in jobs)
        todo_total += len(jobs)
        print("%s: %d species to do -> %s" % (ds["ns"], len(jobs), out))
        if not jobs:
            continue
        results, failures = run_pool(args, jobs, lambda j: annotate_species(args, j, sink),
                                     lambda j: "%s/%s" % (ds["ns"], j["species"]))
        if args.dry_run:
            continue
        sink.sort_in_place()
        results_all.extend(results)
        print("%s: %d written, %d failed" % (ds["ns"], len(results), len(failures)))
        changed = [r["species"] for r in results
                   if r["previous_tags"] and list(r["previous_tags"]) != r["species_tags"]]
        if changed:
            print("  differs from the dataset's species_tags (%d): %s"
                  % (len(changed), ", ".join(changed)))
        report_rig_collisions(read_jsonl(sink.main_path), facts)
    review = [r["species"] for r in results_all if r["needs_review"]]
    if review:
        print("needs review (%d): %s" % (len(review), ", ".join(review)))


def report_rig_collisions(rows, facts):
    """Same skeleton + same descriptor: nothing in the conditioning separates them.

    Two species on DIFFERENT skeletons sharing a descriptor is fine -- the
    joint-name channel still tells them apart.
    """
    by_rig = {}
    for row in rows:
        fact = facts.get(row["species"])
        if fact is not None:
            by_rig.setdefault(fact["signature"], []).append(row)
    clashes = []
    for members in by_rig.values():
        by_text = {}
        for row in members:
            by_text.setdefault(" ".join(row["species_tags"]), []).append(row["species"])
        clashes.extend((text, sorted(g)) for text, g in by_text.items() if len(g) > 1)
    if not clashes:
        print("  [OK] no two species share a skeleton and a descriptor")
        return
    print("  SAME SKELETON + SAME DESCRIPTOR (%d group(s)):" % len(clashes))
    for text, group in sorted(clashes, key=lambda kv: -len(kv[1])):
        print("    %-30s %s" % (text, ", ".join(group)))


# ═══════════════════════════ action labels ══════════════════════════════════
ACTIONS_SYSTEM = (
    "You label animation clips of rigged 3D characters for a motion-generation "
    "dataset. Every answer is CONTROLLED KEYWORDS, never prose. Answer only with "
    "the requested JSON."
)

GROUP_RULES = {
    "locomotion": """\
locomotion -- the character TRAVELS; the ground slides underneath it.
  - The head is the travel mode: walk, run, fly, swim, crawl -- exactly one of
    those -- or roll, jump, fall, hover, which may also ride on one ("run, jump").
  - fly is for a body held up by WINGS. A legless hoverer drifting along is not
    flying: it is walk or run by pace.
  - Gait and manner are modifiers: "walk, forward, trot" (between walking and
    running), "run, forward, fast" (sprint, dash), "fly, forward, glide" (held
    wings), "walk, backward, retreat" (backing away). A sideways step with the
    body still facing ahead is just "walk, left" / "walk, right".
  - The label MUST name a direction, even for a clip authored in place: the
    heading the gait is trying to go. A hover held in place is exempt.""",
    "stationary": """\
stationary -- the character acts WITHOUT travelling across the ground.
  - Heads: idle (any standing / waiting / breathing / emote), attack, hurt (taking
    a hit), roar (its own head -- "roar", never "idle, roar"), rest (lying,
    downed), work, turn, hover (a flyer holding station).
  - Idle variants take what they do: "idle, look", "idle, eat", "idle, sleep",
    "idle, scratch" (grooming, licking), "idle, taunt", "idle, happy",
    "idle, talk", "idle, sit", "idle, rear".
  - Attacks name the strike whenever one is visible: bite, headbutt, kick, swat
    (unarmed lateral sweep of paw / claw / tail), sting, stab, slash, punch,
    smash (heavy overhead or ground slam), throw, spit, cast (magic),
    firebreath, projectile, charge (wind-up or lunge in place), spin; and the
    implement when the strike IS its motion: bow, gun, hammer, shield.
  - rear means the front of the body comes UP off the ground: "idle, rear",
    "attack, rear".
  - Write a direction word only for a strike or a step that really goes one way.""",
    "transition": """\
transition -- the character CHANGES STATE, once, and does not return.
  - Name the change with ONE event word; the direction of the change is in the
    word, never in the order of words: draw, sheathe, stop (a run skids to a
    halt), getup (from the ground back onto its feet), kneel, laydown, sitdown,
    land, takeoff, die, spawn (arrives, emerges, digs out), burrow (digs in,
    leaves), jump, pickup, putdown, and "rear, up" / "rear, down".
  - A SECOND head only says in what posture or context the event happens, event
    word first: "land, hover" (lands out of hovering flight), "land, run",
    "land, jump" (lands out of a jump), "die, hover" (dies in the air).
  - Never write a from-state and a to-state ("run, idle", "idle, attack"): pick
    the event word.
  - Do not spell out an ending state the event already implies: "land", never
    "land, idle"; "getup", never "getup, idle".
  - A turn in place is "turn, left" / "turn, right". A death names the way the
    body falls to the ground ("die, backward") when it plainly falls one way.""",
}

ACTIONS_RULES = """\
TASK
For every clip listed under CLIPS below, return one action_label. Those labels
are the whole answer -- no prose, and no action_group: the group is decided.

WHAT YOU ARE LOOKING AT
The clip as a short video ({frame_count} rendered frames of {src_frames} source
frames), on a 1x1 checker ground from a fixed oblique view. The camera only
trucks sideways to keep the character framed, so TRAVEL SHOWS UP AS THE CHECKER
SLIDING UNDER IT. Playback is several times slower than the real action.

WHAT THE RENDER DOES NOT TELL YOU
1. LEFT AND RIGHT are the CHARACTER's own, and you cannot read them reliably off
   this render. Take left / right from the clip name (Left / Right / L / R).
   When the name has none, leave the side out unless the motion is plainly
   one-sided -- an omitted direction means "any direction", which is never
   wrong, and the missing side is filled in later from the motion itself.
   forward / backward / up / down: the name is the prior; when it says nothing,
   the video decides (the checker sliding, the body rising or sinking). The "Up"
   in GetUp, StandUp, WakeUp, PickUp, LevelUp is not a direction.
2. GROUND CONTACT MEANS NOTHING. Flying, swimming and hovering clips are usually
   authored in place, so the character may be drawn sunk into the grid. When the
   name says Fly / Swim / Hover / Glide / Dive / Jump / Fall, BELIEVE THE NAME.
3. STAYING IN PLACE PROVES NOTHING. Gaits are very often animated in place. An
   in-place gait still gets its direction -- the heading it is trying to go.
4. SPEED IS INVISIBLE at this playback rate. Write "fast" only when the name says
   Fast / Sprint / Dash, "slow" only when it says Slow, never both.
5. WHAT THE CHARACTER HOLDS IS NOT LABELLED. An armed and an empty-handed swing
   are the same label; a visible prop never changes it. bow / gun / hammer /
   shield name the MOTION the implement makes (drawing a bow, firing, hammering,
   a shield bash), never the fact of holding one.

ACTION GROUP -- ALREADY DECIDED
Every clip below is {group_rules}

action_label -- CONTROLLED KEYWORDS, LOWER CASE, COMMA-SEPARATED
Only words from these lists, each spelled exactly as listed, at most {max_words}
words, no repeats. Write the head word(s) first, the main one first; the rest of
the order is fixed for you afterwards.
{blocks}

RULES THAT ARE NOT OPTIONAL
- ONLY WHAT YOU ACTUALLY SEE. A label is a coordinate, not a caption: one extra
  word points the whole clip somewhere else. Three right words beat six hopeful
  ones.
- These are aimed nowhere and NEVER take forward / backward / left / right:
  {no_planar}. up / down still apply to them.
- A death is just "die" -- never "die, fall", never "die, idle".
- Never "walk, run": the gait between them is "walk, trot".
- If the clip name puts a SECOND action on top of the head (Attack, Bite, Roar,
  Jump, Shot) and the lists have a word for it, name it: it is what separates
  this clip from the plain version. Do not invent one the lists cannot spell.
{loop}{siblings}{context}{variant}
REFERENCE LABELS a human reviewer wrote for {group} clips like these:
{examples}

SPECIES
  {species} ("{label}")

CLIPS -- return one label for every one of these
{members}
"""

SIBLING_RULES = """
THESE CLIPS ARE ONE FAMILY
They are the same motion re-authored for the other side, so:
  - MIRROR THEM. The Left (L) member and the Right (R) member carry the SAME
    label except that left and right swap -- same heads, same modifiers.
  - Only ONE video is attached, for {video_clip}. The others are the same
    action; what differs between them is in their names.
{fixed}"""

FIXED_LINE = '  - {clip} is already labelled "{label}" by a human; mirror it.\n'

CONTEXT_BLOCK = """
THE REST OF THIS SPECIES' {group} LIBRARY
{bases}
Every one of those gets its own label. Do not spend on this clip a label that one
of them plainly needs.
"""

VARIANT_RULE = """
SAME MOTION, DIFFERENT PROP
{lines}
It is the SAME motion -- the only difference is what the hands hold. Keep the
same action words; do not re-read it as a different action because the arms move
differently: a landing while holding a weapon is still a landing.
"""

# Said only when the clip is flagged looping, never the other way round. The
# flag says the ends join, NOT that the character stays off the ground.
LOOP_RULE = """
THIS CLIP CYCLES
It is flagged looping: the last frame runs straight back into the first.
  - Do not write "land" or "takeoff" for a contact you cannot see. A cycle that
    is airborne throughout is the mid-air stretch of a jump.
  - THE VIDEO STILL DECIDES: if you SEE a push-off or a touch-down inside the
    cycle, say so.
"""


def labels_schema(clips):
    """One string field per clip, keyed by clip name -- flat, so a truncated
    reply can still be salvaged."""
    return {
        "type": "object",
        "properties": {clip: {"type": "string"} for clip in clips},
        "required": list(clips),
        "additionalProperties": False,
    }


# ── few-shot ─────────────────────────────────────────────────────────────────
def _name_tokens(text):
    return {w for w in decamel(text).split() if len(w) > 1}


class FewShot:
    """Reviewed rows of every dataset, retrieved per call by name and species.

    The reviewed rows are the only source of convention there is; a clip pulls
    the ones closest to it, widest label variety first so a run of identical
    rows cannot take the whole block and teach one answer.
    """

    def __init__(self, clips, n):
        self.n = n
        self.by_group = {}
        for clip in clips:
            if clip["reviewed"] and clip["label"] and not clip["pending_delete"]:
                self.by_group.setdefault(clip["group"], []).append(
                    dict(clip, _tok=_name_tokens(clip["action"])))

    def block(self, group, species, actions, exclude):
        cands = [c for c in self.by_group.get(group, [])
                 if (c["ns"], c["clip"]) not in exclude]
        if not cands or self.n <= 0:
            return "  (none)"
        tok = set().union(*(_name_tokens(a) for a in actions))

        def score(entry):
            overlap = len(tok & entry["_tok"]) / max(1, len(tok | entry["_tok"]))
            return (1.0 if entry["species"] == species else 0.0) + overlap

        ranked = sorted(cands, key=lambda c: (-score(c), c["clip"]))[:self.n * 4]
        picked, used = [], set()
        for entry in ranked:
            if entry["label"] not in used and len(picked) < self.n:
                picked.append(entry)
                used.add(entry["label"])
        for entry in ranked:
            if len(picked) >= self.n:
                break
            if entry not in picked:
                picked.append(entry)
        return "\n".join('  %-24s %-28s -> "%s"'
                         % (readable_species(e["species"])[:24], e["action"][:28],
                            e["label"]) for e in picked)


# ── label resolution and checks ──────────────────────────────────────────────
def resolve_label(raw, head_order=None):
    """``(label, words, unknown, notes)`` -- one model answer as canonical tokens.

    Exact tokens only, like ``parse_action_label``: a piece that is not a
    vocabulary word is reported, never translated or dropped silently.
    *head_order* maps a word set to the head order the dataset already spells
    it with; head order is the condition, so a draft follows the corpus.
    """
    words, unknown, notes = [], [], []
    for chunk in re.split(r"[,;/]+", str(raw or "").lower()):
        for piece in chunk.split():
            piece = piece.strip(" .\"'")
            if not piece:
                continue
            if piece not in CONTROLLED_VOCAB:
                unknown.append(piece)
            elif piece not in words:
                words.append(piece)
    if not takes_planar_direction(words):
        stripped = [w for w in words if w in PLANAR_DIRECTIONS]
        if stripped:
            words = [w for w in words if w not in stripped]
            notes.append("dropped %s: %s is aimed nowhere"
                         % ("/".join(stripped),
                            "/".join(sorted(set(words) & set(NO_PLANAR_DIRECTION_WORDS)))))
    heads = head_words_in(words)
    known = (head_order or {}).get(frozenset(words))
    if known and len(heads) > 1 and tuple(heads) != known:
        words = list(known) + [w for w in words if w not in known]
    try:
        label = canonical_action_label(words)
    except ActionLabelError:
        label = ""
    return label, words, unknown, notes


def validate_label(action, label, words, unknown, group, loop):
    """Per-clip checks -> ``(problems, notes)``.

    *problems* are worth one repair call; *notes* only mark the row for a human.
    """
    problems, notes = [], []
    word_set = set(words)
    if unknown:
        problems.append("%s are not vocabulary words -- write only listed words"
                        % ", ".join(repr(u) for u in unknown))
    if not label:
        problems.append("action_label is empty")
    else:
        try:
            parse_action_label(label)
        except ActionLabelError as exc:
            problems.append(str(exc).split(" Valid tokens")[0])

    # R4: a locomotion label carries a heading.
    heading = set(PLANAR_DIRECTIONS) | set(VERTICAL_WORDS)
    exempt = (word_set & set(NO_HEADING_WORDS)) or not takes_planar_direction(words)
    if group == "locomotion" and label and not exempt and not (word_set & heading):
        problems.append("a locomotion label must name a direction (%s) -- in-place "
                        "gaits included" % "/".join(PLANAR_DIRECTIONS))

    # R5: one axis, one word.
    for pair, advice in _CONFLICTS:
        if set(pair) <= set(CONTROLLED_VOCAB) and set(pair) <= word_set:
            problems.append("'%s' and '%s' together -- %s" % (pair[0], pair[1], advice))
    modes = [w for w in words if w in TRAVEL_MODES]
    if len(modes) > 1:
        problems.append("names %d travel media (%s) -- a clip travels in one"
                        % (len(modes), ", ".join(modes)))

    claimed_speed = (word_set & set(_NAME_SPEED)) - name_speed_words(action)
    if claimed_speed:
        problems.append("the clip name does not say %s, and speed is not visible at "
                        "this playback rate -- drop it" % "/".join(sorted(claimed_speed)))

    name_dirs = name_direction_words(action)
    label_planar = word_set & set(PLANAR_DIRECTIONS)
    opposite = {"left": "right", "right": "left",
                "forward": "backward", "backward": "forward"}
    contradicted = {w for w in label_planar if opposite[w] in name_dirs}
    if contradicted:
        notes.append("label says %s but the clip name says the opposite"
                     % "/".join(sorted(contradicted)))
    side_unsupported = label_planar & {"left", "right"} - name_dirs
    if side_unsupported:
        notes.append("label says %s but the clip name does not"
                     % "/".join(sorted(side_unsupported)))
    missing_vertical = (name_dirs & {"up", "down"}) - word_set
    if missing_vertical:
        notes.append("the clip name says %s but the label does not"
                     % "/".join(sorted(missing_vertical)))
    if loop and (word_set & {"land", "takeoff"}):
        notes.append("the clip loops but the label says %s -- check the contact is "
                     "really in frame" % "/".join(sorted(word_set & {"land", "takeoff"})))
    if _AIRBORNE_NAME_RE.search(action) and label and not (word_set & _AIRBORNE_WORDS):
        notes.append("airborne clip name but the label names nothing airborne")
    return problems, notes


def validate_family(members, fixed, resolved):
    """Cross-clip checks inside one family -> ``{action: [problem, ...]}``.

    Mirror consistency is SYMMETRIC, like the audit's R3: it says the pair
    disagrees with itself, never which half is wrong. A mirror pair spelled
    identically (no side word) is a valid mirror. Two non-mirror members with
    one label are a problem: whatever separates them is in their names.
    """
    labels = {a: resolved[a]["label"] for a in members}
    labels.update(fixed)
    problems = {a: [] for a in members}
    sides = {}
    for action in labels:
        side = clip_side(action)
        if side:
            sides.setdefault(side, []).append(action)
    pair = None
    if len(sides.get("L", ())) == 1 and len(sides.get("R", ())) == 1:
        pair = (sides["L"][0], sides["R"][0])
        left, right = pair
        if labels[left] and labels[right] and mirror_label(labels[left]) != labels[right]:
            message = ("%s %r and %s %r are mirrors: they must be the same label with "
                       "left and right swapped" % (left, labels[left], right, labels[right]))
            for action in pair:
                if action in problems:
                    problems[action].append(message)
    seen = {}
    for action, label in labels.items():
        if label:
            seen.setdefault(label, []).append(action)
    for label, actions in seen.items():
        if len(actions) < 2 or (pair and set(actions) == set(pair)):
            continue
        for action in actions:
            if action in problems:
                problems[action].append(
                    "%s carry the identical label %r; separate them by what their "
                    "names differ by" % (" and ".join(sorted(actions)), label))
    return problems


def _member_line(clip, video_clip):
    marks = []
    if clip["action"] == video_clip:
        marks.append("video attached")
    if clip["is_loop"]:
        marks.append("cycle")
    return '  %-28s reads as: "%s"%s' % (
        clip["action"], decamel(clip["action"]),
        ("   [%s]" % ", ".join(marks)) if marks else "")


def annotate_family(args, job, sink, fewshot, head_order):
    """Label one family of clips in a single call. Returns the review rows."""
    species, group = job["species"], job["group"]
    members = job["members"]
    by_action = {c["action"]: c for c in members}
    actions = [c["action"] for c in members]
    # One video: Left first, then the unsided base -- fixed, so a re-run sends
    # the same frames.
    ordered = sorted((c for c in members if c["gif"] is not None),
                     key=lambda c: ({"L": 0, None: 1, "R": 2}[clip_side(c["action"])],
                                    c["action"]))
    if not ordered:
        raise RuntimeError("no review GIF for %s -- run render_gifs.py" % ", ".join(actions))
    video = ordered[0]
    frame_count = gif_frame_count(video["gif"])

    variant_lines = []
    for clip in members:
        base, prop = variant_base(clip["action"], job["library_labels"])
        if base and job["library_labels"].get(base):
            variant_lines.append('  %s is the "%s" variant of %s, labelled "%s".'
                                 % (clip["action"], prop, base, job["library_labels"][base]))
    other_bases = sorted(set(job["library_bases"]) - {job["base"]})
    context = ""
    if other_bases:
        shown = other_bases[:40]
        listing = "\n".join("  " + name for name in shown)
        if len(other_bases) > len(shown):
            listing += "\n  ... +%d more" % (len(other_bases) - len(shown))
        context = CONTEXT_BLOCK.format(group=group.upper(), bases=listing)
    fixed = job["fixed"]
    siblings = ""
    if len(members) + len(fixed) > 1:
        siblings = SIBLING_RULES.format(
            video_clip=video["action"],
            fixed="".join(FIXED_LINE.format(clip=a, label=l) for a, l in sorted(fixed.items())))

    prompt = ACTIONS_RULES.format(
        frame_count=frame_count or "all", src_frames=video["src_frames"] or "?",
        group_rules=GROUP_RULES.get(group, group),
        max_words=ACTION_LABEL_MAX_WORDS, blocks=vocab_block_text(),
        no_planar=", ".join(NO_PLANAR_DIRECTION_WORDS),
        loop=LOOP_RULE if any(c["is_loop"] for c in members) else "",
        siblings=siblings, context=context,
        variant=VARIANT_RULE.format(lines="\n".join(variant_lines)) if variant_lines else "",
        group=group,
        examples=fewshot.block(group, species, actions,
                               {(c["ns"], c["clip"]) for c in members}),
        species=species, label=readable_species(species),
        members="\n".join(_member_line(c, video["action"]) for c in members),
    )
    if args.dry_run:
        log("=" * 78 + "\n" + prompt + "\nvideo: %s" % video["gif"])
        return None

    schema = labels_schema(actions)

    def answer(text):
        reply = call_llm(args, ACTIONS_SYSTEM, text, video["gif"], schema, "action_label")
        resolved, problems, notes = {}, {}, {}
        for action in actions:
            label, words, unknown, fixes = resolve_label(reply.get(action), head_order)
            resolved[action] = {"label": label, "words": words, "raw": reply.get(action)}
            problems[action], notes[action] = validate_label(
                action, label, words, unknown, group, by_action[action]["is_loop"])
            notes[action] = fixes + notes[action]
        family_problems = validate_family(actions, fixed, resolved)
        for action in actions:
            problems[action] = problems[action] + family_problems[action]
        return resolved, problems, notes

    resolved, problems, notes = answer(prompt)
    if any(problems.values()):
        listing = "\n".join("  - %s: %s" % (a, m) for a in actions for m in problems[a])
        repair = prompt + ("\n\nYour previous answer was rejected:\n%s\nAnswer again "
                           "for every clip, fixing exactly those points." % listing)
        try:
            resolved, problems, notes = answer(repair)
        except RuntimeError as exc:
            for action in actions:
                problems[action] = problems[action] + ["repair call failed: %s" % exc]

    main_rows, reviews = [], []
    for action in actions:
        clip = by_action[action]
        label = resolved[action]["label"]
        # A row that still fails the contract never reaches the contract file;
        # the review row keeps it for a human.
        if label and not problems[action]:
            main_rows.append({"clip": clip["clip"], "action_group": group,
                              "action_label": label})
        reviews.append({
            "clip": clip["clip"], "species": species, "action": action,
            "action_group": group, "action_label": label,
            "previous_label": clip["label"], "llm_raw": resolved[action]["raw"],
            "family": job["base"],
            "family_labels": dict({a: resolved[a]["label"] for a in actions}, **fixed),
            "needs_review": bool(problems[action] or notes[action]),
            "rejected": bool(problems[action]),
            "warnings": problems[action] + notes[action],
            "is_loop": clip["is_loop"], "video": video["clip"],
        })
    sink.write(main_rows, reviews)
    for review in reviews:
        log("  %-40s %-11s %-28s -> %s%s"
            % (review["clip"][:40], group, review["previous_label"][:28] or "(empty)",
               review["action_label"] or "(empty)",
               "   [REJECTED]" if review["rejected"]
               else "   [REVIEW]" if review["needs_review"] else ""))
    return reviews


def read_clip_list(path):
    """Clip keys from a file: ``<clip>`` or ``<species>/<action>`` per line."""
    wanted = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        species, sep, action = line.replace("\\", "/").partition("/")
        wanted.add(clip_key("%s_%s" % (species, action) if sep else line))
    if not wanted:
        raise SystemExit("no clips in %s" % path)
    return wanted


def dataset_head_order(clips):
    """``{group: {frozenset(words): heads}}`` -- how the dataset spells each word set."""
    order = {}
    for clip in clips:
        if not clip["label"]:
            continue
        words = [w.strip() for w in clip["label"].split(",")]
        heads = tuple(head_words_in(words))
        order.setdefault(clip["group"], {}).setdefault(frozenset(words), heads)
    return order


def cmd_actions(args):
    datasets = load_datasets(args.dataset)
    if not datasets:
        raise SystemExit("no dataset matches %r in %s" % (args.dataset, DATASETS_FILE))
    # Ground truth is read from EVERY dataset: the reviewer's labels in one
    # library are what teach another its conventions.
    all_clips = {ds["ns"]: load_clips(ds) for ds in load_datasets("all")}
    fewshot = FewShot([c for clips in all_clips.values() for c in clips], args.shots)
    wanted = read_clip_list(args.clips_from) if args.clips_from else None
    loop_flag = {"yes": True, "no": False, "unknown": None}.get(args.loop)

    all_results = []
    for ds in datasets:
        clips = [c for c in all_clips[ds["ns"]] if not c["pending_delete"]]
        bad_groups = sorted({c["group"] for c in clips} - set(ACTION_GROUPS))
        if bad_groups:
            raise SystemExit("%s: action_group %s is not one of %s"
                             % (ds["ns"], bad_groups, list(ACTION_GROUPS)))
        head_order = dataset_head_order(clips)

        # Families are built over the WHOLE species, then filtered: a filter
        # that cut a mirror pair in half would give back exactly the
        # disagreement families exist to prevent.
        families, library = {}, {}
        for clip in clips:
            key = (clip["species"], clip["group"], clip_family(clip["action"]))
            families.setdefault(key, []).append(clip)
            library.setdefault((clip["species"], clip["group"]), set()).add(key[2])
        species_labels = {}
        for clip in clips:
            if clip["reviewed"] and clip["label"]:
                species_labels.setdefault(clip["species"], {})[clip["action"]] = clip["label"]

        def matched(clip):
            return (fnmatch.fnmatch(clip["species"], args.filter)
                    and (fnmatch.fnmatch(clip["action"], args.clip)
                         or fnmatch.fnmatch(clip["clip"], args.clip))
                    and (wanted is None or clip["clip"] in wanted)
                    and (args.loop == "any" or clip["is_loop"] is loop_flag))

        out = draft_dir(ds, args.out_root)
        sink = Sink(out / ACTION_LABELS_FILE, out / "action_labels.review.jsonl", "clip")
        jobs, skipped_done, no_gif = [], 0, []
        for (species, group, base), family in sorted(families.items()):
            if args.action_group != "any" and group != args.action_group:
                continue
            if not any(matched(c) for c in family):
                continue
            ask = [c for c in family if args.include_reviewed or not c["reviewed"]]
            fixed = {c["action"]: c["label"] for c in family
                     if c not in ask and c["label"]}
            if not ask:
                continue
            if not args.overwrite and all(c["clip"] in sink.done for c in ask):
                skipped_done += len(ask)
                continue
            if not any(c["gif"] is not None for c in ask):
                no_gif.extend(c["clip"] for c in ask)
                continue
            groups = [ask] if len(ask) <= MAX_FAMILY else [[c] for c in ask]
            for members in groups:
                jobs.append({"species": species, "group": group, "base": base,
                             "members": sorted(members, key=lambda c: c["action"]),
                             "fixed": fixed if len(groups) == 1 else {},
                             "library_bases": library[(species, group)],
                             "library_labels": species_labels.get(species, {})})
        if args.limit:
            jobs = jobs[:args.limit]
        if args.overwrite and not args.dry_run:
            sink.invalidate(
                clip["clip"]
                for job in jobs
                for clip in job["members"]
            )
        todo = sum(len(j["members"]) for j in jobs)
        print("%s: %d call(s), %d clip(s) to label, %d already drafted -> %s"
              % (ds["ns"], len(jobs), todo, skipped_done, out))
        if no_gif:
            print("  [SKIP] %d clip(s) have no review GIF (run render_gifs.py): %s%s"
                  % (len(no_gif), ", ".join(no_gif[:8]), " ..." if len(no_gif) > 8 else ""))
        if not jobs:
            continue
        results, failures = run_pool(
            args, jobs,
            lambda job: annotate_family(args, job, sink, fewshot,
                                        head_order.get(job["group"], {})),
            lambda job: "%s/%s (%d)" % (job["species"], job["base"], len(job["members"])))
        if args.dry_run:
            continue
        sink.sort_in_place()
        reviews = [row for batch in results for row in batch]
        all_results.extend(reviews)
        print("%s: %d clip(s) drafted, %d call(s) failed"
              % (ds["ns"], len(reviews), len(failures)))
    if not args.dry_run and all_results:
        summarize_actions(all_results)


def summarize_actions(reviews):
    changed = [r for r in reviews if r["previous_label"] and r["action_label"]
               and r["action_label"] != r["previous_label"]]
    rejected = [r["clip"] for r in reviews if r["rejected"]]
    review = [r["clip"] for r in reviews if r["needs_review"] and not r["rejected"]]
    print("\ndifferent from the current label: %d of %d" % (len(changed), len(reviews)))
    # A label shared inside one species across families: legal when the clips
    # really are interchangeable, but it is how two distinct takes collapse.
    buckets = {}
    for row in reviews:
        if row["action_label"]:
            buckets.setdefault((row["species"], row["action_group"], row["action_label"]),
                               []).append(row["action"])
    shared = sorted(((k, v) for k, v in buckets.items() if len(v) > 1),
                    key=lambda kv: -len(kv[1]))
    if shared:
        print("shared labels inside one species (%d):" % len(shared))
        for (species, _group, label), actions in shared[:10]:
            print("   %-22s %-28s %s" % (species, "'%s'" % label, ", ".join(actions)))
    if rejected:
        print("REJECTED -- kept out of the contract file (%d): %s%s"
              % (len(rejected), ", ".join(rejected[:12]), " ..." if len(rejected) > 12 else ""))
    if review:
        print("needs review (%d): %s%s"
              % (len(review), ", ".join(review[:12]), " ..." if len(review) > 12 else ""))
    print("next: look at the drafts, then `llm_annotate.py apply`, then "
          "tools/prefill_direction_words.py and the serve.py review pass")


# ═══════════════════════════ apply ══════════════════════════════════════════
def apply_actions(ds, drafts, include_reviewed, dry_run):
    """Write action drafts into the dataset's sidecar; returns rows changed."""
    rows = read_action_label_rows(ds["processed"])
    current = {clip_key(r["clip"]): r for r in rows}
    accepted, refused = {}, []
    for draft in drafts:
        clip = clip_key(draft["clip"])
        row = current.get(clip)
        label = draft.get("action_label") or ""
        if row is None:
            refused.append((clip, "no row in %s" % ACTION_LABELS_FILE))
            continue
        if row.get("pending_delete") or (row.get("reviewed") and not include_reviewed):
            continue
        if draft.get("action_group") != row.get("action_group"):
            refused.append((clip, "drafted as %s, the row is %s"
                            % (draft.get("action_group"), row.get("action_group"))))
            continue
        try:
            tokens = parse_action_label(label)
            if canonical_action_label(tokens) != label:
                raise ActionLabelError("not canonically spelled")
        except ActionLabelError as exc:
            refused.append((clip, str(exc)))
            continue
        if label != row.get("action_label"):
            accepted[clip] = label

    # Within a group one word set has one head order (the loader's gate).
    final = {clip: accepted.get(clip, str(r.get("action_label") or ""))
             for clip, r in current.items()}
    spelled = {}
    for clip, label in final.items():
        if clip in accepted or not label:
            continue
        words = [w.strip() for w in label.split(",")]
        spelled.setdefault((current[clip].get("action_group"), frozenset(words)),
                           tuple(head_words_in(words)))
    for clip in sorted(accepted):
        words = accepted[clip].split(", ")
        key = (current[clip].get("action_group"), frozenset(words))
        heads = tuple(head_words_in(words))
        if spelled.setdefault(key, heads) != heads:
            refused.append((clip, "head order %s contradicts the dataset's %s"
                            % (list(heads), list(spelled[key]))))
            del accepted[clip]

    for clip, reason in refused:
        print("  [REFUSED] %s: %s" % (clip, reason))
    if dry_run:
        for clip in sorted(accepted):
            print("  %-48s %-28s -> %s" % (clip, current[clip].get("action_label"),
                                          accepted[clip]))
        return len(accepted)

    def edit(entry):
        label = accepted.get(clip_key(entry["clip"]))
        return autofill_action_label(entry, label) if label else None

    return rewrite_action_label_rows(ds["processed"], edit)


def apply_species(ds, drafts, dry_run):
    path = ds["processed"] / SPECIES_TAGS_FILE
    rows = read_jsonl(path)
    new = {}
    for draft in drafts:
        tags = [normalize_species_tag(t) for t in draft.get("species_tags") or []]
        try:
            check_species_tags(tags, draft.get("species"))
        except ValueError as exc:
            print("  [REFUSED] %s" % exc)
            continue
        new[draft["species"]] = tags
    changed = 0
    out_rows = []
    for row in rows:
        tags = new.pop(row.get("species"), None)
        if tags is not None and list(row.get("species_tags") or []) != tags:
            print("  %-28s %s -> %s" % (row["species"], row.get("species_tags"), tags))
            row = dict(row, species_tags=tags)
            changed += 1
        out_rows.append(row)
    for species, tags in sorted(new.items()):
        print("  %-28s (new) %s" % (species, tags))
        out_rows.append({"species": species, "species_tags": tags})
        changed += 1
    if changed and not dry_run:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for row in out_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    return changed


def cmd_apply(args):
    for ds in load_datasets(args.dataset):
        out = draft_dir(ds, args.out_root)
        if args.what in ("actions", "both"):
            drafts = read_jsonl(out / ACTION_LABELS_FILE)
            if drafts:
                n = apply_actions(ds, drafts, args.include_reviewed, args.dry_run)
                print("%s: %d action label(s) %s (reviewed:false + autofill)"
                      % (ds["ns"], n, "would change" if args.dry_run else "written"))
            else:
                print("%s: no action drafts in %s" % (ds["ns"], out))
        if args.what in ("species", "both"):
            drafts = read_jsonl(out / SPECIES_TAGS_FILE)
            if drafts:
                n = apply_species(ds, drafts, args.dry_run)
                print("%s: %d species row(s) %s%s"
                      % (ds["ns"], n, "would change" if args.dry_run else "written",
                         "" if args.dry_run or not n else
                         " -- species tags are baked into cond.npy: regenerate it"))
            else:
                print("%s: no species drafts in %s" % (ds["ns"], out))


# ═════════════════════════════════ CLI ══════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(parser):
        parser.add_argument("--dataset", default="all",
                            help="namespace glob from datasets.jsonl, or its last part "
                                 "(unitybundles, zoo, ...)")
        parser.add_argument("--out-root", default=None,
                            help="draft directory root (default: <processed>/review/llm)")

    def llm(parser):
        common(parser)
        parser.add_argument("--filter", default="*", help="species glob, e.g. 'RMW_*'")
        parser.add_argument("--limit", type=int, default=0, help="cap the number of calls")
        parser.add_argument("--workers", "-j", type=int, default=MAX_WORKERS)
        parser.add_argument("--overwrite", action="store_true",
                            help="redo entries already in the drafts")
        parser.add_argument("--dry-run", action="store_true",
                            help="print the assembled prompts, call nothing")
        parser.add_argument("--endpoint", default=ENDPOINT)
        parser.add_argument("--model", default=None, help="default: first served model")
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
        parser.add_argument("--timeout", type=float, default=TIMEOUT)
        parser.add_argument("--retries", type=int, default=RETRIES)

    p_species = sub.add_parser("species", help="draft species_tags.jsonl")
    llm(p_species)
    p_species.add_argument("--species-from", default=None, metavar="FILE",
                           help="only the species named in FILE, one per line")

    p_actions = sub.add_parser("actions", help="draft action_labels.jsonl labels")
    llm(p_actions)
    p_actions.add_argument("--clip", default="*",
                           help="glob on the action name ('Fly*') or the clip key")
    p_actions.add_argument("--clips-from", default=None, metavar="FILE",
                           help="only these clips: '<clip>' or '<species>/<action>' per line")
    p_actions.add_argument("--action-group", default="any",
                           choices=("any",) + tuple(ACTION_GROUPS))
    p_actions.add_argument("--loop", default="any", choices=("any", "yes", "no", "unknown"),
                           help="only clips whose is_loop is true / false / absent")
    p_actions.add_argument("--include-reviewed", action="store_true",
                           help="also relabel rows marked reviewed (they stay the "
                                "few-shot examples for everything else)")
    p_actions.add_argument("--shots", type=int, default=SHOTS,
                           help="reviewed examples per call")

    p_apply = sub.add_parser("apply", help="write drafts into the dataset sidecars")
    common(p_apply)
    p_apply.add_argument("--what", default="actions", choices=("actions", "species", "both"))
    p_apply.add_argument("--include-reviewed", action="store_true",
                         help="also overwrite rows marked reviewed")
    p_apply.add_argument("--dry-run", action="store_true", help="report, write nothing")

    args = ap.parse_args()
    _check_vocab()
    if args.mode == "apply":
        cmd_apply(args)
        return 0
    if args.workers > MAX_WORKERS:
        print("--workers %d capped at %d: the local server corrupts JSON under more "
              "concurrency" % (args.workers, MAX_WORKERS))
        args.workers = MAX_WORKERS
    if not args.dry_run:
        args.model = args.model or discover_model(args.endpoint)
        print("endpoint %s  model %s  workers %d" % (args.endpoint, args.model, args.workers))
    started = time.time()
    (cmd_species if args.mode == "species" else cmd_actions)(args)
    print("elapsed %.1fs" % (time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
