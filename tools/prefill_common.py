"""What the two label-prefill tools share.

``prefill_direction_words.py`` walks the
corpus the way the audit does, decide something about a row whose slot is
empty, and write it back as a proposal. The corpus walk, the decode of a
clip into world positions, the proposal record, the CSV and the write are
the same in both, so they live here rather than twice.

Three rules every proposal obeys, enforced here so neither tool can forget:

* A row is only ever FILLED, never rewritten: the edit re-reads the row at
  write time and skips it if the slot is no longer empty, so a word a person
  put there between the dry run and ``--apply`` wins.
* A ``reviewed: true`` row is left alone unless the run explicitly asks for it
  (``--include-reviewed``). That mark means a person watched the GIF and
  settled the label, so an empty slot on such a row is their verdict, not a
  gap. Like the rule above, this is checked again at write time against the
  row as it is on disk, so a row reviewed between the dry run and ``--apply``
  is still spared. A reviewed row remains full evidence: it calibrates and it
  serves as a reference or a mirror partner exactly as before -- it is only
  never the target of a write.
* A written row is marked ``"reviewed": false`` and flagged ``"autofill": true``
  (see ``action_label_sidecar.autofill_action_label``), so the review UI lists it as a
  proposal nobody has signed off yet. The flag says only that a tool wrote the
  label; why it wrote it is in this run's console output and ``--report`` CSV,
  not in the sidecar.

Neither tool renders a page: the rows it fills come back as ``reviewed:false``,
which is what ``dataset/review/serve.py`` + ``index.html`` already filter on, so
the GIFs are reviewed there with everything else.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent.parent
for _candidate in (str(ANYTOP_DIR), str(ANYTOP_DIR.parent)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from data_loaders.truebones.truebones_utils.canonical_features import (  # noqa: E402
    _length_scale_from_cond,
)
from data_loaders.truebones.truebones_utils.cond_schema import load_cond  # noqa: E402
from data_loaders.truebones.truebones_utils.dataset_sources import (  # noqa: E402
    sources_from_cond,
)
from data_loaders.truebones.truebones_utils.features import (  # noqa: E402
    recover_from_bvh_ric_np,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    ACTION_LABELS_FILE,
    LOOP_FLAG_KEY,
    _validate_head_order_consistency,
    canonical_action_label,
    clip_key,
    head_words_in,
    parse_action_label,
)
from tools.action_label_sidecar import (  # noqa: E402
    autofill_action_label,
    read_action_label_rows,
    rewrite_action_label_rows,
)
from tools.audit_action_labels import collect_clips, label_words  # noqa: E402


# ── corpus ──────────────────────────────────────────────────────────────────

def load_corpus(cond_path, action_group=None):
    """``(cond, sources, clips)`` for the whole corpus, minus retiring clips.

    Each clip record is the audit's (clip / species / group / label /
    motion_path / gif_path / labels_path) plus ``key`` (the sidecar key),
    ``root`` (the dataset dir its sidecar lives in), ``reviewed`` and
    ``is_loop`` read off the raw row. A ``pending_delete`` row is dropped:
    the clip is on its way out and must neither receive a word nor serve as
    calibration evidence.
    """
    cond = load_cond(cond_path)
    sources = sources_from_cond(cond, cond_path)
    clips = collect_clips(cond, sources, action_group=action_group)
    marks = {}
    for source in sources:
        for row in read_action_label_rows(source.root):
            marks[(source.root, clip_key(row.get("clip", "")))] = row
    kept = []
    for clip in clips:
        root = str(Path(clip["labels_path"]).parent)
        key = clip_key(clip["clip"])
        row = marks.get((root, key)) or {}
        if row.get("pending_delete"):
            continue
        clip = dict(clip)
        clip["key"] = key
        clip["root"] = root
        clip["reviewed"] = row.get("reviewed") is True
        clip["is_loop"] = row.get(LOOP_FLAG_KEY)
        kept.append(clip)
    return cond, sources, kept


def species_name(clip) -> str:
    """The bare species name (``MB_Unka``) of a clip record."""
    return str(clip["species"]).rsplit("/", 1)[-1]


# ── decode ──────────────────────────────────────────────────────────────────

class DecodedClip:
    """One clip's positions, in the canonical facing frame.

    ``ric`` is the stored root-relative position channel ``(F, J, 3)``: root XZ
    removed, facing +Z, +X to the character's left. ``world`` adds the root's
    integrated XZ travel back (what the clip does on the ground plane -- for a
    locomotion clip the preprocess detrend has already removed the net travel,
    so only the within-cycle surge and sway remain). ``L`` is the skeleton's
    length scale, so distances divided by it read in body lengths.
    """

    def __init__(self, motion_path, entry):
        motion = np.load(motion_path).astype(np.float64)
        self.frames = int(motion.shape[0])
        self.root_index = int(entry["translation_root_index"])
        self.ric = motion[..., :3]
        self.world = recover_from_bvh_ric_np(motion, translation_root_index=self.root_index)
        self.L = float(_length_scale_from_cond(entry))


# ── proposals ───────────────────────────────────────────────────────────────

class Proposal:
    """One row's verdict: a word to write, or a reason it goes to the list.

    ``status`` is one of
      ``write``   -- the measurement decided it and the calibration gate is
                     open: ``--apply`` writes ``proposed``;
      ``review``  -- asymmetric / ambiguous / no evidence usable: listed for
                     a person, nothing written;
      ``keep``    -- decided to be correctly empty (symmetric, still, exempt):
                     reported in the counts only.
    ``evidence`` and ``metrics`` are both for THIS run's output (the console
    summary and the ``--report`` CSV); the sidecar keeps neither.
    """

    def __init__(self, clip, status, reason, proposed="", evidence=None, metrics=None,
                 partner=None):
        self.clip = clip
        self.status = status
        self.reason = reason
        self.proposed = proposed
        self.evidence = dict(evidence or {})
        self.metrics = dict(metrics or {})
        self.partner = partner

    @property
    def label(self) -> str:
        return self.clip["label"]

    def as_row(self) -> dict:
        row = {
            "status": self.status,
            "species": self.clip["species"],
            "clip": self.clip["clip"],
            "group": self.clip["group"],
            "label": self.label,
            "proposed": self.proposed,
            "reason": self.reason,
        }
        for key, value in self.metrics.items():
            row[key] = value
        row["evidence"] = json.dumps(self.evidence, ensure_ascii=False, sort_keys=True)
        return row


def spell_with(label: str, words) -> str:
    """*label* with *words* added, in canonical spelling."""
    return canonical_action_label(list(parse_action_label(label)) + list(words))


# ── the reviewed rule ───────────────────────────────────────────────────────

def add_reviewed_flag(parser) -> None:
    """The ``--include-reviewed`` / ``--skip-reviewed`` pair both tools take."""
    parser.add_argument(
        "--include-reviewed", "--include_reviewed", dest="include_reviewed",
        action="store_true",
        help="Also judge and fill rows marked reviewed:true. Off by default: a "
             "person watched that clip and settled its label, so its empty slot "
             "is their verdict rather than a gap.")
    parser.add_argument(
        "--skip-reviewed", "--skip_reviewed", dest="include_reviewed",
        action="store_false", help="Leave reviewed:true rows alone (the default).")
    parser.set_defaults(include_reviewed=False)


def drop_reviewed(proposals, include_reviewed):
    """*proposals*, minus the ones aimed at a row a person has already signed off.

    Dropped rather than demoted to ``review``: the point of the mark is that
    the clip has BEEN reviewed, so listing it again is noise. Only the target
    of a write is affected -- a reviewed row still calibrates the measurement
    and still serves as a reference or a mirror partner, because that is where
    a hand-checked row is worth most.
    """
    if include_reviewed:
        return list(proposals)
    kept = [p for p in proposals if not p.clip.get("reviewed")]
    dropped = len(proposals) - len(kept)
    if dropped:
        print(f"[OK] {dropped} reviewed row(s) left alone "
              "(pass --include-reviewed to judge and fill them too)")
    return kept


# ── reports ─────────────────────────────────────────────────────────────────

def write_csv(path, proposals) -> int:
    """Every judged row as a CSV line: what the operator reads the run from."""
    rows = [p.as_row() for p in proposals]
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


# ── write-back ──────────────────────────────────────────────────────────────

def _slot_words_of(label: str, slot_vocab) -> list[str]:
    return [word for word in label_words(label) if word in slot_vocab]


def check_head_order(sources, proposals) -> None:
    """Refuse a proposal set that would put two head orders on one word set.

    Runs the sidecar's own cross-row rule over the corpus AS IT WOULD BE after
    the write; it exits the process on a conflict, so nothing is written.
    """
    proposed = {(p.clip["root"], p.clip["key"]): p.proposed
                for p in proposals if p.status == "write"}
    rows = []
    for source in sources:
        for number, row in enumerate(read_action_label_rows(source.root), start=1):
            if row.get("pending_delete"):
                continue
            key = clip_key(row.get("clip", ""))
            label = proposed.get((source.root, key), row.get("action_label", ""))
            tokens = label_words(label)
            rows.append((number, str(row.get("action_group", "")), key, tokens))
    _validate_head_order_consistency(rows)


def apply_proposals(sources, proposals, *, slot, slot_vocab,
                    include_reviewed: bool = False) -> dict[str, int]:
    """Write every ``write`` proposal whose row is STILL empty in *slot*.

    Returns ``{dataset_root: rows written}``. A row that acquired a word in the
    slot since the proposal was computed is skipped and reported: the person
    who put it there outranks the measurement. So is a row marked
    ``reviewed: true`` unless *include_reviewed* -- the tools already drop
    those proposals (``drop_reviewed``), and this second check, on the row as
    it is on disk, catches the one a person reviewed while the run was being
    read.
    """
    by_root: dict[str, dict[str, Proposal]] = {}
    for proposal in proposals:
        if proposal.status != "write":
            continue
        by_root.setdefault(proposal.clip["root"], {})[proposal.clip["key"]] = proposal
    written: dict[str, int] = {}
    for source in sources:
        pending = by_root.get(source.root)
        if not pending:
            continue
        skipped: list[str] = []

        def edit(entry, pending=pending, skipped=skipped):
            key = clip_key(entry.get("clip", ""))
            proposal = pending.get(key)
            if proposal is None or entry.get("pending_delete"):
                return None
            if entry.get("reviewed") is True and not include_reviewed:
                skipped.append(f"{key}: reviewed:true "
                               "(pass --include-reviewed to fill it anyway)")
                return None
            current = str(entry.get("action_label", ""))
            if _slot_words_of(current, slot_vocab):
                skipped.append(f"{key}: now {current!r} (filled by hand since the dry run)")
                return None
            if current != proposal.label:
                # The label changed under us in some other slot: re-spell the
                # proposal on top of what is there now rather than on the
                # snapshot, so the other edit survives.
                proposal.proposed = spell_with(
                    current, _slot_words_of(proposal.proposed, slot_vocab)
                )
            return autofill_action_label(entry, proposal.proposed)

        written[source.root] = rewrite_action_label_rows(source.root, edit)
        for line in skipped:
            print(f"[SKIP] {source.root}/{ACTION_LABELS_FILE}: {line}")
    return written


def summarize(proposals, *, by=("status",)) -> None:
    counts: dict[tuple, int] = {}
    for proposal in proposals:
        key = tuple(getattr(proposal, field, None) if hasattr(proposal, field)
                    else proposal.clip.get(field) for field in by)
        counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts, key=lambda k: tuple(str(x) for x in k)):
        print("  " + " / ".join(str(x) for x in key) + f": {counts[key]}")


def head_of(label: str) -> str:
    heads = head_words_in(label_words(label))
    return heads[0] if heads else ""
