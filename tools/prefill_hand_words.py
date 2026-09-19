#!/usr/bin/env python3
"""Propose the missing HANDS word (``hand1`` / ``hand2``) of an ``action_labels.jsonl`` row.

The hands slot has no training dropout: an empty slot MEANS empty hands (the
content default), and a clip whose character holds something must say
``hand1`` (one hand occupied) or ``hand2`` (both). So every hand-bearing
species' row that holds a weapon or a tool and spells no hand word is a
mislabel that teaches the model "empty" for an armed pose. This tool finds
those rows, under the same rules as the direction prefill:

* Only an EMPTY slot is filled; a hand word already on a row is a verified
  truth and calibrates the tool. A row filled here is marked
  ``"reviewed": false`` and flagged ``"autofill": true``, so it comes back in
  the review UI with the GIFs.
* The verdict comes from the MOTION: the review GIFs of the KI / LH packs do
  not render the props, so what can be seen is the holding POSE. The clip
  name is not consulted.
* ``--dry-run`` is the default; ``--apply`` writes what the calibration allows.

Two layers:

  k-NN ON THE ARM POSE
      Within one species, rows that already carry ``hand1`` / ``hand2`` are the
      references. A clip is described by the temporal median of its arm joints
      (the first joints of each arm chain out of the trunk, relative to the
      trunk joint the chain hangs from, in body lengths) plus the hand-to-hand
      distance -- the holding pose, not the activity. The three nearest
      references vote; a unanimous vote within the distance threshold names
      the class. Calibration is leave-one-out over
      the references, per class: a class under ``--gate`` (0.90 precision) is
      never written.

  SPECIES DEFAULT (``--species-default NAME=hand1,...``)
      A species with no hand word on any row cannot be judged by k-NN. The
      Mini Legion models carry their weapons in the mesh, so the GIF shows
      them: Footman spear + shield (hand2), Mage staff in one hand (hand1),
      DemonHunter twin blades (hand2), Druid staff (hand1). ``DEFAULT_SPECIES``
      lists those four; pass ``--species-default`` to add or override. Every
      empty-slot row of such a species gets the word (a row that already has
      one keeps it). KI_Human / KI_Performer (Basic Motions / Dance) are
      unarmed and stay empty; KI_Slinger (Throwing) holds its projectile only
      during a throw, which is a per-clip judgement and is left to the review.

Usage:
    python tools/prefill_hand_words.py                      # dry run
    python tools/prefill_hand_words.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent.parent
for _candidate in (str(ANYTOP_DIR), str(ANYTOP_DIR.parent)):
    if _candidate not in sys.path:
        sys.path.insert(0, str(_candidate))

from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    HANDS_VOCAB,
)
from tools.audit_action_labels import label_words  # noqa: E402
from tools.prefill_common import (  # noqa: E402
    DecodedClip,
    Proposal,
    apply_proposals,
    check_head_order,
    load_corpus,
    species_name,
    spell_with,
    write_csv,
)

HAND_WORDS = ("hand1", "hand2")

# Species whose rows carry no hand word at all but whose model holds a weapon
# (visible in the review GIF: the Mini Legion packs bake the weapon into the
# mesh). Verified 2026-09-18 on the GIFs; every row written from here is
# reviewed:false like any other proposal.
DEFAULT_SPECIES = {
    "MLH_Footman": "hand2",
    "MLH_Mage": "hand1",
    "MLS_DemonHunter": "hand2",
    "MLS_Druid": "hand1",
}

# Arm chain depth kept in the descriptor: clavicle / upper arm / forearm /
# hand on a full rig; fingers beyond that would swamp the pose.
ARM_CHAIN_DEPTH = 4
KNN_K = 3


# ── arm descriptor ───────────────────────────────────────────────────────────

class ArmLayout:
    """Which joints of a rig describe its holding pose."""

    def __init__(self, entry):
        parents = np.asarray(entry["parents"], dtype=int)
        side = np.asarray(entry["joint_side_labels"])
        count = len(parents)
        children = defaultdict(list)
        for joint, parent in enumerate(parents):
            if parent >= 0:
                children[int(parent)].append(joint)
        # Leg joints: everything on the path from a contact joint up to the
        # first center joint. Whatever sided joints remain are the arms (and
        # wings, shoulder pads -- the same on every clip of one rig, so harmless).
        leg = np.zeros(count, dtype=bool)
        for joint in (int(j) for j in entry.get("contact_joints", [])):
            while joint >= 0 and side[joint] != "center":
                leg[joint] = True
                joint = int(parents[joint])
        self.chains: list[tuple[int, list[int]]] = []   # (trunk joint, arm joints)
        for joint in range(count):
            parent = int(parents[joint])
            if side[joint] == "center" or leg[joint] or parent < 0 or side[parent] != "center":
                continue
            chain = [joint]
            current = joint
            while len(chain) < ARM_CHAIN_DEPTH:
                below = [c for c in children[current] if side[c] == side[joint] and not leg[c]]
                if len(below) != 1:
                    break
                current = below[0]
                chain.append(current)
            self.chains.append((parent, chain))
        self.ok = len(self.chains) >= 2

    def describe(self, decoded: DecodedClip) -> np.ndarray:
        """Temporal median of the arm pose, flattened, in body lengths."""
        ric = decoded.ric
        parts = []
        tips = []
        for trunk, chain in self.chains:
            relative = (ric[:, chain, :] - ric[:, [trunk], :]) / decoded.L
            parts.append(relative.reshape(decoded.frames, -1))
            tips.append(ric[:, chain[-1], :])
        if len(tips) >= 2:
            # Hand-to-hand distance: the one number a two-handed grip fixes.
            span = np.linalg.norm(tips[0] - tips[1], axis=-1) / decoded.L
            parts.append(span[:, None])
        frames = np.concatenate(parts, axis=1)
        return np.median(frames, axis=0)


def hand_class(label: str) -> str | None:
    words = label_words(label)
    for word in HAND_WORDS:
        if word in words:
            return word
    return None


def knn_vote(descriptor, references, k=KNN_K):
    """``(class or None, nearest distance, votes, neighbours)`` from the k nearest references.

    *references* are ``(vector, class, clip)`` triples; ``neighbours`` names
    the k clips that voted, nearest first, so a proposal can say what it
    matched.
    """
    if not references:
        return None, float("inf"), [], []
    vectors = np.stack([r[0] for r in references])
    distances = np.linalg.norm(vectors - descriptor[None, :], axis=1)
    order = np.argsort(distances)[:k]
    votes = [references[i][1] for i in order]
    neighbours = [references[i][2]["clip"] for i in order]
    nearest = float(distances[order[0]])
    winner = votes[0] if len(set(votes)) == 1 else None
    return winner, nearest, votes, neighbours


# ── the run ──────────────────────────────────────────────────────────────────

def run(args) -> int:
    cond, sources, clips = load_corpus(args.cond_path, None)
    print(f"[OK] {len(clips)} clip(s) from {args.cond_path} (pending_delete rows dropped)")

    defaults = dict(DEFAULT_SPECIES)
    for item in args.species_default:
        name, _, word = item.partition("=")
        if word not in HAND_WORDS:
            raise SystemExit(f"--species-default {item!r}: the value must be one of {HAND_WORDS}")
        defaults[name.strip()] = word

    rows_by_species = defaultdict(list)
    for clip in clips:
        rows_by_species[clip["species"]].append(clip)

    layouts: dict[str, ArmLayout] = {}
    decoded: dict[tuple[str, str], np.ndarray] = {}

    def descriptor(clip):
        key = (clip["species"], clip["key"])
        if key not in decoded:
            layout = layouts.setdefault(clip["species"], ArmLayout(cond[clip["species"]]))
            decoded[key] = layout.describe(DecodedClip(clip["motion_path"], cond[clip["species"]])) if layout.ok else None
        return decoded[key]

    # References are the species' OWN labelled rows. Rig-mates (the KI / TNR /
    # TTR models share one skeleton each) were tried as a shared pool and
    # matched relaxed-arm poses across species regardless of what the hands
    # held -- a villager's idle landed on a soldier's death with its rifle --
    # so a row is only ever compared with its own species' holding poses.
    references_by_species: dict[str, list] = {}
    for species, rows in rows_by_species.items():
        refs = []
        for clip in rows:
            if hand_class(clip["label"]) is None:
                continue
            vector = descriptor(clip)
            if vector is not None:
                refs.append((vector, hand_class(clip["label"]), clip))
        if refs:
            references_by_species[species] = refs

    # ── calibration: leave-one-out over the references ──
    # Distance threshold: the nearest-neighbour distance a reference sees
    # among its own class-mates, at the median -- a match further than the
    # typical same-class neighbour is a guess, not a near-duplicate pose.
    nearest_same: list[float] = []
    for refs in references_by_species.values():
        for i, (vector, klass, clip) in enumerate(refs):
            same = [np.linalg.norm(v - vector) for j, (v, c, _) in enumerate(refs) if j != i and c == klass]
            if same:
                nearest_same.append(float(min(same)))
    distance_max = float(np.median(nearest_same)) if nearest_same else float("inf")
    if args.distance_max is not None:
        distance_max = args.distance_max
    confusion = Counter()
    undecided = 0
    for refs in references_by_species.values():
        for i, (vector, klass, clip) in enumerate(refs):
            others = [ref for j, ref in enumerate(refs) if j != i]
            if len(others) < KNN_K:
                undecided += 1
                continue
            winner, nearest, votes, _neighbours = knn_vote(vector, others)
            if winner is None or nearest > distance_max:
                undecided += 1
                continue
            confusion[(klass, winner)] += 1
    classes = list(HAND_WORDS)
    precision = {}
    for predicted in classes:
        hits = confusion[(predicted, predicted)]
        total = sum(confusion[(truth, predicted)] for truth in classes)
        precision[predicted] = hits / total if total else 0.0
    open_classes = {c for c in HAND_WORDS if precision[c] >= args.gate and confusion[(c, c)] > 0}

    print()
    print("=" * 72)
    print(f"  CALIBRATION leave-one-out over {sum(len(r) for r in references_by_species.values())} reference row(s) "
          f"in {len(references_by_species)} species, k={KNN_K}, distance <= {distance_max:.3f}")
    print("=" * 72)
    print("  truth / predicted   " + "".join(f"{c:>8s}" for c in classes))
    for truth in classes:
        print(f"  {truth:18s}  " + "".join(f"{confusion[(truth, p)]:8d}" for p in classes))
    print(f"  undecided (split vote, too far, or under {KNN_K} references): {undecided}")
    for c in classes:
        gate = " -> OPEN" if c in open_classes else f" -> CLOSED (gate {args.gate:.0%}, list only)"
        print(f"  precision {c:6s}: {precision[c]:.1%}{gate}")

    # ── proposals ──
    # k-NN only judges a species that uses the hands axis itself (some rows
    # carry a hand word, so an empty one may be a gap). A species with no
    # hand word anywhere is either unarmed (empty is right) or covered by the
    # default table.
    proposals: list[Proposal] = []
    for species, rows in sorted(rows_by_species.items()):
        name = species_name({"species": species})
        refs = references_by_species.get(species, [])
        default = defaults.get(name)
        if default is None and len(refs) < KNN_K:
            continue
        for clip in rows:
            if hand_class(clip["label"]) is not None:
                continue
            if default is not None:
                proposals.append(Proposal(
                    clip, "write", f"species default {default} (model carries its weapon)",
                    spell_with(clip["label"], [default]),
                    {"measure": "species_default", "word": default}))
                continue
            vector = descriptor(clip)
            if vector is None:
                proposals.append(Proposal(clip, "review", "no arm chains found on this rig"))
                continue
            winner, nearest, votes, neighbours = knn_vote(vector, refs)
            metrics = {"nearest": round(nearest, 3), "votes": "/".join(votes),
                       "neighbours": " | ".join(neighbours)}
            if winner is None:
                proposals.append(Proposal(clip, "review", f"split vote {votes}", metrics=metrics))
                continue
            if nearest > distance_max:
                proposals.append(Proposal(clip, "review", f"nearest reference {nearest:.3f} beyond {distance_max:.3f} ({winner})", metrics=metrics))
                continue
            status = "write" if winner in open_classes else "review"
            proposals.append(Proposal(
                clip, status, f"{winner}: {KNN_K} nearest references agree, nearest {nearest:.3f}",
                spell_with(clip["label"], [winner]),
                {"measure": "arm_pose_knn", **metrics}, metrics))

    print()
    print("=" * 72)
    print("  PROPOSALS")
    print("=" * 72)
    counts = Counter((p.status, p.evidence.get("measure", "") or p.reason.split(":")[0]) for p in proposals)
    for key in sorted(counts):
        print(f"  {key[0]:7s} {key[1]:40s} {counts[key]}")
    by_species = Counter((species_name(p.clip), p.status) for p in proposals)
    for key in sorted(by_species):
        print(f"    {key[0]:22s} {key[1]:7s} {by_species[key]}")
    if args.verbose:
        for p in sorted(proposals, key=lambda p: (p.status, p.clip["species"], p.clip["clip"])):
            print(f"    {p.status:6s} {p.clip['clip']:48s} {p.label!r} -> {p.proposed!r}  {p.reason}")

    if args.report:
        count = write_csv(args.report, proposals)
        print(f"\n[OK] wrote {count} row(s) to {args.report}")
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps([p.as_row() for p in proposals], ensure_ascii=False, indent=1), encoding="utf-8")

    writes = [p for p in proposals if p.status == "write"]
    if not args.apply:
        print(f"\n[OK] dry run: {len(writes)} row(s) would be written; pass --apply to write them")
        return 0
    if not writes:
        print("\n[OK] nothing to write")
        return 0
    check_head_order(sources, writes)
    written = apply_proposals(sources, writes, slot="hands", slot_vocab=set(HANDS_VOCAB))
    for root, count in written.items():
        print(f"[OK] {root}: {count} row(s) written (reviewed:false + autofill)")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Propose missing hand1/hand2 words from the holding pose (dry run by default).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--cond-path", "--cond_path", dest="cond_path", default=None)
    parser.add_argument("--apply", action="store_true", help="Write the 'write' rows (default: dry run).")
    parser.add_argument("--dry-run", dest="apply", action="store_false")
    parser.add_argument("--species-default", action="append", default=[], metavar="NAME=hand1|hand2",
                        help="Hand word every empty-slot row of NAME receives (repeatable; adds to the built-in table).")
    parser.add_argument("--gate", type=float, default=0.90, help="Per-class leave-one-out precision needed to write (default 0.90).")
    parser.add_argument("--distance-max", type=float, default=None,
                        help="k-NN distance threshold (default: 90th percentile of same-class nearest distances).")
    parser.add_argument("--report", default="", help="CSV of every judged row.")
    parser.add_argument("--json", dest="json_path", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.cond_path is None:
        from data_loaders.truebones.truebones_utils.get_opt import DEFAULT_COND_PATH
        args.cond_path = DEFAULT_COND_PATH
    return args


if __name__ == "__main__":
    sys.exit(run(parse_args()))
