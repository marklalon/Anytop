"""
migrate_hand_state_labels.py

Move ``action_labels.jsonl`` from the ``weapon`` + ``1hand`` / ``2hand`` spelling
to the exclusive hands axis ``hand0`` / ``hand1`` / ``hand2`` (HANDS_VOCAB in
motion_labels.py), and fill the axis in for EVERY clip of the species that hold
something in at least one clip.

Why the fill-in is the real job: the old ``weapon`` tag was written on combat
idles and combat locomotion only.  The attack, hurt, die and block clips of the
same armed character carried no tag, so "the untagged clips" of an armed species
were a mix of armed and unarmed poses and a prompt without ``weapon`` still drew
armed ones.  Under the hands axis a clip without a hand-state word means
"unspecified", and ``hand0`` is a positive statement -- both hands empty -- which
is what makes an unarmed idle requestable at all.

The axis counts OCCUPIED hands, not weapon class: sword + shield and a rifle
are both ``hand2``; a dagger and a torch are both ``hand1``; a bow carried at
the side is ``hand1``, drawn it is ``hand2``.

Sources of truth, in order (per clip):

1. ``CLIP_OVERRIDES``       -- an explicit per-clip verdict.
2. ``SPECIES_POLICY``       -- ordered ``(regex on the clip stem, hand)`` rules,
                               ending in the species default.  ``None`` as the
                               hand means the state CHANGES inside the clip
                               (draw / sheathe / lift / put down): the axis is
                               left unspecified.
3. the retired tag          -- ``weapon, 1hand`` -> hand1, ``weapon, 2hand`` ->
                               hand2, which the policy must agree with; a
                               disagreement is printed and the policy loses,
                               since the old tag was a person's verdict.

Every verdict carries a provenance and the report groups the rows by it:

    tag    the retired label already carried the count (``1hand`` / ``2hand``);
           derived, never written in the policy table
    name   the clip stem spells the prop state (Weapon / 2HLong / Shield /
           Rifle / Torch / Axe ...)
    impl   forced by the implement word: a bow is drawn with both hands
    pack   a species-wide convention of the source pack (RTS units never put
           the weapon down, the soldier pack is a rifle pack ...) that nobody
           has confirmed clip by clip -- THESE are the rows to look at

Species outside ``SPECIES_POLICY`` are left without a hand state (the axis is
"unspecified" for them, which is correct for anything that never holds a thing
and for anything without hands).  Species in the same packs that plainly carry
weapons but never received the old tag (TTR_LightInfantry, KI_Slinger, ...) are
listed at the end as candidates: add a policy row and re-run.

Idempotent: a row already carrying a hands word is re-derived the same way, so
the script can be re-run after editing the policy.  The file is rewritten in
place with a ``.bak`` next to it (kept if it already exists).

Usage::

    python tools/dataset_cleanup/migrate_hand_state_labels.py --dry-run
    python tools/dataset_cleanup/migrate_hand_state_labels.py
    python tools/dataset_cleanup/migrate_hand_state_labels.py --dataset-dir dataset/unitybundles/processed
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
from pathlib import Path

_ANYTOP_DIR = Path(__file__).resolve().parent.parent.parent
if str(_ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    HANDS_VOCAB,
    canonical_action_label,
    parse_action_label,
)

RETIRED = {"weapon": None, "1hand": "hand1", "2hand": "hand2"}

# ---------------------------------------------------------------------------
# Per-clip verdicts (win over everything).  Keyed by the clip stem, no ".npy".
# ---------------------------------------------------------------------------
CLIP_OVERRIDES: dict[str, tuple[str | None, str]] = {
    # (hand, provenance)
}

# ---------------------------------------------------------------------------
# Per-species rules: ordered (regex on the stem AFTER the species prefix, hand,
# provenance).  First match wins; the last row is the species default.
# ---------------------------------------------------------------------------
_H0, _H1, _H2, _CHANGES = "hand0", "hand1", "hand2", None

SPECIES_POLICY: dict[str, list[tuple[str, str | None, str]]] = {
    # Caveman pack: every clip exists twice, with and without the club.
    "IAC_Caveman": [(r"Weap(o)?n", _H1, "name"), (r".*", _H0, "name")],
    "IAC_Cavewoman": [(r"Weap(o)?n", _H1, "name"), (r".*", _H0, "name")],

    # Kevin Iglesias packs -----------------------------------------------
    "KI_Archer": [
        (r"^BowShot|^GetArrow", _H2, "name"),        # drawing / aiming the bow
        (r"^CombatIdle", _H1, "pack"),               # bow in hand, not drawn
        (r"^Sheathe|^Unsheathe", _CHANGES, "name"),
        (r".*", _H0, "pack"),                        # the plain set: bow stowed
    ],
    "KI_CasterMage": [
        (r"^CombatIdle", _H1, "pack"),               # staff
        (r".*", _H0, "pack"),
    ],
    "KI_Soldier": [                                  # a rifle pack
        (r"DrawWeapon", _CHANGES, "name"),
        (r"^Salute", _H1, "pack"),
        (r".*", _H2, "pack"),
    ],
    "KI_Villager": [
        (r"^Axe|^Pickaxe", _H2, "name"),
        (r"^Hammer|^Skinning", _H1, "name"),
        (r"^FarmWorking", _H2, "name"),              # hoe, both hands
        (r"^Farm", _H1, "pack"),                     # hoe carried
        (r"^SackPlantingSeeds02", _CHANGES, "name"), # putdown
        (r"^SackPlantingSeeds", _H1, "name"),
        (r"^Sack", _H2, "pack"),
        (r"Lift|Drop", _CHANGES, "name"),
        (r"^Carrying", _H2, "name"),
        (r"^Cooking03", _CHANGES, "name"),           # get / drop an item
        (r"^Cooking", _H1, "pack"),
        (r"^Fishing", _H2, "pack"),
        (r"^Sawing", _H2, "pack"),
        (r"^Shovel", _H2, "name"),
        (r"^WateringCan", _H1, "name"),
        (r".*", _H0, "pack"),
    ],
    "KI_Warrior": [
        (r"2HLong", _H2, "name"),
        (r"Shield", _H2, "name"),                    # sword + shield
        (r"^CombatIdle|^Sprint", _H2, "pack"),
        (r".*", _H1, "pack"),                        # the plain set: one-handed sword
    ],

    # Low-poly hero: the prefix names the loadout ---------------------------
    "LH_Hero": [
        (r"^SpearRelax$", _H1, "pack"),
        (r"^Spear", _H2, "name"),
        (r"^THSword(Relax|Run|Walk|Dash|StrafeLeft|StrafeRight)$", _H1, "pack"),
        (r"^THSword", _H2, "name"),
        (r"^Torch", _H1, "name"),
        (r"^(Fly)?(Crossbow|Longbow)", _H2, "name"),
        (r"^PickUp$", _CHANGES, "name"),
        (r"^(ChopTree|Digging|Strumming)$", _H2, "pack"),
        (r"^(Melee|SpinAttack|SawingWood|HammeringOnAnvil)", _H1, "pack"),
        (r".*", _H0, "pack"),
    ],

    # Mini-legion style RTS units: the weapon never leaves the hand ---------
    "MLH_Archer": [(r"^Attack", _H2, "impl"), (r".*", _H1, "pack")],
    "MLH_Knight": [(r".*", _H2, "pack")],           # sword + shield
    "MLH_Worker": [(r"^Pickup", _CHANGES, "name"), (r"^Work", _H2, "pack"), (r".*", _H1, "pack")],
    "MLS_ElfRanger": [(r"^Attack", _H2, "impl"), (r".*", _H1, "pack")],
    "MLS_Knome": [(r"^WorkPickup", _CHANGES, "name"), (r"^Work", _H2, "pack"), (r".*", _H1, "pack")],

    "RMW_Orc": [(r".*", _H2, "pack")],               # two-handed axe
    "RMW_Skeleton": [(r".*", _H2, "pack")],          # sword + shield (has Defense)

    # Toon RTS packs ----------------------------------------------------------
    "TNR_Archer": [(r"^Attack", _H2, "impl"), (r"^Combat", _H2, "pack"), (r".*", _H1, "pack")],
    "TNR_Cavalry": [(r".*", _H1, "pack")],
    "TNR_CavalryArcher": [(r"^Attack", _H2, "impl"), (r".*", _H1, "pack")],
    "TNR_CavalrySpear": [(r"^Attack|^Charge|^Combat", _H2, "pack"), (r".*", _H1, "pack")],
    "TNR_Infantry": [(r".*", _H1, "pack")],
    "TNR_Mage": [(r".*", _H1, "pack")],
    "TNR_Spearman": [(r"^Attack|^Charge|^Combat", _H2, "pack"), (r".*", _H1, "pack")],
    "TNR_Worker": [
        (r"^Bag|^Wood", _H2, "name"),
        (r"^Attack|^WorkingA|^WorkingB", _H1, "pack"),
        (r".*", _H0, "pack"),
    ],
    "TTR_Archer": [(r"^Attack", _H2, "impl"), (r".*", _H1, "pack")],
    "TTR_Crossbowman": [(r"^Attack", _H2, "impl"), (r".*", _H1, "pack")],
    "TTR_Halberdier": [(r".*", _H2, "pack")],
    "TTR_HeavyCavalry": [(r".*", _H1, "pack")],
    "TTR_HeavySwordman": [(r".*", _H2, "pack")],
    "TTR_Mage": [(r".*", _H1, "pack")],
    "TTR_MountedKnight": [(r".*", _H1, "pack")],
    "TTR_MountedMage": [(r".*", _H1, "pack")],
    "TTR_MountedScout": [(r"^AttackArcher", _H2, "impl"), (r".*", _H1, "pack")],
    "TTR_Spearman": [(r"^Attack", _H2, "pack"), (r".*", _H1, "pack")],
    "TTR_Swordman": [(r".*", _H1, "pack")],
}

# Species that were NOT in scope (never carried the old tag) but whose clip
# names or labels say they hold something.  Reported, never edited.
CANDIDATE_HINT = re.compile(r"Spear|Sword|Shield|Bow|Rifle|Gun|Axe|Hammer|Torch|Staff|Weapon")


def _species_of(clip: str) -> str:
    return "_".join(clip.split("_")[:2])


def _stem_of(clip: str, species: str) -> str:
    stem = clip[:-4] if clip.endswith(".npy") else clip
    return stem[len(species) + 1:]


def decide(clip: str, retired_hand: str | None) -> tuple[str | None, str, bool]:
    """``(hand, provenance, in_scope)`` for one clip."""
    species = _species_of(clip)
    stem = _stem_of(clip, species)
    if stem in CLIP_OVERRIDES:
        hand, why = CLIP_OVERRIDES[stem]
        return hand, f"override:{why}", True
    rules = SPECIES_POLICY.get(species)
    if rules is None:
        return retired_hand, "tag" if retired_hand else "out-of-scope", False
    for pattern, hand, why in rules:
        if re.search(pattern, stem):
            return hand, why, True
    raise AssertionError(f"{species} policy has no catch-all row")


def migrate_row(row: dict, report: dict) -> dict:
    clip = str(row["clip"])
    label = str(row.get("action_label") or "")
    tokens = [t.strip() for t in label.split(",") if t.strip()] if label else []
    retired = [t for t in tokens if t in RETIRED]
    retired_hand = next((RETIRED[t] for t in retired if RETIRED[t]), None)
    kept = [t for t in tokens if t not in RETIRED and t not in HANDS_VOCAB]

    hand, why, in_scope = decide(clip, retired_hand)
    if retired_hand:
        if hand != retired_hand:
            # The old tag was a human verdict; the policy is an inference.
            report["disagreements"].append((clip, label, hand, why, retired_hand))
        hand, why = retired_hand, "tag"
    if not kept and hand:
        # A label needs a head word; a bare hand state is not a label.  (No such
        # row exists today; the guard is so a future one fails visibly.)
        raise SystemExit(f"{clip}: label {label!r} would keep no head word")

    new_tokens = kept + ([hand] if hand else [])
    new_label = canonical_action_label(new_tokens) if new_tokens else ""
    if new_label:
        parse_action_label(new_label)
    if new_label != label:
        report["changed"] += 1
    species = _species_of(clip)
    report["by_species"][species][hand or "(unspecified)"] += 1
    report["by_provenance"][why] += 1
    if in_scope:
        report["rows"].append((why, species, clip, label, new_label))
    elif CANDIDATE_HINT.search(clip):
        report["candidates"][species] += 1
    out = dict(row)
    out["action_label"] = new_label
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default=str(_ANYTOP_DIR / "dataset" / "unitybundles" / "processed"))
    parser.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    parser.add_argument("--show", default="pack", help="comma list of provenances to list row by row (default: pack)")
    args = parser.parse_args()

    path = Path(args.dataset_dir) / "action_labels.jsonl"
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    report: dict = {
        "changed": 0,
        "by_species": collections.defaultdict(collections.Counter),
        "by_provenance": collections.Counter(),
        "disagreements": [],
        "rows": [],
        "candidates": collections.Counter(),
    }
    out_lines = []
    for line in raw_lines:
        if not line.strip():
            out_lines.append(line)
            continue
        row = json.loads(line)
        out_lines.append(json.dumps(migrate_row(row, report), ensure_ascii=False))

    # ---- report ----
    print(f"{path}: {report['changed']} row(s) change")
    print("\nverdicts by provenance:")
    for why, count in report["by_provenance"].most_common():
        print(f"  {count:5d}  {why}")
    print("\nin-scope species (hand state counts):")
    for species in SPECIES_POLICY:
        counts = report["by_species"].get(species)
        if counts:
            print(f"  {species:20s} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if report["disagreements"]:
        print("\nPOLICY vs RETIRED TAG disagreements (tag kept):")
        for clip, label, hand, why, tag_hand in report["disagreements"]:
            print(f"  {clip:45s} {label!r}: policy {hand} ({why}) vs tag {tag_hand}")
    show = {s.strip() for s in args.show.split(",") if s.strip()}
    if show:
        print(f"\nrows by provenance {sorted(show)} -- the ones nobody confirmed clip by clip:")
        for why, species, clip, label, new_label in sorted(report["rows"]):
            if why in show:
                print(f"  [{why}] {clip:45s} {label!r} -> {new_label!r}")
    if report["candidates"]:
        print("\nout-of-scope species whose clips still look armed (add a SPECIES_POLICY row to include):")
        for species, count in sorted(report["candidates"].items()):
            print(f"  {species:28s} {count} clip(s)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"\nbacked up to {backup}")
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
