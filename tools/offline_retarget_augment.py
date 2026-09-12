"""Offline retarget augmentation -- spread rare action labels across species.

A handful of action words live on one or two species only (``dance`` on
KI_Performer, ``salute`` on KI_Soldier, ``push``/``pull`` on LH_Hero), so the
model sees them in a single body geometry.  This tool copies those motions onto
other skeletons offline, so the (species, action) grid gets filled with real
clips instead of relying on zero-shot composition.

Pipeline per (source clip, target species)::

    <donor's own raw .glb/.fbx>                    the file preprocessing read
      -> retarget_glb_to_glb(..., align_facing, ground, fullbody_ik)
      -> <out-root>/<TargetSpecies>/<Action>_<SrcSpecies>Retarget.glb

The world-space transfer places every target joint on its donor counterpart, and
whatever a rotation cannot reach from the target's rest offset is left in the
pose-translation channel -- which on a cross-species pair is the target rig
stretched to the donor's proportions (~5% of body height on a biped pair).
``fullbody_ik`` re-solves that pose on the rigid target skeleton, so the clip
preprocessing reads is a body of the target's own size.  ``--no-fullbody-ik``
turns it off for diagnosing the transfer itself.

The donor is the dataset's own source animation -- ``source_fbx_path`` from
motion_metadata.json, sliced to the same ``source_frame_range`` -- not the
preprocessed ``.npy``.  Staying in native space the whole way is what lets a
travelling donor keep its locomotion: the feature round trip strips the root XZ
of any clip whose sustained travel ``get_motion`` removes, so the restored
GLB came out in place and contradicted its own "forward" label.  Here nothing is
encoded, so root translation survives verbatim and every action group is usable.

The GLB lands under the Anytop ``outputs`` directory (``--out-root``,
``outputs/offline_retarget_augment/retarget``) as a staged, reviewable copy --
never into the dataset.  The trailing ``retarget`` marker is handled explicitly
by ``fbx_filename_rules`` (never a rest-pose reference, exempt from the
variant-codename skip) and survives into the clip name as provenance.

Two modes:

    --mode plan   select sources + targets, write a manifest JSONL, print a
                  summary.  Cheap; nothing is written into the dataset.
    --mode run    execute a manifest: retarget onto the target rig, re-solve the
                  pose on that rig's own rigid skeleton (full-body IK), write the
                  GLBs, and append the matching ``action_labels.jsonl`` rows.

Typical use::

    python tools/offline_retarget_augment.py --mode plan \
        --manifest outputs/offline_retarget_augment/plan.jsonl

    python tools/offline_retarget_augment.py --mode run \
        --manifest outputs/offline_retarget_augment/plan.jsonl \
        --limit 6 --dry-run-labels          # smoke test

    python tools/offline_retarget_augment.py --mode run \
        --manifest outputs/offline_retarget_augment/plan.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANYTOP_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.dirname(ANYTOP_DIR)
for _p in (REPO_ROOT, ANYTOP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_loaders.truebones.truebones_utils.dataset_sources import (  # noqa: E402
    bare_species_name,
    canonical_key,
    load_datasets_manifest,
)
from data_loaders.truebones.truebones_utils.fbx_filename_rules import (  # noqa: E402
    RETARGET_MARKER,
    normalize_action_name,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    canonical_action_label,
)
from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    ACTION_LABELS_FILE,
    MOTION_DIR,
    MOTION_METADATA_FILE,
)
from utils.fullbody_ik import DEFAULT_IK_STRETCH_FACTOR  # noqa: E402
from utils.retarget_pipeline import (  # noqa: E402
    _exporter_view_of_skeleton,
    _load_native_rig_skeleton,
    canonical_match_names_from_raw_skeleton,
)

DEFAULT_COND = os.path.join(ANYTOP_DIR, "dataset", "merged", "cond.npy")
DEFAULT_MANIFEST = os.path.join(
    ANYTOP_DIR, "outputs", "offline_retarget_augment", "plan.jsonl"
)
DEFAULT_WORK_DIR = os.path.join(ANYTOP_DIR, "outputs", "offline_retarget_augment")
DEFAULT_OUT_ROOT = os.path.join(
    ANYTOP_DIR, "outputs", "offline_retarget_augment", "retarget"
)
DATASETS_MANIFEST = os.path.join(ANYTOP_DIR, "dataset", "datasets.jsonl")

# Preprocessing warns when a source file is not 30 FPS, and all 260 species are
# authored at 30, so the export inherits the donor's own rate by default rather
# than forcing one: forcing a rate does not resample, it retimes.  --fps
# overrides when a donor really is authored at something else.
DEFAULT_EXPORT_FPS = None


# ---------------------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------------------
# Per action word: which body plans may receive it, and how far to spread it.
#
# ``humanoid`` is not an object_subset -- it is the biped species that actually
# carry an arm chain (see ``is_humanoid``).  An arm-led motion (salute, lift,
# push) is meaningless on a Chicken or an Egglet, both of which are bipeds.
#
# ``targets`` caps how many target species an action reaches, ``clips_per_target``
# how many distinct source clips each of them receives.  Their product bounds the
# clips a single word adds, so a 34-clip donor like KI_Performer's dance library
# spreads across species instead of being copied wholesale onto each one.
@dataclass(frozen=True)
class ActionPolicy:
    word: str
    subsets: tuple[str, ...]      # allowed object_subsets, or ("humanoid",)
    targets: int
    clips_per_target: int


HUMANOID_ACTIONS = ("dance", "crawl", "salute", "push", "pull", "pickup", "putdown", "lift")
BROAD_ACTIONS = ("swim", "jump")

DEFAULT_POLICIES: tuple[ActionPolicy, ...] = (
    # Arm-led / upright motions: humanoid rigs only.
    ActionPolicy("dance", ("humanoid",), targets=16, clips_per_target=3),
    ActionPolicy("crawl", ("humanoid",), targets=14, clips_per_target=3),
    ActionPolicy("salute", ("humanoid",), targets=14, clips_per_target=1),
    ActionPolicy("push", ("humanoid",), targets=14, clips_per_target=1),
    ActionPolicy("pull", ("humanoid",), targets=14, clips_per_target=1),
    ActionPolicy("pickup", ("humanoid",), targets=14, clips_per_target=2),
    ActionPolicy("putdown", ("humanoid",), targets=14, clips_per_target=2),
    ActionPolicy("lift", ("humanoid",), targets=14, clips_per_target=2),
    # Whole-body modes that read on any legged/winged plan.
    ActionPolicy("swim", ("biped", "quadruped", "winged"), targets=30, clips_per_target=2),
    ActionPolicy("jump", ("biped", "quadruped", "winged"), targets=30, clips_per_target=2),
)

# Joint-name tokens (after the canonical synonym normalisation) that mark an arm
# and a leg chain.  Checked per side, so a one-armed rig does not pass.
_ARM_TOKENS = ("hand", "forearm", "wrist", "elbow")
_LEG_TOKENS = ("thigh", "calf", "shin", "knee", "leg")
_FOOT_TOKENS = ("foot", "ankle", "toe")

# A tail chain separates the theropod/bird bipeds (Tyranno, Raptor2, Flamingo)
# from the humanoids.  They pass every arm/leg test -- a raptor's clavicle,
# elbow and wrist are all there -- but a salute or a dance retargeted onto a
# body that balances on a counterweighted tail is not the motion the label names.
_TAIL_TOKEN = "tail"
_MIN_TAIL_JOINTS = 2

# Mounted rigs (TNR_Cavalry, TTR_MountedKnight) carry a rider and a horse in one
# skeleton, spelled as a duplicated ``... 001`` namespace.  A donor covers only
# one of the two bodies, so the other would stay frozen in its rest pose for the
# whole clip.  Excluded from every action unless --allow-composite.
_COMPOSITE_CORE_JOINTS = ("pelvis", "hips", "head", "spine")


# ---------------------------------------------------------------------------
# Dataset index
# ---------------------------------------------------------------------------
@dataclass
class SpeciesInfo:
    key: str                      # canonical "<namespace>/<Species>"
    bare: str
    namespace: str
    dataset_root: str
    subset: str
    raw_dir: str | None = None    # where this species' source animations live
    tpose_path: str | None = None
    clip_count: int = 0
    words: set = field(default_factory=set)


@dataclass
class ClipInfo:
    clip: str                     # "<Species>_<Action>.npy"
    species: str                  # canonical key
    npy_path: str                 # preprocessed features -- planning stats only
    source_path: str              # the raw animation preprocessing read
    source_frame_range: list | None   # [start, end] slice into that raw file
    action_group: str
    action_label: str
    words: tuple


# Threshold for "this clip actually goes somewhere", as a fraction of body span
# so it is scale-free.  Used to report which donors carry locomotion and to
# assert the export kept it: the native retarget transfers root translation
# verbatim, so a travelling donor producing an in-place export is a real bug,
# not the expected feature-space loss it used to be.
_TRAVEL_EPSILON = 1e-4


def _resolve(path_value: str) -> str:
    path_value = str(path_value)
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(ANYTOP_DIR, path_value))


def _subset_of(cond_entry) -> str:
    tags = list(cond_entry.get("species_tags", []) or [])
    return str(tags[0]).lower() if tags else "?"


def _name_tokens(names) -> list[set]:
    return [set(str(n).lower().replace("_", " ").split()) for n in names]


def is_humanoid(cond_entry, subset: str) -> bool:
    """A biped that carries a left AND right arm chain plus legs and a head.

    The gate is deliberately structural rather than name-based: the corpus spells
    the same joint as ``Hand Left`` and ``Left Hand`` depending on the rig, so a
    token test is the only spelling-independent form.
    """
    if subset != "biped":
        return False
    from utils.skeleton_similarity import normalize_match_name

    tokens = _name_tokens(
        normalize_match_name(n) for n in cond_entry.get("canonical_joint_names", [])
    )

    def has(side: str, wanted: tuple) -> bool:
        return any(side in t and any(w in t for w in wanted) for t in tokens)

    has_head = any("head" in t for t in tokens)
    arms = has("left", _ARM_TOKENS) and has("right", _ARM_TOKENS)
    legs = has("left", _LEG_TOKENS) and has("right", _LEG_TOKENS)
    feet = has("left", _FOOT_TOKENS) and has("right", _FOOT_TOKENS)
    tail_joints = sum(1 for t in tokens if _TAIL_TOKEN in t)
    return bool(has_head and arms and legs and feet and tail_joints < _MIN_TAIL_JOINTS)


def is_composite_rig(cond_entry) -> bool:
    """True when one skeleton holds two bodies (a mount and its rider)."""
    names = {str(n).lower().strip() for n in cond_entry.get("canonical_joint_names", [])}
    return any(
        core in names and f"{core} 001" in names
        for core in _COMPOSITE_CORE_JOINTS
    )


def build_index(cond_path: str):
    """Index every dataset: species metadata, per-clip labels, raw source dirs."""
    cond = np.load(cond_path, allow_pickle=True).item()
    sources = load_datasets_manifest(DATASETS_MANIFEST)

    species: dict[str, SpeciesInfo] = {}
    for key, entry in cond.items():
        species[key] = SpeciesInfo(
            key=key,
            bare=bare_species_name(key),
            namespace=str(entry.get("dataset_namespace") or key.rsplit("/", 1)[0]),
            dataset_root=_resolve(str(entry.get("dataset_root") or "")),
            subset=_subset_of(entry),
        )

    clips: dict[str, ClipInfo] = {}
    for source in sources:
        root = _resolve(source.root)
        meta_path = os.path.join(root, MOTION_METADATA_FILE)
        labels_path = os.path.join(root, ACTION_LABELS_FILE)
        if not os.path.isfile(meta_path) or not os.path.isfile(labels_path):
            print(f"[plan] skipping {source.namespace}: no metadata/labels at {root}")
            continue

        with open(meta_path, encoding="utf-8") as handle:
            motions = json.load(handle)["motions"]

        # The raw source directory is wherever this species' own animations came
        # from; reading it off the metadata keeps the tool correct for all three
        # datasets without hardcoding any drive layout.
        for name, meta in motions.items():
            key = canonical_key(source.namespace, meta["object_type"])
            info = species.get(key)
            if info is None:
                continue
            info.clip_count += 1
            if info.raw_dir is None:
                src = meta.get("source_fbx_path")
                if src:
                    info.raw_dir = os.path.dirname(str(src))

        tpose_path = os.path.join(root, "tpose_reference_paths.jsonl")
        if os.path.isfile(tpose_path):
            with open(tpose_path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    key = canonical_key(source.namespace, row["object_type"])
                    if key in species:
                        species[key].tpose_path = _resolve(row["path"])

        with open(labels_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                clip = row["clip"]
                # sidecar keys are extension-less; metadata is keyed by file name
                meta = motions.get(clip if clip.endswith(".npy") else clip + ".npy")
                if meta is None:
                    continue
                key = canonical_key(source.namespace, meta["object_type"])
                label = str(row.get("action_label") or "")
                words = tuple(w.strip() for w in label.split(",") if w.strip())
                # The raw animation this clip was preprocessed from, plus the
                # slice taken out of it: retargeting reads that file directly,
                # so the donor motion never passes through feature space.
                source_path = str(meta.get("source_fbx_path") or "")
                frame_range = meta.get("source_frame_range")
                if isinstance(frame_range, (list, tuple)) and len(frame_range) == 2:
                    frame_range = [int(frame_range[0]), int(frame_range[1])]
                else:
                    frame_range = None
                clips[f"{source.namespace}::{clip}"] = ClipInfo(
                    clip=clip,
                    species=key,
                    npy_path=os.path.join(
                        root, MOTION_DIR,
                        clip if clip.endswith(".npy") else clip + ".npy"),
                    source_path=source_path,
                    source_frame_range=frame_range,
                    action_group=str(row.get("action_group") or ""),
                    action_label=label,
                    words=words,
                )
                if key in species:
                    species[key].words.update(words)

    return cond, species, clips


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def _donor_travels(clip_info, cond, cache: dict) -> bool:
    """Measure whether a donor clip's own features still carry XZ locomotion.

    Preprocessing's own verdict says only whether it removed sustained travel;
    a treadmill animation authored in place (Dog_Paddle) was never touched and
    still goes nowhere.  The travel is read back the way the decoder does it --
    cumulative XZ velocity of the translation-root joint -- and compared against
    the clip's own body span so the test is scale-free.
    """
    if clip_info.clip in cache:
        return cache[clip_info.clip]
    # Measured unconditionally: preprocessing's verdict is not a reliable
    # stand-in. It flattens the joint named by cond's ``translation_root_index``,
    # and where that index disagrees with the joint the features actually carry
    # the locomotion on, a clip is flagged flattened while its travel is still
    # there (KI_Soldier_Crawling01Forward: flagged, 0.94 of XZ path on joint 0).
    features = np.load(clip_info.npy_path)
    # Every joint is measured rather than the cond's translation_root_index:
    # the index recorded in cond does not always match the joint the features
    # actually carry the locomotion on, and the largest XZ path over all
    # joints is the locomotion either way.
    steps = np.zeros((features.shape[0], features.shape[1], 2), dtype=np.float64)
    steps[1:] = features[:-1][:, :, [9, 11]]
    path = np.cumsum(steps, axis=0)
    net = float(np.linalg.norm(path[-1] - path[0], axis=-1).max())
    span = float(np.ptp(features[:, :, 1])) or 1.0
    result = net > max(_TRAVEL_EPSILON, 0.05 * span)
    cache[clip_info.clip] = result
    return result


def _rank_targets(cond, source_key: str, candidates: list[str]):
    """Candidate species scored by skeleton similarity to the donor (closest first)."""
    from utils.skeleton_similarity import rank_species

    pool = {k: cond[k] for k in candidates}
    if not pool:
        return []
    return rank_species(cond[source_key], pool, query_hint=source_key, top_k=None)


def _target_glb_name(source_clip: str, source_species: str, target_species: str) -> str:
    """``<SourceAction>_<SrcSpecies>Retarget.glb`` in the target's raw directory.

    The source species is folded into the stem so two donors of the same action
    never claim the same clip name on one target, and so a reviewer can read the
    provenance straight off the file.

    Shape: ``<Action>_<SrcSpecies>Retarget.glb`` -- the action name, an
    underscore, the source species, then the ``Retarget`` provenance marker.
    The species+marker is one final token (``KIHumanRetarget``), which keeps the
    filename unambiguous for ``fbx_filename_rules`` and reads as provenance.
    """
    stem = os.path.splitext(source_clip)[0]
    src_bare = bare_species_name(source_species)
    action = stem[len(bare_species_name(source_species)) + 1:] if stem.startswith(
        bare_species_name(source_species) + "_"
    ) else stem
    donor = "".join(ch for ch in src_bare if ch.isalnum())
    action = "".join(ch for ch in action if ch.isalnum())
    return f"{action}_{donor}{RETARGET_MARKER}.glb"


def plan(args) -> list[dict]:
    cond, species, clips = build_index(args.cond)

    policies = [p for p in DEFAULT_POLICIES if not args.actions or p.word in args.actions]
    if args.targets_per_action:
        policies = [
            ActionPolicy(p.word, p.subsets, args.targets_per_action, p.clips_per_target)
            for p in policies
        ]
    if args.clips_per_target:
        policies = [
            ActionPolicy(p.word, p.subsets, p.targets, args.clips_per_target)
            for p in policies
        ]

    humanoid = {k: is_humanoid(cond[k], info.subset) for k, info in species.items()}
    excluded = set(args.exclude_target or ())

    rows: list[dict] = []
    travel_cache: dict[str, bool] = {}
    claimed: set[tuple[str, str]] = set()      # (target species, glb name)
    summary: list[str] = []

    for policy in policies:
        donors = [c for c in clips.values() if policy.word in c.words]
        if args.groups:
            donors = [c for c in donors if c.action_group in args.groups]
        if args.source_species:
            donors = [c for c in donors if bare_species_name(c.species) in args.source_species]
        # The retarget reads the donor's raw file, so a clip whose source has
        # moved or was never recorded cannot be a donor at all -- catching it
        # here keeps the manifest executable instead of failing row by row.
        with_source = [c for c in donors if c.source_path and os.path.isfile(c.source_path)]
        if len(with_source) != len(donors):
            print(
                f"[plan] {policy.word}: {len(donors) - len(with_source)} donor clip(s) "
                f"dropped -- source animation missing on disk"
            )
        donors = with_source
        if not donors:
            summary.append(f"  {policy.word:8s}  no donor clips -- skipped")
            continue
        donors.sort(key=lambda c: c.clip)
        donor_species = sorted({c.species for c in donors})

        # Candidate targets: right body plan, missing the word, restorable.
        candidates = []
        for key, info in species.items():
            if key in donor_species or bare_species_name(key) in excluded:
                continue
            if policy.word in info.words:
                continue          # already owns the action -- nothing to add
            if info.raw_dir is None or not os.path.isdir(info.raw_dir):
                continue
            if not info.tpose_path or not os.path.isfile(info.tpose_path):
                continue
            if not args.allow_composite and is_composite_rig(cond[key]):
                continue
            if "humanoid" in policy.subsets:
                if not humanoid[key]:
                    continue
            elif info.subset not in policy.subsets:
                continue
            candidates.append(key)

        if not candidates:
            summary.append(f"  {policy.word:8s}  no eligible targets -- skipped")
            continue

        # Pair every candidate with the donor species closest to it, then keep the
        # closest pairs.  Ranking per donor rather than once against the word's
        # biggest donor is what keeps a quadruped's swim on quadrupeds and a
        # winged swim on wings when one word has donors from several body plans.
        best: dict[str, tuple[float, str]] = {}
        for donor_key in donor_species:
            for sim in _rank_targets(cond, donor_key, candidates):
                current = best.get(sim.name)
                if current is None or sim.combined_distance < current[0]:
                    best[sim.name] = (float(sim.combined_distance), donor_key)
        ordered = sorted(best.items(), key=lambda item: (item[1][0], item[0]))
        ordered = ordered[: policy.targets]

        donors_by_species: dict[str, list] = defaultdict(list)
        for clip_info in donors:
            donors_by_species[clip_info.species].append(clip_info)
        cursors: dict[str, int] = defaultdict(int)

        # Round-robin within the chosen donor species so different targets receive
        # different clips: coverage of the (species, action) grid is the point,
        # not repeating one clip everywhere.
        added = 0
        for target, (_distance, donor_key) in ordered:
            info = species[target]
            pool = donors_by_species[donor_key]
            for _ in range(policy.clips_per_target):
                donor = pool[cursors[donor_key] % len(pool)]
                cursors[donor_key] += 1
                glb_name = _target_glb_name(donor.clip, donor.species, target)
                if (target, glb_name) in claimed:
                    continue
                claimed.add((target, glb_name))
                action_name = normalize_action_name(
                    info.bare, os.path.splitext(glb_name)[0]
                )
                rows.append({
                    "action_word": policy.word,
                    "source_clip": donor.clip,
                    "source_species": donor.species,
                    "source_file": donor.source_path,
                    "source_frame_range": donor.source_frame_range,
                    "target_species": target,
                    "target_subset": info.subset,
                    "target_tpose": info.tpose_path,
                    "out_glb": os.path.join(
                        os.path.join(args.out_root, info.bare),
                        glb_name,
                    ),
                    "target_clip": f"{info.bare}_{action_name}.npy",
                    "action_group": donor.action_group,
                    "action_label": donor.action_label,
                    "donor_travels": _donor_travels(donor, cond, travel_cache),
                })
                added += 1
        summary.append(
            f"  {policy.word:8s}  donors={len(donors):3d} from "
            f"{len(donor_species)} species -> {len(ordered):3d} targets, "
            f"{added:4d} clips"
        )

    print("\n=== plan ===")
    for line in summary:
        print(line)
    print(f"  {'TOTAL':8s}  {len(rows)} retarget jobs")
    travelling = sum(1 for row in rows if row["donor_travels"])
    if travelling:
        print(f"  {travelling} of them have a TRAVELLING donor -- the native "
              f"retarget carries their root XZ through, and run asserts it did.")

    by_target = defaultdict(int)
    for row in rows:
        by_target[row["target_species"]] += 1
    print(f"  touching {len(by_target)} target species")

    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  manifest -> {args.manifest}")
    return rows


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def _rig_joint_count(rig_path: str) -> int:
    """Joint count of a rig file, read exactly the way the export is verified."""
    from motion_lib import FBX

    _animation, names, _frame_time = FBX.load(rig_path)
    return len(names)


def _expected_frame_count(frame_range) -> int | None:
    """Frames the export should hold, from the donor's recorded source slice."""
    if isinstance(frame_range, (list, tuple)) and len(frame_range) == 2:
        start, end = int(frame_range[0]), int(frame_range[1])
        if end > start:
            return end - start
    return None


def _read_exported_glb(glb_path: str):
    """Read an exported GLB back exactly the way preprocessing will read it.

    Both post-export checks below need the same file, and the bpy import is the
    second most expensive thing a row does after the joint mapping, so it is
    read once and handed to both.
    """
    from motion_lib import FBX

    return FBX.load(glb_path)


def _verify_exported_glb(animation, names, expected_frames: int | None, expected_joints: int) -> None:
    """Check a re-read export the way preprocessing would, and reject a bad one.

    The export reports success even when the glTF exporter writes the rig as
    plain nodes instead of an armature (it happens on T-pose assets whose armature
    object and a mesh share a name -- retargeting that species' OWN clip fails the
    same way).  Such a file is unusable and would only surface much later as a
    preprocessing error, so it is checked here and never left in the raw directory.

    The joint count is compared against the target rig read the same way, not
    against cond: the native export keeps the rig's own skeleton, which is not
    the collapsed joint set cond records.
    """
    if len(names) != expected_joints:
        raise RuntimeError(
            f"exported GLB has {len(names)} joints, expected {expected_joints}"
        )
    if expected_frames is not None and len(animation) != expected_frames:
        raise RuntimeError(
            f"exported GLB has {len(animation)} frames, expected {expected_frames}"
        )


def _exported_glb_travels(animation) -> bool:
    """True when the exported clip actually moves in XZ.

    Measured over every joint, not joint 0, for the same reason ``_donor_travels``
    does it: plenty of rigs park a static wrapper (``root``, ``Dummy Rig``) above
    the joint that carries the locomotion, and joint 0 then never moves.  The
    KI_Soldier crawls carry 39 units of travel on ``Hips`` under a ``root`` that
    sits still -- judged on joint 0 the SOURCE file fails its own test.
    """
    from motion_lib.Animation import positions_global

    global_positions = positions_global(animation)
    span = float(np.ptp(global_positions[..., 1]))
    displacement = global_positions[-1, :, :] - global_positions[0, :, :]
    net = float(np.linalg.norm(displacement[:, [0, 2]], axis=-1).max())
    return net > max(_TRAVEL_EPSILON, 0.05 * span)


def _load_manifest(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Joint-mapping prefetch
# ---------------------------------------------------------------------------
# A batch run asks the LLM for the joint mapping of every distinct skeleton
# pair and re-imports the same donor/target rig through bpy for every clip.
# The source file is read the way ``export_glb`` sees it (through the
# ``Skeleton``'s torch buffers) and the target rig is read raw; both reads are
# memoised per file revision, so each file is imported once.  These helpers
# live here rather than in ``utils.retarget_pipeline`` because they exist only
# for this batch path -- the pipeline keeps just the loaders that
# ``retarget_glb_to_glb`` itself uses.

_SOURCE_RIG_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_SOURCE_RIG_CACHE_LIMIT = 64


def _load_source_rest_skeleton(path: str):
    """Read a motion file's rest skeleton as ``export_glb`` sees it, memoised.

    The exporter's view goes through the ``Skeleton``'s torch buffers, and the
    LLM prompt is built from those offsets -- reading the ``Animation`` directly
    would be a different, though nearly equal, number.  One donor species fans
    out to a dozen targets, and every one of them would otherwise re-import the
    same file.
    """
    from motion_lib import FBX
    from utils.roundtrip_common import build_skeleton

    try:
        stat = os.stat(path)
        key = (os.path.abspath(path), stat.st_size, stat.st_mtime_ns)
    except OSError:
        key = None

    if key is not None and key in _SOURCE_RIG_CACHE:
        _SOURCE_RIG_CACHE.move_to_end(key)
        names, parents, offsets, rest_rotations = _SOURCE_RIG_CACHE[key]
    else:
        # One frame is enough: only the rest skeleton is read out.
        animation, names, _frame_time = FBX.load(path, start=0, end=1, collapse_root=False)
        skeleton = build_skeleton(
            names,
            np.asarray(animation.offsets, dtype=np.float64),
            np.asarray(animation.parents, dtype=np.int32),
            rest_rotations=np.asarray(animation.orients.qs, dtype=np.float64),
        )
        parents, offsets, rest_rotations = _exporter_view_of_skeleton(skeleton)
        if key is not None:
            _SOURCE_RIG_CACHE[key] = (list(names), parents, offsets, rest_rotations)
            while len(_SOURCE_RIG_CACHE) > _SOURCE_RIG_CACHE_LIMIT:
                _SOURCE_RIG_CACHE.popitem(last=False)

    return list(names), parents.copy(), offsets.copy(), rest_rotations.copy()


def _load_pair_skeletons(source_path: str, target_path: str) -> dict:
    """Read both rigs the way :func:`retarget_glb_to_glb` reads them.

    Returns everything the joint mapping depends on -- parents, rest offsets,
    rest rotations and canonical match names for each side -- and nothing that
    depends on the clip, so one read covers every clip a donor rig sends to a
    target rig.

    Both reads are memoised per file, so a donor species fanning out to a dozen
    targets imports its rig once.  A miss goes through bpy, so this must be
    called from the thread that owns the Blender scene; the mapping itself
    (:func:`_resolve_joint_mapping`) is pure numpy plus an HTTP call and can
    then run anywhere.
    """
    source_names, src_parents, src_rest_offsets, src_rest_rotations = (
        _load_source_rest_skeleton(source_path)
    )
    tgt_names, tgt_parents, tgt_rest_offsets, tgt_rest_rotations = (
        _load_native_rig_skeleton(target_path)
    )
    return {
        "src_parents": src_parents,
        "src_rest_offsets": src_rest_offsets,
        "src_rest_rotations": src_rest_rotations,
        "src_match_names": canonical_match_names_from_raw_skeleton(
            source_names, src_parents, src_rest_offsets,
        ),
        "tgt_parents": tgt_parents,
        "tgt_rest_offsets": tgt_rest_offsets,
        "tgt_rest_rotations": tgt_rest_rotations,
        "tgt_match_names": canonical_match_names_from_raw_skeleton(
            tgt_names, tgt_parents, tgt_rest_offsets,
        ),
    }


def _resolve_joint_mapping(skeletons: dict) -> None:
    """Resolve -- and thereby cache -- the joint mapping for one skeleton pair.

    Whenever the two skeletons are not name-identical the retarget asks an LLM
    to map their joints, and that single round trip costs an order of magnitude
    more than everything else a clip's retarget does.  The answer depends only on
    the pair, so it is cached on disk; running this over a batch's distinct pairs
    first -- concurrently, which the endpoint serves by batching -- turns those
    round trips from a per-pair serial wait into one overlapped one.

    Takes the dict :func:`_load_pair_skeletons` returns.  Pure numpy plus the
    API call: no bpy, so it is safe in a worker thread.
    """
    from utils.retarget_core import retarget_world_space_np

    joint_count = len(skeletons["src_match_names"])
    identity_rotations = np.zeros((1, joint_count, 4), dtype=np.float64)
    identity_rotations[..., 0] = 1.0
    retarget_world_space_np(
        src_parents=skeletons["src_parents"],
        src_rest_offsets=skeletons["src_rest_offsets"],
        src_rest_rotations=skeletons["src_rest_rotations"],
        tgt_parents=skeletons["tgt_parents"],
        tgt_rest_offsets=skeletons["tgt_rest_offsets"],
        tgt_rest_rotations=skeletons["tgt_rest_rotations"],
        src_joint_rotations=identity_rotations,
        src_root_translation=np.zeros((1, 3), dtype=np.float64),
        src_root_rotation=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        src_match_names=skeletons["src_match_names"],
        tgt_match_names=skeletons["tgt_match_names"],
        coordinate_search=False,
        verbose=False,
    )


def _prefetch_joint_mappings(rows: list[dict], workers: int) -> None:
    """Warm the joint-mapping cache for every distinct skeleton pair up front.

    Whenever the donor and target rigs are not name-identical the retarget asks
    an LLM to map their joints, and that single round trip dominates a row:
    measured against the local endpoint it is ~6 s, against ~0.5 s for
    everything else the row does, bpy imports included.  The answer depends only
    on the two skeletons -- every clip a donor species sends to a target species
    asks the identical question -- so a 299-row plan holds only ~117 distinct
    questions, and they can be asked concurrently: the endpoint batches them
    (5 answered in 9.4 s, against 6.3 s each when asked one at a time).

    The skeletons are read here on the main thread, because reading them goes
    through bpy and bpy is a single global scene; only the mapping calls, which
    are numpy plus HTTP, are handed to the pool.

    Best effort: a pair that fails here is not reported as a failure, it just
    pays for its own call later in the row loop, where the error lands against a
    named clip.
    """
    # One representative source FILE per donor species: clips of a species share
    # a rig, so they share the question -- and the answer.
    source_file_by_species: dict[str, str] = {}
    for row in rows:
        if row.get("source_file"):
            source_file_by_species.setdefault(row["source_species"], row["source_file"])

    pairs = sorted({
        (row["source_species"], row["target_species"], row["target_tpose"])
        for row in rows
        if row["source_species"] in source_file_by_species and row.get("target_tpose")
    })
    if len(pairs) < 2:
        return

    print(
        f"\n[run] resolving joint mappings for {len(pairs)} skeleton pair(s) "
        f"with {workers} worker(s)"
    )
    start = time.perf_counter()
    skeletons = []
    for source_species, target_species, target_rig in pairs:
        source_file = source_file_by_species[source_species]
        if not os.path.isfile(source_file) or not os.path.isfile(target_rig):
            continue
        try:
            skeletons.append((
                f"{source_species} -> {target_species}",
                _load_pair_skeletons(source_file, target_rig),
            ))
        except Exception as exc:                       # noqa: BLE001 - best effort
            print(f"    skipped {source_species} -> {target_species}: {exc}")

    if not skeletons:
        return

    # The first one runs inline so the shared LLM client is built once, on this
    # thread, instead of being raced into existence by every worker at once.
    label, first = skeletons[0]
    remaining = skeletons[1:]
    try:
        _resolve_joint_mapping(first)
    except BaseException as exc:                       # noqa: BLE001 - SystemExit too
        # The mapping call aborts the process on a dead endpoint; here that is
        # only a warning, and the row loop will fail loudly against a clip.
        print(f"    joint-mapping prefetch unavailable ({label}): {exc}")
        return

    if remaining:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(_resolve_joint_mapping, skeleton): pair_label
                for pair_label, skeleton in remaining
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:           # noqa: BLE001 - SystemExit too
                    print(f"    joint-mapping prefetch failed ({futures[future]}): {exc}")
    print(f"[run] joint mappings ready in {time.perf_counter() - start:.1f}s")


def run(args) -> int:
    from utils.retarget_pipeline import retarget_glb_to_glb

    rows = _load_manifest(args.manifest)
    if args.actions:
        rows = [r for r in rows if r["action_word"] in args.actions]
    if args.groups:
        rows = [r for r in rows if r.get("action_group") in args.groups]
    if args.target_species:
        rows = [
            r for r in rows
            if bare_species_name(r["target_species"]) in args.target_species
        ]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("[run] nothing to do")
        return 0

    os.makedirs(args.work_dir, exist_ok=True)

    # Group by target so consecutive rows reuse the same target rig, and so the
    # per-target joint count is read once.
    rows.sort(key=lambda r: (r["target_species"], r["source_species"], r["source_clip"]))
    target_joint_cache: dict[str, int] = {}

    pending = [r for r in rows if args.force or not os.path.isfile(r["out_glb"])]
    if pending:
        _prefetch_joint_mappings(pending, args.mapping_workers)

    done, failed = [], []
    for index, row in enumerate(rows, start=1):
        target = row["target_species"]
        tag = f"[{index}/{len(rows)}] {row['action_word']}: {row['source_clip']} -> {target}"
        if os.path.isfile(row["out_glb"]) and not args.force:
            print(f"{tag}  SKIP (exists)")
            done.append(row)
            continue
        print(f"\n{tag}")
        try:
            source_file = row["source_file"]
            if not os.path.isfile(source_file):
                raise RuntimeError(f"source animation missing: {source_file}")
            target_rig = row["target_tpose"]
            if not os.path.isfile(target_rig):
                raise RuntimeError(f"target rig missing: {target_rig}")

            if target not in target_joint_cache:
                target_joint_cache[target] = _rig_joint_count(target_rig)

            os.makedirs(os.path.dirname(row["out_glb"]), exist_ok=True)
            # align_facing + ground reproduce what the feature path used to get
            # for free: process_anim canonicalised both rigs to the +Z reference
            # and bake_foot_floor_offset dropped the rebuilt body onto y=0.
            # Native space has neither, and without them a cross-species donor
            # lands facing the wrong way and buried (measured: ~30% of body
            # height for a quadruped-to-quadruped transfer).
            retarget_glb_to_glb(
                source_file,
                target_rig,
                row["out_glb"],
                fps=args.fps,
                align_facing=True,
                ground=True,
                # The world-space transfer stretches the target's bones to reach
                # the donor's proportions and leaves the residual in the pose
                # translation channel; IK puts it back into rotations so the clip
                # preprocessing reads is a rigid body of the target's own size.
                fullbody_ik=not args.no_fullbody_ik,
                fullbody_ik_stretch_factor=args.ik_stretch_factor,
                slice_inds=row.get("source_frame_range"),
                verbose=False,
            )
            expected_frames = _expected_frame_count(row.get("source_frame_range"))
            exported_anim, exported_names, _exported_frame_time = _read_exported_glb(
                row["out_glb"]
            )
            _verify_exported_glb(
                exported_anim, exported_names, expected_frames, target_joint_cache[target]
            )
            # Root translation is transferred verbatim now, so a travelling donor
            # that exports in place means the transfer dropped it -- a bug, not
            # the feature-space loss this check used to tolerate.
            if row.get("donor_travels") and not _exported_glb_travels(exported_anim):
                raise RuntimeError(
                    "donor travels but the exported GLB is in place -- the native "
                    "retarget should carry root XZ through verbatim"
                )
            print(f"    {os.path.basename(source_file)} -> {row['out_glb']}")
            done.append(row)
        except Exception as exc:                       # noqa: BLE001 - reported per row
            # A rejected export must not stay next to the species' real clips:
            # preprocessing would pick it up on the next incremental run.
            if os.path.isfile(row["out_glb"]):
                os.remove(row["out_glb"])
                print(f"    removed rejected export {row['out_glb']}")
            print(f"    FAILED: {exc}")
            if args.traceback:
                traceback.print_exc()
            failed.append({**row, "error": str(exc)})

    _write_labels(done, args)

    report = os.path.join(args.work_dir, "run_report.json")
    with open(report, "w", encoding="utf-8") as handle:
        json.dump({"done": done, "failed": failed}, handle, indent=2, ensure_ascii=False)
    print(f"\n[run] {len(done)} ok, {len(failed)} failed -- report: {report}")
    for row in failed:
        print(f"   FAIL {row['source_clip']} -> {row['target_species']}: {row['error']}")
    return 1 if failed else 0


def _write_labels(rows: list[dict], args) -> None:
    """Append an ``action_labels.jsonl`` row for every clip the GLBs will produce.

    Preprocessing hard-fails on a clip with no label, and the sidecar is the only
    source of truth for the group, so the rows are written now rather than left
    for a later hand pass.  The label is the donor's, canonicalised through the
    same spelling rules the rest of the corpus uses.
    """
    if not rows:
        return
    _, species, _ = build_index(args.cond)
    by_root: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        info = species.get(row["target_species"])
        if info is None or not info.dataset_root:
            continue
        by_root[os.path.join(info.dataset_root, ACTION_LABELS_FILE)].append(row)

    for path, target_rows in sorted(by_root.items()):
        existing = set()
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        existing.add(json.loads(line)["clip"])
        new_rows = []
        for row in target_rows:
            # sidecar keys are extension-less; target_clip is the .npy file name
            new_key = row["target_clip"][:-4] if row["target_clip"].endswith(".npy") \
                else row["target_clip"]
            if new_key in existing:
                continue
            existing.add(new_key)
            words = [w.strip() for w in row["action_label"].split(",") if w.strip()]
            new_rows.append({
                "clip": new_key,
                "action_group": row["action_group"],
                "action_label": canonical_action_label(words),
                "retargeted_from": row["source_clip"],
            })
        if not new_rows:
            print(f"[labels] {path}: nothing new")
            continue
        if args.dry_run_labels:
            print(f"[labels] {path}: would append {len(new_rows)} row(s) (--dry-run-labels)")
            for row in new_rows[:5]:
                print(f"           {json.dumps(row, ensure_ascii=False)}")
            continue
        with open(path, "a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[labels] {path}: appended {len(new_rows)} row(s)")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=("plan", "run"), required=True)
    parser.add_argument("--cond", default=DEFAULT_COND,
                        help=f"Merged cond.npy covering every dataset. Default: {DEFAULT_COND}")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="Plan JSONL to write (plan) or execute (run).")
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR,
                        help="Where the run report goes.")
    parser.add_argument("--actions", nargs="*", default=None,
                        help="Restrict to these action words.")
    parser.add_argument("--targets-per-action", type=int, default=None,
                        help="Override the per-action target-species cap.")
    parser.add_argument("--clips-per-target", type=int, default=None,
                        help="Override how many donor clips each target receives.")
    parser.add_argument("--source-species", nargs="*", default=None,
                        help="plan: restrict donors to these bare species names.")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                        help=f"plan: stage the GLBs under <out-root>/<Species>/. "
                             f"Default: {DEFAULT_OUT_ROOT} -- a review directory under "
                             f"the Anytop outputs folder, never the dataset.")
    parser.add_argument("--exclude-target", nargs="*", default=None,
                        help="plan: bare species names that must never be targets.")
    parser.add_argument("--allow-composite", action="store_true",
                        help="plan: allow mount+rider rigs (TNR_Cavalry, TTR_MountedKnight) "
                             "as targets. Off by default -- a donor animates only one of "
                             "the two bodies and the other stays frozen.")
    parser.add_argument("--target-species", nargs="*", default=None,
                        help="run: only execute rows landing on these bare species.")
    parser.add_argument("--groups", nargs="*", default=None,
                        choices=("locomotion", "stationary", "transition"),
                        help="Restrict to donors in these action groups. plan and run.")
    parser.add_argument("--fps", type=float, default=DEFAULT_EXPORT_FPS,
                        help="run: force the exported GLB's frame rate. Default: the "
                             "donor file's own rate. This retimes rather than "
                             "resamples, so only set it for a mis-tagged source.")
    parser.add_argument("--limit", type=int, default=None,
                        help="run: execute at most this many rows (smoke test).")
    parser.add_argument("--force", action="store_true",
                        help="run: re-export even when the output GLB already exists.")
    parser.add_argument("--no-fullbody-ik", action="store_true",
                        help="run: write the retarget's own pose channels instead of "
                             "re-solving them on the rigid target skeleton. The "
                             "world-space transfer reaches the donor's proportions by "
                             "stretching the target's bones, so this leaves the target "
                             "deformed -- only useful for diagnosing the transfer.")
    parser.add_argument("--ik-stretch-factor", type=float,
                        default=DEFAULT_IK_STRETCH_FACTOR,
                        help="run: bone-length elasticity the IK rebuild may keep "
                             f"(default {DEFAULT_IK_STRETCH_FACTOR}, i.e. +/-10%%). "
                             "Ignored with --no-fullbody-ik.")
    parser.add_argument("--mapping-workers", type=int, default=4,
                        help="run: how many skeleton pairs may ask the joint-mapping "
                             "LLM at once before the row loop starts. That call is the "
                             "most expensive part of a cold run and the endpoint "
                             "batches concurrent requests; 1 disables the overlap.")
    parser.add_argument("--dry-run-labels", action="store_true",
                        help="run: print the action_labels.jsonl rows instead of appending them.")
    parser.add_argument("--traceback", action="store_true",
                        help="run: print a full traceback for each failure.")
    args = parser.parse_args()

    if args.mode == "plan":
        plan(args)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
