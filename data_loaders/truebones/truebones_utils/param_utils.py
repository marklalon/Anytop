import json
import os
from pathlib import Path
import statistics
import numpy as np


_ANYTOP_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DATA_DIR = str((_ANYTOP_ROOT / "dataset/truebones/zoo/Truebone_Z-OO").resolve())
DEFAULT_DATASET_DIR = str((_ANYTOP_ROOT / "dataset/truebones/zoo/truebones_processed").resolve())


def _resolve_project_path(path_value):
        candidate = Path(path_value)
        if candidate.is_absolute():
                return candidate

        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate.exists():
                return cwd_candidate

        project_candidate = (_ANYTOP_ROOT / candidate).resolve()
        if project_candidate.exists():
                return project_candidate

        return project_candidate


def get_raw_data_dir(raw_data_dir=None):
        if raw_data_dir is not None:
                resolved_dir = _resolve_project_path(raw_data_dir)
        else:
                resolved_dir = _resolve_project_path(DEFAULT_RAW_DATA_DIR)
        
        if not resolved_dir.is_dir():
                raise FileNotFoundError(
                        f"Raw BVH directory not found at: {resolved_dir}\n"
                        f"Please provide a valid path using --raw-data-dir argument.\n"
                        f"Example: python preprocess_and_validate.py --raw-data-dir /path/to/Truebone_Z-OO"
                )
        
        return str(resolved_dir)


def get_dataset_dir(dataset_dir=None):
        if dataset_dir is not None:
                return str(_resolve_project_path(dataset_dir))
        return str(_resolve_project_path(DEFAULT_DATASET_DIR))


MOTION_DIR = "motions"
GLB_DIR = "glb"
BVHS_DIR = "bvhs"
MOTION_METADATA_FILE = "motion_metadata.json"
MOTION_TAGS_FILE = "motion_tags.jsonl"
# Per-species motion descriptor (body-plan, size/build, locomotion), maintained as
# a JSONL sidecar alongside motion_tags.jsonl. One object per line:
#   {"species": "Cat", "motion_tags": ["Quadruped", "Small", "Stalking"]}
# This is the single source of truth for the species condition and for the
# object subsets below; do not duplicate the species->tags mapping in code.
SPECIES_MOTION_TAGS_FILE = "species_motion_tags.jsonl"
FOOT_CONTACT_HEIGHT_THRESH = 0.2
FOOT_CONTACT_VEL_THRESH = 0.002
MAX_PATH_LEN = 5.
# Vertical clamp thresholds expressed as a ratio of the character's reference
# body length (measured from the processed skeleton's rest-pose joint span).
# Motion within VERTICAL_CLAMP_MIN_RATIO is left unchanged; only the excess is
# compressed into the [min, max] band.
VERTICAL_CLAMP_MIN_RATIO = 0.3
VERTICAL_CLAMP_MAX_RATIO = 0.5

# Maps object_type -> joint index tuple used to compute the forward direction for
# creatures without usable limb pairs (snakes, fish).
# 2-tuple (neck, head)        -> forward = head - neck
# 3-tuple (base, neck, head)  -> forward = (head - neck) + (neck - base)
CHAIN_FORWARD_JOINTS = {
    'Anaconda': (22, 24),
    'KingCobra': (4, 8),
    'Pirrana': (9, 2, 3),   # kosi → mune → atama (tail to head)
}

def load_species_motion_tags(dataset_dir=None):
        """Load the per-species motion descriptor from ``SPECIES_MOTION_TAGS_FILE``.

        Returns an insertion-ordered ``{species: (tag, ...)}`` mapping. The file is
        the single source of truth for the species condition and for
        ``OBJECT_SUBSETS_DICT`` -- there is no in-code fallback, so a missing or
        malformed file fails loudly rather than silently degrading.
        """
        tags_path = Path(get_dataset_dir(dataset_dir)) / SPECIES_MOTION_TAGS_FILE
        if not tags_path.is_file():
                raise FileNotFoundError(
                        f"Species motion tags file not found at: {tags_path}\n"
                        f"It is the single source of truth for species tags and object subsets."
                )
        species_tags = {}
        with open(tags_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                                continue
                        record = json.loads(line)
                        species = str(record["species"]).strip()
                        motion_tags = tuple(str(tag).strip() for tag in record["motion_tags"])
                        if not species or not motion_tags:
                                raise ValueError(
                                        f"{SPECIES_MOTION_TAGS_FILE}:{line_no} has an empty species or motion_tags."
                                )
                        species_tags[species] = motion_tags
        return species_tags


def build_object_subsets_dict(species_tags):
        """Group species by body-plan (the first motion tag) into ``--object_subsets`` keys.

        Keeps the existing ``OBJECT_SUBSETS_DICT`` contract -- ``"all"`` plus a
        lower-cased key per body-plan -- but sources the membership from the
        species motion tags so the mapping never drifts from the descriptor.
        """
        subsets = {"all": list(species_tags.keys())}
        for species, motion_tags in species_tags.items():
                body_plan = motion_tags[0].strip().lower()
                subsets.setdefault(body_plan, []).append(species)
        return subsets


SPECIES_MOTION_TAGS = load_species_motion_tags()

# Body-plan groupings for ``--object_subsets``. Keys are ``"all"`` plus the
# lower-cased body-plan tag (quadruped / biped / multiped / serpentine /
# aquatic / winged); values are derived from SPECIES_MOTION_TAGS.
OBJECT_SUBSETS_DICT = build_object_subsets_dict(SPECIES_MOTION_TAGS)


def parse_action_tags(raw_action_tags):
        if raw_action_tags is None:
                return tuple()
        if isinstance(raw_action_tags, str):
                tokens = raw_action_tags.replace(';', ',').split(',')
        else:
                tokens = raw_action_tags
        return tuple(token.strip().lower() for token in tokens if str(token).strip())


def parse_action_tag_weights(raw_action_tag_weights):
        """Parse per-action-tag sampling weights into ``{tag: float}``.

        Accepts a ``'tag:weight,tag:weight'`` string (``;`` also allowed as a
        separator) or an already-parsed mapping. Tag names are lower-cased and
        stripped. Empty / ``None`` input yields an empty dict (uniform sampling).
        Weights must be finite and non-negative.
        """
        if raw_action_tag_weights is None:
                return {}
        if isinstance(raw_action_tag_weights, dict):
                items = raw_action_tag_weights.items()
        else:
                if isinstance(raw_action_tag_weights, str):
                        tokens = raw_action_tag_weights.replace(';', ',').split(',')
                else:
                        tokens = raw_action_tag_weights
                items = []
                for token in tokens:
                        token = str(token).strip()
                        if not token:
                                continue
                        if ':' not in token:
                                raise ValueError(
                                        f"Invalid action_tag_weight '{token}', expected 'tag:weight'."
                                )
                        tag, weight = token.rsplit(':', 1)
                        items.append((tag, weight))

        weights = {}
        for tag, weight in items:
                tag = str(tag).strip().lower()
                if not tag:
                        continue
                weight = float(weight)
                if not np.isfinite(weight) or weight < 0:
                        raise ValueError(
                                f"action_tag_weight for '{tag}' must be finite and non-negative, got {weight}."
                        )
                weights[tag] = weight
        return weights

MAX_JOINTS=100
FPS=30
FEATS_LEN=13
SMPL_OFFSETS = np.array([[ 0.0000,  0.0000,  0.0000],
        [ 0.1031,  0.0000,  0.0000],
        [-0.1099,  0.0000,  0.0000],
        [ 0.0000,  0.1316,  0.0000],
        [ 0.0000, -0.3936,  0.0000],
        [ 0.0000, -0.3902,  0.0000],
        [ 0.0000,  0.1432,  0.0000],
        [ 0.0000, -0.4324,  0.0000],
        [ 0.0000, -0.4256,  0.0000],
        [ 0.0000,  0.0574,  0.0000],
        [ 0.0000,  0.0000,  0.1434],
        [ 0.0000,  0.0000,  0.1494],
        [ 0.0000,  0.2194,  0.0000],
        [ 0.1375,  0.0000,  0.0000],
        [-0.1434,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.1030],
        [ 0.0000, -0.1316,  0.0000],
        [ 0.0000, -0.1230,  0.0000],
        [ 0.0000, -0.2568,  0.0000],
        [ 0.0000, -0.2631,  0.0000],
        [ 0.0000, -0.2660,  0.0000],
        [ 0.0000, -0.2699,  0.0000]])
SMPL_PARENTS = np.array([
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
], dtype=np.int32)
_SMPL_REST_POSITIONS = np.zeros_like(SMPL_OFFSETS)
for _joint_index, _parent_index in enumerate(SMPL_PARENTS):
        if _parent_index >= 0:
                _SMPL_REST_POSITIONS[_joint_index] = _SMPL_REST_POSITIONS[_parent_index] + SMPL_OFFSETS[_joint_index]
        else:
                _SMPL_REST_POSITIONS[_joint_index] = SMPL_OFFSETS[_joint_index]

# Average bone length of axial (center/spine/neck/head) joints in the SMPL skeleton.
# These are indices [3, 6, 9, 12, 15] — the central body chain that excludes all
# left/right limb bones.  Used as the target for uniform skeletal scaling so that
# scaling reflects trunk size rather than total max span (which is inflated by
# long limb bones in some species).
HML_REF_AXIAL_BONE_LENGTH = float(np.linalg.norm(
    SMPL_OFFSETS[[3, 6, 9, 12, 15]],
    axis=1,
).mean())

# Maximum rest-pose joint-to-joint span of the 22-joint HumanML / SMPL skeleton.
# Used as a secondary target so compact characters can scale up and wide/long
# characters can scale down without switching fully to max-span normalization.
HML_REF_MAX_SPAN = float(np.linalg.norm(
        _SMPL_REST_POSITIONS[:, None, :] - _SMPL_REST_POSITIONS[None, :, :],
        axis=-1,
).max())

# Geometric blend weight between axial mean bone length and whole-body max span.
# 0 keeps the existing axial-only scaling; 1 becomes pure max-span scaling.
SCALE_BODY_SPAN_BLEND_WEIGHT = 0.5
