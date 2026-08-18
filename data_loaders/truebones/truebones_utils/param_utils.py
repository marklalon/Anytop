import json
import importlib
import os
import sys
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
ACTION_TAGS_FILE = "action_tags.jsonl"
# Sidecar mapping object_type -> portable skinned-mesh T-pose reference path. Kept
# out of cond.npy (no inference path reads it); consumed only by the offline dataset
# GLB tool (data_bridge.restore_glb_from_anytop). Written/refreshed by the dataset
# preprocessing save path and by regenerate_dataset_artifacts.
TPOSE_REFERENCE_SIDECAR = "tpose_reference_paths.jsonl"
# Per-species motion descriptor (body-plan, size/build, locomotion), maintained as
# a JSONL sidecar alongside action_tags.jsonl. One object per line:
#   {"species": "Cat", "species_tags": ["Quadruped", "Small", "Stalking"]}
# This is the single source of truth for the species condition and for the
# object subsets below; do not duplicate the species->tags mapping in code.
SPECIES_TAGS_FILE = "species_tags.jsonl"
# Dataset-specific forward-chain overrides.  The joint indices are tied to a
# particular dataset's skeleton ordering, so they must not live in source code.
CHAIN_FORWARD_JOINTS_FILE = "chain_forward_joints.jsonl"
FOOT_CONTACT_HEIGHT_THRESH = 0.2
FOOT_CONTACT_VEL_THRESH = 0.002
MAX_PATH_LEN = 5.
# Vertical clamp thresholds expressed as a ratio of the character's reference
# body length (measured from the processed skeleton's rest-pose joint span).
# Motion within VERTICAL_CLAMP_MIN_RATIO is left unchanged; only the excess is
# compressed into the [min, max] band.
VERTICAL_CLAMP_MIN_RATIO = 0.3
VERTICAL_CLAMP_MAX_RATIO = 0.5
# Absolute lower bound for the processed translation-root Y height, in the same
# normalized units as the exported motion features.
ROOT_Y_MIN_HEIGHT = -0.5

# Maps object_type -> joint index tuple used to compute the forward direction.
# Loaded from CHAIN_FORWARD_JOINTS_FILE below.  2-tuple (neck, head) means
# ``head - neck``; 3-tuple (base, neck, head) means
# ``(head - neck) + (neck - base)``.
CHAIN_FORWARD_JOINTS = {}

def resolve_species_tags_file(species_tags_file=None, dataset_dir=None):
        """Resolve the species-tag sidecar path used by the current run."""
        if species_tags_file is not None and str(species_tags_file).strip():
                return _resolve_project_path(species_tags_file)
        return Path(get_dataset_dir(dataset_dir)) / SPECIES_TAGS_FILE


def resolve_chain_forward_joints_file(chain_forward_joints_file=None, dataset_dir=None):
        """Resolve the dataset-specific forward-chain sidecar path."""
        if chain_forward_joints_file is not None and str(chain_forward_joints_file).strip():
                return _resolve_project_path(chain_forward_joints_file)
        return Path(get_dataset_dir(dataset_dir)) / CHAIN_FORWARD_JOINTS_FILE


def load_species_tags(dataset_dir=None, species_tags_file=None):
        """Load the per-species motion descriptor from ``SPECIES_TAGS_FILE``.

        Returns an insertion-ordered ``{species: (tag, ...)}`` mapping. The file is
        the single source of truth for the species condition and for
        ``OBJECT_SUBSETS_DICT`` -- there is no in-code fallback, so a missing or
        malformed file fails loudly rather than silently degrading.
        """
        tags_path = resolve_species_tags_file(
                species_tags_file=species_tags_file,
                dataset_dir=dataset_dir,
        )
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
                        species_tags_tuple = tuple(str(tag).strip() for tag in record["species_tags"])
                        if not species or not species_tags_tuple:
                                raise ValueError(
                                        f"{SPECIES_TAGS_FILE}:{line_no} has an empty species or species_tags."
                                )
                        species_tags[species] = species_tags_tuple
        return species_tags


def load_chain_forward_joints(dataset_dir=None, chain_forward_joints_file=None):
        """Load dataset-specific forward-chain joint indices from JSONL."""
        joints_path = resolve_chain_forward_joints_file(
                chain_forward_joints_file=chain_forward_joints_file,
                dataset_dir=dataset_dir,
        )
        if not joints_path.is_file():
                # Most datasets do not need any index-based forward override;
                # they should fall through to semantic head/limb detection.
                return {}

        chain_forward_joints = {}
        with open(joints_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                                continue
                        record = json.loads(line)
                        species = str(record["species"]).strip()
                        raw_indices = record["chain_forward_joints"]
                        if not species:
                                raise ValueError(
                                        f"{CHAIN_FORWARD_JOINTS_FILE}:{line_no} has an empty species."
                                )
                        if not isinstance(raw_indices, (list, tuple)) or len(raw_indices) not in (2, 3):
                                raise ValueError(
                                        f"{CHAIN_FORWARD_JOINTS_FILE}:{line_no} chain_forward_joints "
                                        "must contain 2 or 3 joint indices."
                                )
                        try:
                                indices = tuple(int(index) for index in raw_indices)
                        except (TypeError, ValueError) as exc:
                                raise ValueError(
                                        f"{CHAIN_FORWARD_JOINTS_FILE}:{line_no} contains non-integer "
                                        "chain_forward_joints."
                                ) from exc
                        if any(index < 0 for index in indices):
                                raise ValueError(
                                        f"{CHAIN_FORWARD_JOINTS_FILE}:{line_no} contains negative "
                                        "chain_forward_joints index."
                                )
                        if species in chain_forward_joints:
                                raise ValueError(
                                        f"{CHAIN_FORWARD_JOINTS_FILE}:{line_no} duplicates species "
                                        f"'{species}'."
                                )
                        chain_forward_joints[species] = indices
        return chain_forward_joints


def build_object_subsets_dict(species_tags):
        """Group species by object_subset (the first motion tag) into ``--object_subsets`` keys.

        Keeps the existing ``OBJECT_SUBSETS_DICT`` contract -- ``"all"`` plus a
        lower-cased key per object_subset -- but sources the membership from the
        species motion tags so the mapping never drifts from the descriptor.
        """
        subsets = {"all": list(species_tags.keys())}
        for species, tags in species_tags.items():
                object_subset = tags[0].strip().lower()
                subsets.setdefault(object_subset, []).append(species)
        for object_subset in (
                "quadruped",
                "biped",
                "multiped",
                "serpentine",
                "aquatic",
                "winged",
        ):
                subsets.setdefault(object_subset, [])
        return subsets


def configure_species_tags(species_tags_file=None, dataset_dir=None):
        """Load a run-specific species-tag sidecar into the process globals.

        The preprocessing and artifact-generation modules historically imported
        the tag mapping once at module import time.  Updating these dictionaries
        in place keeps existing imported aliases valid while allowing a custom
        dataset to provide its own ``species_tags.jsonl``.
        """
        loaded_tags = load_species_tags(
                dataset_dir=dataset_dir,
                species_tags_file=species_tags_file,
        )

        rebuilt_subsets = build_object_subsets_dict(loaded_tags)
        rebuilt_subsets["podata"] = (
                rebuilt_subsets["quadruped"]
                + rebuilt_subsets["biped"]
                + rebuilt_subsets["multiped"]
                + rebuilt_subsets["winged"]
        )

        # Some legacy entry points import this module under a short name
        # (``param_utils`` / ``truebones_utils.param_utils``), while the main
        # pipeline uses the package-qualified name.  Synchronize all aliases so
        # a custom sidecar cannot silently update only one copy of the globals.
        param_modules = []
        for module_name in (
                "param_utils",
                "truebones_utils.param_utils",
                "data_loaders.truebones.truebones_utils.param_utils",
        ):
                module = sys.modules.get(module_name)
                if module is not None and module not in param_modules:
                        param_modules.append(module)
        for module in param_modules:
                module.SPECIES_TAGS.clear()
                module.SPECIES_TAGS.update(loaded_tags)
                module.OBJECT_SUBSETS_DICT.clear()
                module.OBJECT_SUBSETS_DICT.update(rebuilt_subsets)

        # These modules keep lazy/derived views over the same mapping.  Refresh
        # them when they were imported before this configuration call (for
        # example, by a direct validation entry point).
        for module_name, module in list(sys.modules.items()):
                if module is None:
                        continue
                if module_name.endswith("physics_joint_annotation"):
                        module._SPECIES_TAGS_LOWER = None
                elif module_name.endswith("animation_utils"):
                        module.FLYING = frozenset(rebuilt_subsets["winged"])
                        module.FISH = frozenset(rebuilt_subsets["aquatic"])
                elif module_name.endswith("skeleton_similarity"):
                        module._GROUP_TAGS_LOWER = {
                                key.lower(): frozenset(tags)
                                for key, tags in loaded_tags.items()
                        }

        return resolve_species_tags_file(
                species_tags_file=species_tags_file,
                dataset_dir=dataset_dir,
        )


def configure_chain_forward_joints(chain_forward_joints_file=None, dataset_dir=None):
        """Load a run-specific chain-forward sidecar into all imported aliases."""
        loaded_joints = load_chain_forward_joints(
                dataset_dir=dataset_dir,
                chain_forward_joints_file=chain_forward_joints_file,
        )

        # The project has legacy short-name and package-qualified imports. Keep
        # every already-imported mapping live so face_orientation sees the same
        # run-specific sidecar in the parent and worker processes.
        modules = []
        for module_name in (
                "param_utils",
                "truebones_utils.param_utils",
                "data_loaders.truebones.truebones_utils.param_utils",
        ):
                module = sys.modules.get(module_name)
                if module is None:
                        try:
                                module = importlib.import_module(module_name)
                        except (ImportError, ModuleNotFoundError):
                                module = None
                if module is not None and module not in modules:
                        modules.append(module)
        for module in modules:
                mapping = getattr(module, "CHAIN_FORWARD_JOINTS", None)
                if isinstance(mapping, dict):
                        mapping.clear()
                        mapping.update(loaded_joints)

        for module in list(sys.modules.values()):
                if module is None or not getattr(module, "__name__", "").endswith("face_orientation"):
                        continue
                mapping = getattr(module, "CHAIN_FORWARD_JOINTS", None)
                if isinstance(mapping, dict):
                        mapping.clear()
                        mapping.update(loaded_joints)

        return resolve_chain_forward_joints_file(
                chain_forward_joints_file=chain_forward_joints_file,
                dataset_dir=dataset_dir,
        )


SPECIES_TAGS = load_species_tags()
CHAIN_FORWARD_JOINTS.update(load_chain_forward_joints())

# object_subset groupings for ``--object_subsets``. Keys are ``"all"`` plus the
# lower-cased first motion tag (quadruped / biped / multiped / serpentine /
# aquatic / winged); values are derived from SPECIES_TAGS.
OBJECT_SUBSETS_DICT = build_object_subsets_dict(SPECIES_TAGS)

# Composite subset: all footed creatures (有足动物), excluding serpentine & aquatic.
# Combines quadruped + biped + multiped + winged.
OBJECT_SUBSETS_DICT["podata"] = (
        OBJECT_SUBSETS_DICT["quadruped"]
        + OBJECT_SUBSETS_DICT["biped"]
        + OBJECT_SUBSETS_DICT["multiped"]
        + OBJECT_SUBSETS_DICT["winged"]
)


def object_subset_for_object_type(object_type):
        """Return the ``object_subset`` key for a species / ``object_type``.

        The object_subset is the lowercased first motion tag from
        ``species_tags.jsonl`` (quadruped / biped / multiped / serpentine /
        aquatic / winged) -- the same key used in ``OBJECT_SUBSETS_DICT``. The
        per-object_subset canonical standardization statistics are bucketed on
        this value, so held-out species inherit the stats of their object_subset.
        The species-name lookup is case-insensitive. Returns ``None`` when the
        species carries no tags entry.
        """
        if object_type is None:
                return None
        key = str(object_type).strip()
        if not key:
                return None
        tags = SPECIES_TAGS.get(key)
        if tags is None:
                lowered = key.lower()
                for species, species_tags in SPECIES_TAGS.items():
                        if species.lower() == lowered:
                                tags = species_tags
                                break
        if not tags:
                return None
        return tags[0].strip().lower()


def parse_action_tags(raw_action_tags):
        if raw_action_tags is None:
                return tuple()
        if isinstance(raw_action_tags, str):
                tokens = raw_action_tags.replace(';', ',').split(',')
        else:
                tokens = raw_action_tags
        return tuple(token.strip().lower() for token in tokens if str(token).strip())


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
