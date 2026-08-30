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
ACTION_LABELS_FILE = "action_labels.jsonl"
# Offline sidecar: action_label string -> frozen T5 mean-pool vector. Built by
# tools/build_action_label_embeddings.py so training never loads a T5 encoder.
ACTION_LABEL_EMBEDDINGS_FILE = "action_label_embs.npy"
# Sidecar mapping object_type -> portable skinned-mesh T-pose reference path. Kept
# out of cond.npy (no inference path reads it); consumed only by the offline dataset
# GLB tool (data_bridge.restore_glb_from_anytop). Written/refreshed by the dataset
# preprocessing save path and by regenerate_dataset_artifacts.
TPOSE_REFERENCE_SIDECAR = "tpose_reference_paths.jsonl"
# The per-species tag sidecars (species_tags.jsonl / chain_forward_joints.jsonl)
# and everything derived from them -- object subsets, forward-chain overrides --
# are owned by ``dataset_tags``; import ``dataset_tags.dataset_tags()`` there
# rather than caching a copy here.
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


def parse_action_words(raw_action_words):
        """Split a comma/semicolon list of controlled-vocabulary words.

        Used by the evaluation reference prior, which filters dataset clips by the
        words their ``action_label`` hits -- never by ``action_group``, which would
        widen the prior from "the attack references" to "everything stationary"
        and make the score meaningless.
        """
        if raw_action_words is None:
                return tuple()
        if isinstance(raw_action_words, str):
                tokens = raw_action_words.replace(';', ',').split(',')
        else:
                tokens = raw_action_words
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

# A bone shorter than this fraction of the skeleton's mean bone length is a rig
# helper sitting on top of its parent, not a limb: it is excluded from the mean
# bone length that drives scale normalization. 53% of the unitybundles species
# carry at least one (MU04_Pollen's RigSpine is 3.8e-5 against 4.43).
DEGENERATE_BONE_LENGTH_RATIO = 0.02

# A bone this many times longer than the skeleton's 90th-percentile bone is a
# rig socket (a held weapon parked away from the body in the T-pose), not
# anatomy, and is excluded from scale statistics. Calibrated on the 260 dataset
# species: the highest anatomical leaf is at 2.94 (TTR_MountedScout's horse tail
# tip), the lowest prop at 3.34 (RMW_EvilMage's Staff01); 3.25 flags 16 joints
# across 14 unitybundles species (Bow, Arrow, Sword, Shield, ...) and nothing in
# truebones/zoo or zoo_upgrade (closest: the rhino's rider marker at 3.22).
PROP_SOCKET_BONE_LENGTH_RATIO = 3.25

# A socket may carry a short chain (MLS_ElfRanger's ``Bow`` -> ``Bow02``), so an
# over-long bone still counts while at most this many joints hang below it, and
# the whole subtree is excluded with it. Above 4 this starts to reach real
# anatomy.
PROP_SOCKET_MAX_SUBTREE_JOINTS = 4
