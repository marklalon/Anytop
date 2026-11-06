"""Static skeleton geometry shared by diffusion and autoencoder."""
import numpy as np

from data_loaders.truebones.truebones_utils.canonical_features import (
    _length_scale_from_cond, rest_pos_from_cond, root_relative_positions,
)


# A bone shorter than this fraction of the rig's longest rest bone is a
# coincident helper joint; its direction is noise, so it reads as zero.
BONE_DIRECTION_MIN_REST_FRACTION = 1e-3

# Per-joint pose geometry: unit bone direction (3) and height over ``L`` (1).
POSE_GEOMETRY_CHANNELS = 4
# Rest geometry per joint (``rest_geometry``): the encoder pose channels of the
# rest, then the parent->joint bone length over ``L``, so each consumer's
# columns are one contiguous slice.
REST_GEOMETRY_DIM = POSE_GEOMETRY_CHANNELS + 1
# Direction and height columns of the rest geometry.
REST_POSE_COLUMNS = slice(0, POSE_GEOMETRY_CHANNELS)
# Denoiser: direction and height plus bone length, which a near
# horizontal bone's direction and height cannot recover.
REST_GEOM_COLUMNS = slice(0, REST_GEOMETRY_DIM)
# Target rest-geometry channel: the denoiser columns plus a present flag, since an
# all-zero geometry (a root on the ground) is a real value.
REST_GEOM_DIM = REST_GEOM_COLUMNS.stop - REST_GEOM_COLUMNS.start + 1


def bone_directions(positions, parents, rest_positions):
    """``[T, J, 3]`` unit parent->child directions, one row per frame.

    Physical positions sit in the skeleton-canonicalized rest-facing frame
    (every rest faces +Z; per-frame root yaw is not removed) with absolute
    height, so a direction means the same world direction on every skeleton
    ("points up", "points back"), unlike the local rotation. The rest pose only decides which bones are coincident
    helpers; it is never an output row. Roots and coincident joints are zero.
    """
    positions = np.asarray(positions, np.float32)
    rest_positions = np.asarray(rest_positions, np.float32)
    parents = np.asarray(parents)
    out = np.zeros_like(positions)
    child = np.flatnonzero(parents >= 0)
    if not len(child):
        return out
    rest_norm = np.linalg.norm(rest_positions[child] - rest_positions[parents[child]], axis=-1)
    keep = rest_norm > BONE_DIRECTION_MIN_REST_FRACTION * max(float(rest_norm.max()), 1e-12)
    vec = positions[:, child] - positions[:, parents[child]]
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    # Non-finite positions are tolerated by the ``feats`` path (nan_to_num), so
    # they must not leave a NaN here: inf/inf in the divide is NaN.
    with np.errstate(invalid='ignore', over='ignore'):
        unit = np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 1e-12)
        out[:, child] = unit * keep[None, :, None]
    return np.nan_to_num(out)


def reference_pose(motion, cond):
    """``[T, J, POSE_GEOMETRY_CHANNELS]`` pose geometry.

    Bone directions plus each joint's absolute height over the rig's ``L``
    (ground contact and posture height, comparable across body sizes), read
    off the all-RIC view of the working-layout ``motion``.
    """
    positions = root_relative_positions(np.asarray(motion, np.float32)[..., :3])
    height = positions[..., 1:2] / float(_length_scale_from_cond(cond))
    pose = np.concatenate([bone_directions(positions, cond['parents'], rest_pos_from_cond(cond)),
                           height], axis=-1)
    return np.nan_to_num(pose).astype(np.float32)


def rest_geometry(cond):
    """``[J, REST_GEOMETRY_DIM]`` rest geometry, built once for both consumers.

    Direction, height and bone length over the character scale L.
    The model reads REST_GEOM_COLUMNS and appends a presence flag.
    """
    rest = np.asarray(rest_pos_from_cond(cond), np.float32)
    parents = np.asarray(cond['parents']).reshape(-1)
    pose = reference_pose(rest[None], cond)[0]
    length = np.zeros((len(rest), 1), np.float32)
    child = np.flatnonzero(parents >= 0)
    length[child, 0] = np.linalg.norm(rest[child] - rest[parents[child]], axis=-1)
    length = length / float(_length_scale_from_cond(cond))
    geometry = np.concatenate([pose, length], axis=-1)
    return np.nan_to_num(geometry).astype(np.float32)
