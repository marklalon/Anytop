"""Inference compatibility for obsolete, parameter-free skeleton dropout settings."""

REMOVED_SKELETON_DROPS = frozenset((
    'rest_geom_drop_prob', 'mirror_twin_drop_prob', 'lateral_drop_prob',
    'decoder_mirror_twin_drop_prob', 'decoder_lateral_drop_prob',
))


def without_removed_skeleton_drops(config):
    return {key: value for key, value in config.items() if key not in REMOVED_SKELETON_DROPS}
