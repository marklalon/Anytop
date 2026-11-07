"""Inference compatibility for obsolete, parameter-free model settings."""

REMOVED_MODEL_OPTIONS = frozenset((
    'rest_geom_drop_prob', 'mirror_twin_drop_prob', 'lateral_drop_prob',
    'decoder_mirror_twin_drop_prob', 'decoder_lateral_drop_prob',
    'temporal_rope', 'cross_limb_dim', 'cross_limb_last_n',
))


def without_removed_options(config):
    return {key: value for key, value in config.items() if key not in REMOVED_MODEL_OPTIONS}
