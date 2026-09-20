"""Side prefill needs actual side-joint motion before naming a side."""

import numpy as np

from tools.prefill_direction_words import Symmetry, side_measure, side_word


def test_center_only_motion_has_no_side_verdict_but_right_limb_motion_does():
    symmetry = Symmetry({
        "symmetry_partner_indices": [-1, 2, 1],
        "joint_side_labels": ["center", "left", "right"],
        "is_symmetric": True,
    })

    class Clip:
        L = 1.0
        ric = np.zeros((30, 3, 3))

    clip = Clip()
    clip.ric[:, 0, 0] = np.linspace(-1.0, 1.0, 30)
    center_only = side_measure(clip, symmetry)
    assert center_only["amp_anti"] > 0.05
    assert center_only["asym_share"] > 0.30
    assert side_word(center_only["left_share"], 0.20) is None

    clip.ric[:, 0, 0] = 0.0
    clip.ric[:, 2, 0] = np.linspace(-1.0, 1.0, 30)
    right_limb = side_measure(clip, symmetry)
    assert side_word(right_limb["left_share"], 0.20) == "right"
