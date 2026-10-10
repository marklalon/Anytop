"""precheck_dataset's contact-coverage check: a limb end standing on the floor
at rest that the contact prefill leaves out is reported."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ANYTOP = Path(__file__).resolve().parents[1]
for path in (_ANYTOP, _ANYTOP / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import precheck_dataset as precheck  # noqa: E402
from data_loaders.truebones.truebones_utils.joint_name_canonical import (  # noqa: E402
    refresh_joint_metadata_in_object_cond,
)
from data_loaders.truebones.truebones_utils.joint_parts import prefill_contacts  # noqa: E402
from data_loaders.truebones.truebones_utils.physics_joint_annotation import (  # noqa: E402
    rest_positions_from_offsets,
)

# A quadruped standing on four feet: hips, spine, two front legs, two rear legs,
# and a tail hanging to the floor.
_NAMES = [
    "Hips", "Spine", "LFrontLeg", "LFrontFoot", "RFrontLeg", "RFrontFoot",
    "LBackLeg", "LBackPad", "RBackLeg", "RBackPad", "Tail1", "Tail2",
]
_PARENTS = np.array([-1, 0, 1, 2, 1, 4, 0, 6, 0, 8, 0, 10], dtype=np.int64)
_POSITIONS = np.array([
    [0.0, 1.0, -0.5], [0.0, 1.0, 0.5],
    [0.3, 0.5, 0.5], [0.3, 0.0, 0.5], [-0.3, 0.5, 0.5], [-0.3, 0.0, 0.5],
    [0.3, 0.5, -0.5], [0.3, 0.0, -0.5], [-0.3, 0.5, -0.5], [-0.3, 0.0, -0.5],
    [0.0, 0.5, -1.0], [0.0, 0.0, -1.5],
])


def _cond(names, positions=_POSITIONS):
    offsets = positions.copy()
    offsets[1:] -= positions[_PARENTS[1:]]
    cond = {
        "joints_names": list(names),
        "parents": _PARENTS,
        "offsets": offsets,
        "object_type": "Synthetic",
        "species_name": "Synthetic",
    }
    refresh_joint_metadata_in_object_cond(cond)
    return cond


def test_inferred_contact_feet_are_not_reported():
    cond = _cond(_NAMES)
    rest = rest_positions_from_offsets(cond["offsets"], cond["parents"])
    contact = {_NAMES[index] for index in prefill_contacts(_NAMES, cond["parents"], rest)}
    assert {"LFrontFoot", "RFrontFoot"} <= contact
    assert "LBackPad" not in contact


def test_a_tip_with_no_limb_word_is_judged_by_its_parent():
    # The rear "Pad" tips carry no limb word, but hang off "BackLeg"; the tail
    # lies on the floor too and is not a limb.
    cond = _cond(_NAMES)
    missing = [cond["joints_names"][index] for index in precheck.floor_limb_ends_missing_contact(cond)]
    assert missing == ["LBackPad", "RBackPad"]


def test_a_tip_under_an_excluded_parent_is_not_reported():
    names = [name.replace("BackLeg", "BackLegIK") for name in _NAMES]
    cond = _cond(names)
    assert precheck.floor_limb_ends_missing_contact(cond) == []


def test_a_floor_level_foot_left_out_of_contact_is_reported():
    cond = _cond(_NAMES)
    # Simulate a prefill that saw only the front feet.
    contact = [_NAMES.index("LFrontFoot"), _NAMES.index("RFrontFoot")]
    cond["joints_names"] = [name.replace("BackPad", "BackHoof") for name in _NAMES]
    missing = [cond["joints_names"][index]
               for index in precheck.floor_limb_ends_missing_contact(cond, contact)]
    assert missing == ["LBackHoof", "RBackHoof"]


def test_a_contact_ancestor_covers_its_toe_tip():
    names = [name.replace("BackPad", "BackToe") for name in _NAMES]
    cond = _cond(names)
    contact = [names.index(n) for n in ("LFrontFoot", "RFrontFoot", "LBackLeg", "RBackLeg")]
    assert precheck.floor_limb_ends_missing_contact(cond, contact) == []


def test_a_tail_hanging_below_the_feet_does_not_lower_the_floor():
    names = [name.replace("BackPad", "BackHoof") for name in _NAMES]
    positions = _POSITIONS.copy()
    positions[names.index("Tail2"), 1] = -0.1
    cond = _cond(names, positions)
    contact = [names.index("LFrontFoot"), names.index("RFrontFoot")]
    missing = [names[index] for index in precheck.floor_limb_ends_missing_contact(cond, contact)]
    assert missing == ["LBackHoof", "RBackHoof"]
