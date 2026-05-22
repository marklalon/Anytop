"""Regression tests for FBX filtering and action-name normalization."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loaders.truebones.truebones_utils.fbx_filename_rules import (  # noqa: E402
    normalize_action_name,
    should_skip_anim,
)

# Test cases: (file, object_type, expected_skip, expected_normalized_action)
tests = [
    # ── Filtering: all-in-one ──
    ('CrabAll.fbx', 'Crab', True, None),
    ('HorseALL.fbx', 'Horse', True, None),
    ('FOXALL.fbx', 'Fox', True, None),
    ('HoundALL.fbx', 'Hound', True, None),
    ('BEARALL.fbx', 'Bear', True, None),
    ('DEERALL.fbx', 'Deer', True, None),
    ('Camel_ALL.fbx', 'Camel', True, None),
    ('Cat-ALL.fbx', 'Cat', True, None),
    ('LionAll.fbx', 'Lion', True, None),
    ('scorpionALL.fbx', 'Scorpion', True, None),
    ('MonkeyAll.fbx', 'Monkey', True, None),
    ('BIRDALL.fbx', 'Bird', True, None),
    ('OstrichALL.fbx', 'Ostrich', True, None),
    # ── Filtering: no action name ──
    ('Fox.fbx', 'Fox', True, None),
    ('Monkey.fbx', 'Monkey', True, None),
    ('Elephant.fbx', 'Elephant', True, None),
    ('Chicken.fbx', 'Chicken', True, None),
    ('Bird.fbx', 'Bird', True, None),
    ('Dog.fbx', 'Dog', True, None),
    ('Ostrich.fbx', 'Ostrich', True, None),
    # ── Filtering: variant codenames ──
    ('FoxA_A02.fbx', 'Fox', True, None),
    ('FoxA_A03.fbx', 'Fox', True, None),
    ('Monkey_B01.fbx', 'Monkey', True, None),
    ('Monkey_B02.fbx', 'Monkey', True, None),
    # ── Normalization: ALL-prefix stripping ──
    ('HorseALL-RunToStop.fbx', 'Horse', False, 'RunToStop'),
    ('HorseALL-Attack.fbx', 'Horse', False, 'Attack'),
    ('HorseALL-RunLoop.fbx', 'Horse', False, 'RunLoop'),
    ('LionAll-Attack.fbx', 'Lion', False, 'Attack'),
    ('LionAll-Run.fbx', 'Lion', False, 'Run'),
    ('LionAll-Walk.fbx', 'Lion', False, 'Walk'),
    # ── Normalization: CamelCase lowercase/spaced ──
    ('atk 1.fbx', 'Fox', False, 'Atk1'),
    ('atk 2.fbx', 'Fox', False, 'Atk2'),
    ('idle 3.fbx', 'Fox', False, 'Idle3'),
    ('die.fbx', 'Fox', False, 'Die'),
    ('run.fbx', 'Fox', False, 'Run'),
    ('walk.fbx', 'Fox', False, 'Walk'),
    ('down loop.fbx', 'Elephant', False, 'DownLoop'),
    ('down in.fbx', 'Elephant', False, 'DownIn'),
    ('down out.fbx', 'Elephant', False, 'DownOut'),
    ('atk 1.fbx', 'Monkey', False, 'Atk1'),
    ('B01 die.fbx', 'Monkey', False, 'B01Die'),
    ('B01 idle.fbx', 'Monkey', False, 'B01Idle'),
    ('B02 atk.fbx', 'Monkey', False, 'B02Atk'),
    ('die2.fbx', 'Fox', False, 'Die2'),
    ('firing.fbx', 'Elephant', False, 'Firing'),
    # ── Filtering: NoSaddle T-pose variant ──
    ('Horse_NoSaddle.fbx', 'Horse', True, None),
    # ── {species}- prefix stripped (was already well-formed but duplicates object_type) ──
    ('Hound-Attack.fbx', 'Hound', False, 'Attack'),
    ('Hound-Idle.fbx', 'Hound', False, 'Idle'),
    ('Crab-Walk.fbx', 'Crab', False, 'Walk'),
    ('DEER-Gallop.fbx', 'Deer', False, 'Gallop'),
    ('Ostrich-Walk.fbx', 'Ostrich', False, 'Walk'),
    ('Chicken-IdlePecking.fbx', 'Chicken', False, 'IdlePecking'),
    ('Camel-Run.fbx', 'Camel', False, 'Run'),
    ('BEAR-Growl.fbx', 'Bear', False, 'Growl'),
    ('Cat-Walk.fbx', 'Cat', False, 'Walk'),
    ('scorpion-Attack1.fbx', 'Scorpion', False, 'Attack1'),
    ('Dog-Back Away.fbx', 'Dog', False, 'BackAway'),
    ('dragon_walk.fbx', 'Dragon', False, 'Walk'),
    # ── Reported real-world regressions ──
    ('T-Rex-bite 90 left.fbx', 'Trex', False, 'Bite90Left'),
    ('T-Rex-Chase Roar.fbx', 'Trex', False, 'ChaseRoar'),
    ('T-Rex-death short.fbx', 'Trex', False, 'DeathShort'),
    ('T-Rex.fbx', 'Trex', True, None),
    ('T-Rex-STILL.fbx', 'Trex', True, None),
    ('BEAR-Attack.fbx', 'BrownBear', False, 'Attack'),
    ('BEARALL.fbx', 'BrownBear', True, None),
    ('Buffalo-Fall.fbx', 'Buffalo', False, 'Fall'),
    ('Piranna-Biting.fbx', 'Pirrana', False, 'Biting'),
    ('PirhannaALL.fbx', 'Pirrana', True, None),
    ('CrabAll-Die.fbx', 'HermitCrab', False, 'Die'),
    ('Parrot-ALL.fbx', 'Parrot2', True, None),
    ('SandMouseA02.fbx', 'SandMouse', True, None),
]


def test_filter_and_normalize_cases():
    for file_path, obj_type, expected_skip, expected_action in tests:
        skip_result = should_skip_anim(file_path, obj_type)
        assert skip_result == expected_skip, (
            f'{file_path} ({obj_type}): skip={skip_result}, expected={expected_skip}'
        )

        if expected_skip or expected_action is None:
            continue

        stem = os.path.splitext(file_path)[0]
        action_result = normalize_action_name(obj_type, stem)
        assert action_result == expected_action, (
            f'{file_path} ({obj_type}): action={action_result}, expected={expected_action}'
        )


if __name__ == '__main__':
    test_filter_and_normalize_cases()
    print(f'{"="*50}')
    print(f'Results: {len(tests)}/{len(tests)} passed, 0 failed')

