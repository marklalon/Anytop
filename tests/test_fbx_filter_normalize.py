"""Quick verification tests for FBX filtering and action-name normalization."""
import os
import re
import sys

# Copy of the two new functions from motion_process.py
def _normalize_action_name(object_type, raw_action):
    obj_lower = object_type.lower()
    if not raw_action:
        return raw_action

    # Step 1 — strip {species}ALL{sep}
    all_prefix = re.compile(rf'^{re.escape(obj_lower)}all[-_\s]', re.IGNORECASE)
    raw_action = all_prefix.sub('', raw_action)
    if not raw_action:
        return raw_action

    # Step 2 — strip {species}{sep}
    species_prefix = re.compile(rf'^{re.escape(obj_lower)}[-_\s]', re.IGNORECASE)
    raw_action = species_prefix.sub('', raw_action)
    if not raw_action:
        return raw_action

    has_spaces = ' ' in raw_action
    is_all_lowercase = raw_action.islower()
    starts_with_lower = raw_action[0].islower() if raw_action else False

    if (has_spaces and starts_with_lower) or is_all_lowercase:
        parts = re.split(r'[^a-zA-Z0-9]+', raw_action)
        parts = [p for p in parts if p]
        if not parts:
            return raw_action
        return ''.join(p[0].upper() + p[1:] for p in parts)
    return raw_action

def _should_skip_fbx(file_path, object_type):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    stem_lower = stem.lower()
    obj_lower = object_type.lower()
    for sep in ('', '_', '-'):
        pattern = re.compile(rf'^{re.escape(obj_lower)}{re.escape(sep)}all$', re.IGNORECASE)
        if pattern.match(stem):
            return True
    # NoSaddle T-pose variants
    nosaddle_pattern = re.compile(
        rf'^{re.escape(obj_lower)}[-_]\s*nosaddle$', re.IGNORECASE
    )
    if nosaddle_pattern.match(stem_lower):
        return True
    if stem_lower == obj_lower:
        return True
    variant1 = re.compile(rf'^{re.escape(obj_lower)}[a-z]_\w+$', re.IGNORECASE)
    variant2 = re.compile(rf'^{re.escape(obj_lower)}_[a-z]\d+$', re.IGNORECASE)
    if variant1.match(stem) or variant2.match(stem):
        return True
    return False

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
    # B01 die starts with uppercase → kept as-is (not a raw lowercase description)
    ('B01 die.fbx', 'Monkey', False, 'B01 die'),
    ('B01 idle.fbx', 'Monkey', False, 'B01 idle'),
    ('B02 atk.fbx', 'Monkey', False, 'B02 atk'),
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
    # Dog-Back Away: strip Dog- → "Back Away" (uppercase start + spaces, left untouched)
    ('Dog-Back Away.fbx', 'Dog', False, 'Back Away'),
]

passed = 0
failed = 0
for file_path, obj_type, expected_skip, expected_action in tests:
    skip_result = _should_skip_fbx(file_path, obj_type)
    skip_ok = (skip_result == expected_skip)

    norm_ok = True
    if not expected_skip and expected_action is not None:
        stem = os.path.splitext(file_path)[0]
        action_result = _normalize_action_name(obj_type, stem)
        norm_ok = (action_result == expected_action)
        if not norm_ok:
            print(f'  [FAIL] {file_path} ({obj_type}): action="{action_result}", expected="{expected_action}"')
    elif skip_result != expected_skip:
        print(f'  [FAIL] {file_path} ({obj_type}): skip={skip_result}, expected={expected_skip}')

    if skip_ok and norm_ok:
        passed += 1
    else:
        failed += 1

print(f'{"="*50}')
print(f'Results: {passed}/{passed+failed} passed, {failed} failed')
sys.exit(0 if failed == 0 else 1)
