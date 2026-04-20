# T5Conditioner Joint Name Preprocessing — Changes & Rationale

File: `model/conditioners.py` · Class: `T5Conditioner`

---

## Background

`T5Conditioner.tokenize()` converts raw skeleton joint names (e.g. `Bip01_L_ForeArm`) into
text strings that are then encoded by the T5 encoder. The quality of these strings directly
affects how well the model generalises across animals with different rigging conventions.

The dataset (Truebones Zoo) was created by multiple artists using different naming conventions,
which introduced several systematic inconsistencies. The changes below fix them.

---

## Changes

### 1. Capitalization normalization

**Location:** `_split_and_replace()`, `else` branch

**Before:**
```python
new_splitted.append(clean_part)
```

**After:**
```python
new_splitted.append(clean_part[0].upper() + clean_part[1:])
```

**Problem:** Some rigs use lowercase-start tokens (e.g. `BN_hand_R_01` → `hand`), while
identical bones in other rigs are uppercase (`BN_R_Hand_01` → `Hand`). T5 tokenizes `hand`
and `Hand` differently, creating spurious embedding distance between anatomically equivalent
joints.

---

### 2. Left/Right word-order normalization

**Location:** end of `_split_and_replace()`

**Before:**
```python
return ' '.join(new_splitted)
```

**After:**
```python
sides = [w for w in new_splitted if w in ("Left", "Right")]
rest  = [w for w in new_splitted if w not in ("Left", "Right")]
return ' '.join(sides + rest)
```

**Problem:** Different rigs place the side indicator at different positions:
- `Bip01_L_Foot` → `Left Foot`
- `jt_Foot_L` → `Foot Left`

The two strings share the same tokens but in different order, giving a cosine similarity of
~0.83 instead of 1.0. Standardising to `{side} {body_part}` eliminates this variance.

---

### 3. `ForeArm` compound-word normalization

**Location:** top of `_split_and_replace()`, before the regex split

**Added:**
```python
s = s.replace('ForeArm', 'Forearm')
```

**Problem:** The regex `(?=[A-Z]|_)` splits on every uppercase letter, so `ForeArm` becomes
`Fore Arm` (two tokens) while `Forearm` (correct casing) stays as one token `Forearm`. T5
encodes these differently, creating distance between animals that spell the same joint two ways.

Example from dataset:
| Raw name | Before fix | After fix |
|---|---|---|
| `Bip01_R_ForeArm` (FireAnt) | `Right Fore Arm` | `Right Forearm` |
| `Bip01_R_Forearm` (Ant) | `Right Forearm` | `Right Forearm` |

---

### 4. Non-anatomical joint filtering

**Location:** new method `_is_anatomical()` + one line added to `tokenize()`

**Added:**
```python
NON_ANATOMICAL_TOKENS = {"Dummy", "Projectile", "Brain", "Ponytail", "Node", "Nub"}

def _is_anatomical(self, processed: str) -> bool:
    words = set(processed.split())
    if words & self.NON_ANATOMICAL_TOKENS:
        return False
    if {"End", "Site"} <= words:
        return False
    return True
```

In `tokenize()`:
```python
entries = [e if self._is_anatomical(e) else "" for e in entries]
```

Non-anatomical joints are mapped to empty string `""`. The downstream T5 encode path already
handles empty strings (they receive a fixed "null" embedding).

**Problem:** Several animals contain rig/game-engine artefact bones that carry no anatomical
meaning. These pollute the per-animal mean embedding used as a joint conditioning signal.

Filtered categories:

| Token | Source | Example raw name |
|---|---|---|
| `Nub` | Structural end-effector caps | `Bip01_HeadNub`, `Bip01_Tail_Nub` |
| `End Site` | BVH end-site markers | `BN_Tai02_end_site` |
| `Dummy` | Game-engine dummy objects | `Dummy01_HeadFire_` |
| `Projectile` | Projectile spawn nodes | `ProjectileNode_Fire` |
| `Brain` | Game logic nodes | `Bip01_Head_Brain` |
| `Ponytail` | Misnamed antenna/mandible chains | `Bip01_Ponytail2_R_Antenna1` |
| `Node` | Generic engine nodes | `ProjectileNode_Fire` |

Concrete impact on FireAnt (45 raw joints → 27 anatomical after filtering):
- 6× `Ponytail Antenna` removed
- 2× `Ponytail Mandible` removed
- `Dummy Head Fire`, `Projectile Node Fire`, `Head Brain` removed
- 6× `*Nub` removed

---

## Cache invalidation

`T5Conditioner` embeddings for the training dataset are cached in:
```
dataset/truebones/zoo/truebones_processed/joint_name_embs_*.npy
```

**Delete this file before the next training run** so the cache is regenerated with the
corrected preprocessing.
