# Bundled Dependencies

This directory includes local copies of third-party libraries in the `motion_lib/` subdirectory.

## BVH Motion Library

**Source:** https://github.com/inbar-2344/Motion

**Location:** `motion_lib/` subdirectory

**Files:**
- `motion_lib/BVH.py` - BVH file parsing and saving
- `motion_lib/Animation.py` - Animation data structure and utilities
- `motion_lib/Quaternions.py` - Quaternion mathematics operations
- `motion_lib/AnimationStructure.py` - Animation structure utilities
- `motion_lib/__init__.py` - Package initialization

**Changes from original:**
- **BVH.py (line ~220-225):** Modified root-folding logic to apply parent rotation to child positions during root joint collapse, preserving Y-height information that would otherwise be lost in coordinate transformations.

**Why bundled:**
- The upstream Motion library had a bug with redundant root joint handling (see root cause fix summary)
- Bundling allows the fix to be shipped with the codebase
- No external git dependency installation required

## Usage

Import from the motion_lib package:

```python
# Load BVH files
from motion_lib import BVH
anim, joint_names, dt = BVH.load('path/to/file.bvh')

# Work with animations
from motion_lib import Animation, Quaternions
from motion_lib.Animation import positions_global, rotations_global

# Use animation structure utilities
from motion_lib import AnimationStructure
parents = AnimationStructure.parents_list(anim.parents)
```

All modules are organized under `motion_lib/` to avoid namespace conflicts with the main project code.

