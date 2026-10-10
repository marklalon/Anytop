"""Single-frame rest-pose BVH from a cond.npy entry.

The rest pose with identity joint rotations *is* the t-pose, so the file is the
cond skeleton itself: already cropped, prop sockets removed, oriented and scaled
into the dataset's canonical training frame. Nothing is read from the source
GLB/FBX.

The hierarchy is written depth-first from joint 0, children in index order --
the layout ``motion_lib.BVH.save`` writes, every joint a ``JOINT`` and no
``End Site``. A cond whose helper joints were appended at the tail is not in
that order, so :func:`tpose_bvh_text` also returns the cond index of each BVH
joint: a reader maps BVH joint ``k`` to cond joint ``order[k]``, never by name.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np


def _bvh_joint_names(cond_entry):
    names = cond_entry.get(
        'canonical_bvh_joint_names',
        cond_entry.get('canonical_joint_names', cond_entry['joints_names']),
    )
    # A BVH reader splits on whitespace: a name must be one token.
    return [re.sub(r'\s+', '_', str(name).strip()) or f'joint_{index}' for index, name in enumerate(names)]


def tpose_dfs_order(parents) -> list[int]:
    """Cond joint indices in the BVH hierarchy order (DFS from joint 0)."""
    parents = np.asarray(parents, dtype=np.int64)
    roots = np.flatnonzero(parents < 0).tolist()
    if roots != [0]:
        raise ValueError(f'a BVH needs exactly one root at joint 0; roots are {roots}')
    children = [[] for _ in range(len(parents))]
    for index, parent in enumerate(parents):
        if parent >= 0:
            children[int(parent)].append(index)
    order = []
    stack = [0]
    while stack:
        index = stack.pop()
        order.append(index)
        stack.extend(reversed(children[index]))
    if len(order) != len(parents):
        raise ValueError(f'DFS from joint 0 reached {len(order)} of {len(parents)} joints')
    return order


def tpose_bvh_text(cond_entry) -> tuple[str, list[int]]:
    """``(bvh text, order)``; BVH joint ``k`` is cond joint ``order[k]``."""
    offsets = np.asarray(cond_entry['offsets'], dtype=np.float64)
    parents = np.asarray(cond_entry['parents'], dtype=np.int64)
    names = _bvh_joint_names(cond_entry)
    if offsets.shape != (len(parents), 3) or len(names) != len(parents):
        raise ValueError(
            f'inconsistent joint counts: offsets={offsets.shape}, parents={len(parents)}, names={len(names)}'
        )
    order = tpose_dfs_order(parents)
    depth = np.zeros(len(parents), dtype=np.int64)
    for index in order[1:]:
        depth[index] = depth[parents[index]] + 1

    lines = ['HIERARCHY']
    open_depths = []
    for index in order:
        while open_depths and open_depths[-1] >= depth[index]:
            closed = open_depths.pop()
            lines.append('\t' * closed + '}')
        tab = '\t' * depth[index]
        keyword = 'ROOT' if index == 0 else 'JOINT'
        x, y, z = offsets[index]
        lines.append(f'{tab}{keyword} {names[index]}')
        lines.append(f'{tab}{{')
        lines.append(f'{tab}\tOFFSET {x:.6f} {y:.6f} {z:.6f}')
        if index == 0:
            lines.append(f'{tab}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation')
        else:
            lines.append(f'{tab}\tCHANNELS 3 Zrotation Yrotation Xrotation')
        open_depths.append(int(depth[index]))
    while open_depths:
        lines.append('\t' * open_depths.pop() + '}')

    # One frame, identity rotations: the root sits at its own offset.
    frame = [*(f'{value:.6f}' for value in offsets[0]), *(['0.000000'] * (3 * len(parents)))]
    lines += ['MOTION', 'Frames: 1', f'Frame Time: {1.0 / 30.0:.6f}', ' '.join(frame)]
    return '\n'.join(lines) + '\n', order


def write_tpose_bvh(cond_entry, path) -> list[int]:
    """Atomically write the t-pose BVH of ``cond_entry`` to ``path``; returns its order."""
    text, order = tpose_bvh_text(cond_entry)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + '.tmp')
    tmp_path.write_text(text, encoding='utf-8', newline='\n')
    os.replace(tmp_path, path)
    return order
