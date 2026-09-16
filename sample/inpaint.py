"""``--inpaint_joints`` / ``--inpaint_frames``: mask construction and the
post-sampling vertical fixes.

Frame ranges are given in OUTPUT frames and mapped onto the sampler's window
(``_map_frame_ranges_to_internal``); joint names resolve through every alias
list the cond carries (``_resolve_inpaint_joint_indices``). After sampling,
the regenerated region is put back on the reference's vertical frame, either
by re-integrating vel-Y across a frame span or by re-seating a freed joint
subtree at its clamped boundary.
"""
import numpy as np
import torch


def _parse_frame_ranges(spec, n_frames):
    """Parse '40-90' / '0-20,150-180' / '30' into a set of frame indices,
    inclusive and clipped to [0, n_frames - 1]. Empty spec => all frames.
    """
    if not spec:
        return set(range(n_frames))
    frames = set()
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            lo_str, hi_str = chunk.split('-', 1)
            lo, hi = int(lo_str), int(hi_str)
        else:
            lo = hi = int(chunk)
        if lo > hi:
            lo, hi = hi, lo
        lo = max(0, lo)
        hi = min(n_frames - 1, hi)
        frames.update(range(lo, hi + 1))
    if not frames:
        raise ValueError(
            f"--inpaint_frames '{spec}' selected no valid frames "
            f"(motion has {n_frames} frames, indices 0..{n_frames - 1})"
        )
    return frames


def _map_frame_ranges_to_internal(spec, source_frames, target_frames, warn_remap=False, periodic=False):
    """Map a frame range given in OUTPUT frames onto the sampler's window.

    ``periodic`` picks the mapping the rest of the pipeline uses for a loop
    window (``--loop``): output frame ``s`` is window time ``s*T/M``, the same
    map ``_prepare_img2img_reference_bundle`` and ``_resample_window_to_output``
    apply in the two directions, so a range names the reference poses it names.
    Otherwise the window is an open clip and its frames span ``M-1`` steps.
    """
    if not spec or int(source_frames) == int(target_frames):
        return spec
    source_frames = int(source_frames)
    target_frames = int(target_frames)
    if source_frames <= 0 or target_frames <= 0:
        raise ValueError(
            f"Cannot map frame ranges with source_frames={source_frames}, target_frames={target_frames}"
        )
    if periodic:
        scale = float(target_frames) / float(source_frames)
    else:
        scale = float(target_frames - 1) / float(source_frames - 1) if source_frames > 1 else 0.0
    mapped_frames = set()
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            lo_str, hi_str = chunk.split('-', 1)
            lo, hi = int(lo_str), int(hi_str)
        else:
            lo = hi = int(chunk)
        if lo > hi:
            lo, hi = hi, lo
        lo = max(0, lo)
        hi = min(source_frames - 1, hi)
        if lo > hi:
            continue
        start = max(0, min(target_frames - 1, int(np.floor(float(lo) * scale))))
        end = max(0, min(target_frames - 1, int(np.ceil(float(hi) * scale))))
        mapped_frames.update(range(start, end + 1))
    if not mapped_frames:
        raise ValueError(
            f"--inpaint_frames '{spec}' selected no valid frames "
            f"(motion has {source_frames} frames, indices 0..{source_frames - 1})"
        )
    internal_runs = _contiguous_frame_runs(mapped_frames)
    internal_spec = ','.join(
        f'{start}-{end}' if start != end else str(start)
        for start, end in internal_runs
    )

    # Frame range remapped from visible to internal length; floor/ceil widening
    # causes ~1-2 frame drift. To inpaint exact frames, set --num_frames to match the model's
    # num_frames so visible == internal and no remapping happens.
    if warn_remap:
        inv_scale = 1.0 / scale if scale > 0.0 else 0.0
        effective_runs = [
            (
                int(np.floor(float(start) * inv_scale)),
                int(np.ceil(float(end) * inv_scale)),
            )
            for start, end in internal_runs
        ]
        effective_spec = ','.join(
            f'{a}-{b}' if a != b else str(a) for a, b in effective_runs
        )
        print(
            f'\033[33m  [WARN] --inpaint_frames remapped: requested visible frames '
            f"'{spec}' (output length {source_frames}) -> internal frames "
            f"'{internal_spec}' (sampler length {target_frames}). "
            f'Effective regenerated visible region is ~{effective_spec}, not the '
            f'exact frames requested (boundaries drift ~1-2 frames from floor/ceil '
            f'widening and the {source_frames}->{target_frames} resolution change).\033[0m'
        )
    return internal_spec


def _contiguous_frame_runs(frame_set):
    """Convert a set of frame indices into a list of [start, end] inclusive
    runs of consecutive frames, sorted ascending.
    """
    if not frame_set:
        return []
    sorted_frames = sorted(frame_set)
    runs = []
    start = prev = sorted_frames[0]
    for f in sorted_frames[1:]:
        if f == prev + 1:
            prev = f
            continue
        runs.append((start, prev))
        start = prev = f
    runs.append((start, prev))
    return runs


def _reanchor_inpaint_root_y_via_velocity(motion_np, spans):
    """Fix inpaint Y misalignment per joint via vel-Y integral + linear ramp.

    For each inpaint span [a, b], replaces pos[..., 1] with cumulative vel-Y
    from a-1 anchored to pos_y[a-1], then ramps to match pos_y[b+1]. This
    closes the Y seam while preserving per-frame articulation.

    No-op when a span touches frame 0 or F-1 (no neighbour on one side).
    """
    if not spans:
        return
    F, J, C = motion_np.shape
    if C < 11 or J == 0:
        return
    # Slice views -> writes propagate to motion_np.
    pos_y = motion_np[:, :, 1]   # (F, J) world Y
    vel_y = motion_np[:, :, 10]  # (F, J) vel[f] = pos[f+1] - pos[f]
    for a, b in spans:
        if a < 1 or b > F - 2 or a > b:
            continue  # no clamped anchor on both sides
        L = b - a + 1
        integrated = pos_y[a - 1:a] + np.cumsum(
            vel_y[a - 1:b], axis=0, dtype=np.float64,
        )  # (L, J)
        integrated_at_b_plus_1 = integrated[-1] + vel_y[b]
        adjust = pos_y[b + 1] - integrated_at_b_plus_1  # (J,) residual to close
        ramp = (np.arange(1, L + 1, dtype=np.float64) / float(L + 1))[:, None]  # (L, 1)
        pos_y[a:b + 1] = (integrated + adjust[None, :] * ramp).astype(
            pos_y.dtype, copy=False,
        )


def _reground_inpaint_joint_y(motion_np, ref_motion_np, free_joint_indices, parents):
    """Re-ground regenerated subtree Y onto the reference.

    ``--inpaint_joints`` frees a subset of joints while the rest stay clamped
    to the reference. The free joints live in the model's own vertical frame
    which can float above the grounded body. This computes a constant offset
    at the subtree *boundary* (free joints whose parent is clamped) and shifts
    the whole free subtree to meet the reference's grounded height.

        boundary = { j in free : parent(j) not in free }
        delta    = mean_{j in boundary, t}( ref_y - gen_y )
        gen_y[:, free] += delta
    """
    if ref_motion_np is None or motion_np.ndim != 3:
        return 0.0
    F, J, C = motion_np.shape
    if C < 2 or F == 0 or ref_motion_np.shape[:2] != (F, J):
        return 0.0
    free = [int(j) for j in free_joint_indices if 0 <= int(j) < J]
    if not free:
        return 0.0
    free_set = set(free)
    parents = np.asarray(parents).reshape(-1)
    # Boundary joints: free joints whose parent is clamped — anchor the reseat here.
    boundary = [
        j for j in free
        if 0 <= int(parents[j]) < J and int(parents[j]) not in free_set
    ]
    if not boundary:
        boundary = free
    b_idx = np.asarray(boundary, dtype=np.int64)
    delta = float(np.mean(
        ref_motion_np[:, b_idx, 1].astype(np.float64)
        - motion_np[:, b_idx, 1].astype(np.float64)
    ))
    if delta == 0.0 or not np.isfinite(delta):
        return 0.0
    f_idx = np.asarray(free, dtype=np.int64)
    motion_np[:, f_idx, 1] += np.asarray(delta, dtype=motion_np.dtype)
    return delta


def _resolve_inpaint_joint_indices(cond_entry, names_arg, include_subtree):
    """Resolve comma-separated joint names to a set of joint indices.

    Names are matched against the union of the raw / canonical / canonical_bvh
    alias lists (all same length and index order). When include_subtree is set,
    every descendant of a selected joint is added too. Empty names_arg => all
    real joints.
    """
    raw_names = list(cond_entry['joints_names'])
    n_joints = len(raw_names)
    canon = list(cond_entry.get('canonical_joint_names', raw_names))
    canon_bvh = list(cond_entry.get('canonical_bvh_joint_names', raw_names))

    if not names_arg:
        base = set(range(n_joints))
    else:
        alias_to_index = {}
        for idx in range(n_joints):
            for alias in (raw_names[idx], canon[idx], canon_bvh[idx]):
                if alias is not None:
                    alias_to_index.setdefault(str(alias), idx)
        base = set()
        invalid = []
        for token in names_arg.split(','):
            token = token.strip()
            if not token:
                continue
            if token in alias_to_index:
                base.add(alias_to_index[token])
            else:
                invalid.append(token)
        if invalid:
            table = ['  idx | raw | canonical | canonical_bvh']
            for idx in range(n_joints):
                table.append(f'  {idx:>3} | {raw_names[idx]} | {canon[idx]} | {canon_bvh[idx]}')
            raise ValueError(
                f"--inpaint_joints: unknown joint name(s) {invalid}.\n"
                "Accepted names (any of the three aliases):\n" + '\n'.join(table)
            )

    if not include_subtree or not base:
        return base, n_joints

    parents = np.asarray(cond_entry['parents'], dtype=np.int64)
    children = [[] for _ in range(n_joints)]
    for j in range(n_joints):
        p = int(parents[j])
        if 0 <= p < n_joints:
            children[p].append(j)
    selected = set(base)
    stack = list(base)
    while stack:
        cur = stack.pop()
        for child in children[cur]:
            if child not in selected:
                selected.add(child)
                stack.append(child)
    return selected, n_joints


def build_inpaint_mask(
    cond_entry,
    inpaint_joints_arg,
    inpaint_include_subtree,
    inpaint_frames_arg,
    batch_size,
    max_joints,
    n_frames,
):
    """Build the inpainting mask tensor [B, max_joints, 1, n_frames].

    Convention: 1.0 = regenerate (free), 0.0 = keep reference (clamped).
    Padding joints (index >= n_joints) stay 0.0. The regenerated region is
    selected-joints x selected-frames; everything else is held to the
    reference during sampling.
    """
    joint_indices, n_joints = _resolve_inpaint_joint_indices(
        cond_entry, inpaint_joints_arg, inpaint_include_subtree
    )
    frame_indices = _parse_frame_ranges(inpaint_frames_arg, n_frames)
    if not joint_indices:
        raise ValueError("--inpaint_joints resolved to an empty joint set")

    mask = np.zeros((max_joints, 1, n_frames), dtype=np.float32)
    j_idx = np.fromiter(
        (j for j in joint_indices if 0 <= j < n_joints), dtype=np.int64
    )
    f_idx = np.fromiter(
        (f for f in frame_indices if 0 <= f < n_frames), dtype=np.int64
    )
    if j_idx.size and f_idx.size:
        mask[np.ix_(j_idx, [0], f_idx)] = 1.0

    n_regen_joints = int(j_idx.size)
    n_regen_frames = int(f_idx.size)
    print(
        f'  Inpaint mask: regenerating {n_regen_joints}/{n_joints} joints x '
        f'{n_regen_frames}/{n_frames} frames '
        f'(joints={sorted(int(j) for j in j_idx)[:20]}'
        f'{"..." if n_regen_joints > 20 else ""}, '
        f'subtree={"on" if inpaint_include_subtree else "off"})'
    )

    mask_t = torch.from_numpy(mask).unsqueeze(0).expand(
        batch_size, -1, -1, -1
    ).contiguous()
    return mask_t
