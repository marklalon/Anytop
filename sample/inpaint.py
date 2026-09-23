"""``--inpaint_joints`` / ``--inpaint_frames``: mask construction and the
post-sampling vertical fixes.

Frame ranges are given in OUTPUT frames and mapped onto the sampler's window
(``_map_frame_ranges_to_internal``); joint names resolve through every alias
list the cond carries (``_resolve_inpaint_joint_indices``). After sampling,
the regenerated region is put back on the reference's vertical frame, either
by re-integrating vel-Y across a frame span or by re-seating a freed joint
subtree at its clamped boundary.

Where a range sits decides what ``--loop`` may do (``inpaint_span_role``, which
``resolve_inpaint_policy`` applies to the WINDOW span the range maps onto, not
to the range as written). A span keeping both window ends clamped is INTERIOR:
the run is exported open whatever the loop verdict was, since a periodic export
resamples those clamped ends too and would hand back frames the reference never
had. A span reaching window frame 0 or the last window frame is a closure
request and keeps the loop condition. A span covering every window frame over
every joint is not an inpaint at all and ``resolve_inpaint_policy`` rejects it.
"""
import sys

import numpy as np
import torch


def _iter_frame_range_bounds(spec, n_frames):
    """Yield inclusive ``(lo, hi)`` pairs from '40-90' / '0-20,150-180' / '30',
    normalized so ``lo <= hi`` and clipped to ``[0, n_frames - 1]``. Chunks
    falling entirely outside the clip are skipped.
    """
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
        hi = min(int(n_frames) - 1, hi)
        if lo > hi:
            continue
        yield lo, hi


def _parse_frame_ranges(spec, n_frames):
    """Parse '40-90' / '0-20,150-180' / '30' into a set of frame indices,
    inclusive and clipped to [0, n_frames - 1]. Empty spec => all frames.
    """
    if not spec:
        return set(range(n_frames))
    frames = set()
    for lo, hi in _iter_frame_range_bounds(spec, n_frames):
        frames.update(range(lo, hi + 1))
    if not frames:
        raise ValueError(
            f"--inpaint_frames '{spec}' selected no valid frames "
            f"(motion has {n_frames} frames, indices 0..{n_frames - 1})"
        )
    return frames


def inpaint_span_role(spec, output_frames):
    """Where a range given in OUTPUT frames sits relative to the clip's ends.

    ``'full'``: every output frame. ``'boundary'``: touches frame 0 or frame
    ``output_frames - 1``, so one end of the clip is regenerated and ``--loop``
    is a closure request the model can act on. ``'interior'``: neither end, so
    both stay clamped to the reference and the loop verdict has nothing to
    decide. An empty spec means every frame.
    """
    output_frames = int(output_frames)
    frames = _parse_frame_ranges(spec, output_frames)
    if len(frames) >= output_frames:
        return 'full'
    if 0 in frames or output_frames - 1 in frames:
        return 'boundary'
    return 'interior'


_FULL_COVERAGE_INPAINT_ERROR = (
    "ERROR: the inpaint mask would free every joint on every window frame, "
    "which is not an inpaint: nothing is left clamped, the reference is "
    "discarded entirely, and the model is handed an all-ones reliability map "
    "(cross_limb_unreliable_mask) far outside the spans it trained on.\n"
    "  To regenerate the whole clip FROM the reference: drop the inpaint flags "
    "and pass --skip_timesteps (higher = more faithful to the reference).\n"
    "  To regenerate the whole clip ignoring the reference: drop "
    "--reference_motion as well.\n"
    "  To regenerate part of it: name fewer frames in --inpaint_frames, or "
    "fewer joints in --inpaint_joints."
)


def check_full_coverage_inpaint_early(frames_arg, joints_arg, num_frames):
    """Reject a whole-mask inpaint before the model is loaded, when it can be
    seen that early: an explicit ``--num_frames`` and no ``--inpaint_joints`` to
    narrow the joint axis. Everything else waits for
    :func:`resolve_inpaint_policy`, which sees the resolved length, the resolved
    joint set and the mapped window frames.
    """
    if joints_arg or not frames_arg:
        return
    if int(num_frames or 0) <= 0:
        return
    if inpaint_span_role(frames_arg, int(num_frames)) == 'full':
        sys.exit(_FULL_COVERAGE_INPAINT_ERROR)


def resolve_inpaint_policy(
    cond_entry,
    *,
    frames_arg,
    joints_arg,
    include_subtree,
    outpaint_range,
    user_inpaint_active,
    output_frames,
    window_frames,
    loop,
):
    """Settle what the requested inpaint means for the run, and return the loop
    condition it is sampled and exported under.

    Both judgements are made on the WINDOW frames the mask will actually free,
    never on the requested output range: the floor/ceil widening of the M -> T
    remap stretches that range, and at M > T it can stretch one naming neither
    clip end into the whole window ('1-118' of M=120 covers all 60 window
    frames of a T=60 checkpoint).

    ``--loop`` is a statement about the WINDOW and it reaches the export as a
    periodic resample of the WHOLE clip, clamped frames included. An inpaint
    that holds both window ends therefore has no use for it: the model only
    regenerates interior frames, and a periodic export would resample the
    clamped head and tail to window times the reference was never placed at
    (|T-M|/M window frames of drift at the tail, and for M > T the last output
    frame blends the reference's OPENING pose). Such a run is exported open,
    which also collapses the mask mapping and the export mapping onto the same
    one: the requested frames are then exactly the frames regenerated.

    The verdict is taken against the open mapping, which is the one a forced-off
    loop then uses; ``--loop`` only ever widens the span, so a range reaching a
    window end under the open mapping reaches it either way and the two agree.

    A mask that frees every joint on every window frame leaves no known region,
    so there is nothing to inpaint against and the run is rejected -- on the
    resolved joint set, so a subtree covering the whole skeleton is caught too.
    That joint set is resolved only once the frame axis is already known to
    cover the window: a span that holds an end cannot free everything whatever
    the joints are, and the caller resolves them again for the mask itself.
    """
    def window_span_role(spec, spec_loop):
        return inpaint_span_role(
            _map_frame_ranges_to_internal(
                spec, output_frames, window_frames, loop=bool(spec_loop),
            ),
            window_frames,
        )

    specs = [
        spec for spec in (
            outpaint_range,
            frames_arg if user_inpaint_active else None,
        ) if spec is not None
    ]
    if specs and loop and all(
        window_span_role(spec, False) == 'interior' for spec in specs
    ):
        print(
            f'  loop: forced off -- the inpaint range keeps both clip ends '
            f'clamped to the reference (window frames 0 and {window_frames - 1}), '
            f'so there is no closure to make and the clamped ends are exported '
            f'unresampled'
        )
        loop = False

    if (
        user_inpaint_active
        and window_span_role(frames_arg, loop) == 'full'
    ):
        freed_joints, real_joint_count = _resolve_inpaint_joint_indices(
            cond_entry, joints_arg, include_subtree,
        )
        if len(freed_joints) >= real_joint_count:
            sys.exit(_FULL_COVERAGE_INPAINT_ERROR)
    return loop


def inpaint_y_anchor_spans(frames_arg, output_frames):
    """Contiguous ``[(a, b)]`` runs of the user's ``--inpaint_frames``, in OUTPUT
    frames -- already the axis ``_reanchor_inpaint_root_y_via_velocity`` walks,
    since it runs on the exported motion. Empty when no range was given.
    """
    if not frames_arg:
        return None
    return _contiguous_frame_runs(_parse_frame_ranges(frames_arg, output_frames))


def _map_frame_ranges_to_internal(spec, source_frames, target_frames, loop=False):
    """Map a frame range given in OUTPUT frames onto the sampler's window.

    Two mappings put content at an output frame, so the freed window span is the
    UNION of both preimages:

    * the reference fills the window end to end
      (``_prepare_img2img_reference_bundle``), output frame ``s`` at window time
      ``s*(T-1)/(M-1)``. Freeing that preimage releases the reference content --
      and the appended pad of an outpaint range -- sitting under the request.
    * the window is exported from window time ``s*T/M`` once ``--loop`` makes it
      periodic (``_resample_window_to_output``). Freeing that preimage is what
      makes the requested EXPORT frames regenerate instead of blending in a
      neighbour that stayed clamped.

    The two coincide when the export is open, so the union only widens a loop
    run. Its wrap partner is deliberately left out: under ``is_loop`` the last
    export frame straddles window ``T-1`` -> window ``0``, and window ``0`` is
    the clamped opening pose the cycle closes onto -- it is freed when the
    request names output frame 0, not because it names the last one.

    One window frame feeds several export frames whenever ``M != T``, so freeing
    a request's preimage necessarily touches its neighbours: a one-frame request
    regenerates one or two output frames around it as well. That spill is
    inherent to the resolution change and only ``M == T`` removes it, which is
    not something a reference run should be asked to arrange, so it is not
    reported -- the seam simply lands a frame or two outside the request.
    """
    if not spec or int(source_frames) == int(target_frames):
        return spec
    source_frames = int(source_frames)
    target_frames = int(target_frames)
    if source_frames <= 0 or target_frames <= 0:
        raise ValueError(
            f"Cannot map frame ranges with source_frames={source_frames}, target_frames={target_frames}"
        )
    scales = [
        float(target_frames - 1) / float(source_frames - 1) if source_frames > 1 else 0.0
    ]
    if loop:
        scales.append(float(target_frames) / float(source_frames))
    mapped_frames = set()
    for lo, hi in _iter_frame_range_bounds(spec, source_frames):
        for scale in scales:
            start = max(0, min(target_frames - 1, int(np.floor(float(lo) * scale))))
            end = max(0, min(target_frames - 1, int(np.ceil(float(hi) * scale))))
            mapped_frames.update(range(start, end + 1))
    if not mapped_frames:
        raise ValueError(
            f"--inpaint_frames '{spec}' selected no valid frames "
            f"(motion has {source_frames} frames, indices 0..{source_frames - 1})"
        )
    return _frame_runs_to_spec(_contiguous_frame_runs(mapped_frames))


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


def _frame_runs_to_spec(runs):
    """Render ``[(start, end), ...]`` back as a '0-19,40' frame spec."""
    return ','.join(
        f'{start}-{end}' if start != end else str(start) for start, end in runs
    )


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
    *,
    output_frames=None,
    loop=False,
):
    """Build the inpainting mask tensor [B, max_joints, 1, n_frames].

    Convention: 1.0 = regenerate (free), 0.0 = keep reference (clamped).
    Padding joints (index >= n_joints) stay 0.0. The regenerated region is
    selected-joints x selected-frames; everything else is held to the
    reference during sampling.

    ``output_frames`` says the frame spec is written in OUTPUT frames and has to
    be mapped onto the window first (``_map_frame_ranges_to_internal``, which
    ``loop`` steers); without it the spec is already in window frames.
    """
    joint_indices, n_joints = _resolve_inpaint_joint_indices(
        cond_entry, inpaint_joints_arg, inpaint_include_subtree
    )
    if output_frames is not None:
        inpaint_frames_arg = _map_frame_ranges_to_internal(
            inpaint_frames_arg, output_frames, n_frames, loop=loop,
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
