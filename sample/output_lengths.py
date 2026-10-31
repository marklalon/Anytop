"""Output length of a generated clip: choosing it, and resampling to it.

The model samples a fixed native window (the checkpoint's ``num_frames``); the
requested output length M only sets the ``resample_speed`` condition and the
final temporal resample of the window. These helpers pick M (explicit flag,
reference length, ``--action_label`` training prior, native window), fit a
reference to it as the one-shot clip it is, and apply the window -> output
resample with the periodic/open convention ``--loop`` dictates.
"""
import contextlib
import io
import math
import sys

import numpy as np

from data_loaders.truebones.data.dataset import (
    invert_resample_motion_features,
    resample_motion_features,
)
from data_loaders.truebones.truebones_utils.animation_utils import detect_loop_from_features
from data_loaders.truebones.truebones_utils.loop_verdict import stored_loop_verdict
from data_loaders.truebones.truebones_utils.motion_labels import (
    head_words_in,
    loop_is_phase_free,
    parse_action_label,
    phase_anchoring_words,
)
from data_loaders.truebones.truebones_utils.param_utils import MAX_SOURCE_FRAMES_MULT
from utils.clip_length_prior import (
    auto_num_frames,
    has_clip_length_prior,
    loop_period_prior,
    merge_prior_pool,
    high_frequency_frame_cap,
)

LOOP_MODES = ('auto', 'on', 'off')


def cap_auto_output_frames(cond_dict, target_type, action_condition, frames, *,
                           internal_num_frames, fallback_cond_loader=None, verbose=True):
    """Cap only an inferred output length; loop policy/period stay untouched."""
    if action_condition is None:
        return int(frames)
    def ask(pool):
        cap = high_frequency_frame_cap(pool, target_type,
            action_group=action_condition['action_group'], action_label=action_condition['action_label'],
            window_frames=internal_num_frames)
        return (cap, 'matching high-frequency training window') if cap is not None else None
    resolved, _ = _ask_prior(cond_dict, fallback_cond_loader, ask)
    if resolved is not None and frames > resolved[0]:
        if verbose:
            print(f'  num_frames (auto) capped: {frames} -> {resolved[0]} ({resolved[1]})')
        return int(resolved[0])
    return int(frames)


def _ask_prior(cond_dict, fallback_cond_loader, ask):
    """Run one prior lookup ``ask(pool)`` on the active cond, then on the
    checkpoint's cond when the active one answers nothing.

    Returns ``(result, pool)``: ``result`` is what ``ask`` returned (``None``
    when neither pool knows the label), ``pool`` is the widest cond consulted,
    so the caller can tell "unbaked" from "no clip matches".

    ``fallback_cond_loader`` is consulted only when the active cond answers
    nothing: a narrow ``--cond_path`` (a new skeleton's own one-species cond) has
    no neighbours to borrow from, and the checkpoint's own snapshot is the pool
    wanted -- the species the weights were actually trained on. It is loaded
    lazily because on the ordinary path it is the same file.
    """
    pool = cond_dict
    resolved = ask(cond_dict)
    if resolved is None and fallback_cond_loader is not None:
        extra = fallback_cond_loader()
        if extra:
            pool = merge_prior_pool(cond_dict, extra)
            resolved = ask(pool)
            if resolved is not None:
                value, explanation = resolved
                resolved = (value, f"{explanation}, from the checkpoint's cond")
    return resolved, pool


def _no_prior_reason(pool, action_condition, *, loop=False):
    """Why a prior lookup came back empty, for the console line."""
    if action_condition is None:
        return 'no --action_label to infer from'
    if not has_clip_length_prior(pool):
        return (
            'no clip-length prior in this cond.npy -- bake one with '
            'tools/regenerate_dataset_artifacts.py, or point --cond_path at '
            'a cond that has one'
        )
    loop_note = ' loop' if loop else ''
    return f'no{loop_note} training clip matches {action_condition["action_label"]!r}'


def reference_loop_verdict(reference_features, translation_root_index):
    """Whether a reference clip closes, as ``(is_loop, explanation)``.

    Every stored clip carries its verdict in its terminal velocity row (see
    ``loop_verdict``): the hand-verified sidecar flag for a dataset clip, the
    detector's proposal for a clip preprocessed on the way in.
    That is read first, since it is the annotation the clip was stored under.
    A tensor with no verdict row -- a generated sample, say -- is judged by the
    same geometric detector preprocessing proposes with (endpoint pose gap
    against the clip's own boundary motion, plus root XZ closure).
    """
    stored = stored_loop_verdict(reference_features, translation_root_index)
    if stored is not None:
        return bool(stored), "the reference's stored loop verdict"
    detected = detect_loop_from_features(
        reference_features, translation_root_index=translation_root_index,
    )
    return bool(detected), (
        "the reference's endpoints "
        + ("close" if detected else "do not close")
        + " (no stored verdict; pose gap + root XZ detector)"
    )


def resolve_loop_condition(
    loop_mode,
    action_condition,
    *,
    reference_features=None,
    translation_root_index=0,
    verbose=True,
):
    """The ``is_loop`` condition a generation runs with, from ``--loop``.

    ``loop_mode`` is ``'on'`` / ``'off'`` / ``'auto'`` (a bool is accepted as
    on/off for programmatic callers).

    ``'auto'`` with a ``--reference_motion`` (``reference_features`` is the
    clip as loaded, on the target skeleton) follows the reference: it is a
    loop when the clip closes (:func:`reference_loop_verdict`). The reference
    sets the window, so the label's habit is beside the point here.

    ``'auto'`` without one follows the ``--action_label``: a loop when the
    label is phase-free (:func:`loop_is_phase_free` -- a gait cycle, a
    wingbeat, an idle, a held posture), the same rule that decides which loop
    clips the loader rolls and tiles. A phase-anchored head (jump, attack,
    roar ...) or event modifier (look, eat, shake ...) is one event in an open
    window. Without a label there is nothing to key the verdict on, so
    ``'off'``.
    """
    if isinstance(loop_mode, bool):
        return loop_mode
    mode = str(loop_mode or 'auto').strip().lower()
    if mode not in LOOP_MODES:
        raise ValueError(f"--loop must be one of {LOOP_MODES}, got {loop_mode!r}")
    if mode != 'auto':
        return mode == 'on'

    if reference_features is not None:
        is_loop, explanation = reference_loop_verdict(reference_features, translation_root_index)
        if verbose:
            print(f'  loop (auto) -> {"on" if is_loop else "off"} ({explanation})')
        return is_loop

    if action_condition is None:
        if verbose:
            print('  loop (auto) -> off (no --action_label to infer from)')
        return False
    label = action_condition['action_label']
    is_loop = loop_is_phase_free(label)
    if verbose:
        if is_loop:
            heads = ', '.join(head_words_in(parse_action_label(label)))
            reason = f'head word(s) {heads}: phase-free'
        else:
            reason = f'phase-anchored by {", ".join(phase_anchoring_words(label)) or "an empty label"}'
        print(f'  loop (auto) -> {"on" if is_loop else "off"} ({reason})')
    return is_loop


def loop_period(cond_dict, target_type, action_condition, *, min_length,
                internal_num_frames, fallback_cond_loader=None):
    """``(period, high_frequency)`` of the label's loop clips for
    ``target_type`` (:func:`utils.clip_length_prior.loop_period_prior`), or
    ``None`` without a label or a matching loop clip."""
    if action_condition is None:
        return None
    resolved, _ = _ask_prior(
        cond_dict, fallback_cond_loader,
        lambda candidates: loop_period_prior(
            candidates,
            target_type,
            action_group=action_condition['action_group'],
            action_label=action_condition['action_label'],
            min_frames=min_length,
            max_frames=MAX_SOURCE_FRAMES_MULT * internal_num_frames,
            window_frames=internal_num_frames,
        ),
    )
    return None if resolved is None else resolved[0]


def _nearest_cycle_count(frames, period, high_frequency):
    """The cycle count ``frames`` frames of a ``period``-frame cycle hold.

    Nearest integer in log space (the speed draw is log-uniform), except for
    a high-frequency cycle, which the loader only slows down, so a cycle
    shorter than ``period`` was never trained: that rounds down. At least 1.
    """
    ratio = float(frames) / float(period)
    if high_frequency:
        return max(1, int(math.floor(ratio + 1e-9)))
    low = max(1, int(math.floor(ratio)))
    return low if ratio <= math.sqrt(low * (low + 1)) else low + 1


def resolve_loop_cycles(requested_cycles, requested_frames, *, loop, action_label,
                        period=None, verbose=True):
    """The loop window's cycle count ``k`` and the output length, from
    ``--loop_cycles`` and ``--num_frames``. Returns ``(k, frames)``: ``k`` is
    an int on a loop window and ``None`` on an open one, ``frames`` the length
    to finalize or ``None`` to keep the caller's own rule.

    ``requested_cycles`` is ``None`` (flag unset) or an int; ``period`` is
    ``(P, high_frequency)`` -- the species' loop
    period for the label (:func:`loop_period`), or a dataset reference's own
    length -- or ``None`` when nothing is known.

    ===========  ==============================  ==============
    given        k                               frames
    ===========  ==============================  ==============
    neither      1                               caller's rule
    k only       k                               k * P
    frames only  nearest to frames / P           frames
    both         k                               frames
    ===========  ==============================  ==============

    Without a period, an unset count is 1 whatever the length.

    A phase-anchored label (not :func:`loop_is_phase_free`) is never tiled in
    training, so it takes k = 1 and refuses a larger one. An open window has
    no k: the flag is refused there.
    """
    if not loop:
        if requested_cycles is not None:
            sys.exit("ERROR: --loop_cycles needs a loop window (--loop on, or --loop auto "
                     "resolving to on).")
        return None, requested_frames
    if action_label and not loop_is_phase_free(action_label):
        if requested_cycles is not None and int(requested_cycles) != 1:
            sys.exit(
                f"ERROR: --loop_cycles {requested_cycles}: {action_label!r} is a phase-anchored "
                "loop, which training never tiles; only --loop_cycles 1 is in distribution."
            )
        if verbose:
            print(f'  loop_cycles -> 1 (phase-anchored label {action_label!r})')
        return 1, requested_frames
    if requested_cycles is not None:
        cycles = int(requested_cycles)
        if requested_frames is None:
            if period is None:
                sys.exit(
                    f"ERROR: --loop_cycles {cycles} without --num_frames needs the cycle "
                    "length of the label's loop clips, and none is known here; pass "
                    "--num_frames too."
                )
            requested_frames = int(round(cycles * period[0]))
            if verbose:
                print(f'  num_frames -> {requested_frames} ({cycles} x period {period[0]})')
        return cycles, requested_frames
    if period is None:
        if verbose:
            print('  loop_cycles -> 1 (no loop period known)')
        return 1, requested_frames
    if requested_frames is None:
        if verbose:
            print(f'  loop_cycles -> 1 (one period, {period[0]} frames)')
        return 1, requested_frames
    cycles = _nearest_cycle_count(requested_frames, period[0], period[1])
    if verbose:
        rule = 'rounded down, high-frequency' if period[1] else 'nearest in log space'
        print(f'  loop_cycles -> {cycles} (num_frames {requested_frames} / period {period[0]}, {rule})')
    return cycles, requested_frames


def reference_loop_cycles(requested_cycles, *, loop, verbose=True):
    """The cycle count of a window generated with a ``--reference_motion``:
    1 on a loop window, ``None`` on an open one.

    A reference is taken to be unseen, so how many cycles it holds is not
    known; the model reads a reference-active loop row as one cycle anyway.
    An explicit ``--loop_cycles`` is refused rather than ignored.
    """
    if requested_cycles is not None:
        sys.exit(
            f"ERROR: --loop_cycles {requested_cycles} cannot be combined with --reference_motion: "
            "the reference's own cycles set the window, and the window is read as one cycle."
        )
    if not loop:
        return None
    if verbose:
        print('  loop_cycles -> 1 (a --reference_motion sets its own cycles)')
    return 1


def warn_loop_cycles_range(cycles, frames, *, period, internal_num_frames, motion_speed_aug):
    """Warn when a loop window's ``(k, L)`` lies outside what training drew.

    The cycle length ``L / k`` must lie in the speed range of the clip,
    ``[P / R, R * P]`` (``[P, R * P]`` for a high-frequency clip, which is only
    slowed down), ``R`` being the checkpoint's ``--motion_speed_aug``; and a
    high-frequency clip's source budget is the window, so a longer loop was
    always cropped open. Returns the warnings, printed as well.
    """
    warnings = []
    if cycles is None or period is None:
        return warnings
    base, high_frequency = int(period[0]), bool(period[1])
    ratio = max(float(motion_speed_aug or 1.0), 1.0)
    low = base if high_frequency else base / ratio
    high = base * ratio
    cycle_length = float(frames) / float(cycles)
    # Half a frame of slack: P is a rounded median, and lengths are integers.
    if not low - 0.5 <= cycle_length <= high + 0.5:
        warnings.append(
            f"cycle length {frames}/{cycles} = {cycle_length:.1f} frames is outside the trained "
            f"[{low:.1f}, {high:.1f}] around period {base}"
            + (" (high-frequency: slowed down only)" if high_frequency else "")
            + f" at motion_speed_aug {ratio:g}"
        )
    if high_frequency and int(frames) > int(internal_num_frames):
        warnings.append(
            f"num_frames {frames} exceeds the {internal_num_frames}-frame source budget of a "
            "high-frequency clip; training crops such a loop open and never trains it as a loop"
        )
    for message in warnings:
        print(f'  WARNING: loop_cycles {cycles}: {message}')
    return warnings


def fit_reference_to_output(reference_features, requested_frames):
    """The ``--outpaint`` fit: the reference as the window will be filled from
    it, for an output of M frames.  Returns ``(features, outpaint_range, note)``.

    The reference is a ONE-SHOT clip whatever the loop condition says: it covers
    exactly the R frames it holds, so R against M is a coverage question. R < M
    leaves ``[R, M)`` with no reference behind it and the caller outpaints that
    span from noise (``outpaint_range`` names those frames). R >= M extends
    nothing and is refused: the default fit (``resample_reference_to_output``)
    is the way to reach a length the reference already covers.

    ``--loop`` is a statement about the WINDOW, not about the reference.  It
    reaches the model as ``is_loop`` -- its circular phase table, the wrap loss
    it trained under -- and it decides how the sampled window is exported
    (``_resample_window_to_output``).  Closing the cycle is then the model's own
    job, and appended frames are the room it has to do it in.  Nothing here
    reads the reference as a period.
    """
    R = int(reference_features.shape[0])
    M = int(requested_frames)
    if R >= M:
        raise ValueError(
            f"--outpaint extends the reference, but it already has R={R} frames for "
            f"--num_frames {M}. Pass --num_frames above {R}, or drop --outpaint to "
            "resample the reference to that length."
        )
    pad = np.repeat(reference_features[-1:], M - R, axis=0)
    return (
        np.concatenate([reference_features, pad], axis=0),
        f'{R}-{M - 1}',
        f'  Reference outpaint: R={R} < M={M} -> appended frames [{R}, {M - 1}]',
    )


def resample_reference_to_output(reference_features, requested_frames):
    """The default fit without ``--outpaint``: the whole reference is resampled
    (endpoint, one-shot) to M frames, so it covers the window end to end at a
    changed tempo and nothing is left for the model to fill.  Same return shape
    as ``fit_reference_to_output``; ``outpaint_range`` is always None.
    """
    R = int(reference_features.shape[0])
    M = int(requested_frames)
    if R == M:
        return reference_features, None, None
    return (
        resample_motion_features(reference_features, M),
        None,
        f'  Reference resampled: R={R} -> M={M} (pass --outpaint to crop/extend instead)',
    )


def _finalize_output_lengths(requested_frames, min_length, internal_num_frames):
    """Validate the requested output frame count M and derive the resample_speed
    conditioning value. Returns ``(requested_output_frames, target_output_frames,
    resample_speed_cond_value)``.
    """
    if requested_frames < min_length or requested_frames > MAX_SOURCE_FRAMES_MULT * internal_num_frames:
        sys.exit(
            f"ERROR: num_frames M={requested_frames} outside "
            f"[min_length={min_length}, "
            f"{MAX_SOURCE_FRAMES_MULT}*num_frames={MAX_SOURCE_FRAMES_MULT * internal_num_frames}]"
        )
    resample_speed = float(requested_frames) / float(internal_num_frames)
    return requested_frames, requested_frames, resample_speed


def _resolve_auto_output_lengths(
    cond_dict,
    target_type,
    action_condition,
    *,
    min_length,
    internal_num_frames,
    default_frames,
    loop,
    fallback_cond_loader=None,
    verbose=True,
):
    """Pick ``--num_frames`` when it was not given, with no reference motion.

    Returns the same triple as :func:`_finalize_output_lengths`. ``target_type``
    is ``None`` for ``--object_type all``, where one length covers every species
    and the prior is therefore pooled corpus-wide. ``loop`` is the RESOLVED
    loop condition (see :func:`resolve_loop_condition`); it restricts the pool
    to loop clips, whose recorded length is one period.

    Without an ``--action_label`` there is nothing to key a length on -- a
    species' clips span idles and attacks and gallops -- so the checkpoint's
    native window stands, which is the behaviour every earlier run had.
    """
    resolved, pool = None, cond_dict
    if action_condition is not None:
        resolved, pool = _ask_prior(
            cond_dict, fallback_cond_loader,
            lambda candidates: auto_num_frames(
                candidates,
                target_type,
                action_group=action_condition['action_group'],
                action_label=action_condition['action_label'],
                loop=loop,
                min_frames=min_length,
                max_frames=MAX_SOURCE_FRAMES_MULT * internal_num_frames,
                window_frames=internal_num_frames,
            ),
        )

    if resolved is not None:
        frames, explanation = resolved
        if verbose:
            print(f'  num_frames (auto) -> {frames} ({explanation})')
    else:
        frames = int(default_frames)
        if action_condition is None:
            reason = 'no --action_label to infer a length from'
        else:
            reason = _no_prior_reason(pool, action_condition, loop=loop)
        if verbose:
            print(f'  num_frames (auto) -> {frames} (checkpoint native window: {reason})')
    return _finalize_output_lengths(frames, min_length, internal_num_frames)


def _all_species_output_lengths(
    cond_dict,
    action_condition,
    *,
    explicit,
    min_length,
    internal_num_frames,
    default_frames,
    loop_mode,
    fallback_cond_loader=None,
    loop_cycles=None,
    motion_speed_aug=1.0,
):
    """``{species: (target_output_frames, resample_speed_cond, is_loop,
    loop_cycles)}`` for --object_type all.

    ``explicit`` is the pair an explicit ``--num_frames`` already fixed, and it
    applies to every species unchanged -- a number the user typed is a decision,
    not a hint. Otherwise each species is resolved on its own prior: one shared
    median would time every species but the average one wrongly, and the length
    is free to differ because it is per-sample in the batch anyway. ``--loop``
    (``loop_mode``, the raw flag) follows the label independently of frequency.
    The cycle count (``loop_cycles``, the raw
    ``--loop_cycles``) is resolved per species against its own loop period
    (:func:`resolve_loop_cycles`); a ``k * P`` past the length range is
    clamped into it.
    """
    is_loop = resolve_loop_condition(loop_mode, action_condition, verbose=False)
    label = action_condition['action_label'] if action_condition is not None else ''
    max_frames = MAX_SOURCE_FRAMES_MULT * internal_num_frames
    lengths = {}
    off_range = []
    for species in cond_dict:
        auto_length_capped = False
        period = loop_period(
            cond_dict, species, action_condition, min_length=min_length,
            internal_num_frames=internal_num_frames, fallback_cond_loader=fallback_cond_loader,
        ) if is_loop else None
        cycles, frames = resolve_loop_cycles(
            loop_cycles, None if explicit is None else explicit[0],
            loop=is_loop, action_label=label, period=period, verbose=False,
        )
        if explicit is not None:
            target, speed = explicit
        elif frames is not None:
            uncapped_frames = frames
            frames = cap_auto_output_frames(cond_dict, species, action_condition, frames,
                internal_num_frames=internal_num_frames, fallback_cond_loader=fallback_cond_loader, verbose=False)
            auto_length_capped = frames != uncapped_frames
            _, target, speed = _finalize_output_lengths(
                int(np.clip(frames, min_length, max_frames)), min_length, internal_num_frames,
            )
        else:
            _, target, speed = _resolve_auto_output_lengths(
                cond_dict,
                species,
                action_condition,
                min_length=min_length,
                internal_num_frames=internal_num_frames,
                default_frames=default_frames,
                loop=is_loop,
                fallback_cond_loader=fallback_cond_loader,
                verbose=False,
            )
        lengths[species] = (target, speed, is_loop, cycles)
        if explicit is None and period is not None and target < period[0]:
            capped_period = cap_auto_output_frames(cond_dict, species, action_condition, period[0],
                internal_num_frames=internal_num_frames, fallback_cond_loader=fallback_cond_loader, verbose=False)
            auto_length_capped |= capped_period < period[0]
        if is_loop and cycles is not None and period is not None and not auto_length_capped:
            with contextlib.redirect_stdout(io.StringIO()):
                if warn_loop_cycles_range(
                    cycles, target, period=period, internal_num_frames=internal_num_frames,
                    motion_speed_aug=motion_speed_aug,
                ):
                    off_range.append(species)

    print(f'  loop ({loop_mode}): {"on" if is_loop else "off"} for every species')
    if is_loop:
        counts = sorted({cycles for *_, cycles in lengths.values()})
        print('  loop_cycles: ' + ', '.join(
            f'{c} x{sum(v[3] == c for v in lengths.values())}' for c in counts
        ))
        if off_range:
            preview = ', '.join(off_range[:6]) + ('...' if len(off_range) > 6 else '')
            print(f'  WARNING: {len(off_range)} species get a cycle length or source budget '
                  f'outside training ({preview})')
    if explicit is None and lengths:
        resolved = sorted(target for target, *_ in lengths.values())
        # One line instead of one per species; each batch header repeats the
        # number next to the species it belongs to.
        print(
            f'  num_frames (auto): per species, median {resolved[len(resolved) // 2]}, '
            f'range {resolved[0]}-{resolved[-1]} over {len(resolved)} species'
        )
    return lengths


def _resample_window_to_output(motion_np, target_output_frames, output_frame_count, loop, *, invert=True):
    """Rescale a sampled window (``output_frame_count`` frames) to the requested
    output length.

    ``--loop`` makes the window periodic -- the model's ``is_loop`` condition,
    its circular phase table and the loader's periodic window resample all say
    so -- so it is rescaled the same way: output frame ``s`` is window time
    ``s*T/M`` and the wrap is interpolated, which leaves the seam an ordinary
    step. Endpoint (open) resampling instead pins the window's ends and leaves
    the exported clip's wrap step at 1 source frame against ``(T-1)/(M-1)``
    inside: a stall at every seam, i.e. a ``--loop`` run that is not a loop.

    The reference goes in the other way as a one-shot clip
    (``_prepare_img2img_reference_bundle``, endpoint resampling), so the two
    directions are not inverse once ``M != T``: a clamped reference pose comes
    out up to ``|M/T - 1|`` output frames from the frame it went in at (1 frame
    at the ``M = MAX_SOURCE_FRAMES_MULT * T`` ceiling, under one below it). An
    inpaint mask is therefore built over the union of both preimages
    (``_map_frame_ranges_to_internal``), and an inpaint holding both clip ends
    forces the export open so the clamped region is not resampled at all.

    An output shorter than the window is the source of a window the loader
    would have upsampled, so it is solved from the window
    (``invert_resample_motion_features``) rather than interpolated out of it:
    interpolation misses every output frame that falls between two window
    frames and flattens content at a period of a few frames. A longer output
    was downsampled into the window, which is not invertible, and is
    interpolated.

    ``invert=False`` keeps the interpolation for every length. A frame-range
    inpaint or outpaint needs it: its mask frees exactly the window frames each
    requested output frame interpolates from and clamps the rest to the
    reference, while every solved output frame reads (with decaying weight)
    the whole window, so clamped reference frames would pick up the
    regenerated ones.
    """
    target_output_frames = int(target_output_frames)
    output_frame_count = int(output_frame_count)
    if target_output_frames == output_frame_count:
        return motion_np
    if invert and 2 <= target_output_frames < output_frame_count:
        return invert_resample_motion_features(
            motion_np, target_output_frames, periodic=bool(loop),
        )
    return resample_motion_features(
        motion_np, target_output_frames, periodic=bool(loop),
    )
