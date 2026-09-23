"""Output length of a generated clip: choosing it, and resampling to it.

The model samples a fixed native window (the checkpoint's ``num_frames``); the
requested output length M only sets the ``resample_speed`` condition and the
final temporal resample of the window. These helpers pick M (explicit flag,
reference length, ``--action_label`` training prior, native window), fit a
reference to it as the one-shot clip it is, and apply the window -> output
resample with the periodic/open convention ``--loop`` dictates.
"""
import sys

import numpy as np

from data_loaders.truebones.data.dataset import resample_motion_features
from data_loaders.truebones.truebones_utils.animation_utils import detect_loop_from_features
from data_loaders.truebones.truebones_utils.loop_verdict import stored_loop_verdict
from data_loaders.truebones.truebones_utils.param_utils import MAX_SOURCE_FRAMES_MULT
from utils.clip_length_prior import (
    auto_loop,
    auto_num_frames,
    has_clip_length_prior,
    merge_prior_pool,
)

LOOP_MODES = ('auto', 'on', 'off')


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
    detector's proposal for a clip preprocessed or retargeted on the way in.
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
    cond_dict,
    target_type,
    action_condition,
    *,
    reference_features=None,
    translation_root_index=0,
    fallback_cond_loader=None,
    verbose=True,
):
    """The ``is_loop`` condition a generation runs with, from ``--loop``.

    ``loop_mode`` is ``'on'`` / ``'off'`` / ``'auto'`` (a bool is accepted as
    on/off for programmatic callers).

    ``'auto'`` with a ``--reference_motion`` (``reference_features`` is the
    clip as loaded, on the target skeleton) follows the reference: it is a
    loop when the clip closes (:func:`reference_loop_verdict`). The reference
    sets the window, so the label's habit is beside the point here.

    ``'auto'`` without one follows the training corpus: the majority of the
    clips carrying ``--action_label`` (the same lookup that picks the auto
    ``--num_frames``, so both are read off the same clips) decides. ``is_loop``
    is a conditioning input the model only ever saw paired with a label the
    way that label's clips were authored, so for a label the corpus holds only
    as loops the open-window pairing is off-distribution -- an unflagged
    ``"jump, up"`` is the case in point. Without a label there is nothing to
    key the verdict on, so ``'off'`` -- the behaviour every earlier run had.

    ``target_type`` ``None`` asks for the corpus-wide answer (``--object_type
    all`` is resolved per species by :func:`_all_species_output_lengths`
    instead).
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

    resolved, pool = None, cond_dict
    if action_condition is not None:
        resolved, pool = _ask_prior(
            cond_dict, fallback_cond_loader,
            lambda candidates: auto_loop(
                candidates,
                target_type,
                action_group=action_condition['action_group'],
                action_label=action_condition['action_label'],
            ),
        )
    if resolved is not None:
        is_loop, explanation = resolved
        if verbose:
            print(f'  loop (auto) -> {"on" if is_loop else "off"} ({explanation})')
        return bool(is_loop)
    if verbose:
        print(f'  loop (auto) -> off ({_no_prior_reason(pool, action_condition)})')
    return False


def fit_reference_to_output(reference_features, requested_frames):
    """The reference as the window will be filled from it, for an output of M
    frames.  Returns ``(features, outpaint_range, note)``.

    The reference is a ONE-SHOT clip whatever the loop condition says: it covers
    exactly the R frames it holds, so R against M is a coverage question.  R > M
    crops the tail off; R < M leaves ``[R, M)`` with no reference behind it and
    the caller outpaints that span from noise (``outpaint_range`` names those
    frames).

    ``--loop`` is a statement about the WINDOW, not about the reference.  It
    reaches the model as ``is_loop`` -- its circular phase table, the wrap loss
    it trained under -- and it decides how the sampled window is exported
    (``_resample_window_to_output``).  Closing the cycle is then the model's own
    job, and appended frames are the room it has to do it in.  Nothing here
    reads the reference as a period.
    """
    R = int(reference_features.shape[0])
    M = int(requested_frames)
    if R > M:
        return reference_features[:M], None, (
            f'  Reference cropped: R={R} > M={M} -> using first {M} frames'
        )
    if R < M:
        pad = np.repeat(reference_features[-1:], M - R, axis=0)
        return (
            np.concatenate([reference_features, pad], axis=0),
            f'{R}-{M - 1}',
            f'  Reference outpaint: R={R} < M={M} -> appended frames [{R}, {M - 1}]',
        )
    return reference_features, None, None


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
):
    """``{species: (target_output_frames, resample_speed_cond, is_loop)}`` for
    --object_type all.

    ``explicit`` is the pair an explicit ``--num_frames`` already fixed, and it
    applies to every species unchanged -- a number the user typed is a decision,
    not a hint. Otherwise each species is resolved on its own prior: one shared
    median would time every species but the average one wrongly, and the length
    is free to differ because it is per-sample in the batch anyway. ``--loop``
    is resolved per species the same way (``loop_mode`` is the raw flag), since
    ``is_loop`` is per-sample too.
    """
    lengths = {}
    for species in cond_dict:
        is_loop = resolve_loop_condition(
            loop_mode, cond_dict, species, action_condition,
            fallback_cond_loader=fallback_cond_loader,
            verbose=False,
        )
        if explicit is not None:
            target, speed = explicit
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
        lengths[species] = (target, speed, is_loop)

    closed = sum(1 for _, _, is_loop in lengths.values() if is_loop)
    print(f'  loop ({loop_mode}): {closed} of {len(lengths)} species closed')
    if explicit is None and lengths:
        resolved = sorted(target for target, _, _ in lengths.values())
        # One line instead of one per species; each batch header repeats the
        # number next to the species it belongs to.
        print(
            f'  num_frames (auto): per species, median {resolved[len(resolved) // 2]}, '
            f'range {resolved[0]}-{resolved[-1]} over {len(resolved)} species'
        )
    return lengths


def _resample_window_to_output(motion_np, target_output_frames, output_frame_count, loop):
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
    """
    target_output_frames = int(target_output_frames)
    if target_output_frames == int(output_frame_count):
        return motion_np
    return resample_motion_features(
        motion_np, target_output_frames, periodic=bool(loop),
    )
