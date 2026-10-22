"""Output length of a generated clip: reading it, and resampling to it.

The model samples a fixed native window (the checkpoint's ``num_frames``, T).
The motion it draws spans ``L`` source frames at the dataset fps; ``L`` is the
output length and is never an input. Without a reference the tempo head reads
it off the finished window (:func:`predict_window_source_frames`); with one it
is the reference's own length. The window's velocity rows are displacement per
window frame and become physical per-frame velocity here
(:func:`window_to_physical`), before the window -> output resample with the
periodic/open convention ``--loop`` dictates.
"""
import numpy as np
import torch

from data_loaders.truebones.data.dataset import (
    invert_resample_motion_features,
    resample_motion_features,
    resample_step_scale,
    velocity_to_source_units,
)
from data_loaders.truebones.truebones_utils.animation_utils import detect_loop_from_features
from data_loaders.truebones.truebones_utils.canonical_features import rebuild_derived_velocity
from data_loaders.truebones.truebones_utils.loop_verdict import stored_loop_verdict
from data_loaders.truebones.truebones_utils.motion_labels import (
    head_words_in,
    loop_is_phase_free,
    parse_action_label,
    phase_anchoring_words,
)
from data_loaders.truebones.truebones_utils.param_utils import MAX_SOURCE_FRAMES_MULT
from model.tempo_head import expected_frames
from utils.model_util import unwrap_anytop_model

LOOP_MODES = ('auto', 'on', 'off')


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


def outpaint_reference(reference_features, appended_frames):
    """The ``--outpaint N`` fit: the reference with ``N`` frames appended, as
    the window will be filled from it.  Returns ``(features, outpaint_range,
    note)``; the appended frames repeat the last reference frame and
    ``outpaint_range`` names them, for the caller to generate from noise.

    The reference is a ONE-SHOT clip whatever the loop condition says, and
    ``--loop`` is a statement about the WINDOW: it reaches the model as
    ``is_loop`` and decides how the sampled window is exported
    (``_resample_window_to_output``). Closing the cycle is the model's job, and
    the appended frames are the room it has to do it in.
    """
    R = int(reference_features.shape[0])
    appended = int(appended_frames)
    if appended <= 0:
        return reference_features, None, None
    pad = np.repeat(reference_features[-1:], appended, axis=0)
    M = R + appended
    return (
        np.concatenate([reference_features, pad], axis=0),
        f'{R}-{M - 1}',
        f'  Reference outpaint: R={R} + {appended} -> appended frames [{R}, {M - 1}]',
    )


def resample_reference_to_output(reference_features, requested_frames):
    """The whole reference resampled (endpoint, one-shot) to M frames, so it
    covers the window end to end at a changed tempo and nothing is left for the
    model to fill.  Same return shape as :func:`outpaint_reference`;
    ``outpaint_range`` is always None.
    """
    R = int(reference_features.shape[0])
    M = int(requested_frames)
    if R == M:
        return reference_features, None, None
    return (
        resample_motion_features(reference_features, M),
        None,
        f'  Reference resampled: R={R} -> M={M} (tempo change)',
    )


def source_frame_budget(window_frames):
    """Most source frames the loader ever puts in one window."""
    return int(MAX_SOURCE_FRAMES_MULT) * int(window_frames)


def clamp_reference_frames(frames, min_length, window_frames):
    """A reference's frame count clamped into the range a window holds:
    ``[min_length, MAX_SOURCE_FRAMES_MULT * window_frames]``."""
    return int(np.clip(int(frames), int(min_length), source_frame_budget(window_frames)))


def predict_window_source_frames(model, sample, model_kwargs, *, autocast_dtype=None):
    """``[B]`` int source frame count of each sampled window.

    One extra conditional forward at timestep 0 on the finished sample, read
    off the tempo head as the expected frame count over its bins. Rows share
    nothing, so a batch can come back with a different length per sample.
    """
    core = unwrap_anytop_model(model)
    if getattr(core, 'tempo_head', None) is None:
        raise ValueError('this checkpoint has no tempo head to read the output length from')
    device = sample.device
    timesteps = torch.zeros(sample.shape[0], dtype=torch.long, device=device)
    if autocast_dtype is None:
        autocast = torch.autocast(device_type=device.type, enabled=False)
    else:
        autocast = torch.autocast(device_type=device.type, dtype=autocast_dtype)
    with torch.no_grad(), autocast:
        _, logits = core(sample, timesteps, return_tempo_logits=True, **model_kwargs)
    frames = expected_frames(logits, core.tempo_head.bin_frames)
    return np.rint(frames.cpu().numpy()).astype(np.int64)


def window_to_physical(motion_np, source_frames, loop):
    """A decoded ``[T, J, F]`` window with physical velocity rows.

    The window's root XZ velocity is displacement per WINDOW frame (the
    loader's ``velocity_to_window_units``); ``source_frames`` L sets the step
    ``resample_step_scale(L, T)`` that turns it back into displacement per
    source frame, and the derived rows are rebuilt at that step, so the export
    resample and every reader of the saved rows see physical velocity.
    """
    n_frames = motion_np.shape[0]
    step = resample_step_scale(int(source_frames), n_frames, periodic=bool(loop))
    physical = velocity_to_source_units(motion_np, step)
    return rebuild_derived_velocity(physical, is_loop=bool(loop), step_scale=step)


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
