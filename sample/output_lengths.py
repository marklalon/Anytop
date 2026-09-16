"""Output length of a generated clip: choosing it, and resampling to it.

The model samples a fixed native window (the checkpoint's ``num_frames``); the
requested output length M only sets the ``resample_speed`` condition and the
final temporal resample of the window. These helpers pick M (explicit flag,
reference length, ``--action_label`` training prior, native window) and apply
the window -> output resample with the periodic/open convention ``--loop``
dictates.
"""
import sys

from data_loaders.truebones.data.dataset import resample_motion_features
from data_loaders.truebones.truebones_utils.param_utils import MAX_SOURCE_FRAMES_MULT
from utils.clip_length_prior import (
    auto_num_frames,
    has_clip_length_prior,
    merge_prior_pool,
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
    and the prior is therefore pooled corpus-wide.

    ``fallback_cond_loader`` is consulted only when the active cond answers
    nothing: a narrow ``--cond_path`` (a new skeleton's own one-species cond) has
    no neighbours to borrow a length from, and the checkpoint's own snapshot is
    the pool wanted -- the species the weights were actually trained on. It is
    loaded lazily because on the ordinary path it is the same file.

    Without an ``--action_label`` there is nothing to key a length on -- a
    species' clips span idles and attacks and gallops -- so the checkpoint's
    native window stands, which is the behaviour every earlier run had.
    """
    pool = cond_dict
    resolved = None
    if action_condition is not None:
        def ask(candidates):
            return auto_num_frames(
                candidates,
                target_type,
                action_group=action_condition['action_group'],
                action_label=action_condition['action_label'],
                loop=loop,
                min_frames=min_length,
                max_frames=MAX_SOURCE_FRAMES_MULT * internal_num_frames,
            )

        resolved = ask(cond_dict)
        if resolved is None and fallback_cond_loader is not None:
            extra = fallback_cond_loader()
            if extra:
                pool = merge_prior_pool(cond_dict, extra)
                resolved = ask(pool)
                if resolved is not None:
                    frames, explanation = resolved
                    resolved = (frames, f"{explanation}, from the checkpoint's cond")

    if resolved is not None:
        frames, explanation = resolved
        if verbose:
            print(f'  num_frames (auto) -> {frames} ({explanation})')
    else:
        frames = int(default_frames)
        if action_condition is None:
            reason = 'no --action_label to infer a length from'
        elif not has_clip_length_prior(pool):
            reason = (
                'no clip-length prior in this cond.npy -- bake one with '
                'tools/regenerate_dataset_artifacts.py, or point --cond_path at '
                'a cond that has one'
            )
        else:
            loop_note = ' loop' if loop else ''
            reason = (
                f'no{loop_note} training clip matches '
                f'{action_condition["action_label"]!r}'
            )
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
    loop,
    fallback_cond_loader=None,
):
    """``{species: (target_output_frames, resample_speed_cond)}`` for --object_type all.

    ``explicit`` is the pair an explicit ``--num_frames`` already fixed, and it
    applies to every species unchanged -- a number the user typed is a decision,
    not a hint. Otherwise each species is resolved on its own prior: one shared
    median would time every species but the average one wrongly, and the length
    is free to differ because it is per-sample in the batch anyway.
    """
    if explicit is not None:
        return {species: explicit for species in cond_dict}

    lengths = {}
    for species in cond_dict:
        _, target, speed = _resolve_auto_output_lengths(
            cond_dict,
            species,
            action_condition,
            min_length=min_length,
            internal_num_frames=internal_num_frames,
            default_frames=default_frames,
            loop=loop,
            fallback_cond_loader=fallback_cond_loader,
            verbose=False,
        )
        lengths[species] = (target, speed)

    resolved = sorted(target for target, _ in lengths.values())
    if resolved:
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

    The reference bundle (``_prepare_img2img_reference_bundle``) fills the
    window with the same mapping in the other direction, so the round trip is
    the identity and the frames a clamp or ``--inpaint_frames`` names are the
    reference poses they name.
    """
    target_output_frames = int(target_output_frames)
    if target_output_frames == int(output_frame_count):
        return motion_np
    return resample_motion_features(
        motion_np, target_output_frames, periodic=bool(loop),
    )
