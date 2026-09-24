"""Everything that goes into ``model_kwargs['y']`` besides the reference.

* ``create_condition`` -- the per-species structural condition (skeleton,
  rest pose, joint-name embeddings, canonical frame stats), collated exactly
  as the training loader collates it.
* ``_resolve_action_condition`` / ``_wrap_action_label_cfg`` -- the
  ``--action_label`` prompt as word-slot ids into the checkpoint's own frozen
  vocabulary, and the optional classifier-free guidance around it.
* ``_resolve_species_emb_override`` -- ``--species_tags``, a re-encoded (or
  cache-hit) species descriptor replacing the cond's baked ``species_emb``.
"""
import sys

import numpy as np
import torch

from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.canonical_features import (
    CANONICAL_FEATURE_SPACE,
    build_canonical_rest_feature,
    get_canonical_global_stats,
    mark_canonical_cond_entry,
)
from data_loaders.truebones.truebones_utils.dataset_tags import (
    check_species_tags,
    parse_species_tags,
)
from data_loaders.truebones.truebones_utils.joint_struct_features import (
    build_joint_struct_features,
)
from model.cfg_sampler import ClassifierFreeActionModel
from sample.generation_runtime import _load_default_cond_cache
from utils.model_util import unwrap_anytop_model


def _parse_species_tags(raw):
    """``--species_tags`` -> tags in sidecar form, or ``[]`` when not given.

    Uses the sidecar's own parser, so a hand-typed ``quadruped; chibi striding``
    reaches T5 as the same text a registered species was encoded from, and a
    descriptor with the wrong slot count fails here instead of being encoded.
    """
    tags = parse_species_tags(str(raw or ''))
    if tags:
        check_species_tags(tags, '--species_tags')
    return list(tags)


def _resolve_species_t5_name(cond_entry):
    """Return the T5 model name used to bake this species' descriptor (species_emb_meta >
    joints_names_embs_meta > t5-base)."""
    for meta_key in ('species_emb_meta', 'joints_names_embs_meta'):
        meta = cond_entry.get(meta_key)
        if isinstance(meta, dict) and meta.get('t5_name'):
            return str(meta['t5_name'])
    return 't5-base'


# Process-local T5 cache keyed by t5 model name for --species_tags re-encoding.
_SPECIES_T5_CACHE = {}


def _get_species_t5_conditioner(t5_name, *, preloaded=None):
    """Return a T5 conditioner, preferring preloaded (if name matches) > cached > fresh."""
    if preloaded is not None and str(getattr(preloaded, 'name', '')) == str(t5_name):
        return preloaded
    cached = _SPECIES_T5_CACHE.get(t5_name)
    if cached is not None:
        return cached
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[generate] Loading T5 '{t5_name}' on {device.upper()} for species-tag re-encoding ...")
    from model.conditioners import T5Conditioner

    conditioner = T5Conditioner(
        name=t5_name,
        finetune=False,
        word_dropout=0.0,
        normalize_text=False,
        device=device,
        autocast_dtype=None,
        local_files_only=True,
    )
    _SPECIES_T5_CACHE[t5_name] = conditioner
    return conditioner


def _encode_species_tags_override(tags, cond_entry, expected_dim, t5_conditioner=None):
    """Re-encode species tags via T5 into a [expected_dim] species_emb override."""
    species_text = ' '.join(tags)
    t5_name = _resolve_species_t5_name(cond_entry)
    print(f"[generate] Re-encoding species tags {tags} via T5 '{t5_name}' ...")
    conditioner = _get_species_t5_conditioner(t5_name, preloaded=t5_conditioner)
    with torch.no_grad():
        tokens = conditioner.tokenize_entries([species_text])
        emb = conditioner(tokens).detach().cpu().numpy().astype(np.float32, copy=False)[0]
    if emb.shape[-1] != int(expected_dim):
        raise ValueError(
            f"--species_tags re-encoding produced dim {emb.shape[-1]} but the model expects "
            f"{expected_dim} (t5_out_dim). The T5 model '{t5_name}' does not match the one used "
            "to build cond.npy."
        )
    return emb


def _find_cached_species_emb(tags, cond_dict, t5_name, expected_dim):
    """Reuse baked species_emb when requested tags match an existing cond entry (same T5 + dim).
    T5 mean-pooling is deterministic given tokenized text, so this is exact."""
    target_text = ' '.join(tags)
    for entry in cond_dict.values():
        if not isinstance(entry, dict):
            continue
        emb = entry.get('species_emb')
        meta = entry.get('species_emb_meta')
        if emb is None or not isinstance(meta, dict):
            continue
        if str(meta.get('t5_name') or '') != str(t5_name):
            continue
        # Normalize whitespace for comparison.
        if ' '.join(str(meta.get('embedding_text', '')).split()) != target_text:
            continue
        emb = np.asarray(emb, dtype=np.float32)
        if emb.shape[-1] == int(expected_dim):
            return emb, str(entry.get('object_type') or '?')
    return None


def _resolve_species_emb_override(
    args,
    model,
    cond_dict,
    object_type,
    *,
    default_cond_file,
    actual_cond_file,
    t5_conditioner=None,
):
    """``--species_tags`` -> a ``[t5_out_dim]`` species descriptor, or ``None``.

    Restyles the target species' motion descriptor. A baked ``species_emb``
    whose tags match (same T5, same dim) is reused as-is -- first from the
    active cond, then from ``default_cond_file`` when the active
    ``--cond_path`` is a small custom cond -- and only otherwise are the tags
    re-encoded through T5 (``t5_conditioner`` is the runtime's preloaded one).
    """
    species_tags = _parse_species_tags(getattr(args, 'species_tags', ''))
    if not species_tags:
        return None

    # Fast-fail if checkpoint ignores species descriptor.
    unwrapped = unwrap_anytop_model(model)
    if not (getattr(unwrapped, 'species_cond', False) or getattr(unwrapped, 'species_joint_cond', False)):
        sys.exit(
            'ERROR: --species_tags was passed but this checkpoint was trained without '
            '--species_cond or --species_joint_cond; the species descriptor is unused, so '
            'the tags would have no effect.'
        )
    target_cond_entry = cond_dict[object_type]
    if 'species_emb' not in target_cond_entry:
        sys.exit(
            f"ERROR: cond entry for '{object_type}' has no baked 'species_emb'; regenerate "
            "cond.npy with species embeddings before using --species_tags."
        )
    override_t5_name = _resolve_species_t5_name(target_cond_entry)
    override_dim = int(np.asarray(target_cond_entry['species_emb']).shape[-1])
    # Reuse baked species_emb if tags match an existing cond entry (same T5 + dim).
    cached = _find_cached_species_emb(
        species_tags, cond_dict, override_t5_name, override_dim,
    )
    if cached is None:
        # The active --cond_path may be a small custom cond lacking the
        # species whose baked tags match; fall back to the default cond DB.
        default_cond_cache = _load_default_cond_cache(default_cond_file, actual_cond_file)
        if default_cond_cache:
            cached = _find_cached_species_emb(
                species_tags, default_cond_cache, override_t5_name, override_dim,
            )
    if cached is not None:
        species_emb_override, cache_src = cached
        print(
            f"[generate] species tags {species_tags} match baked descriptor of "
            f"'{cache_src}'; reusing cached species_emb (skipped T5)."
        )
    else:
        species_emb_override = _encode_species_tags_override(
            species_tags,
            target_cond_entry,
            override_dim,
            t5_conditioner=t5_conditioner,
        )
    default_text = ' '.join(
        (target_cond_entry.get('species_emb_meta') or {}).get('embedding_text', '').split()
    )
    print(
        f"[generate] species descriptor for '{object_type}' overridden: "
        f"'{default_text}' -> '{' '.join(species_tags)}'"
    )
    return species_emb_override


def _resolve_action_condition(args, model):
    """Resolve the checkpoint's action group + ``--action_label`` into one condition.

    Returns ``None`` when no label was requested (the model then falls back to its
    learned unconditional embedding), otherwise a dict carrying the group, the
    label text and the word-level ids the model assembles its slot channels from.
    No T5 runs here and no sidecar is read: the frozen word vectors live in the
    checkpoint, so a generated clip is conditioned on exactly the table the
    weights were trained against.

    ``args.action_group`` is the group this checkpoint was trained on, read out of
    its args.json by parser_util.apply_checkpoint_action_group. There is no
    ``--action_group`` flag at generation: a single-group checkpoint IS its
    group, and a foreign one would describe a different checkpoint. It is empty
    for a checkpoint trained with ``--action_group all`` (and for one predating
    the mandatory training flag), which is not an error: the label carries the
    group already, and the value only ever reaches the clip-length prior, whose
    matchers read an empty group as "any group".
    """
    label = str(getattr(args, 'action_label', '') or '').strip()
    group = str(getattr(args, 'action_group', '') or '').strip().lower()
    if not label:
        return None

    from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (
        action_label_slots,
    )
    from data_loaders.truebones.truebones_utils.motion_labels import (
        ActionLabelError,
        canonical_action_label,
        parse_action_label,
    )

    unwrapped = unwrap_anytop_model(model)
    if not getattr(unwrapped, 'action_label_cond', False):
        sys.exit(
            'ERROR: --action_label was passed but this checkpoint was trained '
            'without --action_label_cond. The label would have no effect.'
        )
    # No group-validity check here: apply_checkpoint_action_group already
    # normalizes anything but ''/a legal group to '' at load time, so ``group``
    # is either one of ACTION_GROUPS or '' (an --action_group all checkpoint,
    # meaning "any group").
    #
    # Labels are exact controlled tokens. An unrecognized one is a HARD ERROR,
    # not a pass-through: there is no synonym translation any more, and letting
    # free text through would sample from wherever T5 puts an out-of-distribution
    # sentence while looking like it worked.
    #
    # A recognizable prompt is still REWRITTEN to its canonical spelling -- it is
    # the string the model fitted, and the one recorded next to the sample -- but
    # the rewrite may only reorder NON-HEAD words: directions bind next to their
    # head, then the remaining modifiers follow. Head-word order is kept as
    # given: the first head word outweighs any later one in the head slot,
    # so reordering them would change the condition; the corpus spells one
    # word set one way, and the prompt's order is the caller's call.
    try:
        tokens = parse_action_label(label)
    except ActionLabelError as exc:
        sys.exit(f"ERROR: --action_label {exc}")
    canonical = canonical_action_label(tokens)
    if canonical != label:
        print(
            f"[generate] --action_label '{label}' -> '{canonical}' "
            f"(canonical spelling: head words in the order given, directions "
            f"next to their head, then other modifiers in vocabulary order; "
            f"this is the string the model trained on)"
        )
        label = canonical
        # Re-parse so the word order handed to the model is the canonical one.
        # Only non-head words moved and every slot pools as a set within
        # itself, so this changes no condition; it keeps generation emitting
        # exactly what the loader emits for the same label.
        tokens = parse_action_label(label)

    # The same contract function the loader calls, so the sampler cannot hand
    # the model a slot assignment training never produced.
    slots = action_label_slots(tokens)
    return {
        'action_group': group,
        'action_label': label,
        'action_slots': {
            'word_ids': np.asarray(slots['word_ids'], dtype=np.int64),
            'slot_ids': np.asarray(slots['slot_ids'], dtype=np.int64),
            'word_mask': np.asarray(slots['word_mask'], dtype=np.bool_),
        },
    }


def _wrap_action_label_cfg(model, args, action_condition):
    """Wrap the denoiser in action-label CFG, or hand it back untouched.

    ``--action_label_cfg_scale 1.0`` (the default) returns ``model`` itself, so the
    common path keeps exactly one model forward per diffusion step. Any other scale
    needs both halves of the CFG contract to exist -- a prompt to guide toward, and
    a checkpoint that trained the null condition -- so both are checked here rather
    than silently costing 2x for a guidance term that is identically zero.
    """
    raw_scale = getattr(args, 'action_label_cfg_scale', None)
    # NOT `or 1.0`: 0.0 is a meaningful scale (pure unconditional sample).
    scale = 1.0 if raw_scale is None else float(raw_scale)
    if scale == 1.0:
        return model
    if scale < 0.0:
        sys.exit(
            f"ERROR: --action_label_cfg_scale must be >= 0, got {scale}. A negative "
            "scale extrapolates AWAY from the prompt."
        )
    if action_condition is None:
        sys.exit(
            "ERROR: --action_label_cfg_scale needs --action_label. With no prompt both "
            "CFG passes are the same unconditional forward, so the guidance term is "
            "exactly zero and sampling would only cost twice as much."
        )
    # Restored from the checkpoint's args.json by parse_and_load_from_model (it is
    # a 'model'-group arg), so this reads how the weights were actually trained.
    drop_prob = float(getattr(args, 'action_label_cfg_drop_prob', 0.0) or 0.0)
    if drop_prob <= 0.0:
        sys.exit(
            "ERROR: --action_label_cfg_scale needs an unconditional mode to guide away "
            "from, but this checkpoint was trained with --action_label_cfg_drop_prob 0, "
            "so it never saw the null condition. Sample with --action_label_cfg_scale 1.0, "
            "or retrain with a non-zero drop probability."
        )
    print(
        f"[generate] action-label CFG: scale={scale:g} on "
        f"{action_condition['action_label']!r} (2 model forwards per diffusion step)"
    )
    return ClassifierFreeActionModel(model, scale)


def _coerce_loop_flag(flag):
    """One ``is_loop`` bool from a batch-level or per-species ``loop`` value.

    ``create_condition`` takes the RESOLVED condition, so the ``--loop`` mode
    strings are refused rather than read as truthiness -- ``bool('off')`` is
    ``True``, which would silently turn an explicit open window into a loop.
    ``'auto'`` is refused too: it is a request to resolve the mode against the
    reference/corpus first (``sample.output_lengths.resolve_loop_condition``).
    """
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        mode = flag.strip().lower()
        if mode in ('on', 'true', '1', 'yes'):
            return True
        if mode in ('off', 'false', '0', 'no', ''):
            return False
        if mode == 'auto':
            raise ValueError(
                "create_condition: loop='auto' is unresolved -- resolve it with "
                "sample.output_lengths.resolve_loop_condition and pass the bool."
            )
        raise ValueError(f"create_condition: loop={flag!r} is not 'on'/'off' or a bool")
    raise TypeError(f"create_condition: loop must be a bool or one of 'on'/'off', got {type(flag).__name__}")


def create_condition(object_types, cond_dict, n_frames, max_joints, feature_len, loop=False, action_condition=None, species_emb_override=None):
    """Build model_kwargs for a batch of object_types.

    action_condition: {'action_group', 'action_label', 'action_slots'} applied
        to every object in the batch, or None for unconditional generation.
    species_emb_override: [t5_out_dim] vector replacing baked species_emb for all objects.
    loop: ask for a closed window -- one bool for the whole batch, or one per
        object_type (--object_type all resolves --loop auto per species). It is
        the whole loop condition -- how many gait cycles the window holds is the
        model's to decide from resample_speed and the species/action prior, so
        nothing here needs a period table.
    """
    if isinstance(loop, (list, tuple)):
        if len(loop) != len(object_types):
            raise ValueError(
                f"create_condition: {len(loop)} loop flags for {len(object_types)} object_types"
            )
        loop_flags = [_coerce_loop_flag(flag) for flag in loop]
    else:
        loop_flags = [_coerce_loop_flag(loop)] * len(object_types)
    batches = list()
    # One entry per species, not per sample: the structural descriptors are a pure
    # function of the cond entry, and this is the same builder the dataset caches
    # at construction -- a second implementation here would silently condition
    # generation on something training never saw.
    joint_struct_by_object = {}
    for object_type, is_loop in zip(object_types, loop_flags):
        if object_type not in cond_dict:
            available = ', '.join(sorted(cond_dict.keys()))
            raise KeyError(
                f"Unknown object_type '{object_type}'. Available object types in cond file: {available}"
            )
        batch = list()
        mark_canonical_cond_entry(cond_dict[object_type])
        parents = cond_dict[object_type]['parents']
        n_joints = len(parents)
        rest_pose = np.nan_to_num(build_canonical_rest_feature(cond_dict[object_type]))
        joint_relations = cond_dict[object_type]['joint_relations']
        joints_graph_dist = cond_dict[object_type]['joints_graph_dist']
        offsets = cond_dict[object_type]['offsets']
        joints_names_embs = cond_dict[object_type]['joints_names_embs']
        batch.append(np.zeros((n_frames, n_joints, feature_len)))
        batch.append(n_frames)
        batch.append(parents)
        batch.append(rest_pose)
        batch.append(offsets)
        batch.append(joints_graph_dist)
        batch.append(joint_relations)
        batch.append(object_type)
        batch.append(joints_names_embs)
        batch.append(max_joints)
        metadata = {
            'is_loop': is_loop,
            'translation_root_index': cond_dict[object_type].get('translation_root_index', 0),
        }
        if 'species_emb' in cond_dict[object_type]:
            metadata['species_emb'] = cond_dict[object_type]['species_emb']
        if species_emb_override is not None:
            metadata['species_emb'] = species_emb_override
        if action_condition is not None:
            metadata['action_group'] = action_condition['action_group']
            metadata['action_label'] = action_condition['action_label']
            metadata['action_slots'] = action_condition['action_slots']
        batch.append(metadata)
        batch.append(object_type)
        # Never None here: build_canonical_rest_feature above standardizes with
        # these same stats and raises a named KeyError when they are absent.
        canonical_mean, canonical_std = get_canonical_global_stats(cond_dict[object_type])
        batch.append({
            # The output coordinate frame is an UNCONDITIONAL model input (see
            # AnyTop.canonical_frame_projection), so generation must carry it just
            # like training does -- the collate stacks it per sample. Decoding the
            # sample back to physical still uses the full cond entry, not y.
            'canonical_feature_mean': canonical_mean,
            'canonical_feature_std': canonical_std,
            'rest_pose_physical': cond_dict[object_type]['rest_pose'],
            'rest_pos_ric_hml': cond_dict[object_type]['rest_pos_ric_hml'],
            'joint_struct': joint_struct_by_object.setdefault(
                object_type,
                build_joint_struct_features(cond_dict[object_type], source=str(object_type)),
            ),
            'feature_space': cond_dict[object_type].get('feature_space', CANONICAL_FEATURE_SPACE),
        })
        batches.append(batch)

    return truebones_batch_collate(batches)
