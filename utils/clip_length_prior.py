"""Per-species clip-length prior: the frame counts a species was animated at.

``--num_frames`` is not a free parameter. A generation of M frames is told
``resample_speed_cond = M / n`` (the model's native window n), which is exactly
what a training clip of M source frames was told, so asking for a length the
corpus never holds for that kind of motion samples the model off its training
distribution -- the usual symptom being a gait that runs at the wrong rate for
the window it is drawn in.

This module is the one place that answers "how long is this motion, normally?".
It has two halves that share one on-disk contract:

* **Baking** (``build_clip_length_prior``, run by
  ``tools/regenerate_dataset_artifacts.py``) walks the dataset's clips and
  writes a per-species table of clip lengths, split by action group + canonical
  action label and by whether the clip is a loop.
* **Lookup** (``auto_num_frames`` and ``auto_loop``, run by
  ``sample/generate.py``) reads that table back out of ``cond.npy`` and returns
  the median length for the requested label, or whether the corpus animates
  that label as a loop.

The table lives in ``cond.npy`` and nowhere else, because cond.npy is the whole
inference contract: generation reads no dataset directory. (The retired
``loop_period_by_action`` table fed generation a per-species cycle count the
same way.) A cond that predates the bake simply has no table and the caller
keeps its own default -- nothing here is ever fatal.

The length recorded is the clip's *source* length as the loader sees it, i.e.
after a loop's redundant closing key is dropped, so a loop entry is one period.
Everything the loader does downstream of that (speed augmentation, tiling, the
crop to the source-frame budget) spreads the distribution around this value
rather than moving it.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from data_loaders.truebones.truebones_utils.motion_labels import (
    head_words_in,
    vocab_words_in,
)

# cond.npy key holding the table, and the table's own schema version. A reader
# that meets a newer schema than it knows ignores the table instead of guessing
# at its shape.
COND_KEY = "clip_length_prior"
PRIOR_SCHEMA_VERSION = 1

# Bucket names inside one label entry.
LOOP_BUCKET = "loop"
ONESHOT_BUCKET = "oneshot"

# Widening past the target species stops as soon as the pool holds this many
# clips, and never looks at more than this many neighbours. Small on purpose:
# the point is a robust median for a species with no clips of its own, not a
# corpus-wide average that erases every species' own tempo.
_SIMILAR_POOL_TARGET = 5
_SIMILAR_SPECIES_LIMIT = 16


def label_key(action_group, action_label) -> str:
    """The table key for one (group, canonical label) pair.

    The group is part of the key because a single-group checkpoint only ever
    samples its own group's labels, and the same head word could carry a
    different duration in another group. A request with an EMPTY group (an
    ``--action_group all`` checkpoint) matches the label part across every
    group; see :func:`_group_matches`.
    """
    group = str(action_group or "").strip().lower()
    label = str(action_label or "").strip()
    return f"{group}|{label}"


def split_label_key(key) -> tuple[str, str]:
    group, _, label = str(key).partition("|")
    return group, label


# -- Baking ------------------------------------------------------------------
def build_clip_length_prior(records: Iterable[tuple[str, str, bool, int]]) -> dict:
    """Assemble one species' table from ``(group, label, is_loop, length)`` rows.

    Lengths are kept as a sorted tuple rather than reduced to a median here: the
    lookup pools several species (and several labels) before it takes one, and a
    median of medians is not the median of the pool.
    """
    by_label: dict[str, dict[str, list[int]]] = {}
    for action_group, action_label, is_loop, length in records:
        length = int(length)
        if length <= 0:
            continue
        key = label_key(action_group, action_label)
        buckets = by_label.setdefault(key, {LOOP_BUCKET: [], ONESHOT_BUCKET: []})
        buckets[LOOP_BUCKET if is_loop else ONESHOT_BUCKET].append(length)
    return {
        "schema": PRIOR_SCHEMA_VERSION,
        "by_label": {
            key: {name: tuple(sorted(values)) for name, values in buckets.items()}
            for key, buckets in sorted(by_label.items())
        },
    }


def source_clip_length(motion, is_loop: bool, drop_closing_frame) -> int:
    """Length of a stored clip as the loader's source, in frames.

    A loop authored with its last frame repeating frame 0 carries one redundant
    frame that the loader drops before anything else, so its period -- the
    length a ``--loop`` generation should ask for -- is one less than the file.
    ``drop_closing_frame`` is injected so this module stays free of the loader
    (and of torch).
    """
    frames = int(np.asarray(motion).shape[0])
    if not is_loop:
        return frames
    return int(np.asarray(drop_closing_frame(motion)).shape[0])


# -- Lookup ------------------------------------------------------------------
def _has_readable_table(cond_entry) -> bool:
    """True when *cond_entry* carries a table this reader can take apart.

    The bar is the table's SHAPE -- a schema this reader understands and the
    ``by_label`` mapping it needs -- not its contents: a cond baked from a corpus
    whose clips carry no ``action_label`` is still baked, and the caller's advice
    differs, because "no training clip matches this label" is not the same answer
    as "bake a prior". A table missing ``by_label`` is a different case again:
    nothing wrote it, so re-baking is the honest advice.

    :func:`_table_for` cannot make these calls, because it returns ``{}`` for
    every one of them.
    """
    table = cond_entry.get(COND_KEY) if isinstance(cond_entry, Mapping) else None
    if not isinstance(table, Mapping):
        return False
    if int(table.get("schema", 0) or 0) != PRIOR_SCHEMA_VERSION:
        return False
    return isinstance(table.get("by_label"), Mapping)


def _table_for(cond_entry) -> dict:
    """The ``{label_key: buckets}`` map of *cond_entry*, or ``{}`` when unusable.

    ``{}`` covers every state that leaves nothing to look up -- no table, an
    unreadable one, and a table that recorded no clips -- so it answers "is there
    a length here", never "was this cond baked". Use
    :func:`_has_readable_table` for the latter.
    """
    if not _has_readable_table(cond_entry):
        return {}
    return dict(cond_entry[COND_KEY]["by_label"])


def has_clip_length_prior(cond_dict: Mapping[str, Mapping[str, object]]) -> bool:
    """True when at least one species in *cond_dict* carries a readable table.

    "Readable" is about the bake, not about the contents: a table that recorded
    no clips still means the bake ran, so the answer is ``True`` and the caller
    reports "no clip matches this label" rather than sending the user to re-bake
    a cond that was already baked.

    Lets a caller tell "this label is simply not in the corpus" apart from "this
    cond.npy predates the bake", which are the same empty lookup but different
    advice to the user.
    """
    return any(_has_readable_table(entry) for entry in cond_dict.values())


def merge_prior_pool(primary, fallback) -> dict:
    """``primary`` widened with every species ``fallback`` adds.

    ``primary`` is the cond being generated against and stays authoritative for
    every skeleton it defines. The prior is a dataset statistic rather than part
    of that skeleton contract, so a species present in both but carrying no
    table in ``primary`` inherits ``fallback``'s -- which is what a narrow
    ``--cond_path`` cut out of a full cond wants.
    """
    pool = dict(fallback)
    for key, entry in primary.items():
        other = fallback.get(key)
        if other is not None and not _table_for(entry) and _table_for(other):
            pool[key] = {**entry, COND_KEY: other[COND_KEY]}
        else:
            pool[key] = entry
    return pool


def _buckets_for(cond_entry, matches):
    """The ``(loop_lengths, oneshot_lengths)`` of one species under the labels
    ``matches`` accepts, pooled across those labels."""
    loops: list[int] = []
    oneshots: list[int] = []
    for key, buckets in _table_for(cond_entry).items():
        if not matches(key) or not isinstance(buckets, Mapping):
            continue
        loops.extend(int(value) for value in (buckets.get(LOOP_BUCKET) or ()))
        oneshots.extend(int(value) for value in (buckets.get(ONESHOT_BUCKET) or ()))
    return loops, oneshots


def _lengths_for(cond_entry, matches, loop_only: bool) -> list[int]:
    """Every recorded length of one species under the labels ``matches`` accepts."""
    loops, oneshots = _buckets_for(cond_entry, matches)
    return loops if loop_only else loops + oneshots


def _loop_flags_for(cond_entry, matches) -> list[bool]:
    """One ``is_loop`` per recorded clip of one species under ``matches``."""
    loops, oneshots = _buckets_for(cond_entry, matches)
    return [True] * len(loops) + [False] * len(oneshots)


def _group_matches(key_group, wanted_group) -> bool:
    """Does a table key's group satisfy the request's?

    An EMPTY ``wanted_group`` means "any group". That is what an
    ``--action_group all`` checkpoint asks for: it was trained on the whole
    corpus, so it has no group of its own to restrict the pool by. Nothing is
    lost by the wildcard -- the label's first head word determines the group on
    its own across the corpus, so at most one group can hold a given label
    anyway, and the wildcard just saves having to say which.
    """
    return not wanted_group or key_group == wanted_group


def _exact_matcher(action_group, action_label):
    group = str(action_group or "").strip().lower()
    wanted = label_key("", action_label).lstrip("|")

    def matches(key):
        key_group, key_label = split_label_key(key)
        return key_label == wanted and _group_matches(key_group, group)

    return matches


def _head_word_matcher(action_group, action_label):
    """Same group (or any, when none is asked for) and same action head words,
    whatever the modifiers.

    'walk, forward, fast' falls back to every 'walk' clip of the group: the head
    word is what sets the duration, a direction or a hands token does not.
    """
    group = str(action_group or "").strip().lower()
    wanted = tuple(head_words_in(vocab_words_in(str(action_label or ""))))
    if not wanted:
        return None

    def matches(key):
        key_group, key_label = split_label_key(key)
        candidate = tuple(head_words_in(vocab_words_in(key_label)))
        return candidate == wanted and _group_matches(key_group, group)

    return matches


def _clamp(value, low, high) -> int:
    return int(min(max(int(value), int(low)), int(high)))


def _pooled_median(lengths: Sequence[int], min_frames, max_frames) -> int:
    """Median of *lengths*, each clamped into the generation's legal range first.

    Clamping per clip rather than on the median mirrors the loader: a clip
    longer than the source-frame budget is CROPPED to the budget and resampled
    from there, so what it contributed to training was the budget, not its own
    length. Taking the median first would let one 400-frame idle drag a
    two-clip species' answer to the ceiling.
    """
    clamped = [_clamp(value, min_frames, max_frames) for value in lengths]
    return int(round(float(np.median(np.asarray(clamped, dtype=np.float64)))))


def _ranked_neighbours(cond_dict, target_type, candidates):
    """``candidates`` ordered most-similar-first to ``target_type``.

    Falls back to the given order when similarity cannot be scored (a cond entry
    without the geometry ``rank_species`` needs): a worse order is still a usable
    pool, and an auto default is not worth failing a generation over.
    """
    if not candidates or target_type is None:
        return list(candidates)
    try:
        from utils.skeleton_similarity import rank_species

        ranked = rank_species(
            cond_dict[target_type],
            {name: cond_dict[name] for name in candidates},
            query_hint=target_type,
        )
        return [entry.name for entry in ranked]
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[clip_length_prior] similarity ranking unavailable ({exc}); pooling in cond order")
        return list(candidates)


def _matchers_for(action_group, action_label):
    """The label matchers in the order the ladder tries them, or ``[]`` for an
    empty label (nothing to look up)."""
    label = str(action_label or "").strip()
    if not label:
        return []
    matchers = [("exact label", _exact_matcher(action_group, label))]
    head_words = _head_word_matcher(action_group, label)
    if head_words is not None:
        words = ", ".join(head_words_in(vocab_words_in(label)))
        matchers.append((f"head word(s) '{words}'", head_words))
    return matchers


def _first_pool(cond_dict, target_type, matchers, collect):
    """The first non-empty pool the ladder finds, as ``(values, explanation)``.

    ``collect(cond_entry, matches)`` lists one species' values under one
    matcher. The ladder, first non-empty pool wins:

    1. the target species' own clips with exactly this label,
    2. its clips with the same action head words,
    3. the most similar species' clips with exactly this label,
    4. the most similar species' clips with the same head words.

    The target species comes first because its own clip IS what the model
    fitted for it; neighbours are borrowed only for a species that was never
    animated doing this, and then nearest-first, so a quadruped's walk is not
    timed by a bird's.

    ``target_type`` ``None`` asks for the corpus-wide answer (``--object_type
    all``, which generates one answer for every species at once).
    """
    # Every label matcher is tried on the target species BEFORE any neighbour is
    # consulted. Duration is set more by the body than by the modifier: a
    # Horse's own "walk, left" cycle times its "walk, forward" far better than a
    # Pigeon's exact "walk, forward" does.
    if target_type is not None:
        for matcher_name, matches in matchers:
            own = collect(cond_dict.get(target_type, {}), matches)
            if own:
                return own, f"{len(own)} {target_type} clip(s) matching {matcher_name}"

    for matcher_name, matches in matchers:
        candidates = [
            name
            for name in cond_dict
            if name != target_type and collect(cond_dict[name], matches)
        ]
        if not candidates:
            continue
        pooled = []
        used: list[str] = []
        for name in _ranked_neighbours(cond_dict, target_type, candidates):
            pooled.extend(collect(cond_dict[name], matches))
            used.append(name)
            # Stop early only while walking a similarity ranking, where the next
            # species is always further away than the last. With no target to
            # rank against (--object_type all) the order is arbitrary, so
            # stopping would answer from whichever species sorted first rather
            # than from the corpus.
            if target_type is None:
                continue
            if len(pooled) >= _SIMILAR_POOL_TARGET or len(used) >= _SIMILAR_SPECIES_LIMIT:
                break
        if pooled:
            scope = "all species" if target_type is None else f"{len(used)} similar species"
            preview = ", ".join(used[:4]) + ("..." if len(used) > 4 else "")
            return (
                pooled,
                f"{len(pooled)} clip(s) from {scope} ({preview}) matching {matcher_name}",
            )
    return None


def auto_num_frames(
    cond_dict: Mapping[str, Mapping[str, object]],
    target_type: Optional[str],
    *,
    action_group: str,
    action_label: str,
    loop: bool,
    min_frames: int,
    max_frames: int,
) -> Optional[tuple[int, str]]:
    """Median training clip length for ``action_label``, or ``None``.

    Returns ``(frames, explanation)`` with ``frames`` clamped into
    ``[min_frames, max_frames]``; ``None`` means the corpus says nothing about
    this label and the caller should keep its own default. The pool is the
    first rung of :func:`_first_pool`'s ladder that holds a clip.

    ``loop`` restricts every tier to loop clips, whose recorded length is one
    period: a one-shot clip's length is not a period and would hand ``--loop`` a
    window several cycles long (or a fraction of one).
    """
    matchers = _matchers_for(action_group, action_label)
    if not matchers:
        return None
    loop_only = bool(loop)
    found = _first_pool(
        cond_dict, target_type, matchers,
        lambda entry, matches: _lengths_for(entry, matches, loop_only),
    )
    if found is None:
        return None
    lengths, explanation = found
    return _pooled_median(lengths, min_frames, max_frames), f"median of {explanation}"


def auto_loop(
    cond_dict: Mapping[str, Mapping[str, object]],
    target_type: Optional[str],
    *,
    action_group: str,
    action_label: str,
) -> Optional[tuple[bool, str]]:
    """Whether the corpus animates ``action_label`` as a loop, or ``None``.

    Returns ``(is_loop, explanation)``; ``None`` means no training clip carries
    this label (or a label sharing its head words) and the caller keeps its own
    default. Same ladder and same pool as :func:`auto_num_frames`, so the loop
    verdict and the length it then gates are read off the same clips.

    The verdict is the pool's majority. ``is_loop`` is a conditioning input the
    model only ever saw paired with the label the way the corpus authored it --
    every loop clip is trained as a loop, every one-shot as open -- so a label
    the corpus holds only as loops (``"jump, up"``, say) has nothing behind the
    open-window pairing and samples off-distribution there. A tie keeps the
    open window, the convention every earlier run defaulted to.
    """
    matchers = _matchers_for(action_group, action_label)
    if not matchers:
        return None
    found = _first_pool(cond_dict, target_type, matchers, _loop_flags_for)
    if found is None:
        return None
    flags, explanation = found
    loops = sum(1 for flag in flags if flag)
    is_loop = loops * 2 > len(flags)
    return is_loop, f"{loops} of {explanation} are loops"
