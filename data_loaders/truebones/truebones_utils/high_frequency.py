"""High-frequency clip detection.

A clip is high-frequency when some joint's rotation is dominated by an
oscillation at a period of a few source frames, which linear resampling would
alias. A fast large move -- a swat, a hit reaction, an arm flail -- also puts
energy at short periods, but spread over a broad band beside its slow
content, so the high band must carry most of the joint's motion. Two kinds
qualify:

* strong -- large high-band amplitude: a frame-locked insect wing beat,
  including a beat that starts or stops inside the clip;
* sustained -- a small, locally confined oscillation that keeps going through
  the clip at a clearly short period: a wing-tip or feather tremble on a
  gliding bird. The upper harmonics of a short cycle and a slower chatter
  near the band edge (mandibles, fangs) do not qualify.

An isolated one-frame excursion -- a joint flipping for a single key and back,
a data glitch -- is broadband too and on an otherwise still joint dominates its
spectrum, so it is bridged by its neighbours before scoring. An excursion
with comparable motion within one high-band period (a beat) is not isolated
and is kept.

Linear resampling to fewer frames aliases such content (a period-4 beat
downsampled by 1.8 turns into a slow beat pattern, a period-2 one into
near-stillness), so the loader never downsamples these clips: they are only
slowed down, tiled and cropped within the window, never squeezed into it (see
``MotionDataset._prepare_sample``).

The verdict is baked per species into ``cond.npy`` by
``tools/regenerate_dataset_artifacts.py`` (``HIGH_FREQUENCY_COND_KEY``), so
the loader and the review page read it instead of scoring every clip.
"""

import numpy as np

# Longest period, in source frames, counted as high-frequency content.
HIGH_FREQUENCY_MAX_PERIOD = 6.0
# Share of the joint's motion RMS that the high band must carry to count at all.
HIGH_FREQUENCY_SHARE = 0.5
# High-band RMS at and above which a joint is high-frequency on its own.
HIGH_FREQUENCY_STRONG_RMS = 0.3
# High-band RMS at and above which a sustained oscillation is high-frequency.
HIGH_FREQUENCY_SUSTAINED_RMS = 0.04
# Longest dominant period, in source frames, of a sustained oscillation.
HIGH_FREQUENCY_SUSTAINED_MAX_PERIOD = 5.4
# Share of frames on which the high-band energy must stay present to be sustained.
HIGH_FREQUENCY_SUSTAINED_COVERAGE = 0.9
# Fewest cycles of the dominant high-band period over the clip to be sustained.
HIGH_FREQUENCY_SUSTAINED_MIN_CYCLES = 6
# A frame is covered when its smoothed high-band energy reaches this fraction
# of the joint's mean high-band energy.
HIGH_FREQUENCY_PRESENT_FRACTION = 0.25
# Score at and above which a clip is high-frequency.
HIGH_FREQUENCY_SCORE_THRESHOLD = 1.0
# Clips shorter than this have too few spectral bins to judge and score 0.
HIGH_FREQUENCY_MIN_FRAMES = 8
# Smallest 6-D rotation jump into and out of a frame for it to be an excursion.
HIGH_FREQUENCY_SPIKE_JUMP = 1.0
# An excursion's neighbours must be this much closer to each other than either
# jump, i.e. the joint returns to where it was.
HIGH_FREQUENCY_SPIKE_RETURN = 0.3
# Bump when the score definition changes, so a cond baked by an older
# definition is refused until regenerated.
HIGH_FREQUENCY_SCORE_VERSION = 5
# cond entry key: {"version": HIGH_FREQUENCY_SCORE_VERSION, "clips": sorted
# motion file names ("<clip>.npy") of the species' high-frequency clips}.
HIGH_FREQUENCY_COND_KEY = "high_frequency_clips"


def _moving_mean(values, width):
    """Centred box mean over axis 0, zero-padded at both ends."""
    left = (width - 1) // 2
    padded = np.pad(values, [(left + 1, width - left - 1)] + [(0, 0)] * (values.ndim - 1))
    cumulative = np.cumsum(padded, axis=0)
    return (cumulative[width:] - cumulative[:-width]) / width


def _bridge_isolated_spikes(rotations):
    """``rotations`` [T, J, 6] with every isolated one-frame excursion replaced
    by the mean of its two neighbours."""
    jumps = np.linalg.norm(np.diff(rotations, axis=0), axis=2)
    jump_in, jump_out = jumps[:-1], jumps[1:]
    smaller = np.minimum(jump_in, jump_out)
    bridge = np.linalg.norm(rotations[2:] - rotations[:-2], axis=2)
    candidates = ((smaller >= HIGH_FREQUENCY_SPIKE_JUMP)
                  & (bridge < HIGH_FREQUENCY_SPIKE_RETURN * smaller))
    window = int(HIGH_FREQUENCY_MAX_PERIOD)
    fixed = rotations
    for k, joint in zip(*np.nonzero(candidates)):
        # Frame k + 1 jumps in at jumps[k] and out at jumps[k + 1]. Any other
        # comparable jump within one high-band period makes it part of an
        # oscillation, not a lone excursion.
        around = np.concatenate([jumps[max(0, k - window):k, joint], jumps[k + 2:k + 2 + window, joint]])
        if around.size and around.max() >= HIGH_FREQUENCY_SPIKE_RETURN * smaller[k, joint]:
            continue
        if fixed is rotations:
            fixed = rotations.copy()
        fixed[k + 1, joint] = 0.5 * (rotations[k, joint] + rotations[k + 2, joint])
    return fixed


def high_frequency_score(motion) -> float:
    """Largest per-joint high-band RMS of the 6-D rotation channels, in units
    of the threshold the joint qualifies for.

    ``motion`` is a stored clip ``[T, J, F]`` (rotations in channels 3-8). The
    high band is every period of at most ``HIGH_FREQUENCY_MAX_PERIOD`` frames;
    its RMS is taken from the one-sided spectrum of the mean-removed channels,
    in 6-D units (roughly radians for small angles). A joint whose high band
    carries at least ``HIGH_FREQUENCY_SHARE`` of its motion is measured against
    ``HIGH_FREQUENCY_STRONG_RMS``, a sustained one (see the module docstring)
    against the lower ``HIGH_FREQUENCY_SUSTAINED_RMS``; any other joint scores
    0. Maximised over joints so a single fast joint marks the clip. Isolated
    one-frame excursions are bridged first (see the module docstring).
    """
    motion = np.asarray(motion)
    frames = int(motion.shape[0])
    if frames < HIGH_FREQUENCY_MIN_FRAMES:
        return 0.0
    rotations = _bridge_isolated_spikes(motion[:, :, 3:9].astype(np.float64))
    rotations = rotations - rotations.mean(axis=0)
    spectrum = np.fft.rfft(rotations, axis=0)
    # One-sided weights: every bin stands for itself and its negative-frequency
    # mirror, except the Nyquist bin of an even-length clip, which has none.
    weights = np.full(spectrum.shape[0], 2.0)
    if frames % 2 == 0:
        weights[-1] = 1.0
    high = np.fft.rfftfreq(frames) >= 1.0 / HIGH_FREQUENCY_MAX_PERIOD
    # [bins, J] power, summed over the rotation channels.
    bin_power = (weights[:, None] * (np.abs(spectrum) ** 2).sum(axis=2)) / frames ** 2
    high_power = bin_power[high]
    high_rms = np.sqrt(high_power.sum(axis=0))
    share = high_rms / np.maximum(np.sqrt(bin_power.sum(axis=0)), 1e-12)

    # Bin index k of the dominant high-band bin is k cycles over the clip.
    cycles = np.flatnonzero(high)[high_power.argmax(axis=0)]
    period = frames / cycles
    # High-band signal back in time; its energy smoothed over one max period
    # so a sinusoid's zero crossings do not read as gaps.
    band = np.where(high[:, None, None], spectrum, 0.0)
    energy = (np.fft.irfft(band, n=frames, axis=0) ** 2).sum(axis=2)
    energy = _moving_mean(energy, int(HIGH_FREQUENCY_MAX_PERIOD))
    present = energy >= HIGH_FREQUENCY_PRESENT_FRACTION * energy.mean(axis=0)
    coverage = present.mean(axis=0)

    dominant = share >= HIGH_FREQUENCY_SHARE
    sustained = (dominant
                 & (coverage >= HIGH_FREQUENCY_SUSTAINED_COVERAGE)
                 & (cycles >= HIGH_FREQUENCY_SUSTAINED_MIN_CYCLES)
                 & (period <= HIGH_FREQUENCY_SUSTAINED_MAX_PERIOD))
    threshold = np.where(sustained, HIGH_FREQUENCY_SUSTAINED_RMS, HIGH_FREQUENCY_STRONG_RMS)
    return float(np.where(dominant, high_rms / threshold, 0.0).max())


def is_high_frequency(score: float) -> bool:
    return float(score) >= HIGH_FREQUENCY_SCORE_THRESHOLD


def bake_high_frequency_clips(motion_names) -> dict:
    """cond value for one species from its high-frequency motion file names."""
    return {"version": HIGH_FREQUENCY_SCORE_VERSION, "clips": sorted(str(name) for name in motion_names)}


def high_frequency_clips(object_cond, cond_source="cond.npy") -> frozenset:
    """Motion file names of the species' high-frequency clips, as baked in its
    cond entry. A missing or stale bake is an error: the verdict decides
    whether a clip may be downsampled, so there is no safe default."""
    baked = object_cond.get(HIGH_FREQUENCY_COND_KEY)
    if not isinstance(baked, dict) or baked.get("version") != HIGH_FREQUENCY_SCORE_VERSION:
        found = baked.get("version") if isinstance(baked, dict) else None
        raise RuntimeError(
            f"{cond_source} for '{object_cond.get('object_type')}' has high-frequency "
            f"verdicts of version {found}, expected {HIGH_FREQUENCY_SCORE_VERSION}. "
            "Re-run tools/regenerate_dataset_artifacts.py (and merge_dataset_cond.py "
            "for a merged cond)."
        )
    return frozenset(baked["clips"])
