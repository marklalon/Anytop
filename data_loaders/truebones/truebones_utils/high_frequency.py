"""High-frequency clip detection.

A clip is high-frequency when some joint's rotation carries strong content at a
period of a few source frames -- a frame-locked insect wing beat, a one- or
two-frame snap. Linear resampling to fewer frames aliases such content (a
period-4 beat downsampled by 1.8 turns into a slow beat pattern, a period-2
one into near-stillness), so the loader never downsamples these clips: they
are only slowed down, tiled and cropped within the window, never squeezed into
it (see ``MotionDataset._prepare_sample``).
"""

import numpy as np

# Longest period, in source frames, counted as high-frequency content.
HIGH_FREQUENCY_MAX_PERIOD = 6.0
# Score at and above which a clip is high-frequency.
HIGH_FREQUENCY_SCORE_THRESHOLD = 0.3
# Clips shorter than this have too few spectral bins to judge and score 0.
HIGH_FREQUENCY_MIN_FRAMES = 8
# Bump when the score definition changes, so cached scores are recomputed.
HIGH_FREQUENCY_SCORE_VERSION = 2


def high_frequency_score(motion) -> float:
    """Largest per-joint RMS of the 6-D rotation channels at periods of at most
    ``HIGH_FREQUENCY_MAX_PERIOD`` frames.

    ``motion`` is a stored clip ``[T, J, F]`` (rotations in channels 3-8). The
    RMS is taken from the one-sided spectrum of the mean-removed channels, in
    6-D units (roughly radians for small angles), and maximised over joints so
    a single fast joint marks the clip.
    """
    motion = np.asarray(motion)
    frames = int(motion.shape[0])
    if frames < HIGH_FREQUENCY_MIN_FRAMES:
        return 0.0
    rotations = motion[:, :, 3:9].astype(np.float64)
    rotations = rotations - rotations.mean(axis=0)
    spectrum = np.fft.rfft(rotations, axis=0)
    # One-sided weights: every bin stands for itself and its negative-frequency
    # mirror, except the Nyquist bin of an even-length clip, which has none.
    weights = np.full(spectrum.shape[0], 2.0)
    if frames % 2 == 0:
        weights[-1] = 1.0
    high = np.fft.rfftfreq(frames) >= 1.0 / HIGH_FREQUENCY_MAX_PERIOD
    power = (weights[high, None, None] * np.abs(spectrum[high]) ** 2).sum(axis=(0, 2))
    return float(np.sqrt(power / frames ** 2).max())


def is_high_frequency(score: float) -> bool:
    return float(score) >= HIGH_FREQUENCY_SCORE_THRESHOLD
