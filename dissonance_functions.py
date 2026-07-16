from numbers import Number
from typing import Iterable, List, Sequence, Tuple

import numpy as np


Spectrum = Sequence[Tuple[float, float]]


# ---------------------------------------------------------------------------
# Published Sethares sensory-dissonance parameters
# ---------------------------------------------------------------------------
# These are the constants from W. A. Sethares, "Tuning, Timbre, Spectrum,
# Scale" (and the 1993 JASA paper "Local consonance and the relationship
# between timbre and scale"), as used in his reference `dissmeasure` routine.
#
# The roughness contributed by a single pair of partials is modelled as a
# difference of two exponentials in the frequency separation between them:
#
#     d(f_low, f_high) = a * ( C1 * exp(A1 * s * df) + C2 * exp(A2 * s * df) )
#
# where
#     df   = f_high - f_low                     (frequency separation, Hz)
#     s    = D_STAR / (S1 * f_low + S2)          (scales df by critical band)
#     a    = amplitude weighting of the pair
#
# The `s` term rescales the frequency difference so that the point of maximum
# roughness tracks the width of the auditory critical band, which grows with
# the (lower) frequency of the pair. The curve is zero at unison (df = 0,
# because C1 + C2 = 0) and decays back to zero once the partials are far apart.
D_STAR = 0.24   # separation (in scaled units) of maximum dissonance
S1 = 0.0207     # critical-band slope
S2 = 18.96      # critical-band offset
C1 = 5.0        # weight of the first (rising) exponential
C2 = -5.0       # weight of the second (falling) exponential
A1 = -3.51      # decay rate of the first exponential
A2 = -5.75      # decay rate of the second exponential


def _is_numeric_sequence(value: object) -> bool:
    """Return True when a value looks like a sequence of numeric scalars."""
    if isinstance(value, (str, bytes, dict)):
        return False
    try:
        iterator = iter(value)
    except TypeError:
        return False

    for item in iterator:
        if not isinstance(item, Number):
            return False
    return True


def _normalize_spectrum(note: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Convert a note-like input into a list of (frequency, amplitude) pairs.

    Supported forms include:
    - [(220.0, 0.8), (440.0, 0.4)]
    - ([220.0, 440.0], [0.8, 0.4])
    - {"frequencies": [220.0, 440.0], "amplitudes": [0.8, 0.4]}
    """
    if isinstance(note, dict):
        if "frequencies" in note and "amplitudes" in note:
            frequencies = note["frequencies"]
            amplitudes = note["amplitudes"]
            if len(frequencies) != len(amplitudes):
                raise ValueError("Frequency and amplitude arrays must have the same length.")
            return [(float(f), float(a)) for f, a in zip(frequencies, amplitudes)]
        if "frequency" in note and "amplitude" in note:
            return [(float(note["frequency"]), float(note["amplitude"]))]
        raise ValueError("Dictionary spectra must contain 'frequencies'/'amplitudes' or 'frequency'/'amplitude'.")

    if isinstance(note, (tuple, list)) and len(note) == 2:
        first, second = note
        if _is_numeric_sequence(first) and _is_numeric_sequence(second) and len(first) == len(second):
            return [(float(f), float(a)) for f, a in zip(first, second)]

    normalized: List[Tuple[float, float]] = []
    for item in note:
        if isinstance(item, dict):
            frequency = item.get("frequency")
            amplitude = item.get("amplitude")
            if frequency is None or amplitude is None:
                raise ValueError("Each dictionary entry must include 'frequency' and 'amplitude'.")
            normalized.append((float(frequency), float(amplitude)))
        else:
            try:
                if len(item) != 2:
                    raise ValueError
                frequency, amplitude = item
            except (TypeError, ValueError) as exc:
                raise ValueError("Each spectrum entry must be a (frequency, amplitude) pair.") from exc
            normalized.append((float(frequency), float(amplitude)))
    return normalized


def _spectrum_arrays(note: Spectrum) -> Tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, amplitudes) as float arrays, dropping non-positive freqs.

    Partials at f <= 0 Hz are not physically meaningful and are discarded so
    they do not contaminate the pairwise roughness sum.
    """
    pairs = _normalize_spectrum(note)
    if not pairs:
        return np.empty(0), np.empty(0)

    freqs = np.array([f for f, _ in pairs], dtype=float)
    amps = np.array([a for _, a in pairs], dtype=float)

    keep = freqs > 0.0
    return freqs[keep], amps[keep]


def _pairwise_dissonance(
    note_a: Spectrum,
    note_b: Spectrum,
    model: str = "product",
) -> float:
    """Sethares sensory dissonance between the partials of two spectra.

    Every partial of ``note_a`` is compared against every partial of
    ``note_b`` using the published Sethares roughness equation, and the
    contributions of all pairs are summed.

    ``model`` selects how each pair is weighted by its two amplitudes:
    - "product": a = amp_a * amp_b  (weight by both partials, the default)
    - "min":     a = min(amp_a, amp_b)  (as in Sethares' reference `dissmeasure`)
    """
    freqs_a, amps_a = _spectrum_arrays(note_a)
    freqs_b, amps_b = _spectrum_arrays(note_b)

    if freqs_a.size == 0 or freqs_b.size == 0:
        return 0.0

    # Broadcast every partial of A (rows) against every partial of B (cols)
    # so the whole cross-product of pairs is evaluated in one vectorized pass.
    fa = freqs_a[:, np.newaxis]
    fb = freqs_b[np.newaxis, :]
    aa = amps_a[:, np.newaxis]
    ab = amps_b[np.newaxis, :]

    # The critical-band scaling uses the *lower* frequency of each pair; the
    # roughness curve itself depends only on the absolute separation.
    f_low = np.minimum(fa, fb)
    df = np.abs(fa - fb)

    # Amplitude weighting for each pair.
    if model == "min":
        weight = np.minimum(aa, ab)
    elif model == "product":
        weight = aa * ab
    else:
        raise ValueError("model must be 'product' or 'min'.")

    # Sethares roughness: s rescales df by the critical bandwidth at f_low,
    # then two exponentials form the rise-and-fall roughness curve.
    s = D_STAR / (S1 * f_low + S2)
    roughness = weight * (C1 * np.exp(A1 * s * df) + C2 * np.exp(A2 * s * df))

    return float(np.sum(roughness))


def sethares_dissonance(
    *notes: Iterable[Tuple[float, float]],
    model: str = "product",
) -> float:
    """Return the summed Sethares sensory dissonance between two or more spectra.

    Each note can be given as one of the following:
    - a list of (frequency, amplitude) pairs
    - a pair of numeric arrays: ([frequencies], [amplitudes])
    - a dictionary with 'frequencies' and 'amplitudes'

    When more than two notes are supplied, the dissonance is summed over every
    unordered pair of notes.

    Examples:
        sethares_dissonance(note_a, note_b)
        sethares_dissonance(note_a, note_b, note_c, note_d)
    """
    if len(notes) < 2:
        return 0.0

    total = 0.0
    for index, note_a in enumerate(notes):
        for note_b in notes[index + 1:]:
            total += _pairwise_dissonance(note_a, note_b, model=model)

    return total
