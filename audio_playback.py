"""Continuous, click-free chord playback for interactive dragging.

A one-shot "stop the old buffer, synthesize and play a new one" approach
pops audibly every time the pitch changes, because sd.stop() truncates the
waveform mid-cycle and the new buffer restarts its phase from zero. Instead,
this module runs a single persistent output stream with a bank of sine
oscillators whose frequencies can be retargeted at any time: the callback
keeps a running phase per oscillator, so frequency updates are instant and
never discontinuous. Only the overall volume is ramped on start/stop, which
avoids the on/off click without adding any audible pitch-change latency.
"""

import threading
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except OSError:
    # sounddevice imports fine but raises OSError if PortAudio can't find a device.
    AUDIO_AVAILABLE = False


SAMPLE_RATE = 44100
BLOCK_SIZE = 256
_VOLUME_RAMP_SECONDS = 0.015  # fade time for start/stop, short enough to feel instant


class _ChordVoice:
    """A persistent additive-synthesis voice whose oscillator frequencies can be updated live."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._lock = threading.Lock()
        self._freqs = np.zeros(0)
        self._osc_amps = np.zeros(0)
        self._phases = np.zeros(0)
        self._volume = 0.0
        self._target_volume = 0.0
        self._stream: Optional["sd.OutputStream"] = None

    def _ensure_stream(self, n_oscillators: int) -> None:
        """(Re)create the output stream if the oscillator count has changed."""
        if self._stream is not None and self._freqs.size == n_oscillators:
            return
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

        self._freqs = np.zeros(n_oscillators)
        self._osc_amps = np.zeros(n_oscillators)
        self._phases = np.zeros(n_oscillators)

        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frames, _time_info, _status) -> None:
        with self._lock:
            freqs = self._freqs
            osc_amps = self._osc_amps
            phases = self._phases
            volume = self._volume
            target_volume = self._target_volume

        if freqs.size == 0:
            outdata[:, 0] = 0.0
            return

        # Advance each oscillator's phase across this block from where it left off,
        # so changing `freqs` between callbacks never introduces a phase jump.
        t = (np.arange(frames) / self._sample_rate)[:, np.newaxis]
        phase_matrix = phases[np.newaxis, :] + 2.0 * np.pi * freqs[np.newaxis, :] * t
        wave = np.sin(phase_matrix) @ osc_amps

        ramp_samples = min(frames, max(1, int(_VOLUME_RAMP_SECONDS * self._sample_rate)))
        volume_env = np.full(frames, target_volume)
        if volume != target_volume:
            volume_env[:ramp_samples] = np.linspace(volume, target_volume, ramp_samples, endpoint=True)

        outdata[:, 0] = (wave * volume_env).astype(np.float32)

        with self._lock:
            self._phases = (phases + 2.0 * np.pi * freqs * frames / self._sample_rate) % (2.0 * np.pi)
            self._volume = float(volume_env[-1])

    def set_notes(self, note_specs: Sequence[Tuple[Sequence[float], Sequence[float]]]) -> None:
        """Retarget the oscillators to a new chord and (re)start audible volume."""
        frequencies = []
        amplitudes = []
        for freqs, amps in note_specs:
            frequencies.extend(freqs)
            amplitudes.extend(amps)

        freqs_arr = np.array(frequencies, dtype=float)
        amps_arr = np.array(amplitudes, dtype=float)
        total = np.sum(amps_arr)
        if total > 0:
            amps_arr = amps_arr / total * 0.8

        self._ensure_stream(len(freqs_arr))
        with self._lock:
            self._freqs[:] = freqs_arr
            self._osc_amps[:] = amps_arr
            self._target_volume = 1.0

    def stop(self) -> None:
        with self._lock:
            self._target_volume = 0.0


_voice = _ChordVoice()


def play_chord(note_specs: Sequence[Tuple[Sequence[float], Sequence[float]]]) -> None:
    """Start (or instantly retune) the persistent chord voice.

    Safe to call repeatedly while the same voice is already sounding --
    frequencies update in place with no stop/restart click.
    """
    if not AUDIO_AVAILABLE:
        print("Audio playback unavailable: no working sound device found.")
        return
    _voice.set_notes(note_specs)


def stop_chord() -> None:
    """Fade the currently playing chord to silence."""
    if AUDIO_AVAILABLE:
        _voice.stop()
