import argparse
from typing import List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from dissonance_functions import sethares_dissonance
from audio_playback import play_chord, stop_chord
from menu import clear_screen, select_from_menu

DEFAULT_START_FREQ = 440.0  # A4


def _as_freq_amp_arrays(note) -> Tuple[List[float], List[float]]:
    """Normalize the CLI/plot note representation into (frequencies, amplitudes) lists."""
    if isinstance(note, dict):
        return [float(f) for f in note["frequencies"]], [float(a) for a in note["amplitudes"]]
    frequencies, amplitudes = note
    return [float(f) for f in frequencies], [float(a) for a in amplitudes]


class DissonancePlot:
    """An interactive dissonance-vs-offset plot.

    Hovering highlights the nearest sample with a marker, a vertical guide
    line, and an annotation showing the offset and dissonance value.
    Holding the mouse button down plays the base note stacked with the note
    shifted by that offset; releasing it stops playback. Dragging while held
    down retunes the playing note as it crosses into a new semitone.
    """

    def __init__(
        self,
        base_note: Tuple[Sequence[float], Sequence[float]],
        offsets: Sequence[float],
        dissonance_values: Sequence[float],
    ) -> None:
        self.base_note = base_note
        self.offsets = np.asarray(offsets, dtype=float)
        self.dissonance_values = np.asarray(dissonance_values, dtype=float)
        self._is_pressed = False
        self._last_played_semitone = None

        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        self._build_static_plot()
        self._build_highlight_artists()

        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("axes_leave_event", self._on_leave)

    def _build_static_plot(self) -> None:
        ax = self.ax
        ax.plot(self.offsets, self.dissonance_values, color="#3b6fb6", linewidth=1.8, zorder=2)
        ax.fill_between(self.offsets, self.dissonance_values, color="#3b6fb6", alpha=0.12, zorder=1)

        ax.set_xlabel("Offset above base note (semitones)")
        ax.set_ylabel("Sensory dissonance")
        ax.set_title("Dissonance vs. Interval  (hover to inspect, hold to hear)")
        ax.set_xlim(float(self.offsets.min()), float(self.offsets.max()))
        top = float(self.dissonance_values.max())
        ax.set_ylim(0.0, top * 1.1 if top > 0 else 1.0)
        ax.grid(True, alpha=0.25)
        self.fig.tight_layout()

    def _build_highlight_artists(self) -> None:
        ax = self.ax
        (self.marker,) = ax.plot([], [], "o", color="#d1495b", markersize=8, zorder=4)
        self.vline = ax.axvline(self.offsets[0], color="#d1495b", linewidth=1, linestyle="--", alpha=0.0, zorder=3)
        self.annotation = ax.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d1495b", alpha=0.9),
            fontsize=9,
            visible=False,
            zorder=5,
        )

    def _nearest_index(self, x: float) -> int:
        return int(np.argmin(np.abs(self.offsets - x)))

    def _on_move(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            self._hide_highlight()
            return

        idx = self._nearest_index(event.xdata)
        offset = self.offsets[idx]
        value = self.dissonance_values[idx]

        self.marker.set_data([offset], [value])
        self.vline.set_xdata([offset, offset])
        self.vline.set_alpha(0.6)
        self.annotation.xy = (offset, value)
        self.annotation.set_text(f"{offset:+.2f} st\nD = {value:.4f}")
        self.annotation.set_visible(True)
        self.fig.canvas.draw_idle()

        if self._is_pressed:
            self._retune_if_new_note(offset)

    def _retune_if_new_note(self, offset: float) -> None:
        """While the mouse is held down, restart playback if dragging crossed into a new semitone."""
        semitone = round(offset)
        if semitone != self._last_played_semitone:
            self._last_played_semitone = semitone
            self._play_offset(offset)

    def _on_leave(self, _event) -> None:
        self._hide_highlight()
        if self._is_pressed:
            self._is_pressed = False
            self._last_played_semitone = None
            stop_chord()

    def _hide_highlight(self) -> None:
        self.marker.set_data([], [])
        self.vline.set_alpha(0.0)
        self.annotation.set_visible(False)
        self.fig.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        self._is_pressed = True
        idx = self._nearest_index(event.xdata)
        offset = self.offsets[idx]
        self._last_played_semitone = round(offset)
        self._play_offset(offset)

    def _on_release(self, _event) -> None:
        if self._is_pressed:
            self._is_pressed = False
            self._last_played_semitone = None
            stop_chord()

    def _play_offset(self, offset: float) -> None:
        base_frequencies, base_amplitudes = self.base_note
        ratio = 2.0 ** (offset / 12.0)
        shifted_frequencies = [freq * ratio for freq in base_frequencies]

        play_chord([(base_frequencies, base_amplitudes), (shifted_frequencies, base_amplitudes)])

    def show(self) -> None:
        plt.show()


def plot_dissonance_vs_offset(
    start_note,
    min_offset: float = 0.0,
    max_offset: float = 24.0,
    steps: int = 300,
    output_path: str = None,
    show: bool = True,
    interactive: bool = True,
) -> Tuple[List[float], List[float]]:
    """Plot (and optionally play) the dissonance between a note and a shifted copy of it.

    The x-axis is the offset in semitones from the starting note, and only
    goes upward (min_offset defaults to 0). The y-axis is the dissonance
    value returned by sethares_dissonance. With interactive=True, hovering
    over the curve highlights the nearest point and holding the mouse button
    down plays that interval as a chord until released.
    """
    base_frequencies, base_amplitudes = _as_freq_amp_arrays(start_note)
    base_note = (base_frequencies, base_amplitudes)

    offsets = np.linspace(min_offset, max_offset, steps)
    dissonance_values = np.empty(steps, dtype=float)

    for i, offset in enumerate(offsets):
        ratio = 2.0 ** (offset / 12.0)
        shifted_note = ([freq * ratio for freq in base_frequencies], base_amplitudes)
        dissonance_values[i] = sethares_dissonance(base_note, shifted_note)

    if interactive:
        plot = DissonancePlot(base_note, offsets, dissonance_values)
        if output_path is not None:
            plot.fig.savefig(output_path, dpi=200, bbox_inches="tight")
        if show:
            plot.show()
        else:
            plt.close(plot.fig)
    else:
        plt.figure(figsize=(9, 5))
        plt.plot(offsets, dissonance_values, color="#3b6fb6", linewidth=1.8)
        plt.fill_between(offsets, dissonance_values, color="#3b6fb6", alpha=0.12)
        plt.xlabel("Offset above base note (semitones)")
        plt.ylabel("Sensory dissonance")
        plt.title("Dissonance vs. Interval")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    return offsets.tolist(), dissonance_values.tolist()


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI parser for plotting dissonance vs. offset.

    --start-freq is intentionally left without a default: passing it on the
    command line skips the interactive menu entirely (for scripting/testing);
    omitting it launches the arrow-key menu instead, which prompts for it.
    """
    parser = argparse.ArgumentParser(description="Plot dissonance between a starting note and shifted copies of it.")
    parser.add_argument("--start-freq", type=float, default=None, help="Base frequency for the starting note (Hz); skips the menu if given")
    parser.add_argument("--start-amp", type=float, default=0.8, help="Amplitude for the starting note")
    parser.add_argument("--min-offset", type=float, default=0.0, help="Minimum semitone offset (>= 0)")
    parser.add_argument("--max-offset", type=float, default=24.0, help="Maximum semitone offset")
    parser.add_argument("--steps", type=int, default=300, help="Number of samples to plot")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the plot image")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    parser.add_argument("--no-interactive", action="store_true", help="Disable hover/hold interactivity")
    return parser


def _prompt_for_start_freq(default: float = DEFAULT_START_FREQ) -> float:
    while True:
        raw = input(f"Enter base frequency in Hz [{default:g} = A4]: ").strip()
        if not raw:
            return default
        try:
            freq = float(raw)
        except ValueError:
            print("Please enter a numeric frequency.")
            continue
        if freq <= 0:
            print("Frequency must be positive.")
            continue
        return freq


def _generate_plot(args: argparse.Namespace, start_freq: float) -> None:
    start_note = ([start_freq], [args.start_amp])
    plot_dissonance_vs_offset(
        start_note,
        min_offset=max(0.0, args.min_offset),
        max_offset=args.max_offset,
        steps=args.steps,
        output_path=args.output,
        show=not args.no_show,
        interactive=not args.no_interactive,
    )


def _action_plot_dissonance(args: argparse.Namespace) -> None:
    _generate_plot(args, _prompt_for_start_freq())


# Menu entries as (label, handler) pairs. To add a new feature later, write a
# handler that takes `args` and append it here -- the menu grows automatically.
MENU_ACTIONS = [
    ("Plot dissonance vs. offset", _action_plot_dissonance),
]


def run_menu(args: argparse.Namespace) -> None:
    """Show an arrow-key menu of MENU_ACTIONS in a loop until the user quits."""
    labels = [label for label, _ in MENU_ACTIONS] + ["Quit"]
    quit_index = len(MENU_ACTIONS)

    while True:
        choice = select_from_menu(labels, title="Music Experimentation -- choose an action:")
        if choice is None or choice == quit_index:
            return
        MENU_ACTIONS[choice][1](args)
        clear_screen()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.start_freq is not None:
        _generate_plot(args, args.start_freq)
        return

    run_menu(args)


if __name__ == "__main__":
    main()
