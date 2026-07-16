"""A minimal arrow-key + Enter driven terminal menu.

Falls back to plain numbered input when there's no interactive console
attached (e.g. piped/redirected stdin), so the same code works both when
run normally and when scripted or tested headlessly.
"""

import os
import sys
from typing import List, Optional

if os.name == "nt":
    import ctypes
    import msvcrt

    def _enable_ansi() -> None:
        """Turn on ANSI escape-sequence support in the Windows console, if attached."""
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            pass

    _enable_ansi()
else:
    msvcrt = None


_ARROW_PREFIXES = (b"\x00", b"\xe0")
_UP = b"H"
_DOWN = b"P"
_ENTER = b"\r"
_ESC = b"\x1b"


def select_from_menu(options: List[str], title: Optional[str] = None, selected: int = 0) -> Optional[int]:
    """Show `options`; Up/Down moves the highlight, Enter selects, Esc cancels.

    Returns the chosen index, or None if the user cancelled.
    """
    if msvcrt is None or not sys.stdin.isatty():
        return _select_fallback(options, title)

    _render(options, selected, title, first=True)
    while True:
        key = msvcrt.getch()
        if key in _ARROW_PREFIXES:
            arrow = msvcrt.getch()
            if arrow == _UP:
                selected = (selected - 1) % len(options)
                _render(options, selected, title)
            elif arrow == _DOWN:
                selected = (selected + 1) % len(options)
                _render(options, selected, title)
        elif key == _ENTER:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return selected
        elif key == _ESC:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return None


def _render(options: List[str], selected: int, title: Optional[str], first: bool = False) -> None:
    """Redraw the menu in place using ANSI cursor-movement codes."""
    lines = len(options) + (1 if title else 0)
    if not first:
        sys.stdout.write(f"\x1b[{lines}A")

    if title:
        sys.stdout.write("\x1b[2K" + title + "\n")

    for i, label in enumerate(options):
        sys.stdout.write("\x1b[2K")
        if i == selected:
            sys.stdout.write(f"\x1b[7m> {label}\x1b[0m\n")
        else:
            sys.stdout.write(f"  {label}\n")

    sys.stdout.flush()


def clear_screen() -> None:
    """Clear the terminal, if an interactive console is attached."""
    if not sys.stdout.isatty():
        return
    os.system("cls" if os.name == "nt" else "clear")


def _select_fallback(options: List[str], title: Optional[str]) -> Optional[int]:
    """Plain numbered-input menu for non-interactive or non-Windows consoles."""
    if title:
        print(title)
    for i, label in enumerate(options):
        print(f"  {i + 1}. {label}")

    while True:
        raw = input(f"Select 1-{len(options)}: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= choice <= len(options):
            return choice - 1
        print(f"Please enter a number between 1 and {len(options)}.")
