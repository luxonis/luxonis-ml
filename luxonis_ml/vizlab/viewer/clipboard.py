"""Best-effort clipboard access through whatever the platform provides.

A viewer copying a clicked annotation must not fail — or add a dependency —
because no clipboard tool is installed, so `copy` shells out to the first
available system utility and simply reports that it found none. The caller
prints the payload either way.

The X11/Wayland clipboards belong to a *running* process rather than to the
system, so a helper is spawned to own the selection (``xclip``/``xsel`` fork for
exactly this reason); writing it from this process would lose the content the
moment the viewer exits. That fork is also why `copy` must not read the helper's
output — the forked child inherits those pipes and holds them open for as long
as it owns the selection, so waiting for them to close means waiting for the
*next* copy — and why `copy_later` exists: even done right, taking ownership
costs a couple hundred milliseconds, which an interactive loop cannot spend.
"""

import queue
import shutil
import subprocess
import threading
from collections.abc import Sequence

from loguru import logger

#: Clipboard writers to try, in order, each reading the text from stdin. The
#: Wayland and X11 tools come first (a Wayland session may still offer ``xclip``
#: through XWayland, but ``wl-copy`` is the native one), then macOS, then the
#: Windows/WSL console tool.
_WRITERS: tuple[tuple[str, ...], ...] = (
    ("wl-copy",),
    ("xclip", "-selection", "clipboard"),
    ("xsel", "--clipboard", "--input"),
    ("pbcopy",),
    ("clip.exe",),
)


def copy(
    text: str, writers: "Sequence[Sequence[str]] | None" = None
) -> str | None:
    """Put ``text`` on the system clipboard, if a tool is available.

    Blocks until the tool has taken ownership of the selection, which is not
    instant (``wl-copy`` takes ~200 ms); call `copy_later` from anything
    interactive.

    Args:
        text: The text to copy.
        writers: Candidate commands to try, each reading stdin; defaults to the
            platform tools this module knows.

    Returns:
        The name of the tool that took the text, or ``None`` when none was
        available or all of them failed.

    Examples:
        >>> copy("hi", writers=[])  # no tool available

    """
    for command in _WRITERS if writers is None else writers:
        if not command:
            continue
        executable = shutil.which(command[0])
        if executable is None:
            continue
        try:
            subprocess.run(
                [executable, *command[1:]],
                input=text.encode("utf-8"),
                check=True,
                # Never a pipe: the helper forks to own the selection and the
                # child inherits it, so reading to EOF would block until the
                # clipboard is replaced (see the module docstring).
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        return command[0]
    return None


class _BackgroundWriter:
    """One thread performing queued copies, started on first use.

    Only the newest queued text is written: while a helper is taking ownership
    (long enough for a second click to arrive), the texts behind it are already
    obsolete, and the clipboard must end up holding the last thing clicked.

    The thread is a daemon, so a copy still in flight never holds up interpreter
    exit — and a helper that already owns the selection owns it independently of
    this process anyway.
    """

    def __init__(self) -> None:
        self._pending: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def submit(self, text: str) -> None:
        """Queue ``text`` and return without waiting for it to be written."""
        self._start()
        self._pending.put(text)

    def _start(self) -> None:
        """Start the writer thread unless it is already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._drain, name="vizlab-clipboard", daemon=True
            )
            self._thread.start()

    def _drain(self) -> None:
        """Copy queued texts forever, skipping superseded ones."""
        while True:
            if copy(self._newest(self._pending.get())) is None:
                logger.debug(
                    "No clipboard tool was found, so nothing could be copied."
                )

    def _newest(self, text: str) -> str:
        """Return the last text queued so far, starting from ``text``."""
        while True:
            try:
                text = self._pending.get_nowait()
            except queue.Empty:
                return text


_BACKGROUND = _BackgroundWriter()


def copy_later(text: str) -> None:
    """Queue ``text`` for the clipboard and return immediately.

    An interactive caller cannot wait for a clipboard helper to start (see the
    module docstring), and does not need to know whether one exists — a failure
    is logged rather than returned. See `_BackgroundWriter` for what happens to
    several texts queued in quick succession.

    Args:
        text: The text to copy.

    """
    _BACKGROUND.submit(text)
