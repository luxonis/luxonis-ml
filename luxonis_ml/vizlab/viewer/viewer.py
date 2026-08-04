"""The `Viewer`: an interactive, backend-agnostic image window with hover.

`Viewer` presents `Frame` objects (a scene plus its interaction maps), fits them
to the screen, routes mouse moves through the hover map to show per-annotation
tooltips, and runs the keyboard loop — all over a pluggable `WindowBackend`
(OpenCV by default). Callers build frames with `Renderable.frame` and never
touch windowing themselves. Composition retains interactions automatically, so
calling `frame()` on a grid or panel works the same way as on one `Image`.
"""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from rich import print_json

from luxonis_ml.typing import ParamValue
from luxonis_ml.vizlab import io
from luxonis_ml.vizlab.interaction.frame import Frame
from luxonis_ml.vizlab.render.capture import ClickMap, HitMap, PickMap
from luxonis_ml.vizlab.scene.image import Renderable
from luxonis_ml.vizlab.tooltip import Tooltip

from . import clipboard
from .backend import MouseHandler, WindowBackend
from .cv2_backend import Cv2Backend
from .hud import render_controls_card
from .layers import LayerState
from .tooltip_render import TooltipCard, blit_rgba_on_bgr, prepare_tooltip

#: A per-window callback that re-renders the window's `Frame` for a `LayerState`.
RenderFn = Callable[[LayerState], Frame]

#: What a `Viewer` does with the source data of a clicked annotation.
PickFn = Callable[[ParamValue], None]

#: Longest string `report_pick` prints in full. Comfortably above anything a
#: person writes into dataset metadata (a path, a note, a URL) and far below the
#: run-length data of even a small mask, which is the value this exists for.
_READABLE_LIMIT = 200

#: `wait`'s poll timeout while nothing is moving — long enough to idle cheaply.
_IDLE_POLL_MS = 20
#: ...and right after a hover redraw, so a moving tooltip is not held a whole
#: idle poll behind the cursor. The pointer stopping restores the idle timeout.
_ACTIVE_POLL_MS = 1

#: Key that writes the displayed frames to disk (see `Viewer.save`).
_SAVE_KEY = "s"

#: Runs of characters replaced when a window name becomes a filename.
_UNSAFE_IN_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(name: str) -> str:
    """Turn a window name into a filename stem that is safe on any platform."""
    return _UNSAFE_IN_NAME.sub("-", name).strip("-.") or "frame"


def report_pick(source: "ParamValue") -> None:
    """Print a clicked annotation's source data as JSON and copy it.

    The default `Viewer` pick handler. The two destinations are wanted for
    different things, so they are not given the same text: the terminal gets a
    version fit to *read*, with any value too long for a person elided (see
    `_readable`), while the clipboard gets the annotation *whole*, so what is
    pasted back is the real thing. Each elision says how much it stands for, so
    the printout never implies the copy was cut short too.

    Printing is also what makes a click useful where nothing can be copied at
    all. The clipboard write is queued rather than awaited (see
    `luxonis_ml.vizlab.viewer.clipboard.copy_later`): taking ownership of a
    selection costs enough to stutter the viewer that called this.

    Args:
        source: The annotation's source data (see `Annotation.source`). Values
            JSON cannot represent are stringified rather than raising, since a
            click must never take down an interactive session.

    """
    print_json(_dumps(_readable(source)))
    clipboard.copy_later(_dumps(source))


def _dumps(source: "ParamValue") -> str:
    """Render ``source`` as the JSON a pick is reported in."""
    return json.dumps(source, indent=2, ensure_ascii=False, default=str)


def _readable(source: "ParamValue") -> "ParamValue":
    """Return ``source`` with values too long to read replaced by a summary.

    A segmentation mask is carried as run-length data, which for a single
    720p instance runs past 70,000 characters — several hundred wrapped
    terminal lines of digits, burying the class, the identity and the metadata
    that the click was about. It is worth keeping (it is what makes the copied
    annotation reproduce the mask) but never worth reading, and the same goes
    for anything else that long.
    """
    if isinstance(source, str):
        if len(source) <= _READABLE_LIMIT:
            return source
        return f"… {len(source)} characters, copied in full …"
    if isinstance(source, Mapping):
        return {key: _readable(value) for key, value in source.items()}
    if isinstance(source, Sequence):
        return [_readable(item) for item in source]
    return source


def _unique_path(directory: Path, stem: str) -> Path:
    """Return ``<stem>.png`` in ``directory``, numbering past any that exist.

    Saving several times in one session builds up a numbered series instead of
    overwriting the previous shot.
    """
    path = directory / f"{stem}.png"
    index = 1
    while path.exists():
        path = directory / f"{stem}-{index}.png"
        index += 1
    return path


@dataclass(frozen=True, slots=True)
class PreparedFrame:
    """A fully rendered frame ready for immediate backend presentation.

    `Viewer.prepare` performs scene rendering, screen fitting, pixel conversion,
    and interaction-map scaling. The resulting value is independent of the
    window backend and can therefore be prepared on a background thread while
    another frame is displayed.

    Attributes:
        bgr: Display-ready BGR pixels.
        hitmap: Hover regions scaled to the BGR pixels.
        clickmap: Click regions scaled to the BGR pixels.
        pickmap: Annotation source regions scaled to the BGR pixels.

    """

    bgr: np.ndarray
    hitmap: HitMap
    clickmap: ClickMap
    pickmap: PickMap = field(default_factory=PickMap.empty)


@dataclass
class _HoverState:
    """Per-window state used to redraw the hover tooltip on mouse-move.

    ``render``, when set, makes the window *interactive*: layer-control keys
    re-render it through this callback with the viewer's current `LayerState`.
    """

    base: np.ndarray
    hitmap: HitMap
    clickmap: ClickMap = field(default_factory=ClickMap.empty)
    pickmap: PickMap = field(default_factory=PickMap.empty)
    hover: Tooltip | None = None
    mouse: tuple[int, int] = field(default=(0, 0))
    dirty: bool = False
    #: Action string from a panel click, awaiting the `wait` loop to apply it.
    pending: str | None = None
    #: Source data of an annotation clicked in the image, awaiting the same loop.
    #: A pick map never stores ``None``, so ``None`` means "nothing clicked".
    picked: "ParamValue | None" = None
    render: RenderFn | None = None
    #: Buffer the tooltip is drawn into, so `base` stays a clean backdrop to
    #: restore from. Allocated on the window's first hover.
    scratch: np.ndarray | None = None
    #: The card rendered for `hover`, reused for as long as the pointer stays
    #: within the same region and only its position changes.
    card: TooltipCard | None = None
    #: Region of `scratch` the last tooltip covered, restored before the next
    #: one is drawn — repainting the whole frame per mouse-move is wasteful.
    painted: tuple[int, int, int, int] | None = None
    #: What `Viewer.save` names this window's file, when the caller knows the
    #: subject better than the window title does (the source image, say). Falls
    #: back to the window name.
    save_as: str | None = None

    @property
    def shown(self) -> np.ndarray:
        """The frame currently on screen — the scratch when a tooltip is up."""
        if self.painted is None or self.scratch is None:
            return self.base
        return self.scratch

    def clear_hover(self) -> None:
        """Drop the tooltip and every buffer derived from the current frame."""
        self.hover = None
        self.scratch = None
        self.card = None
        self.painted = None
        self.dirty = False


#: Qt's key codes for the same cluster. The Qt OpenCV build reports these
#: instead of X11 keysyms, and their low byte is a control character rather
#: than a letter — harmless, but equally useless as a binding.
_QT_KEYS = {
    0x01000010: "home",
    0x01000011: "end",
    0x01000012: "left",
    0x01000013: "up",
    0x01000014: "right",
    0x01000015: "down",
    0x01000016: "pageup",
    0x01000017: "pagedown",
}

#: Final bytes of the CSI escape sequences a terminal-style backend sends for
#: the same keys: ``ESC [ D`` is Left, ``ESC [ 5 ~`` is Page Up.
_CSI_KEYS = {
    "A": "up",
    "B": "down",
    "C": "right",
    "D": "left",
    "H": "home",
    "F": "end",
}
_CSI_TILDE_KEYS = {"5": "pageup", "6": "pagedown"}

#: How long to wait for the rest of an escape sequence before concluding the
#: Escape key itself was pressed. The remaining bytes are already queued when
#: the ``ESC`` arrives, so this only has to cover the trip through the backend.
_ESCAPE_TAIL_MS = 15

#: The navigation keysyms worth naming. Everything else in the special range
#: still reports nothing, so an unbound function key cannot alias onto a
#: letter's binding.
_NAMED_KEYS = {
    0xFF50: "home",
    0xFF51: "left",
    0xFF52: "up",
    0xFF53: "right",
    0xFF54: "down",
    0xFF55: "pageup",
    0xFF56: "pagedown",
    0xFF57: "end",
}


def _key_char(key: int) -> str:
    """Map a backend key code to a character, or to a named key.

    The low byte is what identifies a character key: the GTK/Qt OpenCV builds
    return it with modifier and state bits set above it. Special keys, though,
    arrive as X11 keysyms (``0xFF00``-``0xFFFF``: arrows, Insert, Home, the
    function keys), whose low byte would otherwise alias onto an unrelated
    letter and trigger that letter's binding.

    The arrows and the page/home/end cluster come back as their names — a
    caller can bind ``"left"`` the way it binds ``"p"``, and a name can never
    collide with a single-character binding. Every other special key still
    reports nothing.

    Examples:
        >>> _key_char(ord("q"))
        'q'
        >>> _key_char(0xFF51)
        'left'
        >>> _key_char(0xFFC0)  # F3, unbound
        ''
        >>> _key_char(-1)
        ''

    """
    if key == -1:
        return ""
    if key in _QT_KEYS:
        return _QT_KEYS[key]
    keysym = key & 0xFFFF
    if 0xFF00 <= keysym <= 0xFFFF:
        return _NAMED_KEYS.get(keysym, "")
    return chr(key & 0xFF)


class Viewer:
    """Show `Image` frames interactively, with screen-fit and hover tooltips.

    Two interaction models, matching the backend flavor (see `WindowBackend`):

    - **pull** (the default `Cv2Backend`): call `show` for each window, then
      `wait` to block for a keypress while hover tooltips redraw; `show_blocking`
      shows a single non-interactive frame.
    - **push** (event-driven backends): call `show`, then `run` once with an
      ``on_key`` callback and return to the backend's own loop; hover tooltips
      then redraw inline on each mouse move.

    Clicking works in both models: a click on a panel control or legend swatch
    applies it, and a click on an annotation reports the data that annotation was
    drawn from — by default printing it as JSON and copying it (see ``on_pick``).

    Use `destroy_stale` to close windows no longer in use and `close` to tear
    everything down.

    Args:
        backend: The window backend to drive; defaults to `Cv2Backend`.
        hud: Whether to float the controls HUD over each interactive window. Turn
            it off when the controls are shown elsewhere (e.g. a side panel that
            already lists them, as ``luxonis_ml data inspect`` does).
        save_dir: Where the ``s`` key writes frames (see `save`), created on the
            first save. Defaults to the working directory; give it a directory
            of its own so a session's shots collect somewhere predictable
            instead of scattering over the caller's cwd.
        on_pick: What to do with the source data of a clicked annotation (see
            `Annotation.source`). ``None`` uses the default: print it as JSON
            and copy it to the clipboard. Pass a callback to route it elsewhere,
            or ``lambda source: None`` to make clicking annotations inert.

    """

    def __init__(
        self,
        backend: WindowBackend | None = None,
        *,
        hud: bool = True,
        save_dir: "str | Path | None" = None,
        on_pick: PickFn | None = None,
    ) -> None:
        self._backend: WindowBackend = (
            backend if backend is not None else Cv2Backend()
        )
        self._hud = hud
        self._save_dir = Path.cwd() if save_dir is None else Path(save_dir)
        self._on_pick: PickFn = report_pick if on_pick is None else on_pick
        self._screen = self._backend.screen_size()
        #: One key read ahead while scanning an escape sequence and
        #: found not to belong to it, returned by the next `_poll`.
        self._pushback: int | None = None
        self._windows: dict[str, _HoverState] = {}
        self._live: set[str] = set()
        # Push (event-driven) backends redraw hover inline instead of in a loop.
        self._driven = False
        # Shared layer toggles for every interactive window this viewer shows.
        self._layers = LayerState()

    @property
    def screen(self) -> tuple[int, int] | None:
        """The backend's screen size in pixels, or ``None`` if unavailable."""
        return self._screen

    @property
    def save_dir(self) -> Path:
        """Where the ``s`` key writes frames (see `save`)."""
        return self._save_dir

    @property
    def layers(self) -> LayerState:
        """The shared `LayerState` driving interactive windows.

        Configure it before showing frames — e.g. set ``layers.classes`` so the
        ``c`` key can cycle a class focus — and read it back to see the current
        toggles.
        """
        return self._layers

    def _prepare(
        self, frame: Frame
    ) -> tuple[np.ndarray, HitMap, ClickMap, PickMap]:
        """Render ``frame`` to a screen-fitted BGR image and scale its maps.

        The image is rendered once at its natural size to learn its dimensions
        (which may already reflect a `Image.render_at` size); if that overflows
        the screen it is re-rendered smaller — so labels stay crisp rather than
        being resampled afterwards — and every interaction map is scaled to match.
        """
        rgba = frame.render()
        hitmap = frame.hitmap
        clickmap = frame.clickmap
        pickmap = frame.pickmap
        out_h, out_w = rgba.shape[:2]
        fit = 1.0
        if self._screen is not None:
            fit = min(
                0.9 * self._screen[0] / out_w,
                0.9 * self._screen[1] / out_h,
                1.0,
            )
        if fit < 1.0:
            size = (max(1, round(out_w * fit)), max(1, round(out_h * fit)))
            rgba = frame.render(size)
            hitmap = hitmap.scaled(fit)
            clickmap = clickmap.scaled(fit)
            pickmap = pickmap.scaled(fit)
        return io.export(rgba, "bgr"), hitmap, clickmap, pickmap

    def _open(self, name: str, frame: np.ndarray) -> None:
        """Create (if needed), size, and center the window for ``frame``."""
        height, width = frame.shape[:2]
        if name not in self._live:
            self._backend.create_window(name)
            self._live.add(name)
        self._backend.resize(name, width, height)
        if self._screen is not None:
            self._backend.center(name, width, height, self._screen)

    def show(
        self,
        name: str,
        frame: Frame,
        *,
        render: RenderFn | None = None,
        save_as: str | None = None,
    ) -> None:
        """Present ``frame`` in window ``name`` and arm its hover tooltips.

        Args:
            name: The window identifier (created on first use).
            frame: The `Frame` to show — an image plus its hover `HitMap` (from
                `Renderable.frame`). The map is in
                the image's native pixels and is scaled internally to match the
                shown frame.
            render: Optional callback that rebuilds this window's `Frame` for a
                `LayerState`. When given, the window becomes interactive: the
                layer-control keys (see `layers`) re-render it through this
                callback and a small controls HUD is drawn on it.
            save_as: What to name this frame's file if it is saved (see `save`),
                without a suffix. A window shown once per subject should pass
                the subject's own name — its source image, say — since the
                window title stays the same across all of them.

        """
        self.show_prepared(
            name, self.prepare(frame), render=render, save_as=save_as
        )

    def prepare(self, frame: Frame) -> PreparedFrame:
        """Render and screen-fit ``frame`` without touching the window backend.

        This method is safe to run ahead of presentation. It performs the
        expensive scene work that `show_prepared` intentionally avoids.

        Args:
            frame: Scene and interaction maps to prepare.

        Returns:
            Display-ready pixels and aligned interaction maps.

        """
        bgr, hitmap, clickmap, pickmap = self._prepare(frame)
        return PreparedFrame(bgr, hitmap, clickmap, pickmap)

    def show_prepared(
        self,
        name: str,
        prepared: PreparedFrame,
        *,
        render: RenderFn | None = None,
        save_as: str | None = None,
    ) -> None:
        """Present a value returned by `prepare` without rendering it again.

        Args:
            name: The window identifier.
            prepared: Display-ready frame data.
            render: Optional callback used for interactive layer re-renders.
            save_as: What to name this frame's file if it is saved (see `show`).

        """
        bgr = prepared.bgr.copy()
        self._open(name, bgr)
        if render is not None:
            self._draw_hud(bgr)
        state = _HoverState(
            base=bgr,
            hitmap=prepared.hitmap,
            clickmap=prepared.clickmap,
            pickmap=prepared.pickmap,
            render=render,
            save_as=save_as,
        )
        self._windows[name] = state
        self._backend.set_mouse_handler(name, self._handler(name, state))
        self._backend.show(name, bgr)

    def _draw_hud(self, frame: np.ndarray) -> None:
        """Draw the controls HUD (current `LayerState`) at the frame's lower-left.

        The type size scales with the frame so the HUD stays proportional on both
        small and large (screen-fitted) windows, matching the hover tooltips. A
        no-op when the viewer was created with ``hud=False`` (controls shown
        elsewhere).
        """
        if not self._hud:
            return
        height, width = frame.shape[:2]
        size = int(min(22, max(12, round(min(width, height) / 52))))
        card = render_controls_card(self._layers.controls(), size)
        inset = round(size * 1.1)
        y = height - card.shape[0] - inset
        # Frosted-glass backdrop behind the card, matching the hover tooltips.
        blit_rgba_on_bgr(frame, card, inset, y, blur=size * 0.7)

    def _controllable(self) -> bool:
        """Whether any open window has a re-render callback (is interactive)."""
        return any(s.render is not None for s in self._windows.values())

    def _rerender_all(self) -> None:
        """Rebuild every open interactive window."""
        for name, state in self._windows.items():
            self._rerender(name, state)

    def _handle_control_key(self, char: str) -> bool:
        """Apply a viewer or layer control key; return whether it was consumed.

        The save key belongs to the viewer rather than to `LayerState`: it
        writes what is on screen instead of changing what is drawn. With no
        window open there is nothing to save, so the key falls through to the
        caller rather than being swallowed.
        """
        if char.lower() == _SAVE_KEY and self._windows:
            self._save_shown()
            return True
        return self._handle_layer_key(char)

    def _handle_layer_key(self, char: str) -> bool:
        """Apply a layer key and re-render, returning whether it was handled."""
        if not self._controllable() or not self._layers.handle(char):
            return False
        self._rerender_all()
        return True

    def _rerender(self, name: str, state: _HoverState) -> None:
        """Rebuild ``state``'s frame for the current layers and repaint it."""
        if state.render is None:
            return
        frame = state.render(self._layers)
        bgr, hitmap, clickmap, pickmap = self._prepare(frame)
        self._draw_hud(bgr)
        state.base = bgr
        state.hitmap = hitmap
        state.clickmap = clickmap
        state.pickmap = pickmap
        state.clear_hover()
        self._backend.show(name, bgr)

    def show_blocking(self, name: str, display: Renderable) -> str:
        """Show one frame (no hover) and block until a key; return its char."""
        bgr, *_ = self._prepare(Frame(display))
        self._open(name, bgr)
        self._windows.pop(name, None)
        self._backend.show(name, bgr)
        return _key_char(self._backend.poll_key(0))

    def wait(self) -> str:
        """Block for a keypress, redrawing hover tooltips; return its char.

        For pull backends (`Cv2Backend`); push backends use `run` instead.

        The poll timeout is what mouse-moves are delivered through, so it also
        paces hover redraws: it stays short while the tooltip is moving and
        relaxes as soon as the pointer settles.
        """
        poll = _IDLE_POLL_MS
        while True:
            key_result = self._handle_polled_key(self._poll(poll))
            if key_result is not None:
                handled, char = key_result
                if not handled:
                    return char
                continue
            if self._consume_pending_action() or self._consume_pending_pick():
                continue
            poll = _ACTIVE_POLL_MS if self._flush_hover() else _IDLE_POLL_MS

    def _poll(self, timeout_ms: int) -> int:
        """Poll for a key, returning any byte an escape scan put back first."""
        if self._pushback is not None:
            key, self._pushback = self._pushback, None
            return key
        return self._backend.poll_key(timeout_ms)

    def _read_escape_sequence(self) -> str | None:
        """Name the navigation key an ``ESC [ ...`` sequence stands for.

        Some builds deliver the arrows as terminal escape sequences rather than
        as key codes. Read one byte at a time, the leading ``ESC`` is
        indistinguishable from the Escape key — which is how pressing Left came
        to quit a presentation whose quit key is Escape. Peeking at the next
        byte or two tells them apart; a byte that turns out not to belong to a
        sequence is pushed back rather than swallowed.

        Returns:
            The key's name, or ``None`` when this really was a bare Escape.

        """
        following = self._poll(_ESCAPE_TAIL_MS)
        if _key_char(following) != "[":
            if following != -1:
                self._pushback = following
            return None
        final = _key_char(self._poll(_ESCAPE_TAIL_MS))
        if final in _CSI_KEYS:
            return _CSI_KEYS[final]
        if final in _CSI_TILDE_KEYS:
            self._poll(_ESCAPE_TAIL_MS)  # the sequence's trailing "~"
            return _CSI_TILDE_KEYS[final]
        return None

    def _handle_polled_key(self, key: int) -> tuple[bool, str] | None:
        """Return ``(handled, char)`` for a key, or ``None`` when none arrived."""
        if key == -1:
            return None
        char = _key_char(key)
        if char == "\x1b":
            char = self._read_escape_sequence() or "\x1b"
        return self._handle_control_key(char), char

    def _consume_pending_action(self) -> bool:
        """Apply and clear the first queued panel action, if any."""
        pending = next(
            (
                state.pending
                for state in self._windows.values()
                if state.pending is not None
            ),
            None,
        )
        if pending is None:
            return False
        for state in self._windows.values():
            state.pending = None
        self._apply_action(pending)
        return True

    def _consume_pending_pick(self) -> bool:
        """Report and clear the first queued annotation pick, if any."""
        picked = next(
            (
                state.picked
                for state in self._windows.values()
                if state.picked is not None
            ),
            None,
        )
        if picked is None:
            return False
        for state in self._windows.values():
            state.picked = None
        self._pick(picked)
        return True

    def _flush_hover(self) -> bool:
        """Redraw every window whose hover state changed, reporting whether any did."""
        redrawn = False
        for name, state in self._windows.items():
            if not state.dirty:
                continue
            state.dirty = False
            self._render_hover(name, state)
            redrawn = True
        return redrawn

    def run(self, on_key: Callable[[str], None]) -> None:
        """Enter driven mode: the backend's own event loop delivers events.

        Present frames with `show` as usual; hover tooltips then redraw inline on
        each mouse move (there is no `wait` loop), and every keypress is delivered
        to ``on_key`` as a one-character string. Returns immediately — the
        backend's native loop (Qt, a notebook kernel, a web socket) keeps calling
        back until `close`; the caller keeps that loop alive.

        Requires a push-capable backend; pull backends like `Cv2Backend` raise
        ``NotImplementedError`` (use `wait` instead).

        Args:
            on_key: Called with the pressed key's character on each keypress.

        """
        self._driven = True

        def dispatch(key: int) -> None:
            char = _key_char(key)
            if self._handle_control_key(char):
                return
            on_key(char)

        self._backend.set_key_handler(dispatch)

    def _render_hover(self, name: str, state: _HoverState) -> None:
        """Present ``state``'s base frame, with its current tooltip if any.

        The tooltip is drawn into a scratch buffer that is otherwise a copy of
        the base frame, and only the region the previous card covered is
        restored — a mouse-move repaints a card-sized patch, not a whole frame.
        """
        if state.hover is None and state.painted is None:
            self._backend.show(name, state.base)
            return
        if state.scratch is None:
            state.scratch = state.base.copy()
        frame = state.scratch
        if state.painted is not None:
            x0, y0, x1, y1 = state.painted
            frame[y0:y1, x0:x1] = state.base[y0:y1, x0:x1]
            state.painted = None
        if state.hover is not None:
            if state.card is None or state.card.tooltip is not state.hover:
                state.card = prepare_tooltip(frame, state.hover)
            if state.card is not None:
                state.painted = state.card.draw(frame, state.mouse)
        self._backend.show(name, frame)

    def save(self, directory: "str | Path | None" = None) -> list[Path]:
        """Write every open window's current frame to a PNG file.

        The frame is written exactly as displayed — hover tooltip and controls
        HUD included — so the file matches what was on screen. Bound to the
        ``s`` key, replacing the one the OpenCV Qt window used to offer.

        Each file is named after what the window is showing — the ``save_as``
        given to `show`, else the window name — with a number appended rather
        than overwriting a file already there.

        Args:
            directory: Where to write, created if missing; defaults to
                `save_dir`.

        Returns:
            The paths written, one per open window, in window order.

        Raises:
            OSError: If the directory cannot be created or a file not written.

        """
        target = self._save_dir if directory is None else Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        written = []
        for name, state in self._windows.items():
            path = _unique_path(target, _slug(state.save_as or name))
            io.save(io.load_rgba(state.shown, "bgr"), path)
            written.append(path)
        return written

    def _save_shown(self) -> None:
        """Save every window for the ``s`` key, reporting where it landed.

        A failure here (an unwritable directory, say) must not take down an
        interactive session, so it is reported rather than raised.
        """
        try:
            written = self.save()
        except OSError as error:
            logger.error(f"Could not save the current frame: {error}")
            return
        for path in written:
            logger.info(f"Saved the current view to {path}")

    def destroy_stale(self, current: set[str]) -> None:
        """Close every open window whose name is not in ``current``."""
        for name in self._live - current:
            self._backend.destroy_window(name)
            self._windows.pop(name, None)
        self._live &= current

    def close(self) -> None:
        """Destroy all windows and reset the viewer."""
        self._backend.close()
        self._live.clear()
        self._windows.clear()

    def _handler(self, name: str, state: _HoverState) -> MouseHandler:
        """Build the mouse handler tracking hover and dispatching panel clicks.

        A click on a panel control/legend region applies its action to the shared
        `LayerState` and re-renders; a move updates the hover tooltip. Pull mode
        defers the work to `wait` (via ``dirty``/``pending``); driven mode does it
        inline (there is no loop to defer to).
        """

        def handler(x: int, y: int, clicked: bool) -> None:
            if clicked:
                self._handle_click(state, x, y)
                return
            self._handle_hover(name, state, x, y)

        return handler

    def _handle_click(self, state: _HoverState, x: int, y: int) -> None:
        """Dispatch or queue what was clicked at ``(x, y)``.

        A control (panel button, legend swatch) wins over an annotation: the
        maps do not overlap in practice, but a control is chrome the click was
        certainly aimed at, whereas the picked annotation only reports itself.
        """
        action = state.clickmap.hit(x, y)
        if action is not None:
            if self._driven:
                self._apply_action(action)
            else:
                state.pending = action
            return
        source = state.pickmap.hit(x, y)
        if source is None:
            return
        if self._driven:
            self._pick(source)
        else:
            state.picked = source

    def _handle_hover(
        self,
        name: str,
        state: _HoverState,
        x: int,
        y: int,
    ) -> None:
        """Update one window's hover state and schedule its redraw."""
        tooltip = state.hitmap.hit(x, y)
        position = (int(x), int(y))
        changed = tooltip is not state.hover or (
            tooltip is not None and position != state.mouse
        )
        if not changed:
            return
        state.hover = tooltip
        state.mouse = position
        if self._driven:
            self._render_hover(name, state)
        else:
            state.dirty = True

    def _pick(self, source: "ParamValue") -> None:
        """Hand a clicked annotation's source data to the pick handler.

        Like `_save_shown`, a failure here is reported instead of raised: the
        handler is caller-supplied (and the default one talks to the clipboard),
        and neither is worth ending an interactive session over.
        """
        try:
            self._on_pick(source)
        except Exception as error:
            logger.error(f"Could not report the clicked annotation: {error}")

    def _apply_action(self, action: str) -> None:
        """Apply a panel-click ``action`` to the layers and re-render every window.

        ``"key:<k>"`` presses control key ``k`` (same as the keyboard); ``"class:
        <name>"`` toggles that class's visibility (a legend click); ``"classes:
        toggle"`` flips every class on/off (the legend's master switch).
        """
        kind, _, arg = action.partition(":")
        if kind == "key":
            if arg.lower() == _SAVE_KEY:
                self._save_shown()
                return
            self._layers.handle(arg)
        elif kind == "class":
            self._layers.toggle_class(arg)
        elif kind == "classes" and arg == "toggle":
            self._layers.toggle_all_classes()
        else:
            return
        self._rerender_all()
