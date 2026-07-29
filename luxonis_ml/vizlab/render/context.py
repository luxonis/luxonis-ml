"""Resolved ambient state for one complete render."""

from collections.abc import Mapping
from dataclasses import dataclass

from luxonis_ml.vizlab.style import (
    Style,
    StyleValue,
    current_default_style,
    current_style_overrides,
)


@dataclass(frozen=True)
class RenderEnvironment:
    """A snapshot of scoped style state at the render boundary.

    Rendering resolves ambient style once and threads this immutable value
    through the whole scene. An annotation therefore never observes a different
    scope halfway through a render, and cache keys depend on the same state that
    drawing consumes.
    """

    default_style: Style | None
    style_overrides: Mapping[str, StyleValue]

    @classmethod
    def current(cls) -> "RenderEnvironment":
        """Capture the style state in effect for the current context."""
        return cls(
            default_style=current_default_style(),
            style_overrides=dict(current_style_overrides()),
        )
