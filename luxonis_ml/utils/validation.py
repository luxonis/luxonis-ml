"""Human-readable rendering of pydantic validation errors.

Pydantic's default `ValidationError` message is accurate but noisy: it
repeats the raw error type, the truncated input value, and a documentation
URL for every problem, and it reports union failures once per union member.
This module turns such an error into a short list of `ValidationProblem`
entries and renders them either as plain text or as a `rich`_ panel.

Example:
    Render a failed validation as a panel::

        from pydantic import ValidationError
        from rich.console import Console

        from luxonis_ml.utils.validation import render_validation_error

        try:
            MyModel(**data)
        except ValidationError as e:
            Console().print(render_validation_error(e, model=MyModel))

Passing ``model`` is optional but recommended: knowing the model lets the
formatter suggest the closest valid field name for a misspelled key.

.. _rich:
    https://rich.readthedocs.io/en/stable/

"""

import difflib
import re
import sys
import types
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import ErrorDetails, ValidationError
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

__all__ = [
    "ValidationProblem",
    "format_validation_error",
    "install_excepthook",
    "iter_validation_problems",
    "record_validated_model",
    "render_validation_error",
]

_MODEL_ATTR = "_luxonis_validated_model"
"""Attribute used to remember which model a `ValidationError` came from."""

_HOOK_ATTR = "_luxonis_validation_hook"
"""Marker telling an already installed exception hook from any other."""

_MAX_VALUE_LENGTH = 70
"""Longest input value repr shown before it is truncated."""

_SUGGESTION_CUTOFF = 0.6
"""Minimum `difflib` similarity for a "did you mean" suggestion."""

_QUOTED = re.compile(r"'([^']*)'")

_WRAPPED_TYPE = re.compile(
    r"(?:is-instance|is-subclass|nullable|definition-ref"
    r"|json-or-python|lax-or-strict|default"
    r"|function-(?:after|before|plain|wrap))\[(?P<inner>.+)\]"
)
"""Pydantic-internal wrappers around a union member's real type name."""

_BUILTIN_TYPE_TAGS = frozenset(
    {
        "int",
        "float",
        "str",
        "bytes",
        "bool",
        "list",
        "tuple",
        "set",
        "frozenset",
        "dict",
        "none",
        "null",
        "date",
        "time",
        "datetime",
        "timedelta",
        "decimal",
        "uuid",
        "path",
    }
)
"""Union member tags pydantic uses for non-model types."""


@dataclass(frozen=True)
class ValidationProblem:
    """A single validation failure, phrased for a human reader.

    Attributes:
        location: Dotted path to the offending value, using ``[i]`` for
            sequence indices, e.g. ``model.inputs[0].dtype``. Empty for a
            problem concerning the top-level value itself.
        message: What is wrong, as a lowercase sentence fragment.
        hint: Optional follow-up suggestion, such as the closest matching
            field name for a misspelled key.
        value: Optional repr of the offending input value.

    """

    location: str
    message: str
    hint: str | None = None
    value: str | None = None


def record_validated_model(
    error: ValidationError, model: type[BaseModel]
) -> ValidationError:
    """Remember which model produced ``error`` and return it unchanged.

    Call sites that catch a `ValidationError` only to re-raise it can use
    this to pass the model along, so that a later `format_validation_error`
    call, including the one made by the exception hook, can suggest valid
    field names.

    Args:
        error: The caught validation error.
        model: The model class that was being validated.

    Returns:
        The very same ``error`` object, so it can be re-raised directly.

    Example:
        .. code-block:: python

            try:
                record = DatasetRecord(**data)
            except ValidationError as e:
                raise record_validated_model(e, DatasetRecord) from None

    """
    setattr(error, _MODEL_ATTR, model)
    return error


def format_validation_error(
    error: ValidationError,
    *,
    model: type[BaseModel] | None = None,
    title: str | None = None,
) -> str:
    """Format a validation error as readable plain text.

    Args:
        error: The validation error to format.
        model: Model that was validated. Used to suggest valid field names
            for misspelled keys. Defaults to the model recorded by
            `record_validated_model`, if any.
        title: Headline for the message. Defaults to a line naming the
            model that failed to validate.

    Returns:
        A multi-line message with one indented block per problem.

    """
    problems = list(iter_validation_problems(error, model=model))
    lines = [title if title is not None else _default_title(error, problems)]
    for problem in problems:
        lines.append("")
        if problem.location:
            lines.append(f"  {problem.location}")
            prefix = "    "
        else:
            prefix = "  "
        lines.append(f"{prefix}{problem.message}")
        if problem.value is not None:
            lines.append(f"{prefix}got: {problem.value}")
        if problem.hint is not None:
            lines.append(f"{prefix}{problem.hint}")
    return "\n".join(lines)


def render_validation_error(
    error: ValidationError,
    *,
    model: type[BaseModel] | None = None,
    title: str | None = None,
) -> RenderableType:
    """Render a validation error as a `rich`_ panel.

    Args:
        error: The validation error to render.
        model: Model that was validated. Used to suggest valid field names
            for misspelled keys. Defaults to the model recorded by
            `record_validated_model`, if any.
        title: Headline shown in the panel border. Defaults to a line
            naming the model that failed to validate.

    Returns:
        A renderable suitable for `rich.console.Console.print`.

    .. _rich:
        https://rich.readthedocs.io/en/stable/

    """
    problems = list(iter_validation_problems(error, model=model))
    blocks: list[RenderableType] = []
    for problem in problems:
        body = Text(problem.message)
        if problem.value is not None:
            body.append("\ngot: ", style="dim")
            body.append(problem.value, style="yellow")
        if problem.hint is not None:
            body.append(f"\n{problem.hint}", style="green")

        if not problem.location:
            blocks.append(body)
            continue
        blocks.append(
            Group(
                Text(problem.location, style="bold cyan"),
                # Indent as a renderable rather than with spaces, so that
                # wrapped lines stay aligned under the first one.
                Padding(body, (0, 0, 0, 2)),
            )
        )

    # `fit` rather than the default: the border should hug the errors,
    # not stretch to the width of the terminal.
    return Panel.fit(
        Group(*_interleave(blocks, Text())),
        title=title if title is not None else _default_title(error, problems),
        title_align="left",
        border_style="red",
        padding=(1, 2),
    )


def iter_validation_problems(
    error: ValidationError,
    *,
    model: type[BaseModel] | None = None,
) -> Iterator[ValidationProblem]:
    """Convert a validation error into humanized problems.

    Union failures reported once per union member are collapsed into a
    single problem, and duplicate problems are dropped.

    Args:
        error: The validation error to inspect.
        model: Model that was validated. Used to suggest valid field names
            for misspelled keys. Defaults to the model recorded by
            `record_validated_model`, if any.

    Yields:
        One problem per distinct validation failure.

    """
    model = model or getattr(error, _MODEL_ATTR, None)
    seen: set[tuple[str, str]] = set()
    for group in _group_union_errors(error.errors(), model):
        problem = _to_problem(group, model)
        key = (problem.location, problem.message)
        if key not in seen:
            seen.add(key)
            yield problem


def install_excepthook() -> None:  # pragma: no cover
    """Print uncaught validation errors as a panel instead of a traceback.

    Replaces `sys.excepthook` with one that renders `ValidationError` using
    `render_validation_error` and delegates every other exception to the
    previously installed hook. Calling this more than once is harmless.

    """
    previous = sys.excepthook
    if getattr(previous, _HOOK_ATTR, False):
        return

    def hook(
        exc_type: type[BaseException],
        exc: BaseException,
        traceback: types.TracebackType | None,
    ) -> None:
        if not isinstance(exc, ValidationError):
            previous(exc_type, exc, traceback)
            return
        Console(stderr=True).print(render_validation_error(exc))

    setattr(hook, _HOOK_ATTR, True)
    sys.excepthook = hook


def _default_title(
    error: ValidationError, problems: Sequence[ValidationProblem]
) -> str:
    count = len(problems)
    plural = "problem" if count == 1 else "problems"
    return f"Invalid {error.title} — {count} {plural} found"


def _interleave(
    blocks: Sequence[RenderableType], separator: RenderableType
) -> list[RenderableType]:
    out: list[RenderableType] = []
    for i, block in enumerate(blocks):
        if i:
            out.append(separator)
        out.append(block)
    return out


def _group_union_errors(
    errors: Sequence[ErrorDetails],
    model: type[BaseModel] | None,
) -> list[list[ErrorDetails]]:
    """Group the errors produced by the members of a failed union.

    Every member of a union is validated against the *same* input at the
    *same* location, so errors sharing both are candidates for grouping.

    Args:
        errors: Raw error details, in the order pydantic reported them.
        model: Model that was validated, if known.

    Returns:
        Groups of errors, each holding either a single unrelated error or
        all the member failures of one union.

    """
    groups: dict[tuple[tuple[Any, ...], int], list[ErrorDetails]] = {}
    order: list[tuple[tuple[Any, ...], int]] = []
    for error in errors:
        key = (tuple(error["loc"][:-1]), id(error.get("input")))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(error)

    out: list[list[ErrorDetails]] = []
    for key in order:
        group = groups[key]
        if len(group) > 1 and _is_union_group(group, model):
            out.append(group)
        else:
            out.extend([error] for error in group)
    return out


def _is_union_group(
    group: Sequence[ErrorDetails], model: type[BaseModel] | None
) -> bool:
    """Tell a failed union apart from sibling fields that both failed.

    Sharing an input is not enough on its own: two sibling fields given
    the same bad value report the same ``input`` object, because equal
    small strings and integers are interned.

    Args:
        group: Errors sharing a parent location and an input value.
        model: Model that was validated, if known.

    Returns:
        ``True`` if the errors are the member failures of one union.

    """
    if any(not error["loc"] for error in group):
        return False

    tags = [tag for error in group if isinstance(tag := error["loc"][-1], str)]
    if len(tags) != len(group) or len(set(tags)) != len(tags):
        return False

    # Union members are labelled with the *type* they failed to match,
    # never with a field name, so knowing the fields settles it.
    fields = set(_valid_field_names(model, group[0]["loc"][:-1]))
    if fields:
        return fields.isdisjoint(tags)

    return all(_looks_like_type_name(tag) for tag in tags)


def _clean_type_name(tag: str) -> str:
    """Strip pydantic's internal wrappers from a union member tag.

    ``is-instance[Category]`` names the same type a user wrote as
    ``Category``, so only the inner name is worth showing. Genuine
    generics such as ``list[int]`` are left alone.

    Args:
        tag: The union member tag as it appears in the error location.

    Returns:
        The type name to show.

    """
    match = _WRAPPED_TYPE.fullmatch(tag)
    if match is None:
        return tag
    # A wrapper may carry the validator name first, as in
    # "function-after[check(), Category]".
    return match["inner"].rsplit(", ", 1)[-1]


def _looks_like_type_name(tag: str) -> bool:
    return (
        tag in _BUILTIN_TYPE_TAGS
        or not tag.isidentifier()  # e.g. "list[int]", "function-after[...]"
        or tag[:1].isupper()  # e.g. a model name
    )


def _to_problem(
    group: Sequence[ErrorDetails], model: type[BaseModel] | None
) -> ValidationProblem:
    if len(group) > 1:
        return _union_problem(group)

    error = group[0]
    loc = tuple(error["loc"])
    message, hint = _describe(error, loc, model)
    return ValidationProblem(
        location=_format_location(loc),
        message=message,
        hint=hint,
        value=_format_input(error),
    )


def _union_problem(group: Sequence[ErrorDetails]) -> ValidationProblem:
    alternatives = ", ".join(
        _clean_type_name(str(error["loc"][-1]))
        for error in group
        if error["loc"]
    )
    return ValidationProblem(
        location=_format_location(tuple(group[0]["loc"][:-1])),
        message=f"does not match any of the allowed types: {alternatives}",
        value=_format_input(group[0], always=True),
    )


def _describe(
    error: ErrorDetails,
    loc: tuple[Any, ...],
    model: type[BaseModel] | None,
) -> tuple[str, str | None]:
    """Phrase one error as a message and an optional hint.

    Args:
        error: The raw error details.
        loc: The error location, as reported by pydantic.
        model: Model that was validated, if known.

    Returns:
        A ``(message, hint)`` pair.

    """
    error_type = error["type"]
    ctx = error.get("ctx") or {}

    if error_type == "missing":
        return "this field is required, but is missing", None

    if error_type == "extra_forbidden":
        name = str(loc[-1]) if loc else "?"
        return (
            f"unexpected field {name!r}",
            _suggest(name, _valid_field_names(model, loc[:-1])),
        )

    if error_type in {"literal_error", "enum"}:
        expected = str(ctx.get("expected", "")).strip()
        message = f"expected {expected}" if expected else error["msg"]
        return message, _suggest(error.get("input"), _QUOTED.findall(expected))

    if error_type in {"value_error", "assertion_error"}:
        return str(ctx.get("error", error["msg"])), None

    # Pydantic's own message is already short and specific for the many
    # remaining types, e.g. "Input should be a valid integer".
    message = error["msg"]
    return message[:1].lower() + message[1:], None


def _format_location(loc: Sequence[Any]) -> str:
    parts: list[str] = []
    for part in loc:
        if isinstance(part, int):
            if parts:
                parts[-1] += f"[{part}]"
            else:
                parts.append(f"[{part}]")
        elif "[" in part:
            # A pydantic-internal marker such as "function-after[...]".
            continue
        else:
            parts.append(part)
    return ".".join(parts)


def _format_input(error: ErrorDetails, *, always: bool = False) -> str | None:
    """Return a short repr of the offending value, if it is informative.

    Args:
        error: The raw error details.
        always: Show the value even for error types whose input is the
            enclosing container rather than the offending value.

    Returns:
        A truncated repr, or ``None`` when the value adds nothing.

    """
    if not always and error["type"] in {"missing", "extra_forbidden"}:
        return None

    value = repr(error["input"])
    if len(value) > _MAX_VALUE_LENGTH:
        value = f"{value[: _MAX_VALUE_LENGTH - 1]}…"
    return value


def _suggest(value: Any, candidates: Sequence[str]) -> str | None:
    if not isinstance(value, str) or not candidates:
        return None
    matches = difflib.get_close_matches(
        value, candidates, n=1, cutoff=_SUGGESTION_CUTOFF
    )
    if not matches:
        return None
    return f"did you mean {matches[0]!r}?"


def _valid_field_names(
    model: type[BaseModel] | None, loc: Sequence[Any]
) -> list[str]:
    """Return the field names accepted at ``loc`` inside ``model``.

    Args:
        model: The root model, or ``None`` if it is unknown.
        loc: Location of the *enclosing* value, i.e. the error location
            without its final component.

    Returns:
        Every field name accepted at that location. Empty if the location
        cannot be resolved.

    """
    if model is None:
        return []

    models = [model]
    for part in loc:
        if isinstance(part, int):
            # Indexing a sequence does not change the element model.
            continue
        nested: list[type[BaseModel]] = []
        for candidate in models:
            field = candidate.model_fields.get(part)
            if field is not None:
                nested.extend(_nested_models(field.annotation))
        if not nested:
            return []
        models = nested

    return [name for candidate in models for name in candidate.model_fields]


def _nested_models(annotation: Any) -> list[type[BaseModel]]:
    """Collect the models reachable from a type annotation.

    Containers and unions are unwrapped, so ``list[Input] | None`` yields
    ``[Input]``.

    Args:
        annotation: The annotation to inspect.

    Returns:
        Every `pydantic.BaseModel` subclass found inside ``annotation``.

    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    if get_origin(annotation) in {Union, types.UnionType} or get_args(
        annotation
    ):
        return [
            model
            for argument in get_args(annotation)
            for model in _nested_models(argument)
        ]
    return []
