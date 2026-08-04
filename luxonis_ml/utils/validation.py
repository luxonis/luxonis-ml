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
formatter tell a field name from a mapping key or from one of pydantic's
internal union member tags, and suggest the closest valid field name for a
misspelled key.

.. _rich:
    https://rich.readthedocs.io/en/stable/

"""

import difflib
import re
import sys
import types
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Union, get_args, get_origin

from pydantic import AliasChoices, AliasPath, BaseModel
from pydantic.fields import FieldInfo
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

_SUGGESTION_CUTOFF = 0.75
"""Minimum `difflib` similarity for a "did you mean" suggestion.

A misspelling scores well above this: one transposed letter in a short
name still matches it at 0.80, and a dropped one at 0.91. Merely related
names do not — ``task_name`` matches ``class_name`` at only 0.74, and
suggesting it would send the reader off to change the wrong thing.
"""

_QUOTED = re.compile(r"'([^']*)'")

_WRAPPER_TAGS = (
    "is-instance",
    "is-subclass",
    "nullable",
    "definition-ref",
    "json-or-python",
    "lax-or-strict",
    "default",
    "chain",
    "function-after",
    "function-before",
    "function-plain",
    "function-wrap",
)
"""Pydantic-internal wrappers around a union member's real type name."""

_WRAPPED_TYPE = re.compile(rf"(?:{'|'.join(_WRAPPER_TAGS)})\[(?P<inner>.+)\]")

_TAG_HEADS = frozenset(
    {
        *_WRAPPER_TAGS,
        "list",
        "dict",
        "tuple",
        "set",
        "frozenset",
        "union",
        "literal",
        "generator",
    }
)
"""Names that may precede the ``[`` of a union member tag."""

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

_OWN_NAME_ERRORS = frozenset({"missing", "extra_forbidden"})
"""Error types reported under a key of their own, never under a type tag."""


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
                record_validated_model(e, DatasetRecord)
                raise

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
        model: Model that was validated. Used to resolve error locations
            and to suggest valid field names for misspelled keys. Defaults
            to the model recorded by `record_validated_model`, if any.
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
        model: Model that was validated. Used to resolve error locations
            and to suggest valid field names for misspelled keys. Defaults
            to the model recorded by `record_validated_model`, if any.
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

    if title is None:
        title = _default_title(error, problems)

    # `fit` rather than the default: the border should hug the errors,
    # not stretch to the width of the terminal. The title goes in as
    # `Text`, so that a model name such as "list[int]" is not mistaken
    # for console markup and swallowed.
    return Panel.fit(
        Group(*_interleave(blocks, Text())),
        title=Text(title),
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
        model: Model that was validated. Used to resolve error locations
            and to suggest valid field names for misspelled keys. Defaults
            to the model recorded by `record_validated_model`, if any.

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


def install_excepthook(*, use_rich: bool = True) -> None:
    """Summarize uncaught validation errors below their traceback.

    Wraps `sys.excepthook` with one that first lets the previously
    installed hook run — so the traceback still shows where the error came
    from, and crash reporters still see it — and then prints a readable
    summary of the `ValidationError` as the last thing the reader sees.
    Calling this more than once is harmless.

    Args:
        use_rich: If True, the summary is a `rich` panel. If False, it is
            plain text.

    """
    previous = sys.excepthook
    if getattr(previous, _HOOK_ATTR, False):
        return

    def hook(
        exc_type: type[BaseException],
        exc: BaseException,
        traceback: types.TracebackType | None,
    ) -> None:
        previous(exc_type, exc, traceback)
        if not isinstance(exc, ValidationError):
            return
        if use_rich:
            Console(stderr=True).print(render_validation_error(exc))
        else:
            sys.stderr.write(f"{format_validation_error(exc)}\n")

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

    Every member of a union is validated against the same input at the
    same location, and pydantic labels each failure with the member's type
    rather than with a field name. Errors sharing such a parent location
    therefore belong together — including the ones a container member
    reports one level deeper, e.g. ``value.list[Marker][0]``.

    Args:
        errors: Raw error details, in the order pydantic reported them.
        model: Model that was validated, if known.

    Returns:
        Groups of errors, each holding either a single unrelated error or
        all the member failures of one union.

    """
    by_parent: dict[tuple[Any, ...], list[ErrorDetails]] = {}
    for error in errors:
        by_parent.setdefault(tuple(error["loc"][:-1]), []).append(error)

    unions = {
        parent
        for parent, group in by_parent.items()
        if len(group) > 1 and _is_union_group(group, model)
    }

    out: list[list[ErrorDetails]] = []
    position: dict[tuple[Any, ...], int] = {}
    for error in errors:
        parent = _union_parent(tuple(error["loc"]), unions, model)
        if parent is None:
            out.append([error])
        elif parent in position:
            out[position[parent]].append(error)
        else:
            position[parent] = len(out)
            out.append([error])
    return out


def _union_parent(
    loc: tuple[Any, ...],
    unions: set[tuple[Any, ...]],
    model: type[BaseModel] | None,
) -> tuple[Any, ...] | None:
    """Find the failed union ``loc`` reports a member failure of.

    Args:
        loc: The error location, as reported by pydantic.
        unions: Locations known to hold a failed union.
        model: Model that was validated, if known.

    Returns:
        The location of the innermost union ``loc`` belongs to, or
        ``None`` if it belongs to none.

    """
    for length in range(len(loc) - 1, -1, -1):
        parent = loc[:length]
        if parent not in unions:
            continue
        if length == len(loc) - 1:
            return parent
        tag = loc[length]
        if isinstance(tag, str) and _is_member_tag(parent, tag, model):
            return parent
    return None


def _is_member_tag(
    parent: Sequence[Any], tag: str, model: type[BaseModel] | None
) -> bool:
    """Tell whether ``tag`` names a member of the union at ``parent``."""
    annotations, _ = _resolve_location(model, parent)
    members = _union_members(annotations)
    if members:
        return any(_tag_matches(tag, member) for member in members)
    return _is_certain_type_tag(tag)


def _is_union_group(
    group: Sequence[ErrorDetails], model: type[BaseModel] | None
) -> bool:
    """Tell a failed union apart from siblings that all happen to fail.

    A shared parent location is not enough on its own: the entries of a
    mapping and the extra keys of a model share one too, and reporting
    those as the members of a union would replace every real message with
    a list of type names that are really the user's own keys.

    Args:
        group: Errors sharing a parent location.
        model: Model that was validated, if known.

    Returns:
        ``True`` if the errors are the member failures of one union.

    """
    if any(not error["loc"] for error in group):
        return False

    # A missing or forbidden key is reported under the name of the key
    # itself, so its last location component is never a member tag.
    if any(error["type"] in _OWN_NAME_ERRORS for error in group):
        return False

    tags = [tag for error in group if isinstance(tag := error["loc"][-1], str)]
    if len(tags) != len(group) or len(set(tags)) != len(tags):
        return False

    annotations, _ = _resolve_location(model, group[0]["loc"][:-1])
    members = _union_members(annotations)
    if members:
        return all(
            any(_tag_matches(tag, member) for member in members)
            for tag in tags
        )
    if annotations:
        # The location resolved to something that is not a union.
        return False

    # Nothing to resolve against, so go by the shape of the tags alone,
    # and only when at least one of them could not be anything else.
    return all(_looks_like_type_tag(tag) for tag in tags) and any(
        _is_certain_type_tag(tag) for tag in tags
    )


def _clean_type_name(tag: str) -> str:
    """Strip pydantic's internal wrappers from a union member tag.

    ``is-instance[Category]`` names the same type a user wrote as
    ``Category``, so only the inner name is worth showing. Wrappers nest
    and carry keyword-prefixed alternatives, as in
    ``lax-or-strict[lax=…,strict=is-instance[Path]]``, so the last
    alternative is unwrapped repeatedly. Genuine generics such as
    ``list[int]`` are left alone.

    Args:
        tag: The union member tag as it appears in the error location.

    Returns:
        The type name to show.

    """
    match = _WRAPPED_TYPE.fullmatch(tag)
    if match is not None:
        # A wrapper may carry the validator name or the alternative it did
        # not take first, as in "function-after[check(), Category]".
        inner = _split_top_level(match["inner"])[-1]
        return _clean_type_name(inner.partition("=")[2] or inner)

    head, bracket, rest = tag.partition("[")
    if bracket and head in _TAG_HEADS and rest.endswith("]"):
        # A generic keeps its own name but not its arguments' wrappers,
        # so "list[is-instance[Marker]]" reads as "list[Marker]".
        arguments = _split_top_level(rest[:-1])
        return f"{head}[{', '.join(map(_clean_type_name, arguments))}]"
    return tag


def _split_top_level(tag: str) -> list[str]:
    """Split a tag on the commas that are not inside brackets."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, char in enumerate(tag):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(tag[start:i].strip())
            start = i + 1
    parts.append(tag[start:].strip())
    return parts


def _is_certain_type_tag(tag: str) -> bool:
    """Tell whether ``tag`` cannot be anything but a member tag.

    Args:
        tag: A location component.

    Returns:
        ``True`` for the tags pydantic spells in a way no user key does.

    """
    if tag in _BUILTIN_TYPE_TAGS:
        return True
    head, bracket, _ = tag.partition("[")
    # A hyphen rules out a Python name, so "is-instance[...]" and friends
    # are unmistakable even though the list of them is open-ended.
    return bool(bracket) and (head in _TAG_HEADS or "-" in head)


def _looks_like_type_tag(tag: str) -> bool:
    return _is_certain_type_tag(tag) or (
        tag.isidentifier() and tag[:1].isupper()  # e.g. a model name
    )


def _to_problem(
    group: Sequence[ErrorDetails], model: type[BaseModel] | None
) -> ValidationProblem:
    if len(group) > 1:
        return _union_problem(group, model)

    error = group[0]
    loc = tuple(error["loc"])
    message, hint = _describe(error, loc, model)
    return ValidationProblem(
        location=_format_location(loc, model),
        message=message,
        hint=hint,
        value=_format_input(error),
    )


def _union_problem(
    group: Sequence[ErrorDetails], model: type[BaseModel] | None
) -> ValidationProblem:
    # Members that failed inside a container of their own report one
    # location component more than the rest; the shortest location is the
    # one that ends on a member tag.
    outermost = min(group, key=lambda error: len(error["loc"]))
    depth = len(outermost["loc"]) - 1

    alternatives: list[str] = []
    for error in group:
        name = _clean_type_name(str(error["loc"][depth]))
        if name not in alternatives:
            alternatives.append(name)

    return ValidationProblem(
        location=_format_location(tuple(outermost["loc"][:depth]), model),
        message="does not match any of the allowed types: "
        + ", ".join(alternatives),
        value=_format_input(outermost, always=True),
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
        # A bare `assert x > 0` or `raise ValueError()` carries no message
        # of its own, in which case pydantic's own text is all there is.
        detail = str(ctx.get("error", "")).strip()
        if detail:
            return detail, None
        return _lower_first(error["msg"].strip(", ")), None

    # Pydantic's own message is already short and specific for the many
    # remaining types, e.g. "Input should be a valid integer".
    return _lower_first(error["msg"]), None


def _lower_first(message: str) -> str:
    """Lowercase the first word of a message unless it is an acronym.

    A second capital means the first one is not the start of an ordinary
    sentence but of a name such as ``UUID`` or ``JSON``, which would read
    as a typo if it were lowercased.

    Args:
        message: The message to adjust.

    Returns:
        The message, starting lowercase where that is safe.

    """
    if message[1:2].isupper():
        return message
    return message[:1].lower() + message[1:]


def _format_location(loc: Sequence[Any], model: type[BaseModel] | None) -> str:
    """Render an error location as a path the user can act on.

    Args:
        loc: The error location, as reported by pydantic.
        model: Model that was validated, if known.

    Returns:
        A dotted path with sequence indices as subscripts, leaving out
        the union member tags pydantic puts in the middle of a location.

    """
    _, is_tag = _resolve_location(model, loc)
    parts: list[str] = []
    for part, tag in zip(loc, is_tag, strict=True):
        if tag:
            continue
        if isinstance(part, int):
            if parts:
                parts[-1] += f"[{part}]"
            else:
                parts.append(f"[{part}]")
        else:
            parts.append(str(part))
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
    if not always and error["type"] in _OWN_NAME_ERRORS:
        return None

    value = repr(error["input"])
    if len(value) > _MAX_VALUE_LENGTH:
        # Keep both ends: what tells two long paths apart is their tail.
        head = (_MAX_VALUE_LENGTH - 1) // 2
        tail = _MAX_VALUE_LENGTH - 1 - head
        value = f"{value[:head]}…{value[-tail:]}"
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
    """Return the keys accepted at ``loc`` inside ``model``.

    Args:
        model: The root model, or ``None`` if it is unknown.
        loc: Location of the *enclosing* value, i.e. the error location
            without its final component.

    Returns:
        Every key accepted at that location, using validation aliases
        where a field has them. Empty if the location cannot be resolved.

    """
    annotations, _ = _resolve_location(model, loc)
    names: list[str] = []
    for annotation in annotations:
        for candidate in _models_in(annotation):
            for name in _accepted_keys(candidate):
                if name not in names:
                    names.append(name)
    return names


def _accepted_keys(model: type[BaseModel]) -> list[str]:
    """Return the keys ``model`` accepts, aliases included.

    A field with a validation alias is only reachable under that alias,
    unless the model also allows population by field name, so suggesting
    the field name would be suggesting a key that is still rejected.

    Args:
        model: The model to inspect.

    Returns:
        Every accepted key, in field declaration order.

    """
    by_name = model.model_config.get("populate_by_name", False)
    keys: list[str] = []
    for name, field in model.model_fields.items():
        aliases = _validation_aliases(field)
        for key in aliases or [name]:
            if key not in keys:
                keys.append(key)
        if aliases and by_name and name not in keys:
            keys.append(name)
    return keys


def _validation_aliases(field: FieldInfo) -> list[str]:
    """Return the top-level keys ``field`` may be given under."""
    alias = field.validation_alias
    if alias is None:
        return []
    if isinstance(alias, str):
        return [alias]
    choices = alias.choices if isinstance(alias, AliasChoices) else [alias]
    keys: list[str] = []
    for choice in choices:
        if isinstance(choice, str):
            keys.append(choice)
        elif isinstance(choice, AliasPath) and choice.path:
            keys.append(str(choice.path[0]))
    return keys


def _resolve_location(
    model: type[BaseModel] | None, loc: Sequence[Any]
) -> tuple[list[Any], list[bool]]:
    """Walk ``loc`` through ``model``, classifying every component.

    A location component is a field name, a sequence index, a mapping key
    or one of pydantic's union member tags, and only the annotations along
    the way can tell which. Resolution stops at the first component that
    cannot be matched; from there on components are judged by their shape
    alone.

    Args:
        model: The root model, or ``None`` if it is unknown.
        loc: The location to walk.

    Returns:
        The annotations reachable at ``loc``, and one flag per component
        saying whether it is a union member tag rather than user data.

    """
    annotations: list[Any] = [model] if model is not None else []
    is_tag: list[bool] = []
    for part in loc:
        if isinstance(part, int):
            is_tag.append(False)
            annotations = _element_annotations(annotations)
            continue
        if not annotations:
            is_tag.append(_is_certain_type_tag(part))
            continue

        members = _matching_members(annotations, part)
        if members is not None:
            is_tag.append(True)
            annotations = members
            continue

        is_tag.append(False)
        # A mapping key can be spelled like anything, including like a
        # field of the mapping's own value model, so mappings win.
        annotations = _mapping_values(annotations) or _field_annotations(
            annotations, part
        )
    return annotations, is_tag


def _union_members(annotations: Sequence[Any]) -> list[Any]:
    """Return every member of every union among ``annotations``."""
    return [
        member
        for annotation in annotations
        if _is_union(annotation)
        for member in get_args(annotation)
    ]


def _matching_members(
    annotations: Sequence[Any], tag: str
) -> list[Any] | None:
    """Return the union members ``tag`` names, or ``None`` if it names none."""
    matched = [
        member
        for member in _union_members(annotations)
        if _tag_matches(tag, member)
    ]
    return matched or None


def _field_annotations(annotations: Sequence[Any], key: str) -> list[Any]:
    """Return the annotations of the field accepted under ``key``."""
    out: list[Any] = []
    for annotation in annotations:
        for model in _models_in(annotation):
            field = _field_for_key(model, key)
            if field is not None:
                out.append(field.annotation)
    return out


def _field_for_key(model: type[BaseModel], key: str) -> FieldInfo | None:
    by_name = model.model_config.get("populate_by_name", False)
    for name, field in model.model_fields.items():
        aliases = _validation_aliases(field)
        if key in aliases or (key == name and (not aliases or by_name)):
            return field
    return None


def _mapping_values(annotations: Sequence[Any]) -> list[Any]:
    """Return the value annotations of every mapping in ``annotations``."""
    out: list[Any] = []
    for annotation in annotations:
        for candidate in _flatten_union(annotation):
            args = get_args(candidate)
            if len(args) == 2 and _origin_is(candidate, Mapping):
                out.append(args[1])
    return out


def _element_annotations(annotations: Sequence[Any]) -> list[Any]:
    """Return the element annotations of every sequence in ``annotations``."""
    out: list[Any] = []
    for annotation in annotations:
        for candidate in _flatten_union(annotation):
            args = get_args(candidate)
            if args and _origin_is(candidate, Sequence):
                out.append(args[0])
            else:
                # Not a sequence we can look into, e.g. a model that
                # pydantic indexed for reasons of its own.
                out.append(candidate)
    return out


def _origin_is(annotation: Any, protocol: type) -> bool:
    origin = get_origin(annotation)
    return (
        isinstance(origin, type)
        and issubclass(origin, protocol)
        and not issubclass(origin, (str, bytes))
    )


def _is_union(annotation: Any) -> bool:
    return get_origin(annotation) in {Union, types.UnionType}


def _flatten_union(annotation: Any) -> list[Any]:
    if _is_union(annotation):
        return list(get_args(annotation))
    return [annotation]


def _annotation_head(annotation: Any) -> str | None:
    """Return the name pydantic would tag ``annotation`` with."""
    if annotation is None or annotation is type(None):
        return "none"
    origin = get_origin(annotation)
    if origin is not None:
        return getattr(origin, "__name__", None) or getattr(
            origin, "_name", None
        )
    if isinstance(annotation, type):
        return annotation.__name__
    return None


def _tag_matches(tag: str, annotation: Any) -> bool:
    """Tell whether ``tag`` is how pydantic would name ``annotation``."""
    head = _annotation_head(annotation)
    if head is None:
        return False
    return _clean_type_name(tag).partition("[")[0].lower() == head.lower()


def _models_in(annotation: Any) -> list[type[BaseModel]]:
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
    if _is_union(annotation) or get_args(annotation):
        return [
            model
            for argument in get_args(annotation)
            for model in _models_in(argument)
        ]
    return []
