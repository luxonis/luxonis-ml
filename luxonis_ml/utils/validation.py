"""Human-readable rendering of pydantic validation errors.

Pydantic's default `ValidationError` message is accurate but noisy: it
repeats the raw error type, the truncated input value, and a documentation
URL for every problem, and it reports union failures once per union member.
This module turns such an error into a short list of `ValidationProblem`
entries and renders them either as plain text or as a `rich`_ panel.

Example:
    .. python::

        from pydantic import ValidationError
        from rich.console import Console

        from luxonis_ml.utils.validation import render_validation_error

        try:
            MyModel(**data)
        except ValidationError as e:
            Console().print(render_validation_error(e))

The model that failed is recovered from the error's own traceback. Knowing
it lets the formatter tell a field name from a mapping key or from one of
pydantic's internal union member tags, and suggest the closest valid field
name for a misspelled key. Pass ``model`` explicitly for an error that
arrives without its traceback, or one raised by a `pydantic.TypeAdapter`,
which validates something that need not be a model at all.

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

from pydantic import AliasChoices, AliasPath, BaseModel, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import ErrorDetails
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

__all__ = [
    "ValidationProblem",
    "format_validation_error",
    "install_excepthook",
    "iter_validation_problems",
    "render_validation_error",
]

_HOOK_ATTR = "_luxonis_validation_hook"
_MAX_VALUE_LENGTH = 70
_SUGGESTION_CUTOFF = 0.75
_QUOTED = re.compile(r"'([^']*)'")

_WRAPPER_TAGS = (
    "is-instance",
    "is-subclass",
    "nullable",
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
_BUILTIN_TYPE_TAGS = frozenset(
    {
        "int",
        "float",
        "str",
        "bytes",
        "bool",
        "date",
        "time",
        "datetime",
        "timedelta",
        "decimal",
        "uuid",
    }
)
_OWN_NAME_ERRORS = frozenset({"missing", "extra_forbidden"})


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
            to the model recovered from the error's traceback.
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
            to the model recovered from the error's traceback.
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

        if blocks:
            blocks.append(Text())
        if not problem.location:
            blocks.append(body)
            continue
        blocks.append(
            Group(
                Text(problem.location, style="bold cyan"),
                Padding(body, (0, 0, 0, 2)),
            )
        )

    if title is None:
        title = _default_title(error, problems)

    return Panel.fit(
        Group(*blocks),
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
            to the model recovered from the error's traceback.

    Yields:
        One problem per distinct validation failure.

    """
    model = model or _validated_model(error)
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


def _validated_model(error: ValidationError) -> type[BaseModel] | None:
    """Recover the model that raised ``error`` from its traceback.

    Pydantic enters its Rust validator from a single Python frame, and that
    frame still holds the model — as ``cls`` for `BaseModel.model_validate`
    and its siblings, as ``self`` for ``Model(**data)``. Only pydantic's own
    frames are searched, so a model that merely happens to be the ``self``
    of a calling method cannot be mistaken for the one being validated, and
    the innermost match wins. A `pydantic.TypeAdapter` yields nothing, since
    what it validates need not be a model at all.
    """
    found: type[BaseModel] | None = None
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_globals.get("__name__", "").startswith("pydantic"):
            for name in ("cls", "self"):
                owner = frame.f_locals.get(name)
                if isinstance(owner, BaseModel):
                    owner = type(owner)
                if isinstance(owner, type) and issubclass(owner, BaseModel):
                    found = owner
                    break
        traceback = traceback.tb_next
    return found


def _default_title(
    error: ValidationError, problems: Sequence[ValidationProblem]
) -> str:
    count = len(problems)
    plural = "problem" if count == 1 else "problems"
    return f"Invalid {error.title} — {count} {plural} found"


def _group_union_errors(
    errors: Sequence[ErrorDetails],
    model: type[BaseModel] | None,
) -> list[list[ErrorDetails]]:
    """Group failures reported for members of the same union."""
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
    annotations, _ = _resolve_location(model, parent)
    members = _union_members(annotations)
    if members:
        return any(_tag_matches(tag, member) for member in members)
    return _is_certain_type_tag(tag)


def _is_union_group(
    group: Sequence[ErrorDetails], model: type[BaseModel] | None
) -> bool:
    if any(error["type"] in _OWN_NAME_ERRORS for error in group):
        return False

    tags = [
        tag
        for error in group
        for tag in error["loc"][-1:]
        if isinstance(tag, str)
    ]
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
        return False

    return all(_looks_like_type_tag(tag) for tag in tags) and any(
        _is_certain_type_tag(tag) for tag in tags
    )


def _clean_type_name(tag: str) -> str:
    match = _WRAPPED_TYPE.fullmatch(tag)
    if match is not None:
        inner = _split_top_level(match["inner"])[-1]
        return _clean_type_name(inner.partition("=")[2] or inner)

    head, bracket, rest = tag.partition("[")
    if bracket and head in _TAG_HEADS and rest.endswith("]"):
        arguments = _split_top_level(rest[:-1])
        return f"{head}[{', '.join(map(_clean_type_name, arguments))}]"
    return tag


def _split_top_level(tag: str) -> list[str]:
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
    if tag in _BUILTIN_TYPE_TAGS:
        return True
    head, bracket, _ = tag.partition("[")
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
        value=_format_input(outermost),
    )


def _describe(
    error: ErrorDetails,
    loc: tuple[Any, ...],
    model: type[BaseModel] | None,
) -> tuple[str, str | None]:
    """Phrase one error as a message and an optional hint."""
    error_type = error["type"]
    ctx = error.get("ctx") or {}

    if error_type == "missing":
        return "this field is required, but is missing", None

    if error_type == "extra_forbidden":
        name = str(loc[-1])
        return (
            f"unexpected field {name!r}",
            _suggest(name, _valid_field_names(model, loc[:-1])),
        )

    if error_type in {"literal_error", "enum"}:
        expected = str(ctx.get("expected", "")).strip()
        message = f"expected {expected}" if expected else error["msg"]
        return message, _suggest(error.get("input"), _QUOTED.findall(expected))

    if error_type in {"value_error", "assertion_error"}:
        detail = str(ctx.get("error", "")).strip()
        if detail:
            return detail, None
        return _lower_first(error["msg"].strip(", ")), None

    return _lower_first(error["msg"]), None


def _lower_first(message: str) -> str:
    if message[1:2].isupper():
        return message
    return message[:1].lower() + message[1:]


def _format_location(loc: Sequence[Any], model: type[BaseModel] | None) -> str:
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


def _format_input(error: ErrorDetails) -> str | None:
    if error["type"] in _OWN_NAME_ERRORS:
        return None

    value = repr(error["input"])
    if len(value) > _MAX_VALUE_LENGTH:
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
    annotations, _ = _resolve_location(model, loc)
    keys: list[str] = []
    for annotation in annotations:
        for candidate in _models_in(annotation):
            keys.extend(_accepted_fields(candidate))
    return list(dict.fromkeys(keys))


def _accepted_fields(model: type[BaseModel]) -> dict[str, FieldInfo]:
    by_name = model.model_config.get("populate_by_name", False)
    fields: dict[str, FieldInfo] = {}
    for name, field in model.model_fields.items():
        aliases = _validation_aliases(field)
        for key in aliases or [name]:
            fields.setdefault(key, field)
        if aliases and by_name:
            fields.setdefault(name, field)
    return fields


def _validation_aliases(field: FieldInfo) -> list[str]:
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
        if members:
            is_tag.append(True)
            annotations = members
            continue

        is_tag.append(False)
        annotations = _mapping_values(annotations) or _field_annotations(
            annotations, part
        )
    return annotations, is_tag


def _union_members(annotations: Sequence[Any]) -> list[Any]:
    return [
        member
        for annotation in annotations
        if _is_union(annotation)
        for member in get_args(annotation)
    ]


def _matching_members(annotations: Sequence[Any], tag: str) -> list[Any]:
    return [
        member
        for member in _union_members(annotations)
        if _tag_matches(tag, member)
    ]


def _field_annotations(annotations: Sequence[Any], key: str) -> list[Any]:
    out: list[Any] = []
    for annotation in annotations:
        for model in _models_in(annotation):
            field = _accepted_fields(model).get(key)
            if field is not None:
                out.append(field.annotation)
    return out


def _mapping_values(annotations: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for annotation in annotations:
        for candidate in _flatten_union(annotation):
            args = get_args(candidate)
            if len(args) == 2 and _origin_is(candidate, Mapping):
                out.append(args[1])
    return out


def _element_annotations(annotations: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for annotation in annotations:
        for candidate in _flatten_union(annotation):
            args = get_args(candidate)
            if args and _origin_is(candidate, Sequence):
                out.append(args[0])
            else:
                out.append(candidate)
    return out


def _origin_is(annotation: Any, base: type) -> bool:
    origin = get_origin(annotation)
    return (
        isinstance(origin, type)
        and issubclass(origin, base)
        and not issubclass(origin, (str, bytes))
    )


def _is_union(annotation: Any) -> bool:
    return get_origin(annotation) in {Union, types.UnionType}


def _flatten_union(annotation: Any) -> list[Any]:
    if _is_union(annotation):
        return list(get_args(annotation))
    return [annotation]


def _annotation_head(annotation: Any) -> str | None:
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
    head = _annotation_head(annotation)
    if head is None:
        return False
    return _clean_type_name(tag).partition("[")[0].lower() == head.lower()


def _models_in(annotation: Any) -> list[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    if _is_union(annotation) or get_args(annotation):
        return [
            model
            for argument in get_args(annotation)
            for model in _models_in(argument)
        ]
    return []
