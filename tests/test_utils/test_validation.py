from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from rich.console import Console

from luxonis_ml.utils.validation import (
    format_validation_error,
    iter_validation_problems,
    record_validated_model,
    render_validation_error,
)


class Resize(BaseModel):
    model_config = ConfigDict(extra="forbid")

    height: int = 256
    width: int = 256


class Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    epochs: int = 10
    precision: Literal["float16", "float32"] = "float32"
    resize: Resize = Resize()
    stages: list[Resize] = []


def catch(model: type[BaseModel], **data) -> ValidationError:
    with pytest.raises(ValidationError) as info:
        model(**data)
    return info.value


def render(error: ValidationError, **kwargs) -> str:
    console = Console(width=100, no_color=True, legacy_windows=False)
    with console.capture() as capture:
        console.print(render_validation_error(error, **kwargs))
    return capture.get()


def test_missing_field():
    message = format_validation_error(catch(Cfg), model=Cfg)
    assert "Invalid Cfg — 1 problem found" in message
    assert "name" in message
    assert "this field is required, but is missing" in message


def test_drops_pydantic_noise():
    message = format_validation_error(catch(Cfg, name="a", epochs="x"))
    assert "errors.pydantic.dev" not in message
    assert "[type=" not in message
    assert "input_type=" not in message


def test_suggests_closest_field_name():
    message = format_validation_error(catch(Cfg, name="a", epocs=1), model=Cfg)
    assert "unexpected field 'epocs'" in message
    assert "did you mean 'epochs'?" in message


def test_suggests_closest_nested_field_name():
    error = catch(Cfg, name="a", resize={"heigth": 5})
    assert "did you mean 'height'?" in format_validation_error(
        error, model=Cfg
    )


def test_suggests_through_a_list_field():
    error = catch(Cfg, name="a", stages=[{"widht": 5}])
    message = format_validation_error(error, model=Cfg)
    assert "stages[0].widht" in message
    assert "did you mean 'width'?" in message


def test_suggests_literal_value():
    error = catch(Cfg, name="a", precision="float 16")
    message = format_validation_error(error, model=Cfg)
    assert "did you mean 'float16'?" in message


def test_no_suggestion_without_model():
    message = format_validation_error(catch(Cfg, name="a", epocs=1))
    assert "unexpected field 'epocs'" in message
    assert "did you mean" not in message


def test_no_suggestion_when_nothing_is_close():
    message = format_validation_error(
        catch(Cfg, name="a", zzzzzzzz=1), model=Cfg
    )
    assert "unexpected field 'zzzzzzzz'" in message
    assert "did you mean" not in message


def test_indices_are_formatted_as_subscripts():
    error = catch(Cfg, name="a", stages=[{"height": 1}, {"height": "tall"}])
    assert "stages[1].height" in format_validation_error(error, model=Cfg)


def test_long_values_are_truncated():
    error = catch(Cfg, name="a", epochs="x" * 500)
    message = format_validation_error(error, model=Cfg)
    assert "…" in message
    assert max(map(len, message.splitlines())) < 120


def test_record_validated_model_enables_suggestions():
    error = catch(Cfg, name="a", epocs=1)
    assert record_validated_model(error, Cfg) is error
    assert "did you mean 'epochs'?" in format_validation_error(error)


class Sub(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: int


class Union_(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int | str | list[int]
    y: Sub | list[Sub]


def test_union_members_are_collapsed():
    error = catch(Union_, x={"n": 1}, y=3.5)
    assert len(error.errors()) == 5

    problems = list(iter_validation_problems(error, model=Union_))
    assert len(problems) == 2
    assert problems[0].location == "x"
    assert (
        problems[0].message
        == "does not match any of the allowed types: int, str, list[int]"
    )
    assert problems[1].location == "y"
    assert "Sub, list[Sub]" in problems[1].message


class Siblings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p: int
    q: int


@pytest.mark.parametrize("model", [Siblings, None])
def test_sibling_fields_are_not_collapsed(model: type[BaseModel] | None):
    """Two fields given the same bad value share one interned ``input``
    object, which must not be mistaken for a failed union.
    """
    error = catch(Siblings, p="a", q="a")
    problems = list(iter_validation_problems(error, model=model))
    assert [problem.location for problem in problems] == ["p", "q"]
    assert all("allowed types" not in p.message for p in problems)


def test_duplicate_problems_are_reported_once():
    error = catch(Cfg, name="a", stages=[{"height": "x"}, {"height": "x"}])
    locations = [p.location for p in iter_validation_problems(error)]
    assert locations == ["stages[0].height", "stages[1].height"]


def test_rich_rendering():
    output = render(catch(Cfg, name="a", epocs=1), model=Cfg)
    assert "Invalid Cfg" in output
    assert "unexpected field 'epocs'" in output
    assert "did you mean 'epochs'?" in output


def test_rich_rendering_keeps_wrapped_lines_indented():
    error = catch(Cfg, name="a", precision="nope")
    lines = [
        line
        for line in render(error, model=Cfg).splitlines()
        if "float16" in line or "float32" in line
    ]
    assert lines
    # Every line of the wrapped "expected ..." message is indented under
    # the location that introduces it.
    assert all(line.startswith("│    ") for line in lines)


def test_custom_title():
    message = format_validation_error(catch(Cfg), title="Bad config")
    assert message.startswith("Bad config")


def test_custom_validator_message_is_kept_verbatim():
    class Strict(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def check(cls, v: int) -> int:
            if v % 2:
                raise ValueError("value must be even")
            return v

    message = format_validation_error(catch(Strict, value=3), model=Strict)
    assert "value must be even" in message
    assert "Value error," not in message


class Marker:
    """An arbitrary type, validated by pydantic with an isinstance check."""


def test_union_type_names_drop_pydantic_wrappers():
    class Tagged(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        value: int | Marker

    error = catch(Tagged, value=["a"])
    (problem,) = iter_validation_problems(error, model=Tagged)
    assert "is-instance[" not in problem.message
    assert problem.message.endswith("int, Marker")


def test_panel_hugs_its_content():
    """The border used to stretch to the full terminal width."""
    output = render(catch(Cfg, name="a", epocs=1), model=Cfg)
    lines = [line for line in output.splitlines() if line.strip()]
    assert max(len(line) for line in lines) < 60  # console is 100 wide
