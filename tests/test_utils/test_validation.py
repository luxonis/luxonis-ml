import io
import json
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Literal
from uuid import UUID

import pytest
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)
from rich.console import Console

from luxonis_ml.utils.validation import (
    format_validation_error,
    install_excepthook,
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


class Trainer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = 10
    use_rich: bool = True
    save_ckpt: bool = True


def test_extra_keys_sharing_a_value_are_not_collapsed():
    """Two misspelled keys given the same value used to be merged into
    one fabricated "allowed types" problem, because equal small values
    are interned and so share an ``id``.
    """
    message = format_validation_error(
        catch(Trainer, use_rihc=True, save_ckp=True), model=Trainer
    )

    assert "2 problems found" in message
    assert "allowed types" not in message
    assert "did you mean 'use_rich'?" in message
    assert "did you mean 'save_ckpt'?" in message


class Sources(BaseModel):
    files: dict[str, int]


@pytest.mark.parametrize("model", [Sources, None])
def test_mapping_entries_are_not_collapsed(model: type[BaseModel] | None):
    """Capitalized mapping keys used to be taken for type names, which
    replaced both real messages with a list of the user's own keys.
    """
    error = catch(Sources, files={"CAM_A": "x", "CAM_B": "x"})
    problems = list(iter_validation_problems(error, model=model))

    assert [p.location for p in problems] == ["files.CAM_A", "files.CAM_B"]
    assert all("allowed types" not in p.message for p in problems)


class Bracketed(BaseModel):
    files: dict[str, Resize]


@pytest.mark.parametrize("model", [Bracketed, None])
def test_mapping_keys_with_brackets_stay_in_the_location(
    model: type[BaseModel] | None,
):
    """A "[" used to be read as a pydantic-internal marker and dropped,
    pointing the reader at a key that does not exist.
    """
    error = catch(Bracketed, files={"img[1].png": {"height": "x"}})
    (problem,) = iter_validation_problems(error, model=model)

    assert problem.location == "files.img[1].png.height"


def test_bare_assert_still_says_something():
    """A bare ``assert x >= 0`` in a validator raises an
    `AssertionError` carrying no message, and ``str()`` of it is empty.

    It is raised here rather than asserted because pytest rewrites the
    asserts of a test module and would fill the message in for us.
    """

    class Positive(BaseModel):
        v: int

        @field_validator("v")
        @classmethod
        def check(cls, value: int) -> int:
            raise AssertionError

    (problem,) = iter_validation_problems(catch(Positive, v=-1))
    assert problem.message == "assertion failed"


def test_value_error_without_a_message_still_says_something():
    class Rejecting(BaseModel):
        v: int

        @field_validator("v")
        @classmethod
        def check(cls, value: int) -> int:
            raise ValueError

    (problem,) = iter_validation_problems(catch(Rejecting, v=1))
    assert problem.message == "value error"


class Pathish(BaseModel):
    x: int | Path


def test_union_type_names_survive_nested_wrappers():
    """Pydantic tags a `pathlib.Path` member with a nest of wrappers
    whose own commas used to be mistaken for the separator.
    """
    error = catch(Pathish, x=[1, 2])
    (problem,) = iter_validation_problems(error, model=Pathish)

    assert problem.message == (
        "does not match any of the allowed types: int, Path"
    )


class Aliased(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_name: str | None = Field(
        None, validation_alias=AliasChoices("class", "class_name")
    )
    instance_id: int = -1


def test_suggestions_use_validation_aliases():
    """``class`` is the key this field is actually given under, and it
    used to be missing from the candidates entirely.
    """
    message = format_validation_error(
        catch(Aliased, clas="cat"), model=Aliased
    )
    assert "did you mean 'class'?" in message


def test_no_suggestion_for_a_merely_related_name():
    """``task_name`` is not a misspelling of ``class_name``; suggesting
    it would send the reader off to rename the wrong thing.
    """
    message = format_validation_error(
        catch(Aliased, task_name="a"), model=Aliased
    )
    assert "unexpected field 'task_name'" in message
    assert "did you mean" not in message


def test_union_members_are_collapsed_in_json_mode():
    """``errors()`` builds a fresh input object per error in JSON mode,
    so grouping members by the identity of their input silently stopped
    working there.
    """
    with pytest.raises(ValidationError) as info:
        Union_.model_validate_json('{"x": {"n": 1}, "y": 3.5}')

    problems = list(iter_validation_problems(info.value, model=Union_))
    assert [p.location for p in problems] == ["x", "y"]
    assert all("allowed types" in p.message for p in problems)


def test_union_members_failing_inside_a_container_are_listed():
    """A container member reports one location component more than its
    siblings, which used to drop it from the list of allowed types and
    add its element failures as problems of their own.
    """

    class Boxed(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        value: int | Marker | list[Marker]

    error = catch(Boxed, value=["a", 1])
    (problem,) = iter_validation_problems(error, model=Boxed)

    assert problem.message == (
        "does not match any of the allowed types: int, Marker, list[Marker]"
    )
    assert problem.value == "['a', 1]"


def test_panel_title_is_not_read_as_markup():
    output = render(catch(Cfg, name="a", epocs=1), title="Config [dev]")
    assert "Config [dev]" in output


def test_panel_title_keeps_a_parametrized_model_name():
    with pytest.raises(ValidationError) as info:
        TypeAdapter(list[int]).validate_python(["x"])

    assert "Invalid list[int]" in render(info.value)


def test_panel_title_with_a_closing_tag_does_not_raise():
    """An unbalanced "[/]" used to raise `rich.errors.MarkupError` from
    inside the exception hook, printing nothing at all.
    """
    assert "Bad [/]" in render(catch(Cfg), title="Bad [/]")


def test_long_values_keep_their_tail():
    """What tells two long paths apart is their tail."""
    value = "/prefix/" + "d/" * 40 + "img_0001.png"
    error = catch(Cfg, name="a", epochs=value)
    (problem,) = iter_validation_problems(error, model=Cfg)

    assert problem.value is not None
    assert problem.value.startswith("'/prefix/")
    assert problem.value.endswith("img_0001.png'")


class Layers(BaseModel):
    layers: dict[str, Resize]


def test_suggests_through_a_mapping_key():
    """The walk used to treat a mapping key as a field name and give up
    on the first one it could not find.
    """
    message = format_validation_error(
        catch(Layers, layers={"cam-1": {"heigth": 5}}), model=Layers
    )
    assert "layers.cam-1.heigth" in message
    assert "did you mean 'height'?" in message


class Either(BaseModel):
    x: Resize | list[Resize]


def test_suggests_through_a_union_member():
    """Pydantic names the union member in the location, which used to be
    reported to the user as if it were a field of their own.
    """
    message = format_validation_error(
        catch(Either, x={"heigth": 5}), model=Either
    )
    assert "x.heigth" in message
    assert "did you mean 'height'?" in message


def test_acronyms_are_not_lowercased():
    class Ident(BaseModel):
        u: UUID

    (problem,) = iter_validation_problems(catch(Ident, u=5), model=Ident)
    assert problem.message.startswith("UUID input should be")


@pytest.fixture
def excepthook() -> Iterator[list[tuple]]:
    """Install the hook over a recording one and restore it afterwards."""
    previous = sys.excepthook
    seen: list[tuple] = []
    sys.excepthook = lambda *args: seen.append(args)
    yield seen
    sys.excepthook = previous


def test_excepthook_keeps_the_previous_hook(
    excepthook: list[tuple], capsys: pytest.CaptureFixture[str]
):
    """The panel is a summary, not a replacement: dropping the traceback
    also drops the file and line, and silences crash reporters.
    """
    install_excepthook()
    error = catch(Cfg, name="a", epocs=1)

    sys.excepthook(type(error), error, None)

    assert len(excepthook) == 1
    assert "unexpected field 'epocs'" in capsys.readouterr().err


def test_excepthook_delegates_other_exceptions(
    excepthook: list[tuple], capsys: pytest.CaptureFixture[str]
):
    install_excepthook()
    error = RuntimeError("boom")

    sys.excepthook(RuntimeError, error, None)

    assert len(excepthook) == 1
    assert capsys.readouterr().err == ""


def test_excepthook_honours_use_rich(
    excepthook: list[tuple], capsys: pytest.CaptureFixture[str]
):
    install_excepthook(use_rich=False)
    error = catch(Cfg, name="a", epocs=1)

    sys.excepthook(type(error), error, None)

    printed = capsys.readouterr().err
    assert len(excepthook) == 1
    assert "unexpected field 'epocs'" in printed
    assert "╭" not in printed


def test_excepthook_is_installed_only_once(excepthook: list[tuple]):
    install_excepthook()
    installed = sys.excepthook

    install_excepthook()

    assert sys.excepthook is installed
    assert excepthook == []


def test_augmentation_config_names_the_model_it_validated():
    """`AlbumentationsEngine` validates user YAML, so its errors are
    among the ones that most need to name the key the user meant.
    """
    from luxonis_ml.data import AlbumentationsEngine

    with pytest.raises(ValidationError) as info:
        AlbumentationsEngine(
            32,
            32,
            {"task/boundingbox": "boundingbox"},
            {"task/boundingbox": 1},
            ["image"],
            [{"name": "Flip", "parms": {"p": 1.0}}],
        )

    assert "did you mean 'params'?" in format_validation_error(info.value)


def test_archive_inspection_names_the_model_it_validated(tmp_path: Path):
    """``luxonis_ml nn_archive inspect`` validates an archive a user
    built, and used to report its errors without any suggestions.
    """
    from luxonis_ml.nn_archive.__main__ import inspect

    payload = json.dumps({"config_version": "1.0", "modle": {}}).encode()
    archive = tmp_path / "archive.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("config.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValidationError) as excinfo:
        inspect(archive)

    assert "did you mean 'model'?" in format_validation_error(excinfo.value)
