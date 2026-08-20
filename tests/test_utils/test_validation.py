import io
import json
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
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


def catch_anonymous(model: type[BaseModel], **data) -> ValidationError:
    """Catch an error the formatter cannot trace back to its model.

    `TypeAdapter` validates from a frame of its own, so nothing in the
    traceback names the model — the same position the formatter is in when
    the error is handed to it by third-party code.
    """
    with pytest.raises(ValidationError) as info:
        TypeAdapter(model).validate_python(data)
    return info.value


Catcher = Callable[..., ValidationError]
CATCHERS: list[Catcher] = [catch, catch_anonymous]


def missing_letter(word: str, index: int) -> str:
    return word[:index] + word[index + 1 :]


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
    error = catch(
        Cfg,
        name="a",
        resize={missing_letter("height", 4): 5},
    )
    assert "did you mean 'height'?" in format_validation_error(
        error, model=Cfg
    )


def test_suggests_through_a_list_field():
    wrong = missing_letter("width", 3)
    error = catch(Cfg, name="a", stages=[{wrong: 5}])
    message = format_validation_error(error, model=Cfg)
    assert f"stages[0].{wrong}" in message
    assert "did you mean 'width'?" in message


def test_suggests_literal_value():
    error = catch(Cfg, name="a", precision="float 16")
    message = format_validation_error(error, model=Cfg)
    assert "did you mean 'float16'?" in message


def test_no_suggestion_without_model():
    message = format_validation_error(catch_anonymous(Cfg, name="a", epocs=1))
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


def test_model_is_recovered_from_the_traceback():
    error = catch(Cfg, name="a", epocs=1)
    assert "did you mean 'epochs'?" in format_validation_error(error)


def test_model_is_recovered_through_intervening_frames():
    """The pydantic frame is the innermost one, however deep in the call
    stack the validation happened.
    """

    def build(**data: Any) -> Cfg:
        return Cfg(**data)

    def call_build() -> Cfg:
        return build(name="a", epocs=1)

    with pytest.raises(ValidationError) as info:
        call_build()

    assert "did you mean 'epochs'?" in format_validation_error(info.value)


def test_an_unrelated_model_in_the_caller_is_not_recovered():
    """Only pydantic's own frames are searched. A caller that happens to be
    a model would otherwise be mistaken for the one that failed, and would
    answer the misspelling with a field name of its own.
    """

    class Bystander(BaseModel):
        epoch: int = 0

        def validate_something(self) -> None:
            TypeAdapter(Cfg).validate_python({"name": "a", "epocs": 1})

    with pytest.raises(ValidationError) as info:
        Bystander().validate_something()

    message = format_validation_error(info.value)
    assert "unexpected field 'epocs'" in message
    assert "did you mean" not in message


class Sub(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: int


class Union_(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int | str | list[int]
    y: Sub | list[Sub]


def test_union_members_are_collapsed():
    error = catch(Union_, x={"n": 1}, y=3.5)
    # Pinned on purpose: this module is an adapter over how pydantic shapes
    # union errors, so a change in how many it reports should fail loudly
    # here rather than quietly reshape the output.
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


@pytest.mark.parametrize("catcher", CATCHERS)
def test_sibling_fields_are_not_collapsed(catcher: Catcher):
    error = catcher(Siblings, p="a", q="a")
    problems = list(iter_validation_problems(error))
    assert [problem.location for problem in problems] == ["p", "q"]
    assert all("allowed types" not in p.message for p in problems)


def test_equal_list_items_keep_separate_locations():
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
    pass


def test_union_type_names_drop_pydantic_wrappers():
    class Tagged(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        value: int | Marker

    error = catch(Tagged, value=["a"])
    (problem,) = iter_validation_problems(error, model=Tagged)
    assert "is-instance[" not in problem.message
    assert problem.message.endswith("int, Marker")


def test_panel_hugs_its_content():
    output = render(catch(Cfg, name="a", epocs=1), model=Cfg)
    lines = [line for line in output.splitlines() if line.strip()]
    assert max(len(line) for line in lines) < 60  # console is 100 wide


def test_extra_keys_sharing_a_value_are_not_collapsed():
    class Trainer(BaseModel):
        model_config = ConfigDict(extra="forbid")

        epochs: int = 10
        use_rich: bool = True
        save_ckpt: bool = True

    message = format_validation_error(
        catch(Trainer, use_rihc=True, save_ckp=True), model=Trainer
    )

    assert "2 problems found" in message
    assert "allowed types" not in message
    assert "did you mean 'use_rich'?" in message
    assert "did you mean 'save_ckpt'?" in message


class Sources(BaseModel):
    files: dict[str, int]


@pytest.mark.parametrize("catcher", CATCHERS)
def test_mapping_entries_are_not_collapsed(catcher: Catcher):
    error = catcher(Sources, files={"CAM_A": "x", "CAM_B": "x"})
    problems = list(iter_validation_problems(error))

    assert [p.location for p in problems] == ["files.CAM_A", "files.CAM_B"]
    assert all("allowed types" not in p.message for p in problems)


def test_mapping_keys_named_after_a_type_are_not_collapsed():
    error = catch_anonymous(Sources, files={"path": "x", "none": "y"})
    problems = list(iter_validation_problems(error))

    assert [p.location for p in problems] == ["files.path", "files.none"]
    assert all("allowed types" not in p.message for p in problems)


class Bracketed(BaseModel):
    files: dict[str, Resize]


@pytest.mark.parametrize("catcher", CATCHERS)
def test_mapping_keys_with_brackets_stay_in_the_location(catcher: Catcher):
    error = catcher(Bracketed, files={"img[1].png": {"height": "x"}})
    (problem,) = iter_validation_problems(error)

    assert problem.location == "files.img[1].png.height"


def test_bare_assert_still_says_something():
    class Positive(BaseModel):
        v: int

        @field_validator("v")
        @classmethod
        def check(cls, _value: int) -> int:
            raise AssertionError

    (problem,) = iter_validation_problems(catch(Positive, v=-1))
    assert problem.message == "assertion failed"


def test_value_error_without_a_message_still_says_something():
    class Rejecting(BaseModel):
        v: int

        @field_validator("v")
        @classmethod
        def check(cls, _value: int) -> int:
            raise ValueError

    (problem,) = iter_validation_problems(catch(Rejecting, v=1))
    assert problem.message == "value error"


def test_union_type_names_survive_nested_wrappers():
    class Pathish(BaseModel):
        x: int | Path

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
    wrong = missing_letter("class", 4)
    message = format_validation_error(
        catch(Aliased, **{wrong: "cat"}), model=Aliased
    )
    assert "did you mean 'class'?" in message


class NameOnly(BaseModel):
    model_config = ConfigDict(
        extra="forbid", validate_by_alias=False, validate_by_name=True
    )

    class_name: str = Field(validation_alias="class")


def test_suggestions_skip_an_alias_the_model_rejects():
    wrong = missing_letter("class", 4)
    message = format_validation_error(
        catch(NameOnly, **{wrong: "cat"}), model=NameOnly
    )
    assert f"unexpected field {wrong!r}" in message
    assert "did you mean 'class'?" not in message


def test_suggestions_use_the_name_the_model_accepts():
    wrong = missing_letter("class_name", 4)
    message = format_validation_error(
        catch(NameOnly, **{wrong: "cat"}), model=NameOnly
    )
    assert "did you mean 'class_name'?" in message


def test_no_suggestion_for_a_merely_related_name():
    message = format_validation_error(
        catch(Aliased, task_name="a"), model=Aliased
    )
    assert "unexpected field 'task_name'" in message
    assert "did you mean" not in message


def test_union_members_are_collapsed_in_json_mode():
    with pytest.raises(ValidationError) as info:
        Union_.model_validate_json('{"x": {"n": 1}, "y": 3.5}')

    problems = list(iter_validation_problems(info.value, model=Union_))
    assert [p.location for p in problems] == ["x", "y"]
    assert all("allowed types" in p.message for p in problems)


def test_union_members_failing_inside_a_container_are_listed():
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
    assert "Bad [/]" in render(catch(Cfg), title="Bad [/]")


def test_long_values_keep_their_tail():
    value = "/prefix/" + "d/" * 40 + "img_0001.png"
    error = catch(Cfg, name="a", epochs=value)
    (problem,) = iter_validation_problems(error, model=Cfg)

    assert problem.value is not None
    assert problem.value.startswith("'/prefix/")
    assert problem.value.endswith("img_0001.png'")


def test_fixed_length_tuple_elements_resolve_by_position():
    class Pair(BaseModel):
        pair: tuple[int, Resize]

    wrong = missing_letter("height", 4)
    message = format_validation_error(
        catch(Pair, pair=(1, {wrong: 5})), model=Pair
    )
    assert f"pair[1].{wrong}" in message
    assert "did you mean 'height'?" in message


def test_variadic_tuple_elements_keep_their_type():
    class Many(BaseModel):
        items: tuple[Resize, ...]

    wrong = missing_letter("width", 3)
    message = format_validation_error(
        catch(Many, items=({wrong: 5},)), model=Many
    )
    assert f"items[0].{wrong}" in message
    assert "did you mean 'width'?" in message


def test_suggests_through_a_mapping_key():
    class Layers(BaseModel):
        layers: dict[str, Resize]

    wrong = missing_letter("height", 4)
    message = format_validation_error(
        catch(Layers, layers={"cam-1": {wrong: 5}}), model=Layers
    )
    assert f"layers.cam-1.{wrong}" in message
    assert "did you mean 'height'?" in message


def test_suggests_through_a_union_member():
    class Either(BaseModel):
        x: Resize | list[Resize]

    wrong = missing_letter("height", 4)
    message = format_validation_error(
        catch(Either, x={wrong: 5}), model=Either
    )
    assert f"x.{wrong}" in message
    assert "did you mean 'height'?" in message


def test_acronyms_are_not_lowercased():
    class Ident(BaseModel):
        u: UUID

    (problem,) = iter_validation_problems(catch(Ident, u=5), model=Ident)
    assert problem.message.startswith("UUID input should be")


@pytest.fixture
def excepthook(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    seen: list[tuple] = []
    monkeypatch.setattr(sys, "excepthook", lambda *args: seen.append(args))
    return seen


def test_excepthook_keeps_the_previous_hook(
    excepthook: list[tuple], capsys: pytest.CaptureFixture[str]
):
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


def test_excepthook_can_switch_to_plain_text(
    excepthook: list[tuple], capsys: pytest.CaptureFixture[str]
):
    install_excepthook(use_rich=True)
    installed = sys.excepthook

    install_excepthook(use_rich=False)
    error = catch(Cfg, name="a", epocs=1)
    sys.excepthook(type(error), error, None)

    printed = capsys.readouterr().err
    assert sys.excepthook is installed
    assert len(excepthook) == 1
    assert "unexpected field 'epocs'" in printed
    assert "╭" not in printed


def test_excepthook_can_be_disabled(excepthook: list[tuple]):
    previous = sys.excepthook
    install_excepthook()

    install_excepthook(enabled=False)

    assert sys.excepthook is previous
    assert excepthook == []


def test_augmentation_config_names_the_model_it_validated():
    from luxonis_ml.data import AlbumentationsEngine

    wrong = missing_letter("params", 3)
    with pytest.raises(ValidationError) as info:
        AlbumentationsEngine(
            height=32,
            width=32,
            targets={"task/boundingbox": "boundingbox"},
            n_classes={"task/boundingbox": 1},
            source_names=["image"],
            config=[{"name": "Flip", wrong: {"p": 1.0}}],
        )

    assert "did you mean 'params'?" in format_validation_error(info.value)


def test_archive_inspection_names_the_model_it_validated(tmp_path: Path):
    from luxonis_ml.nn_archive.__main__ import inspect

    wrong = missing_letter("model", 3)
    payload = json.dumps({"config_version": "1.0", wrong: {}}).encode()
    archive = tmp_path / "archive.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("config.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValidationError) as excinfo:
        inspect(archive)

    assert "did you mean 'model'?" in format_validation_error(excinfo.value)
