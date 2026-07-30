"""Source acquisition: local paths, archives and remote providers."""

import os
import sys
import types
import zipfile
from pathlib import Path
from typing import (
    Any,
    cast,
)

import pytest
from pydantic import SecretStr

import luxonis_ml.data.parsers.source as parser_source
from luxonis_ml.data import (
    PARSERS_REGISTRY,
    Layout,
)
from luxonis_ml.data.parsers.source import prepare_source


def test_trailing_separator_in_source_keeps_dataset_name(tempdir: Path):
    """A trailing separator must not erase the derived dataset name.

    Regression: the name came from ``rsplit("/", 1)[-1]``, which is ``""`` for
    a path written with a trailing slash. The dataset was then created as
    ``""``, writing its storage directories straight into the datasets root and
    merging into whatever a previous trailing-slash import had left there, and
    the remote download target collapsed to the working directory.

    Splitting on ``"/"`` alone then made the whole thing worse on Windows,
    where a local path has none: the derived name became the entire path.
    """
    dataset_dir = tempdir / "trailing_source"
    dataset_dir.mkdir()

    assert prepare_source(f"{dataset_dir}/", None) == (
        dataset_dir,
        "trailing_source",
    )
    assert prepare_source(f"{dataset_dir}{os.sep}", None) == (
        dataset_dir,
        "trailing_source",
    )
    assert prepare_source(str(dataset_dir), None) == (
        dataset_dir,
        "trailing_source",
    )

    # A path that is nothing but separators has no name to derive at all.
    with pytest.raises(ValueError, match="Could not derive a dataset name"):
        prepare_source("/", None)


def test_prepare_source_routes_and_extracts_zip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    local = tmp_path / "local"
    local.mkdir()

    class FileSystem:
        source_path = local

        @classmethod
        def download(cls, source: str, destination: Path) -> Path:
            assert source
            assert destination
            return cls.source_path

    monkeypatch.setattr(parser_source, "LuxonisFileSystem", FileSystem)
    source, name = parser_source.prepare_source(
        "https://example.com/local", tmp_path
    )
    assert (source, name) == (local, "local")

    monkeypatch.setattr(
        parser_source,
        "_download_roboflow_dataset",
        lambda source, local_path: (local, "roboflow-name"),
    )
    assert parser_source.prepare_source(
        "roboflow://workspace/project/1/coco", tmp_path
    ) == (local, "roboflow-name")

    monkeypatch.setattr(
        parser_source,
        "_download_ultralytics_dataset",
        lambda source, local_path: (local, "ultralytics-name"),
    )
    assert parser_source.prepare_source(
        "ultralytics://user/datasets/project", tmp_path
    ) == (local, "ultralytics-name")

    archive = tmp_path / "wrapped.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("wrapper/train/bird/image.jpg", b"image")
    FileSystem.source_path = archive
    source, name = parser_source.prepare_source(archive, tmp_path)
    assert name == archive.name
    assert source == tmp_path / "wrapped" / "wrapper"


def test_resolve_extracted_zip_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    multiple = tmp_path / "multiple"
    (multiple / "one").mkdir(parents=True)
    (multiple / "two").mkdir()
    assert parser_source._resolve_extracted_zip_root(multiple) == multiple

    unrecognized = tmp_path / "unrecognized"
    only = unrecognized / "wrapper"
    (only / "content").mkdir(parents=True)

    class NeverParser:
        @classmethod
        def detect(cls, source: Path) -> None:
            return None

    monkeypatch.setattr(PARSERS_REGISTRY, "values", lambda: [NeverParser])
    assert (
        parser_source._resolve_extracted_zip_root(unrecognized) == unrecognized
    )

    marker = tmp_path / "marker"
    wrapped = marker / "wrapper"
    (wrapped / "train").mkdir(parents=True)
    assert parser_source._resolve_extracted_zip_root(marker) == wrapped

    recognized = tmp_path / "recognized"
    recognized_wrapper = recognized / "wrapper"
    (recognized_wrapper / "content").mkdir(parents=True)

    class RecognizedParser:
        @classmethod
        def detect(cls, source: Path) -> Layout | None:
            return Layout({None: {}}) if source == recognized_wrapper else None

    monkeypatch.setattr(PARSERS_REGISTRY, "values", lambda: [RecognizedParser])
    assert (
        parser_source._resolve_extracted_zip_root(recognized)
        == recognized_wrapper
    )


def _fake_roboflow_module(location: Path) -> types.ModuleType:
    module = types.ModuleType("roboflow")

    class Version:
        @staticmethod
        def download(
            export_format: str, destination: str
        ) -> types.SimpleNamespace:
            assert export_format == "coco"
            # Compared as a path, not as a string: the destination is
            # built with the platform separator, which is a backslash on
            # Windows.
            assert Path(destination).name == "project"
            return types.SimpleNamespace(location=str(location))

    class Project:
        @staticmethod
        def version(version: int) -> Version:
            assert version == 2
            return Version()

    class Workspace:
        @staticmethod
        def project(project: str) -> Project:
            assert project == "project"
            return Project()

    class Roboflow:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "secret"

        @staticmethod
        def workspace(workspace: str) -> Workspace:
            assert workspace == "workspace"
            return Workspace()

    cast(Any, module).Roboflow = Roboflow
    return module


def test_download_roboflow_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(parser_source.environ, "ROBOFLOW_API_KEY", None)
    with pytest.raises(RuntimeError, match="ROBOFLOW_API_KEY"):
        parser_source._download_roboflow_dataset(
            "roboflow://workspace/project/2/coco", tmp_path
        )

    monkeypatch.setattr(
        parser_source.environ,
        "ROBOFLOW_API_KEY",
        SecretStr("secret"),
    )
    monkeypatch.setattr(parser_source, "find_spec", lambda name: object())
    monkeypatch.setitem(
        sys.modules,
        "roboflow",
        _fake_roboflow_module(tmp_path / "downloaded"),
    )

    with pytest.raises(ValueError, match="Incorrect Roboflow"):
        parser_source._download_roboflow_dataset(
            "roboflow://workspace/project/2", tmp_path
        )
    with pytest.raises(ValueError, match="must be an integer"):
        parser_source._download_roboflow_dataset(
            "roboflow://workspace/project/latest/coco", tmp_path
        )

    assert parser_source._download_roboflow_dataset(
        "roboflow://workspace/project/2/coco", tmp_path
    ) == (tmp_path / "downloaded", "project")


class _Response:
    def __init__(
        self,
        *,
        ok: bool,
        status_code: int,
        payload: dict[str, Any] | ValueError,
        text: str = "",
        reason: str = "",
    ) -> None:
        self.ok = ok
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.reason = reason

    def json(self) -> dict[str, Any]:
        if isinstance(self._payload, ValueError):
            raise self._payload
        return self._payload


def test_download_ultralytics_reference_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(parser_source.environ, "ULTRALYTICS_API_KEY", None)
    with pytest.raises(RuntimeError, match="ULTRALYTICS_API_KEY"):
        parser_source._download_ultralytics_dataset(
            "ultralytics://user/datasets/project", tmp_path
        )

    monkeypatch.setattr(
        parser_source.environ,
        "ULTRALYTICS_API_KEY",
        SecretStr("secret"),
    )
    with pytest.raises(ValueError, match="must be an integer"):
        parser_source._download_ultralytics_dataset(
            "ultralytics://user/datasets/project?v=latest", tmp_path
        )
    with pytest.raises(ValueError, match="must be >= 1"):
        parser_source._download_ultralytics_dataset(
            "ultralytics://user/datasets/project?v=0", tmp_path
        )
    with pytest.raises(ValueError, match="Incorrect Ultralytics"):
        parser_source._download_ultralytics_dataset(
            "ultralytics://user/projects/project", tmp_path
        )


def test_download_ultralytics_api_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        parser_source.environ,
        "ULTRALYTICS_API_KEY",
        SecretStr("secret"),
    )
    reference = "ultralytics://user/datasets/project"

    monkeypatch.setattr(
        parser_source.requests,
        "get",
        lambda *args, **kwargs: _Response(
            ok=False,
            status_code=401,
            payload={"error": "unauthorized"},
        ),
    )
    with pytest.raises(RuntimeError, match=r"401.*unauthorized"):
        parser_source._download_ultralytics_dataset(reference, tmp_path)

    monkeypatch.setattr(
        parser_source.requests,
        "get",
        lambda *args, **kwargs: _Response(
            ok=False,
            status_code=500,
            payload=ValueError(),
            text="server failed",
        ),
    )
    with pytest.raises(RuntimeError, match=r"500.*server failed"):
        parser_source._download_ultralytics_dataset(reference, tmp_path)

    dataset_response = _Response(
        ok=True,
        status_code=200,
        payload={
            "dataset": {
                "_id": "dataset-id",
                "slug": "project",
                "name": "Project",
            }
        },
    )
    export_json_error = _Response(
        ok=False,
        status_code=422,
        payload={"error": "bad export"},
        reason="Unprocessable",
    )
    responses = iter([dataset_response, export_json_error])
    monkeypatch.setattr(
        parser_source.requests,
        "get",
        lambda *args, **kwargs: next(responses),
    )
    with pytest.raises(RuntimeError, match="422 Unprocessable: bad export"):
        parser_source._download_ultralytics_dataset(reference, tmp_path)

    export_text_error = _Response(
        ok=False,
        status_code=503,
        payload=ValueError(),
        text="",
        reason="Unavailable",
    )
    responses = iter([dataset_response, export_text_error])
    monkeypatch.setattr(
        parser_source.requests,
        "get",
        lambda *args, **kwargs: next(responses),
    )
    with pytest.raises(RuntimeError, match="503 Unavailable"):
        parser_source._download_ultralytics_dataset(reference, tmp_path)


@pytest.mark.parametrize("version", [None, 3])
def test_download_ultralytics_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: int | None,
):
    monkeypatch.setattr(
        parser_source.environ,
        "ULTRALYTICS_API_KEY",
        SecretStr("secret"),
    )
    dataset_response = _Response(
        ok=True,
        status_code=200,
        payload={
            "dataset": {
                "_id": "dataset-id",
                "slug": "project",
                "name": "Project",
            }
        },
    )
    export_response = _Response(
        ok=True,
        status_code=200,
        payload={"downloadUrl": "https://example.com/export.ndjson"},
    )
    responses = iter([dataset_response, export_response])
    monkeypatch.setattr(
        parser_source.requests,
        "get",
        lambda *args, **kwargs: next(responses),
    )

    def download(url: str, destination: Path, *, timeout: float) -> None:
        assert url.endswith("export.ndjson")
        assert timeout == 120.0
        destination.write_text("downloaded")

    monkeypatch.setattr(parser_source, "download_remote_file", download)
    suffix = f"?v={version}" if version is not None else ""
    destination, name = parser_source._download_ultralytics_dataset(
        f"ultralytics://user/datasets/project{suffix}",
        tmp_path,
    )
    assert destination.name == (
        f"project.v{version}.ndjson"
        if version is not None
        else "project.ndjson"
    )
    assert destination.read_text() == "downloaded"
    assert name == "Project"
