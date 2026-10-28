"""Tests for runtime dependencies in NN Archive manifests and tar files."""

import json
import tarfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from luxonis_ml.nn_archive import ArchiveGenerator, Config


def _config() -> dict:
    return {
        "config_version": "1.1",
        "model": {
            "metadata": {"name": "custom-model", "path": "model.dlc"},
            "inputs": [],
            "outputs": [],
            "heads": [],
        },
    }


def _dependencies() -> dict:
    return {
        "snpe_udo": [
            {
                "registration": "runtime/libFirstReg.so",
                "implementations": [
                    {
                        "runtime": "dsp",
                        "path": "runtime/libFirstImplDsp.so",
                    },
                    {
                        "runtime": "cpu",
                        "path": "runtime/libFirstImplCpu.so",
                    },
                ],
            },
            {
                "registration": "runtime/libSecondReg.so",
                "implementations": [
                    {
                        "runtime": "dsp",
                        "path": "runtime/libSecondImplDsp.so",
                    }
                ],
            },
        ]
    }


def test_runtime_dependencies_round_trip_in_archive(tmp_path: Path) -> None:
    config = _config()
    config["model"]["runtime_dependencies"] = _dependencies()
    expected_paths = {
        "runtime/libFirstReg.so",
        "runtime/libFirstImplDsp.so",
        "runtime/libFirstImplCpu.so",
        "runtime/libSecondReg.so",
        "runtime/libSecondImplDsp.so",
    }
    local_paths = {}
    for archive_path in expected_paths:
        local_path = tmp_path / Path(archive_path).name
        local_path.write_bytes(archive_path.encode())
        local_paths[archive_path] = local_path
    model_path = tmp_path / "model.dlc"
    model_path.write_bytes(b"model")

    generator = ArchiveGenerator(
        archive_name="custom-model",
        save_path=tmp_path,
        cfg_dict=config,
        executables_paths=[model_path],
        runtime_dependency_paths=local_paths,
    )
    archive_path = generator.make_archive()

    with tarfile.open(archive_path) as tar:
        assert set(tar.getnames()) == {
            "config.json",
            "model.dlc",
            *expected_paths,
        }
        config_file = tar.extractfile("config.json")
        assert config_file is not None
        saved_config = json.load(config_file)
        assert saved_config["model"]["runtime_dependencies"] == _dependencies()
        assert (
            Config.model_validate(saved_config).model_dump()["model"][
                "runtime_dependencies"
            ]
            == _dependencies()
        )
        for dependency_path in expected_paths:
            member = tar.extractfile(dependency_path)
            assert member is not None
            assert member.read() == dependency_path.encode()


def test_ordinary_archive_omits_runtime_dependencies(tmp_path: Path) -> None:
    config = _config()
    config.pop("config_version")
    assert Config.model_validate(config).config_version == "1.0"
    config["config_version"] = "1.0"
    assert (
        "runtime_dependencies"
        not in Config.model_validate(config).model_dump()["model"]
    )
    model_path = tmp_path / "model.dlc"
    model_path.write_bytes(b"model")
    generator = ArchiveGenerator(
        archive_name="ordinary-model",
        save_path=tmp_path,
        cfg_dict=config,
        executables_paths=[model_path],
    )

    with tarfile.open(generator.make_archive()) as tar:
        config_file = tar.extractfile("config.json")
        assert config_file is not None
        saved_config = json.load(config_file)
        assert "runtime_dependencies" not in saved_config["model"]


@pytest.mark.parametrize(
    "bad_path",
    [
        "/lib.so",
        "C:/runtime/lib.so",
        "C:runtime/lib.so",
        "//server/share/lib.so",
        "../lib.so",
        "a/../lib.so",
        "a\\lib.so",
        "./lib.so",
        ".",
        "runtime/lib\x00.so",
        "config.json",
    ],
)
def test_dependency_paths_must_be_inside_archive(bad_path: str) -> None:
    config = _config()
    config["model"]["runtime_dependencies"] = {
        "snpe_udo": [
            {
                "registration": bad_path,
                "implementations": [
                    {"runtime": "dsp", "path": "runtime/libImpl.so"}
                ],
            }
        ]
    }

    with pytest.raises(ValidationError):
        Config.model_validate(config)


def test_implementation_path_must_be_inside_archive() -> None:
    config = _config()
    config["model"]["runtime_dependencies"] = _dependencies()
    config["model"]["runtime_dependencies"]["snpe_udo"][0]["implementations"][
        0
    ]["path"] = "../libImpl.so"

    with pytest.raises(ValidationError):
        Config.model_validate(config)


def test_generator_requires_all_declared_dependency_files(
    tmp_path: Path,
) -> None:
    config = _config()
    config["model"]["runtime_dependencies"] = _dependencies()
    with pytest.raises(ValueError, match="missing"):
        ArchiveGenerator(
            archive_name="custom-model",
            save_path=tmp_path,
            cfg_dict=config,
            executables_paths=[],
        )


def test_generator_rejects_missing_local_file(tmp_path: Path) -> None:
    config = _config()
    config["model"]["runtime_dependencies"] = {
        "snpe_udo": [
            {
                "registration": "runtime/libReg.so",
                "implementations": [
                    {"runtime": "dsp", "path": "runtime/libImpl.so"}
                ],
            }
        ]
    }
    with pytest.raises(FileNotFoundError):
        ArchiveGenerator(
            archive_name="custom-model",
            save_path=tmp_path,
            cfg_dict=config,
            executables_paths=[],
            runtime_dependency_paths={
                "runtime/libReg.so": tmp_path / "missingReg.so",
                "runtime/libImpl.so": tmp_path / "missingImpl.so",
            },
        )


def test_generator_bundles_symlinked_library_bytes(tmp_path: Path) -> None:
    config = _config()
    config["model"]["runtime_dependencies"] = {
        "snpe_udo": [
            {
                "registration": "runtime/libReg.so",
                "implementations": [
                    {"runtime": "dsp", "path": "runtime/libImpl.so"}
                ],
            }
        ]
    }
    model_path = tmp_path / "model.dlc"
    model_path.write_bytes(b"model")
    registration_path = tmp_path / "libReg.so"
    registration_path.write_bytes(b"registration")
    implementation_target = tmp_path / "libImpl.so.1"
    implementation_target.write_bytes(b"implementation")
    implementation_link = tmp_path / "libImpl.so"
    try:
        implementation_link.symlink_to(implementation_target.name)
    except (NotImplementedError, OSError):
        pytest.skip("Creating symlinks is unavailable on this host")

    archive_path = ArchiveGenerator(
        archive_name="symlinked-library",
        save_path=tmp_path,
        cfg_dict=config,
        executables_paths=[model_path],
        runtime_dependency_paths={
            "runtime/libReg.so": registration_path,
            "runtime/libImpl.so": implementation_link,
        },
    ).make_archive()

    with tarfile.open(archive_path) as tar:
        member = tar.getmember("runtime/libImpl.so")
        assert member.isfile()
        contents = tar.extractfile(member)
        assert contents is not None
        assert contents.read() == b"implementation"
