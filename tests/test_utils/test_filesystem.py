import pytest

from luxonis_ml.utils.filesystem import LuxonisFileSystem, _get_protocol_and_path


def test_protocol():
    assert _get_protocol_and_path("foo://bar/baz") == ("foo", "bar/baz")
    assert _get_protocol_and_path("gs://foo/bar") == ("gcs", "foo/bar")
    assert _get_protocol_and_path("local_path/to/file") == (
        "file",
        "local_path/to/file",
    )

    with pytest.raises(ValueError):
        LuxonisFileSystem("foo://bar")
