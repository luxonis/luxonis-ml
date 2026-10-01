"""Sample identity across datasets, used to pair the samples of two datasets.

The comparison command pairs a ground-truth sample with a prediction sample
when both name the same source files. A sample's identity is thus its sorted
``(source, filename)`` pairs.
"""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from luxonis_ml.data.loaders.luxonis_loader import LuxonisLoader
    from luxonis_ml.typing import LoaderOutput

SampleIdentity: TypeAlias = tuple[tuple[str, str], ...]
"""The identity of a sample across datasets.

The sorted ``(source, filename)`` pairs of its files.
"""


def sample_identity(filenames: "Mapping[str, str]") -> SampleIdentity:
    """Stable sample identity from a source-name/filename map.

    Args:
        filenames: The file name of each source, keyed by source name.

    Returns:
        The sorted ``(source, filename)`` pairs.

    Raises:
        ValueError: If ``filenames`` is empty.

    """
    if not filenames:
        raise ValueError(
            "Dataset comparison requires loader filename metadata to match "
            "samples by identity."
        )
    return tuple(
        sorted(
            (str(source), str(filename))
            for source, filename in filenames.items()
        )
    )


def identity_label(identity: SampleIdentity) -> str:
    """Human-readable form of an identity, for reports and error messages.

    Args:
        identity: The identity to show.

    Returns:
        The pairs as ``source=filename``, joined with commas.

    """
    return ", ".join(f"{source}={filename}" for source, filename in identity)


def identity_index(
    loader: "LuxonisLoader",
    dataset_name: str,
    *,
    matches: "Callable[[LoaderOutput], bool] | None" = None,
) -> tuple[dict[SampleIdentity, int], dict[SampleIdentity, bool]]:
    """Map unique identities to loader indices and filter decisions.

    Args:
        loader: The loader whose samples are indexed.
        dataset_name: Name reported when a duplicate identity is found.
        matches: Optional sample filter; without one every sample is selected.

    Returns:
        The ``identity -> loader index`` map, and the ``identity -> selected``
        decisions.

    Raises:
        ValueError: If two samples share an identity, which would make the
            pairing ambiguous.

    """
    indexed: dict[SampleIdentity, int] = {}
    selected: dict[SampleIdentity, bool] = {}
    for index in range(len(loader)):
        identity = sample_identity(loader.get_filenames(index))
        if identity in indexed:
            raise ValueError(
                f"Dataset '{dataset_name}' contains duplicate sample "
                f"identity: {identity_label(identity)}."
            )
        indexed[identity] = index
        selected[identity] = (
            matches(loader[index]) if matches is not None else True
        )
    return indexed, selected
