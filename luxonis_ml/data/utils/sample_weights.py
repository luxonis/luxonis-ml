"""Resolve training weights from record-level dataset provenance."""

from collections.abc import Mapping
from math import isfinite

from luxonis_ml.typing import Params


class SourceDatasetWeighting:
    """Map ``sample_metadata['source_dataset']`` to a loss multiplier.

    Missing or unlisted sources have weight 1.0. For outputs that combine
    records (Mosaic, MixUp, CutMix), use the arithmetic mean of contributing
    record weights from ``batch_augmentation_metadata``. This is a weight
    for the whole augmented sample, without estimating pixel or object area.
    Stored metadata is never modified.
    """

    def __init__(self, weights: Mapping[str, float]) -> None:
        self.weights = dict(weights)
        for source, weight in self.weights.items():
            if not isinstance(source, str) or not source:
                raise ValueError(
                    "Source dataset names must be nonempty strings"
                )
            if not isfinite(weight) or weight < 0:
                raise ValueError(
                    f"Weight for {source!r} must be finite and nonnegative"
                )

    def __call__(self, metadata: Params) -> float:
        """Return the multiplier for one loader output's metadata."""
        contributors = metadata.get("batch_augmentation_metadata")
        if contributors is not None:
            if not isinstance(contributors, list) or not contributors:
                raise ValueError(
                    "Expected nonempty batch_augmentation_metadata"
                )
            weights = []
            for contributor in contributors:
                if not isinstance(contributor, dict) or not isinstance(
                    contributor.get("sample_metadata"), dict
                ):
                    raise TypeError("Expected contributor sample_metadata")
                weights.append(self(contributor["sample_metadata"]))
            # Divide before summing to avoid overflow for large finite weights.
            return sum(weight / len(weights) for weight in weights)
        source = metadata.get("source_dataset")
        if source is not None and not isinstance(source, str):
            raise ValueError(
                "sample_metadata['source_dataset'] must be a string"
            )
        return float(self.weights.get(source, 1.0))
