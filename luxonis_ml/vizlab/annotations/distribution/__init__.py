"""The class-distribution overlay and the painters for its modes.

`ClassDistribution` handles the data — scores, ordering, selection, cell layout —
and `charts` paints whichever mode was chosen. Splitting them keeps a mode
readable on its own: none of the painters knows how the numbers were produced.
"""

from .annotation import ClassDistribution, DistributionMode, ValueFormat

__all__ = ["ClassDistribution", "DistributionMode", "ValueFormat"]
