from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
from typing_extensions import TypeAlias

from ..utils.enums import LabelType

Labels: TypeAlias = Dict[str, Tuple[np.ndarray, LabelType]]
"""C{Labels} is a dictionary mappping task names to their L{LabelType} and annotations
as L{numpy arrays<np.ndarray>}."""


LuxonisLoaderOutput: TypeAlias = Tuple[np.ndarray, Labels]
"""C{LuxonisLoaderOutput} is a tuple of an image as a L{numpy array<np.ndarray>} and a
dictionary of task group names and their annotations as L{Annotations}."""


class BaseLoader(ABC):
    """Base abstract loader class.

    Enforces the L{LuxonisLoaderOutput} output label structure.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the dataset.

        @rtype: int
        @return: Length of the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> LuxonisLoaderOutput:
        """Loads sample from dataset.

        @type idx: int
        @param idx: Index of the sample to load.
        @rtype: LuxonisLoaderOutput
        @return: Sample's data in L{LuxonisLoaderOutput} format.
        """
        pass
