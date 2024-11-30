from abc import ABC, abstractmethod
from typing import Dict, Iterator, Tuple, Type

import numpy as np
from typing_extensions import TypeAlias

from luxonis_ml.utils import AutoRegisterMeta, Registry

Labels: TypeAlias = Dict[str, np.ndarray]
"""Dictionary mappping task names to the annotations as C{np.ndarray}"""


LuxonisLoaderOutput: TypeAlias = Tuple[np.ndarray, Labels]
"""C{LuxonisLoaderOutput} is a tuple of an image as a C{np.ndarray>} and
a dictionary of task group names and their annotations as
L{Annotations}."""

LOADERS_REGISTRY: Registry[Type["BaseLoader"]] = Registry(name="loaders")


class BaseLoader(
    ABC, metaclass=AutoRegisterMeta, registry=LOADERS_REGISTRY, register=False
):
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
        @rtype: L{LuxonisLoaderOutput}
        @return: Sample's data in C{LuxonisLoaderOutput} format.
        """
        pass

    def __iter__(self) -> Iterator[LuxonisLoaderOutput]:
        """Iterates over the dataset.

        @rtype: Iterator[L{LuxonisLoaderOutput}]
        @return: Iterator over the dataset.
        """
        for i in range(len(self)):
            yield self[i]
