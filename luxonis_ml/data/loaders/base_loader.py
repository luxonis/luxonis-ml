from abc import ABC, abstractmethod
from typing import Iterator, Type

from luxonis_ml.data.utils.types import LuxonisLoaderOutput
from luxonis_ml.utils import AutoRegisterMeta, Registry

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
