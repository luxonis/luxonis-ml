from abc import ABC, abstractmethod
from collections.abc import Iterator

from luxonis_ml.typing import LoaderOutput
from luxonis_ml.utils import AutoRegisterMeta, Registry

LOADERS_REGISTRY: Registry[type["BaseLoader"]] = Registry(name="loaders")


class BaseLoader(
    ABC, metaclass=AutoRegisterMeta, registry=LOADERS_REGISTRY, register=False
):
    """Base abstract loader class.

    Implementations return samples in the loader output format used by
    Luxonis datasets.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples.

        """
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> LoaderOutput:
        """Load a sample from the dataset.

        Args:
            idx: Index of the sample to load.

        Returns:
            Sample data in loader output format.

        """
        ...

    def __iter__(self) -> Iterator[LoaderOutput]:
        """Iterate over the dataset.

        Returns:
            Iterator over loader outputs.

        """
        for i in range(len(self)):
            yield self[i]
