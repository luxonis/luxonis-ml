import random
from collections.abc import Iterator

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def restore_global_rngs() -> Iterator[None]:
    """Undo the process-global reseeding `BatchCompose` does on construction.

    Building an engine seeds `random` and `numpy.random` for the whole
    process, so without this every test scheduled after one of these on the
    same xdist worker would inherit a fixed RNG stream.
    """
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    yield
    random.setstate(random_state)
    np.random.set_state(numpy_state)
