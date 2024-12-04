from typing import Dict, Tuple

import numpy as np
from typing_extensions import TypeAlias

Labels: TypeAlias = Dict[str, np.ndarray]
"""Dictionary mappping task names to the annotations as C{np.ndarray}"""


LuxonisLoaderOutput: TypeAlias = Tuple[np.ndarray, Labels]
"""C{LuxonisLoaderOutput} is a tuple of an image as a C{np.ndarray>} and
a dictionary of task group names and their annotations as
L{Annotations}."""
