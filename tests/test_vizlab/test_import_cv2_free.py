"""Guard: importing vizlab (and its viewer) must not pull in heavy backends.

cv2 and ipywidgets are heavy and, on some hosts, unavailable. vizlab core and the
viewer import their backends only lazily (inside functions), so a bare import must
leave ``cv2`` and ``ipywidgets`` out of ``sys.modules``. Run in a subprocess
because the data-package conftest imports cv2 into this process.
"""

import subprocess
import sys


def test_vizlab_and_viewer_import_without_backends() -> None:
    code = (
        "import sys\n"
        "import luxonis_ml.vizlab\n"
        "import luxonis_ml.vizlab.viewer\n"
        "assert 'cv2' not in sys.modules, 'cv2 imported at import time'\n"
        "assert 'ipywidgets' not in sys.modules, 'ipywidgets imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
