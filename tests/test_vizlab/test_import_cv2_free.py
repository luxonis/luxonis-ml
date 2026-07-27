"""Guard: importing vizlab (and its viewer) must not pull in OpenCV.

cv2 is heavy and, on some hosts, fails to load without extra system libraries.
vizlab core and the viewer import it only lazily (inside functions), so a bare
import must leave ``cv2`` out of ``sys.modules``. Run in a subprocess because the
data-package conftest imports cv2 into this process.
"""

import subprocess
import sys


def test_vizlab_and_viewer_import_without_cv2() -> None:
    code = (
        "import sys\n"
        "import luxonis_ml.vizlab\n"
        "import luxonis_ml.vizlab.viewer\n"
        "assert 'cv2' not in sys.modules, 'cv2 imported at import time'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
