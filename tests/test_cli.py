import subprocess
import sys

import pytest


@pytest.mark.parametrize("cmd", ["data", "archive", "fs", "checkhealth"])
def test_cli_commands(cmd: str):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "luxonis_ml",
            cmd,
            "--help",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
