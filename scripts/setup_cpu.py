"""Install the locked CPU extra with the macOS forced-Metal override."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="cafe-cpu-build-") as directory:
        hook = Path(directory) / "cpu_only.cmake"
        shutil.copyfile(Path("configs/cpu_only.cmake"), hook)
        env = {
            **os.environ,
            "CMAKE_ARGS": f"-DCMAKE_PROJECT_INCLUDE={hook} -DGGML_METAL=OFF -DGGML_CUDA=OFF",
        }
        subprocess.run(
            [
                "uv",
                "sync",
                "--locked",
                "--extra",
                "cpu",
                "--reinstall-package",
                "llama-cpp-python",
                "--no-cache",
            ],
            env=env,
            check=True,
        )
