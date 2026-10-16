"""Record notebook schema/syntax checks, without running GPU code."""

import json
from pathlib import Path

from cafe_agent.kaggle.notebook_validation import validate_notebook

if __name__ == "__main__":
    report = {
        "notebooks": [
            validate_notebook(path) for path in sorted(Path("notebooks").glob("*.ipynb"))
        ],
        "gpu_training_run": False,
        "gpu_inference_run": False,
    }
    Path("results/notebook_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
