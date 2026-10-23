"""Run the S3 API evaluator or recompute the complete Phase 5 report offline."""

import sys

if __name__ == "__main__":
    if sys.argv[1:] == ["--metrics-only"]:
        from cafe_agent.reporting.cli import main
    else:
        # Preserve historical --config/--output-dir S3 workflows.
        from cafe_agent.eval.cli import main

    main()
