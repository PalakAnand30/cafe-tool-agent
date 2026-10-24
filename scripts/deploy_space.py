"""Prepare a deployment by default; --deploy explicitly creates/updates the Space."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from cafe_agent.serve.deployment import prepare_bundle, publish_bundle
from cafe_agent.serve.settings import serve_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument(
        "--public", action="store_true", help="Explicitly create/make only the demo Space public"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory, receipt = prepare_bundle(root, root / ".cache/space-build")
    if args.deploy:
        load_dotenv()
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("Set HF_TOKEN in .env before deployment")
        api = HfApi(token=token)
        username = os.environ.get("HF_USERNAME") or api.whoami()["name"]
        repo = username + "/" + serve_settings().space_name
        receipt.update(
            repo=repo,
            revision=publish_bundle(api, repo, directory, token, public=args.public),
            published=True,
        )
    receipt["bundle"] = str(directory.relative_to(root))
    (root / "results/space_deployment.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Prepared {len(receipt['files'])} files in {directory.relative_to(root)}")
    if receipt["published"]:
        print(f"Space updated: https://huggingface.co/spaces/{receipt['repo']}")
    else:
        print("Dry run complete. No Space was created or changed. Use --deploy to publish.")


if __name__ == "__main__":
    main()
