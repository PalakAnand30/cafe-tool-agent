"""Non-secret package provenance for user-run notebook jobs."""

import importlib.metadata
import json


def package_provenance() -> dict:
    package = importlib.metadata.distribution("cafe-tool-agent")
    direct = json.loads(package.read_text("direct_url.json") or "{}")
    return {
        "version": package.version,
        "git_commit": direct.get("vcs_info", {}).get("commit_id"),
    }


def installed_versions() -> dict[str, str]:
    return dict(
        sorted(
            (package.metadata["Name"], package.version)
            for package in importlib.metadata.distributions()
        )
    )
