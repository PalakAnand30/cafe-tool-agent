"""One worker owns the shared model and ephemeral customer sessions."""

import os

import uvicorn

from .settings import serve_settings


def main() -> None:
    settings = serve_settings()
    uvicorn.run(
        "cafe_agent.serve.app:create_app",
        factory=True,
        host=os.environ.get("CAFE_HOST", settings.host),
        port=int(os.environ.get("PORT", settings.port)),
        workers=1,
    )


if __name__ == "__main__":
    main()
