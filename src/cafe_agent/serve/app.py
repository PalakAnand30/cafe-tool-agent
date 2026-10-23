"""FastAPI chat endpoints and a mounted Gradio café demo."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from .model import load_client
from .service import ChatModel, ChatRequest, ChatResponse, ChatService, SessionError, SessionRequest
from .settings import ServeSettings, serve_settings


def create_app(
    *,
    client: ChatModel | None = None,
    settings: ServeSettings | None = None,
    with_ui: bool = True,
) -> FastAPI:
    settings = settings or serve_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = client or load_client(settings)
        app.state.chat = ChatService(engine, settings)
        try:
            yield
        finally:
            app.state.chat.close()
            if client is None:
                engine.close()

    app = FastAPI(title="Café ordering assistant", lifespan=lifespan)

    @app.exception_handler(SessionError)
    async def session_error(request: Request, exc: SessionError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"error": exc.code})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": "INVALID_REQUEST"})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": "llama.cpp", "n_gpu_layers": 0}

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        return request.app.state.chat.chat(body)

    @app.post("/session/advance")
    def advance(body: SessionRequest, request: Request) -> dict:
        return request.app.state.chat.advance(body.session_id)

    @app.post("/session/reset", status_code=204)
    def reset(body: SessionRequest, request: Request) -> Response:
        request.app.state.chat.reset(body.session_id)
        return Response(status_code=204)

    if with_ui:
        from .ui import mount_ui

        @app.get("/", include_in_schema=False)
        def index() -> RedirectResponse:
            return RedirectResponse("/demo/")

        return mount_ui(app, settings)
    return app
