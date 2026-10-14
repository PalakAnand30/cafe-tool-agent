"""Café API factory, with no database or server created at import time."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from cafe_agent.config import load_runtime
from cafe_agent.domain.models import ErrorCode, ErrorResult
from cafe_agent.tools.schemas import CheckAvailabilityArgs, CreateOrderArgs

from .service import CafeService, error


def create_app(service: CafeService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.cafe = service or CafeService(database=load_runtime().api.database_path)
        try:
            yield
        finally:
            if service is None:
                app.state.cafe.close()

    app = FastAPI(title="Café ordering API", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        result = error(
            ErrorCode.INVALID_ARGUMENTS,
            "Request failed validation.",
            details={"fields": [".".join(map(str, e["loc"])) for e in exc.errors()]},
        )
        return JSONResponse(status_code=422, content=result.model_dump(mode="json"))

    def invoke(request: Request, name: str, arguments: dict[str, Any]) -> JSONResponse:
        result = request.app.state.cafe.call(name, arguments)
        status = 200
        if isinstance(result, ErrorResult):
            code = result.error.code
            status = (
                404
                if code
                in {ErrorCode.ITEM_NOT_FOUND, ErrorCode.ORDER_NOT_FOUND, ErrorCode.UNKNOWN_TOOL}
                else 422
            )
            if code in {
                ErrorCode.OUT_OF_STOCK,
                ErrorCode.CANNOT_CANCEL,
                ErrorCode.ORDER_LIMIT_REACHED,
            }:
                status = 409
        return JSONResponse(status_code=status, content=result.model_dump(mode="json"))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/tools/{name}")
    def tool(
        request: Request, name: str, arguments: Annotated[dict[str, Any], Body()]
    ) -> JSONResponse:
        return invoke(request, name, arguments)

    @app.get("/menu")
    def menu(request: Request, category: str | None = None) -> JSONResponse:
        return invoke(request, "get_menu", {} if category is None else {"category": category})

    @app.post("/availability")
    def availability(request: Request, arguments: CheckAvailabilityArgs) -> JSONResponse:
        return invoke(request, "check_availability", arguments.model_dump())

    @app.post("/orders")
    def order(request: Request, arguments: CreateOrderArgs) -> JSONResponse:
        return invoke(request, "create_order", arguments.model_dump())

    @app.get("/orders/{order_id}")
    def status(request: Request, order_id: str) -> JSONResponse:
        return invoke(request, "get_order_status", {"order_id": order_id})

    @app.post("/orders/{order_id}/cancel")
    def cancel(request: Request, order_id: str) -> JSONResponse:
        return invoke(request, "cancel_order", {"order_id": order_id})

    return app
