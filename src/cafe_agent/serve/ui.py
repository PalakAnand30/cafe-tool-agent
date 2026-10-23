"""Chat, complete tool traces and an order board backed by the same chat service."""

import os
from html import escape
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import ValidationError

from .service import ChatRequest, SessionError
from .settings import ServeSettings


def order_rows(state: dict[str, Any]) -> list[list[str]]:
    return [
        [
            order["order_id"],
            order["status"],
            ", ".join(
                f"{line['quantity']} × {line.get('size') or ''} {line['item_name']}"
                + (f" ({', '.join(line['modifiers'])})" if line["modifiers"] else "")
                for line in order["items"]
            ),
            order["total"],
        ]
        for order in state["orders"]
    ]


def order_board(state: dict[str, Any]) -> str:
    rows = order_rows(state)
    if not rows:
        return "<p>No orders yet.</p>"
    headings = "".join(f"<th>{value}</th>" for value in ["Order", "Status", "Items", "Total (GBP)"])
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return (
        f'<table style="width:100%;text-align:left"><thead><tr>{headings}</tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def mount_ui(app: FastAPI, settings: ServeSettings) -> FastAPI:
    # Set before Gradio's lazy import; no analytics or public sharing tunnel.
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
    import gradio as gr

    def submit(message: str, history: list[dict] | None, session_id: str | None) -> tuple:
        try:
            result = app.state.chat.chat(ChatRequest(session_id=session_id, message=message))
        except SessionError as exc:
            raise gr.Error(exc.code.replace("_", " ").capitalize()) from None
        except ValidationError:
            raise gr.Error("Enter a message before sending.") from None
        history = [
            *(history or []),
            {"role": "user", "content": message},
            {"role": "assistant", "content": result.reply},
        ]
        state = result.state.model_dump(mode="json")
        return (
            "",
            history,
            result.session_id,
            [entry.model_dump(mode="json") for entry in result.trace],
            order_board(state),
            f"Café time: {state['minute']} min",
            result.error or "",
        )

    def reset(session_id: str | None) -> tuple:
        if session_id:
            app.state.chat.reset(session_id)
        return "", [], None, [], order_board({"orders": []}), "Café time: 0 min", ""

    def advance(session_id: str | None) -> tuple:
        if not session_id:
            return order_board({"orders": []}), "Café time: 0 min"
        try:
            state = app.state.chat.advance(session_id)
        except SessionError as exc:
            raise gr.Error(exc.code.replace("_", " ").capitalize()) from None
        return order_board(state), f"Café time: {state['minute']} min"

    with gr.Blocks(title="The café counter", analytics_enabled=False) as demo:
        gr.Markdown("# The café counter\nOrder a drink, browse the menu, or ask about your order.")
        gr.Markdown(
            "A simulated café. Your session has its own orders. CPU replies can take a moment."
        )
        session_id = gr.State(None)
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(
                    type="messages",
                    label="Your conversation",
                    height=420,
                    allow_tags=False,
                    allow_file_downloads=False,
                )
                message = gr.Textbox(
                    label="Message", placeholder="One medium latte, please.", max_lines=3
                )
                send = gr.Button("Send", variant="primary")
                gr.Examples(
                    [[value] for value in settings.examples], inputs=message, label="Try a request"
                )
                problem = gr.Textbox(label="Request status", interactive=False)
            with gr.Column(scale=2):
                gr.Markdown("### Your orders")
                board = gr.HTML(order_board({"orders": []}))
                clock = gr.Markdown("Café time: 0 min")
                tick = gr.Button(f"Advance café time {settings.advance_minutes} minutes")
                clear = gr.Button("Start a new café session")
                with gr.Accordion("Tool calls", open=False):
                    trace = gr.JSON(label="Calls and results")
        outputs = [message, chat, session_id, trace, board, clock, problem]
        for event in [send.click, message.submit]:
            event(
                submit,
                [message, chat, session_id],
                outputs,
                api_name=False,
                concurrency_id="cafe",
                concurrency_limit=1,
            )
        clear.click(
            reset,
            session_id,
            [message, chat, session_id, trace, board, clock, problem],
            api_name=False,
            concurrency_id="cafe",
            concurrency_limit=1,
        )
        tick.click(
            advance,
            session_id,
            [board, clock],
            api_name=False,
            concurrency_id="cafe",
            concurrency_limit=1,
        )
    demo.queue(max_size=settings.queue_size, default_concurrency_limit=1)
    return gr.mount_gradio_app(
        app,
        demo,
        path="/demo",
        show_api=False,
        show_error=False,
        blocked_paths=[
            str(Path(name).resolve()) for name in [".env", ".git", "models", "data", "results"]
        ],
    )
