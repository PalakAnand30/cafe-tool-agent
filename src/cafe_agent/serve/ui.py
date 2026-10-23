"""A warm café counter with menu cards, conversation and a live receipt."""

import os
from functools import partial
from pathlib import Path

from fastapi import FastAPI
from pydantic import ValidationError

from .presentation import ASSETS, hero, masthead, menu_cards, order_board
from .service import ChatRequest, SessionError
from .settings import ServeSettings


def mount_ui(app: FastAPI, settings: ServeSettings) -> FastAPI:
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
    import gradio as gr

    def submit(message: str, history: list[dict] | None, session_id: str | None) -> tuple:
        try:
            result = app.state.chat.chat(ChatRequest(session_id=session_id, message=message))
        except SessionError as exc:
            notices = {
                "SESSION_EXPIRED": "This session has ended. Choose Start fresh to begin again.",
                "TURN_LIMIT_RESET_REQUIRED": "Choose Start fresh to begin a new café session.",
                "MESSAGE_TOO_LONG": "Try a shorter message so we can help with your order.",
                "SESSION_CAPACITY": "The counter is busy. Please try again in a moment.",
            }
            raise gr.Error(notices.get(exc.code, "Please start a new café session.")) from None
        except ValidationError:
            raise gr.Error("Enter a message before sending.") from None
        history = [
            *(history or []),
            {"role": "user", "content": message},
            {"role": "assistant", "content": result.reply},
        ]
        state = result.state.model_dump(mode="json")
        trace = {
            "source": result.source,
            "error": result.error,
            "calls": [entry.model_dump(mode="json") for entry in result.trace],
        }
        notice = "Please check your receipt, then choose Start fresh." if result.error else ""
        return (
            "",
            history,
            result.session_id,
            trace,
            order_board(state),
            f"Café clock · {state['minute']} min",
            notice,
        )

    def reset(session_id: str | None) -> tuple:
        if session_id:
            app.state.chat.reset(session_id)
        return "", [], None, {}, order_board({"orders": []}), "Café clock · 0 min", ""

    def advance(session_id: str | None) -> tuple:
        if not session_id:
            return order_board({"orders": []}), "Café clock · 0 min"
        try:
            state = app.state.chat.advance(session_id)
        except SessionError:
            raise gr.Error("This session has ended. Choose Start fresh to begin again.") from None
        return order_board(state), f"Café clock · {state['minute']} min"

    theme = gr.themes.Base(
        font=["Arial", "Helvetica", "sans-serif"], font_mono=["Courier New", "monospace"]
    )
    with gr.Blocks(
        title="The café counter",
        theme=theme,
        css=(ASSETS / "theme.css").read_text(),
        head="<style>" + (ASSETS / "shell.css").read_text() + "</style>",
        analytics_enabled=False,
    ) as demo:
        session_id = gr.State(None)
        gr.HTML(masthead(), elem_id="cafe-masthead")
        gr.HTML(hero(), elem_id="cafe-hero")
        with gr.Row(elem_id="cafe-main"):
            with gr.Column(scale=7, elem_id="counter-main"):
                gr.HTML(
                    '<div class="section-heading"><h2>Find your little favourite.</h2>'
                    "<span>Our menu, at a glance</span></div>"
                )
                choices = [
                    ("featured", "Favourites"),
                    ("coffee", "Coffee"),
                    ("tea", "Tea"),
                    ("cold_drinks", "Cold drinks"),
                    ("food", "Food"),
                ]
                with gr.Row(elem_id="menu-tabs"):
                    tabs = [
                        gr.Button(
                            label,
                            variant="primary" if key == "featured" else "secondary",
                            min_width=65,
                        )
                        for key, label in choices
                    ]
                gallery = gr.HTML(
                    menu_cards("featured", settings.featured_items), elem_id="cafe-menu"
                )

                def choose_menu(group: str) -> tuple:
                    return (
                        menu_cards(group, settings.featured_items),
                        *(
                            gr.update(variant="primary" if key == group else "secondary")
                            for key, _ in choices
                        ),
                    )

                for (key, _), tab in zip(choices, tabs, strict=True):
                    tab.click(
                        partial(choose_menu, key),
                        outputs=[gallery, *tabs],
                        api_name=False,
                        queue=False,
                    )
                gr.HTML(
                    '<div class="chat-heading"><div><strong>A seat at the counter.</strong>'
                    "<small>What can we get started for you?</small></div>"
                    '<span class="assistant-badge">Café assistant</span></div>',
                    elem_id="chat-heading",
                )
                chat = gr.Chatbot(
                    type="messages",
                    show_label=False,
                    height=320,
                    elem_id="cafe-chat",
                    allow_tags=False,
                    allow_file_downloads=False,
                    placeholder="A warm welcome. What are you in the mood for?",
                )
                with gr.Row(elem_id="cafe-composer"):
                    message = gr.Textbox(
                        label="Your message",
                        show_label=False,
                        scale=8,
                        placeholder="A coffee, a bite, or a little help choosing…",
                        lines=1,
                        max_lines=3,
                        elem_id="cafe-message",
                        container=False,
                    )
                    send = gr.Button(
                        "Send ↗", variant="primary", scale=1, min_width=89, elem_id="cafe-send"
                    )
                problem = gr.Markdown("", elem_id="cafe-problem")
                with gr.Column(elem_id="cafe-examples"):
                    gr.Examples(
                        [[value] for value in settings.examples],
                        inputs=message,
                        label="Not sure where to start? Try a little inspiration.",
                    )
            with gr.Column(scale=3, elem_id="counter-sidebar"):
                gr.HTML(
                    '<div class="order-heading"><h2>Your little order.</h2>'
                    "<span>THE RECEIPT</span></div>",
                    elem_id="order-heading",
                )
                board = gr.HTML(order_board({"orders": []}), elem_id="cafe-orders")
                clock = gr.Markdown("Café clock · 0 min", elem_id="cafe-clock")
                tick = gr.Button(
                    f"Let {settings.advance_minutes} minutes pass", elem_id="cafe-tick"
                )
                gr.Markdown(
                    "Let a little time pass to see your order being prepared.",
                    elem_classes=["sidebar-note"],
                )
                clear = gr.Button("Start fresh", elem_id="cafe-reset")
                with gr.Accordion("Behind the counter", open=False, elem_id="cafe-details"):
                    gr.Markdown("Menu reads come straight from the café. Orders use the assistant.")
                    trace = gr.JSON(label="Calls & results", value={})
        gr.HTML(
            '<div class="demo-footer">A simulated café. A real little coffee moment.'
            " &nbsp;·&nbsp; No payments or real orders.</div>"
        )
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
            reset, session_id, outputs, api_name=False, concurrency_id="cafe", concurrency_limit=1
        )
        tick.click(
            advance,
            session_id,
            [board, clock],
            api_name=False,
            concurrency_id="cafe",
            concurrency_limit=1,
        )
        chat.clear(
            reset,
            session_id,
            outputs,
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
