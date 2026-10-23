"""Gradio web chat for the Digital Music Store.

Run from the project root:
    python app.py

Each browser session gets its own thread_id (a UUID in gr.State), which is
passed to the graph config so the checkpointer keeps every session's
conversation separate. All graph logic lives in chat_service.py.
"""

import html
from typing import Iterator

import gradio as gr
from langgraph.graph.state import CompiledStateGraph

from chat_service import (
    Done,
    Progress,
    Reply,
    TurnResult,
    forget_thread,
    get_graph,
    new_thread_id,
    stream_turn,
)
from config import get_settings

# Turns answered at the same time across all users. Gradio's default is 1
# (everyone waits for everyone); each turn makes several OpenAI calls, so
# keep this modest to stay clear of rate limits.
MAX_CONCURRENT_TURNS = 4

TITLE = "Digital Music Store: Customer Support"

INTRO = (
    "Ask about our **music catalog** (artists, albums, songs, genres) or your "
    "**invoices and purchases**. You'll be asked to verify your identity with "
    "your Customer ID, email or phone number."
)

PLACEHOLDER = (
    "**Welcome!** Start by telling us who you are and what you need, e.g. "
    "*\"Hi, I'm customer 14. What rock albums do you have?\"*"
)

MESSAGE_PLACEHOLDER = "Type your message and press Enter…"
WAITING_PLACEHOLDER = "Your Customer ID, email or phone number…"
ENDED_PLACEHOLDER = "This conversation has ended. Click New conversation to start again."

EXAMPLES = [
    "Hi, I'm customer 14. What rock albums do you have?",
    "My email is mphilips12@shaw.ca. Show me my invoices.",
    "What albums does AC/DC have?",
    "Show my invoices and also what jazz albums you have",
]


STATUS_ICONS = {"idle": "", "success": "✓", "waiting": "✋", "error": "⚠", "ended": "⛔"}

# Gradio 6 takes css in launch(), not gr.Blocks(). Tinted, semi-transparent
# backgrounds and the theme's text colour keep it readable in light and dark.
APP_CSS = """
#status .status {
  display: inline-flex; align-items: center; gap: .5em;
  padding: .35em .85em; border-radius: 999px; border: 1px solid transparent;
  font-size: .95em; color: var(--body-text-color);
}
#status .status-idle { color: var(--body-text-color-subdued); }
#status .status-processing { background: rgba(59, 130, 246, .12); border-color: rgba(59, 130, 246, .4); }
#status .status-success { background: rgba(34, 197, 94, .12); border-color: rgba(34, 197, 94, .45); }
#status .status-waiting { background: rgba(245, 158, 11, .14); border-color: rgba(245, 158, 11, .5); }
#status .status-error { background: rgba(239, 68, 68, .12); border-color: rgba(239, 68, 68, .5); }
#status .status-ended { background: rgba(107, 114, 128, .14); border-color: rgba(107, 114, 128, .5); }
#status .status-spinner {
  width: .9em; height: .9em; border-radius: 50%;
  border: 2px solid currentColor; border-right-color: transparent;
  animation: status-spin .8s linear infinite;
}
@keyframes status-spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) {
  #status .status-spinner { animation: none; border-right-color: currentColor; opacity: .6; }
}
"""


def status_html(state: str, text: str) -> str:
    """The status badge: an icon and words (never colour alone).

    role="status" / aria-live let screen readers announce changes. The text
    is escaped, so nothing in it can be read as HTML.
    """
    if state == "processing":
        icon = '<span class="status-spinner" aria-hidden="true"></span>'
    elif STATUS_ICONS.get(state):
        icon = f'<span class="status-icon" aria-hidden="true">{STATUS_ICONS[state]}</span>'
    else:
        icon = ""
    return (
        f'<div class="status status-{state}" role="status" aria-live="polite">'
        f"{icon}<span>{html.escape(text)}</span></div>"
    )


IDLE_STATUS = status_html("idle", "Ready.")


def final_status(result: TurnResult) -> str:
    """The status badge once a turn has finished."""
    if result.status == "success":
        return status_html("success", f"Answered in {result.elapsed_seconds:.1f} s")
    if result.status == "waiting":
        return status_html("waiting", "Waiting for your input: your Customer ID, email or phone number")
    if result.status == "ended":
        return status_html("ended", "Conversation ended. Start a new conversation to try again.")
    if result.status == "error":
        return status_html("error", "Something went wrong. Please try again.")
    return IDLE_STATUS


def _inputs(enabled: bool) -> tuple:
    """Updates for the textbox, Send and New conversation."""
    return gr.Textbox(interactive=enabled), gr.Button(interactive=enabled), gr.Button(interactive=enabled)


def _inputs_after(status: str) -> tuple:
    """Textbox, Send and New conversation once a turn has finished.

    Waiting: the placeholder asks for an identifier. Ended (locked out):
    only New conversation stays usable, since every message would just get
    the same reply. Driven by the graph's state each turn; the page keeps no
    flag of its own.
    """
    if status == "ended":
        return (
            gr.Textbox(interactive=False, placeholder=ENDED_PLACEHOLDER),
            gr.Button(interactive=False),
            gr.Button(interactive=True),
        )
    placeholder = WAITING_PLACEHOLDER if status == "waiting" else MESSAGE_PLACEHOLDER
    return gr.Textbox(interactive=True, placeholder=placeholder), gr.Button(interactive=True), gr.Button(interactive=True)


def add_user_message(text: str, history: list) -> tuple:
    """Step 1 (runs at once): show the customer's message and lock the input.

    Outputs: chatbot, textbox, pending text, Send, New conversation, status.
    Blank input changes nothing.
    """
    history = list(history or [])
    text = (text or "").strip()
    if not text:
        return history, gr.skip(), "", gr.skip(), gr.skip(), gr.skip()
    history.append({"role": "user", "content": text})
    _, locked_send, locked_reset = _inputs(False)
    processing = status_html("processing", "Processing…")
    return history, gr.Textbox(value="", interactive=False), text, locked_send, locked_reset, processing


def respond(graph: CompiledStateGraph, thread_id: str, text: str, history: list) -> Iterator[tuple]:
    """Step 2: run the turn and stream progress and the reply into the page.

    Yields: chatbot, status, textbox, Send, New conversation.
    """
    history = list(history or [])
    if not (text or "").strip():
        yield history, gr.skip(), *_inputs(True)
        return

    replied = False
    for event in stream_turn(graph, thread_id, text):
        if isinstance(event, Progress):
            yield history, status_html("processing", event.stage), gr.skip(), gr.skip(), gr.skip()
        elif isinstance(event, Reply):
            message = {"role": "assistant", "content": event.text}
            if replied:  # a corrected reply replaces the first, never a second bubble
                history[-1] = message
            else:
                history.append(message)
                replied = True
            yield history, gr.skip(), gr.skip(), gr.skip(), gr.skip()
        elif isinstance(event, Done):
            result = event.result
            if not replied and result.reply:  # e.g. the friendly error message
                history.append({"role": "assistant", "content": result.reply})
            yield history, final_status(result), *_inputs_after(result.status)


def reset_conversation(graph: CompiledStateGraph, old_thread_id: str) -> tuple:
    """New conversation: a fresh thread for this browser session only.

    Deletes the old conversation's checkpoints, then returns: new thread id,
    empty chat, cleared and enabled textbox, enabled Send, empty pending
    text, idle status.
    """
    forget_thread(graph, old_thread_id)
    return (
        new_thread_id(),
        [],
        gr.Textbox(value="", interactive=True, placeholder=MESSAGE_PLACEHOLDER),
        gr.Button(interactive=True),
        "",
        status_html("idle", "New conversation started."),
    )


def build_ui(graph: CompiledStateGraph | None = None) -> gr.Blocks:
    """Build the chat page.

    ``graph`` is for tests; by default the real app graph is built on first
    use (get_graph), so building the page needs no API key and opens fast.
    """

    def current_graph() -> CompiledStateGraph:
        return graph if graph is not None else get_graph()

    with gr.Blocks(title=TITLE, fill_height=True) as demo:
        gr.Markdown(f"# {TITLE}\n{INTRO}")

        # A callable default: Gradio calls it for every new browser session,
        # so each session gets its own conversation thread.
        thread_id = gr.State(new_thread_id)

        chatbot = gr.Chatbot(
            label="Conversation",
            placeholder=PLACEHOLDER,
            height=480,
            resizable=True,
        )
        with gr.Row():
            message = gr.Textbox(
                placeholder=MESSAGE_PLACEHOLDER,
                show_label=False,
                scale=5,
                autofocus=True,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        with gr.Row():
            new_conversation = gr.Button("New conversation", variant="secondary")
        status = gr.HTML(IDLE_STATUS, elem_id="status")

        gr.Examples(examples=EXAMPLES, inputs=message, label="Try an example")

        # The text being answered: step 1 clears the textbox, so step 2
        # can't read it from there.
        pending = gr.State("")

        def run_turn_in_page(thread, text, history):
            yield from respond(current_graph(), thread, text, history)

        for trigger in (message.submit, send.click):
            trigger(
                add_user_message,
                inputs=[message, chatbot],
                outputs=[chatbot, message, pending, send, new_conversation, status],
                queue=False,
            ).then(
                run_turn_in_page,
                inputs=[thread_id, pending, chatbot],
                outputs=[chatbot, status, message, send, new_conversation],
                concurrency_limit=MAX_CONCURRENT_TURNS,
                concurrency_id="chat_turn",  # one shared limit for Enter and Send
            )

        new_conversation.click(
            lambda old_thread: reset_conversation(current_graph(), old_thread),
            inputs=[thread_id],
            outputs=[thread_id, chatbot, message, send, pending, status],
            queue=False,
        )

    demo.components = {  # for the event wiring and tests
        "thread_id": thread_id,
        "chatbot": chatbot,
        "message": message,
        "send": send,
        "new_conversation": new_conversation,
        "status": status,
        "pending": pending,
    }
    demo.current_graph = current_graph
    return demo


if __name__ == "__main__":
    settings = get_settings()
    build_ui().queue().launch(
        server_name=settings.host,  # 127.0.0.1 unless HOST says otherwise (Docker: 0.0.0.0)
        server_port=settings.port,
        css=APP_CSS,
    )
