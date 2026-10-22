"""The Digital Music Store application graph.

  START -> verify_info -(verified)-> load_memory -> supervisor -> create_memory -> END
                       \\-(not verified)-> END

- verify_info is verification.py's flow (add_verification_flow), compiled as
  a subgraph: verify_customer, ask_for_identifier, human_input (interrupt)
  and verification_failed. It pauses with interrupt() until the customer is
  verified or runs out of attempts, and resumes where it paused.
- The edge after verify_info only lets a verified customer_id through, so
  an unverified user can never reach memory or either agent.
- The verified SystemMessage and customer_id stay in state, so the
  supervisor, the invoice agent and its tools all receive them.

Run every turn on the same thread_id; resume a pause with Command(resume=...):

    app = build_app()
    config = {"configurable": {"thread_id": session_id}}
    out = app.invoke({"messages": [("user", text)]}, config)
    if "__interrupt__" in out:
        question = out["__interrupt__"][0].value["question"]
        out = app.invoke(Command(resume=next_text), config)
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from memory import create_memory, load_memory
from supervisor import SupervisorState, build_supervisor
from verification import VerificationState, add_verification_flow


class AppState(SupervisorState):
    # Failed verification attempts; see verification.MAX_VERIFICATION_ATTEMPTS.
    verification_attempts: int


def build_verify_info() -> CompiledStateGraph:
    """verification.py's flow as one stage that ends when it is done."""
    builder = StateGraph(VerificationState)
    add_verification_flow(builder, next_node=END)
    return builder.compile(name="verify_info")


def after_verify_info(state: AppState) -> str:
    """Only a verified customer goes on; everyone else stops here."""
    return "load_memory" if state.get("customer_id") is not None else END


def build_app(
    supervisor: CompiledStateGraph | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Compile the application graph.

    A checkpointer is required for interrupt()/resume, so one is always
    used (in-memory by default). The store keeps memory across
    conversations. Tests pass a supervisor built with fakes.
    """
    builder = StateGraph(AppState)
    builder.add_node("verify_info", build_verify_info())
    builder.add_node("load_memory", load_memory)
    builder.add_node("supervisor", supervisor or build_supervisor())
    builder.add_node("create_memory", create_memory)

    builder.add_edge(START, "verify_info")
    builder.add_conditional_edges("verify_info", after_verify_info, ["load_memory", END])
    builder.add_edge("load_memory", "supervisor")
    builder.add_edge("supervisor", "create_memory")
    builder.add_edge("create_memory", END)

    return builder.compile(
        checkpointer=checkpointer or MemorySaver(),
        store=store if store is not None else InMemoryStore(),
        name="music_store_app",
    )
