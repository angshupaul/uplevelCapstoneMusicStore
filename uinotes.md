# UI Notes

How the Digital Music Store's web chat works.

The code is in two files:

| File | What it holds |
|---|---|
| `chat_service.py` | all graph logic for one chat turn, with no Gradio code, so it can be tested without a browser |
| `app.py` | the Gradio page: layout, event wiring, status badges |

## How to run it

```
python app.py
```

Then open **http://127.0.0.1:7860**. The port comes from `PORT` in `.env` (default 7860).

The app only listens on `127.0.0.1`, so it's reachable from your own machine only. See "Limitations" before exposing it.

The graph, and with it the LLM connection, is only built when the first message is sent, so the page opens instantly.

## The screen

| Part | What it does |
|---|---|
| Header | the title, and one line on what the assistant can help with |
| Chat (`gr.Chatbot`) | the conversation; shows a welcome hint until the first message |
| Message box + **Send** | type and press Enter, or click Send |
| **New conversation** | starts over in a fresh, separate conversation |
| Status badge | what's happening right now (see "Status badges") |
| Examples | four starter messages; clicking one fills the message box, it doesn't send |

## Sessions: one conversation per browser tab

Every browser session gets its own **thread ID**, a random UUID:

```python
thread_id = gr.State(new_thread_id)   # a function, not a value
```

- **Why a function?** Gradio calls it once for every new browser session. Passing a value instead (`gr.State(new_thread_id())`) would give *every* visitor the same ID, and they'd all share one conversation.
- The ID goes into the graph config as `{"configurable": {"thread_id": thread_id}}`. The checkpointer (`MemorySaver`) stores each thread's messages and state separately, so two tabs never see each other's conversation, verification or pending question.
- **One graph for everyone.** `get_graph()` builds the app graph once per process, and all sessions share it. The thread ID keeps them apart, not separate graphs.
- **A browser refresh starts a new session:** new thread ID, empty chat. The old conversation stays in memory until the app restarts.

## One chat turn

Sending a message runs two steps in sequence:

```
Enter / Send
   │
   ▼
add_user_message   (instant, queue=False)
   • shows your message in the chat right away
   • clears and locks the message box, Send and New conversation
   • status: "Processing…"
   │
   ▼
respond            (a generator: updates the page as it goes)
   • status: "Checking your identity…" → "Looking that up…"
   • the reply appears
   • status: "Saving your preferences…"
   • final badge, and the inputs are unlocked again
```

- **Why lock the inputs?** A double-click would otherwise send two turns into the same conversation at the same time. New conversation is locked too, so the thread can't change while an answer is on its way.
- **Why a `pending` state?** Step 1 clears the message box, so step 2 can't read the text from it. The text travels through a hidden `gr.State`.
- **Why the reply comes before "Saving your preferences…":** `create_memory` makes an extra LLM call *after* the answer is ready. `stream_turn` hands the reply over as soon as the supervisor stage finishes, so the customer doesn't wait for preference saving.

### The service layer: `stream_turn`

`app.py` never talks to the graph directly. It iterates `chat_service.stream_turn(graph, thread_id, text)`, which yields:

| Event | Meaning |
|---|---|
| `Progress(stage)` | a step has finished; `stage` describes the next one |
| `Reply(text)` | the text to show in the chat |
| `Done(TurnResult)` | always last: the final `status`, `reply` and `elapsed_seconds` |

`TurnResult.status` is one of:

| Status | When |
|---|---|
| `success` | the question was answered |
| `waiting` | the graph is paused, waiting for an identifier |
| `ended` | verification failed 3 times; this conversation is locked |
| `error` | something failed; the reply is a friendly message, and the technical detail is only logged |
| `ignored` | the message was blank |

`stream_turn` **never raises**, so a failing LLM call can't break the page. `run_turn()` is a shortcut that returns only the final `TurnResult`, used by tests and scripts.

### Only the final reply is shown

The graph's state contains much more than the chat: the verified SystemMessage, supervisor handoffs, tool calls, and both the specialist's answer and the supervisor's reply. The page keeps **its own** chat history and only adds the final reply (`final_reply()`: the latest assistant message without tool calls after the customer's last message). Internal messages, the `customer_id` and saved preferences never appear on screen.

## Interrupt and resume (identity check)

When the customer isn't verified yet, the graph pauses inside `verify_info` with `interrupt()`. The UI handles it like this:

1. **After every turn**, the snapshot is checked:

   ```python
   snapshot = graph.get_state(config)
   # waiting:  snapshot.next == ('verify_info',)
   #           snapshot.interrupts[0].value == {"question": "Could you share your Customer ID…"}
   ```

2. **If it's waiting**, the question is shown as an assistant message, the badge says ✋ "Waiting for your input", and the message box placeholder changes to *"Your Customer ID, email or phone number…"*.
3. **When the customer types again**, `stream_turn` sees the pending question and **resumes** the paused graph instead of starting a new turn:

   ```python
   if pending_question(graph.get_state(config)) is not None:
       payload = Command(resume=text)                  # answer the question
   else:
       payload = {"messages": [HumanMessage(text)]}    # a new message
   ```

4. After verification, the graph answers the customer's **original** question (e.g. "Show my invoices"), not the identifier they typed.

The page keeps no "we are waiting" flag of its own; every turn asks the graph. So the UI can never get out of step with the graph.

### Why it checks `snapshot.interrupts`, not just `snapshot.next`

`snapshot.next` is non-empty whenever the graph has unfinished work. That includes a turn that **crashed** midway (e.g. an LLM timeout inside the supervisor), which leaves pending work but no question. If "waiting" were decided by `snapshot.next` alone, the customer's next message would be sent as a resume value for a question nobody asked. `pending_question()` requires an actual interrupt, so after a crash the next message simply starts a fresh turn. A test proves this: switching back to the `snapshot.next`-only rule makes it fail.

### When verification fails

After 3 failed attempts the conversation ends for good (the attempt limit is per conversation):

- the badge shows ⛔ "Conversation ended. Start a new conversation to try again.";
- the message box and Send are **disabled**, so only New conversation still works;
- even a valid ID sent afterwards would get the same locked-out reply.

## Status badges

The status is a small HTML badge built by `status_html(state, text)`:

| State | Looks like | When |
|---|---|---|
| idle | "Ready." (muted) | on load, after New conversation |
| processing | spinner + stage, e.g. "Looking that up…" | while a turn runs |
| success | ✓ "Answered in 3.2 s" (green) | answered; time is measured to when the reply appeared |
| waiting | ✋ "Waiting for your input: your Customer ID, email or phone number" (amber) | identity question pending |
| error | ⚠ "Something went wrong. Please try again." (red) | a turn failed; the conversation stays usable |
| ended | ⛔ "Conversation ended…" (grey) | verification locked out |

- **Not colour alone:** every state has an icon and words, for colour-blind users and both themes.
- **Screen readers:** `role="status"` and `aria-live="polite"`, so changes are announced.
- **Reduced motion:** the spinner stops for users who turned on "reduce motion".
- **Escaped:** the text goes through `html.escape`, so nothing in it can be read as HTML.
- **Light and dark:** semi-transparent tints and the theme's own text colour (`var(--body-text-color)`). The CSS is `APP_CSS`, passed to `launch(css=APP_CSS)`.

## New conversation

`reset_conversation(graph, old_thread_id)`:

1. **Deletes the old conversation's checkpoints** with `forget_thread()`, which calls `checkpointer.delete_thread()`. This removes the main graph's checkpoints *and* any paused verification subgraph's. Other sessions are untouched.
2. Gives the session a **new thread ID**.
3. Clears the chat and message box, unlocks everything (including after "ended"), and shows "New conversation started."

Saved **music preferences are kept**: they belong to the customer (in the store), not to the conversation, so they come back once the customer verifies again.

## Gradio 6 specifics

Gradio **6.29.1** is installed (`requirements.txt` pins `gradio>=6,<7`). Things that differ from older guides:

| What | Gradio 6 |
|---|---|
| `gr.Chatbot(type="messages")` | **crashes**: `type` no longer exists; the messages format is the only one |
| `gr.Blocks(css=..., theme=...)` | `css` and `theme` moved to **`launch()`** |
| Chat history received by a handler | each message's `content` is a **list of blocks**, e.g. `[{"text": "hi", "type": "text"}]`; appending plain `{"role": ..., "content": "text"}` dicts still works |
| Leave an output unchanged | return `gr.skip()` |
| Concurrency | the default is **1 per event**, meaning everyone waits for everyone; `respond` uses `concurrency_limit=4` (`MAX_CONCURRENT_TURNS`), shared by Enter and Send through `concurrency_id="chat_turn"` |

## Tests

### Offline: `tests/ui_test.py` (29 tests, about 4 seconds, no LLM)

They use the real app graph with the fakes from `app_test.py`, plus a music agent that can be made to fail on purpose.

- **Service:**
  - success with the final reply only; the reply arrives before preferences are saved;
  - waiting, then resume, then success; 3 failures give `ended`, and it stays ended;
  - blank input is ignored; an agent crash gives a friendly error and the next message starts fresh; a broken graph never raises;
  - sessions are isolated; thread IDs are unique; `forget_thread` deletes only its thread.
- **Handlers:**
  - instant echo and input locking, and blank input changes nothing;
  - progress, then reply, then final badge, with the right input state for waiting and ended;
  - a second reply replaces the first; an error never shows technical detail;
  - reset gives a clean page and deletes the old thread.
- **Badges and page:** every state's badge, accessibility attributes, escaping, CSS coverage; the page builds without the graph; each session gets its own ID.

### Live check (real LLM, two sessions)

`python app.py` was driven by `gradio_client`. Each `Client` is a separate session, like two browser tabs.

| Session | Message | Result |
|---|---|---|
| 1 | Show my invoices | verification question · waiting |
| 2 | What AC/DC albums do you have? | its own verification question · waiting |
| 1 | my phone is 780-434-4554 | resumed; the original invoice question answered · 14.7 s |
| 1 | What AC/DC albums do you have? | 2 albums · 4.8 s |
| 1 | Show my invoices and also what jazz albums you have | both answered · 39.5 s |
| 1 | What's the weather like today? | off-topic reply · 0.7 s |
| 2 | my customer ID is 5 | resumed; session 2's own question answered (as customer 5) · 5.0 s |

The message appeared before the reply every time, and session 2 never saw anything from session 1.

### How to run

```
python -m pytest tests/ui_test.py -v
```

## Limitations and follow-ups

- **Mixed questions are slow** (about 40 s): two specialists run one after the other, then the supervisor writes the combined reply. Streaming the reply token by token, or a lighter final step, would help.
- **New conversation resets the verification limit,** because the 3-attempt limit is per conversation.
- **Every handler is also an API endpoint** (`/add_user_message`, `/run_turn_in_page`, …). That's fine while the app only listens on `127.0.0.1`. Before exposing it: set `api_name=False` on the handlers, add `auth=` to `launch()`, and don't use `share=True`. Verification by customer ID alone is weak, so a public link would let anyone try IDs.
- **Everything is in memory:** conversations and preferences are lost when the app restarts.
- **A browser refresh starts a new conversation.**
