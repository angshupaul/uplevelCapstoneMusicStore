# Technical Specification

**Project:** Digital Music Store - AI Multi-Agent Customer Support Assistant
**Audience:** For anyone who wants to learn multi-agent AI application. See [ARCHITECTURE.md](ARCHITECTURE.md) for the diagrams.

## 1. Goal

Build a chat assistant that can:

| # | Requirement |
|---|---|
| R1 | Verify the customer by Customer ID, email or phone before helping them |
| R2 | Answer music catalog questions in natural language |
| R3 | Show the verified customer's own invoices and purchases, and nobody else's |
| R4 | Remember the customer's music preferences between conversations |
| R5 | Answer only from database results - never make up facts |
| R6 | Politely decline off-topic questions |

## 2. Technology stack

| Purpose | Technology |
|---|---|
| Language | Python 3.12 |
| Agent workflow | LangGraph, `langgraph-supervisor` |
| LLM access | LangChain + OpenAI (`ChatOpenAI`, e.g. `gpt-4o-mini`) |
| Database | SQLite (in memory) via SQLAlchemy, Chinook sample data |
| Data validation | Pydantic |
| Web UI | Gradio 6 |
| Tests | pytest (offline with fake LLMs, plus optional live tests) |
| Packaging | Docker |

## 3. Components

| Component | File | Responsibility |
|---|---|---|
| Web UI | `app.py` | Chat page, status badge, "New conversation" button |
| Chat service | `chat_service.py` | Runs one turn: new message or resume after a pause; streams the reply |
| App graph | `graph.py` | Connects the stages: verify → load memory → supervisor → save memory |
| Verification | `verification.py` | Extracts the identifier, looks up the customer, max 3 attempts |
| Supervisor | `supervisor.py` | Classifies the request and routes it to the agents |
| Music agent | `music_agent.py` | Answers catalog questions using catalog tools |
| Invoice agent | `invoice_agent.py` | Answers account questions using account tools |
| Memory | `memory.py` | Loads and saves music preferences |
| Tools | `tools.py` | SQL-backed functions the agents call |
| Database | `database.py` | Loads Chinook into memory; runs safe, parameterized queries |
| Config / LLM | `config.py`, `llm.py` | Reads `.env`; creates one shared chat model |

## 4. Agents and their tools

**Music agent** (catalog - anyone verified may ask)

| Tool | What it does |
|---|---|
| `search_albums_by_artist` | Albums by an artist |
| `search_tracks_by_artist` | Songs by an artist |
| `browse_tracks_by_genre` | Sample of songs in a genre |
| `search_songs_by_title` | Find songs by name |
| `get_track_details` | Album, genre, length and price of one song |

**Invoice agent** (account - only the verified customer's data)

| Tool | What it does |
|---|---|
| `get_customer_invoices` | List the customer's invoices and totals |
| `get_customer_purchased_tracks` | Songs the customer has bought |
| `get_invoice_line_items` | Items in one invoice |
| `get_invoice_support_rep` | The support employee for an invoice |

**Supervisor routing rules**

| Request is about | Goes to |
|---|---|
| Music catalog | Music agent |
| Own account | Invoice agent |
| Both | Invoice agent first, then music agent |
| Anything else | Fixed polite reply, no agent runs |

## 5. Shared state (what the graph remembers during a conversation)

| Field | Meaning |
|---|---|
| `messages` | The chat history |
| `customer_id` | Set only after successful verification |
| `verification_attempts` | Failed tries so far (limit 3) |
| `route` | Which agents this request needs, in order |
| `loaded_memory` | The customer's saved preferences, for the music agent |

## 6. Data

- **Chinook database** (`DBSet/Chinook_Sqlite.sql`): Artist, Album, Track, Genre, Customer, Invoice, InvoiceLine, Employee, and more. Loaded into memory at startup.
- **Conversation memory:** LangGraph `MemorySaver`, one thread per browser tab.
- **Preference memory:** LangGraph `InMemoryStore`, keyed by `("memory_profile", customer_id)`.

Example saved profile:

```json
{ "customer_id": 14, "music_preferences": ["jazz", "AC/DC"] }
```

## 7. Safety and quality rules

- **Privacy:** account tools receive the *verified* `customer_id` from graph state. The LLM cannot see or change it.
- **Grounding:** agents use only tool results, quote numbers exactly, and say clearly when nothing is found or results are partial.
- **Safe SQL:** all queries use parameters, never string-built user input.
- **Memory hygiene:** only stated likes are saved (not questions, dislikes or instructions); identifiers are never sent to the memory extractor.
- **Secrets:** the API key lives in `.env`, which is git-ignored.

## 8. Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` (or `LLM_API_KEY`) | yes | - |
| `OPENAI_CHAT_MODEL` (or `LLM_MODEL`) | yes | - |
| `LLM_TEMPERATURE` | no | `0.0` |
| `HOST` / `PORT` | no | `127.0.0.1` / `7860` |

## 9. Run and test

```bash
pip install -r requirements.txt
python app.py                        # open http://127.0.0.1:7860
python -m pytest tests/ -v           # offline tests, no API key needed
python -m pytest -m live -v -s       # live LLM checks (small cost)
```

Try it with customer **14** (Mark Philips): "my customer ID is 14", then "What AC/DC albums do you have?"

## 10. Known limitations

- Everything is in memory: restarting the app clears chats and preferences.
- Verification is basic (knowing an ID, email or phone is enough) - not real authentication.
- Mixed questions are slower because two agents run one after the other.
- Preferences can be added but not removed.

## 11. Suggested classroom extensions

1. Save preferences to a real database so they survive restarts.
2. Add a "forget my preferences" feature.
3. Add a third agent (for example, a refunds agent) and a new routing rule.
4. Replace basic verification with a login.
