# Chinook In-Memory SQLite Setup — Steps

This document describes how the Chinook database is loaded into an in-memory
SQLite database with SQLAlchemy. These steps were written before implementation
and updated to reflect the actual code.

## 1. Source file

- Path: `DBSet/Chinook_Sqlite.sql` (Chinook v1.4.5, ~595 KB, UTF-8)
- Contents:
  - `DROP TABLE IF EXISTS` statements
  - `CREATE TABLE` statements for 11 tables: `Album`, `Artist`, `Customer`,
    `Employee`, `Genre`, `Invoice`, `InvoiceLine`, `MediaType`, `Playlist`,
    `PlaylistTrack`, `Track`
  - `CREATE INDEX` statements for the foreign key columns
  - 24 multi-row `INSERT INTO ... VALUES (...), (...)` statements

## 2. Prerequisites

- Python 3.10+ (tested with Python 3.14)
- SQLAlchemy 2.x: `pip install -r requirements.txt`

## 3. Design decisions

| Concern | Decision | Reason |
|---|---|---|
| Database location | `sqlite://` (in-memory) | Fast, no file left on disk |
| Connection pooling | `StaticPool` + `check_same_thread=False` | Each new SQLite connection to `:memory:` creates a **separate, empty** database. `StaticPool` reuses one connection, so every session sees the loaded data. |
| Running the script | Raw DBAPI `executescript()` | The file holds many statements. SQLAlchemy's `text()` runs only one statement per call, and splitting on `;` is unreliable because the data contains semicolons inside string literals. |
| File encoding | Read with `encoding="utf-8-sig"` | Handles a possible BOM and the non-ASCII artist and track names |
| Foreign keys | Enable `PRAGMA foreign_keys=ON` **after** loading | The script creates `Album` before `Artist` and inserts rows in an order that can break FK checks during load |
| Engine caching | `@lru_cache(maxsize=1)` on `get_engine()` | Ensures the script is loaded exactly once; every caller gets the same engine and sees the same in-memory data |
| Session factory caching | `@lru_cache(maxsize=1)` on `_session_factory()` | One `sessionmaker` instance is reused; individual sessions are still created per call to `get_session()` |
| Query execution | `execute_query(sql, params)` returns JSON | Provides a safe, parameterized interface for LLM-generated SQL; values are bound, never interpolated |
| JSON serialization | `_json_default` fallback | `Decimal`, `datetime`/`date`/`time`, and `bytes`/`bytearray`/`memoryview` values from SQLite are converted to JSON-safe types |

## 4. Implementation steps

1. **Find the SQL file** relative to the project root
   (`DBSet/Chinook_Sqlite.sql`) so the script works from any working directory.
2. **Create the engine:**
   ```python
   engine = create_engine(
       "sqlite://",
       connect_args={"check_same_thread": False},
       poolclass=StaticPool,
   )
   ```
3. **Read the SQL script** as UTF-8 text (with `encoding="utf-8-sig"`).
4. **Run the script** through the raw DBAPI connection:
   ```python
   raw = engine.raw_connection()
   try:
       raw.driver_connection.executescript(sql_script)
       raw.commit()
   finally:
       raw.close()   # returns the connection to StaticPool; it stays open
   ```
5. **Turn on foreign key enforcement** with `PRAGMA foreign_keys = ON`.
6. **Check the load:**
   - Use `sqlalchemy.inspect(engine).get_table_names()` to confirm all 11 tables exist.
   - Count the rows in each table and compare them with the expected Chinook counts
     (Artist 275, Album 347, Track 3503, Customer 59, Employee 8,
     Invoice 412, InvoiceLine 2240, Genre 25, MediaType 5, Playlist 18,
     PlaylistTrack 8715).
   - Run a sample join query (top-selling artists) to confirm the relationships work.
7. **Expose helpers** for the rest of the project:
   - `get_engine()` — returns the loaded engine (created once, cached with `@lru_cache`).
   - `get_session()` — returns a new SQLAlchemy `Session` bound to that engine.
     Use as a context manager: `with get_session() as session:`.
   - `get_metadata()` — returns `MetaData` reflected from the loaded database,
     giving access to `Table` objects for SQLAlchemy Core queries.
   - `execute_query(sql, params)` — runs a parameterized SQL statement and returns
     the result as a JSON string (see §4a below).

### 4a. `execute_query` contract

```python
execute_query(sql: str, params: Mapping[str, Any] | None = None) -> str
```

Parameters are bound by name, e.g.:
```python
execute_query("SELECT * FROM Artist WHERE ArtistId = :id", {"id": 1})
```

Return value (always valid JSON):

| Case | Shape |
|---|---|
| Row-returning query | `{"columns": [...], "rows": [{...}], "row_count": n}` |
| Non-row statement | `{"rows_affected": n}` |
| SQLAlchemy error | `{"error": "..."}` |

Special Python types serialised automatically: `Decimal` → `float`;
`datetime`/`date`/`time` → ISO-8601 string; `bytes`/`bytearray`/`memoryview` → hex string.

## 5. Files

| File | Purpose |
|---|---|
| `DB_SETUP_STEPS.md` | This document |
| `requirements.txt` | Python dependencies (`SQLAlchemy>=2.0`) |
| `database.py` | Engine creation, script loading, session helpers, `execute_query` |
| `verify_db.py` | Checks all 11 table counts and runs a top-artists join query |
| `self_check_customer_count.py` | Unit-style self-check for `execute_query`; cross-validates against SQLAlchemy Core count via `get_metadata()` |

## 6. How to run

```powershell
pip install -r requirements.txt
python verify_db.py
python self_check_customer_count.py
```
