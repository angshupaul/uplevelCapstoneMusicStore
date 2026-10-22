"""Load the Chinook database into an in-memory SQLite instance via SQLAlchemy."""

import json
import re
from datetime import date, datetime, time
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import Engine, MetaData, create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = Path(__file__).resolve().parent
SQL_FILE = PROJECT_ROOT / "DBSet" / "Chinook_Sqlite.sql"

_NON_DIGITS = re.compile(r"\D")

EXPECTED_TABLES = (
    "Album",
    "Artist",
    "Customer",
    "Employee",
    "Genre",
    "Invoice",
    "InvoiceLine",
    "MediaType",
    "Playlist",
    "PlaylistTrack",
    "Track",
)


def _create_in_memory_engine() -> Engine:
    # StaticPool keeps a single connection alive; every new connection to
    # ":memory:" would otherwise open a separate, empty database.
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _load_sql_script(engine: Engine, sql_file: Path) -> None:
    sql_script = sql_file.read_text(encoding="utf-8-sig")

    # The script contains many statements, so run it with sqlite3's
    # executescript() on the underlying DBAPI connection.
    raw = engine.raw_connection()
    try:
        raw.driver_connection.executescript(sql_script)
        raw.commit()
    finally:
        raw.close()  # returns the connection to the StaticPool; it stays open

    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the engine for the loaded in-memory Chinook database."""
    if not SQL_FILE.exists():
        raise FileNotFoundError(f"Chinook SQL file not found: {SQL_FILE}")
    engine = _create_in_memory_engine()
    _load_sql_script(engine, SQL_FILE)
    return engine


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine())


def get_session() -> Session:
    """Return a new ORM session bound to the in-memory Chinook database."""
    return _session_factory()()


def get_metadata() -> MetaData:
    """Return MetaData reflected from the loaded database."""
    metadata = MetaData()
    metadata.reflect(bind=get_engine())
    return metadata


def health_check() -> dict[str, Any]:
    """Verify the expected tables exist and a COUNT(*) succeeds on each.

    Never raises. Returns::

        {"healthy": bool, "missing_tables": [...], "row_counts": {...}, "error": str | None}
    """
    report: dict[str, Any] = {
        "healthy": False,
        "missing_tables": [],
        "row_counts": {},
        "error": None,
    }
    try:
        engine = get_engine()
        existing = set(inspect(engine).get_table_names())
        report["missing_tables"] = [t for t in EXPECTED_TABLES if t not in existing]

        with engine.connect() as conn:
            for table in EXPECTED_TABLES:
                if table in existing:
                    # Safe to interpolate: names come from EXPECTED_TABLES, not user input.
                    count = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar_one()
                    report["row_counts"][table] = count
    except (SQLAlchemyError, FileNotFoundError) as exc:
        report["error"] = str(exc.__cause__ or exc)
        return report

    report["healthy"] = not report["missing_tables"]
    return report


def _json_default(value: Any) -> Any:
    """Convert values the json module can't serialize on its own."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def normalize_phone(phone: str | None) -> str | None:
    """Strip every non-digit from a phone number, keeping a leading ``+``.

    ``"+55 (12) 3923-5555"`` -> ``"+551239235555"``
    ``"(403) 262-3443"``     -> ``"4032623443"``

    Returns ``None`` for ``None`` (Chinook's Phone/Fax columns are nullable)
    and ``""`` when the input contains no digits.
    """
    if phone is None:
        return None
    stripped = phone.strip()
    digits = _NON_DIGITS.sub("", stripped)
    if not digits:
        return ""
    return f"+{digits}" if stripped.startswith("+") else digits


def execute_query(sql: str, params: Mapping[str, Any] | None = None) -> str:
    """Execute a SQL statement with bound parameters and return a JSON string.

    Parameters are bound by name, e.g. ``"... WHERE ArtistId = :artist_id"``
    with ``{"artist_id": 1}``. Values are never interpolated into the SQL.

    Returns one of:
      - row-returning queries: ``{"columns": [...], "rows": [{...}], "row_count": n}``
      - other statements:      ``{"rows_affected": n}``
      - on failure:            ``{"error": "..."}``
    """
    try:
        with get_engine().begin() as conn:
            result = conn.execute(text(sql), dict(params or {}))
            if result.returns_rows:
                columns = list(result.keys())
                rows = [dict(row) for row in result.mappings()]
                payload = {"columns": columns, "rows": rows, "row_count": len(rows)}
            else:
                payload = {"rows_affected": result.rowcount}
    except SQLAlchemyError as exc:
        payload = {"error": str(exc.__cause__ or exc)}

    return json.dumps(payload, default=_json_default, ensure_ascii=False)


if __name__ == "__main__":
    report = health_check()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["healthy"] else 1)
