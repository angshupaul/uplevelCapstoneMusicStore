"""Self-check: run "SELECT COUNT(*) FROM Customer" through execute_query."""

import json
import sys

from sqlalchemy import func, select

from database import execute_query, get_engine, get_metadata

QUERY = "SELECT COUNT(*) FROM Customer"
EXPECTED_COUNT = 59  # Chinook v1.4.5


def main() -> int:
    raw_json = execute_query(QUERY)
    print(f"Query : {QUERY}")
    print(f"Result: {raw_json}\n")

    payload = json.loads(raw_json)
    checks: list[tuple[str, bool]] = []

    checks.append(("No error returned", "error" not in payload))
    checks.append(("Returned exactly one row", payload.get("row_count") == 1))
    checks.append(("Returned exactly one column", len(payload.get("columns", [])) == 1))

    count = None
    if payload.get("rows"):
        count = next(iter(payload["rows"][0].values()))
    checks.append(("Count is an integer", isinstance(count, int)))
    checks.append((f"Count equals expected {EXPECTED_COUNT}", count == EXPECTED_COUNT))

    # Cross-check against an independent count built with SQLAlchemy Core.
    customer = get_metadata().tables["Customer"]
    with get_engine().connect() as conn:
        core_count = conn.execute(select(func.count()).select_from(customer)).scalar_one()
    checks.append((f"Count matches SQLAlchemy Core count ({core_count})", count == core_count))

    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    all_ok = all(ok for _, ok in checks)
    print(f"\nSelf-check {'PASSED' if all_ok else 'FAILED'} (customer count = {count})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
