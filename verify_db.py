"""Verify the in-memory Chinook database loaded correctly."""

from sqlalchemy import inspect, text

from database import get_engine, get_session

EXPECTED_COUNTS = {
    "Album": 347,
    "Artist": 275,
    "Customer": 59,
    "Employee": 8,
    "Genre": 25,
    "Invoice": 412,
    "InvoiceLine": 2240,
    "MediaType": 5,
    "Playlist": 18,
    "PlaylistTrack": 8715,
    "Track": 3503,
}

TOP_ARTISTS_SQL = text(
    """
    SELECT ar.Name, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS Revenue
    FROM InvoiceLine il
    JOIN Track t   ON t.TrackId = il.TrackId
    JOIN Album al  ON al.AlbumId = t.AlbumId
    JOIN Artist ar ON ar.ArtistId = al.ArtistId
    GROUP BY ar.ArtistId
    ORDER BY Revenue DESC
    LIMIT 5
    """
)


def main() -> None:
    engine = get_engine()
    tables = sorted(inspect(engine).get_table_names())
    print(f"Tables loaded ({len(tables)}): {', '.join(tables)}\n")

    all_ok = True
    with get_session() as session:
        print(f"{'Table':<15}{'Rows':>8}{'Expected':>10}  Status")
        for table, expected in EXPECTED_COUNTS.items():
            count = session.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar_one()
            ok = count == expected
            all_ok &= ok
            print(f"{table:<15}{count:>8}{expected:>10}  {'OK' if ok else 'MISMATCH'}")

        print("\nTop 5 artists by revenue:")
        for name, revenue in session.execute(TOP_ARTISTS_SQL):
            print(f"  {name:<30} ${revenue}")

    print("\nVerification", "PASSED" if all_ok else "FAILED")


if __name__ == "__main__":
    main()
