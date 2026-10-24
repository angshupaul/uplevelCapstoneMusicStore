"""Look up a Chinook customer by CustomerId and print their details and invoices.

Run from the project root:
    python scripts/Chinookdb_lookup.py
    python scripts/Chinookdb_lookup.py 14
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text  # noqa: E402

from database import get_engine  # noqa: E402

DEFAULT_CUSTOMER_ID = 14


def lookup_customer(customer_id: int):
    """Return the customer row for customer_id, or None if not found."""
    query = text(
        "SELECT CustomerId, FirstName, LastName, Email, Phone "
        "FROM Customer WHERE CustomerId = :customer_id"
    )
    with get_engine().connect() as conn:
        return conn.execute(query, {"customer_id": customer_id}).mappings().first()


def lookup_invoices(customer_id: int):
    """Return all invoices for customer_id, ordered by InvoiceDate."""
    query = text(
        "SELECT InvoiceId, InvoiceDate, BillingCity, BillingCountry, Total "
        "FROM Invoice WHERE CustomerId = :customer_id ORDER BY InvoiceDate"
    )
    with get_engine().connect() as conn:
        return conn.execute(query, {"customer_id": customer_id}).mappings().all()


def lookup_purchased_tracks(customer_id: int):
    """Return every track on the customer's invoices with composer and album title."""
    query = text(
        "SELECT t.Name AS TrackName, t.Composer, al.Title AS AlbumTitle "
        "FROM Invoice i "
        "JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId "
        "JOIN Track t ON t.TrackId = il.TrackId "
        "JOIN Album al ON al.AlbumId = t.AlbumId "
        "WHERE i.CustomerId = :customer_id "
        "ORDER BY i.InvoiceDate, i.InvoiceId, il.InvoiceLineId"
    )
    with get_engine().connect() as conn:
        return conn.execute(query, {"customer_id": customer_id}).mappings().all()


def lookup_purchased_artists(customer_id: int):
    """Return distinct artists of the customer's purchased tracks with track counts."""
    query = text(
        "SELECT ar.ArtistId, ar.Name AS ArtistName, COUNT(*) AS TrackCount "
        "FROM Invoice i "
        "JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId "
        "JOIN Track t ON t.TrackId = il.TrackId "
        "JOIN Album al ON al.AlbumId = t.AlbumId "
        "JOIN Artist ar ON ar.ArtistId = al.ArtistId "
        "WHERE i.CustomerId = :customer_id "
        "GROUP BY ar.ArtistId, ar.Name "
        "ORDER BY ar.Name"
    )
    with get_engine().connect() as conn:
        return conn.execute(query, {"customer_id": customer_id}).mappings().all()


def main() -> None:
    customer_id = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CUSTOMER_ID
    row = lookup_customer(customer_id)
    if row is None:
        print(f"No customer found with CustomerId {customer_id}")
        return
    print(f"CustomerId : {row['CustomerId']}")
    print(f"Name       : {row['FirstName']} {row['LastName']}")
    print(f"Email      : {row['Email']}")
    print(f"Phone      : {row['Phone']}")

    invoices = lookup_invoices(customer_id)
    print(f"\nInvoices   : {len(invoices)}")
    for inv in invoices:
        print(
            f"  InvoiceId {inv['InvoiceId']} | {inv['InvoiceDate']} | "
            f"{inv['BillingCity']}, {inv['BillingCountry']} | Total {inv['Total']}"
        )

    tracks = lookup_purchased_tracks(customer_id)
    print(f"\nTracks     : {len(tracks)}")
    for trk in tracks:
        print(
            f"  {trk['TrackName']} | Composer: {trk['Composer'] or 'N/A'} | "
            f"Album: {trk['AlbumTitle']}"
        )

    artists = lookup_purchased_artists(customer_id)
    print(f"\nArtists    : {len(artists)}")
    for art in artists:
        print(f"  ArtistId {art['ArtistId']} | {art['ArtistName']} | Tracks {art['TrackCount']}")


if __name__ == "__main__":
    main()
