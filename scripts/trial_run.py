"""Call search tools and print the results.

Run from the project root:
    python scripts/trial_run.py

Each tool runs only if its output JSON is missing from ProjectExecution.
"""
# This script tests the search and browse tools by invoking them and saving their results to JSON files. 
# Purpose is to verify the correctness of the JSON files, based on various scenarios..

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools import (  # noqa: E402
    browse_tracks_by_genre,
    get_customer_invoices,
    get_invoice_line_items,
    get_invoice_support_rep,
    get_track_details,
    search_albums_by_artist,
    search_songs_by_title,
    search_tracks_by_artist,
)

EXECUTION_DIR = (
    Path(r"C:\AIAgents\CapstoneProject\Capstone Project Resources")
    / "4. Multi-Agent AI Customer Support"
    / "ProjectExecution"
)
EXECUTION_DIR.mkdir(parents=True, exist_ok=True)

#ARTIST_NAME = "AC/DC"
#ARTIST_NAME = "Queen"
ARTIST_NAME = "Led Zeppelin"
#ARTIST_NAME = "Aquaman"
GENRE = "Rock"
TITLE = "rhapsody"
#TRACK_ID = 2254
TRACK_ID = 2469
# Chinook CustomerId 1: Luís Gonçalves (DBSet/Chinook_Sqlite.sql)
CUSTOMER_ID = 5
INVOICE_ID = 90


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote results to {path}")


tracks_path = EXECUTION_DIR / "search_tracks_by_artist.json"
if not tracks_path.exists():
    result = search_tracks_by_artist.invoke({"artist_name": ARTIST_NAME})
    print(result)
    albums_result = search_albums_by_artist.invoke({"artist_name": ARTIST_NAME})
    write_json(
        tracks_path,
        {
            "artist_name": ARTIST_NAME,
            "search_albums_by_artist": json.loads(albums_result),
            "search_tracks_by_artist": json.loads(result),
        },
    )
else:
    print(f"Skipping search_tracks_by_artist; {tracks_path.name} already exists")

genre_path = EXECUTION_DIR / "browseTracksByGenre.json"
if not genre_path.exists():
    genre_result = browse_tracks_by_genre.invoke({"genre": GENRE})
    print(genre_result)
    write_json(genre_path, json.loads(genre_result))
else:
    print(f"Skipping browse_tracks_by_genre; {genre_path.name} already exists")

title_path = EXECUTION_DIR / "search_songs_by_title.json"
if not title_path.exists():
    title_result = search_songs_by_title.invoke({"title": TITLE})
    print(title_result)
    write_json(title_path, json.loads(title_result))
else:
    print(f"Skipping search_songs_by_title; {title_path.name} already exists")

details_path = EXECUTION_DIR / "get_track_details.json"
if not details_path.exists():
    track_details_result = get_track_details.invoke({"track_id": TRACK_ID})
    print(track_details_result)
    write_json(details_path, json.loads(track_details_result))
else:
    print(f"Skipping get_track_details; {details_path.name} already exists")

invoices_path = EXECUTION_DIR / "get_customer_invoices.json"
if not invoices_path.exists():
    invoices_result = get_customer_invoices.invoke({"customer_id": CUSTOMER_ID})
    print(invoices_result)
    write_json(invoices_path, json.loads(invoices_result))
else:
    print(f"Skipping get_customer_invoices; {invoices_path.name} already exists")

invoice_support_rep_path = EXECUTION_DIR / "get_invoice_support_rep.json"
if not invoice_support_rep_path.exists():
    invoice_support_rep_result = get_invoice_support_rep.invoke(
        {"invoice_id": INVOICE_ID}
    )
    print(invoice_support_rep_result)
    write_json(invoice_support_rep_path, json.loads(invoice_support_rep_result))
else:
    print(
        "Skipping get_invoice_support_rep; "
        f"{invoice_support_rep_path.name} already exists"
    )

invoice_line_items_path = EXECUTION_DIR / "get_invoice_line_item.json"
if not invoice_line_items_path.exists():
    invoice_line_items_result = get_invoice_line_items.invoke(
        {"invoice_id": INVOICE_ID}
    )
    print(invoice_line_items_result)
    write_json(invoice_line_items_path, json.loads(invoice_line_items_result))
else:
    print(
        "Skipping get_invoice_line_items; "
        f"{invoice_line_items_path.name} already exists"
    )
