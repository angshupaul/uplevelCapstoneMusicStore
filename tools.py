"""LangChain tools for the Digital Music Store agents."""

import json
import re
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BeforeValidator

from database import execute_query

MAX_RESULTS = 100


def _to_str(value: object) -> str:
    """Turn any argument into a string, so a number or null never fails
    LangChain's schema check; the tool itself then validates the text."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


# Every tool parameter is a string. The schema shown to the agent says
# "string", but 42 or null are converted instead of raising an error.
StrArg = Annotated[str, BeforeValidator(_to_str)]

# The verified customer's id, filled from graph state by the ToolNode. It is
# hidden from the agent's tool schema, so the LLM can never choose whose data
# a tool reads. Direct Python callers can still pass it, e.g. in tests.
VerifiedCustomerId = Annotated[str, BeforeValidator(_to_str), InjectedState("customer_id")]

_WHOLE_NUMBER = re.compile(r"[+-]?\d+", re.ASCII)

# Largest integer SQLite can store; bigger values would crash the query.
_SQLITE_MAX_INT = 2**63 - 1


def _parse_int(value: str, name: str, default: int | None = None) -> tuple[int | None, str | None]:
    """Parse a string argument as a whole number.

    Returns (number, None) on success or (None, error_json) on failure. A blank
    value returns default when one is given, otherwise it is an error.
    """
    # _to_str again: injected arguments (VerifiedCustomerId) skip validation,
    # so they arrive as the raw state value, e.g. 14 or None.
    text = _to_str(value).strip()
    if not text:
        if default is not None:
            return default, None
        return None, json.dumps({"error": f"{name} is required, e.g. 42"})
    if not _WHOLE_NUMBER.fullmatch(text):
        return None, json.dumps(
            {"error": f"{name} must be a whole number, e.g. 42; got {value!r}"}
        )
    number = int(text)
    if abs(number) > _SQLITE_MAX_INT:
        return None, json.dumps({"error": f"{name} is too large; got {value!r}"})
    return number, None


def _parse_id(value: str, name: str) -> tuple[int | None, str | None]:
    """Parse a required ID (CustomerId, InvoiceId, TrackId): a positive whole number."""
    number, error = _parse_int(value, name)
    if error:
        return None, error
    if number < 1:
        return None, json.dumps({"error": f"{name} must be a positive integer; got {value!r}"})
    return number, None


def _parse_optional_id(value: str, name: str) -> tuple[int | None, str | None]:
    """Like _parse_id, but a blank value means "not given" and returns (None, None)."""
    if not _to_str(value).strip():
        return None, None
    return _parse_id(value, name)


def _not_found(message: str) -> str:
    """JSON for a lookup that found nothing.

    "error" is kept for bad input and database failures; "message" is a
    sentence the agent can pass straight on to the customer.
    """
    return json.dumps({"message": message}, ensure_ascii=False)


def tool_exception_json(error: Exception) -> str:
    """Report an exception raised inside a tool as the tools' usual error JSON.

    For ToolNode(handle_tool_errors=...). Errors the tools return themselves
    (bad input, database failures) are already JSON and reach the LLM
    unchanged; this covers anything unexpected, so the agent loop keeps going
    and the LLM can report the failure instead of the graph crashing.
    """
    return json.dumps({"error": f"{type(error).__name__}: {error}"})


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards so user input is matched literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@tool
def search_albums_by_artist(artist_name: StrArg, limit: StrArg = "25") -> str:
    """Find albums by an artist, using a partial, case-insensitive name match.

    Use this when the user asks what albums an artist has, e.g. "albums by AC/DC"
    or "anything from zeppelin". A fragment of the name is enough.

    Args:
        artist_name: Full or partial artist name, e.g. "zeppelin".
        limit: Maximum number of albums to return, as a whole number in a
            string (1-100, default "25").

    Returns:
        JSON with columns AlbumId, AlbumTitle, ArtistId, ArtistName, TrackCount,
        and has_more: true if more albums match than were returned (search
        again with a higher limit), false if the list is complete. If nothing
        matches, rows is empty and message says so.
    """
    artist_name = artist_name.strip()
    if not artist_name:
        return json.dumps({"error": "artist_name must not be empty"})
    limit, error = _parse_int(limit, "limit", default=25)
    if error:
        return error
    limit = max(1, min(limit, MAX_RESULTS))

    sql = """
        SELECT al.AlbumId, al.Title AS AlbumTitle, ar.ArtistId, ar.Name AS ArtistName,
               COUNT(t.TrackId) AS TrackCount
        FROM Album AS al
        JOIN Artist AS ar ON ar.ArtistId = al.ArtistId
        LEFT JOIN Track AS t ON t.AlbumId = al.AlbumId
        WHERE ar.Name LIKE :pattern ESCAPE '\\'
        GROUP BY al.AlbumId
        ORDER BY ar.Name, al.Title
        LIMIT :limit
    """
    # SQLite's LIKE is case-insensitive for ASCII letters. One extra row is
    # fetched to tell whether the list was cut off at the limit.
    params = {"pattern": f"%{_escape_like(artist_name)}%", "limit": limit + 1}
    albums = json.loads(execute_query(sql, params))
    if "error" in albums:
        return json.dumps(albums)
    albums["has_more"] = len(albums["rows"]) > limit
    albums["rows"] = albums["rows"][:limit]
    albums["row_count"] = len(albums["rows"])
    if not albums["rows"]:
        albums["message"] = f"No albums found for artist: {artist_name}"
    return json.dumps(albums, ensure_ascii=False)


MAX_TRACK_SAMPLE = 20

# Shared FROM/WHERE so the count and the sample always cover the same tracks.
_TRACKS_BY_ARTIST = """
    FROM Track AS t
    JOIN Album AS al ON al.AlbumId = t.AlbumId
    JOIN Artist AS ar ON ar.ArtistId = al.ArtistId
    LEFT JOIN Genre AS g ON g.GenreId = t.GenreId
    WHERE ar.Name LIKE :pattern ESCAPE '\\'
"""


@tool
def search_tracks_by_artist(
    artist_name: StrArg, sample_size: StrArg = str(MAX_TRACK_SAMPLE)
) -> str:
    """Count an artist's tracks and return a sample of them, using a partial,
    case-insensitive name match.

    Use this when the user asks about an artist's songs, e.g. "how many songs
    does Queen have?" or "show me some tracks by miles davis".

    Args:
        artist_name: Full or partial artist name, e.g. "queen".
        sample_size: Number of tracks to list, as a whole number in a string
            (1-20, default "20").

    Returns:
        JSON with total_tracks (count of ALL matching tracks), sample_size
        (number of tracks listed), and tracks: a list with TrackId, TrackName,
        AlbumTitle, ArtistName, GenreName, Milliseconds, UnitPrice. If nothing
        matches, tracks is empty and message says so.
    """
    artist_name = artist_name.strip()
    if not artist_name:
        return json.dumps({"error": "artist_name must not be empty"})
    sample_size, error = _parse_int(sample_size, "sample_size", default=MAX_TRACK_SAMPLE)
    if error:
        return error
    sample_size = max(1, min(sample_size, MAX_TRACK_SAMPLE))
    pattern = f"%{_escape_like(artist_name)}%"

    count = json.loads(execute_query(
        f"SELECT COUNT(*) AS total {_TRACKS_BY_ARTIST}", {"pattern": pattern}
    ))
    if "error" in count:
        return json.dumps(count)

    sample = json.loads(execute_query(
        f"""
        SELECT t.TrackId, t.Name AS TrackName, al.Title AS AlbumTitle,
               ar.Name AS ArtistName, g.Name AS GenreName, t.Milliseconds, t.UnitPrice
        {_TRACKS_BY_ARTIST}
        ORDER BY ar.Name, al.Title, t.TrackId
        LIMIT :limit
        """,
        {"pattern": pattern, "limit": sample_size},
    ))
    if "error" in sample:
        return json.dumps(sample)

    result = {
        "total_tracks": count["rows"][0]["total"],
        "sample_size": sample["row_count"],
        "tracks": sample["rows"],
    }
    if not result["tracks"]:
        result["message"] = f"No tracks found for artist: {artist_name}"
    return json.dumps(result, ensure_ascii=False)


MAX_GENRE_SAMPLE = 10

# Shared FROM/WHERE so the summary and the sample always cover the same tracks.
_TRACKS_BY_GENRE = """
    FROM Track AS t
    JOIN Genre AS g ON g.GenreId = t.GenreId
    JOIN Album AS al ON al.AlbumId = t.AlbumId
    JOIN Artist AS ar ON ar.ArtistId = al.ArtistId
    WHERE g.Name LIKE :pattern ESCAPE '\\'
"""


@tool
def browse_tracks_by_genre(genre: StrArg, sample_size: StrArg = str(MAX_GENRE_SAMPLE)) -> str:
    """Browse a genre: one representative track from each of up to 10 different
    artists, using a partial, case-insensitive genre name match.

    Use this when the user wants to explore or get recommendations for a genre,
    e.g. "what jazz do you have?" or "show me some blues". Every track in the
    sample is by a different artist: the artist's first track in the genre
    (lowest TrackId). The same genre always returns the same tracks.

    Args:
        genre: Full or partial genre name, e.g. "jazz" or "metal".
        sample_size: Number of artists (and tracks) to show, as a whole
            number in a string (1-10, default "10").

    Returns:
        JSON with matched_genres, total_tracks and total_artists (for the whole
        genre), sample_size, and tracks: one per artist with TrackId, TrackName,
        AlbumTitle, ArtistName, GenreName, Milliseconds, UnitPrice, TimesSold,
        ArtistTrackCount. If no genre matches, tracks is empty, message says
        so, and available_genres lists every genre.
    """
    genre = genre.strip()
    if not genre:
        return json.dumps({"error": "genre must not be empty"})
    sample_size, error = _parse_int(sample_size, "sample_size", default=MAX_GENRE_SAMPLE)
    if error:
        return error
    sample_size = max(1, min(sample_size, MAX_GENRE_SAMPLE))
    pattern = f"%{_escape_like(genre)}%"

    summary = json.loads(execute_query(
        f"""
        SELECT COUNT(*) AS total_tracks, COUNT(DISTINCT ar.ArtistId) AS total_artists
        {_TRACKS_BY_GENRE}
        """,
        {"pattern": pattern},
    ))
    if "error" in summary:
        return json.dumps(summary)
    totals = summary["rows"][0]

    genres = json.loads(execute_query(
        "SELECT Name FROM Genre WHERE Name LIKE :pattern ESCAPE '\\' ORDER BY Name, GenreId",
        {"pattern": pattern},
    ))
    if "error" in genres:
        return json.dumps(genres)
    matched_genres = [r["Name"] for r in genres["rows"]]

    if not matched_genres:
        # Give the agent the valid names so it can retry or suggest one.
        all_genres = json.loads(execute_query("SELECT Name FROM Genre ORDER BY Name, GenreId"))
        return json.dumps(
            {
                "matched_genres": [],
                "total_tracks": 0,
                "total_artists": 0,
                "sample_size": 0,
                "tracks": [],
                "message": f"No genre found matching: {genre}",
                "available_genres": [r["Name"] for r in all_genres.get("rows", [])],
            },
            ensure_ascii=False,
        )

    # Number each artist's tracks by TrackId and keep only the first (rn = 1),
    # then pick the artists with the most tracks in the genre. Every ORDER BY
    # ends on a unique id, so the same genre always returns identical results.
    # Ranking by TrackId rather than sales keeps the pick fixed even when new
    # sales are recorded.
    sample = json.loads(execute_query(
        f"""
        WITH genre_tracks AS (
            SELECT t.TrackId, t.Name AS TrackName, al.Title AS AlbumTitle,
                   ar.ArtistId, ar.Name AS ArtistName, g.Name AS GenreName,
                   t.Milliseconds, t.UnitPrice,
                   (SELECT COALESCE(SUM(il.Quantity), 0)
                    FROM InvoiceLine AS il
                    WHERE il.TrackId = t.TrackId) AS TimesSold
            {_TRACKS_BY_GENRE}
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY ArtistId ORDER BY TrackId) AS rn,
                   COUNT(*) OVER (PARTITION BY ArtistId) AS ArtistTrackCount
            FROM genre_tracks
        )
        SELECT TrackId, TrackName, AlbumTitle, ArtistName, GenreName,
               Milliseconds, UnitPrice, TimesSold, ArtistTrackCount
        FROM ranked
        WHERE rn = 1
        ORDER BY ArtistTrackCount DESC, ArtistName, ArtistId
        LIMIT :limit
        """,
        {"pattern": pattern, "limit": sample_size},
    ))
    if "error" in sample:
        return json.dumps(sample)

    result = {
        "matched_genres": matched_genres,
        "total_tracks": totals["total_tracks"],
        "total_artists": totals["total_artists"],
        "sample_size": sample["row_count"],
        "tracks": sample["rows"],
    }
    # A genre can exist with no tracks in it.
    if not result["tracks"]:
        result["message"] = f"No tracks found for genre: {', '.join(matched_genres)}"
    return json.dumps(result, ensure_ascii=False)


MAX_TITLE_RESULTS = 10

# Shared FROM/WHERE so the count and the results always cover the same tracks.
_TRACKS_BY_TITLE = """
    FROM Track AS t
    LEFT JOIN Album AS al ON al.AlbumId = t.AlbumId
    LEFT JOIN Artist AS ar ON ar.ArtistId = al.ArtistId
    LEFT JOIN Genre AS g ON g.GenreId = t.GenreId
    WHERE t.Name LIKE :contains ESCAPE '\\'
"""


@tool
def search_songs_by_title(title: StrArg, limit: StrArg = str(MAX_TITLE_RESULTS)) -> str:
    """Find songs by title, using a partial, case-insensitive match.

    Use this when the user names a song or part of one, e.g. "do you have
    Bohemian Rhapsody?" or "songs with 'love' in the title".

    Args:
        title: Full or partial song title, e.g. "rhapsody".
        limit: Maximum number of songs to return, as a whole number in a
            string (1-10, default "10").

    Returns:
        JSON with total_matches (count of ALL matching songs), result_count
        (number listed), and tracks: best matches first (exact title, then
        titles starting with the text, then titles containing it), each with
        TrackId, TrackName, ArtistName, AlbumTitle, GenreName, Composer,
        Milliseconds, UnitPrice. If nothing matches, tracks is empty and
        message says so.
    """
    title = title.strip()
    if not title:
        return json.dumps({"error": "title must not be empty"})
    limit, error = _parse_int(limit, "limit", default=MAX_TITLE_RESULTS)
    if error:
        return error
    limit = max(1, min(limit, MAX_TITLE_RESULTS))
    escaped = _escape_like(title)
    params = {
        "exact": escaped,
        "prefix": f"{escaped}%",
        "contains": f"%{escaped}%",
    }

    count = json.loads(execute_query(
        f"SELECT COUNT(*) AS total {_TRACKS_BY_TITLE}", {"contains": params["contains"]}
    ))
    if "error" in count:
        return json.dumps(count)

    results = json.loads(execute_query(
        f"""
        SELECT t.TrackId, t.Name AS TrackName, ar.Name AS ArtistName,
               al.Title AS AlbumTitle, g.Name AS GenreName, t.Composer,
               t.Milliseconds, t.UnitPrice
        {_TRACKS_BY_TITLE}
        ORDER BY
            CASE
                WHEN t.Name LIKE :exact ESCAPE '\\' THEN 0
                WHEN t.Name LIKE :prefix ESCAPE '\\' THEN 1
                ELSE 2
            END,
            t.Name, ar.Name, t.TrackId
        LIMIT :limit
        """,
        {**params, "limit": limit},
    ))
    if "error" in results:
        return json.dumps(results)

    result = {
        "total_matches": count["rows"][0]["total"],
        "result_count": results["row_count"],
        "tracks": results["rows"],
    }
    if not result["tracks"]:
        result["message"] = f"No songs found with title: {title}"
    return json.dumps(result, ensure_ascii=False)


def _format_duration(milliseconds: int) -> str:
    """Format milliseconds as m:ss, or h:mm:ss for an hour or longer."""
    minutes, seconds = divmod(round(milliseconds / 1000), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


@tool
def get_track_details(track_id: StrArg) -> str:
    """Get the full details of one specific track by its TrackId.

    Use this after a search tool has found a track and the user wants to know
    more about it, e.g. "tell me more about that song" or "how long is it?".
    TrackIds come from the other tools' results.

    Args:
        track_id: The track's TrackId as a string of digits, e.g. "1".

    Returns:
        JSON with the track's name, composer, album (and its position on the
        album), artist, genre, media type, duration, file size, price, the
        playlists it appears on, and sales totals. If no track has that
        TrackId, returns only a message saying so.
    """
    track_id, error = _parse_id(track_id, "track_id")
    if error:
        return error

    # Track number = position on the album, counting the album's tracks by TrackId.
    track = json.loads(execute_query(
        """
        SELECT t.TrackId, t.Name AS TrackName, t.Composer,
               t.Milliseconds, t.Bytes, t.UnitPrice,
               al.AlbumId, al.Title AS AlbumTitle,
               ar.ArtistId, ar.Name AS ArtistName,
               g.GenreId, g.Name AS GenreName,
               mt.MediaTypeId, mt.Name AS MediaType,
               (SELECT COUNT(*) FROM Track AS t2
                WHERE t2.AlbumId = t.AlbumId AND t2.TrackId <= t.TrackId) AS TrackNumber,
               (SELECT COUNT(*) FROM Track AS t2
                WHERE t2.AlbumId = t.AlbumId) AS AlbumTrackCount
        FROM Track AS t
        LEFT JOIN Album AS al ON al.AlbumId = t.AlbumId
        LEFT JOIN Artist AS ar ON ar.ArtistId = al.ArtistId
        LEFT JOIN Genre AS g ON g.GenreId = t.GenreId
        LEFT JOIN MediaType AS mt ON mt.MediaTypeId = t.MediaTypeId
        WHERE t.TrackId = :track_id
        """,
        {"track_id": track_id},
    ))
    if "error" in track:
        return json.dumps(track)
    if not track["rows"]:
        return _not_found(f"No track found with TrackId: {track_id}")
    details = track["rows"][0]

    playlists = json.loads(execute_query(
        """
        SELECT p.PlaylistId, p.Name
        FROM PlaylistTrack AS pt
        JOIN Playlist AS p ON p.PlaylistId = pt.PlaylistId
        WHERE pt.TrackId = :track_id
        ORDER BY p.Name, p.PlaylistId
        """,
        {"track_id": track_id},
    ))
    if "error" in playlists:
        return json.dumps(playlists)

    sales = json.loads(execute_query(
        """
        SELECT COALESCE(SUM(il.Quantity), 0) AS TimesSold,
               COUNT(DISTINCT il.InvoiceId) AS InvoiceCount,
               ROUND(COALESCE(SUM(il.UnitPrice * il.Quantity), 0), 2) AS TotalRevenue
        FROM InvoiceLine AS il
        WHERE il.TrackId = :track_id
        """,
        {"track_id": track_id},
    ))
    if "error" in sales:
        return json.dumps(sales)

    details["Duration"] = _format_duration(details["Milliseconds"])
    details["SizeMB"] = round(details["Bytes"] / 1_000_000, 2) if details["Bytes"] else None
    details["Playlists"] = playlists["rows"]
    details["Sales"] = sales["rows"][0]
    return json.dumps(details, ensure_ascii=False)


@tool
def get_customer_invoices(customer_id: VerifiedCustomerId) -> str:
    """Get all invoices for the verified customer, most recent first.

    Use this when a customer asks about their purchase history, e.g. "show my
    past orders" or "when was my last purchase?". It always looks up the
    customer you are helping; it takes no arguments.

    Returns:
        JSON with CustomerId, CustomerName, invoice_count, total_spent, and
        invoices: newest first, each with InvoiceId, InvoiceDate, BillingCity,
        BillingState, BillingCountry, ItemCount and Total. If the customer has
        no invoices, invoices is empty and message says so. If no customer
        has that CustomerId, returns only a message saying so.
    """
    customer_id, error = _parse_id(customer_id, "customer_id")
    if error:
        return error

    customer = json.loads(execute_query(
        """
        SELECT CustomerId, FirstName || ' ' || LastName AS CustomerName
        FROM Customer
        WHERE CustomerId = :customer_id
        """,
        {"customer_id": customer_id},
    ))
    if "error" in customer:
        return json.dumps(customer)
    if not customer["rows"]:
        return _not_found(f"No customer found with CustomerId: {customer_id}")

    # InvoiceDate is stored as ISO text ("2025-12-22 00:00:00"), so sorting
    # the text sorts by date. InvoiceId breaks ties on the same day.
    invoices = json.loads(execute_query(
        """
        SELECT i.InvoiceId, i.InvoiceDate, i.BillingCity, i.BillingState,
               i.BillingCountry,
               (SELECT COALESCE(SUM(il.Quantity), 0)
                FROM InvoiceLine AS il
                WHERE il.InvoiceId = i.InvoiceId) AS ItemCount,
               i.Total
        FROM Invoice AS i
        WHERE i.CustomerId = :customer_id
        ORDER BY i.InvoiceDate DESC, i.InvoiceId DESC
        """,
        {"customer_id": customer_id},
    ))
    if "error" in invoices:
        return json.dumps(invoices)

    result = {
        **customer["rows"][0],
        "invoice_count": invoices["row_count"],
        "total_spent": round(sum(inv["Total"] for inv in invoices["rows"]), 2),
        "invoices": invoices["rows"],
    }
    if not result["invoices"]:
        result["message"] = f"No invoices found for customer: {result['CustomerName']}"
    return json.dumps(result, ensure_ascii=False)


MAX_PURCHASE_PAGE = 50

# One row per InvoiceLine = one purchased track. Shared so the count and the
# page always cover the same purchases.
_PURCHASED_TRACKS = """
    FROM InvoiceLine AS il
    JOIN Invoice AS i ON i.InvoiceId = il.InvoiceId
    JOIN Track AS t ON t.TrackId = il.TrackId
    JOIN Album AS al ON al.AlbumId = t.AlbumId
    JOIN Artist AS ar ON ar.ArtistId = al.ArtistId
    LEFT JOIN Genre AS g ON g.GenreId = t.GenreId
    LEFT JOIN MediaType AS mt ON mt.MediaTypeId = t.MediaTypeId
"""

_PURCHASED_COLUMNS = """
    SELECT il.InvoiceLineId, il.InvoiceId, i.InvoiceDate, i.CustomerId,
           t.TrackId, t.Name AS TrackName, ar.Name AS ArtistName,
           al.Title AS AlbumTitle, g.Name AS GenreName, mt.Name AS MediaType,
           il.UnitPrice, il.Quantity, il.UnitPrice * il.Quantity AS LineTotal
"""

# UnitPrice has only two values (0.99 / 1.99), so date and line id keep the
# order stable within each price.
_PURCHASED_ORDER = "ORDER BY il.UnitPrice DESC, i.InvoiceDate DESC, il.InvoiceLineId DESC"


def fetch_purchased_tracks(
    customer_id: int | None = None, limit: int | None = None, offset: int = 0
) -> str:
    """Every purchased track (one row per InvoiceLine), highest unit price first.

    Not an agent tool: with customer_id=None it covers all customers' purchases.
    Use it for internal reports; agents should use get_customer_purchased_tracks.
    UnitPrice is the price actually paid (InvoiceLine), not today's catalog price.
    limit=None returns every row.
    """
    where = "WHERE i.CustomerId = :customer_id" if customer_id is not None else ""
    params: dict = {"customer_id": customer_id, "offset": max(0, int(offset))}

    count = json.loads(execute_query(
        f"SELECT COUNT(*) AS total {_PURCHASED_TRACKS} {where}", params
    ))
    if "error" in count:
        return json.dumps(count)

    # SQLite needs a LIMIT before OFFSET; -1 means "no limit".
    params["limit"] = -1 if limit is None else max(1, int(limit))
    page = json.loads(execute_query(
        f"""
        {_PURCHASED_COLUMNS}
        {_PURCHASED_TRACKS}
        {where}
        {_PURCHASED_ORDER}
        LIMIT :limit OFFSET :offset
        """,
        params,
    ))
    if "error" in page:
        return json.dumps(page)

    return json.dumps(
        {
            "total_purchases": count["rows"][0]["total"],
            "offset": params["offset"],
            "result_count": page["row_count"],
            "purchases": page["rows"],
        },
        ensure_ascii=False,
    )


@tool
def get_customer_purchased_tracks(
    customer_id: VerifiedCustomerId, limit: StrArg = "20", offset: StrArg = "0"
) -> str:
    """List every track the verified customer has bought, highest unit price first.

    Use this when a customer asks what they've purchased, e.g. "what songs have
    I bought?" or "what's the most expensive thing I bought?". Each row is one
    purchased track, not a whole invoice; a track bought twice appears twice.
    Use offset to page through long lists. It always looks up the customer you
    are helping.

    Args:
        limit: Number of purchases to return, as a whole number in a string
            (1-50, default "20").
        offset: Number of purchases to skip, for paging, as a whole number in
            a string (default "0").

    Returns:
        JSON with total_purchases, offset, result_count, and purchases: each
        with InvoiceLineId, InvoiceId, InvoiceDate, CustomerId, TrackId,
        TrackName, ArtistName, AlbumTitle, GenreName, MediaType, UnitPrice,
        Quantity, LineTotal. If the page is empty (no purchases, or offset is
        past the end), message says so. If no customer has that CustomerId,
        returns only a message saying so.
    """
    customer_id, error = _parse_id(customer_id, "customer_id")
    if error:
        return error
    limit, error = _parse_int(limit, "limit", default=20)
    if error:
        return error
    limit = max(1, min(limit, MAX_PURCHASE_PAGE))
    offset, error = _parse_int(offset, "offset", default=0)
    if error:
        return error

    customer = json.loads(execute_query(
        "SELECT FirstName || ' ' || LastName AS CustomerName"
        " FROM Customer WHERE CustomerId = :customer_id",
        {"customer_id": customer_id},
    ))
    if "error" in customer:
        return json.dumps(customer)
    if not customer["rows"]:
        return _not_found(f"No customer found with CustomerId: {customer_id}")

    result = json.loads(fetch_purchased_tracks(customer_id, limit, offset))
    if "error" not in result and not result["purchases"]:
        name = customer["rows"][0]["CustomerName"]
        if result["total_purchases"] == 0:
            result["message"] = f"No purchases found for customer: {name}"
        else:
            result["message"] = (
                f"No more purchases: offset {result['offset']} is past the end"
                f" of {result['total_purchases']} purchases for customer: {name}"
            )
    return json.dumps(result, ensure_ascii=False)


def _invoice_not_found(invoice_id: int, customer_id: int | None) -> str:
    """Not-found message for the invoice tools.

    Used both when the invoice doesn't exist and when it belongs to another
    customer, so other customers' invoice ids are not revealed.
    """
    owner = f" for CustomerId: {customer_id}" if customer_id is not None else ""
    return _not_found(f"No invoice found with InvoiceId: {invoice_id}{owner}")


@tool
def get_invoice_support_rep(invoice_id: StrArg, customer_id: VerifiedCustomerId = "") -> str:
    """Find the support representative responsible for a specific invoice.

    Use this when a customer has a question or problem with one of their
    orders, e.g. "who can help me with invoice 98?" or "who do I contact about
    my last order?". Invoices have no rep of their own: the rep is the one
    assigned to the customer who placed the invoice (Invoice -> Customer ->
    Employee). Only the verified customer's own invoices can be looked up.

    Args:
        invoice_id: The invoice's InvoiceId as a string of digits, e.g. "98".

    Returns:
        JSON with InvoiceId, InvoiceDate, Total, CustomerId, CustomerName, and
        SupportRep: EmployeeId, RepName, Title, Email, Phone, and ManagerName
        (who the rep reports to, for escalation). If the customer has no rep
        assigned, SupportRep is null and message says so. If no invoice has
        that InvoiceId or it belongs to a different customer, returns only a
        message saying so.
    """
    invoice_id, error = _parse_id(invoice_id, "invoice_id")
    if error:
        return error
    customer_id, error = _parse_optional_id(customer_id, "customer_id")
    if error:
        return error

    # LEFT JOINs keep the invoice even if the customer has no SupportRepId or
    # the rep reports to no one (e.g. the General Manager).
    result = json.loads(execute_query(
        """
        SELECT i.InvoiceId, i.InvoiceDate, i.Total,
               c.CustomerId, c.FirstName || ' ' || c.LastName AS CustomerName,
               e.EmployeeId, e.FirstName || ' ' || e.LastName AS RepName,
               e.Title, e.Email, e.Phone,
               m.FirstName || ' ' || m.LastName AS ManagerName
        FROM Invoice AS i
        JOIN Customer AS c ON c.CustomerId = i.CustomerId
        LEFT JOIN Employee AS e ON e.EmployeeId = c.SupportRepId
        LEFT JOIN Employee AS m ON m.EmployeeId = e.ReportsTo
        WHERE i.InvoiceId = :invoice_id
        """,
        {"invoice_id": invoice_id},
    ))
    if "error" in result:
        return json.dumps(result)
    # Same message for "missing" and "someone else's" so other customers'
    # invoice ids are not revealed.
    if not result["rows"] or (
        customer_id is not None and result["rows"][0]["CustomerId"] != customer_id
    ):
        return _invoice_not_found(invoice_id, customer_id)
    row = result["rows"][0]

    rep_fields = ("EmployeeId", "RepName", "Title", "Email", "Phone", "ManagerName")
    rep = {k: row.pop(k) for k in rep_fields}
    row["SupportRep"] = rep if rep["EmployeeId"] is not None else None
    if row["SupportRep"] is None:
        row["message"] = (
            f"No support rep is assigned to customer {row['CustomerName']}"
            f" for InvoiceId: {invoice_id}"
        )
    return json.dumps(row, ensure_ascii=False)


@tool
def get_invoice_line_items(invoice_id: StrArg, customer_id: VerifiedCustomerId = "") -> str:
    """Get the line items (the tracks purchased) on one specific invoice.

    Use this when a customer asks what was in a particular order, e.g. "what
    did I buy on invoice 98?" or "what was in my last order?". Get InvoiceIds
    from get_customer_invoices. Only the verified customer's own invoices can
    be looked up.

    Args:
        invoice_id: The invoice's InvoiceId as a string of digits, e.g. "98".

    Returns:
        JSON with InvoiceId, InvoiceDate, CustomerId, CustomerName,
        BillingCity, BillingState, BillingCountry, Total, item_count, and
        line_items in receipt order, each with InvoiceLineId, TrackId,
        TrackName, ArtistName, AlbumTitle, GenreName, MediaType, Duration,
        UnitPrice, Quantity, LineTotal. If the invoice has no lines,
        line_items is empty and message says so. If no invoice has that
        InvoiceId or it belongs to a different customer, returns only a
        message saying so.
    """
    invoice_id, error = _parse_id(invoice_id, "invoice_id")
    if error:
        return error
    customer_id, error = _parse_optional_id(customer_id, "customer_id")
    if error:
        return error

    header = json.loads(execute_query(
        """
        SELECT i.InvoiceId, i.InvoiceDate, i.CustomerId,
               c.FirstName || ' ' || c.LastName AS CustomerName,
               i.BillingCity, i.BillingState, i.BillingCountry, i.Total
        FROM Invoice AS i
        JOIN Customer AS c ON c.CustomerId = i.CustomerId
        WHERE i.InvoiceId = :invoice_id
        """,
        {"invoice_id": invoice_id},
    ))
    if "error" in header:
        return json.dumps(header)
    # Same message for "missing" and "someone else's" so other customers'
    # invoice ids are not revealed.
    if not header["rows"] or (
        customer_id is not None and header["rows"][0]["CustomerId"] != customer_id
    ):
        return _invoice_not_found(invoice_id, customer_id)

    # UnitPrice is the price actually paid (InvoiceLine), not today's catalog
    # price. InvoiceLineId order is the order the lines appear on the receipt.
    lines = json.loads(execute_query(
        f"""
        SELECT il.InvoiceLineId, t.TrackId, t.Name AS TrackName,
               ar.Name AS ArtistName, al.Title AS AlbumTitle,
               g.Name AS GenreName, mt.Name AS MediaType, t.Milliseconds,
               il.UnitPrice, il.Quantity, il.UnitPrice * il.Quantity AS LineTotal
        {_PURCHASED_TRACKS}
        WHERE il.InvoiceId = :invoice_id
        ORDER BY il.InvoiceLineId
        """,
        {"invoice_id": invoice_id},
    ))
    if "error" in lines:
        return json.dumps(lines)

    for line in lines["rows"]:
        line["Duration"] = _format_duration(line.pop("Milliseconds"))

    result = {
        **header["rows"][0],
        "item_count": sum(line["Quantity"] for line in lines["rows"]),
        "line_items": lines["rows"],
    }
    if not result["line_items"]:
        result["message"] = f"No line items found for InvoiceId: {invoice_id}"
    return json.dumps(result, ensure_ascii=False)
