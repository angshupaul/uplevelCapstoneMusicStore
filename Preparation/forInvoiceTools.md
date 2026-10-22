Here's how I'd approach it. I only read and searched `Chinook_Sqlite.sql`; I didn't run any queries or code against the database.

## What the data shows

| Finding | Evidence from the `.sql` file | What it means for the design |
|---|---|---|
| **`InvoiceLine` already has one row per purchased track** | 2,240 rows, each with `InvoiceId`, `TrackId`, `UnitPrice` and `Quantity` | Use `InvoiceLine` as the main table. `Invoice` holds one row per order, so it's the wrong level for this. |
| **`Quantity` is always 1** | 2,240 of 2,240 rows | One row really is one track. No need to split rows by quantity. |
| **There are only two prices** | 2,129 rows at $0.99 and 111 at $1.99 | Sorting by price only splits the rows into two groups. You need a second sort key, or the order inside each group can change from run to run. |
| **Every $1.99 item is a video** | All 213 tracks priced $1.99 are "Protected MPEG-4 video file", in TV Shows, Drama, Sci Fi & Fantasy, Science Fiction and Comedy | "Highest price first" really means "TV episodes first, then music". |
| **The price paid always equals the catalog price today** | 0 rows where `InvoiceLine.UnitPrice` differs from `Track.UnitPrice` | Still sort on `InvoiceLine.UnitPrice`, the price actually paid. `Track.UnitPrice` is today's catalog price and could change later. |
| **Some tracks were bought more than once** | 1,984 distinct tracks; 256 bought more than once | One row per purchase means the same track can appear several times. That's correct for what you asked. |
| **No missing links** | No `InvoiceLine` rows point to a missing track; no tracks without an album | Plain joins are safe for Track, Invoice, Album and Artist. Keep `LEFT JOIN` for Genre, because `Track.GenreId` can be empty. |
| **Indexes exist on the join columns** | Indexes on `InvoiceLine.InvoiceId` and `InvoiceLine.TrackId` | The joins will be fast. Sorting 2,240 rows needs no index. |

## Recommended approach

1. **Start from `InvoiceLine`.** Join `Invoice` to get the date and customer, `Track` for the name, and Album, Artist, Genre and MediaType for context.
2. **Sort by `il.UnitPrice DESC`, then `i.InvoiceDate DESC`, then `il.InvoiceLineId DESC`.** That's highest price first, then most recent purchase first, then a final key so the order never changes between runs.
3. **Return a page of results plus the total.** 2,240 rows is fine for a normal Python function, but far too many to send to an AI model in one go. As an agent tool, return `total_purchases` plus a page controlled by `limit` and `offset`, the same pattern as the other tools.
4. **Limit it to one customer when it's an agent tool.** All purchases across all invoices include every customer's history. That's fine for an internal report, but an AI agent should only see the logged-in customer's purchases, the same concern as with `get_customer_invoices`. I'd split it in two:
   - `fetch_purchased_tracks()`: a plain function, not a tool, covering all customers, for reports.
   - `get_customer_purchased_tracks`: the tool the agent uses, which always requires a `customer_id`.

## Query

```sql
SELECT il.InvoiceLineId,
       il.InvoiceId,
       i.InvoiceDate,
       i.CustomerId,
       t.TrackId,
       t.Name               AS TrackName,
       ar.Name              AS ArtistName,
       al.Title             AS AlbumTitle,
       g.Name               AS GenreName,
       mt.Name              AS MediaType,
       il.UnitPrice,                          -- price actually paid
       il.Quantity,
       il.UnitPrice * il.Quantity AS LineTotal
FROM InvoiceLine AS il
JOIN Invoice     AS i  ON i.InvoiceId  = il.InvoiceId
JOIN Track       AS t  ON t.TrackId    = il.TrackId
JOIN Album       AS al ON al.AlbumId   = t.AlbumId
JOIN Artist      AS ar ON ar.ArtistId  = al.ArtistId
LEFT JOIN Genre     AS g  ON g.GenreId      = t.GenreId
LEFT JOIN MediaType AS mt ON mt.MediaTypeId = t.MediaTypeId
-- WHERE i.CustomerId = :customer_id        -- add for the per-customer tool
ORDER BY il.UnitPrice DESC,
         i.InvoiceDate DESC,
         il.InvoiceLineId DESC
LIMIT :limit OFFSET :offset;
```

## Draft code for `tools.py`

```python
MAX_PURCHASE_PAGE = 50

# Shared FROM so the count and the page always cover the same purchases.
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

# Price only has two values (0.99 / 1.99), so date and line id make the order stable.
_PURCHASED_ORDER = "ORDER BY il.UnitPrice DESC, i.InvoiceDate DESC, il.InvoiceLineId DESC"


def fetch_purchased_tracks(customer_id: int | None = None,
                           limit: int | None = None, offset: int = 0) -> str:
    """Every purchased track (one row per InvoiceLine), highest unit price first.

    Not an agent tool: with customer_id=None it covers all customers. Use it for
    internal reports; agents should use get_customer_purchased_tracks.
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
        f"{_PURCHASED_COLUMNS} {_PURCHASED_TRACKS} {where} {_PURCHASED_ORDER} "
        "LIMIT :limit OFFSET :offset",
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
def get_customer_purchased_tracks(customer_id: int, limit: int = 20, offset: int = 0) -> str:
    """List every track one customer has bought, highest unit price first.

    Use this when a customer asks what they've purchased, e.g. "what songs have
    I bought?" or "what was the most expensive thing I bought?". Each row is one
    purchased track, not a whole invoice. Use offset to page through long lists.
    Only pass the CustomerId of the customer you are currently helping.

    Args:
        customer_id: The customer's CustomerId, e.g. 1.
        limit: Number of purchases to return (1-50, default 20).
        offset: Number of purchases to skip, for paging (default 0).
    """
    customer_id = int(customer_id)
    if customer_id < 1:
        return json.dumps({"error": "customer_id must be a positive integer"})
    limit = max(1, min(int(limit), MAX_PURCHASE_PAGE))
    return fetch_purchased_tracks(customer_id, limit, offset)
```

## Decisions for you

- **One row per purchase, or one row per distinct track?** As you asked, the draft lists every purchase, so a track bought twice appears twice. If you'd rather list distinct tracks with a "bought N times" count, use `GROUP BY t.TrackId` with `COUNT(*)` and `MAX(il.UnitPrice)`.
- **Customer name in the all-customer function:** I've included only `CustomerId`. Joining `Customer` to show names is easy, but it adds personal data to every row.

One tidy-up: while profiling the file, one command briefly wrote a temporary file to `/tmp` instead of the scratchpad. I've deleted it, and nothing in the project was changed.