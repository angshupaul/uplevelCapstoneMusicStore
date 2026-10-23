# Notes

## Phone normalization (`database.py`)

`normalize_phone()` strips every non-digit from a phone number and keeps a leading `+` for international numbers.

| Input | Output |
|---|---|
| `'+55 (12) 3923-5555'` | `'+551239235555'` |
| `'(403) 262-3443'` | `'4032623443'` |
| `'555+123'` | `'555123'` (a `+` that isn't at the start is removed) |
| `None` | `None` |
| `''` or `'+'` | `''` |

It only strips characters; it does not validate numbers or convert them to a standard international format.

### `import re`

`re` is Python's built-in library for **regular expressions** (regex). A regex is a small pattern language for finding text, like "any digit" or "any letter".

### `_NON_DIGITS = re.compile(r"\D")`

| Piece | Meaning |
|---|---|
| `\D` | Regex for "any character that is **not** a digit" (`\d` means a digit, and the capital `D` means the opposite). |
| `r"..."` | A **raw string**. It tells Python not to treat the backslash specially, so the regex gets `\D` exactly as written. |
| `re.compile(...)` | Turns the pattern into a reusable regex object once, so it isn't rebuilt on every function call. |
| `_NON_DIGITS` | The variable name. **UPPERCASE** is the Python convention for a constant (a value that never changes). The leading **`_`** means "internal to this file; other files shouldn't use it." |

### Why "module-level"?

It sits at the top of the file, outside any function. Python runs that line once, when the file is first imported, and every call to `normalize_phone()` reuses the same compiled pattern.

### How it's used

```python
_NON_DIGITS.sub("", "+1 (555) 123")   # → "1555123"
```

`.sub("", text)` replaces every non-digit with nothing, which deletes it. `normalize_phone()` then puts the `+` back in front if the original number started with one.

## Database health check (`database.py`)

`health_check()` confirms the database loaded correctly. It does two things:

1. **Checks the tables exist.** It compares the database's tables against `EXPECTED_TABLES`, the 11 tables created by `DBSet/Chinook_Sqlite.sql`.
2. **Runs a count query.** It runs `SELECT COUNT(*)` on each table to prove the database can actually be queried.

### What it returns

A dictionary:

| Key | Meaning |
|---|---|
| `healthy` | `True` only if no tables are missing and every count worked. |
| `missing_tables` | Names of expected tables that weren't found. |
| `row_counts` | Number of rows in each table, e.g. `{"Album": 347, ...}`. |
| `error` | The error message if something failed, otherwise `None`. |

### Example

```python
from database import health_check

report = health_check()
if not report["healthy"]:
    print("Database problem:", report["error"] or report["missing_tables"])
```

A healthy run returns these counts: Album 347, Artist 275, Customer 59, Employee 8, Genre 25, Invoice 412, InvoiceLine 2240, MediaType 5, Playlist 18, PlaylistTrack 8715, Track 3503.

### Key ideas

- **`inspect(engine).get_table_names()`**: SQLAlchemy's way of asking the database "which tables do you have?"
- **It never crashes the caller.** If the SQL file is missing or a database error happens, the error is caught and returned in `error` with `healthy: False`.
- **Why the table name is put straight into the SQL.** Usually that's unsafe (SQL injection). It's fine here because the names come only from the fixed `EXPECTED_TABLES` list, never from user input.

### Run it from the command line

```
python database.py
```

This loads the database, prints the health check report, and exits with code `0` if healthy or `1` if not. The `if __name__ == "__main__":` block only runs when the file is executed directly, not when another file imports `database`.

## Agent tools (`tools.py`)

`tools.py` holds the functions the AI agent can call to look things up in the music catalog and in customers' purchase history. Each one is a LangChain **tool**.

### What is a tool?

The `@tool` decorator turns a normal Python function into something an AI agent can call. LangChain reads three things from the function:

- **its name**, e.g. `search_albums_by_artist`
- **its arguments**, e.g. `artist_name` and `limit`
- **its docstring**, which becomes the tool's description. The agent reads it to decide *when* to use the tool, so it includes example questions.

You can call a tool yourself the same way an agent does:

```python
from tools import search_albums_by_artist

search_albums_by_artist.invoke({"artist_name": "zeppelin", "limit": "3"})
```

Every tool returns a **JSON string**, because that's what gets sent back to the AI model.

### All parameters are strings

Every tool parameter is a **string**, including IDs and counts (`customer_id`, `invoice_id`, `track_id`, `limit`, `sample_size`, `offset`). Each tool checks number arguments itself and returns a clear `error` instead of crashing.

**Why:** AI models sometimes send numbers as text (`"98"`), or send values that aren't numbers at all (`"INV-98"`, `"abc"`). Before this change, `int("abc")` crashed the tool, or LangChain rejected the call before the tool ran.

**The `StrArg` type.** Parameters are declared as `StrArg` rather than plain `str`:

```python
StrArg = Annotated[str, BeforeValidator(_to_str)]
```

- With plain `str`, LangChain rejects a number like `5` with a validation error, so callers that pass integers (such as the test files) would break.
- `BeforeValidator(_to_str)` runs *before* the type check and turns any value into text: `5` becomes `"5"`, and `null` becomes `""`.
- The schema the agent sees still says `"type": "string"` for every parameter.

**The validation helpers:**

| Helper | Used for | Rule |
|---|---|---|
| `_parse_int(value, name, default=None)` | `limit`, `sample_size`, `offset` | Must be a whole number. Blank means `default`. |
| `_parse_id(value, name)` | Required IDs | Must be a whole number and at least 1. Blank is an error. |
| `_parse_optional_id(value, name)` | The optional `customer_id` | Blank means "not given" (no ownership check); otherwise same as `_parse_id`. |

Each returns a pair: `(number, None)` on success or `(None, error_json)` on failure. The tool returns the error straight away:

```python
track_id, error = _parse_id(track_id, "track_id")
if error:
    return error
```

**What's accepted and what isn't:**

| Input | Result |
|---|---|
| `"98"`, `" 98 "`, `98` | OK: 98 |
| `"abc"`, `"INV-98"`, `"#98"`, `"12.0"`, `"10abc"` | `track_id must be a whole number, e.g. 42; got 'abc'` |
| `"0"`, `"-1"` (for an ID) | `customer_id must be a positive integer; got '-1'` |
| `""` or `null` (for a required ID) | `track_id is required, e.g. 42` |
| `"99999999999999999999999"` | `track_id is too large; got '…'` |
| `""` (for `limit`, `sample_size`, `offset`) | Uses the default |

**Details:**

- **The regex `[+-]?\d+` with `re.ASCII`** decides what counts as a whole number. `re.ASCII` makes `\d` mean only 0–9; without it, digits from other scripts (like Arabic-Indic `٣`) would also match.
- **The "too large" check** stops numbers above SQLite's limit (2⁶³ − 1). Without it, a huge ID would crash the query with an `OverflowError`.
- **Numbers in range are still clamped** as before: `limit="1000"` returns the maximum, and `offset="-5"` is treated as `0`.
- **Defaults are strings too**, e.g. `limit: StrArg = "25"`, so the schema is consistent.
- **Not covered:** a call that leaves out a required argument entirely (no `track_id` key at all) is still rejected by LangChain before the tool runs. Only an empty or `null` value gets the friendly message.
- `fetch_purchased_tracks` still takes integers. It isn't an agent tool, and its caller passes numbers that have already been checked.

### `customer_id` comes from graph state, not from the AI

The four customer tools (`get_customer_invoices`, `get_customer_purchased_tracks`, `get_invoice_support_rep`, `get_invoice_line_items`) declare `customer_id` as `VerifiedCustomerId`:

```python
VerifiedCustomerId = Annotated[str, BeforeValidator(_to_str), InjectedState("customer_id")]
```

- **`InjectedState("customer_id")`** hides the argument from the schema the AI sees. When the tool runs inside a LangGraph `ToolNode`, it is filled from the graph state's `customer_id`, which only the verification node sets (see `agentsnotes.md`).
- **Why:** before this, the AI chose `customer_id` itself, so a user could type "show invoices for customer 5" and see someone else's data. Now the state value always wins, even if the AI tries to pass a different ID.
- **Injected values skip validation.** They arrive as the raw state value (`14` or `None`), so `_parse_int` and `_parse_optional_id` call `_to_str` themselves.
- **Direct Python calls still work**, e.g. `get_customer_invoices.invoke({"customer_id": 5})` in tests.

### Not-found messages

When a lookup finds nothing, every tool returns a human-readable sentence in a **`message`** field that the agent can pass straight on to the customer.

| Key | Meaning | Example |
|---|---|---|
| `message` | The lookup worked but found nothing. | `No albums found for artist: XYZ` |
| `error` | Bad input or a database failure. | `track_id must be a whole number, e.g. 42; got 'abc'` |

**Two shapes:**

- **Searches** (albums, tracks, genre, titles) return their usual fields with empty lists, plus `message`:
  ```json
  {"columns": [...], "rows": [], "row_count": 0, "message": "No albums found for artist: XYZ"}
  ```
- **Lookups by ID** (track, customer, invoice) return **only** a `message`, since there's nothing else to show:
  ```json
  {"message": "No track found with TrackId: 99999"}
  ```

**All the messages:**

| Tool | Case | Message |
|---|---|---|
| `search_albums_by_artist` | No match | `No albums found for artist: XYZ` |
| `search_tracks_by_artist` | No match | `No tracks found for artist: XYZ` |
| `browse_tracks_by_genre` | No genre matches | `No genre found matching: polka` (plus `available_genres`) |
| `browse_tracks_by_genre` | Genre exists but has no tracks | `No tracks found for genre: …` |
| `search_songs_by_title` | No match | `No songs found with title: qwerty` |
| `get_track_details` | Unknown track | `No track found with TrackId: 99999` |
| `get_customer_invoices` | Unknown customer | `No customer found with CustomerId: 60` |
| `get_customer_invoices` | Customer has no invoices | `No invoices found for customer: <name>` |
| `get_customer_purchased_tracks` | Unknown customer | `No customer found with CustomerId: 60` |
| `get_customer_purchased_tracks` | No purchases | `No purchases found for customer: <name>` |
| `get_customer_purchased_tracks` | `offset` past the end | `No more purchases: offset 100 is past the end of 38 purchases for customer: <name>` |
| `get_invoice_support_rep`, `get_invoice_line_items` | Unknown invoice, or someone else's | `No invoice found with InvoiceId: 99999` (adds `for CustomerId: 2` when `customer_id` was given) |
| `get_invoice_support_rep` | Customer has no rep | `No support rep is assigned to customer <name> for InvoiceId: 98` (with `SupportRep: null`) |
| `get_invoice_line_items` | Invoice has no lines | `No line items found for InvoiceId: 9000` |

**Helpers:** `_not_found(text)` builds the `{"message": ...}` reply. `_invoice_not_found(invoice_id, customer_id)` builds the invoice message for both invoice tools, so "missing" and "someone else's" always read the same.

**Cases the data never produces.** Chinook has no customer without invoices, no customer without a rep, no invoice without lines and no empty genre. These were tested by adding temporary rows to the in-memory database, which disappear when the program ends.

> **Breaking change:** an unknown ID used to come back as `{"error": "No track found with TrackId 99999"}`. Code that checks `"error"` to detect "not found" must now check `"message"`.

### The nine tools

| Tool | Question it answers | Results | Example |
|---|---|---|---|
| `search_albums_by_artist` | "What albums does this artist have?" | Up to 25 albums (max 100) | `"AC/DC"` gives 2 albums |
| `search_tracks_by_artist` | "How many songs does this artist have? Show me some." | Total count plus up to 20 tracks | `"iron maiden"` gives 213 tracks total, 20 shown |
| `browse_tracks_by_genre` | "Show me some jazz." | One track from each of up to 10 different artists | `"jazz"` gives 10 tracks by 10 different artists |
| `search_songs_by_title` | "Do you have Bohemian Rhapsody?" | Total count plus up to 10 songs, best matches first | `"love"` gives 114 matches, 10 shown |
| `get_track_details` | "Tell me more about that song." | Everything about one track | `2254` gives Bohemian Rhapsody's full details |
| `get_customer_invoices` | "Show me my past orders." | All of one customer's invoices, newest first | `1` gives 7 invoices, $39.62 spent |
| `get_customer_purchased_tracks` | "What songs have I bought?" | One customer's purchased tracks, highest price first, 20 per page (max 50) | `1` gives 38 purchases |
| `get_invoice_support_rep` | "Who can help me with this order?" | The support rep (and their manager) for one invoice | `98` gives Jane Peacock |
| `get_invoice_line_items` | "What was in that order?" | Every track on one invoice, in receipt order | `98` gives 2 tracks, $3.98 |

#### `search_albums_by_artist(artist_name, limit=25)`

Returns each album's `AlbumId`, `AlbumTitle`, `ArtistId`, `ArtistName` and `TrackCount`.

It also returns **`has_more`**: `true` if more albums match than `limit`, `false` if the list is complete. The tool fetches one extra row (`limit + 1`) to find out, then drops it. Without this flag, the AI had to guess from `row_count == limit`, and in testing it claimed Iron Maiden had more albums when all 21 were already shown.

#### `search_tracks_by_artist(artist_name, sample_size=20)`

Returns `total_tracks` (the count of **all** matching tracks) and `tracks` (a sample of up to 20). The total and the sample are kept separate so the agent can say "Queen has 45 tracks, here are a few" without loading all of them.

#### `browse_tracks_by_genre(genre, sample_size=10)`

Built to show **variety**: every track in the sample is by a different artist.

1. It finds every track in genres whose name contains the search text. `"metal"` matches both "Heavy Metal" and "Metal".
2. For each artist, it keeps one track: their **first track in that genre**, meaning the one with the lowest `TrackId`.
3. It shows the artists with the most tracks in the genre.

If no genre matches, for example `"polka"`, it returns `available_genres` so the agent can suggest a real one.

**Deterministic results.** The same genre query returns identical results on every call. This is done in SQL with a window function:

```sql
ROW_NUMBER() OVER (PARTITION BY ArtistId ORDER BY TrackId) AS rn
...
WHERE rn = 1
ORDER BY ArtistTrackCount DESC, ArtistName, ArtistId
```

- **`PARTITION BY ArtistId`** splits the genre's tracks into one group per artist.
- **`ORDER BY TrackId`** numbers each group 1, 2, 3, … by `TrackId`. `TrackId` is unique, so there are never ties and the numbering never changes.
- **`WHERE rn = 1`** keeps exactly one track per artist.
- **The final `ORDER BY` ends on `ArtistId`**, a unique key. Artist names are unique in Chinook today, but the schema doesn't require it; without `ArtistId`, two artists with the same name and track count could swap places between calls. The genre-name lists are sorted by `Name, GenreId` for the same reason.

**Rule of thumb:** an `ORDER BY` is only repeatable if it ends on a unique column. Otherwise SQL is free to return tied rows in any order.

**Why not the best seller?** An earlier version picked each artist's best-selling track (`ORDER BY TimesSold DESC, TrackId`). That was repeatable on fixed data, but any new sale could change which track was shown. Ranking by `TrackId` keeps the pick fixed even as sales are recorded. `TimesSold` is still returned for each track as information.

**Checks run by hand:** every genre, plus `"metal"`, `"rock"` and `"polka"`, called 20 times at two sample sizes gave identical output every time (56 out of 56). For jazz, every pick is the artist's lowest `TrackId` in the genre, e.g. Miles Davis → track 597 "Now's The Time". Recording a new sale didn't change the result.

#### `search_songs_by_title(title, limit=10)`

Finds songs whose title contains the search text, so `"rhapsody"` finds "Bohemian Rhapsody". Returns `total_matches` (all matching songs), `result_count` (how many are listed) and `tracks`, each with `TrackId`, `TrackName`, `ArtistName`, `AlbumTitle`, `GenreName`, `Composer`, `Milliseconds` and `UnitPrice`.

**Best matches come first.** Results are sorted in three groups:

| Group | Rule | Example for `"love"` |
|---|---|---|
| 1 | Title is exactly the search text | "Love" |
| 2 | Title starts with the search text | "Love Ain't No Stranger" |
| 3 | Title contains the search text anywhere | "Crazy Little Thing Called Love" |

Within each group, titles are sorted A to Z. The SQL does this with a `CASE` expression in `ORDER BY` that gives each row a group number (0, 1 or 2) to sort on.

#### `get_track_details(track_id)`

Unlike the other tools, this one doesn't search. It takes one exact `TrackId` and returns everything the database knows about that track. The search tools find a song and return its `TrackId`; this tool then gives the details.

**Example: track `2254`**

| Field | Value | Where it comes from |
|---|---|---|
| Name / Composer | Bohemian Rhapsody / Mercury, Freddie | `Track` |
| Album | Greatest Hits I, track 1 of 17 | `Album` |
| Artist | Queen | `Artist` |
| Genre | Rock | `Genre` |
| Format | MPEG audio file, 11.62 MB | `MediaType`, `Track.Bytes` |
| Duration / Price | 5:59 / $0.99 | `Track` |
| Playlists | "Music" (ID 1), "Music" (ID 8) | `PlaylistTrack` + `Playlist` |
| Sales | Sold once, $0.99 revenue | `InvoiceLine` |

**How it works:** it runs three small queries and combines the results into one JSON object:

1. **The track itself**, joined to its album, artist, genre and media type.
2. **The playlists** it appears on.
3. **Sales totals:** `TimesSold`, `InvoiceCount` and `TotalRevenue`. A track that never sold shows `0` instead of being left out.

**Fields worked out in Python:**

- `Duration`: milliseconds turned into `m:ss`, or `h:mm:ss` for an hour or longer (a video track shows `1:28:07`). This is done by the helper `_format_duration()`.
- `SizeMB`: `Bytes` divided by 1,000,000.
- `TrackNumber`: Chinook has no track-number column, so this is the track's position on its album when the album's tracks are sorted by `TrackId`. It isn't guaranteed to match the real album order.

**Errors:** an ID that doesn't exist (e.g. `3504`) returns `{"message": "No track found with TrackId: 3504"}`. An ID of `0` or a negative number returns an `error`, and so does an ID that isn't a whole number, e.g. `"abc"` or `"12.0"` (see "All parameters are strings").

Chinook has several playlists with the same name, which is why each playlist's ID is included.

#### `get_customer_invoices(customer_id)`

Takes one exact `CustomerId` and returns all of that customer's invoices (orders), **most recent first**.

**Example: customer `1` (Luís Gonçalves)**

| InvoiceId | Date | Items | Total |
|---|---|---|---|
| 382 | 2025-08-07 | 9 | $8.91 |
| 327 | 2024-12-07 | 14 | $13.86 |
| 316 | 2024-10-27 | 2 | $1.98 |
| … | … | … | … |
| 98 | 2022-03-11 | 2 | $3.98 |

Totals: `invoice_count` 7, `total_spent` $39.62.

**What it returns:** `CustomerId`, `CustomerName`, `invoice_count`, `total_spent`, and `invoices`. Each invoice has `InvoiceId`, `InvoiceDate`, `BillingCity`, `BillingState`, `BillingCountry`, `ItemCount` and `Total`.

**How it works:**

1. **Checks the customer exists** in the `Customer` table. If not (e.g. `60`), it returns the message `No customer found with CustomerId: 60`.
2. **Gets their invoices** from the `Invoice` table, sorted with `ORDER BY InvoiceDate DESC`. `DESC` means descending, so the newest comes first.
3. **Counts the items** on each invoice by adding up `Quantity` in the `InvoiceLine` table.
4. **Adds up `total_spent`** in Python from the invoice totals.

**Sorting text dates:** SQLite stores `InvoiceDate` as text like `"2025-08-07 00:00:00"`. Because the year comes first, then month, then day, sorting the text A to Z also sorts by date, so no conversion is needed. Two invoices on the same day are sorted by `InvoiceId`.

**No limit:** unlike the search tools, it returns *all* invoices. That's safe here because each Chinook customer has only 6 or 7.

**Privacy:** it returns only the customer's name, not their email, phone or street address.

> **Security note:** the tool returns invoices for *any* `CustomerId` it's given. The docstring tells the agent to pass only the ID of the customer it's helping, but that's an instruction to the AI, not a real security check. In a real app, the customer ID should come from the login step, not from the chat, so one customer can't see another's orders.

#### Purchased tracks: `fetch_purchased_tracks` and `get_customer_purchased_tracks`

These list **purchased tracks** sorted by unit price, **highest first**. Each row is one track that was bought, not a whole invoice.

**Invoice vs. InvoiceLine:** an `Invoice` is one order, like a receipt. An `InvoiceLine` is one item on that receipt. Since we want one row per purchased track, the query starts from `InvoiceLine`. Every Chinook `InvoiceLine` has `Quantity = 1`, so one line really is one track.

**Two parts:**

| Name | Kind | Who it's for | Covers |
|---|---|---|---|
| `fetch_purchased_tracks(customer_id=None, limit=None, offset=0)` | Plain Python function | Internal reports | **All** customers, or one if `customer_id` is given. All 2,240 purchases by default. |
| `get_customer_purchased_tracks(customer_id, limit=20, offset=0)` | Agent tool (`@tool`) | The AI agent | **One** customer only, 1–50 rows per page |

They're split so the AI agent can never list every customer's purchases. Only the per-customer version is a tool.

**Example: customer `1`** has 38 purchases. The first page starts:

| Price | Date | Track | Artist |
|---|---|---|---|
| $1.99 | 2022-03-11 | Take the Celestra | Battlestar Galactica |
| $1.99 | 2022-03-11 | Experiment In Terra | Battlestar Galactica |
| $0.99 | 2025-08-07 | Paranoid | Ozzy Osbourne |
| … | … | … | … |

**What each row contains:** `InvoiceLineId`, `InvoiceId`, `InvoiceDate`, `CustomerId`, `TrackId`, `TrackName`, `ArtistName`, `AlbumTitle`, `GenreName`, `MediaType`, `UnitPrice`, `Quantity` and `LineTotal` (`UnitPrice × Quantity`). The response also has `total_purchases`, `offset` and `result_count`.

**Facts from the data that shaped the design:**

- **Only two prices exist:** $0.99 (2,129 purchases) and $1.99 (111). Every $1.99 item is a video (TV shows, drama, sci-fi), so "highest price first" puts videos first.
- **Sorting needs tie-breakers.** With only two prices, `ORDER BY UnitPrice DESC` alone would leave the order inside each price group random. The full sort is `UnitPrice DESC, InvoiceDate DESC, InvoiceLineId DESC`: highest price, then newest purchase, then a final key so the order never changes.
- **Which price to use:** `InvoiceLine.UnitPrice` is the price the customer **actually paid**, while `Track.UnitPrice` is **today's** catalog price. They match in Chinook now, but could differ if prices change, so the query uses the price paid.
- **Repeat purchases appear more than once.** 256 tracks were bought more than once, and each purchase is its own row.

**Paging with `limit` and `offset`:** `limit` is how many rows to return and `offset` is how many to skip first. To read 20 at a time, use `offset=0`, then `20`, then `40`, and so on. `total_purchases` tells you when you've reached the end. In SQLite, `LIMIT -1` means "no limit", which is how `fetch_purchased_tracks` returns everything when `limit=None`.

**Checks run by hand:**

- All 2,240 rows come back in the right order, and they add up to $2,328.60 of revenue.
- Pages of 1,000 join back into the same full list.
- For all 59 customers, the counts are correct and only that customer's own rows come back.

**Errors:** an ID of `0` or below, or a non-numeric `customer_id`, `limit` or `offset`, returns an `error`. An unknown customer (e.g. `60`) returns a not-found `message`. A negative `offset` is treated as `0`. An `offset` past the end returns no rows, still gives the total, and adds a `message` saying the end was reached.

> **Security note:** same as `get_customer_invoices`. The tool returns purchases for any `CustomerId` it's given, so in a real app the ID must come from the login step, not from the chat.

#### `get_invoice_support_rep(invoice_id, customer_id="")`

Finds the support representative responsible for one invoice, so the agent can tell a customer who to contact about an order.

**Invoices have no rep column.** The rep is stored on the *customer* (`Customer.SupportRepId`), so the query follows a chain of links:

```
Invoice.CustomerId  →  Customer.SupportRepId  →  Employee (the rep)
                                                 Employee.ReportsTo  →  Employee (the manager)
```

```sql
FROM Invoice AS i
JOIN Customer AS c ON c.CustomerId = i.CustomerId
LEFT JOIN Employee AS e ON e.EmployeeId = c.SupportRepId   -- the rep
LEFT JOIN Employee AS m ON m.EmployeeId = e.ReportsTo      -- the rep's manager
WHERE i.InvoiceId = :invoice_id
```

The `Employee` table is joined **twice**, with two aliases: `e` for the rep and `m` for the manager. This is a **self-join**: `ReportsTo` points back into the same table.

**Why `LEFT JOIN`:** a plain `JOIN` would drop the invoice completely if the customer had no rep or the rep had no manager (the General Manager reports to no one). `LEFT JOIN` keeps the invoice and fills the missing columns with `NULL`.

**Example: invoice `98`**

| Field | Value |
|---|---|
| Invoice | 98, 2022-03-11, $3.98 |
| Customer | 1, Luís Gonçalves |
| Rep | Jane Peacock (EmployeeId 3), Sales Support Agent, jane@chinookcorp.com, +1 (403) 262-3443 |
| Manager | Nancy Edwards |

**What it returns:** `InvoiceId`, `InvoiceDate`, `Total`, `CustomerId`, `CustomerName`, and a nested `SupportRep` object with `EmployeeId`, `RepName`, `Title`, `Email`, `Phone` and `ManagerName`. The rep's columns are moved into `SupportRep` in Python. If the customer has no rep, `SupportRep` is `null`.

**Facts from the data:**

- Every one of the 59 customers has a rep, so `SupportRep` is never `null` in Chinook today. The `null` handling is there in case that changes.
- Only three employees are reps, all Sales Support Agents who report to Nancy Edwards:

| Rep | Customers | Invoices |
|---|---|---|
| Jane Peacock (3) | 21 | 146 |
| Margaret Park (4) | 20 | 140 |
| Steve Johnson (5) | 18 | 126 |

**Optional ownership check:** if `customer_id` is given (an empty string or `null` means it isn't), the invoice must belong to that customer. If it belongs to someone else, the tool returns the **same** "No invoice found" message as for an invoice that doesn't exist. That way it doesn't confirm that another customer's invoice ID is real.

**Errors:** an ID of `0` or below, or a non-numeric ID (e.g. `"INV-98"`), returns an `error`. An unknown invoice (e.g. `99999`) or the wrong owner returns a not-found `message`.

> **Limitation:** Chinook doesn't keep a history of rep assignments. The tool returns the customer's **current** rep, which may not be the rep who looked after them when the invoice was created.

> **Security note:** inside the graph, `customer_id` is injected from the verified state, so the agent can only see the verified customer's invoices. It is still optional for direct Python calls, where leaving it empty skips the check.

#### `get_invoice_line_items(invoice_id, customer_id="")`

Lists every track purchased on one invoice, like reading a receipt line by line. It's the natural next step after `get_customer_invoices`: that tool lists a customer's orders, and this one opens a single order.

**How it differs from `get_customer_purchased_tracks`:**

| | `get_customer_purchased_tracks` | `get_invoice_line_items` |
|---|---|---|
| Covers | All of one customer's purchases | One invoice only |
| Order | Highest price first | Receipt order (`InvoiceLineId`) |
| Paging | Yes, 20 per page (max 50) | No: an invoice has at most 14 lines |
| Invoice details | Repeated on every row | Given once, at the top |

**Two queries:**

1. **The invoice header**: invoice date, customer name, billing city/state/country and `Total`. This also tells us whether the invoice exists and who owns it.
2. **The line items**: reuses the shared `_PURCHASED_TRACKS` joins (`InvoiceLine` → `Invoice` → `Track` → `Album` → `Artist`, plus `Genre` and `MediaType`), filtered with `WHERE il.InvoiceId = :invoice_id` and sorted by `InvoiceLineId`.

It doesn't reuse `_PURCHASED_COLUMNS` (which repeats `InvoiceDate` and `CustomerId` on every row) or `_PURCHASED_ORDER` (price order suits a purchase history, not a receipt).

**Example: invoice `98`** (customer 1, Luís Gonçalves, São José dos Campos, Brazil)

| Line | Track | Artist | Media | Duration | Price |
|---|---|---|---|---|---|
| 531 | Experiment In Terra | Battlestar Galactica (Classic) | Protected MPEG-4 video file | 48:44 | $1.99 |
| 532 | Take the Celestra | Battlestar Galactica (Classic) | Protected MPEG-4 video file | 48:48 | $1.99 |

`item_count` 2, `Total` $3.98.

**What it returns:** `InvoiceId`, `InvoiceDate`, `CustomerId`, `CustomerName`, `BillingCity`, `BillingState`, `BillingCountry`, `Total`, `item_count` (the sum of `Quantity`), and `line_items`. Each line has `InvoiceLineId`, `TrackId`, `TrackName`, `ArtistName`, `AlbumTitle`, `GenreName`, `MediaType`, `Duration`, `UnitPrice`, `Quantity` and `LineTotal`.

**Design choices:**

- **`Duration` replaces `Milliseconds`**, formatted as `m:ss` (or `h:mm:ss`) by the existing `_format_duration()` helper.
- **Price paid, not catalog price.** `UnitPrice` comes from `InvoiceLine`, the same rule as the purchased-tracks tool.
- **No subtotal field.** The lines always add up to `Invoice.Total`, so a separate subtotal would add nothing.
- **Privacy:** billing street address and postal code are left out, as in `get_customer_invoices`.
- **Ownership check** works exactly like `get_invoice_support_rep`: if `customer_id` is given and doesn't match, the tool returns the same "No invoice found" message as for a missing invoice.

**Facts from the data:**

- Invoices have 1 to 14 lines (412 invoices in total), and none has zero lines.
- For every invoice, the sum of `LineTotal` equals `Invoice.Total`, and `InvoiceLine.UnitPrice` equals `Track.UnitPrice`.
- Every purchased track has an album, genre and media type.

**Checks run by hand:** all 412 invoices return line totals that add up to `Total`, with `item_count` equal to the number of lines. Invoice 327 (the largest, 14 lines, $13.86) comes back in receipt order.

**Errors:** an `invoice_id` or `customer_id` that is `0` or below or not a whole number returns an `error`. An unknown invoice (e.g. `99999`) or the wrong owner returns a not-found `message`.

> **Security note:** same as `get_invoice_support_rep`. Inside the graph, `customer_id` is injected from the verified state.

### Key ideas

- **Fuzzy matching with `LIKE`.** `WHERE ar.Name LIKE '%zeppelin%'` means "the name contains *zeppelin* anywhere". The `%` symbols mean "any text here". In SQLite, `LIKE` ignores upper/lower case, so `"IRON maiden"` still finds Iron Maiden.
- **Escaping wildcards.** `%` and `_` are special characters in `LIKE`. `_escape_like()` puts a backslash in front of them so that searching for `"%"` looks for a literal percent sign instead of matching every artist.
- **Bound parameters stop SQL injection.** User input goes in as `:pattern` and is never pasted into the SQL text. Input like `"'; DROP TABLE Album; --"` is just searched for as a name and does no harm.
- **Limits are clamped.** `max(1, min(limit, MAX))` keeps a number inside a safe range. A huge limit can't flood the AI with thousands of rows, and `0` or a negative number still returns at least 1 result.
- **Shared `FROM`/`WHERE` text.** `_TRACKS_BY_ARTIST`, `_TRACKS_BY_GENRE`, `_TRACKS_BY_TITLE` and `_PURCHASED_TRACKS` are written once and used by both the count query and the sample query, so the total always describes the same tracks as the sample.
- **Window functions (genre tool).** `ROW_NUMBER() OVER (PARTITION BY ArtistId ORDER BY TrackId)` numbers each artist's tracks 1, 2, 3, … by `TrackId`. Keeping only `rn = 1` gives exactly one track per artist, and the same one every time.
- **Not found is a `message`, not an `error`.** See "Not-found messages" above.

### Testing

`tests/tools_test.py` has 25 tests for `search_albums_by_artist`. Run them from the project root:

```
python -m pytest tests/tools_test.py -v
```

The other tools have no dedicated test file of their own yet, but the agent tests run them for real against the database:

- `tests/music_agent_test.py` exercises `search_tracks_by_artist`, `browse_tracks_by_genre`, `search_songs_by_title` and `get_track_details`, including bad input and not-found results.
- `tests/invoice_agent_test.py` exercises `get_customer_invoices`, `get_customer_purchased_tracks` (and so `fetch_purchased_tracks`), `get_invoice_support_rep` and `get_invoice_line_items`. This includes the ownership checks: another customer's invoice is reported as not found.

These tests check what the agents do with each tool's result rather than every option of each tool, so edge cases such as limits and paging are still only partly covered.
