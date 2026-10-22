"""Tests for tools.search_albums_by_artist against the in-memory Chinook database.

Run from the project root with either:
    python TestingTool/toolstest.py
    python -m pytest TestingTool/toolstest.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Make the project root importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools import MAX_RESULTS, search_albums_by_artist  # noqa: E402

EXPECTED_COLUMNS = ["AlbumId", "AlbumTitle", "ArtistId", "ArtistName", "TrackCount"]


def search(artist_name: str, **kwargs) -> dict:
    """Call the tool the way an agent would and parse its JSON output."""
    return json.loads(search_albums_by_artist.invoke({"artist_name": artist_name, **kwargs}))


# --- Matching ---------------------------------------------------------------

def test_exact_name_returns_that_artists_albums():
    result = search("AC/DC")
    assert result["row_count"] == 2
    assert {r["AlbumTitle"] for r in result["rows"]} == {
        "For Those About To Rock We Salute You",
        "Let There Be Rock",
    }


def test_partial_name_matches():
    result = search("zeppelin", limit=100)
    artists = {r["ArtistName"] for r in result["rows"]}
    assert artists == {"Led Zeppelin", "Dread Zeppelin"}


@pytest.mark.parametrize("name", ["iron maiden", "IRON MAIDEN", "IrOn MaIdEn"])
def test_match_is_case_insensitive(name):
    result = search(name, limit=100)
    assert result["row_count"] > 0
    assert all(r["ArtistName"] == "Iron Maiden" for r in result["rows"])


def test_surrounding_whitespace_is_ignored():
    assert search("  AC/DC  ") == search("AC/DC")


def test_unknown_artist_returns_no_rows():
    result = search("no such artist xyz")
    assert result["row_count"] == 0
    assert result["rows"] == []


# --- Output shape -----------------------------------------------------------

def test_result_has_expected_columns():
    result = search("AC/DC")
    assert result["columns"] == EXPECTED_COLUMNS
    for row in result["rows"]:
        assert list(row) == EXPECTED_COLUMNS


def test_track_counts_are_correct():
    rows = {r["AlbumTitle"]: r["TrackCount"] for r in search("AC/DC")["rows"]}
    assert rows == {"For Those About To Rock We Salute You": 10, "Let There Be Rock": 8}


def test_results_sorted_by_artist_then_title():
    rows = search("a", limit=MAX_RESULTS)["rows"]
    keys = [(r["ArtistName"], r["AlbumTitle"]) for r in rows]
    assert keys == sorted(keys)


# --- Limit ------------------------------------------------------------------

def test_default_limit_is_25():
    assert search("a")["row_count"] == 25


def test_custom_limit_is_respected():
    assert search("a", limit=5)["row_count"] == 5


@pytest.mark.parametrize("limit, expected", [(0, 1), (-10, 1), (1000, MAX_RESULTS)])
def test_limit_is_clamped(limit, expected):
    assert search("a", limit=limit)["row_count"] == expected


def test_has_more_when_results_are_cut_off():
    result = search("AC/DC", limit=1)
    assert result["row_count"] == 1
    assert result["has_more"] is True


@pytest.mark.parametrize("limit", [2, 25])
def test_no_more_when_list_is_complete(limit):
    # AC/DC has exactly 2 albums, so a limit of 2 is not a cut-off.
    result = search("AC/DC", limit=limit)
    assert result["row_count"] == 2
    assert result["has_more"] is False


# --- Input safety -----------------------------------------------------------

@pytest.mark.parametrize("name", ["", "   "])
def test_empty_name_returns_error(name):
    assert "error" in search(name)


@pytest.mark.parametrize("wildcard", ["%", "_", "%%"])
def test_like_wildcards_are_matched_literally(wildcard):
    # Unescaped, these would match every artist.
    assert search(wildcard)["row_count"] == 0


def test_sql_injection_attempt_is_harmless():
    result = search("'; DROP TABLE Album; --")
    assert result["row_count"] == 0
    # The Album table must still be there and queryable.
    assert search("AC/DC")["row_count"] == 2


# --- Tool metadata ----------------------------------------------------------

def test_tool_schema_exposed_to_agent():
    assert search_albums_by_artist.name == "search_albums_by_artist"
    assert set(search_albums_by_artist.args) == {"artist_name", "limit"}
    assert search_albums_by_artist.description


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
