import pandas as pd
import pytest

from qa import COLUMNS, build_records, exact_lookup, find_title, load_data, unique_sources


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_dataset_loads_with_required_columns(data):
    assert set(COLUMNS).issubset(data.columns)
    assert len(data) > 7000


def test_records_have_stable_field_order(data):
    first, meta = build_records(data.head(1))[0]
    assert first.splitlines()[0].startswith("Title: ")
    assert [line.split(":")[0] for line in first.splitlines()] == list(COLUMNS)
    assert meta["title"] == str(data.iloc[0]["Title"])


def test_director_lookup_for_named_title(data):
    result = exact_lookup("Who directed The Irishman?", data)
    assert result is not None
    assert "Martin Scorsese" in result.answer
    assert result.sources[0]["title"] == "The Irishman"


def test_short_titles_do_not_hijack_questions():
    titles = ["9", "21", "Up", "The Irishman"]
    assert find_title("who is the director of a film from 2021 about 9 friends", titles) is None
    assert find_title('who directed "9"?', titles) == "9"


def test_longest_matching_title_wins():
    titles = ["Love", "Love Actually", "Actually"]
    assert find_title("who directed love actually", titles) == "Love Actually"


def test_titles_with_symbols_match_on_word_boundaries():
    assert find_title("who directed '3%'?", ["3%"]) == "3%"
    assert find_title("director of 23% off", ["3%"]) is None


def test_non_director_questions_are_not_routed(data):
    assert exact_lookup("Recommend a thriller like The Irishman", data) is None


def test_missing_director_is_reported_honestly():
    df = pd.DataFrame([{c: "Unknown" for c in COLUMNS} | {"Title": "Some Show"}])
    result = exact_lookup("Who directed Some Show?", df)
    assert result is not None and "does not list a director" in result.answer


def test_unique_sources_keeps_order_and_dedupes():
    out = unique_sources([{"title": "A"}, {"title": "B"}, {"title": "A"}])
    assert [s["title"] for s in out] == ["A", "B"]
