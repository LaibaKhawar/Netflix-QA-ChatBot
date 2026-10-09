"""Data loading, document building, and exact-lookup routing for the Netflix Q&A app.

This module only depends on pandas so it can be tested without model downloads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "netflix_titles.csv"

# Ordered, so every run builds identical document text (a set would not be).
COLUMNS = (
    "Title", "Category", "Director", "Cast", "Country", "Release_Date",
    "Rating", "Duration", "Type", "Description",
)

# Titles this short ("9", "21", "Up") only match when the user quotes them.
MIN_UNQUOTED_TITLE_LENGTH = 4
DIRECTOR_CUES = ("director", "directed", "who made", "filmmaker")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the source dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    dataframe = pd.read_csv(path).fillna("Unknown")
    missing = set(COLUMNS).difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    return dataframe


def build_records(dataframe: pd.DataFrame) -> list[tuple[str, dict[str, str]]]:
    """Return (page_content, metadata) pairs, one per title."""
    records = []
    for _, row in dataframe.iterrows():
        text = "\n".join(f"{column}: {row[column]}" for column in COLUMNS)
        metadata = {
            "title": str(row["Title"]),
            "category": str(row["Category"]),
            "director": str(row["Director"]),
            "release_date": str(row["Release_Date"]),
            "rating": str(row["Rating"]),
            "genres": str(row["Type"]),
        }
        records.append((text, metadata))
    return records


def _title_pattern(title: str) -> re.Pattern[str]:
    # Word-boundary match that also works for titles starting or ending in symbols ("3%", "'89").
    return re.compile(r"(?<!\w)" + re.escape(title.casefold()) + r"(?!\w)")


def _quoted_spans(question: str) -> list[str]:
    return [m.casefold().strip() for m in re.findall(r"[\"“”'‘’*]([^\"“”'‘’*]{1,80})[\"“”'‘’*]", question)]


def find_title(question: str, titles: list[str]) -> str | None:
    """Return the longest catalogue title mentioned in the question, or None."""
    normalized = question.casefold()
    quoted = set(_quoted_spans(question))
    best: str | None = None
    for title in titles:
        folded = title.casefold()
        if folded in quoted:
            return title
        if len(folded) < MIN_UNQUOTED_TITLE_LENGTH:
            continue
        if (best is None or len(title) > len(best)) and _title_pattern(title).search(normalized):
            best = title
    return best


@dataclass
class LookupResult:
    answer: str
    sources: list[dict[str, str]] = field(default_factory=list)


def is_director_question(question: str) -> bool:
    normalized = question.casefold()
    return any(cue in normalized for cue in DIRECTOR_CUES)


def exact_lookup(question: str, dataframe: pd.DataFrame) -> LookupResult | None:
    """Answer director questions about a named title straight from the dataset, without an LLM."""
    if not is_director_question(question):
        return None
    title = find_title(question, dataframe["Title"].astype(str).tolist())
    if title is None:
        return None
    rows = dataframe[dataframe["Title"].astype(str) == title]
    sources = [
        {"title": title, "category": str(r["Category"]), "director": str(r["Director"]),
         "release_date": str(r["Release_Date"]), "rating": str(r["Rating"]), "genres": str(r["Type"])}
        for _, r in rows.iterrows()
    ]
    directors = sorted({s["director"] for s in sources if s["director"] != "Unknown"})
    if not directors:
        return LookupResult(f"The catalogue does not list a director for {title}.", sources)
    if len(sources) > 1:
        return LookupResult(f"The catalogue has {len(sources)} entries titled {title}, directed by {', '.join(directors)}.", sources)
    return LookupResult(f"The director of {title} is {directors[0]}.", sources)


def unique_sources(metadatas: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deduplicate retrieved chunk metadata by title, keeping retrieval order."""
    seen: set[str] = set()
    out = []
    for m in metadatas:
        key = m.get("title", "")
        if key and key not in seen:
            seen.add(key)
            out.append(m)
    return out
