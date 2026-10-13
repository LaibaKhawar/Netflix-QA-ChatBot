# Netflix Q&A Chatbot

A retrieval-augmented question-answering app for exploring a Netflix catalogue in natural language. Director questions about a named title are answered by an exact lookup over the dataset; everything else goes through semantic retrieval and an LLM.

**Case study:** [Read it on the portfolio](https://www.laibakhawar.com/work/netflix-qa)

## How it works

```text
Question ──► Router ──► director question about a named title? ──► exact pandas lookup ──► answer + source
                │
                └──► all-MiniLM-L6-v2 embeddings in Chroma (top 4 titles) ──► Llama 3.1 8B on Groq ──► answer + sources
```

- **Router.** A question is routed to the exact lookup when it asks about a director and names a catalogue title. Titles are matched on word boundaries and the longest match wins; titles shorter than four characters (such as "9" or "21") only match when quoted, so they cannot hijack unrelated questions.
- **Retriever.** Each title becomes a document with a fixed field order (title, category, director, cast, country, release date, rating, duration, genres, description), embedded locally with `sentence-transformers/all-MiniLM-L6-v2` and indexed in an in-memory Chroma store.
- **Generator.** `llama-3.1-8b-instant` on Groq at temperature 0, through LangChain's `ConversationalRetrievalChain` with chat history.
- **Interface.** Every answer shows which route produced it and an expandable list of the source titles used.

## Retrieval-only mode

Without a `GROQ_API_KEY`, the app still runs: exact lookups work as normal, and other questions return the four most relevant titles from the catalogue instead of a generated answer. The sidebar says when this mode is active.

## Project structure

```text
.
├── app.py                 # Streamlit interface, retrieval and generation
├── qa.py                  # Data loading, document building, routing (pandas only)
├── tests/test_qa.py       # Unit tests for routing and document building
├── data/netflix_titles.csv
├── .env.example
└── requirements.txt
```

## Getting started

Requires Python 3.10 to 3.12 (the pinned LangChain release does not support 3.13+).

```bash
git clone https://github.com/LaibaKhawar/Netflix-QA-ChatBot.git
cd Netflix-QA-ChatBot
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # optional: add a Groq API key for generated answers
streamlit run app.py
```

A [Groq API key](https://console.groq.com/keys) (free tier available) enables generated answers. The first launch downloads the embedding model and indexes the 7,789 titles, which takes around half a minute.

## Tests

```bash
pip install pytest
pytest
```

## Example questions

- Who directed *The Irishman*?
- Find family-friendly animated movies.
- Which titles are about space exploration?

## Limitations

- Answers are limited to the included dataset and can still contain model errors; check the listed sources before relying on an answer.
- The vector index is rebuilt in memory on each cold start.
