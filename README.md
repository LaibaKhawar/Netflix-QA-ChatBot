# Netflix Q&A Chatbot

A retrieval-augmented question-answering application for exploring a Netflix catalogue in natural language. It combines deterministic lookups for precise metadata questions with semantic retrieval and an LLM for broader queries.

## Highlights

- Semantic search using Sentence Transformers and Chroma
- Grounded conversational answers with retrieved catalogue context
- Exact lookup path for deterministic director questions
- Dataset schema validation and clear setup errors
- Streamlit chat interface with conversation history
- Secrets loaded from environment variables—never committed to source control

## Architecture

```text
Question → exact lookup (when applicable)
         → vector search → relevant title metadata → Groq LLM → answer
```

## Project structure

```text
.
├── app.py
├── data/
│   └── netflix_titles.csv
├── .env.example
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/LaibaKhawar/Netflix-QA-ChatBot.git
cd Netflix-QA-ChatBot
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Add a [Groq API key](https://console.groq.com/keys) to `.env`, then run:

```bash
streamlit run app.py
```

## Example questions

- Who directed *The Irishman*?
- Find family-friendly titles from the catalogue.
- Which descriptions mention space exploration?

## Responsible use

Answers are limited by the included dataset and may still contain model errors. The retrieval context should be inspected before using an answer for consequential decisions.
