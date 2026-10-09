"""Streamlit application for grounded question answering over Netflix data."""

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # Chroma telemetry is noisy and not needed

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from qa import build_records, exact_lookup, load_data, unique_sources

load_dotenv()

TOP_K = 4
SUGGESTIONS = [
    "Who directed The Irishman?",
    "Find family-friendly animated movies.",
    "Which titles are about space exploration?",
    "Recommend a Korean crime drama series.",
]
ROUTE_LABELS = {
    "exact": "Exact lookup · answered from the dataset, no LLM",
    "rag": f"RAG · top {TOP_K} retrieved titles + Llama 3.1 8B",
    "retrieval": "Retrieval only · no GROQ_API_KEY set, so no generated answer",
}


@st.cache_data
def cached_data():
    return load_data()


@st.cache_resource(show_spinner="Indexing titles with all-MiniLM-L6-v2…")
def create_vectorstore(_dataframe) -> Chroma:
    """Create an in-memory semantic index from title metadata."""
    documents = [Document(page_content=text, metadata=meta) for text, meta in build_records(_dataframe)]
    chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60).split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings)


def answer(question: str, dataframe, vectorstore: Chroma, history: list[dict]) -> dict:
    """Route the question and return {answer, route, sources}."""
    lookup = exact_lookup(question, dataframe)
    if lookup is not None:
        return {"answer": lookup.answer, "route": "exact", "sources": lookup.sources}

    if not os.getenv("GROQ_API_KEY"):
        docs = vectorstore.similarity_search(question, k=TOP_K)
        sources = unique_sources([d.metadata for d in docs])
        return {
            "answer": "These are the most relevant titles in the catalogue. Add a GROQ_API_KEY to get a written answer grounded in them.",
            "route": "retrieval",
            "sources": sources,
        }

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    chain = ConversationalRetrievalChain.from_llm(
        llm, vectorstore.as_retriever(search_kwargs={"k": TOP_K}), return_source_documents=True
    )
    pairs = [(turn["question"], turn["answer"]) for turn in history]
    result = chain.invoke({"question": question, "chat_history": pairs})
    sources = unique_sources([d.metadata for d in result.get("source_documents", [])])
    return {"answer": result["answer"], "route": "rag", "sources": sources}


def render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        st.caption(ROUTE_LABELS[turn["route"]])
        if turn["sources"]:
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for s in turn["sources"]:
                    st.markdown(
                        f"**{s['title']}** · {s['category']} · {s['release_date']} · {s['rating']}  \n"
                        f"Director: {s['director']}  \n"
                        f"<small>{s['genres']}</small>",
                        unsafe_allow_html=True,
                    )


def main() -> None:
    st.set_page_config(page_title="Netflix Q&A", page_icon="🎬", layout="centered")
    st.title("Netflix Q&A")
    st.caption("Grounded questions over the included Netflix catalogue. Director questions are answered by exact lookup; everything else by retrieval-augmented generation.")

    try:
        dataframe = cached_data()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    with st.sidebar:
        st.subheader("Catalogue")
        st.metric("Titles", f"{len(dataframe):,}")
        st.metric("Movies / TV shows", f"{(dataframe['Category'] == 'Movie').sum():,} / {(dataframe['Category'] == 'TV Show').sum():,}")
        st.divider()
        st.subheader("Pipeline")
        st.markdown(
            "1. **Router**: director questions about a named title go to an exact pandas lookup.\n"
            "2. **Retriever**: all-MiniLM-L6-v2 embeddings in Chroma, top 4 titles.\n"
            "3. **Generator**: Llama 3.1 8B on Groq, temperature 0, with chat memory."
        )
        if not os.getenv("GROQ_API_KEY"):
            st.warning("No GROQ_API_KEY found. Running in retrieval-only mode: you will see the retrieved titles but no generated answer.")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.pop("history", None)
            st.rerun()

    vectorstore = create_vectorstore(dataframe)
    history: list[dict] = st.session_state.setdefault("history", [])

    for turn in history:
        render_turn(turn)

    question = st.chat_input("Ask about a title, director, genre, or description")
    if not history and not question:
        st.write("Try one of these:")
        cols = st.columns(2)
        for i, suggestion in enumerate(SUGGESTIONS):
            if cols[i % 2].button(suggestion, use_container_width=True):
                question = suggestion

    if question:
        with st.spinner("Retrieving…"):
            try:
                turn = {"question": question, **answer(question, dataframe, vectorstore, history)}
            except Exception as error:  # surface API and network failures in the UI
                st.error(f"Could not answer: {error}")
                st.stop()
        history.append(turn)
        st.rerun()


if __name__ == "__main__":
    main()
