"""Streamlit application for grounded question answering over Netflix data."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATA_PATH = Path(__file__).parent / "data" / "netflix_titles.csv"
REQUIRED_COLUMNS = {
    "Title", "Category", "Director", "Cast", "Country", "Release_Date",
    "Rating", "Duration", "Type", "Description",
}


@st.cache_data
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the source dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    dataframe = pd.read_csv(path).fillna("Unknown")
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    return dataframe


@st.cache_resource(show_spinner="Indexing titles…")
def create_vectorstore(dataframe: pd.DataFrame) -> Chroma:
    """Create an in-memory semantic index from title metadata."""
    documents = [
        Document(page_content="\n".join(f"{column}: {row[column]}" for column in REQUIRED_COLUMNS))
        for _, row in dataframe.iterrows()
    ]
    chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60).split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings)


def exact_lookup(question: str, dataframe: pd.DataFrame) -> str | None:
    """Answer deterministic title/director questions without invoking an LLM."""
    normalized = question.casefold()
    if "director" not in normalized:
        return None
    for _, row in dataframe.iterrows():
        if str(row["Title"]).casefold() in normalized:
            return f"The director of {row['Title']} is {row['Director']}."
    return None


def main() -> None:
    st.set_page_config(page_title="Netflix Q&A", page_icon="🎬")
    st.title("🎬 Netflix Q&A")
    st.caption("Ask grounded questions about the included Netflix catalogue.")

    if not os.getenv("GROQ_API_KEY"):
        st.info("Add GROQ_API_KEY to a .env file to start chatting.")
        st.stop()

    try:
        dataframe = load_data()
        vectorstore = create_vectorstore(dataframe)
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    history = st.session_state.setdefault("chat_history", [])
    question = st.chat_input("Ask about a title, director, genre, or description")
    for prompt, answer in history:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            st.write(answer)

    if question:
        answer = exact_lookup(question, dataframe)
        if answer is None:
            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
            chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(search_kwargs={"k": 4}))
            answer = chain.invoke({"question": question, "chat_history": history})["answer"]
        history.append((question, answer))
        st.rerun()


if __name__ == "__main__":
    main()
