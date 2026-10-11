"""
Week 2 · Day 4 — Multi-PDF RAG App with Sources (Anthropic version)

Upload several PDFs at once, index them into ONE FAISS store, ask questions
across all of them, and get every answer back with file name + page number.

LLM:        Claude (Anthropic) via langchain-anthropic
Embeddings: Anthropic has no embeddings API, so this uses a free local
            HuggingFace model (sentence-transformers). No extra key needed.

Run:
    export ANTHROPIC_API_KEY="your-key"
    streamlit run rag_app.py
"""

import os
import tempfile

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------- settings
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800        # small enough that one chunk ≈ one idea on one page
CHUNK_OVERLAP = 120
TOP_K = 5               # chunks retrieved per question

st.set_page_config(page_title="Multi-PDF RAG", page_icon="📚")
st.title("📚 Multi-PDF RAG with Sources")
st.caption(f"Answers by Claude ({CLAUDE_MODEL}) · local embeddings · FAISS")

# API key comes from the environment, never from the code
if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("Set the ANTHROPIC_API_KEY environment variable, then restart the app.")
    st.stop()


# ------------------------------------------------------- cached resources
@st.cache_resource
def get_embeddings():
    # Downloads the model once (~90 MB), then runs locally on CPU
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource
def get_llm():
    return ChatAnthropic(model=CLAUDE_MODEL, max_tokens=1024)


# --------------------------------------------------------------- indexing
def build_store(files):
    """Load every uploaded PDF, split, and embed all chunks into ONE store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            path = os.path.join(tmp, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())

            pages = PyPDFLoader(path).load()   # one Document per page
            for p in pages:
                # PyPDFLoader stores the temp path; keep the real file name
                p.metadata["source"] = f.name
            # split_documents copies metadata (source, page) into every chunk
            chunks.extend(splitter.split_documents(pages))

    if not chunks:
        return None, 0
    store = FAISS.from_documents(chunks, get_embeddings())
    return store, len(chunks)


files = st.file_uploader(
    "Upload PDFs", type="pdf", accept_multiple_files=True
)

if files:
    # Rebuild only when the set of uploaded files changes
    upload_key = tuple(sorted((f.name, f.size) for f in files))
    if st.session_state.get("upload_key") != upload_key:
        with st.spinner(f"Indexing {len(files)} PDF(s)..."):
            store, n_chunks = build_store(files)
        if store is None:
            st.error("No text could be extracted. Are these scanned PDFs?")
            st.stop()
        st.session_state.store = store
        st.session_state.n_chunks = n_chunks
        st.session_state.upload_key = upload_key
        st.session_state.history = []

    st.success(
        f"Uploaded: {', '.join(f.name for f in files)}  \n"
        f"Indexed {st.session_state.n_chunks} chunks."
    )


# ---------------------------------------------------------------- answering
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions using ONLY the context excerpts provided. "
     "Each excerpt is labelled with its file name and page.\n"
     "Rules:\n"
     "1. Cite the file and page for each fact, like [handbook.pdf p.7].\n"
     "2. If excerpts from different files or pages DISAGREE, do not pick one "
     "quietly. Say they conflict and state what each source says, with citations.\n"
     "3. If the context does not contain the answer, say you could not find it "
     "in the uploaded documents. Do not use outside knowledge."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def page_no(doc):
    # PyPDFLoader pages are 0-based; show the human page number
    return doc.metadata.get("page", 0) + 1


def format_context(docs):
    return "\n\n".join(
        f"[{d.metadata['source']} p.{page_no(d)}]\n{d.page_content}" for d in docs
    )


def answer(question):
    retriever = st.session_state.store.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)
    chain = PROMPT | get_llm()
    reply = chain.invoke({"context": format_context(docs), "question": question})

    # Unique (file, page) pairs, in retrieval order
    sources = list(dict.fromkeys((d.metadata["source"], page_no(d)) for d in docs))
    return reply.content, sources, docs


if "store" in st.session_state:
    # Show earlier Q&A in this session
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["q"])
        with st.chat_message("assistant"):
            st.write(item["a"])
            st.caption("Sources: " + " | ".join(item["s"]))

    question = st.chat_input("Ask a question across all uploaded PDFs")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and asking Claude..."):
                text, sources, docs = answer(question)
            src_labels = [f"{name} - page {pg}" for name, pg in sources]
            st.write(text)
            st.caption("Sources: " + " | ".join(src_labels))

            # Chunk-size check: many distinct pages for one answer is a hint
            if len(sources) >= 5:
                st.info(
                    f"This answer drew on {len(sources)} different pages. "
                    "If that seems scattered, try a larger CHUNK_SIZE."
                )
            with st.expander("Show retrieved excerpts"):
                for d in docs:
                    st.markdown(f"**{d.metadata['source']} - page {page_no(d)}**")
                    st.text(d.page_content[:600])

        st.session_state.history.append(
            {"q": question, "a": text, "s": src_labels}
        )
else:
    st.info("Upload one or more PDFs to begin.")