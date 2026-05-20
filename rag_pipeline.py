"""
turbovec-rag: Local RAG pipeline using turbovec + LangGraph + Claude
"""

import os
from pathlib import Path
from typing import TypedDict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from turbovec.langchain import TurboQuantVectorStore

load_dotenv()

INDEX_PATH = "data/index.tv"
DOCS_DIR = "data/sample_docs"


# ─── State ────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    context: List[str]
    answer: str


# ─── Embeddings ───────────────────────────────────────────────────────────────

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ─── Index management ─────────────────────────────────────────────────────────

def build_index(docs_dir: str = DOCS_DIR, index_path: str = INDEX_PATH) -> TurboQuantVectorStore:
    print(f"Loading documents from '{docs_dir}' ...")
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    raw_docs = loader.load()

    if not raw_docs:
        raise ValueError(f"No documents found in '{docs_dir}'. Add .txt files and try again.")

    print(f"Chunking {len(raw_docs)} document(s) ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"   → {len(chunks)} chunks created")

    print("Embedding and indexing with turbovec ...")
    embeddings = get_embeddings()
    
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    vectorstore = TurboQuantVectorStore.from_documents(
        chunks, embeddings, persist_path=index_path
    )
    print(f"Index saved → {index_path}")

    return vectorstore


def load_index(index_path: str = INDEX_PATH) -> TurboQuantVectorStore:
    print(f"📂 Loading index from '{index_path}' ...")
    return TurboQuantVectorStore.load(index_path, get_embeddings())


def get_vectorstore(force_rebuild: bool = False) -> TurboQuantVectorStore:
    if not force_rebuild and Path(INDEX_PATH).exists():
        return load_index()
    return build_index()


# ─── LangGraph nodes ──────────────────────────────────────────────────────────

def make_retrieve_node(retriever):
    def retrieve(state: RAGState) -> RAGState:
        docs = retriever.invoke(state["question"])
        state["context"] = [doc.page_content for doc in docs]
        print(f"🔍 Retrieved {len(docs)} chunk(s)")
        return state
    return retrieve


def make_generate_node():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
    )

    def generate(state: RAGState) -> RAGState:
        context = "\n\n---\n\n".join(state["context"])

        prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.
If answer is not in context, say "I don't know".

Context:
{context}

Question:
{state['question']}
"""

        response = llm.invoke(prompt)
        state["answer"] = response.content
        return state

    return generate

# ─── Graph assembly ───────────────────────────────────────────────────────────

def build_rag_graph(vectorstore: TurboQuantVectorStore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("generate", make_generate_node())

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def ask(question: str, rag) -> str:
    result = rag.invoke({"question": question, "context": [], "answer": ""})
    return result["answer"]


def main():
    vectorstore = get_vectorstore()
    rag = build_rag_graph(vectorstore)

    print("\nRAG pipeline ready. Type 'quit' to exit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        answer = ask(question, rag)
        print(f"\nAnswer:\n{answer}\n{'─'*60}\n")


if __name__ == "__main__":
    main()
