"""
RAG Pipeline — Core Logic
--------------------------
Flow: PDF → Extract Text → Chunk → Embed → FAISS → Retrieve → LLM (Groq)
"""

import os
from typing import Optional, Generator
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# ── Choose your embedding mode ──────────────────────────────────────────────
# Option A (recommended): Real semantic embeddings — better retrieval quality
#   pip install sentence-transformers langchain-huggingface
#   then set USE_REAL_EMBEDDINGS = True
#
# Option B (lightweight/testing): Fake embeddings — no extra install needed
#   Results will be random — only use for pipeline testing, NOT production
# ─────────────────────────────────────────────────────────────────────────────
USE_REAL_EMBEDDINGS = False   # ← flip to True when sentence-transformers is installed

if USE_REAL_EMBEDDINGS:
    from langchain_huggingface import HuggingFaceEmbeddings
else:
    from langchain.embeddings import FakeEmbeddings  # fixed import path

load_dotenv()

GROQ_MODEL    = "llama-3.1-8b-instant"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not found in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""
)


class RAGPipeline:

    def __init__(self):
        self.vectorstore: Optional[FAISS] = None
        self._chunk_count: int = 0

        # ── Embeddings ───────────────────────────────────────────────────────
        if USE_REAL_EMBEDDINGS:
            print("Loading HuggingFace embeddings (first run downloads ~90MB)...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            print("⚠️  Lightweight mode ON — using FakeEmbeddings (for testing only)")
            self.embeddings = FakeEmbeddings(size=384)

        # ── LLM ─────────────────────────────────────────────────────────────
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to your .env file.\n"
                "Get a free key at: https://console.groq.com"
            )

        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.2,
            groq_api_key=groq_key,
            streaming=True,
        )

        print("✅ RAG Pipeline ready.")

    # ── INGEST ───────────────────────────────────────────────────────────────
    def ingest(self, pdf_path: str) -> dict:
        """Load PDF → split into chunks → embed → store in FAISS."""

        # 1. Load
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # 2. Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(docs)
        self._chunk_count = len(chunks)

        # 3. Embed + index
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local("faiss_index")

        return {"chunks": self._chunk_count}

    # ── QUERY (blocking) ─────────────────────────────────────────────────────
    def query(self, question: str) -> dict:
        """Retrieve relevant chunks → ask LLM → return full answer."""
        if not self.is_ready():
            raise RuntimeError("No document loaded. Upload a PDF first.")

        docs = self._retrieve(question)
        context = "\n\n".join(d.page_content[:500] for d in docs)
        sources = list({f"Page {d.metadata.get('page', 0) + 1}" for d in docs})

        prompt = RAG_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)

        return {
            "answer": response.content.strip(),
            "sources": sources,
            "chunks_used": len(docs),
        }

    # ── QUERY (streaming) ────────────────────────────────────────────────────
    def stream_query(self, question: str) -> Generator[str, None, None]:
        """Same as query() but yields tokens as they arrive (for SSE / streaming)."""
        if not self.is_ready():
            raise RuntimeError("No document loaded. Upload a PDF first.")

        docs = self._retrieve(question)
        context = "\n\n".join(d.page_content for d in docs)
        prompt = RAG_PROMPT.format(context=context, question=question)

        for chunk in self.llm.stream(prompt):
            yield chunk.content

    # ── HELPERS ──────────────────────────────────────────────────────────────
    def _retrieve(self, question: str, k: int = 3):
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        ).invoke(question)

    def is_ready(self) -> bool:
        return self.vectorstore is not None

    def doc_count(self) -> int:
        return self._chunk_count

    def reset(self) -> None:
        self.vectorstore = None
        self._chunk_count = 0