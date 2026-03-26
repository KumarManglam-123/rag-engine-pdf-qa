"""
RAG Pipeline — Core Logic
Flow: PDF → Chunk → Embed → FAISS → Retrieve → LLM (Groq)
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# 🔥 NEW (memory)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.prompts import PromptTemplate

load_dotenv()

# ✅ CONFIG
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""
)


class RAGPipeline:
    def __init__(self):
        self.vectorstore: Optional[FAISS] = None
        self.qa_chain = None
        self._chunk_count = 0

        # 🔥 MEMORY
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise EnvironmentError("GROQ_API_KEY not set.")

        # ✅ LLM
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.2,
            groq_api_key=groq_key,
            streaming=True
        )

        # Load existing FAISS index
        if os.path.exists("faiss_index"):
            try:
                self.vectorstore = FAISS.load_local(
                    "faiss_index",
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self._build_chain()
                print("Loaded existing FAISS index.")
            except Exception:
                print("Failed to load FAISS index.")

        print("RAG Pipeline ready.")

    # ---------------------------------------------------
    # INGEST
    # ---------------------------------------------------
    def ingest(self, pdf_path: str) -> dict:
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
        except Exception as e:
            raise ValueError(f"Error loading PDF: {str(e)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "]
        )

        chunks = splitter.split_documents(documents)
        self._chunk_count = len(chunks)

        if self.vectorstore:
            self.vectorstore.add_documents(chunks)
        else:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        try:
            self.vectorstore.save_local("faiss_index")
        except Exception:
            pass

        # 🔥 rebuild chain AFTER ingest
        self._build_chain()

        return {
            "chunks": self._chunk_count,
            "model": EMBEDDING_MODEL
        }

    # ---------------------------------------------------
    # 🔥 QUERY WITH MEMORY
    # ---------------------------------------------------
    def query(self, question: str, top_k: int = 3) -> dict:
        if not self.qa_chain:
            raise ValueError("Pipeline not ready.")

        start_time = time.time()

        result = self.qa_chain.invoke({"question": question})

        end_time = time.time()

        sources = []
        for doc in result.get("source_documents", []):
            page = doc.metadata.get("page", "?")
            sources.append(f"Page {page + 1}")

        return {
            "answer": result["answer"].strip(),
            "sources": list(set(sources)),
            "chunks_used": len(result.get("source_documents", [])),
            "response_time": f"{end_time - start_time:.2f}s"
        }

    # ---------------------------------------------------
    # 🔥 STREAMING QUERY (ChatGPT typing)
    # ---------------------------------------------------
    def stream_query(self, question: str, top_k: int = 3):
        if not self.vectorstore:
            raise ValueError("No document uploaded.")

        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )

        docs = retriever.invoke(question)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = RAG_PROMPT.format(
            context=context,
            question=question
        )

        for chunk in self.llm.stream(prompt):
            yield chunk.content

    # ---------------------------------------------------
    # 🔥 BUILD CHAIN WITH MEMORY
    # ---------------------------------------------------
    def _build_chain(self):
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=True
        )

    # ---------------------------------------------------
    # HELPERS
    # ---------------------------------------------------
    def is_ready(self) -> bool:
        return self.vectorstore is not None

    def doc_count(self) -> int:
        return self._chunk_count

    def reset(self):
        self.vectorstore = None
        self.qa_chain = None
        self.memory.clear()   # 🔥 IMPORTANT
        self._chunk_count = 0