# 🚀 RAG Engine — PDF Q&A with Streaming + Memory

A full-stack AI application that allows users to upload PDFs and ask questions about them using a Retrieval-Augmented Generation (RAG) pipeline powered by FAISS, embeddings, and LLMs.

---

## ✨ Features

* 📄 Upload and process PDF documents
* 🔍 Intelligent chunking and semantic search (FAISS)
* 🤖 LLM-powered question answering (Groq - Llama 3)
* ⚡ Real-time streaming responses (ChatGPT-style typing effect)
* 🧠 Conversational memory (context-aware multi-turn chat)
* 🌐 Full-stack UI with interactive chat interface

---

## 🏗️ Tech Stack

**Backend**

* FastAPI
* LangChain
* FAISS (Vector Database)
* HuggingFace Embeddings (all-MiniLM-L6-v2)
* Groq LLM (Llama 3)

**Frontend**

* HTML, CSS, JavaScript (Vanilla JS)
* Streaming UI using Fetch API + ReadableStream

---

## ⚙️ Architecture

```text
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS
                                         ↓
                                   Retriever
                                         ↓
                              LLM (Groq - Llama3)
                                         ↓
                           Streaming Response (UI)
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/KumarManglam-123/rag-engine-pdf-qa
cd rag-engine
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

### 5. Run the server

```bash
uvicorn app.main:app --reload
```

---

### 6. Open in browser

```
http://127.0.0.1:8000
```

---

## 📸 Demo

* Upload a PDF
* Ask questions
* Get real-time streaming answers
* Continue conversation with memory

---

## 🧠 Key Highlights

* Built a complete RAG pipeline from scratch
* Implemented streaming responses using async generators
* Integrated conversational memory for multi-turn dialogue
* Designed efficient document retrieval using FAISS
* Created a responsive ChatGPT-like UI

---

## 📌 Future Improvements

* Multi-document support
* Source highlighting in UI
* Chat history sidebar
* Authentication system
* Deployment (Render / Railway / Docker)

---

## 🤝 Contributing

Feel free to fork and improve this project!

---

## 📄 License

MIT License
