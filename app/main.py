from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import os
import shutil

from app.rag_pipeline import RAGPipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

rag = RAGPipeline()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ---------------------------
# Upload PDF
# ---------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = rag.ingest(file_path)

    return {
        "message": "PDF processed",
        "chunks": result["chunks"]
    }


# ---------------------------
# NORMAL QUERY (fallback)
# ---------------------------
@app.post("/query")
async def query(request: QueryRequest):
    result = rag.query(request.question, request.top_k)
    return result


# ---------------------------
# 🔥 STREAMING QUERY (NEW)
# ---------------------------
@app.post("/stream")
async def stream(request: QueryRequest):

    def generate():
        try:
            for chunk in rag.stream_query(request.question, request.top_k):
                yield chunk
        except Exception as e:
            yield f"\nError: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")


# ---------------------------
# STATUS
# ---------------------------
@app.get("/status")
async def status():
    return {
        "ready": rag.is_ready(),
        "documents_loaded": rag.doc_count()
    }


# ---------------------------
# RESET
# ---------------------------
@app.delete("/reset")
async def reset():
    rag.reset()
    return {"message": "reset done"}