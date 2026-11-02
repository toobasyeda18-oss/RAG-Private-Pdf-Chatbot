import json
import os
import re
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_core import ask, ask_stream, qdrant, llm, COLLECTION, MODEL_NAME
from indexer import index_pdf

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Private PDF Chatbot API")


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/full")
def full_health():
    result = {
        "api": "ok",
        "qdrant": "unreachable",
        "ollama": "unreachable",
        "model_in_use": repr(MODEL_NAME),
        "installed_models": [],
        "model_installed": False,
    }

    try:
        qdrant.get_collections()
        result["qdrant"] = "ok"
    except Exception as error:
        result["qdrant"] = str(error)[:200]

    try:
        text = str(llm.list())
        result["ollama"] = "ok"
        names = re.findall(r"(?:model|name)['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", text)
        result["installed_models"] = names
        result["model_installed"] = MODEL_NAME in names or MODEL_NAME in text
    except Exception as error:
        result["ollama"] = str(error)[:200]

    return result


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    save_path = os.path.join(UPLOAD_DIR, "current.pdf")
    with open(save_path, "wb") as saved_file:
        shutil.copyfileobj(file.file, saved_file)

    chunk_count = index_pdf(save_path, file.filename)
    if chunk_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No readable text found. Scanned PDFs are not supported yet.",
        )
    return {"message": "PDF processed", "chunks": chunk_count, "name": file.filename}


@app.post("/ask")
def ask_question(body: Question):
    if not qdrant.collection_exists(COLLECTION):
        raise HTTPException(status_code=400, detail="Upload a PDF first.")
    return ask(body.question)


@app.post("/ask_stream")
def ask_stream_endpoint(body: Question):
    if not qdrant.collection_exists(COLLECTION):
        raise HTTPException(status_code=400, detail="Upload a PDF first.")

    def event_lines():
        try:
            for event in ask_stream(body.question):
                yield json.dumps(event) + "\n"
        except Exception as error:
            yield json.dumps({"type": "error", "text": str(error)[:300]}) + "\n"

    return StreamingResponse(event_lines(), media_type="application/x-ndjson")


@app.get("/chunks")
def list_chunks():
    if not qdrant.collection_exists(COLLECTION):
        return {"document": None, "count": 0, "chunks": []}

    points, _ = qdrant.scroll(
        collection_name=COLLECTION,
        limit=500,
        with_payload=True,
        with_vectors=False,
    )

    chunks = []
    for point in points:
        chunks.append(
            {
                "id": point.id,
                "page": point.payload.get("page"),
                "text": point.payload["text"],
            }
        )

    document = points[0].payload.get("source") if points else None
    total = qdrant.count(collection_name=COLLECTION, exact=True).count
    return {"document": document, "count": total, "chunks": chunks}

@app.get("/metrics")
def get_metrics():
    path = "eval/results.json"
    if not os.path.exists(path):
        return {"runs": []}
    with open(path, encoding="utf-8") as results_file:
        return {"runs": json.load(results_file)}