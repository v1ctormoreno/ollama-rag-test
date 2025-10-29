import os, uuid, tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import requests
import pandas as pd
from pypdf import PdfReader
import docx

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("COLLECTION_NAME", "docs")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b-instruct-q4_0")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embeddings CPU
emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)

# Qdrant init
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

try:
    qdrant.get_collection(COLLECTION)
except Exception:
    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=len(emb.embed_query("x")), distance=Distance.COSINE),
    )

# --------- Utils de parsing ---------

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)


def extract_text_from_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def extract_text_from_xlsx(path: str) -> str:
    # Concatena todas las hojas; para hojas grandes, genera filas como frases
    xl = pd.ExcelFile(path)
    pieces = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet).fillna("")
        # Limita columnas demasiado grandes
        df = df.astype(str)
        pieces.append(f"### Sheet: {sheet}\n" + df.to_csv(index=False))
    return "\n".join(pieces)


def docs_from_file(tmp_path: str, filename: str) -> List[str]:
    name = filename.lower()
    if name.endswith(".pdf"):
        text = extract_text_from_pdf(tmp_path)
    elif name.endswith(".docx") or name.endswith(".doc"):
        text = extract_text_from_docx(tmp_path)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        text = extract_text_from_xlsx(tmp_path)
    else:
        # Fallback: tratar como texto plano
        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)
    return chunks


def upsert_chunks(chunks: List[str], meta: dict):
    vectors = [emb.embed_query(c) for c in chunks]
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=v, payload={"text": c, **meta})
        for v, c in zip(vectors, chunks)
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)


class ChatRequest(BaseModel):
    q: str
    top_k: int = 4


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    chunks = docs_from_file(path, file.filename)
    upsert_chunks(chunks, {"filename": file.filename})
    os.remove(path)
    return {"status": "ok", "chunks": len(chunks)}


def retrieve(query: str, k: int):
    qvec = emb.embed_query(query)
    res = qdrant.search(collection_name=COLLECTION, query_vector=qvec, limit=k)
    return res


def call_ollama_chat(prompt: str) -> str:
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Eres un asistente que sólo usa el contexto proporcionado. Si falta información, dilo claramente."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # API de Ollama devuelve {message: {content: ...}}
    return data.get("message", {}).get("content", "")


@app.post("/chat")
async def chat(req: ChatRequest):
    hits = retrieve(req.q, req.top_k)
    ctx = "\n---\n".join(h.payload["text"] for h in hits)
    prompt = f"Contexto:\n{ctx}\n\nPregunta: {req.q}\nRespuesta concisa y con referencias a filename si procede."
    answer = call_ollama_chat(prompt)
    sources = [
        {"score": float(h.score), "filename": h.payload.get("filename", ""), "preview": h.payload["text"][:180]}
        for h in hits
    ]
    return {"answer": answer, "sources": sources}


@app.get("/health")
async def health():
    return {"ok": True}
