import os
import uuid
import tempfile
from pathlib import Path
from typing import Annotated, TypedDict, cast

import fitz
from docx import Document as DocxDocument
from google import genai
from google.genai import types
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pptx.api import Presentation
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import Points
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

_ = load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
QDRANT_URL       = os.getenv("QDRANT_URL")
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "researchlens")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K            = int(os.getenv("TOP_K_RESULTS", "5"))
EMBED_MODEL      = "gemini-embedding-001"
EMBED_DIM        = 3072  # fixed output dim for text-embedding-004
CHAT_MODEL = "gemini-2.5-flash"

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ── FastAPI setup ─────────────────────────────────────────────────────────────

app = FastAPI(title="ResearchLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your Lovable domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ensure Qdrant collection exists ───────────────────────────────────────────

def ensure_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        _ = qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    # Always ensure the index exists
    _ = qdrant.create_payload_index(
        COLLECTION_NAME, "document_id", PayloadSchemaType.KEYWORD
    )

ensure_collection()

# ── Helpers ───────────────────────────────────────────────────────────────────

class PageText(TypedDict):
    page: int
    text: str


class ChunkPayload(TypedDict):
    document_id: str
    filename: str
    page: int
    text: str


def _first_embedding_values(result: types.EmbedContentResponse) -> list[float]:
    if not result.embeddings or result.embeddings[0].values is None:
        raise HTTPException(status_code=502, detail="Embedding API returned no vector.")
    return result.embeddings[0].values


def _require_payload(payload: object | None) -> ChunkPayload:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Search hit missing payload.")
    return cast(ChunkPayload, cast(object, payload))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def extract_pdf(path: str) -> list[PageText]:
    """Return list of {page, text} dicts from a PDF."""
    doc = fitz.open(path)
    pages: list[PageText] = []
    for i in range(doc.page_count):
        text = doc[i].get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def extract_pptx(path: str) -> list[PageText]:
    """Return list of {page, text} dicts from a PPTX (page = slide number)."""
    prs = Presentation(path)
    slides: list[PageText] = []
    for i, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = " ".join(run.text for run in para.runs).strip()
                    if line:
                        texts.append(line)
        if texts:
            slides.append({"page": i, "text": " ".join(texts)})
    return slides

def extract_docx(path: str) -> list[PageText]:
    """Return list of {page, text} dicts from a DOCX file."""
    doc = DocxDocument(path)
    chunks, current, page = [], [], 1
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        current.append(text)
        # Treat every 10 paragraphs as a "page" for citation purposes
        if len(current) >= 10:
            chunks.append({"page": page, "text": " ".join(current)})
            page += 1
            current = []
    if current:
        chunks.append({"page": page, "text": " ".join(current)})
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        result = gemini_client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )
        embeddings.append(_first_embedding_values(result))
    return embeddings


def embed_query(text: str) -> list[float]:
    result = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    return _first_embedding_values(result)

# ── Pydantic models ────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int


class Citation(BaseModel):
    source: str
    page: int | None = None
    snippet: str
    score: float


class QueryRequest(BaseModel):
    document_id: str | None = None   # filter to one doc, or None = search all
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: str   # "high" | "medium" | "low"

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "ResearchLens API"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: Annotated[UploadFile, File()]):
    """
    Accept a PDF or PPTX file, extract text, chunk it,
    embed with Gemini, and store in Qdrant.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    filename = file.filename
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".pptx", ".ppt", ".docx"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Upload PDF or PPTX.",
        )

    document_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        _ = tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            pages = extract_pdf(tmp_path)
        elif suffix == ".docx":
            pages = extract_docx(tmp_path)
        else:
            pages = extract_pptx(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not pages:
        raise HTTPException(status_code=422, detail="No readable text found in file.")

    # Build chunks with metadata
    all_chunks: list[str] = []
    all_meta: list[ChunkPayload] = []
    for page_info in pages:
        for chunk in chunk_text(page_info["text"]):
            all_chunks.append(chunk)
            all_meta.append({
                "document_id": document_id,
                "filename": filename,
                "page": page_info["page"],
                "text": chunk,
            })

    if not all_chunks:
        raise HTTPException(status_code=422, detail="Could not extract any text chunks.")

    # Embed in batches of 100 (Gemini limit)
    batch_size = 10
    all_vectors = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        vecs = embed_texts(batch)
        all_vectors.extend(vecs)

    # Upsert into Qdrant
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload=cast(dict[str, object], cast(object, meta)),
        )
        for vec, meta in zip(all_vectors, all_meta)
    ]
    _ = qdrant.upsert(collection_name=COLLECTION_NAME, points=cast(Points, points))

    return IngestResponse(
        document_id=document_id,
        filename=filename,
        chunks_indexed=len(points),
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Embed the question, retrieve top-K chunks from Qdrant,
    and ask Gemini to answer with citations.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    query_vec = embed_query(req.question)

    # Optional filter: restrict to one document
    search_filter = None
    if req.document_id:
        search_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=req.document_id))]
        )

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=TOP_K,
        query_filter=search_filter,
        with_payload=True,
    )

    if not hits:
        return QueryResponse(
            answer="No relevant content found in the uploaded documents.",
            citations=[],
            confidence="low",
        )

    # Build context block for Gemini
    context_parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        p = _require_payload(hit.payload)
        loc = f"p.{p['page']}" if p.get("page") else "unknown page"
        context_parts.append(f"[{i}] {p['filename']} ({loc}):\n{p['text']}")
    context = "\n\n".join(context_parts)

    system_prompt = """You are ResearchLens, a precise research assistant.
Answer the user's question using ONLY the provided context excerpts.
Rules:
- Be concise and direct (2-4 sentences for simple questions, up to a short paragraph for complex ones).
- Cite sources inline using [1], [2] etc. matching the numbered excerpts.
- If the context does not contain enough information, say so clearly.
- Never fabricate facts or add information not present in the excerpts."""

    user_prompt = f"""Context excerpts:
{context}

Question: {req.question}

Answer (with inline citations):"""

    response = gemini_client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    answer_text = (response.text or "").strip()

    # Build citations list
    citations: list[Citation] = []
    for i, hit in enumerate(hits, start=1):
        p = _require_payload(hit.payload)
        # Only include citations that were actually referenced
        if f"[{i}]" in answer_text:
            text = str(p["text"])
            citations.append(Citation(
                source=p["filename"],
                page=p["page"],
                snippet=text[:200] + ("…" if len(text) > 200 else ""),
                score=round(hit.score, 3),
            ))

    # Confidence based on top hit score
    top_score = hits[0].score if hits else 0
    confidence = "high" if top_score > 0.82 else "medium" if top_score > 0.65 else "low"

    return QueryResponse(
        answer=answer_text,
        citations=citations,
        confidence=confidence,
    )


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """Remove all vectors for a given document from Qdrant."""
    _ = qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
    )
    return {"deleted": document_id}
