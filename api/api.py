# api.py
# Purpose: FastAPI backend for the TDS Virtual TA project. Provides a multimodal RAG (Retrieval-Augmented Generation) API for answering questions using course content and Discourse forum data.
# - Loads embeddings from GitHub, builds a FAISS index, and exposes endpoints for question answering.
# - Supports text and image queries, uses Jina AI for embeddings, and aipipe.org for LLM-based answer generation.
# - Includes CORS and request logging middleware for debugging and cross-origin support.

import os
import sys
import io
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import faiss
import json
import time
import requests
from fastapi.middleware.cors import CORSMiddleware

# Base URL for loading embeddings from GitHub
GITHUB_BASE_URL = "https://raw.githubusercontent.com/tunafishhyyyy/TDS_project1_static/refs/heads/main/"

# List of embedding JSON files to load
JSON_FILES = [
    "CourseContentData.json",
    "discourse_posts_part1.json",
    "discourse_posts_part2.json",
    "discourse_posts_part3.json",
    "discourse_posts_part4.json",
    "discourse_posts_part5.json",
    "discourse_posts_part6.json",
    "discourse_posts_part7.json",
    "discourse_posts_part8.json",
]

# --- Config ---
print("[INFO] Starting server initialization...", file=sys.stderr)

# API keys and endpoints for embedding and LLM services
JINA_API_KEY = "jina_70a5793453b54df79e9cac3be028b8d6oWwMsK6SCTd-3EFSjAZMgDRnZBPf"
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-clip-v2"
AIPIPE_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDE0OTlAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.pjDLSX8DwPmGkdOAQSSeHuPcM4M8XVjErw80zQumoVs"
AIPIPE_CHAT_URL = "https://aipipe.org/openrouter/v1/chat/completions"

# Headers for Jina API requests
headers = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json"
}

def get_text_embedding(text):
    """Get text embedding from Jina AI cloud API."""
    print(f"[INFO] Requesting text embedding (length={len(text)})...", file=sys.stderr)
    data = {
        "input": [{"text": text}],
        "model": JINA_MODEL
    }
    response = requests.post(JINA_API_URL, headers=headers, json=data)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(f"[ERROR] Jina API response: {response.text}", file=sys.stderr)
        raise
    print("[INFO] Received text embedding.", file=sys.stderr)
    return np.array(response.json()["data"][0]["embedding"], dtype=np.float32)

def get_image_embedding(image_b64):
    """Get image embedding from Jina AI cloud API using base64 image input."""
    print("[INFO] Requesting image embedding...", file=sys.stderr)
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode()
    data = {
        "input": [{"image": img_b64}],
        "model": JINA_MODEL
    }
    response = requests.post(JINA_API_URL, headers=headers, json=data)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(f"[ERROR] Jina API response: {response.text}", file=sys.stderr)
        raise
    print("[INFO] Received image embedding.", file=sys.stderr)
    return np.array(response.json()["data"][0]["embedding"], dtype=np.float32)

def call_aipipe_chat_api(question, context_docs):
    """Call the aipipe.org LLM API to generate an answer using the provided context."""
    print("[INFO] Calling aipipe.org LLM API for answer generation...", file=sys.stderr)
    context = "\n\n".join(context_docs)
    prompt = f"You are a helpful assistant for the IIT Madras TDS course. Use the following context to answer the user's question as accurately as possible.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for the IIT Madras TDS course."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(AIPIPE_CHAT_URL, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(f"[ERROR] aipipe.org API response: {response.text}", file=sys.stderr)
        raise
    data = response.json()
    print("[INFO] Received answer from LLM API.", file=sys.stderr)
    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"].strip()
    return "No answer generated."

# --- Load embeddings from GitHub ---
def load_json_from_github(filename):
    """Download and return JSON data from the specified GitHub file URL."""
    url = GITHUB_BASE_URL + filename
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Load all embeddings from GitHub into memory
embedding_data = {}
for fname in JSON_FILES:
    try:
        embedding_data[fname] = load_json_from_github(fname)
        print(f"[INFO] Loaded {fname} from GitHub.")
    except Exception as e:
        print(f"[WARN] Could not load {fname} from GitHub: {e}")

if not embedding_data:
    raise RuntimeError("No embeddings loaded!")

# --- Build all embeddings, documents, and metadata from loaded data ---
all_embeddings = []
all_documents = []
all_metadatas = []
start_embed = time.time()
for fname, data in embedding_data.items():
    # If the JSON has an 'embeddings' key, use it
    if isinstance(data, dict) and 'embeddings' in data:
        data = data['embeddings']
    for item in data:
        if 'embedding' in item:
            all_embeddings.append(item['embedding'])
            all_documents.append(item.get('text', item.get('document', '')))
            all_metadatas.append(item.get('metadata', {}))

if not all_embeddings:
    raise RuntimeError("No embeddings loaded!")

# Convert embeddings to numpy array and build FAISS index
embeddings_np = np.array(all_embeddings, dtype='float32')
faiss.normalize_L2(embeddings_np)
dim = embeddings_np.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings_np)
print(f"[INFO] Loaded {len(all_embeddings)} embeddings into FAISS index in {time.time() - start_embed:.2f} seconds.", file=sys.stderr)

print("[INFO] FastAPI app initialization starting...", file=sys.stderr)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins (adjust as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to log all incoming requests for debugging
@app.middleware("http")
async def log_requests(request, call_next):
    body = await request.body()
    print(f"[REQUEST LOG] {request.method} {request.url} | Body: {body.decode('utf-8', errors='replace')}", file=sys.stderr)
    response = await call_next(request)
    return response

# Request/response models for the API
class QueryWithImageRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 string

class Link(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[Link]

def get_query_embedding(text, image_b64=None):
    """Get the embedding for a query, optionally combining text and image embeddings."""
    txt_emb = get_text_embedding(text)
    if image_b64:
        try:
            img_emb = get_image_embedding(image_b64)
            emb = (txt_emb + img_emb) / 2
        except Exception as e:
            print(f"[WARN] Image embedding failed: {e}. Using text only.", file=sys.stderr)
            emb = txt_emb
    else:
        emb = txt_emb
    return emb.astype(np.float32)

# Main API endpoint for answering questions
@app.post("/api/", response_model=AnswerResponse)
async def query_with_image(request: QueryWithImageRequest):
    print(f"[INFO] Received API request: question='{request.question[:50]}...' image={'yes' if request.image else 'no'}", file=sys.stderr)
    try:
        query_emb = get_query_embedding(request.question, request.image)
        print("[INFO] Query embedding computed.", file=sys.stderr)
        query_emb = query_emb.reshape(1, -1)
        faiss.normalize_L2(query_emb)
        k = 5  # Retrieve top 5 documents instead of 3
        D, I = index.search(query_emb, k)
        print(f"[INFO] Top {k} documents retrieved from FAISS.", file=sys.stderr)
        docs = [all_documents[idx] for idx in I[0]]
        metas = [all_metadatas[idx] for idx in I[0]]
        answer = call_aipipe_chat_api(request.question, docs)
        print("[INFO] Answer generated and ready to return.", file=sys.stderr)
        links = []
        for i, meta in enumerate(metas):
            url = meta.get('original_url') or meta.get('url')
            text = docs[i]
            if url:
                links.append(Link(url=url, text=text))
        return AnswerResponse(answer=answer, links=links[:5])
    except Exception as e:
        print(f"[ERROR] Exception in /api/: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))

# Health check/root endpoint
@app.get("/")
def root():
    print("[INFO] Root endpoint hit.", file=sys.stderr)
    return {"message": "Multimodal RAG API is running!"}
