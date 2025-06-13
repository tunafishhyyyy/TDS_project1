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
import glob

# --- Config ---
print("[INFO] Starting server initialization...", file=sys.stderr)

JINA_API_KEY = "jina_70a5793453b54df79e9cac3be028b8d6oWwMsK6SCTd-3EFSjAZMgDRnZBPf"
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-clip-v2"
AIPIPE_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDE0OTlAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.pjDLSX8DwPmGkdOAQSSeHuPcM4M8XVjErw80zQumoVs"
AIPIPE_CHAT_URL = "https://aipipe.org/openrouter/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json"
}

def get_text_embedding(text):
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

# --- Load embeddings from JSON files ---
def load_embeddings_from_json(json_path):
    # Use absolute path relative to the api directory (same as this file)
    abs_path = os.path.join(os.path.dirname(__file__), json_path)
    print(f"[INFO] Loading embeddings from {abs_path} ...", file=sys.stderr)
    with open(abs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and 'embeddings' in data:
        data = data['embeddings']
    embeddings = [item['embedding'] for item in data if 'embedding' in item]
    documents = [item.get('text', item.get('document', '')) for item in data]
    metadatas = [item.get('metadata', {}) for item in data]
    print(f"[INFO] {len(embeddings)} embeddings loaded from {json_path}.", file=sys.stderr)
    return embeddings, documents, metadatas

# --- Load all embeddings and metadata ---
all_embeddings = []
all_documents = []
all_metadatas = []
start_embed = time.time()
# Load CourseContentData.json as before
for json_file in ["CourseContentData.json"]:
    try:
        embeddings, documents, metadatas = load_embeddings_from_json(json_file)
        if embeddings:
            all_embeddings.extend(embeddings)
            all_documents.extend(documents)
            all_metadatas.extend(metadatas)
    except Exception as e:
        print(f"[WARN] Could not load {json_file}: {e}", file=sys.stderr)
# Load all discourse_posts_part*.json files
for json_file in glob.glob(os.path.join(os.path.dirname(__file__), "discourse_posts_part*.json")):
    try:
        # Pass only the filename to loader (it will resolve to api dir)
        embeddings, documents, metadatas = load_embeddings_from_json(os.path.basename(json_file))
        if embeddings:
            all_embeddings.extend(embeddings)
            all_documents.extend(documents)
            all_metadatas.extend(metadatas)
    except Exception as e:
        print(f"[WARN] Could not load {json_file}: {e}", file=sys.stderr)

if not all_embeddings:
    raise RuntimeError("No embeddings loaded!")

embeddings_np = np.array(all_embeddings, dtype='float32')
faiss.normalize_L2(embeddings_np)
dim = embeddings_np.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings_np)
print(f"[INFO] Loaded {len(all_embeddings)} embeddings into FAISS index in {time.time() - start_embed:.2f} seconds.", file=sys.stderr)

print("[INFO] FastAPI app initialization starting...", file=sys.stderr)

app = FastAPI()

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

@app.post("/api/", response_model=AnswerResponse)
async def query_with_image(request: QueryWithImageRequest):
    print(f"[INFO] Received API request: question='{request.question[:50]}...' image={'yes' if request.image else 'no'}", file=sys.stderr)
    try:
        query_emb = get_query_embedding(request.question, request.image)
        print("[INFO] Query embedding computed.", file=sys.stderr)
        query_emb = query_emb.reshape(1, -1)
        faiss.normalize_L2(query_emb)
        k = 3
        D, I = index.search(query_emb, k)
        print(f"[INFO] Top {k} documents retrieved from FAISS.", file=sys.stderr)
        docs = [all_documents[idx] for idx in I[0]]
        metas = [all_metadatas[idx] for idx in I[0]]
        answer = call_aipipe_chat_api(request.question, docs)
        print("[INFO] Answer generated and ready to return.", file=sys.stderr)
        links = []
        for meta in metas:
            url = meta.get('original_url') or meta.get('url')
            text = meta.get('title') or meta.get('text') or "Related link"
            if url:
                links.append(Link(url=url, text=text))
        return AnswerResponse(answer=answer, links=links[:3])
    except Exception as e:
        print(f"[ERROR] Exception in /api/: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    print("[INFO] Root endpoint hit.", file=sys.stderr)
    return {"message": "Multimodal RAG API is running!"}
