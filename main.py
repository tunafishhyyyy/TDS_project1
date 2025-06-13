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
print("[INFO] Starting server initialization...")

JINA_API_KEY = "jina_70a5793453b54df79e9cac3be028b8d6oWwMsK6SCTd-3EFSjAZMgDRnZBPf"
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-clip-v2"

headers = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json"
}

def get_text_embedding(text):
    data = {
        "input": [{"text": text}],
        "model": JINA_MODEL
    }
    response = requests.post(JINA_API_URL, headers=headers, json=data)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(f"[ERROR] Jina API response: {response.text}")
        raise
    return np.array(response.json()["data"][0]["embedding"], dtype=np.float32)

def get_image_embedding(image_b64):
    # Decode base64 image
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Convert to bytes for API
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
        print(f"[ERROR] Jina API response: {response.text}")
        raise
    return np.array(response.json()["data"][0]["embedding"], dtype=np.float32)

# --- Load embeddings from JSON files ---
def load_embeddings_from_json(json_path):
    print(f"[INFO] Loading embeddings from {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and 'embeddings' in data:
        data = data['embeddings']
    embeddings = [item['embedding'] for item in data if 'embedding' in item]
    documents = [item.get('text', item.get('document', '')) for item in data]
    metadatas = [item.get('metadata', {}) for item in data]
    print(f"[INFO] {len(embeddings)} embeddings loaded from {json_path}.")
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
        print(f"[WARN] Could not load {json_file}: {e}")
# Load all discourse_posts_part*.json files
for json_file in glob.glob("discourse_posts_part*.json"):
    try:
        embeddings, documents, metadatas = load_embeddings_from_json(json_file)
        if embeddings:
            all_embeddings.extend(embeddings)
            all_documents.extend(documents)
            all_metadatas.extend(metadatas)
    except Exception as e:
        print(f"[WARN] Could not load {json_file}: {e}")

if not all_embeddings:
    raise RuntimeError("No embeddings loaded!")

embeddings_np = np.array(all_embeddings, dtype='float32')
# Normalize for cosine similarity
faiss.normalize_L2(embeddings_np)
dim = embeddings_np.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings_np)
print(f"[INFO] Loaded {len(all_embeddings)} embeddings into FAISS index in {time.time() - start_embed:.2f} seconds.")

print("[INFO] FastAPI app initialization starting...")

# --- FastAPI Setup ---
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
            # Combine text and image embeddings (average)
            emb = (txt_emb + img_emb) / 2
        except Exception as e:
            print(f"[WARN] Image embedding failed: {e}. Using text only.")
            emb = txt_emb
    else:
        emb = txt_emb
    return emb.astype(np.float32)

@app.post("/api/", response_model=AnswerResponse)
async def query_with_image(request: QueryWithImageRequest):
    try:
        query_emb = get_query_embedding(request.question, request.image)
        query_emb = query_emb.reshape(1, -1)
        faiss.normalize_L2(query_emb)
        k = 3
        D, I = index.search(query_emb, k)
        docs = [all_documents[idx] for idx in I[0]]
        metas = [all_metadatas[idx] for idx in I[0]]
        answer = docs[0][:400] if docs else "No relevant answer found."
        links = []
        for meta in metas:
            url = meta.get('original_url') or meta.get('url')
            text = meta.get('title') or meta.get('text') or "Related link"
            if url:
                links.append(Link(url=url, text=text))
        return AnswerResponse(answer=answer, links=links[:3])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Multimodal RAG API is running!"}
