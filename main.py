import io
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import torch
import faiss
import json
import time
import open_clip

# --- Config ---
print("[INFO] Starting server initialization...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-H-14', pretrained='laion2b_s32b_b79k'
)
tokenizer = open_clip.get_tokenizer('ViT-H-14')
model = model.to(device).eval()

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
for json_file in ["CourseContentData.json", "discourse_posts.json"]:
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

def get_image_embedding(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_emb = model.encode_image(image_tensor)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
    return image_emb.cpu().numpy().astype(np.float32)[0]

def get_text_embedding(text):
    text_tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb.cpu().numpy().astype(np.float32)[0]

def combine_embeddings(img_emb, txt_emb):
    if img_emb is not None and txt_emb is not None:
        return ((img_emb + txt_emb) / 2).astype(np.float32)
    elif img_emb is not None:
        return img_emb.astype(np.float32)
    else:
        return txt_emb.astype(np.float32)

@app.post("/api/", response_model=AnswerResponse)
async def query_with_image(request: QueryWithImageRequest):
    try:
        img_emb = None
        if request.image:
            try:
                image_bytes = base64.b64decode(request.image)
                img_emb = get_image_embedding(image_bytes)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid base64 image data")
        txt_emb = get_text_embedding(request.question)
        query_emb = combine_embeddings(img_emb, txt_emb)
        # Normalize query embedding for cosine similarity
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
