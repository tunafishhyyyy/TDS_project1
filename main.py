import io
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import torch
import open_clip
import chromadb
import json

# --- Config ---
COLLECTION_NAME = "tds_collection"

# --- Load CLIP Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model = model.to(device).eval()

# --- Load ChromaDB (in-memory) ---
client = chromadb.Client()  # In-memory only
collection = client.get_or_create_collection(COLLECTION_NAME)

# --- Load embeddings from JSON files ---
def load_embeddings_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both list of dicts and dict with 'embeddings' key
    if isinstance(data, dict) and 'embeddings' in data:
        data = data['embeddings']
    ids = [str(i) for i in range(len(data))]
    embeddings = [item['embedding'] for item in data if 'embedding' in item]
    documents = [item.get('text', item.get('document', '')) for item in data]
    metadatas = [item.get('metadata', {}) for item in data]
    return ids, embeddings, documents, metadatas

# Load all embeddings (add your JSON files here)
total_embedded = 0
for json_file in ["CourseContentData.json", "discourse_posts.json", "ts_book.json"]:
    try:
        ids, embeddings, documents, metadatas = load_embeddings_from_json(json_file)
        if embeddings:
            collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            total_embedded += len(embeddings)
    except Exception as e:
        pass  # Ignore missing files or bad formats
print(f"[INFO] Loaded {total_embedded} embeddings into ChromaDB collection '{COLLECTION_NAME}'")

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

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=3,
            include=["metadatas", "documents"]
        )
        # Compose answer from top document(s)
        docs = results['documents'][0]
        metas = results['metadatas'][0]
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
