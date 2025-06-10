'''connect to existing database by specifying the same directory
"import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(persist_directory="db/"))
# or
client = chromadb.PersistentClient(path="db/")"

or try this 
"from langchain_chroma import Chroma

# Set these to match your setup
COLLECTION_NAME = "your_collection_name"
PERSIST_DIRECTORY = "path/to/your/chroma_db"
EMBEDDING_FUNCTION = your_embedding_function  # e.g., OpenAIEmbeddings(...)

# Load the persisted Chroma vector database
vector_db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=EMBEDDING_FUNCTION,
    persist_directory=PERSIST_DIRECTORY,
)"


'''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import base64

app = FastAPI()

class QueryWithImageRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64-encoded image string

@app.post("/api/")
async def query_with_image(request: QueryWithImageRequest):
    # Access the question
    question = request.question

    # Optionally handle the image
    if request.image:
        try:
            image_bytes = base64.b64decode(request.image)
            # Now you can process image_bytes as needed (e.g., save, analyze, etc.)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
    else:
        image_bytes = None

    # Here, you would pass 'question' and 'image_bytes' to your RAG/LLM pipeline as needed
    # For demonstration, just echo back
    return {
        "received_question": question,
        "image_received": image_bytes is not None
    }
import io
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
import open_clip
import chromadb
from chromadb.config import Settings

# --- Config ---
COLLECTION_NAME = "your_collection_name"
PERSIST_DIRECTORY = "path/to/your/chroma_db"

# --- Load CLIP Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model = model.to(device).eval()

# --- Load ChromaDB ---
client = chromadb.Client(Settings(persist_directory=PERSIST_DIRECTORY))
collection = client.get_collection(COLLECTION_NAME)

# --- FastAPI Setup ---
app = FastAPI()

class QueryWithImageRequest(BaseModel):
    question: str
    image: str  # base64 string

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
    # Simple average; you can use other strategies
    return ((img_emb + txt_emb) / 2).astype(np.float32)

@app.post("/api/")
async def query_with_image(request: QueryWithImageRequest):
    try:
        # Decode image and get embeddings
        image_bytes = base64.b64decode(request.image)
        img_emb = get_image_embedding(image_bytes)
        txt_emb = get_text_embedding(request.question)
        query_emb = combine_embeddings(img_emb, txt_emb)

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=3,  # Top 3 results
            include=["metadatas", "documents"]
        )

        # Format results
        hits = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            hits.append({
                "document": doc,
                "metadata": meta
            })

        return {
            "results": hits
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Multimodal RAG API is running!"}
