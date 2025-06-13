import numpy as np
import faiss
import json

# Load your embeddings and metadata from JSON (example: CourseContentData.json)
with open("CourseContentData.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare embeddings and metadata
embeddings = np.array([item['embedding'] for item in data], dtype='float32')
texts = [item['text'] for item in data]
metadatas = [item['metadata'] for item in data]

# Build FAISS index (L2/Euclidean or cosine similarity)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # For cosine, normalize first
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Example query (replace with your embedding function)
def get_query_embedding(text):
    # Dummy: use the first embedding as a stand-in
    # Replace with your OpenCLIP embedding function
    return embeddings[0]

query_text = "What is the grading policy?"
query_emb = get_query_embedding(query_text).reshape(1, -1)
faiss.normalize_L2(query_emb)

# Search
k = 3  # top 3
D, I = index.search(query_emb, k)

print("Top results:")
for rank, idx in enumerate(I[0]):
    print(f"Rank {rank+1} | Score: {D[0][rank]:.4f}")
    print(f"Text: {texts[idx][:200]}")
    print(f"Metadata: {metadatas[idx]}")
    print()
