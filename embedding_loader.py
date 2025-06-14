# NOTE: This is just a test program and is not used in the main application.
import chromadb
import json
# Load your embeddings from JSON
with open("embeddings.json", "r") as f:
    data = json.load(f)

# Assume your data is a list of dicts with keys: 'id', 'embedding', 'text', 'metadata'
ids = [item['id'] for item in data]
embeddings = [item['embedding'] for item in data]
documents = [item['text'] for item in data]
metadatas = [item.get('metadata', {}) for item in data]

client = chromadb.Client()
collection = client.get_or_create_collection("my_collection")

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas
)