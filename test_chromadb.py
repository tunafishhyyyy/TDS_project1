# NOTE: This is just a test program and is not used in the main application.

import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("test")
collection.add(
    ids=["1"],
    embeddings=[[0.1]*512],
    documents=["test doc"],
    metadatas=[{"a": 1}]
)
print("Minimal insert succeeded!")