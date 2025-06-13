# TDS_project1

## Overview

This project is an automated Q&A API for the IIT Madras Online Degree "Tools in Data Science" (TDS) course. It answers student questions using both course content and Discourse forum discussions, supporting multimodal queries (text and optional image). The backend is built with FastAPI and uses in-memory vector search for fast retrieval.

## Approach

1. **Data Extraction**
   - Course content is scraped as Markdown files using Playwright and converted to JSON with embeddings.
   - Discourse forum posts are downloaded as JSON and converted to a unified format with embeddings.

2. **Embedding Generation**
   - All text data is embedded using the OpenCLIP ViT-B-32 model with OpenAI weights, ensuring consistency between offline (data prep) and online (API query) embeddings.

3. **Vector Database**
   - Embeddings and metadata are stored in-memory using **FAISS** for fast similarity search. (Previously ChromaDB, now replaced due to performance issues on Windows.)

4. **API**
   - A FastAPI app exposes a `/api/` endpoint that accepts a question and optional base64 image, computes embeddings, retrieves relevant content, and returns an answer with supporting links.

## Models Used

- **OpenCLIP ViT-B-32 (pretrained='openai')**
  - Used for both text and image embedding.
  - Ensures that all embeddings (offline and online) are compatible for similarity search.

## Libraries Used

- fastapi
- uvicorn
- faiss-cpu  # (in-memory vector search)
- open-clip-torch
- torch
- pillow
- numpy
- pydantic
- markdownify
- playwright
- beautifulsoup4 (for HTML to text in Discourse posts)

## Data Pipeline & Commands

### 1. Install dependencies

```powershell
pip install -r requirements.txt
playwright install
```

### 2. Scrape and prepare course content

```powershell
python .\course_content_dld.py
python .\md_to_json.py
```

- Scrapes course content to `tds_pages_md/` and converts to `CourseContentData.json` with embeddings.

### 3. Download and prepare Discourse forum data

```powershell
python .\discourse_dld.py   # Requires your Discourse cookies for access
python .\discourse_to_json.py
```

- Downloads forum topics to `discourse_json/` and converts to `discourse_posts.json` with embeddings.

### 4. Run the API locally

```powershell
uvicorn main:app --reload
```

- The API will be available at [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Test with Swagger UI at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Note:**

- **Server startup may take some time** (up to a few minutes on first run or on resource-constrained machines). This is because the OpenCLIP ViT-B-32 model weights must be loaded into memory and all embeddings from the JSON files are loaded into the in-memory FAISS index for fast search.
- **ChromaDB is no longer used.** All vector search is now handled by FAISS for maximum compatibility and speed, especially on Windows.

## API Usage

- **POST /api/**
  - Request: `{ "question": "...", "image": "<base64>" (optional) }`
  - Response: `{ "answer": "...", "links": [ {"url": "...", "text": "..."}, ... ] }`

## Data Format Example

```json
{
  "id": "1",
  "embedding": [0.123, 0.456, ...],
  "text": "Some content",
  "metadata": {"original_url": "https://..."}
}
```

## Notes

- The same OpenCLIP model is used for both data prep and API queries.
- All embeddings are stored in-memory for fast search (current data size is safe for most modern machines and cloud functions).
- **In-memory embedding decision:**
  - This decision was made after checking the file sizes:
    - `CourseContentData.json`: ~1.96 MB
    - `discourse_posts.json`: ~57.1 MB
  - These sizes are reasonable for in-memory use on most modern machines and even on many cloud/serverless platforms (under 100 MB total).
  - If you deploy to Vercel or similar, check their memory limits (Vercel’s serverless functions typically allow 512 MB–1 GB RAM).
  - If you expect the data to grow much larger (hundreds of MBs or more), you may need to consider sharding, disk-based vector DB, or filtering before loading.
- `.gitignore` is set to exclude intermediate and large files except the main embedding JSONs.

---

**For any issues, check the debug output in the scripts for skipped or malformed files.**
