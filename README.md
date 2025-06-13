# TDS_project1

## Overview

This project is an automated Q&A API for the IIT Madras Online Degree "Tools in Data Science" (TDS) course. It answers student questions using both course content and Discourse forum discussions, supporting true multimodal queries (text and optional image). The backend is built with FastAPI and uses in-memory vector search for fast retrieval.

## Approach

1. **Data Extraction**
   - Course content is scraped as Markdown files using Playwright and converted to JSON with embeddings.
   - Discourse forum posts are downloaded as JSON and converted to a unified format with embeddings.

2. **Embedding Generation**
   - All text and image data is embedded using the Jina AI `jina-clip-v2` cloud API, ensuring consistency between offline (data prep) and online (API query) embeddings.
   - If both text and image are present, their embeddings are averaged to form a multimodal representation.

3. **Vector Database**
   - Embeddings and metadata are stored in-memory using **FAISS** for fast similarity search. (Previously ChromaDB, now replaced due to performance issues on Windows.)
   - Embeddings are loaded from multiple partial files (e.g., `discourse_posts_part1.json`, `discourse_posts_part2.json`, etc.) and `CourseContentData.json` at startup. No manual merging is required.
   - In-memory storage is used for all embeddings. With 10 files of ~4 MB each (~40 MB total), this is efficient and serverless-friendly.

4. **API**
   - A FastAPI app exposes a `/api/` endpoint that accepts a question and optional base64 image, computes embeddings (using Jina AI), retrieves relevant content, and returns an answer with supporting links.

## Models Used

- **Jina AI jina-clip-v2 (cloud API)**
  - Used for both text and image embedding.
  - Ensures that all embeddings (offline and online) are compatible for similarity search.
  - No local model weights or GPU required; all embedding is done via API calls.

## Libraries Used

- fastapi
- uvicorn
- faiss-cpu  # (in-memory vector search)
- pillow      # (for image processing)
- numpy
- pydantic
- markdownify
- playwright
- requests    # (for Jina AI API calls)
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

- Scrapes course content to `tds_pages_md/` and converts to `CourseContentData.json` with multimodal embeddings.

### 3. Download and prepare Discourse forum data (with multiprocessing)

```powershell
python .\discourse_dld.py   # Requires your Discourse cookies for access
python .\discourse_to_json.py
```

- Downloads forum topics to `discourse_json/` and converts to multiple partial files (e.g., `discourse_posts_part1.json`, `discourse_posts_part2.json`, etc.) with multimodal embeddings. No need to merge these files manually.

### 4. Run the API locally

```powershell
uvicorn main:app --reload
```

- The API will be available at [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Test with Swagger UI at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Note:**

- **Server startup is fast** (no local model loading). All embeddings are loaded from JSON files into the in-memory FAISS index for fast search.
- **ChromaDB is no longer used.** All vector search is now handled by FAISS for maximum compatibility and speed, especially on Windows.
- **No local model weights or GPU required.** All embedding is done via the Jina AI API.
- **In-memory storage is serverless-friendly:** With ~40 MB of embeddings, this approach works well on Vercel and similar platforms (which typically allow 512 MB–1 GB RAM per function). Cold start times remain low at this scale.
- **Vercel/serverless note:** Vercel does not provide a built-in database, but you can use in-memory storage for small/medium datasets. For persistent or larger-scale storage, use an external managed database or vector DB.

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

- The same Jina AI model is used for both data prep and API queries.
- All embeddings are stored in-memory for fast search (current data size is safe for most modern machines and cloud/serverless platforms).
- **In-memory embedding decision:**
  - This decision was made after checking the file sizes:
    - 10 partial files × ~4 MB each = ~40 MB total
    - These sizes are reasonable for in-memory use on most modern machines and even on many cloud/serverless platforms (under 100 MB total).
  - If you deploy to Vercel or similar, check their memory limits (Vercel’s serverless functions typically allow 512 MB–1 GB RAM).
  - If you expect the data to grow much larger (hundreds of MBs or more), you may need to consider sharding, disk-based vector DB, or filtering before loading.
- `.gitignore` is set to exclude intermediate and large files except the main embedding JSONs.

---

**For any issues, check the debug output in the scripts for skipped or malformed files.**
