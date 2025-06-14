# TDS_project1

## Overview

This project is an automated Q&A API for the IIT Madras Online Degree "Tools in Data Science" (TDS) course. It answers student questions using both course content and Discourse forum discussions, supporting true multimodal queries (text and optional image). The backend is built with FastAPI and uses in-memory vector search for fast retrieval.

## What Makes This Project Truly Multimodal

This project is truly multimodal because it natively supports both text and image data at every stage of the pipeline:

- **Data Ingestion:** Both course content and forum posts are processed for text and images. Images are extracted and included alongside text for embedding.
- **Embedding Generation:** The Jina AI `jina-clip-v2` model is used to generate embeddings for both text and images. If a data point contains both, their embeddings are averaged to create a single multimodal representation.
- **Query Handling:** The API endpoint accepts both a text question and an optional image (as base64). When both are provided, the system computes embeddings for each and combines them, ensuring the query itself is multimodal.
- **Retrieval:** All embeddings in the vector database are multimodal, so similarity search works seamlessly for text-only, image-only, or combined text+image queries.
- **Answer Generation:** The retrieved context (which may be text, images, or both) is used to generate answers, making the system robust to multimodal information needs.

This end-to-end multimodal support enables richer, more accurate retrieval and answer generation, going beyond traditional text-only Q&A systems.

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
    - In-memory storage is used for all embeddings. With 10 files of ~8 MB each (~80 MB total), this is efficient and serverless-friendly.
4. **API**
    - A FastAPI app exposes a `/api/` endpoint that accepts a question and optional base64 image, computes embeddings (using Jina AI), retrieves relevant content, and returns an answer with supporting links.

## Models Used

- **Jina AI jina-clip-v2 (cloud API)**
  - Used for both text and image embedding.
  - Ensures that all embeddings (offline and online) are compatible for similarity search.
  - No local model weights or GPU required; all embedding is done via API calls.

## Libraries Used

**Production (API):**

- fastapi
- faiss-cpu  # (in-memory vector search)
- pillow      # (for image processing)
- numpy
- pydantic
- requests    # (for Jina AI API calls)

**Development only:**

- uvicorn         # (for local dev server)
- markdownify     # (for markdown to text conversion)
- playwright      # (for scraping course content)
- beautifulsoup4  # (for HTML to text in Discourse posts)

## Data Pipeline & Commands

### 1. Install dependencies

**For production (deployment):**

```powershell
pip install -r requirements.txt
```

**For development (local dev, scraping, etc):**

```powershell
pip install -r requirements-dev.txt
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

**Note:** To run `discourse_dld.py`, you must add your Discourse session cookie for authentication. This is required to access and scrape posts from the Discourse course page.

### 4. Run the API locally

```powershell
uvicorn main:app --reload
```

- The API will be available at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) when running locally.
- When deployed, the API will be available at your Vercel deployment URL, e.g. `https://your-vercel-project.vercel.app/api/` (replace with your actual Vercel URL).

**Note:**

- **Server startup is fast** (no local model loading). All embeddings are loaded from JSON files into the in-memory FAISS index for fast search.
- **No local model weights or GPU required.** All embedding is done via the Jina AI API.
- **In-memory storage is serverless-friendly:** With ~80 MB of embeddings, this approach works well on Vercel and similar platforms (which typically allow 512 MB–1 GB RAM per function). Cold start times remain low at this scale.
- **Vercel/serverless note:** Vercel does not provide a built-in database, but you can use in-memory storage for small/medium datasets. For persistent or larger-scale storage, use an external managed database or vector DB.

## API Usage

- **POST /api/**
  - Request: `{ "question": "...", "image": "<base64>" (optional) }`
  - Response: `{ "answer": "...", "links": [ {"url": "...", "text": "..."}, ... ] }`

## Data Format Example

This is the format of the embedding files generated by `discourse_to_json.py` and `md_to_json.py`:

```json
{
  "id": "1",
  "embedding": [0.123, 0.456, ...],
  "text": "Some content",
  "metadata": {"original_url": "https://..."
}
```

## How Answer Generation Works (RAG with GPT-4o-mini via aipipe.org)

When a user sends a question (and optional image) to the `/api/` endpoint, the system:

1. Computes a multimodal embedding for the query using Jina AI's `jina-clip-v2`.
2. Searches the in-memory FAISS index for the most relevant course and forum content.
3. Selects the top 5 most relevant text chunks as "context" for answer generation.
4. Sends a request to the [aipipe.org](https://aipipe.org/) API, using the `gpt-4o-mini` model, with a prompt that includes:
    - The user's question
    - The retrieved context (top 5 relevant text chunks)
5. The LLM (GPT-4o-mini) generates a synthesized answer using both the question and the provided context.
6. The API returns the generated answer and supporting links to the user.

**API call details:**

The backend sends a POST request to `https://api.aipipe.org/v1/chat/completions` with a payload like:

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant for the IIT Madras TDS course."},
    {"role": "user", "content": "<prompt with context and question>"}
  ]
}
```

- The context is a concatenation of the top 5 retrieved documents, followed by the user's question.
- The API key is sent in the `Authorization` header.

**This approach enables Retrieval-Augmented Generation (RAG):**

- The answer is not just a snippet from the database, but a synthesized response generated by GPT-4o-mini, grounded in the most relevant course/forum content.
- This ensures accurate, context-aware answers for user queries.

## Environment Variables

This project uses environment variables to securely manage API keys and authentication tokens. Before running the API or data preparation scripts, set the following environment variables:

- `JINA_API_KEY`: Your Jina AI API key (required for embedding generation)
- `AIPIPE_API_KEY`: Your AIPipe API key (required for answer generation and evaluation)
- `DISCOURSE_COOKIE`: Your Discourse session cookie (required for scraping Discourse posts)

You can set these variables in your shell, CI environment, or in a `.env` file (if using a tool like `python-dotenv`).

Example for Windows PowerShell:

```powershell
$env:JINA_API_KEY = "your-jina-key"
$env:AIPIPE_API_KEY = "your-aipipe-key"
$env:DISCOURSE_COOKIE = "your-discourse-cookie"
```

Example for Unix/macOS:

```bash
export JINA_API_KEY="your-jina-key"
export AIPIPE_API_KEY="your-aipipe-key"
export DISCOURSE_COOKIE="your-discourse-cookie"
```

## Discourse Cookie Handling (Full Browser Cookie Support)

### How to get and use your Discourse cookies for scraping

Discourse authentication may require more than just the `_forum_session` cookie. To ensure maximum compatibility, this project supports using the full browser cookie string for authenticated requests.

#### Recommended workflow:

1. Run `python get_discourse_cookie.py`. This will open a browser window.
2. Log in to Discourse with Google in the browser window. Complete any 2FA or CAPTCHA if prompted.
3. After you are fully logged in and see the Discourse forum page, return to the terminal and press Enter.
4. The script will extract **all cookies** from your browser session and write them to your `.env` file as a single line:
   ```
   DISCOURSE_COOKIE=gcl_au=...; _ga=...; ...; _forum_session=...; ...
   ```
5. The `discourse_dld.py` script will automatically read this value from `.env`, parse all cookies, and use them for requests. This mimics browser behavior and avoids 403 errors.

- You do **not** need to manually copy-paste cookies from your browser; the script automates this.
- If you ever get a 403 error, repeat the above process to refresh your cookies.
- If you want to use a different account, log out in the browser window before running the script.

**Note:**
- The cookie string should be in the exact format as seen in your browser's request headers (e.g., `name1=value1; name2=value2; ...`).
- The script will parse and use all cookies, not just `_forum_session`.
- This approach is robust to SSO, CSRF, and other authentication mechanisms that require multiple cookies.

## Notes

- The same Jina AI model is used for both data prep and API queries.
- All embeddings are stored in-memory for fast search (current data size is safe for most modern machines and cloud/serverless platforms).
- **In-memory embedding decision:**
  - This decision was made after checking the file sizes:
    - 10 partial files × ~8 MB each = ~80 MB total
    - These sizes are reasonable for in-memory use on most modern machines and even on many cloud/serverless platforms (under 100 MB total).
  - If you deploy to Vercel or similar, check their memory limits (Vercel’s serverless functions typically allow 512 MB–1 GB RAM).
  - If you expect the data to grow much larger (hundreds of MBs or more), you may need to consider sharding, disk-based vector DB, or filtering before loading.
- `.gitignore` is set to exclude intermediate and large files except the main embedding JSONs.

## Embeddings Storage Note

**Important:** Due to Vercel's deployment size restrictions, all embedding files (e.g., `CourseContentData.json`, `discourse_posts_part*.json`) have been moved to a separate public GitHub repository: [TDS_project1_static](https://github.com/tunafishhyyyy/TDS_project1_static). The API dynamically loads these embeddings at runtime from the GitHub raw URLs, so no large embedding files are stored in this project repository. This allows for successful deployment on Vercel and keeps the main repo lightweight.

---

**For any issues, check the debug output in the scripts for skipped or malformed files.**

## Evaluation results

```
headers: {"content-type":"application/json"}
Group 1/4 [████████████████████████████████████████] 100% | 1/1 | <https://tds-project1-umber.vercel.app/a>
Group 2/4 [████████████████████████████████████████] 100% | 1/1 | <https://tds-project1-umber.vercel.app/a>
Group 3/4 [████████████████████████████████████████] 100% | 1/1 | <https://tds-project1-umber.vercel.app/a>
Group 4/4 [████████████████████████████████████████] 100% | 1/1 | <https://tds-project1-umber.vercel.app/a>

┌────────────────────────┬────────────────────────┬────────────────────────┬────────────────────────┐
│ image                  │ link                   │ question               │ [https://tds-project1… │
│                        │                        │                        │ {{prompt}}             │
├────────────────────────┼────────────────────────┼────────────────────────┼────────────────────────┤
│ UklGRpgWAABXRUJQVlA4I… │ https://discourse.onl… │ The question asks to   │ [FAIL] Expected output │
│                        │                        │ use gpt-3.5-turbo-0125 │ to contain             │
│                        │                        │ model but the ai-proxy │ "https://discourse.on… │
│                        │                        │ provided by Anand sir  │ ---                    │
│                        │                        │ only supports          │ {"answer":"If the AI   │
│                        │                        │ gpt-4o-mini. So should │ proxy provided by      │
│                        │                        │ we just use            │ Anand sir only         │
│                        │                        │ gpt-4o-mini or use the │ supports the           │
│                        │                        │ OpenAI API for gpt3.5  │ gpt-4o-mini model,     │
│                        │                        │ turbo?                 │ then you should use    │
│                        │                        │                        │ that model for your    │
│                        │                        │                        │ tasks. H...            │
├────────────────────────┼────────────────────────┼────────────────────────┼────────────────────────┤
│                        │ https://discourse.onl… │ If a student scores    │ [PASS] {"answer":"If a │
│                        │                        │ 10/10 on GA4 as well   │ student scores 10/10   │
│                        │                        │ as a bonus, how would  │ on GA4 and is on the   │
│                        │                        │ it appear on the       │ bonus list, the        │
│                        │                        │ dashboard?             │ dashboard would        │
│                        │                        │                        │ reflect their score as │
│                        │                        │                        │ 11/10. The bonus mark  │
│                        │                        │                        │ is added to the        │
│                        │                        │                        │ original score, so     │
│                        │                        │                        │ students who qualify   │
│                        │                        │                        │ will see this          │
│                        │                        │                        │ adjustment in their    │
│                        │                        │                        │ overall                │
│                        │                        │                        │ score.","links...      │
├────────────────────────┼────────────────────────┼────────────────────────┼────────────────────────┤
│                        │                        │ I know Docker but have │ [FAIL] Expected output │
│                        │                        │ not used Podman        │ to contain             │
│                        │                        │ before. Should I use   │ "https://tds.s-anand.… │
│                        │                        │ Docker for this        │ ---                    │
│                        │                        │ course?                │ {"answer":"Yes, you    │
│                        │                        │                        │ can definitely use     │
│                        │                        │                        │ Docker for this        │
│                        │                        │                        │ course, and it will    │
│                        │                        │                        │ work just fine for     │
│                        │                        │                        │ your needs. Although   │
│                        │                        │                        │ Podman can be used in  │
│                        │                        │                        │ place of Docker, if    │
│                        │                        │                        │ you are more familiar  │
│                        │                        │                        │ wi...                  │
├────────────────────────┼────────────────────────┼────────────────────────┼────────────────────────┤
│                        │                        │ When is the TDS Sep    │ [PASS] {"answer":"The  │
│                        │                        │ 2025 end-term exam?    │ specific date for the  │
│                        │                        │                        │ TDS September 2025     │
│                        │                        │                        │ end-term exam has not  │
│                        │                        │                        │ been provided in the   │
│                        │                        │                        │ given context. You may │
│                        │                        │                        │ need to check the TDS  │
│                        │                        │                        │ course page or any     │
│                        │                        │                        │ official announcements │
│                        │                        │                        │ for the exact date of  │
│                        │                        │                        │ the                    │
│                        │                        │                        │ exam.","links":[{"url… │
└────────────────────────┴────────────────────────┴────────────────────────┴────────────────────────┘
================================================================================================

✔ Evaluation complete. ID: eval-3WL-2025-06-14T00:21:29

» Run promptfoo view to use the local web viewer
» Do you want to share this with your team? Sign up for free at <https://promptfoo.app>
» This project needs your feedback. What's one thing we can improve? <https://forms.gle/YFLgTe1dKJKNSCsU7>
================================================================================================

» Run promptfoo view to use the local web viewer
» Do you want to share this with your team? Sign up for free at <https://promptfoo.app>
» This project needs your feedback. What's one thing we can improve? <https://forms.gle/YFLgTe1dKJKNSCsU7>
» Run promptfoo view to use the local web viewer
» Run promptfoo view to use the local web viewer
» Do you want to share this with your team? Sign up for free at <https://promptfoo.app>
» This project needs your feedback. What's one thing we can improve? <https://forms.gle/YFLgTe1dKJKNSCsU7>
================================================================================================

Successes: 2
Failures: 2
Errors: 0
» Run promptfoo view to use the local web viewer
» Do you want to share this with your team? Sign up for free at <https://promptfoo.app>
» This project needs your feedback. What's one thing we can improve? <https://forms.gle/YFLgTe1dKJKNSCsU7>
================================================================================================

Successes: 2
Failures: 2
Errors: 0
Pass Rate: 50.00%
Duration: 2s (concurrency: 4)

Done.
