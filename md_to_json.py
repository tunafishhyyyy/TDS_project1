import os
import json
import re
import requests
import base64
from PIL import Image
from io import BytesIO

# --- Config ---
MD_DIR = "tds_pages_md"
OUTPUT_JSON = "CourseContentData.json"
JINA_API_KEY = "jina_70a5793453b54df79e9cac3be028b8d6oWwMsK6SCTd-3EFSjAZMgDRnZBPf"
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-clip-v2"  # Updated for multimodal

headers = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json"
}

def get_text_embedding(text):
    print(f"[INFO] Requesting text embedding (length={len(text)})...")
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
    print("[INFO] Received text embedding.")
    return response.json()["data"][0]["embedding"]

def get_image_embedding_from_path(image_path):
    print(f"[INFO] Requesting image embedding for: {image_path}")
    try:
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
        return get_image_embedding_from_bytes(img_bytes)
    except Exception as e:
        print(f"[WARN] Image embedding failed for {image_path}: {e}")
        return None

def get_image_embedding_from_url(url):
    print(f"[INFO] Downloading image from URL: {url}")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img_bytes = resp.content
        return get_image_embedding_from_bytes(img_bytes)
    except Exception as e:
        print(f"[WARN] Image download/embed failed for {url}: {e}")
        return None

def get_image_embedding_from_bytes(img_bytes):
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
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
        print(f"[INFO] Received image embedding.")
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"[WARN] Image embedding from bytes failed: {e}")
        return None

def parse_frontmatter(md_text):
    # Extract YAML frontmatter (between --- lines)
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', md_text, re.DOTALL)
    if not match:
        return {"title": None, "original_url": None, "downloaded_at": None}, md_text
    frontmatter, content = match.groups()
    meta = {}
    for line in frontmatter.splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            meta[key.strip()] = value.strip().strip('"')
    return meta, content.strip()

def extract_image_paths(md_text):
    # Find image paths in markdown: ![alt](path)
    return re.findall(r'!\[.*?\]\((.*?)\)', md_text)

def main():
    print(f"[INFO] Starting markdown to JSON embedding process.")
    entries = []
    md_files = [f for f in os.listdir(MD_DIR) if f.endswith('.md')]
    print(f"[INFO] Found {len(md_files)} markdown files in {MD_DIR}.")
    skipped_empty = 0
    for idx, fname in enumerate(sorted(md_files), 1):
        print(f"[INFO] Processing file {idx}/{len(md_files)}: {fname}")
        with open(os.path.join(MD_DIR, fname), encoding='utf-8') as f:
            md_text = f.read()
        meta, content = parse_frontmatter(md_text)
        if not content.strip():
            print(f"[SKIP] {fname}: empty content after frontmatter.")
            skipped_empty += 1
            continue
        try:
            text_emb = get_text_embedding(content)
        except Exception as e:
            print(f"[ERROR] {fname}: text embedding failed: {e}")
            continue
        # Try to embed the first image if present
        image_emb = None
        image_paths = extract_image_paths(content)
        if image_paths:
            print(f"[INFO] Found {len(image_paths)} image(s) in {fname}.")
        for img_path in image_paths:
            if img_path.startswith("http://") or img_path.startswith("https://"):
                image_emb = get_image_embedding_from_url(img_path)
                if image_emb:
                    break
            else:
                img_full_path = os.path.join(MD_DIR, img_path)
                if os.path.exists(img_full_path):
                    image_emb = get_image_embedding_from_path(img_full_path)
                    if image_emb:
                        break
                else:
                    print(f"[WARN] Image file not found: {img_full_path}")
        # Combine embeddings if both exist
        if image_emb:
            print(f"[INFO] Combining text and image embeddings for {fname}.")
            embedding = [(t + i) / 2 for t, i in zip(text_emb, image_emb)]
        else:
            embedding = text_emb
        entry = {
            "id": str(idx),
            "embedding": embedding,
            "text": content,
            "metadata": {
                "title": meta.get("title"),
                "original_url": meta.get("original_url"),
                "downloaded_at": meta.get("downloaded_at"),
                "filename": fname
            }
        }
        entries.append(entry)
        if len(entries) % 10 == 0:
            print(f"[INFO] Processed {len(entries)} valid files so far...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Done! Wrote {len(entries)} entries to {OUTPUT_JSON}")
    print(f"[INFO] Skipped {skipped_empty} files due to empty content.")

if __name__ == "__main__":
    main()
