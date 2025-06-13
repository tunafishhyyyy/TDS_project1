import os
import json
import re
import requests
from html import unescape
from bs4 import BeautifulSoup
import base64
from PIL import Image
from io import BytesIO
from multiprocessing import Process, cpu_count

# --- Config ---
DISCOURSE_DIR = "discourse_json"
OUTPUT_JSON = "discourse_posts.json"
JINA_API_KEY = "jina_00a607e7edbb45cf867f8650d18dab7alc_ecQksF4ub_6XAQ4ivawwZhnd_"
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-clip-v2"

headers = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json"
}

def get_text_embedding(text):
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
    return response.json()["data"][0]["embedding"]

def get_image_embedding_from_path(image_path):
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
        print("[INFO] Received image embedding.")
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"[WARN] Image embedding from bytes failed: {e}")
        return None

def html_to_text(html):
    soup = BeautifulSoup(unescape(html), "html.parser")
    return soup.get_text(separator="\n").strip()

def extract_image_paths_from_html(html):
    soup = BeautifulSoup(unescape(html), "html.parser")
    return [img['src'] for img in soup.find_all('img') if img.get('src')]

def process_json_files(file_group, group_idx, output_dir, output_prefix):
    entries = []
    for fname in file_group:
        with open(os.path.join(output_dir, fname), encoding='utf-8') as f:
            data = json.load(f)
        topic_id = data.get("id")
        topic_title = data.get("title")
        topic_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{data.get('slug')}/{topic_id}" if topic_id and data.get('slug') else None
        posts = data.get("post_stream", {}).get("posts", [])
        for post in posts:
            post_id = post.get("id")
            cooked = post.get("cooked", "")
            text = html_to_text(cooked)
            if not text.strip():
                print(f"[SKIP] {fname} post_id={post_id}: empty content after HTML to text.")
                continue
            try:
                text_emb = get_text_embedding(text)
            except Exception as e:
                print(f"[ERROR] {fname} post_id={post_id}: text embedding failed: {e}")
                continue
            # Try to embed the first image if present
            image_emb = None
            image_paths = extract_image_paths_from_html(cooked)
            for img_path in image_paths:
                if img_path.startswith("http://") or img_path.startswith("https://"):
                    image_emb = get_image_embedding_from_url(img_path)
                    if image_emb:
                        break
                else:
                    img_full_path = os.path.join(output_dir, img_path)
                    if os.path.exists(img_full_path):
                        image_emb = get_image_embedding_from_path(img_full_path)
                        if image_emb:
                            break
            # Combine embeddings if both exist
            if image_emb:
                embedding = [(t + i) / 2 for t, i in zip(text_emb, image_emb)]
            else:
                embedding = text_emb
            entry = {
                "id": f"{topic_id}_{post_id}",
                "embedding": embedding,
                "text": text,
                "metadata": {
                    "topic_id": topic_id,
                    "post_id": post_id,
                    "topic_title": topic_title,
                    "original_url": topic_url + f"/{post.get('post_number')}" if topic_url and post.get('post_number') else topic_url,
                    "created_at": post.get("created_at"),
                    "username": post.get("username"),
                    "name": post.get("name"),
                    "reply_to_post_number": post.get("reply_to_post_number"),
                    "filename": fname
                }
            }
            entries.append(entry)
    part_file = f"{output_prefix}_part{group_idx+1}.json"
    with open(part_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Group {group_idx+1}: Wrote {len(entries)} entries to {part_file}")

def main():
    json_files = [f for f in os.listdir(DISCOURSE_DIR) if f.endswith('.json')]
    n_proc = min(cpu_count(), 8)  # Use up to 8 processes by default
    chunk_size = (len(json_files) + n_proc - 1) // n_proc
    file_groups = [json_files[i*chunk_size:(i+1)*chunk_size] for i in range(n_proc)]
    procs = []
    for idx, group in enumerate(file_groups):
        if not group:
            continue
        p = Process(target=process_json_files, args=(group, idx, DISCOURSE_DIR, 'discourse_posts'))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("[INFO] All processes finished. Please merge the part files into one JSON if needed.")

if __name__ == "__main__":
    main()
