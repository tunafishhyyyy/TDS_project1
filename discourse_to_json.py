import os
import json
import re
import torch
import numpy as np
from html import unescape
from bs4 import BeautifulSoup
import open_clip

# --- Config ---
DISCOURSE_DIR = "discourse_json"
OUTPUT_JSON = "discourse_posts.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-H-14', pretrained='laion2b_s32b_b79k'
)
tokenizer = open_clip.get_tokenizer('ViT-H-14')
model = model.to(device).eval()

def get_text_embedding(text):
    text_tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb.cpu().numpy().astype(np.float32)[0].tolist()

def html_to_text(html):
    # Remove HTML tags and unescape entities
    soup = BeautifulSoup(unescape(html), "html.parser")
    return soup.get_text(separator="\n").strip()

def main():
    entries = []
    json_files = [f for f in os.listdir(DISCOURSE_DIR) if f.endswith('.json')]
    idx = 1
    skipped_empty = 0
    for fname in sorted(json_files):
        with open(os.path.join(DISCOURSE_DIR, fname), encoding='utf-8') as f:
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
                skipped_empty += 1
                continue
            try:
                embedding = get_text_embedding(text)
            except Exception as e:
                print(f"[ERROR] {fname} post_id={post_id}: embedding failed: {e}")
                continue
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
            idx += 1
            if idx % 20 == 0:
                print(f"Processed {idx} posts...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"Done! Wrote {len(entries)} entries to {OUTPUT_JSON}")
    print(f"Skipped {skipped_empty} posts due to empty content.")

if __name__ == "__main__":
    main()
