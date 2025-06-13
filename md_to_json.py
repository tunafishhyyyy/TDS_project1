import os
import json
import re
import torch
import numpy as np
import open_clip

# --- Config ---
MD_DIR = "tds_pages_md"
OUTPUT_JSON = "CourseContentData.json"

# --- Load OpenCLIP Model (text only) ---
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

def main():
    entries = []
    md_files = [f for f in os.listdir(MD_DIR) if f.endswith('.md')]
    skipped_empty = 0
    for idx, fname in enumerate(sorted(md_files), 1):
        with open(os.path.join(MD_DIR, fname), encoding='utf-8') as f:
            md_text = f.read()
        meta, content = parse_frontmatter(md_text)
        if not content.strip():
            print(f"[SKIP] {fname}: empty content after frontmatter.")
            skipped_empty += 1
            continue
        try:
            embedding = get_text_embedding(content)
        except Exception as e:
            print(f"[ERROR] {fname}: embedding failed: {e}")
            continue
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
            print(f"Processed {len(entries)} valid files...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"Done! Wrote {len(entries)} entries to {OUTPUT_JSON}")
    print(f"Skipped {skipped_empty} files due to empty content.")

if __name__ == "__main__":
    main()
