# NOTE: This is just a test program and is not used in the main application.

import json
with open("discourse_posts.json", encoding="utf-8") as f:
    data = json.load(f)
print(set(len(item["embedding"]) for item in data))