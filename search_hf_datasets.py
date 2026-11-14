"""
Query Hugging Face Datasets API via standard urllib
"""
import sys
import json
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def search_hf(query=None, sort="downloads", limit=15):
    if query:
        url = f"https://huggingface.co/api/datasets?search={query}&limit={limit}"
    else:
        url = f"https://huggingface.co/api/datasets?sort={sort}&direction=-1&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

print("=" * 85)
print(" 📚 TOP HUGGING FACE STORY & NOVEL DATASETS")
print("=" * 85)

categories = ["novel", "fiction", "stories", "creative-writing", "writingprompts", "gutenberg"]
seen = set()

for cat in categories:
    print(f"\n[Category: '{cat}']")
    data = search_hf(query=cat, limit=8)
    for d in data:
        did = d.get("id", "")
        if did in seen:
            continue
        seen.add(did)
        dl = d.get("downloads", 0) or 0
        likes = d.get("likes", 0) or 0
        print(f"  * {did:<48} | DLs: {dl:>9,} | Likes: {likes:>5}")

print("\n[Top Story Datasets by Likes]")
data = search_hf(query="story", sort="likes", limit=12)
for d in data:
    did = d.get("id", "")
    dl = d.get("downloads", 0) or 0
    likes = d.get("likes", 0) or 0
    print(f"  * {did:<48} | DLs: {dl:>9} | Likes: {likes:>5}")

print("\n" + "=" * 85)
