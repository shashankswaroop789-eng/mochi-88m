"""
Find the best trending creative writing, fiction, and story datasets on Hugging Face
"""
import sys
import json
import urllib.request
import urllib.parse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

queries = ["story", "fiction", "novel", "creative writing", "narrative", "roleplay", "literature"]
results = {}

for q in queries:
    encoded_q = urllib.parse.quote(q)
    url = f"https://huggingface.co/api/datasets?search={encoded_q}&sort=trendingScore&direction=-1&limit=20"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            for d in data:
                did = d.get("id", "")
                if did not in results:
                    results[did] = {
                        "likes": d.get("likes", 0) or 0,
                        "downloads": d.get("downloads", 0) or 0,
                        "tag": q
                    }
    except Exception as e:
        print(f"Error fetching {q}: {e}")

# Sort by likes and downloads
sorted_res = sorted(results.items(), key=lambda x: (x[1]["likes"], x[1]["downloads"]), reverse=True)

print("=" * 95)
print(f" 🔥 TOP TRENDING STORY, FICTION & NOVEL DATASETS ON HUGGING FACE ({len(sorted_res)} found)")
print("=" * 95)
for i, (did, info) in enumerate(sorted_res[:35]):
    print(f"{i+1:2d}. {did:<52} | Likes: {info['likes']:<5} | Downloads: {info['downloads']:<8} | Tag: {info['tag']}")
print("=" * 95)
