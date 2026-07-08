#!/usr/bin/env python3
"""修复短摘要：用完整页面文本替代 exintro"""
import json, urllib.request, urllib.parse, time, os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EX_DIR = os.path.join(OUT_DIR, "extracts")
PV_DIR = os.path.join(OUT_DIR, "pageviews")

SHORT_CONCEPTS = ["Beethoven", "Flash_Player", "NFT", "Queen_Elizabeth_II",
                   "Shakespeare", "Trump", "Yahoo!"]

def fetch_full(query):
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": "1",
        "exlimit": "1", "titles": query, "format": "json"
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "pan-meme-experiment/0.1"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(2 ** (attempt + 2), 30)
                print(f"  rate limit, wait {wait}s")
                time.sleep(wait)
            else:
                raise
        except Exception:
            time.sleep(3)
    return None

for concept in SHORT_CONCEPTS:
    name = concept.replace("%2B", "+").replace("%27", "'").replace("_", " ")
    print(f"Fixing: {concept} -> '{name}'")
    data = fetch_full(name)
    if data:
        pages = data.get("query", {}).get("pages", {})
        extract = list(pages.values())[0].get("extract", "")
        if len(extract) >= 100:
            with open(os.path.join(EX_DIR, f"{concept}.json"), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ {concept}: {len(extract)} chars")
        else:
            print(f"  ✗ {concept}: still only {len(extract)} chars")
    else:
        print(f"  ✗ {concept}: fetch failed")
    time.sleep(2)

print("\nDone.")
