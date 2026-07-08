#!/usr/bin/env python3
"""实验 009: 采集 Wikipedia 月浏览量 + 页面摘要（带重试）"""
import json, urllib.request, urllib.parse, time, os

CONCEPTS = {
    "Stable": [
        "Newton", "Darwin", "Democracy", "Gravity", "Evolution",
        "Oxygen", "Shakespeare", "DNA", "Photosynthesis", "Calculus",
        "Periodic_table", "Beethoven", "Philosophy", "Algebra", "Carbon"
    ],
    "Burst": [
        "ChatGPT", "COVID-19", "Bitcoin", "NFT", "Elon_Musk",
        "Taylor_Swift", "Ukraine", "Barbie_(film)", "Oppenheimer_(film)",
        "Queen_Elizabeth_II", "Black_Lives_Matter", "Brexit",
        "GameStop_short_squeeze", "Trump", "Tesla,_Inc."
    ],
    "Decay": [
        "Adobe_Flash", "MySpace", "BlackBerry", "Internet_Explorer",
        "Nokia", "Yahoo!", "Windows_XP", "LimeWire", "Vine_(service)",
        "Google%2B", "Second_Life", "FarmVille", "Clubhouse_(app)",
        "Flash_Player", "Myspace"
    ],
    "Oscillatory": [
        "Christmas", "Thanksgiving", "Halloween", "Super_Bowl",
        "Olympic_Games", "FIFA_World_Cup", "Academy_Awards",
        "Grammy_Awards", "Eurovision_Song_Contest", "Black_Friday_(shopping)",
        "Easter", "Valentine%27s_Day", "Ramadan", "Diwali",
        "New_Year%27s_Eve"
    ],
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def fetch_with_retry(url, max_retries=5):
    """带指数退避的 HTTP GET"""
    req = urllib.request.Request(url, headers={"User-Agent": "pan-meme-experiment/0.1"})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 429:
                    wait = min(2 ** (attempt + 2), 60)
                    time.sleep(wait)
                    continue
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(2 ** (attempt + 2), 60)
                print(f"    rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif e.code == 404:
                return None  # 页面不存在
            else:
                raise
        except Exception:
            time.sleep(2)
    return None

def main():
    # 收集已有的
    pv_done = set(f.replace(".json","") for f in os.listdir(os.path.join(OUT_DIR, "pageviews")) if f.endswith(".json"))
    ex_done = set(f.replace(".json","") for f in os.listdir(os.path.join(OUT_DIR, "extracts")) if f.endswith(".json"))
    total = sum(len(v) for v in CONCEPTS.values())
    done_pv, done_ex, fail_pv, fail_ex = len(pv_done), len(ex_done), 0, 0

    for etype, concepts in CONCEPTS.items():
        for concept in concepts:
            # 浏览量
            if concept not in pv_done:
                url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/{concept}/monthly/2015070100/2025070100"
                data = fetch_with_retry(url)
                if data and "items" in data:
                    with open(os.path.join(OUT_DIR, "pageviews", f"{concept}.json"), "w") as f:
                        json.dump(data, f, indent=2)
                    print(f"  ✓ pv  {concept}: {len(data['items'])} months")
                    done_pv += 1
                elif data is None and "404" in str(data if data else ""):
                    print(f"  - pv  {concept}: 404 (skip)")
                else:
                    print(f"  ✗ pv  {concept}: failed after retries")
                    fail_pv += 1
                time.sleep(1.5)  # Wikimedia 建议 ≥1s 间隔

            # 摘要
            if concept not in ex_done:
                params = urllib.parse.urlencode({
                    "action": "query", "prop": "extracts", "exintro": "1",
                    "explaintext": "1",
                    "titles": concept.replace("%2B", "+").replace("%27", "'"),
                    "format": "json"
                })
                url = f"https://en.wikipedia.org/w/api.php?{params}"
                data = fetch_with_retry(url)
                if data:
                    pages = data.get("query", {}).get("pages", {})
                    extract = list(pages.values())[0].get("extract", "")
                    with open(os.path.join(OUT_DIR, "extracts", f"{concept}.json"), "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    n = len(extract)
                    ok = "✓" if n >= 100 else "⚠"
                    print(f"  {ok} ex  {concept}: {n} chars")
                    done_ex += 1
                else:
                    print(f"  - ex  {concept}: failed/404 (skip)")
                time.sleep(1.5)

    print(f"\n===== {done_pv} pv + {done_ex} ex | {fail_pv} pv fail =====")

    # 保存元数据
    with open(os.path.join(OUT_DIR, "concepts.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in CONCEPTS.items()}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
