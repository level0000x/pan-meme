"""Download additional Wikipedia articles using direct REST API."""
import requests
import os
import time
import json

OUTPUT_DIR = "experiments/010-missing-archetypes/data/fulltext"

new_articles = {
    "Innovation": [
        "IPhone", "Tesla,_Inc.", "Facebook", "Amazon_(company)", "Netflix",
        "Artificial_intelligence", "CRISPR", "SpaceX", "Bitcoin", "YouTube"
    ],
    "Conflict": [
        "World_War_I", "World_War_II", "Cold_War", "Vietnam_War",
        "American_Civil_War", "French_Revolution", "Russian_Revolution",
        "Napoleonic_Wars", "Crusades", "Peloponnesian_War"
    ],
    "Discovery": [
        "Theory_of_relativity", "Quantum_mechanics", "DNA", "Plate_tectonics",
        "Periodic_table", "Natural_selection", "Big_Bang", "Calculus",
        "Electromagnetism", "Germ_theory_of_disease"
    ],
    "Culture": [
        "Renaissance", "Industrial_Revolution", "Age_of_Enlightenment",
        "Romanticism", "Modernism", "Surrealism", "Impressionism",
        "Hip_hop_music", "Cinema_of_the_United_States", "Anime"
    ],
}

def get_page_text(title):
    """Get Wikipedia page text using REST API."""
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
    }
    try:
        resp = requests.get(url, params=params, timeout=30, 
                          headers={"User-Agent": "ResearchBot/1.0 (academic research)"})
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return None  # Page not found
            return page_data.get("extract", "")
    except Exception as e:
        print(f"    API error: {e}")
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_downloaded = 0
    total_chars = 0
    
    for archetype, titles in new_articles.items():
        print(f"\n=== {archetype} ===")
        for title in titles:
            safe_name = title.replace("/", "_").replace("\\", "_").replace(":", "_")
            save_path = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")
            
            if os.path.exists(save_path):
                size = os.path.getsize(save_path)
                print(f"  [EXISTS] {title} ({size} chars)")
                total_downloaded += 1
                total_chars += size
                continue
            
            print(f"  Downloading: {title}...", end=" ", flush=True)
            content = get_page_text(title)
            if content and len(content) > 100:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"OK ({len(content)} chars)")
                total_downloaded += 1
                total_chars += len(content)
            elif content:
                print(f"TOO SHORT ({len(content)} chars)")
            else:
                print("NOT FOUND")
            
            time.sleep(0.3)
    
    print(f"\n=== Summary ===")
    print(f"  New articles downloaded: {total_downloaded}")
    print(f"  Total characters: {total_chars}")
    all_txt = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
    print(f"  Total .txt files in directory: {len(all_txt)}")

if __name__ == "__main__":
    main()