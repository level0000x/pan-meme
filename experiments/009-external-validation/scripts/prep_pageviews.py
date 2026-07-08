#!/usr/bin/env python3
"""预处理：将 pageviews JSON 转为纯文本（每行 = 一个月浏览量）"""
import json, os

PV_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pageviews")

for fname in os.listdir(PV_DIR):
    if not fname.endswith(".json"):
        continue
    concept = fname.replace(".json", "")
    with open(os.path.join(PV_DIR, fname)) as f:
        data = json.load(f)

    items = data.get("items", [])
    if not items:
        continue

    # 归一化到 [0,1]
    views = [it["views"] for it in items]
    max_v = max(views) if views else 1
    norm = [v / max_v for v in views]

    # 保存纯文本
    out_path = os.path.join(PV_DIR, f"{concept}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(f"{v:.6f}" for v in norm))

    # 同时保存原始值
    raw_path = os.path.join(PV_DIR, f"{concept}_raw.txt")
    with open(raw_path, "w") as f:
        f.write("\n".join(str(v) for v in views))

print(f"Done: {len(os.listdir(PV_DIR))} files processed")
