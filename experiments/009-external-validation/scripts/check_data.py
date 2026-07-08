#!/usr/bin/env python3
"""数据质量检查：筛选有效概念，准备验证输入"""
import json, os, csv
import numpy as np  # 需要 pip install numpy

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EX_DIR = os.path.join(OUT_DIR, "extracts")
PV_DIR = os.path.join(OUT_DIR, "pageviews")

# 分类映射
TYPE_MAP = {
    "Stable": ["Newton","Darwin","Democracy","Gravity","Evolution","Oxygen","Shakespeare",
                "DNA","Photosynthesis","Calculus","Periodic_table","Beethoven",
                "Philosophy","Algebra","Carbon"],
    "Burst": ["ChatGPT","COVID-19","Bitcoin","NFT","Elon_Musk","Taylor_Swift","Ukraine",
               "Barbie_(film)","Oppenheimer_(film)","Queen_Elizabeth_II","Black_Lives_Matter",
               "Brexit","GameStop_short_squeeze","Trump","Tesla,_Inc."],
    "Decay": ["Adobe_Flash","MySpace","BlackBerry","Internet_Explorer","Nokia","Yahoo!",
               "Windows_XP","LimeWire","Vine_(service)","Google%2B","Second_Life",
               "FarmVille","Clubhouse_(app)","Flash_Player","Myspace"],
    "Oscillatory": ["Christmas","Thanksgiving","Halloween","Super_Bowl","Olympic_Games",
                     "FIFA_World_Cup","Academy_Awards","Grammy_Awards","Eurovision_Song_Contest",
                     "Black_Friday_(shopping)","Easter","Valentine%27s_Day","Ramadan",
                     "Diwali","New_Year%27s_Eve"],
}

valid = []
for etype, concepts in TYPE_MAP.items():
    for concept in concepts:
        ex_path = os.path.join(EX_DIR, f"{concept}.json")
        pv_path = os.path.join(PV_DIR, f"{concept}.json")

        has_ex = os.path.exists(ex_path)
        has_pv = os.path.exists(pv_path)

        if not has_ex or not has_pv:
            continue

        # 检查摘要长度
        with open(ex_path, encoding="utf-8") as f:
            ex_data = json.load(f)
        pages = ex_data.get("query", {}).get("pages", {})
        extract = list(pages.values())[0].get("extract", "")
        n_chars = len(extract)
        if n_chars < 100:
            continue

        # 检查浏览量月份数
        with open(pv_path) as f:
            pv_data = json.load(f)
        items = pv_data.get("items", [])
        if len(items) < 12:
            continue

        # 浏览量统计
        views = [it["views"] for it in items]
        mean_v = np.mean(views)
        std_v = np.std(views)
        cv = std_v / mean_v if mean_v > 0 else 999

        # 真实趋势：首尾对比
        first_half = np.mean(views[:len(views)//2]) if len(views) >= 4 else 0
        second_half = np.mean(views[len(views)//2:]) if len(views) >= 4 else 0
        trend_ratio = second_half / first_half if first_half > 0 else 1

        # 真实分类
        if trend_ratio > 1.5:
            real_type = "Growth"
        elif trend_ratio < 0.6:
            real_type = "Declining"
        elif cv < 0.3:
            real_type = "Stable"
        else:
            real_type = "Fluctuating"

        valid.append({
            "concept": concept, "type": etype, "real_type": real_type,
            "n_chars": n_chars, "n_months": len(items),
            "mean_views": mean_v, "std_views": std_v, "cv": cv,
            "trend_ratio": trend_ratio,
            "extract": extract,
        })

# 输出
print(f"有效概念: {len(valid)}")
for etype in ["Stable","Burst","Decay","Oscillatory"]:
    n = sum(1 for v in valid if v["type"] == etype)
    names = [v["concept"] for v in valid if v["type"] == etype]
    print(f"  {etype}: {n} — {', '.join(names)}")

print(f"\n{'概念':<25} {'类型':<12} {'真实':<12} {'字符':>6} {'月数':>5} {'均值':>8} {'CV':>6} {'趋势':>5}")
print("-" * 90)
for v in valid:
    print(f"{v['concept']:<25} {v['type']:<12} {v['real_type']:<12} {v['n_chars']:>6} {v['n_months']:>5} {v['mean_views']:>8.0f} {v['cv']:>6.2f} {v['trend_ratio']:>5.2f}")

# 保存摘要文本（供 updown_rs 读取）
for v in valid:
    txt_path = os.path.join(OUT_DIR, "extracts", f"{v['concept']}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(v["extract"][:2000])  # 截取前 2000 字符

# 保存有效概念列表
csv_path = os.path.join(os.path.dirname(OUT_DIR), "results", "valid_concepts.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["concept","type","real_type","n_chars","n_months","mean_views","cv","trend_ratio"])
    w.writeheader()
    for v in valid:
        w.writerow({k: v[k] for k in w.fieldnames})

# 保存完整数据 JSON（给 Rust 测试用）
json_path = os.path.join(os.path.dirname(OUT_DIR), "results", "dataset.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "n_concepts": len(valid),
        "concepts": [{k: v[k] for k in ["concept","type","real_type","n_chars","n_months","mean_views","cv","trend_ratio"]} for v in valid]
    }, f, indent=2, ensure_ascii=False)

print(f"\n保存: {csv_path}")
print(f"保存: {json_path}")
