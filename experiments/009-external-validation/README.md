# 实验 009：真实世界历时数据验证

**日期**：2026-07-08  
**版本**：v4.3 计划  
**状态**：方案设计阶段

## 目的

填补项目最关键短板——**外部验证**。泛模因理论的五维 ODE 系统声称可描述任意信息-结构模式的演化轨迹。若此声称成立，ODE 从模因初始结构状态出发的模拟轨迹，应与同一模因在真实世界中的历时频率变化存在可量化的吻合度。

### 假设 H₄（外部预测假设）

> 对于从 Wikipedia 页面内容构建的初始模因状态，ODE 模拟的能流密度 ρ(t) 与同一页面在后续时间段内的月浏览量时间序列的 Pearson 相关系数显著不为零（ρ̄ > 0，p < 0.05），且该相关性在 ≥70% 的样本上成立。

## 实验设计

### 数据源

| 数据 | 来源 | 格式 | 验证状态 |
|------|------|------|----------|
| 页面浏览量 | [Wikimedia Pageviews API](https://wikimedia.org/api/rest_v1/metrics/pageviews/) | JSON（月粒度） | ✅ 已验证 |
| 页面摘要文本 | [Wikipedia API](https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext) | JSON（纯文本） | ✅ en 已验证，zh 待调试 |
| 输入推荐 | 选取英文 Wikipedia 概念词（有标准摘要文本 + 浏览量数据） | — | — |

### 概念词选取

选取 50-100 个英文 Wikipedia 概念词，覆盖三种演化类型：

| 类型 | 预期原型 | 示例词 | 选取标准 |
|------|----------|--------|----------|
| **稳定型** | Stone / StableCore | "Newton", "Gravity", "Democracy" | 浏览量在 2015-2025 期间平稳，标准差 < 均值 |
| **爆发型** | Burst / Source | "ChatGPT", "COVID-19", "NFT" | 存在明显峰值，峰值/谷值比 > 5 |
| **衰减型** | Decay / Transient | "Flash_Player", "MySpace" | 浏览量单调下降，年降幅 > 20% |
| **周期型** | Oscillatory | "Christmas", "Olympics" | 年周期明显，自相关显著 |

每类 15-25 个词，确保统计效力。

### 具体步骤

```
Step 1: 数据采集（离线脚本）
   ├── 1a. 对每个概念词，调用 Wikipedia API 获取页面摘要（前 1000 字）
   ├── 1b. 对每个概念词，调用 Pageviews API 获取 2015.07-2025.07 月浏览量
   ├── 1c. 保存到 experiments/009-external-validation/data/*.json
   └── 1d. 归一化：浏览量归一化到 [0,1]

Step 2: 模因初始状态构建（updown_rs 管线）
   ├── extract_ngrams(页面摘要) → Phase 1: 浮现
   ├── Phase 2: 编码 → CW 复形
   ├── Phase 3: 分解（γ=2.0）→ 主社区五维状态
   └── 取最大社区的 (D₀, B₀, ρ₀, R₀, S₀) 作为该模因的初始状态

Step 3: ODE 模拟
   ├── 从初始状态 + 11 参数出发
   ├── t_max = 120（对应 120 个月 = 10 年）
   ├── dt 对应 1 个月
   └── 提取 ρ(t) 和 D(t) 时间序列

Step 4: 对比验证
   ├── Pearson r(ρ_ode, views_real)
   ├── 统计显著性 p 值
   ├── Spearman ρ（排序相关性，对非线性关系更稳健）
   └── DTW 距离（动态时间规整，对齐不同速率的演化）

Step 5: 分类分析
   ├── 按 ODE 原型分类
   ├── 按真实轨迹形分类（增长/衰减/稳定/波动）
   ├── 混淆矩阵：ODE 分类 vs 真实分类
   └── 分组对比：不同原型的预测准确度差异

Step 6: 参数稳健性
   ├── 11 参数各自 ±20% 振动
   ├── 计算最大 ρ 和最小 ρ
   └── 报告稳健区间
```

### 验收标准

| 指标 | 判定标准 | 含义 |
|------|----------|------|
| 平均 Pearson r | ρ̄ > 0.3 且 p < 0.01 | ODE 轨迹解释了 ≥9% 的真实变异性 |
| 显著比例 | ≥70% 样本 p < 0.05 | 大多数模因的预测显著优于随机基线 |
| 原型一致性 | 混淆矩阵对角占优 | ODE 分类与真实轨迹分类一致 |
| 基石型 vs 泡沫型 | ρ̄(Stone) > ρ̄(Burst) | 理论预期方向成立 |
| 参数稳健性 | Δρ < 0.1（±20% 振动） | 结果对参数不敏感 |

### 内部对比基线

| 基线方法 | 说明 |
|----------|------|
| 随机游走 | 从 ρ(0) 出发，每步 N(0, 0.01) |
| AR(1) | ρ(t) = φ·ρ(t-1) + ε, φ=0.9 |
| 指数衰减 | ρ(t) = ρ₀·exp(-λt) |
| 常值预测 | ρ(t) = ρ₀ |

要求 ODE 模拟的 ρ̄ 显著优于所有基线。

## 目录结构

```
experiments/009-external-validation/
├── README.md                    # 本文件
├── design.md                    # 实验方案（本文件）
├── data/                        # 原始数据
│   ├── concepts.json            #   概念词列表 + 元数据
│   ├── pageviews/               #   逐个概念的月浏览量 JSON
│   └── extracts/                #   逐个概念的页面摘要
├── results/                     # 处理后的结果
│   ├── summary.csv              #   逐概念: 初始状态 + r 值 + p 值 + 原型
│   └── diagnostics/             #   每个概念的详细诊断
└── scripts/                     # 数据采集脚本
    ├── fetch_pageviews.py       #   采集月浏览量
    └── fetch_extracts.py        #   采集页面摘要
```

## 数据采集脚本

### fetch_pageviews.py

```python
#!/usr/bin/env python3
"""采集 Wikipedia 页面月浏览量 (2015.07-2025.07)"""
import json, urllib.request, time

CONCEPTS = ["Newton", "Gravity", "ChatGPT", "COVID-19", ...]  # 填充概念列表
BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents"

for concept in CONCEPTS:
    url = f"{BASE}/{concept}/monthly/2015070100/2025070100"
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.load(resp)
        with open(f"data/pageviews/{concept}.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ {concept}: {len(data['items'])} months")
    except Exception as e:
        print(f"✗ {concept}: {e}")
    time.sleep(0.1)  # 礼貌速率限制
```

### fetch_extracts.py

```python
#!/usr/bin/env python3
"""采集 Wikipedia 页面摘要（前 1000 字）"""
import json, urllib.request, urllib.parse, time

CONCEPTS = ["Newton", "Gravity", "ChatGPT", "COVID-19", ...]

for concept in CONCEPTS:
    params = urllib.parse.urlencode({
        "action": "query",
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "titles": concept,
        "format": "json"
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.load(resp)
        with open(f"data/extracts/{concept}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        extract = list(data["query"]["pages"].values())[0].get("extract", "")[:100]
        print(f"✓ {concept}: {len(extract)} chars → '{extract}...'")
    except Exception as e:
        print(f"✗ {concept}: {e}")
    time.sleep(0.1)
```

## 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| 中文 Wikipedia API 不可用 | 中 | 使用英文 Wikipedia 替代，概念词用英文 |
| 页面摘要为空或过短 | 高 | 取消 exintro 限制，用 full page extract |
| 单概念词管线产出 0 边 | 低 | 已验证：1000 字摘要足够产生 PSI 图 |
| ODE ρ 和浏览量时间尺度不匹配 | 中 | 用 DTW 对齐 + 归一化，允许尺度差异 |
| 样本量不足（<50） | 中 | 扩大概念词列表，爬虫采集 |

## 复现

```bash
# Step 1: 数据采集
cd experiments/009-external-validation
python scripts/fetch_pageviews.py
python scripts/fetch_extracts.py

# Step 2: 运行验证（待实现）
cd updown_rs
cargo test --lib experiment_009_external_validation -- --nocapture
```

## 待定事项

- [ ] 确定 50-100 个概念词最终列表
- [ ] 中文 Wikipedia extract API 调试
- [ ] 实现 `experiment_009_external_validation` 测试
- [ ] 验证数据采集脚本
