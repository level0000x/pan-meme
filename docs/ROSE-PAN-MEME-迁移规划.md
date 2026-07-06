# ROSE-SCA 数据处理层 → pan-meme 迁移规划

**版本**: v1.0 | **日期**: 2026-07-05 | **状态**: 规划中

---

## 1. 背景

ROSE-SCA 的 L0 给定物层（数据摄入、WikiLine TSV 生成、知识库构建）目前由 10+ 个独立的 Python 脚本组成（`ingest_wordnet.py`、`ingest_conceptnet_full.py`、`ingest_wikidata_zh.py` 等），缺乏统一的管线框架和数学可验证性。

pan-meme 项目提供了四阶段数学双射管线（Φ_A→Φ_B→Φ_C→Φ_D），每个阶段有严格的定理保证（定理1-10，附录D完整证明）。将 ROSE 的数据处理迁移到 pan-meme 管线，可以：

- 统一数据摄入流程（所有数据源走同一管线）
- 获得双向可逆性保证（Φ⁻¹(Φ(I))=I）
- 获得可验证凭证（Merkle 树替代 JSONL 操作日志）
- 获得动力学演化能力（五维 ODE，ROSE 此前没有）

## 2. 架构定位

```
ROSE-SCA 七层架构                    pan-meme 管线
═══════════════════                  ══════════════
L0 给定物层 ─────────────────→ 模块一: 浮现（数学化）
  (数据摄入、TSV生成)              I → Ψ → M

L1 操作层（目光循环）    ←──→ 模块一的 ↑↓ 循环
  (gaze_up/gaze_down)           公理3: 归类/分解互逆

L2 知识层                     ←── 模块二: 几何化
  (containment+connection)        M → G（胞腔复形）

L4 推理层                     ←── 模块一的 Reasoner
  (Warshall/Johnson)             Warshall 传递闭包

—                              ←── 模块三: 模因化
  (ROSE 没有)                     G → Q（五维状态）

—                              ←── 模块四: 绑定
  (ROSE 用 JSONL 操作日志)        Q → Credential（Merkle）

—                              ←── ODE 动力学
  (ROSE 没有)                     五维方程 → 三类原型
```

pan-meme 是 ROSE 的数据处理前端：输入原始数据，输出 WikiLine TSV（兼容 ROSE 现有格式）和可验证凭证。

## 3. 迁移映射

### 3.1 模块级映射

| ROSE 现有模块 | 行数 | → pan-meme 模块 | 迁移方式 |
|:---|:---|:---|:---|
| `ingest_wordnet.py` | ~200 | `module1_input/adapter.py` | 原始数据 → Tokenizer → RelationExtractor → MathModel |
| `ingest_conceptnet_full.py` | ~200 | 同上 | 同上 |
| `ingest_wikidata_zh.py` | ~260 | 同上 | 同上 |
| `ingest_sumo.py` | ~200 | 同上 | 同上 |
| `ingest_multimodal.py` | ~465 | `plugins/modalities/` | 多模态 Tokenizer 已设计 |
| `predicate_mapper.py` | ~150 | `module1_input/rule_extractor.py` | 规则提取替代固定映射表 |
| `reasoning_engine.py` | ~1135 | `module1_input/reasoner.py` | Warshall 已有，补 Johnson |
| `wiki_writer.py` | ~100 | `tsv_bridge.py`（新） | 输出 WikiLine TSV |
| `rose_shared.py` 日志 | ~200 | `module4_bind/` | JSONL → Merkle 凭证 |

### 3.2 数据流映射

```
ROSE 现有流程:
  raw data → ingest_*.py → predicate_mapper → wiki_writer → WikiLine TSV

pan-meme 流程:
  raw data → Tokenizer → CycleEngine → RelationExtractor → Reasoner
  → Completeness → ConceptComposer → RuleExtractor → Consistency
  → MathModel → Geometrizer → MemeDecomposer → CredentialBinder
  → tsv_bridge → WikiLine TSV
```

## 4. 实现计划

### 前提

pan_meme 已有骨架代码（`core/types.py`、`core/pipeline.py`、插件系统、模块接口定义），但大部分模块的实际实现为空桩。需要补全实现。

### 阶段 A：补全 pan_meme 管线

| # | 文件 | 行数估计 | 功能 | 依赖 |
|:---|:---|:---|:---|:---|
| A1 | `module1_input/cycle.py` | 150 | ↑↓ 循环：关键词匹配归类/分解，收敛检测 | 无 |
| A2 | `module1_input/relation_extractor.py` | 200 | 关系提取：共现强度 + 相似度 → Ψ=(V,E,w) | 无 |
| A3 | `module1_input/reasoner.py` | 250 | 推理：Warshall 传递闭包 + Johnson 稀疏 | 无 |
| A4 | `module1_input/adapter.py` | 200 | 模块一主入口：raw → MathModel，含 inverse() | A1, A2, A3 |
| A5 | `module2_geo/geometrizer.py` | 150 | 几何化：MathModel → GeometricObject | A4 |
| A6 | `module3_meme/decomposer.py` | 200 | 模因化：连通分量 → 五维状态 → 参数 | A5 |
| A7 | `module4_bind/binder.py` | 200 | 分层哈希 + Merkle 树 + 凭证 | A6 |
| A8 | `engines/ode_solver.py` | 200 | 五维 ODE 求解器 + 三类原型分类 | A6 |

### 阶段 B：桥接 ROSE

| # | 文件 | 行数估计 | 功能 | 依赖 |
|:---|:---|:---|:---|:---|
| B1 | `tsv_bridge.py` | 150 | PipelineData → WikiLine TSV（9字段） | A8 |
| B2 | `rose_ingest.py` | 150 | 一键摄入：WordNet/ConceptNet/Wikipedia/SUMO | B1 |

### 阶段 C：验证

| # | 内容 | 方法 |
|:---|:---|:---|
| C1 | 双射验证 | 随机文本 round-trip：Φ⁻¹(Φ(I))=I |
| C2 | Merkle 验证 | 有效凭证 → True，篡改 → 精确定位 |
| C3 | 三类原型 | 合成案例参数 → 基石/过客/泡沫 |
| C4 | TSV 兼容 | 输出与 ROSE 现有 TSV 格式一致 |

## 5. 技术约束

| 约束 | 说明 |
|:---|:---|
| Python 标准库 | 原型阶段零外部依赖（和 ROSE `api_server.py` 一致） |
| 不依赖 jieba | 中文分词用纯 Python 实现（基于前缀词典） |
| 不依赖 numpy | 矩阵操作用嵌套 list + 手动实现 |
| 不依赖 scipy | ODE 用 RK4/RK45 手动实现 |
| 不依赖 networkx | 图算法手动实现（BFS/DFS/Warshall） |
| 输出格式 | WikiLine TSV（9字段，制表符分隔，UTF-8） |
| 后续升级 | Python 编排层不变，热路径切 C（和 ROSE 一致） |

## 6. 输出格式：WikiLine TSV 桥接

WikiLine TSV 是 ROSE-SCA 的知识表示格式，9 个字段：

```
record_id	entity	predicate	target	source	state	conflicts	parent	timestamp
```

pan-meme 的 `PipelineData` 到 TSV 的映射：

| PipelineData 字段 | → TSV 字段 | 映射规则 |
|:---|:---|:---|
| psi.nodes[i] | entity | 节点名 |
| psi.edges (containment) | predicate=containment | 边权 > 阈值 → containment |
| psi.edges (connection) | predicate=connection | 边权 > 阈值 → connection |
| psi.edges (equivalence) | predicate=equivalence | 边权 > 阈值 → equivalence |
| 来源标识 | source | wordnet / conceptnet / wikipedia_zh / sumo |
| — | state | unheld（初始状态） |
| — | conflicts | -（无冲突） |
| hierarchy.parent | parent | 层级树的父节点 |
| — | timestamp | ISO 8601 当前时间 |

## 7. 配置

```python
# pm_core/config.py（对标 rose_shared.py 的 ROSE_CONFIG）

class PM_CONFIG:
    # 模块一：浮现
    CYCLE_MAX_ROUNDS = 20          # ↑↓ 循环最大轮次
    CYCLE_CONVERGE_THRESHOLD = 3   # 连续 N 轮无变化即收敛
    RELATION_THRESHOLD = 0.5       # 边权阈值（T）
    COOCCURRENCE_WINDOW = 100      # 滑动窗口大小
    TRANSITIVE_DECAY = 0.9         # 传递闭包衰减因子
    MAX_COMPONENT_RATIO = 0.8      # 最大连通分量占比
    MAX_ISOLATED_RATIO = 0.2       # 最大孤立节点占比

    # 模块二：几何化
    # （无额外配置，直接映射）

    # 模块三：模因化
    CELL_COUNT_MAX = 10000         # 胞腔复形最大顶点数
    LEVEL_DEPTH_MAX = 20           # 最大层级深度

    # 模块四：绑定
    HASH_ALGO = "sha256"           # 哈希算法

    # ODE 动力学
    ODE_METHOD = "RK45"            # 求解方法
    ODE_ATOL = 1e-8                # 绝对容差
    ODE_RTOL = 1e-8                # 相对容差
    ODE_T_SPAN = (0, 100)          # 时间范围
    ODE_MAX_STEP = 0.1             # 最大步长

    # TSV 桥接
    TSV_OUTPUT_DIR = "../sighted-wiki/data/wiki_tsv/records"
    TSV_SOURCE_MAP = {
        "wordnet": "wordnet",
        "conceptnet": "conceptnet",
        "wikipedia_zh": "wikipedia_zh",
        "sumo": "sumo",
    }
```

## 8. 测试策略

| 层级 | 内容 | 方法 |
|:---|:---|:---|
| 单元 | 每个模块独立测试 | 固定输入 → 预期输出 |
| 集成 | 模块一全链路 | 文本 → Tokenizer → MathModel |
| 端到端 | 全管线 + TSV 输出 | 原始数据 → WikiLine TSV |
| 属性 | 双射性验证 | Φ⁻¹(Φ(I)) = I（随机文本 100 次） |
| 兼容 | ROSE 可读 | 生成的 TSV 被 `api_server.py` 正常加载 |

## 9. 与 ROSE 主项目的集成

迁移完成后，ROSE 的数据摄入变为：

```
# 旧方式
python ingest_wordnet.py wordnet_data/ output/

# 新方式
python rose_ingest.py --source wordnet --input wordnet_data/
python rose_ingest.py --source conceptnet --input conceptnet.csv.gz
python rose_ingest.py --source wikipedia_zh --input wiki_batch.json
python rose_ingest.py --source sumo --input sumo_merge.kif
```

ROSE 的 `api_server.py` 无需任何改动——它只读 TSV 文件，不关心 TSV 是怎么生成的。

## 10. 后续演进

| 阶段 | 内容 |
|:---|:---|
| 原型（当前） | Python 标准库，零依赖，验证管线正确性 |
| 热路径优化 | Reasoner/Tokenizer 切 C（和 ROSE C 运行时统一） |
| 分布式 | 参考 `PB_ARCHITECTURE.md`，Spark/GraphBLAS 扩展 |
| 形式化 | Lean 4 验证定理 1-10（和 Lv-00 的 Lean 文件统一） |