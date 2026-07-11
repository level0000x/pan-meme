# 泛模因理论（Pan-Meme Theory）文档索引

> 最后更新：2026-07-12（v2.7）

---

## 核心文档（当前版本 v2.7）

这两个文件是理论的权威来源。所有其他文档要么是历史版本，要么是补充材料。

| 文件 | 说明 |
|------|------|
| **[泛模因理论-完整知识库.md](./泛模因理论-完整知识库.md)** | 完整知识库。包含全部 40 条 ■ 定理/引理（含证明）、三公理体系（P₁/P₄/P₅）、N 算子动力学、§11 格结构定理群、§12 验证状态与优先级表。**理论的第一参考源。** |
| **[泛模因理论-推导链.md](./泛模因理论-推导链.md)** | 推导链。从物理定律到公理到定理的完整推导路径。包含 B-0（物理约束→格结构）、B-1 至 B-5 物理推导（四条验证路径）、B-1.4 纯信息论安全网、B-7 公理最小性、B-7.5 过渡节、B-8 一致性证明、B-9 连续流方程。 |

---

## 历史版本（已被 v2.7 替代）

这些文件是理论的早期版本。它们的内容已被吸收、重写或精简到上述两个核心文档中。保留它们作为历史参考——可以追溯理论演化的路径，但不应作为当前理论的引用来源。

| 文件 | 原版本 | 说明 |
|------|--------|------|
| [泛模因理论.md](./泛模因理论.md) | v1–v2 八命题版 | 理论的早期基础描述。包含八条命题公理（后被精简为 P₁/P₄/P₅ 三公理体系）。其核心论述已迁移至完整知识库 §1–§5。 |
| [formal-concept-analysis-proof.md](./formal-concept-analysis-proof.md) | v2 FCA 证明 | 从公理到 FCA 概念格的早期数学推导。其证明内容已整合到推导链 B-0（物理约束→格结构）和 B-8（FCA 一致性证明）。 |
| [pan-meme-complete-proof.md](./pan-meme-complete-proof.md) | v2 完整证明链 | 早期完整数学推导——从公理经 FCA、CW 胞腔复形、Betti 分解到 ODE 系统。谱路径（Chain A/B）已被 N 算子的 Chain C 替代。其 CW 复形部分仍与当前理论有历史关联。 |
| [pan-meme-mathematics.md](./pan-meme-mathematics.md) | v2–v3 谱桥 | 谱桥数学框架——离散组合结构到连续动力学映射的早期版本。Chain C 采用纯离散 N 迭代后不再需要谱桥。保留作为谱路径的数学参考。 |
| [pan-meme-aesthetic-proof.md](./pan-meme-aesthetic-proof.md) | v2 约束传播 | 约束传播证明——从概念格蕴涵约束到约束流形测地线的早期推导。其约束传播思想已以不同形式存在于 §11.2D 参数可辨识性定理中。 |
| [pan-meme-geodesic-proof.md](./pan-meme-geodesic-proof.md) | v2 测地线 | 谱桥测地线证明——谱社区划分、热迹参数、ODE 动力学的早期推导。与 pan-meme-mathematics.md 构成谱路径的完整版本。 |
| [pan-meme-variational-proof.md](./pan-meme-variational-proof.md) | v2 变分原理 | 变分原理证明——拉格朗日力学框架下的约束传播与测地线轨迹。其变分思想已以不同形式存在于 §6.2 ODE 解和 §6.17D KL Lyapunov 中。 |
| [proof-supplement-complete.md](./proof-supplement-complete.md) | v2 附录 D 补全 | 附录 D 的数学补全——双射性、信息量统一、ODE 推导与紧致化。对应 v2 的 Chain A/B 路径，Chain C 已不再使用这些构造。 |
| [spectral-bridge-design.md](./spectral-bridge-design.md) | v2–v3 谱桥设计 | 谱桥设计文档——离散加权图到连续动力学的数学补全。Chain C 采用纯离散路径后不再需要。保留作为谱-离散对比的数学参考。 |

---

## 参考与补充材料（仍具参考价值）

| 文件 | 说明 |
|------|------|
| [related-work.md](./related-work.md) | 相关文献综述。覆盖普适达尔文主义、模因学、信息论、代数拓扑等七个领域。为理论提供外部学术定位。**与当前 v2.7 兼容。** |
| [information-structural-dynamics.md](./information-structural-dynamics.md) | 信息-结构动力学。从八命题公理推导完整链的早期版本。其附录 B 与完整知识库有交叉引用。**部分内容已迁移，建议以完整知识库为准。** |
| [可复用资产-完整注释.md](./可复用资产-完整注释.md) | 项目中所有可复用资产（代码模块、数据结构、算法）的集中索引。标注来源、状态与依赖。**与当前代码库兼容。** |

---

## 规划与流程文档

| 文件 | 说明 |
|------|------|
| [ROSE-PAN-MEME-迁移规划.md](./ROSE-PAN-MEME-迁移规划.md) | ROSE-SCA 数据处理层迁移至 pan-meme 系统的规划。涉及统一数据摄入、可验证凭证、动力学演化。 |
| [verification-plan.md](./verification-plan.md) | 验证计划。将核心假设拆分为收敛性、统一性与可逆性三个子命题，提出可操作实验准则。 |
| [v4.2-changelog.md](./v4.2-changelog.md) | 工程版本 v4.2 变更记录。包括验证计划完善、原型多样性提升、实验数据处理优化。 |
| [综合审计-证明-断裂-路径.md](./综合审计-证明-断裂-路径.md) | 历史审计报告。记录各阶段证明中的"断裂点"与缺失论证。**其中列出的断裂点已全部在 v2.7 中闭合。** |
| [superpowers/plans/2026-07-07-pan-meme-rewrite.md](./superpowers/plans/2026-07-07-pan-meme-rewrite.md) | Pan-Meme 完整重写计划 v4。从数学构造出发重写四阶段管线的规划文档。 |

---

## 实验产物

| 路径 | 说明 |
|------|------|
| [assets/phi_pipeline/](./assets/phi_pipeline/) | φ 管线实验输出。包含原型分布图（archetype_distribution.png）、ODE 收敛图（ode_convergence.png）、样本轨迹图（sample_trajectories.png）、管线摘要 JSON（pipeline_summary.json）。 |

---

## 阅读路径建议

**首次阅读**：泛模因理论-完整知识库.md → 泛模因理论-推导链.md（第二部分 B-1 至 B-5）

**深入研究**：推导链全文 → 完整知识库 §6 + §11 定理群 → related-work.md

**追溯历史**：泛模因理论.md（八命题版）→ pan-meme-complete-proof.md（CW 复形路径）→ 综合审计-证明-断裂-路径.md（已闭合的断裂点历史）

**工程开发**：可复用资产-完整注释.md → 代码库根目录的 Cargo.toml 和工作区结构