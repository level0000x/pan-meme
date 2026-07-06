# 泛模因几何工具 — 模块一：规则提取器
# 数学对应：前提3（Ψ ↦ (F, C) 双射）— 从关系网络Ψ中提取规则域F与约束域C
# 论文位置：定义2（规则域F、约束域C）, 前提3（Ψ↦(F,C)双射唯一确定）, 附录D.3

from typing import List, Tuple, Dict, Set, Optional
import numpy as np

from pan_meme.core.types import RelationNetwork, RuleDef, ConstraintDef


class RuleExtractor:
    """前提3实现：从关系网络 Ψ = (V, E, w) 中提取规则域 F 和约束域 C。

    数学对应：
    - 前提3：Ψ ↦ (F, C) 是双射 — 即给定 Ψ，其规则域和约束域被唯一确定
    - 定义2：F = {(f_i, supp_i)} — 每个规则由一个模式描述和支撑集组成
    - 定义2：C = {(c_j, dom_j)} — 每个约束由一个条件和定义域组成

    提取策略：
    1. 幂律检测：若度分布的变异系数 cv > 3，则网络呈现无标度特性，
       据此提取"无标度传播规则"，置信度 = cv / 10
    2. 团检测：搜索 3-clique（三角形），每个三角形生成一条传递性规则，
       置信度由三角形边权均值决定
    3. 孤立节点检测：度数为 0 的节点触发 node_connectivity_min 约束

    用法:
        extractor = RuleExtractor()
        rules = extractor.extract_rules(psi)
        constraints = extractor.extract_constraints(psi)
    """

    # ================================================================
    # 公有方法：extract_rules — 从Ψ提取规则域F
    # 数学对应：前提3 — F = { (f, supp) | f 由 Ψ 的结构特征唯一确定 }
    # ================================================================

    def extract_rules(self, psi: RelationNetwork) -> List[RuleDef]:
        """从关系网络 Ψ 中提取规则域 F。

        数学对应：
        - 前提3：F = Φ_F(Ψ) — 规则域由 Ψ 的结构唯一确定
        - 定义2：(f, supp) ∈ F，其中 supp ⊆ V ∪ E

        提取步骤：
        1. 幂律检测：计算度分布变异系数 cv。
           若 cv > 3，则创建无标度传播规则（confidence = cv / 10）
        2. 团检测：搜索所有 3-clique（三角形），
           为每个三角形创建传递性规则（confidence = 边权均值）

        参数:
          psi: 输入关系网络，含节点集 V、边集 E 和权重向量 w

        返回:
          List[RuleDef] — 提取出的规则列表（幂律规则 + 传递性规则）
        """
        rules: List[RuleDef] = []

        # ── 步骤1：幂律检测 — 计算度分布变异系数 ──
        # 数学对应：前提3 — 无标度网络的度分布满足 P(k) ∝ k^{-γ}
        # 当 cv(度) > 3 时，表明度分布高度偏斜，符合无标度特征
        n: int = len(psi.nodes)
        if n == 0:
            return rules

        degree: np.ndarray = self._compute_degrees(psi, n)
        cv: float = self._coefficient_of_variation(degree)

        if cv > 3.0:
            # 无标度传播规则：支撑集 = 度最高的前 k 个节点（k = |V| // 4，至少 2 个）
            # 数学对应：定义2 F域 — 传播规则描述信息在网络中的优先扩散路径
            # confidence = cv / 10，上限 1.0，确保置信度在合理区间
            confidence: float = min(cv / 10.0, 1.0)
            k_support: int = max(2, n // 4)
            top_k_indices: List[int] = self._top_k_indices(degree, k_support)

            rules.append(RuleDef(
                pattern=(
                    "scale_free_propagation: "
                    f"cv={cv:.2f}, top_{k_support}_hubs"
                ),
                support=top_k_indices,
                confidence=round(confidence, 4),
                source="extracted",
            ))

        # ── 步骤2：团检测 — 搜索 3-clique（三角形）──────────────
        # 数学对应：前提3 — 3-clique 编码传递性关系：
        #   若 e(a,b) 且 e(b,c) 且 e(a,c) 均存在，则 {a,b,c} 构成传递性三元组
        triangles: List[Tuple[int, int, int]] = self._find_triangles(psi, n)

        for tri in triangles:
            # 计算三角形边权均值作为规则置信度
            # 数学对应：定义2 — confidence ∈ [0,1] 量化规则可靠性
            w_avg: float = self._triangle_weight_mean(psi, tri)
            rules.append(RuleDef(
                pattern=f"transitivity: 3_clique({tri[0]},{tri[1]},{tri[2]})",
                support=list(tri),
                confidence=round(w_avg, 4),
                source="extracted",
            ))

        return rules

    # ================================================================
    # 公有方法：extract_constraints — 从Ψ提取约束域C
    # 数学对应：前提3 — C = { (c, dom) | c 由 Ψ 的拓扑异常触发 }
    # ================================================================

    def extract_constraints(self, psi: RelationNetwork) -> List[ConstraintDef]:
        """从关系网络 Ψ 中提取约束域 C。

        数学对应：
        - 前提3：C = Φ_C(Ψ) — 约束域由 Ψ 的拓扑异常唯一确定
        - 定义2：(c, dom) ∈ C，其中 dom ⊆ V ∪ E

        提取策略：
        1. 孤立节点检测：度数为 0 的节点无任何连接，
           触发 node_connectivity_min 约束，要求至少 min_degree ≥ 1
        2. 该约束编码了网络结构完整性的必要条件

        参数:
          psi: 输入关系网络

        返回:
          List[ConstraintDef] — 提取出的约束列表
        """
        constraints: List[ConstraintDef] = []
        n: int = len(psi.nodes)
        if n == 0:
            return constraints

        degree: np.ndarray = self._compute_degrees(psi, n)

        # ── 孤立节点检测 ──
        # 数学对应：定义2 C域 — 孤立节点违反连通性基本假设，
        # 必须通过约束显式标记，供后续自洽性验证和修复使用
        isolated: List[int] = [int(i) for i in range(n) if degree[i] == 0]

        if isolated:
            constraints.append(ConstraintDef(
                condition="node_connectivity_min",
                domain=isolated,
                description=(
                    f"检测到 {len(isolated)} 个孤立节点（度=0），"
                    f"索引: {isolated}。违反最低连通性要求 min_degree ≥ 1。"
                ),
                confidence=1.0,
                source="extracted",
            ))

        return constraints

    # ================================================================
    # 内部方法：度分布计算
    # 数学对应：定义1 — deg(i) = |{j : (i,j) ∈ E ∨ (j,i) ∈ E}|
    # ================================================================

    def _compute_degrees(self, psi: RelationNetwork, n: int) -> np.ndarray:
        """计算网络中每个节点的度数。

        数学对应：
        - 定义1：deg(v) = |{u ∈ V : (v,u) ∈ E}|
        - 对于无向网络，入度与出度等同，直接计数邻边

        参数:
          psi: 关系网络
          n: 节点总数

        返回:
          np.ndarray of shape (n,) — 每个节点的度数值，dtype=np.float32
        """
        degree: np.ndarray = np.zeros(n, dtype=np.float32)
        for src, dst in psi.edges:
            degree[src] += 1.0
            degree[dst] += 1.0
        return degree

    # ================================================================
    # 内部方法：变异系数计算
    # 数学对应：统计量 cv = σ / μ，度量度分布的离散程度
    # ================================================================

    def _coefficient_of_variation(self, data: np.ndarray) -> float:
        """计算数据列的变异系数 cv = σ / μ。

        数学对应：
        - cv = σ(度) / μ(度) — 标准化离散度量
        - cv > 3 指示高度偏斜分布，支持无标度假设

        参数:
          data: 数值数组（通常为度序列），dtype=np.float32

        返回:
          float — 变异系数（若 μ ≈ 0，返回 0.0）
        """
        mu: float = float(np.mean(data))
        if mu < 1e-12:
            return 0.0
        sigma: float = float(np.std(data))
        return sigma / mu

    # ================================================================
    # 内部方法：前k大元素索引
    # 数学对应：支撑集 supp 取度最高的节点 — 枢纽节点是传播规则的核心
    # ================================================================

    def _top_k_indices(self, degree: np.ndarray, k: int) -> List[int]:
        """返回度数最高的前 k 个节点的索引。

        使用 argsort 降序排列，取前 k 个索引。

        参数:
          degree: 度数组，shape=(n,), dtype=np.float32
          k: 需要返回的节点数

        返回:
          List[int] — 度最高的前 k 个节点索引
        """
        # 降序排列索引：argsort 默认升序，[::-1] 取降序
        sorted_indices: np.ndarray = np.argsort(degree)[::-1]
        k_actual: int = min(k, len(sorted_indices))
        return [int(idx) for idx in sorted_indices[:k_actual]]

    # ================================================================
    # 内部方法：三角形检测
    # 数学对应：3-clique — 完全子图 K₃，编码传递性关系
    # ================================================================

    def _find_triangles(self, psi: RelationNetwork,
                        n: int) -> List[Tuple[int, int, int]]:
        """在关系网络中搜索所有 3-clique（三角形）。

        数学对应：
        - 3-clique = {i, j, k} 当 (i,j), (j,k), (i,k) ∈ E
        - 三角形编码传递性：若 i↔j 且 j↔k 则 i↔k

        算法：构建邻接集合 adj[i] = {所有 i 的邻居}，
        然后对每条边 (i, j) 检查 adj[i] ∩ adj[j] 的交集。

        参数:
          psi: 关系网络
          n: 节点总数

        返回:
          List[Tuple[int,int,int]] — 三角形顶点三元组列表（每个三角形仅出现一次）
        """
        # 构建邻接集合（无向图）
        adj: List[Set[int]] = [set() for _ in range(n)]
        for src, dst in psi.edges:
            # 排除自环（虽然正常输入不应有，但防御性处理）
            if src != dst:
                adj[src].add(dst)
                adj[dst].add(src)

        triangles: List[Tuple[int, int, int]] = []
        used: Set[Tuple[int, int, int]] = set()  # 去重用

        # 对每条边 (i, j)，检查 adj[i] ∩ adj[j]
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue  # 只处理 i < j，避免重复
                # 计算邻居交集
                common: Set[int] = adj[i] & adj[j]
                for k in common:
                    if k <= j:
                        continue  # 只保留 i < j < k 的唯一表示
                    tri: Tuple[int, int, int] = (i, j, k)
                    if tri not in used:
                        used.add(tri)
                        triangles.append(tri)

        return triangles

    # ================================================================
    # 内部方法：三角形边权均值
    # 数学对应：confidence(tri) = avg( w(i,j), w(j,k), w(i,k) )
    # ================================================================

    def _triangle_weight_mean(self, psi: RelationNetwork,
                              tri: Tuple[int, int, int]) -> float:
        """计算三角形三条边的权重均值，作为规则置信度。

        数学对应：
        - 定义2：confidence ∈ [0,1] — 规则可靠性
        - 对于三角形 {i,j,k}，confidence = (w_ij + w_jk + w_ik) / 3
        - 边权越高，传递性关系越可信

        参数:
          psi: 关系网络
          tri: 三角形顶点三元组 (i, j, k)

        返回:
          float — 三条边权重的算术均值，∈ [0, 1]
        """
        # 构建边到权重的快速查找映射
        edge_weight_map: Dict[Tuple[int, int], float] = {}
        for (s, d), w in zip(psi.edges, psi.weights):
            edge_weight_map[(s, d)] = float(w)
            edge_weight_map[(d, s)] = float(w)  # 无向图双向映射

        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])

        w_ij: float = edge_weight_map.get((i, j), 0.0)
        w_jk: float = edge_weight_map.get((j, k), 0.0)
        w_ik: float = edge_weight_map.get((i, k), 0.0)

        return (w_ij + w_jk + w_ik) / 3.0
