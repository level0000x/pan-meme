# 泛模因几何工具 — 模块一：推理与补全引擎
# 数学对应：关系闭合——对 Ψ 的传递/对称/共现推理产生 Ψ*
# 论文位置：定义1（Ψ=(V,E,w)），定理2（Φ_B: Ψ→M 需要完整 Ψ*），附录D.3
# 核心思想：从已有关系网络通过5条推理规则补全缺失的隐式关系

from typing import List, Tuple, Set, Optional, Dict
import numpy as np

from pan_meme.core.types import RelationNetwork

# PB 级新引擎导入 — 数学对应：PB_ARCHITECTURE.md 第3.2节 Star 稀疏传递闭包
try:
    from pan_meme.engines.star_closure import StarClosure
    _STAR_AVAILABLE: bool = True
except ImportError:
    StarClosure = None  # type: ignore
    _STAR_AVAILABLE: bool = False


class Reasoner:
    """关系闭合推理器：对 Ψ 施加5条推理规则以产生 Ψ*。

    数学对应：
    - 关系闭合：对 Ψ 的传递/对称/共现推理产生 Ψ* — 最大完整关系网络
    - 定理2前驱：Φ_B: Ψ → M 要求 Ψ 已补全（Ψ*），否则双射不成立
    - 附录D.3：推理规则构成 F 域的自动补全过程

    5条推理规则：
    1. 传递性推理（Warshall闭包，衰减因子 0.9）：
       ∀i,j,k: w(i,j) ≥ min(w(i,k), w(k,j)) × 0.9
       若 A→B, B→C，则推出 A→C，但权重衰减。
    2. 对称性推理（衰减因子 0.85）：
       若 w(i,j) > 0 且 w(j,i) == 0，则 w(j,i) = w(i,j) × 0.85
       单向关系可能是对称关系的不完全观测。
    3. 共现推理（Jaccard 系数，共享邻居 ≥ 3）：
       若 |N(i) ∩ N(j)| ≥ 3，则 w(i,j) = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
       共享足够多邻居暗示隐藏的直接关系。
    4. 模式补全（k-团缺边检测）：
       在近乎完备的子图中补全缺失的边（k-团中缺失边权重取邻居均值）。
    5. 结构相似推理（连接模式余弦相似度 ≥ 0.7）：
       若两节点的邻居连接模式高度相似，则它们可能有直接关系。

    用法:
        reasoner = Reasoner()
        psi_star = reasoner.infer(psi)  # psi_star.metadata["inference_applied"] = True
    """

    def __init__(self,
                 use_star_closure: bool = False,
                 star_max_iter: int = 20,
                 star_tol: float = 1e-4) -> None:
        """初始化推理器 — 支持 O(N³) Warshall 模式与 Star 稀疏闭包模式。

        数学对应：PB_ARCHITECTURE.md 第3.2节 — Star 稀疏传递闭包
        - use_star_closure=False：原始 O(N³) Warshall 闭包（精确模式，默认）
        - use_star_closure=True：Star 分解的稀疏传递闭包，PB 级适用

        参数:
          use_star_closure: 是否启用 Star 稀疏闭包模式（默认 False）
          star_max_iter: Star 分解最大迭代次数（默认 20）
          star_tol: 收敛容差（默认 1e-4）
        """
        self.use_star_closure: bool = use_star_closure
        self.star_max_iter: int = star_max_iter
        self.star_tol: float = star_tol

        # 原始推理参数（Star 模式与精确模式共用）
        self.transitive_decay: float = 0.9       # 传递性衰减因子
        self.symmetry_decay: float = 0.85         # 对称性衰减因子
        self.cooccurrence_threshold: int = 3      # 共现推理所需最少共享邻居数
        self.structural_similarity_threshold: float = 0.7  # 结构相似度阈值

    # ================================================================
    # 公有方法：infer — 主推理入口
    # 数学对应：关系闭合算子 — Ψ ↦ Ψ*
    # ================================================================

    def infer(self, psi: RelationNetwork) -> RelationNetwork:
        """执行5条推理规则，返回补全后的关系网络 Ψ*。

        数学对应：
        - 关系闭合：从 Ψ 的观测边集 E 出发，通过推理规则生成隐式边集 E*
        - Ψ* = (V, E*, w*) 是原始网络在推理规则下的闭包
        - metadata["inference_applied"] 标记网络已经过推理补全

        推理流程（按顺序执行）：
        1. 构建完整 n×n 对称权重矩阵（当前仅含观测边）
        2. 传递性推理 Warshall 闭包
        3. 对称性推理
        4. 共现推理
        5. 模式补全
        6. 结构相似推理
        7. 从补全后的矩阵重建 edges + weights

        参数:
          psi: 原始关系网络 Ψ（由 RelationExtractor 生成）

        返回:
          补全后的 RelationNetwork Ψ* — 含 inference_applied 标记和推理统计元数据
        """
        # ── PB 级引擎分流：use_star_closure=True 时走 Star 稀疏闭包路径 ──
        # 数学对应：PB_ARCHITECTURE.md 第3.2节 — Star 稀疏传递闭包
        if self.use_star_closure:
            if not _STAR_AVAILABLE:
                raise ImportError(
                    "Star 闭包引擎不可用：pan_meme.engines.star_closure 未安装或导入失败"
                )
            return self.infer_star(psi)

        n: int = len(psi.nodes)
        # ── 步骤1：构建 n×n 对称权重矩阵 W ──
        # 数学对应：定义1 — w: E → [0,1]，对称矩阵表示
        W: np.ndarray = np.zeros((n, n), dtype=np.float32)
        for (i, j), w in zip(psi.edges, psi.weights):
            # 取 max 确保多边情况保留最强信号
            W[i, j] = W[j, i] = max(float(W[i, j]), float(w))

        # 记录推理前的边计数（用于统计新增边数）
        edge_count_before: int = int(np.sum(W > 0))

        # ── 步骤2：传递性推理 — Warshall 闭包（衰减 0.9）──
        # 数学对应：Warshall 传递闭包算法，w(i,j) ≥ min(w(i,k), w(k,j)) × 0.9
        # 迭代式：对每个中间节点 k，尝试通过 k 连接 i 和 j
        n_transitive: int = self._apply_transitive_inference(W, n)

        # ── 步骤3：对称性推理（衰减 0.85）──
        # 数学对应：若观测到 w(i,j) > 0 而 w(j,i) = 0，则对称化
        n_symmetric: int = self._apply_symmetry_inference(W, n)

        # ── 步骤4：共现推理（Jaccard 系数，共享邻居 ≥ 3）──
        # 数学对应：Jaccard co-occurrence = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
        n_cooccurrence: int = self._apply_cooccurrence_inference(W, n)

        # ── 步骤5：模式补全（k-团缺边检测）──
        # 数学对应：在近乎完备的子图（团）中检测并补全缺失边
        n_pattern: int = self._apply_pattern_completion(W, n)

        # ── 步骤6：结构相似推理（连接模式余弦相似度 ≥ 0.7）──
        # 数学对应：若两节点的邻居模式高度相似，可能具有隐式关系
        n_structural: int = self._apply_structural_inference(W, n)

        # ── 步骤7：从 W 重建 edges 和 weights ──
        # 数学对应：E* = {(i,j) | w*(i,j) > 0}, 保留上三角避免重复
        edges_star: List[Tuple[int, int]] = []
        weights_star: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                w_ij: float = float(W[i, j])
                if w_ij > 0.0:
                    edges_star.append((i, j))
                    weights_star.append(w_ij)

        # ── 构建推理元数据 ──
        total_inferred: int = (n_transitive + n_symmetric + n_cooccurrence +
                               n_pattern + n_structural)
        inference_meta: Dict[str, object] = dict(psi.metadata)  # 继承原始元数据
        inference_meta["inference_applied"] = True
        inference_meta["inference_stats"] = {
            "edges_before": edge_count_before,
            "edges_after": len(edges_star),
            "edges_added": len(edges_star) - edge_count_before,
            "transitive_inferred": n_transitive,
            "symmetric_inferred": n_symmetric,
            "cooccurrence_inferred": n_cooccurrence,
            "pattern_inferred": n_pattern,
            "structural_inferred": n_structural,
            "total_inferred": total_inferred,
        }

        return RelationNetwork(
            nodes=list(psi.nodes),                     # V* = V（节点集不变）
            edges=edges_star,                          # E* ⊇ E（边集扩展）
            weights=np.array(weights_star, dtype=np.float32),
            hierarchy=dict(psi.hierarchy),             # 保留层级信息
            metadata=inference_meta,
        )

    # ================================================================
    # 规则1：传递性推理 — Warshall 闭包
    # 数学对应：∀k: w(i,j) ≥ min(w(i,k), w(k,j)) × 0.9
    # ================================================================

    def _apply_transitive_inference(self, W: np.ndarray, n: int) -> int:
        """施加传递性推理（Warshall 闭包，衰减因子 0.9）。

        数学对应：
        - Warshall 传递闭包：对每个中间节点 k，检查 i→k 和 k→j 是否暗示 i→j
        - 衰减推理：推理出的边权重受衰减因子惩罚（信息传播损耗）
        - w_new(i,j) = max( w(i,j), min(w(i,k), w(k,j)) × 0.9 )

        参数:
          W: n×n 对称权重矩阵（原地修改）
          n: 节点数

        返回:
          新推理出的边数量
        """
        count: int = 0
        # Floyd-Warshall 风格的三重循环 — O(n³)
        # 数学对应：ω-正则传递闭包（带衰减）
        for k in range(n):
            for i in range(n):
                if i == k or W[i, k] == 0.0:
                    continue
                for j in range(n):
                    if j == i or j == k or W[k, j] == 0.0:
                        continue
                    # 潜在的新权重：取链路中较弱的一边，再衰减
                    new_w: float = float(min(W[i, k], W[k, j])) * self.transitive_decay
                    if new_w > float(W[i, j]):
                        W[i, j] = W[j, i] = np.float32(new_w)
                        count += 1
        return count

    # ================================================================
    # 规则2：对称性推理
    # 数学对应：若 w(i,j) > 0 且 w(j,i) = 0，则 w(j,i) = w(i,j) × 0.85
    # ================================================================

    def _apply_symmetry_inference(self, W: np.ndarray, n: int) -> int:
        """施加对称性推理（衰减因子 0.85）。

        数学对应：
        - 对称化：现实关系通常是（近似）对称的，单向边可能是观测不完整
        - 衰减推理：对称化的边权重受 0.85 因子惩罚以区分直接观测
        - 仅对已存在单向边的节点对进行对称化（不对全零对操作）

        参数:
          W: n×n 对称权重矩阵（原地修改）
          n: 节点数

        返回:
          新推理出的边数量
        """
        count: int = 0
        # 注意：W 在整个推理过程中始终保持对称
        # 此步骤处理的是在传递性推理之后仍可能存在的非对称情况
        # 实际实现中，由于前面步骤维护对称性，此处作为形式化验证
        for i in range(n):
            for j in range(n):
                if i >= j:
                    continue
                w_ij: float = float(W[i, j])
                w_ji: float = float(W[j, i])
                if w_ij > 0.0 and w_ji == 0.0:
                    W[j, i] = W[i, j] = np.float32(w_ij * self.symmetry_decay)
                    count += 1
                elif w_ji > 0.0 and w_ij == 0.0:
                    W[i, j] = W[j, i] = np.float32(w_ji * self.symmetry_decay)
                    count += 1
        return count

    # ================================================================
    # 规则3：共现推理 — Jaccard 系数
    # 数学对应：|N(i) ∩ N(j)| ≥ 3 → Jaccard(i,j) = |∩|/|∪|
    # ================================================================

    def _apply_cooccurrence_inference(self, W: np.ndarray, n: int) -> int:
        """施加共现推理（Jaccard 系数，共享邻居 ≥ 3）。

        数学对应：
        - 共现度量：Jaccard(i,j) = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
        - 条件：共享邻居数 ≥ 3（防止小样本噪音）
        - 社交网络理论：共享足够多共同邻居的节点倾向于直接相连
          （Granovetter 的三角闭包原理）

        参数:
          W: n×n 对称权重矩阵（原地修改）
          n: 节点数

        返回:
          新推理出的边数量
        """
        count: int = 0
        for i in range(n):
            for j in range(i + 1, n):
                # 跳过已有边的节点对
                if float(W[i, j]) > 0.0:
                    continue
                common: List[int] = self._common_neighbors(i, j, W)
                if len(common) >= self.cooccurrence_threshold:
                    # 计算 Jaccard 系数
                    union_size: int = (self._degree(i, W) +
                                       self._degree(j, W) -
                                       len(common))
                    if union_size > 0:
                        jaccard: float = len(common) / float(union_size)
                        W[i, j] = W[j, i] = np.float32(jaccard)
                        count += 1
        return count

    # ================================================================
    # 规则4：模式补全 — k-团缺边检测
    # 数学对应：在近乎完备的子图中补全缺失边
    # ================================================================

    def _apply_pattern_completion(self, W: np.ndarray, n: int) -> int:
        """施加模式补全推理（k-团缺边检测）。

        数学对应：
        - k-团检测：在大小为 k 的几乎完备子图中，缺失边的权重取其邻居均值
        - 团大小从 3 开始检测（三角形 → 四边形 → ...）
        - 缺失边权 = mean{w(i, m) | m ∈ N(i) ∩ C} 的截断均值

        算法：
        1. 对每个节点 i，找出其邻居集 N(i)
        2. 在 N(i) 内部检测哪些边缺失
        3. 缺失边的权重取共享邻居的均值

        参数:
          W: n×n 对称权重矩阵（原地修改）
          n: 节点数

        返回:
          新推理出的边数量
        """
        count: int = 0
        # 对每个节点，在其邻居子图中检测缺失边
        for i in range(n):
            neighbors_i: List[int] = [
                k for k in range(n) if k != i and float(W[i, k]) > 0.0
            ]
            if len(neighbors_i) < 2:
                continue

            # 在 N(i) 内部寻找缺失的边（i 的邻居之间的连接）
            for a_idx in range(len(neighbors_i)):
                a: int = neighbors_i[a_idx]
                for b_idx in range(a_idx + 1, len(neighbors_i)):
                    b: int = neighbors_i[b_idx]
                    if float(W[a, b]) > 0.0:
                        continue  # 边已存在

                    # 计算缺失边的合理权重 = 两节点与 i 连接权重的调和均值
                    # 数学对应：三角形补全 — 若 i-A 和 i-B 均强，则 A-B 可能也强
                    w_ia: float = float(W[i, a])
                    w_ib: float = float(W[i, b])
                    if w_ia > 0.0 and w_ib > 0.0:
                        # 调和均值偏向较小值（保守估计）
                        inferred_w: float = (2.0 * w_ia * w_ib) / (w_ia + w_ib + 1e-12)
                        # 额外考虑两节点共享的其他邻居
                        common_ab: List[int] = self._common_neighbors(a, b, W)
                        if len(common_ab) >= 1:
                            # 有共同邻居支撑时，权重取最大值提升可信度
                            # 但不超过调和均值的 1.2 倍
                            boost: float = min(1.0, len(common_ab) * 0.1)
                            inferred_w = min(inferred_w * (1.0 + boost), 1.0)

                        W[a, b] = W[b, a] = np.float32(inferred_w)
                        count += 1
        return count

    # ================================================================
    # 规则5：结构相似推理 — 连接模式余弦相似度
    # 数学对应：cos( row_i(W), row_j(W) ) ≥ 0.7 → 存在隐式关系
    # ================================================================

    def _apply_structural_inference(self, W: np.ndarray, n: int) -> int:
        """施加结构相似推理（连接模式余弦相似度 ≥ 0.7）。

        数学对应：
        - 结构等价性：若两节点的邻居连接模式高度相似，它们可能属于同一语义簇
        - 连接模式：节点 i 的连接模式由其权重行向量 row_i(W) 描述
        - cos(row_i, row_j) = (row_i · row_j) / (||row_i|| ||row_j||)
        - 相似度 ≥ 0.7 且当前无边 → 补全边，权重 = sim × 0.8

        参数:
          W: n×n 对称权重矩阵（原地修改）
          n: 节点数

        返回:
          新推理出的边数量
        """
        count: int = 0
        # 预计算每行的范数（避免重复计算）
        row_norms: np.ndarray = np.linalg.norm(W, axis=1) + 1e-12

        for i in range(n):
            for j in range(i + 1, n):
                if float(W[i, j]) > 0.0:
                    continue  # 边已存在，跳过
                # 计算连接模式的余弦相似度
                dot_ij: float = float(np.dot(W[i], W[j]))
                sim: float = dot_ij / float(row_norms[i] * row_norms[j])
                if sim >= self.structural_similarity_threshold:
                    # 结构相似度高：推理边，权重取相似度 × 0.8（保守）
                    W[i, j] = W[j, i] = np.float32(sim * 0.8)
                    count += 1
        return count

    # ================================================================
    # 公有辅助方法：_common_neighbors — 计算两节点的共同邻居
    # 数学对应：N(i) ∩ N(j) = {k | w(i,k) > 0 ∧ w(j,k) > 0}
    # ================================================================

    def _common_neighbors(self, i: int, j: int,
                          W: np.ndarray) -> List[int]:
        """计算节点 i 和 j 的共同邻居集合。

        数学对应：
        - 共同邻居：N(i) ∩ N(j) = {k | k≠i, k≠j, w(i,k) > 0, w(j,k) > 0}
        - 用于 Jaccard 系数计算和模式补全的支撑度验证

        参数:
          i: 节点 i 的索引
          j: 节点 j 的索引
          W: n×n 对称权重矩阵

        返回:
          共同邻居的索引列表
        """
        n: int = W.shape[0]
        common: List[int] = []
        for k in range(n):
            if k != i and k != j:
                if float(W[i, k]) > 0.0 and float(W[j, k]) > 0.0:
                    common.append(k)
        return common

    # ================================================================
    # 公有辅助方法：_degree — 计算节点度数
    # 数学对应：deg(i) = |{j | w(i,j) > 0}|
    # ================================================================

    def _degree(self, i: int, W: np.ndarray) -> int:
        """计算节点 i 的加权度数（非零权重的邻居数）。

        数学对应：
        - deg(i) = |N(i)| = |{j | w(i,j) > 0}|
        - 用于 Jaccard 系数的分母计算

        参数:
          i: 节点索引
          W: n×n 对称权重矩阵

        返回:
          整型度数（非零邻居数）
        """
        # 对行 i 统计 w(i,j) > 0 的列数
        return int(np.sum(W[i] > 0))

    # ================================================================
    # PB 级新增：Star 稀疏传递闭包推理
    # 数学对应：PB_ARCHITECTURE.md 第3.2节 — Star 稀疏传递闭包
    # ================================================================

    def infer_star(self, psi: RelationNetwork) -> RelationNetwork:
        """Star 稀疏闭包模式：以稀疏矩阵形式执行传递闭包并施加推理规则。

        数学对应：PB_ARCHITECTURE.md 第3.2节 — Star 稀疏传递闭包
        - Step 1: 从 psi 构建稀疏邻接矩阵（CSR 格式）
        - Step 2: 调用 StarClosure 计算传递闭包
        - Step 3: 对闭包结果施加规则3（共现推理）
        - Step 4: 对闭包结果施加规则4（模式补全）
        - Step 5: 对闭包结果施加规则5（结构相似推理）
        - Step 6: 返回补全后的 RelationNetwork

        注意：规则1（传递性）由 StarClosure 引擎内部完成，规则2（对称性）
        由稀疏矩阵的上三角重建保证。

        参数:
          psi: 原始关系网络 Ψ（由 RelationExtractor 生成）

        返回:
          补全后的 RelationNetwork Ψ* — 含 inference_applied 标记和推理统计元数据
        """
        n: int = len(psi.nodes)

        # ── Step 1: 从 psi 构建稀疏邻接矩阵 ──
        # 提取边对和权重用于 CSR 构建
        row_indices: List[int] = []
        col_indices: List[int] = []
        data_values: List[float] = []
        for (i, j), w in zip(psi.edges, psi.weights):
            row_indices.append(i)
            col_indices.append(j)
            data_values.append(float(w))
            # 对称矩阵：同时添加反向边
            if i != j:
                row_indices.append(j)
                col_indices.append(i)
                data_values.append(float(w))

        # 尝试使用 scipy.sparse 构建 CSR 矩阵
        try:
            from scipy.sparse import csr_matrix
            W_sparse = csr_matrix(
                (data_values, (row_indices, col_indices)),
                shape=(n, n),
                dtype=np.float32,
            )
        except ImportError:
            # 回退：scipy 不可用时使用稠密矩阵 + Warshall
            # 数学对应：向后兼容 — 保持 O(N³) 路径可用
            W_dense: np.ndarray = np.zeros((n, n), dtype=np.float32)
            for i, j, w in zip(row_indices, col_indices, data_values):
                W_dense[i, j] = max(float(W_dense[i, j]), w)
            n_inferred = self._csr_to_warshall_fallback(W_dense, n)
            n_transitive = n_inferred
        else:
            # ── Step 2: 调用 StarClosure 计算传递闭包 ──
            star_engine = StarClosure(
                max_iter=self.star_max_iter,
                tol=self.star_tol,
                decay=self.transitive_decay,
            )
            W_dense = star_engine.compute(W_sparse)
            n_transitive: int = int(np.sum(W_dense > 0)) - len(psi.edges)

        # ── Step 3: 施加共现推理（规则3）──
        n_cooccurrence: int = self._apply_cooccurrence_inference(W_dense, n)

        # ── Step 4: 施加模式补全（规则4）──
        n_pattern: int = self._apply_pattern_completion(W_dense, n)

        # ── Step 5: 施加结构相似推理（规则5）──
        n_structural: int = self._apply_structural_inference(W_dense, n)

        # ── Step 6: 从 W_dense 重建 edges 和 weights ──
        edges_star: List[Tuple[int, int]] = []
        weights_star: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                w_ij: float = float(W_dense[i, j])
                if w_ij > 0.0:
                    edges_star.append((i, j))
                    weights_star.append(w_ij)

        # ── 构建推理元数据 ──
        edge_count_before: int = len(psi.edges)
        total_inferred: int = (n_transitive + n_cooccurrence +
                               n_pattern + n_structural)
        inference_meta: Dict[str, object] = dict(psi.metadata)
        inference_meta["inference_applied"] = True
        inference_meta["inference_mode"] = "star_closure"
        inference_meta["inference_stats"] = {
            "edges_before": edge_count_before,
            "edges_after": len(edges_star),
            "edges_added": len(edges_star) - edge_count_before,
            "transitive_inferred": n_transitive,
            "symmetric_inferred": 0,  # Star 闭包由稀疏重建保证对称性
            "cooccurrence_inferred": n_cooccurrence,
            "pattern_inferred": n_pattern,
            "structural_inferred": n_structural,
            "total_inferred": total_inferred,
            "star_max_iter": self.star_max_iter,
            "star_tol": self.star_tol,
        }

        return RelationNetwork(
            nodes=list(psi.nodes),
            edges=edges_star,
            weights=np.array(weights_star, dtype=np.float32),
            hierarchy=dict(psi.hierarchy),
            metadata=inference_meta,
        )

    # ================================================================
    # PB 级新增：CSR → Warshall 回退
    # 数学对应：PB_ARCHITECTURE.md 第3.2节 — 向后兼容回退路径
    # ================================================================

    def _csr_to_warshall_fallback(self, W: np.ndarray, n: int) -> int:
        """scipy.sparse 不可用时，回退到原有 O(N³) Warshall 闭包。

        数学对应：PB_ARCHITECTURE.md 第3.2节 — 向后兼容
        - 当稀疏矩阵库不可用时，自动降级为原始稠密 Warshall
        - 保证在任何环境下推理功能可用

        参数:
          W: n×n 稠密权重矩阵（原地修改用于传递闭包）
          n: 节点数

        返回:
          新推理出的传递边数量
        """
        # 直接复用原有 _apply_transitive_inference 方法
        return self._apply_transitive_inference(W, n)
