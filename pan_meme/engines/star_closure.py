"""Star 传递闭包引擎（稀疏矩阵版）
数学对应：PB_ARCHITECTURE.md 第3.2节 — GraphBLAS Star 算法替代 O(N³) Warshall

核心思想：
  1. Semi-naive 传递闭包：从邻接矩阵 A 出发，迭代计算 Δ = A·A 的非零新边，
     累积到 A 中，直至 Δ 无新边（收敛），复杂度 O(N²·k)，k << N
  2. 增量更新：仅针对新增边传播传递后果，不重算全闭包，适合流式场景
  3. 稀疏矩阵加速：使用 scipy.sparse.csr_matrix 存储邻接矩阵，
     所有矩阵运算在稀疏空间中进行，跳过零元乘法

算法复杂度：
  - 全闭包：O(N² · max_iterations) 最坏，实际 O(N² · k) 仅传播 k 步
  - 增量更新：O(|new_edges| · N) 仅传播新增边的传递链
  - 全 Warshall：O(N³) 对比
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

# scipy 为可选依赖：仅在需要稀疏矩阵时导入
try:
    from scipy.sparse import csr_matrix, lil_matrix, find
    _SCIPY_SPARSE_AVAILABLE: bool = True
except ImportError:
    _SCIPY_SPARSE_AVAILABLE: bool = False
    csr_matrix = None  # type: ignore
    lil_matrix = None  # type: ignore

from pan_meme.core.types import RelationNetwork


# ============================================================
# StarClosure — Semi-naive 传递闭包引擎
# 数学对应：PB_ARCHITECTURE.md 第3.2节
# ============================================================

class StarClosure:
    """Star 传递闭包引擎：用 Semi-naive 算法在稀疏矩阵上计算传递闭包。

    数学对应：PB_ARCHITECTURE.md 第3.2节 — GraphBLAS Star 算法
    - 替代经典 O(N³) Warshall 算法
    - 利用稀疏性降低到 O(N²·k)，k 为实际迭代传播步数
    - 支持增量模式：仅传播新增边的传递后果

    算法（Semi-naive 传递闭包）：
      Δ = A          # Δ: 本轮新增边
      while Δ 有非零元素:
          A = A + Δ   # 累积新边
          Δ = A · A   # 传播一步（矩乘）
          Δ = Δ - nonzero(A)  # 仅保留真正的新边
      结束条件：Δ 全零 或 达到 max_iterations
    """

    def __init__(self, max_iterations: int = 20,
                 convergence_tol: float = 1e-4) -> None:
        """初始化传递闭包参数。

        参数:
          max_iterations: 最大迭代次数，防止不收敛时无限循环，默认 20
          convergence_tol: 收敛容差，当 Δ 中所有值 < tol 时判定收敛，默认 1e-4
        """
        if not _SCIPY_SPARSE_AVAILABLE:
            raise ImportError(
                "StarClosure 需要 scipy 库。请执行: pip install scipy"
            )
        self.max_iterations: int = max_iterations
        self.convergence_tol: float = convergence_tol
        # 内部状态：最后一次计算的统计信息
        self._last_stats: Dict[str, object] = {}

    # ================================================================
    # 核心算法：Semi-naive 传递闭包
    # 数学对应：PB_ARCHITECTURE.md 第3.2节 — Star 算法
    # ================================================================

    def compute_closure(
        self,
        adj_matrix: "csr_matrix",
        edge_weights: Optional[np.ndarray] = None,
    ) -> Tuple["csr_matrix", np.ndarray, dict]:
        """计算稀疏邻接矩阵的传递闭包。

        数学对应：PB_ARCHITECTURE.md 第3.2节
        - 输入：n×n 稀疏邻接矩阵 A（csr_matrix 格式）
        - 输出：传递闭包 A*，其中 A*[i,j] = 1 若存在 i→j 路径

        算法（Semi-naive 传递闭包）：
          Δ = A                                          # (1) 初始 Δ = A
          for iteration in 1..max_iterations:
              A_accum = A + Δ                            # (2) 累积新边
              Δ_new = A_accum · A_accum                  # (3) 传播一步
              Δ_new = Δ_new - nonzero(A_accum)           # (4) 仅保留新边
              A = A_accum
              if Δ_new 中无显著值: break
              Δ = Δ_new

        边权重传播：当 edge_weights 提供时，新边权重 = max-min 路径权重：
          w_new(i,k) = max( w(i,k), min( w(i,j), w(j,k) ) )
          其中 j 是中间节点。这保证了权重单调不增（路径越长权重越小）。

        参数:
          adj_matrix: n×n 稀疏邻接矩阵（csr_matrix 格式），
                      非零元素表示直接边
          edge_weights: 边权重数组 shape=(nnz,)，与 adj_matrix.data 对应。
                        若为 None，则使用 adj_matrix.data 的值作为权重。

        返回:
          (A_closure, weights, stats) —
            A_closure: 传递闭包后的邻接矩阵（csr_matrix）
            weights:   闭包后每条边的权重数组
            stats:     收敛统计 {iterations, edges_added, converged}
        """
        n: int = adj_matrix.shape[0]
        if adj_matrix.shape[1] != n:
            raise ValueError(f"邻接矩阵必须是方阵，实际 shape={adj_matrix.shape}")

        # ── 初始化 ──
        # 用 lil_matrix 方便按索引修改（Delta 阶段），最终转为 csr 加速乘法
        A_lil: "lil_matrix" = adj_matrix.tolil()
        A_csr: "csr_matrix" = adj_matrix.tocsr()

        # 初始化边权重：若未提供则用矩阵值
        if edge_weights is None:
            # 从 csr_matrix.data 获取原始权重
            edge_weights = A_csr.data.copy().astype(np.float64)

        # 构建权重字典：(i,j) → w_ij，便于 O(1) 查询
        rows_A, cols_A = A_csr.nonzero()
        weight_dict: Dict[Tuple[int, int], float] = {}
        for idx in range(len(rows_A)):
            weight_dict[(rows_A[idx], cols_A[idx])] = float(edge_weights[idx])

        edges_before: int = A_csr.nnz
        edges_added_total: int = 0
        converged: bool = False

        # ── Semi-naive 迭代 ──
        # 数学对应：Δ.A 表示当前邻接矩阵未覆盖的新边集合
        # Δ = A; while Δ非零: A+=Δ; Δ=A·A; Δ=Δ-nonzero(A)
        Delta_lil: "lil_matrix" = A_csr.tolil()  # 当前轮的新边

        for iteration in range(1, self.max_iterations + 1):
            # ── (2) 累积新边到 A ──
            # A = A + Δ（在 lil 空间中合并）
            Delta_csr: "csr_matrix" = Delta_lil.tocsr()
            rows_delta, cols_delta = Delta_csr.nonzero()
            data_delta = Delta_csr.data

            new_edges_this_round: int = 0
            for idx in range(len(rows_delta)):
                i: int = rows_delta[idx]
                j: int = cols_delta[idx]
                w_new: float = float(data_delta[idx])
                if w_new < self.convergence_tol:
                    continue
                key: Tuple[int, int] = (i, j)
                if key not in weight_dict:
                    # 新边：添加
                    weight_dict[key] = w_new
                    A_lil[i, j] = w_new
                    new_edges_this_round += 1
                else:
                    # 已存在：取 max 权重
                    w_old: float = weight_dict[key]
                    if w_new > w_old:
                        weight_dict[key] = w_new
                        A_lil[i, j] = w_new

            edges_added_total += new_edges_this_round

            # ── 若本轮无新边，收敛 ──
            if new_edges_this_round == 0:
                converged = True
                break

            # ── (3) Δ = A · A（传播一步） ──
            A_csr = A_lil.tocsr()
            # 稀疏矩阵乘法：A²[i,k] = max_j min( A[i,j], A[j,k] )
            # 数学对应：路径权重的 max-min 组合 — 传递闭包的半环结构
            Delta_next_csr: "csr_matrix" = self._sparse_maxmin_multiply(A_csr, A_csr)

            # ── (4) Δ = Δ - nonzero(A) → 仅保留真正新边 ──
            # 将 Δ_next 减去 A_csr 中已存在的边
            Delta_new_lil: "lil_matrix" = Delta_next_csr.tolil()
            rows_A_existing, cols_A_existing = A_csr.nonzero()
            for idx in range(len(rows_A_existing)):
                i_e: int = rows_A_existing[idx]
                j_e: int = cols_A_existing[idx]
                # 若 A 中已有该边，从 Δ 中移除
                if (i_e, j_e) in weight_dict:
                    Delta_new_lil[i_e, j_e] = 0.0

            Delta_lil = Delta_new_lil

        # ── 构建最终输出 ──
        A_closure: "csr_matrix" = A_lil.tocsr()
        edges_after: int = A_closure.nnz

        # 构建与 A_closure.data 一致的权重数组
        weights_final: np.ndarray = A_closure.data.astype(np.float64)

        # ── 收敛统计 ──
        stats: dict = {
            "iterations": iteration,
            "edges_before": edges_before,
            "edges_after": edges_after,
            "edges_added": edges_added_total,
            "converged": converged,
            "convergence_tol": self.convergence_tol,
        }
        self._last_stats = stats

        return A_closure, weights_final, stats

    # ================================================================
    # 增量更新：仅传播新增边的传递后果
    # 数学对应：PB_ARCHITECTURE.md 第3.2节 — 增量模式
    # ================================================================

    def incremental_update(
        self,
        A: np.ndarray,
        new_edges: List[Tuple[int, int, int]],
        new_weights: List[float],
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """增量更新传递闭包：仅传播新增边的传递后果，不重算全闭包。

        数学对应：PB_ARCHITECTURE.md 第3.2节 — 增量 Star 算法
        - 输入：已计算闭包的邻接矩阵 A 和新增边列表
        - 输出：更新后的闭包 A'

        算法：
        1. 将 new_edges 插入 A 得到 A_plus
        2. 仅对新增边计算 Δ = new_edges · A_plus（正向传播）
        3. 再计算 Δ' = A_plus · Δ（逆向传播）
        4. 将 Δ 和 Δ' 合并到 A

        复杂度：O(|new_edges| · N) vs 全重算的 O(N²·k)

        参数:
          A:          当前 n×n 邻接矩阵（稠密 numpy 数组）
          new_edges:  新增边列表 [(i, j, w_ij), ...]
          new_weights: 新增边的权重列表

        返回:
          (A_updated, weights_updated, stats) —
            A_updated: 更新后的邻接矩阵
            weights_updated: 更新后的边权重数组
            stats: 增量统计 {new_edges_count, edges_propagated, converged}
        """
        n: int = A.shape[0]
        # ── 转为稀疏格式以加速 ──
        A_csr: "csr_matrix" = csr_matrix(A)

        # ── 构建权重字典 ──
        weight_dict: Dict[Tuple[int, int], float] = {}
        rows_existing, cols_existing = A_csr.nonzero()
        for idx in range(len(rows_existing)):
            weight_dict[(rows_existing[idx], cols_existing[idx])] = float(
                A_csr.data[idx]
            )

        # ── 插入新增边 ──
        A_lil: "lil_matrix" = A_csr.tolil()
        new_edges_count: int = 0
        newly_added: List[Tuple[int, int, float]] = []
        for edge, w in zip(new_edges, new_weights):
            i, j, _ = edge
            key: Tuple[int, int] = (i, j)
            if key not in weight_dict or w > weight_dict[key]:
                weight_dict[key] = w
                A_lil[i, j] = w
                newly_added.append((i, j, w))
                new_edges_count += 1

        # ── 传播新增边的传递后果 ──
        # 数学对应：传递闭包的增量传播
        edges_propagated: int = 0
        A_updated_csr: "csr_matrix" = A_lil.tocsr()

        # 对每条新增边 (i,j)，寻找所有 k 使得存在 (j,k)，则 (i,k) 为新路径
        # 正向传播：i → j（新增）→ k（已存在）
        # 反向传播：p → i（已存在）→ j（新增）
        for (i, j, w_ij) in newly_added:
            # ── 正向传播：i → j → k ──
            # 从 A_updated_csr 第 j 行获取所有出边
            row_j_start: int = A_updated_csr.indptr[j]
            row_j_end: int = A_updated_csr.indptr[j + 1]
            for ptr in range(row_j_start, row_j_end):
                k: int = A_updated_csr.indices[ptr]
                w_jk: float = float(A_updated_csr.data[ptr])
                # 路径权重 = min(w_ij, w_jk)
                w_ik: float = min(w_ij, w_jk)
                key_ik: Tuple[int, int] = (i, k)
                if key_ik != (i, j) and (
                    key_ik not in weight_dict or w_ik > weight_dict[key_ik]
                ):
                    weight_dict[key_ik] = w_ik
                    A_lil[i, k] = w_ik
                    edges_propagated += 1

            # ── 反向传播：p → i → j ──
            # 从 A_updated_csr 第 i 列获取所有入边
            rows_all, cols_all = A_updated_csr.nonzero()
            for idx in range(len(rows_all)):
                p: int = rows_all[idx]
                c: int = cols_all[idx]
                if c == i:
                    w_pi: float = float(A_updated_csr.data[idx])
                    w_pj: float = min(w_pi, w_ij)
                    key_pj: Tuple[int, int] = (p, j)
                    if key_pj != (i, j) and (
                        key_pj not in weight_dict or w_pj > weight_dict[key_pj]
                    ):
                        weight_dict[key_pj] = w_pj
                        A_lil[p, j] = w_pj
                        edges_propagated += 1

        # ── 构建输出 ──
        A_final_csr: "csr_matrix" = A_lil.tocsr()
        A_updated: np.ndarray = A_final_csr.toarray()
        weights_updated: np.ndarray = A_final_csr.data.astype(np.float64)

        stats: dict = {
            "new_edges_count": new_edges_count,
            "edges_propagated": edges_propagated,
            "converged": True,  # 增量模式总是单轮传播
        }

        return A_updated, weights_updated, stats

    # ================================================================
    # 从 RelationNetwork 构建稀疏矩阵
    # 数学对应：将 Ψ = (V, E, w) 转换为 n×n 邻接矩阵
    # ================================================================

    def from_relation_network(
        self, psi: RelationNetwork
    ) -> Tuple["csr_matrix", np.ndarray]:
        """从 RelationNetwork 构建稀疏邻接矩阵和边权重。

        数学对应：
        - 定义1：Ψ = (V, E, w) → 邻接矩阵 A ∈ {0,1}^{n×n}
        - A[i,j] = w_ij 若 (i,j) ∈ E，否则 0
        - 对称性：若 (i,j) 有边，则 A[i,j] = A[j,i] = w_ij

        参数:
          psi: RelationNetwork 关系网络

        返回:
          (A_csr, weights) — 稀疏邻接矩阵和边权重数组
        """
        n: int = len(psi.nodes)
        edges: List[Tuple[int, int]] = psi.edges
        weights: np.ndarray = psi.weights

        # ── 构建 COO 格式数据 ──
        row_indices: List[int] = []
        col_indices: List[int] = []
        data_values: List[float] = []

        for idx, (i, j) in enumerate(edges):
            w: float = float(weights[idx]) if idx < len(weights) else 1.0
            # 对称添加
            row_indices.append(i)
            col_indices.append(j)
            data_values.append(w)
            if i != j:
                row_indices.append(j)
                col_indices.append(i)
                data_values.append(w)

        A_csr: "csr_matrix" = csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(n, n),
            dtype=np.float64,
        )

        return A_csr, A_csr.data

    # ================================================================
    # 将传递闭包结果转回 RelationNetwork
    # 数学对应：逆映射 — A* → Ψ*
    # ================================================================

    def to_relation_network(
        self,
        A: "csr_matrix",
        weights: np.ndarray,
        psi: RelationNetwork,
    ) -> RelationNetwork:
        """将传递闭包结果转回 RelationNetwork 格式。

        数学对应：
        - A* 是传递闭包后的邻接矩阵
        - 提取 A* 中所有非零元素作为边集 E*
        - 权重来自 weights 数组
        - 保留原始 psi 的 hierarchy 元数据，更新 metadata 标记为闭包结果

        参数:
          A:       传递闭包后的邻接矩阵（csr_matrix）
          weights: 边权重数组
          psi:     原始 RelationNetwork（用于继承 hierarchy 元数据）

        返回:
          新的 RelationNetwork，包含闭包后的边集
        """
        n: int = A.shape[0]

        # ── 提取非零元素 ──
        edges_new: List[Tuple[int, int]] = []
        weights_new: List[float] = []

        # 只取上三角（避免重复），因为无向图中 A 对称
        A_lil: "lil_matrix" = A.tolil()
        for i in range(n):
            row = A_lil.rows[i]
            row_data = A_lil.data[i]
            for idx, j in enumerate(row):
                if i < j:
                    w_val: float = float(row_data[idx])
                    if w_val >= self.convergence_tol:
                        edges_new.append((i, j))
                        weights_new.append(w_val)

        # ── 继承 hierarchy，更新 metadata ──
        hierarchy_new: Dict[str, object] = dict(psi.hierarchy) if psi.hierarchy else {}
        hierarchy_new["closure_applied"] = True

        metadata_new: Dict[str, object] = dict(psi.metadata) if psi.metadata else {}
        metadata_new["node_count"] = n
        metadata_new["edge_count"] = len(edges_new)
        metadata_new["closure_applied"] = True
        metadata_new["closure_stats"] = self._last_stats

        return RelationNetwork(
            nodes=psi.nodes,
            edges=edges_new,
            weights=np.array(weights_new, dtype=np.float32),
            hierarchy=hierarchy_new,
            metadata=metadata_new,
        )

    # ================================================================
    # 内部方法：稀疏 max-min 矩阵乘法
    # 数学对应：传递闭包的半环结构 — (max, min) 半环上的矩阵乘法
    # ================================================================

    def _sparse_maxmin_multiply(
        self, A: "csr_matrix", B: "csr_matrix"
    ) -> "csr_matrix":
        """稀疏 max-min 矩阵乘法：C[i,k] = max_j min(A[i,j], B[j,k])。

        数学对应：
        - 标准矩阵乘：C[i,k] = Σ_j A[i,j] · B[j,k]
        - max-min 半环乘：C[i,k] = max_j min(A[i,j], B[j,k])
        - 这保证了传递闭包中路径权重为瓶颈权重（最弱链的强度）

        参数:
          A: n×n 稀疏矩阵
          B: n×n 稀疏矩阵

        返回:
          C = A ⊗ B（max-min 半环乘法结果）
        """
        n: int = A.shape[0]
        C: "lil_matrix" = lil_matrix((n, n), dtype=np.float64)

        # 对每个 i，找出所有 (i,j) 边，然后对每个 j 找 (j,k) 边
        A_csr = A.tocsr()
        B_csr = B.tocsr()

        for i in range(n):
            # A[i,*] 的非零列
            a_row_start: int = A_csr.indptr[i]
            a_row_end: int = A_csr.indptr[i + 1]
            if a_row_start == a_row_end:
                continue

            for a_ptr in range(a_row_start, a_row_end):
                j: int = A_csr.indices[a_ptr]
                a_val: float = float(A_csr.data[a_ptr])
                if a_val < self.convergence_tol:
                    continue

                # B[j,*] 的非零列
                b_row_start: int = B_csr.indptr[j]
                b_row_end: int = B_csr.indptr[j + 1]
                for b_ptr in range(b_row_start, b_row_end):
                    k: int = B_csr.indices[b_ptr]
                    b_val: float = float(B_csr.data[b_ptr])
                    if b_val < self.convergence_tol:
                        continue

                    # max-min：min(A[i,j], B[j,k])，然后取 max over j
                    candidate: float = min(a_val, b_val)
                    existing: float = float(C[i, k])
                    if candidate > existing and candidate >= self.convergence_tol:
                        C[i, k] = candidate

        return C.tocsr()
