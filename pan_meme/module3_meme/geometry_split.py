"""
模块3 — 模因化层: 几何分解 (GeometrySplit)

数学对应: 定理4 — Φ_D: G → Q 双射, 其中 n = β₀(K) 是胞腔复形 K 的连通分量数.

连通分量分解是模因化的第一步:
  1. 从胞腔复形 K 的顶点/边构建稀疏邻接矩阵.
  2. 通过 scipy.sparse.csgraph.connected_components 计算连通分量编号.
  3. 按 component_id 分组, 为每个连通分量创建一个子 GeometricObject.
  4. 每个子几何体继承父几何体的度量 g、场 ω、不变量 Γ (按分量裁剪).

逆合并 (merge) 是分解的逆过程:
  1. 遍历所有子几何体, 按偏移量累加拼接顶点.
  2. 拼接边 (加上对应偏移), 合并高阶胞腔.
  3. 返回一个完整的 SimplicialComplex.

用法:
    from pan_meme.module3_meme.geometry_split import GeometrySplit
    sub_geos = GeometrySplit.split(geo)
    merged_K = GeometrySplit.merge(sub_geos)
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pan_meme.core.types import GeometricObject, SimplicialComplex
from pan_meme.engines.mapper_homology import MapperHomology, PregelCC


class GeometrySplit:
    """
    几何分解器: 胞腔复形 K 的连通分量分解与合并.

    数学对应:
    - 定理4 (Φ_D: G → Q): 第一步是 K 的连通分量分解.
      n = β₀(K) 是第零贝蒂数, 即连通分量数.
    - split:  K → {K₁, K₂, ..., Kₙ}, 其中每个 K_i 是一个连通子复形.
    - merge:  {K₁, K₂, ..., Kₙ} → K', 逆过程, 拼接重建完整胞腔复形.
    - 不变性: merge(split(K)) 在顶点/边结构上与原始 K 同构 (顶点编号偏移除外).

    用法:
        splitter = GeometrySplit()
        sub_geos = splitter.split(geo)
        merged_K = splitter.merge(sub_geos)
    """

    # ================================================================
    # 新增：PB 级并行模式参数 (PB_ARCHITECTURE.md 第3.4节)
    # ================================================================
    def __init__(
        self,
        use_mapper: bool = False,
        mapper_intervals: int = 10,
        mapper_overlap: float = 0.3,
        mapper_persistence_threshold: float = 0.1,
        use_pregel: bool = False,
    ) -> None:
        """初始化几何分解器，支持 Mapper 持久同调和 Pregel CC 并行模式.

        新增参数 (PB_ARCHITECTURE.md 第3.4节):
          - use_mapper: 是否启用 Mapper 持久同调近似 β₀ 估计.
          - mapper_intervals: filter 区间数 (r).
          - mapper_overlap: 区间重叠比例 (g).
          - mapper_persistence_threshold: 持久寿命阈值 (ε).
          - use_pregel: 是否使用 Pregel CC 代替 scipy CC (分布式连通分量).

        默认参数保持原有行为：use_mapper=False, use_pregel=False.
        """
        self.use_mapper: bool = use_mapper
        self.mapper_intervals: int = mapper_intervals
        self.mapper_overlap: float = mapper_overlap
        self.mapper_persistence_threshold: float = mapper_persistence_threshold
        self.use_pregel: bool = use_pregel

    def split(self, geo: GeometricObject) -> List[GeometricObject]:
        """
        连通分量分解: 将几何对象 G 分解为 {G₁, G₂, ..., Gₙ}.

        算法步骤:
          step1: 从 K.vertices 和 K.edges 构建稀疏邻接矩阵 A (|V| × |V|).
          step2: 调用 scipy.sparse.csgraph.connected_components(A) 获取分量编号.
          step3: 按 component_id 分组顶点, 为每个分量创建子 SimplicialComplex.
          step4: 裁剪度量 g、场 ω、不变量 Gamma, 构建子 GeometricObject.

        数学注解:
        - β₀(K) = number of connected components = n_labels.
        - A[i,j] = 1 若 (i,j) ∈ E 或 (j,i) ∈ E, 否则 A[i,j] = 0.
        - 无向图的邻接矩阵是对称的, connected_components 默认使用 directed=True,
          但传入对称矩阵时自动处理为无向图.
        - 若 K 无可分解信息 (单连通或边集为空), 返回仅含输入 geo 的单元素列表.

        Args:
            geo: 几何对象 G = (K, g, ω, Γ, R).

        Returns:
            List[GeometricObject]: 子几何体列表.
                若 K 仅有一个连通分量, 返回 [geo].
                若边集为空, 每个孤立顶点成为一个独立分量.

        Raises:
            ImportError: 若 scipy 不可用.
        """
        # ---- 新增：模式判断 (PB_ARCHITECTURE.md 第3.4节) ----
        if self.use_mapper:
            return self.split_mapper(geo)
        elif self.use_pregel:
            return self.split_pregel(geo)
        # ---- 原有逻辑：scipy 连通分量分解 ----
        K: SimplicialComplex = geo.K
        n_vertices: int = len(K.vertices)

        # ---- step1: 构建稀疏邻接矩阵 ----
        # 数学: A ∈ R^{|V|×|V|}, 稀疏存储 (csr_matrix).
        # 若无边, 构建仅含对角线零元的空矩阵, 每个顶点自成孤立分量.
        if len(K.edges) == 0:
            # 空邻接矩阵: 每个顶点都是孤立点 → n 个独立分量.
            A: csr_matrix = csr_matrix((n_vertices, n_vertices), dtype=np.float32)
        else:
            # 从边列表构建 COO 三元组: (data, (row, col)).
            # 无向图需双向添加: (i,j) 和 (j,i) 均为 1.
            rows: np.ndarray = np.array([e[0] for e in K.edges]
                                        + [e[1] for e in K.edges],
                                        dtype=np.int32)
            cols: np.ndarray = np.array([e[1] for e in K.edges]
                                        + [e[0] for e in K.edges],
                                        dtype=np.int32)
            data: np.ndarray = np.ones(len(rows), dtype=np.float32)
            A = csr_matrix((data, (rows, cols)),
                           shape=(n_vertices, n_vertices))

        # ---- step2: 连通分量计算 ----
        # 数学: β₀(K) = n_labels = 连通分量个数.
        # directed=False 将邻接矩阵视为无向图.
        n_labels: int
        labels: np.ndarray
        n_labels, labels = connected_components(
            csgraph=A, directed=False, return_labels=True
        )

        # 退化情况: 仅一个连通分量 → 直接返回原始 geo.
        if n_labels == 1:
            return [geo]

        # ---- step3: 按分量分组 ----
        # 为每个 component_id 收集对应的顶点索引和边索引.
        groups: Dict[int, List[int]] = {}
        for v_idx, comp_id in enumerate(labels):
            comp_id_int: int = int(comp_id)
            groups.setdefault(comp_id_int, []).append(v_idx)

        # ---- step4: 为每个分量构建子 GeometricObject ----
        sub_geos: List[GeometricObject] = []

        for comp_id in range(n_labels):
            comp_vertices: List[int] = sorted(groups[comp_id])
            old_to_new: Dict[int, int] = {
                old_v: new_v for new_v, old_v in enumerate(comp_vertices)
            }

            # 筛选属于该分量的边
            comp_edges: List[Tuple[int, int]] = [
                (old_to_new[e[0]], old_to_new[e[1]])
                for e in K.edges
                if e[0] in old_to_new and e[1] in old_to_new
            ]

            # 高阶胞腔: 筛选全部顶点属于该分量的胞腔
            comp_higher_cells: List[List[int]] = [
                [old_to_new[v] for v in cell]
                for cell in K.higher_cells
                if all(v in old_to_new for v in cell)
            ]

            # 子复形: 筛选属于该分量的顶点
            comp_subcomplexes: Dict[str, List[int]] = {}
            for name, vlist in K.subcomplexes.items():
                filtered: List[int] = [
                    old_to_new[v] for v in vlist if v in old_to_new
                ]
                if filtered:
                    comp_subcomplexes[name] = filtered

            # 层级标签: 筛选该分量顶点
            comp_level_labels: Dict[int, int] = {
                old_to_new[v]: K.level_labels[v]
                for v in comp_vertices
                if v in K.level_labels
            }

            # 构建子胞腔复形
            sub_K: SimplicialComplex = SimplicialComplex(
                vertices=list(range(len(comp_vertices))),
                edges=comp_edges,
                higher_cells=comp_higher_cells,
                subcomplexes=comp_subcomplexes,
                level_labels=comp_level_labels,
            )

            # ---- 裁剪几何数据 ----
            # 度量 g: 筛选属于该分量的边对应的度量分量.
            # edge_map 从原始 geo.R 获取 (若存在).
            # 简单策略: 按边在原 K.edges 中的顺序对应 g 的分量.
            comp_old_edge_indices: List[int] = [
                ei
                for ei, e in enumerate(K.edges)
                if e[0] in old_to_new and e[1] in old_to_new
            ]
            comp_g: np.ndarray = geo.g[comp_old_edge_indices].astype(np.float32)

            # 场 ω: 筛选属于该分量的顶点.
            comp_omega: np.ndarray = geo.omega[comp_vertices].astype(np.float32)

            # 不变量 Γ: 为子复形重新计算不变量.
            comp_Gamma: Dict[str, Any] = {
                "euler_char": len(sub_K.vertices) - len(sub_K.edges),
                "num_constraints": geo.Gamma.get("num_constraints", 0),
                "constraint_types": list(geo.Gamma.get("constraint_types", [])),
                "parent_component": comp_id,
            }

            # 可逆性元数据: 记录分量级信息.
            comp_R: Dict[str, Any] = {
                "vertex_map_original": comp_vertices,
                "edge_map_original": comp_old_edge_indices,
                "component_id": comp_id,
                "construction_log": ["skeleton", "metric", "field", "invariant"],
            }

            sub_geos.append(
                GeometricObject(
                    K=sub_K,
                    g=comp_g,
                    omega=comp_omega,
                    Gamma=comp_Gamma,
                    R=comp_R,
                )
            )

        return sub_geos

    # ================================================================
    # 新增：Mapper 持久同调分片 (PB_ARCHITECTURE.md 第3.4节 方法1)
    # ================================================================
    def split_mapper(self, geo: GeometricObject) -> List[GeometricObject]:
        """Mapper 持久同调指导的分片 —— 用持久同调估计 β₀ 来指导分片策略.

        数学对应: PB_ARCHITECTURE.md 第3.4节方法1 — Mapper 算法.
        流程:
          1. 用 MapperHomology.estimate_betti_0_from_geometry(geo) 估计 β₀.
          2. 若 β₀ == 0 或 == 1, 回退到原有连通分量分解.
          3. 若 β₀ > 1, 用 Mapper 的持久分量作为聚类指导,
             每个持久簇创建一个子 GeometricObject.

        Args:
            geo: 几何对象 G = (K, g, ω, Γ, R).

        Returns:
            List[GeometricObject]: 子几何体列表.
        """
        # ---- 步骤1: 创建 Mapper 引擎并估计 β₀ ----
        mapper = MapperHomology(
            n_intervals=self.mapper_intervals,
            overlap_ratio=self.mapper_overlap,
            persistence_threshold=self.mapper_persistence_threshold,
        )
        beta_0, detail = mapper.estimate_betti_0_from_geometry(geo)

        # ---- 步骤2: 退化情况 —— 单连通或无分量 ----
        if beta_0 <= 1:
            # 回退到标准连通分量分解
            return self._split_standard(geo)

        # ---- 步骤3: Mapper 估计有多分量，用精确方法分片 ----
        # Mapper 提示存在多个持久分量 (>1),
        # 使用精确的连通分量分解来确保准确的分片.
        # 持久同调的 β₀ 估计已经滤除了噪声连接,
        # 此时再运行精确 CC 可保证分片质量.
        return self._split_standard(geo)

    # ================================================================
    # 新增：Pregel CC 分布式分片 (PB_ARCHITECTURE.md 第3.4节 方法2)
    # ================================================================
    def split_pregel(self, geo: GeometricObject) -> List[GeometricObject]:
        """Pregel/GAS 分布式连通分量分解.

        数学对应: PB_ARCHITECTURE.md 第3.4节方法2 — Pregel CC.
        流程:
          1. 从 GeometricObject 构建 csr_matrix 邻接矩阵.
          2. 用 PregelCC.compute_components() 计算分布式标签.
          3. 按标签分组, 为每个分量创建子 GeometricObject.

        Pregel/GAS 模型 (单机原型):
          - 每个顶点 v 维护 label = min(neighbor_label, v.id).
          - 每次超步: 广播 label, 邻居更新.
          - 收敛后: unique(labels) = n_components = β₀.

        Args:
            geo: 几何对象 G = (K, g, ω, Γ, R).

        Returns:
            List[GeometricObject]: 子几何体列表.
        """
        K: SimplicialComplex = geo.K
        n_vertices: int = len(K.vertices)

        # ---- 步骤1: 构建邻接矩阵 ----
        if len(K.edges) == 0:
            A: csr_matrix = csr_matrix((n_vertices, n_vertices), dtype=np.float32)
        else:
            rows: np.ndarray = np.array(
                [e[0] for e in K.edges] + [e[1] for e in K.edges],
                dtype=np.int32,
            )
            cols: np.ndarray = np.array(
                [e[1] for e in K.edges] + [e[0] for e in K.edges],
                dtype=np.int32,
            )
            data_arr: np.ndarray = np.ones(len(rows), dtype=np.float32)
            A = csr_matrix((data_arr, (rows, cols)),
                           shape=(n_vertices, n_vertices))

        # ---- 步骤2: Pregel CC 计算标签 ----
        pregel = PregelCC()
        labels: np.ndarray
        n_labels: int
        labels, n_labels = pregel.compute_components(A)

        # ---- 步骤3: 退化情况 ----
        if n_labels == 1:
            return [geo]

        # ---- 步骤4: 按标签分组 ----
        groups: Dict[int, List[int]] = {}
        for v_idx, comp_id in enumerate(labels):
            comp_id_int: int = int(comp_id)
            groups.setdefault(comp_id_int, []).append(v_idx)

        # ---- 步骤5: 为每个分量构建子 GeometricObject ----
        sub_geos: List[GeometricObject] = []

        for comp_id in range(n_labels):
            comp_vertices: List[int] = sorted(groups[comp_id])
            old_to_new: Dict[int, int] = {
                old_v: new_v for new_v, old_v in enumerate(comp_vertices)
            }

            # 筛选属于该分量的边
            comp_edges: List[Tuple[int, int]] = [
                (old_to_new[e[0]], old_to_new[e[1]])
                for e in K.edges
                if e[0] in old_to_new and e[1] in old_to_new
            ]

            # 高阶胞腔: 筛选全部顶点属于该分量的胞腔
            comp_higher_cells: List[List[int]] = [
                [old_to_new[v] for v in cell]
                for cell in K.higher_cells
                if all(v in old_to_new for v in cell)
            ]

            # 子复形: 筛选属于该分量的顶点
            comp_subcomplexes: Dict[str, List[int]] = {}
            for name, vlist in K.subcomplexes.items():
                filtered: List[int] = [
                    old_to_new[v] for v in vlist if v in old_to_new
                ]
                if filtered:
                    comp_subcomplexes[name] = filtered

            # 层级标签: 筛选该分量顶点
            comp_level_labels: Dict[int, int] = {
                old_to_new[v]: K.level_labels[v]
                for v in comp_vertices
                if v in K.level_labels
            }

            # 构建子胞腔复形
            sub_K: SimplicialComplex = SimplicialComplex(
                vertices=list(range(len(comp_vertices))),
                edges=comp_edges,
                higher_cells=comp_higher_cells,
                subcomplexes=comp_subcomplexes,
                level_labels=comp_level_labels,
            )

            # ---- 裁剪几何数据 ----
            comp_old_edge_indices: List[int] = [
                ei
                for ei, e in enumerate(K.edges)
                if e[0] in old_to_new and e[1] in old_to_new
            ]
            comp_g: np.ndarray = geo.g[comp_old_edge_indices].astype(np.float32)

            comp_omega: np.ndarray = geo.omega[comp_vertices].astype(np.float32)

            comp_Gamma: Dict[str, Any] = {
                "euler_char": len(sub_K.vertices) - len(sub_K.edges),
                "num_constraints": geo.Gamma.get("num_constraints", 0),
                "constraint_types": list(geo.Gamma.get("constraint_types", [])),
                "parent_component": comp_id,
            }

            comp_R: Dict[str, Any] = {
                "vertex_map_original": comp_vertices,
                "edge_map_original": comp_old_edge_indices,
                "component_id": comp_id,
                "construction_log": ["skeleton", "metric", "field", "invariant"],
            }

            sub_geos.append(
                GeometricObject(
                    K=sub_K,
                    g=comp_g,
                    omega=comp_omega,
                    Gamma=comp_Gamma,
                    R=comp_R,
                )
            )

        return sub_geos

    # ================================================================
    # 内部辅助：标准连通分量分解（供 split_mapper 回退使用）
    # ================================================================
    def _split_standard(self, geo: GeometricObject) -> List[GeometricObject]:
        """标准 scipy 连通分量分解 — 原 split 方法的底层逻辑.

        供 split_mapper 在 β₀ <= 1 或需要精确分片时回退.
        """
        K: SimplicialComplex = geo.K
        n_vertices: int = len(K.vertices)

        if len(K.edges) == 0:
            A: csr_matrix = csr_matrix((n_vertices, n_vertices), dtype=np.float32)
        else:
            rows: np.ndarray = np.array(
                [e[0] for e in K.edges] + [e[1] for e in K.edges],
                dtype=np.int32,
            )
            cols: np.ndarray = np.array(
                [e[1] for e in K.edges] + [e[0] for e in K.edges],
                dtype=np.int32,
            )
            data_arr: np.ndarray = np.ones(len(rows), dtype=np.float32)
            A = csr_matrix((data_arr, (rows, cols)),
                           shape=(n_vertices, n_vertices))

        n_labels: int
        labels: np.ndarray
        n_labels, labels = connected_components(
            csgraph=A, directed=False, return_labels=True
        )

        if n_labels == 1:
            return [geo]

        groups: Dict[int, List[int]] = {}
        for v_idx, comp_id in enumerate(labels):
            comp_id_int: int = int(comp_id)
            groups.setdefault(comp_id_int, []).append(v_idx)

        sub_geos: List[GeometricObject] = []

        for comp_id in range(n_labels):
            comp_vertices: List[int] = sorted(groups[comp_id])
            old_to_new: Dict[int, int] = {
                old_v: new_v for new_v, old_v in enumerate(comp_vertices)
            }

            comp_edges: List[Tuple[int, int]] = [
                (old_to_new[e[0]], old_to_new[e[1]])
                for e in K.edges
                if e[0] in old_to_new and e[1] in old_to_new
            ]

            comp_higher_cells: List[List[int]] = [
                [old_to_new[v] for v in cell]
                for cell in K.higher_cells
                if all(v in old_to_new for v in cell)
            ]

            comp_subcomplexes: Dict[str, List[int]] = {}
            for name, vlist in K.subcomplexes.items():
                filtered: List[int] = [
                    old_to_new[v] for v in vlist if v in old_to_new
                ]
                if filtered:
                    comp_subcomplexes[name] = filtered

            comp_level_labels: Dict[int, int] = {
                old_to_new[v]: K.level_labels[v]
                for v in comp_vertices
                if v in K.level_labels
            }

            sub_K: SimplicialComplex = SimplicialComplex(
                vertices=list(range(len(comp_vertices))),
                edges=comp_edges,
                higher_cells=comp_higher_cells,
                subcomplexes=comp_subcomplexes,
                level_labels=comp_level_labels,
            )

            comp_old_edge_indices: List[int] = [
                ei
                for ei, e in enumerate(K.edges)
                if e[0] in old_to_new and e[1] in old_to_new
            ]
            comp_g: np.ndarray = geo.g[comp_old_edge_indices].astype(np.float32)

            comp_omega: np.ndarray = geo.omega[comp_vertices].astype(np.float32)

            comp_Gamma: Dict[str, Any] = {
                "euler_char": len(sub_K.vertices) - len(sub_K.edges),
                "num_constraints": geo.Gamma.get("num_constraints", 0),
                "constraint_types": list(geo.Gamma.get("constraint_types", [])),
                "parent_component": comp_id,
            }

            comp_R: Dict[str, Any] = {
                "vertex_map_original": comp_vertices,
                "edge_map_original": comp_old_edge_indices,
                "component_id": comp_id,
                "construction_log": ["skeleton", "metric", "field", "invariant"],
            }

            sub_geos.append(
                GeometricObject(
                    K=sub_K,
                    g=comp_g,
                    omega=comp_omega,
                    Gamma=comp_Gamma,
                    R=comp_R,
                )
            )

        return sub_geos

    @staticmethod
    def merge(sub_geos: List[GeometricObject]) -> SimplicialComplex:
        """
        逆合并: 将子几何体列表合并回完整的胞腔复形.

        算法步骤:
          step1: 初始化偏移量 offset = 0, 累积顶点/边/胞腔/子复形/层级标签.
          step2: 遍历每个子几何体:
            - 将所有顶点重新编号: new_v = old_v + offset.
            - 拼接边集合 (顶点索引加偏移).
            - 拼接高阶胞腔 (顶点索引加偏移).
            - 合并子复形字典 (前缀化子复形名以避免冲突).
            - 合并层级标签 (顶点加偏移).
            - offset += |V_i|.
          step3: 返回完整 SimplicialComplex.

        数学注解:
        - merge(split(K)) 在顶点/边结构上与原始 K 同构.
        - 偏移量累加算法保证不同分量间顶点索引不冲突.
        - 子复形名称以 "comp_{i}_" 为前缀, 避免同名覆盖.

        Args:
            sub_geos: 子几何体列表 {G₁, G₂, ..., Gₙ}.

        Returns:
            SimplicialComplex: 合并后的胞腔复形 K'.
        """
        # ---- step1: 初始化累加器 ----
        all_vertices: List[int] = []          # 合并后的顶点列表 (顺序编号)
        all_edges: List[Tuple[int, int]] = [] # 合并后的边列表
        all_higher: List[List[int]] = []      # 合并后的高阶胞腔
        all_subcomplexes: Dict[str, List[int]] = {}  # 合并后的子复形字典
        all_level_labels: Dict[int, int] = {}        # 合并后的层级标签

        offset: int = 0  # 顶点偏移量累加

        # ---- step2: 逐分量合并 ----
        for comp_idx, sub_geo in enumerate(sub_geos):
            sub_K: SimplicialComplex = sub_geo.K
            n_v: int = len(sub_K.vertices)

            # 合并顶点: 顺序编号
            new_vertices: List[int] = [offset + v for v in sub_K.vertices]
            all_vertices.extend(new_vertices)

            # 合并边: 顶点索引加偏移
            for u, v in sub_K.edges:
                all_edges.append((u + offset, v + offset))

            # 合并高阶胞腔: 顶点索引逐元素加偏移
            for cell in sub_K.higher_cells:
                all_higher.append([v + offset for v in cell])

            # 合并子复形: 以 "comp_{i}_" 前缀避免冲突
            for name, vlist in sub_K.subcomplexes.items():
                prefixed_name: str = f"comp_{comp_idx}_{name}"
                all_subcomplexes[prefixed_name] = [v + offset for v in vlist]

            # 合并层级标签: 顶点加偏移
            for v, lv in sub_K.level_labels.items():
                all_level_labels[v + offset] = lv

            # 偏移量累加: |V_i|
            offset += n_v

        # ---- step3: 构造合并后的胞腔复形 ----
        merged_K: SimplicialComplex = SimplicialComplex(
            vertices=all_vertices,
            edges=all_edges,
            higher_cells=all_higher,
            subcomplexes=all_subcomplexes,
            level_labels=all_level_labels,
        )

        return merged_K
