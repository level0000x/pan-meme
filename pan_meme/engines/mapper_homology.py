# 泛模因几何工具 — 引擎模块：Mapper 持久同调 β₀ 估计引擎
# 数学对应：PB_ARCHITECTURE.md 第3.4节 — 用 Mapper 算法在线估计连通分量数 β₀(K)
# 论文位置：定理4 — Φ_D: G → Q 双射, 其中 n = β₀(K) 是胞腔复形 K 的连通分量数
# 核心思想：当 K 在 PB 级不可完整构建时, 通过持久同调 + Mapper 算法在线估计 β₀,
#           并用 Pregel/GAS 分布式连通分量模型提供并行精确计算路径.

import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as cc_scipy

from pan_meme.core.types import GeometricObject, SimplicialComplex


# ============================================================
# MapperHomology — Mapper 持久同调 β₀ 估计引擎
# ============================================================

class MapperHomology:
    """Mapper 持久同调引擎 — 在线估计胞腔复形 K 的连通分量数 β₀(K).

    数学对应：PB_ARCHITECTURE.md 第3.4节 方法1 — Mapper 算法.
    当胞腔复形 K 在 PB 级不可完整构建时, 通过以下流程估计 β₀:
      1. 对 token embedding 做 filter function (密度估计)
      2. 将 filter 值域分为 r 个重叠区间
      3. 每个区间内做单链聚类 (single-linkage clustering)
      4. 持久图 (persistence diagram) 的 β₀ 条对应跨区间连通分量寿命
      5. β₀ 估计 = 寿命 > ε 的 component 数量

    策略:
      - 初次全量: Mapper 近似 → 快速获得 n 数量级
      - 后续增量: 并行 CC 算法, 增量更新标签

    用法:
        mapper = MapperHomology(n_intervals=10, overlap_ratio=0.3)
        beta_0, lifetimes = mapper.estimate_betti_0(data, embeddings)
        # 或从几何对象直接估计
        beta_0, detail = mapper.estimate_betti_0_from_geometry(geo)
    """

    # ----------------------------------------------------------------
    # 构造与初始化
    # ----------------------------------------------------------------

    def __init__(
        self,
        n_intervals: int = 10,
        overlap_ratio: float = 0.3,
        min_cluster_size: int = 3,
        persistence_threshold: float = 0.1,
    ) -> None:
        """初始化 Mapper 持久同调引擎.

        数学对应: 第3.4节 — 参数化 Mapper 算法:
          - r = n_intervals: filter 值域被分为 r 个重叠区间
          - g = overlap_ratio: 相邻区间重叠比例 (默认 30%)
          - min_cluster_size: 单链聚类的最小簇大小
          - ε = persistence_threshold: β₀ 寿命阈值, 寿命 < ε 的 component 视为噪声

        Args:
            n_intervals: filter 值域被分为的区间数 (r).
            overlap_ratio: 相邻区间重叠比例, ∈ [0, 1), 默认 0.3.
            min_cluster_size: 单链聚类的最小簇大小, 默认 3.
            persistence_threshold: β₀ 寿命阈值 ε, 默认 0.1.
        """
        self.n_intervals: int = n_intervals
        """filter 值域被分为的区间数 (r)."""

        self.overlap_ratio: float = overlap_ratio
        """相邻区间重叠比例, ∈ [0, 1)."""

        self.min_cluster_size: int = min_cluster_size
        """单链聚类的最小簇大小."""

        self.persistence_threshold: float = persistence_threshold
        """β₀ 寿命阈值 ε — 寿命 < ε 的 component 视为噪声."""

    # ----------------------------------------------------------------
    # filter 函数 — 密度估计
    # ----------------------------------------------------------------

    def _filter_function(self, embeddings: np.ndarray) -> np.ndarray:
        """密度估计 filter 函数 — 使用 k-NN 距离均值.

        数学对应: 第3.4节步骤1 — filter function f: ℝ^d → ℝ.
        f(x_i) = mean(||x_i - x_j|| for x_j ∈ kNN(x_i))
        值越大表示该点越稀疏 (低密度), 值为负表示该点稠密.

        算法:
          1. 默认 k = min(10, n//2), 保证小数据集可用.
          2. 计算成对欧氏距离, 对每个点取最近 k 个邻居的距离均值.
          3. 返回 filter_values ∈ ℝⁿ.

        Args:
            embeddings: 嵌入向量矩阵, shape=(n_points, d).

        Returns:
            filter_values: 每个点的密度估计值, shape=(n_points,).
                值越大越稀疏.
        """
        n_points: int = embeddings.shape[0]
        k: int = min(10, max(1, n_points // 2))

        # ---- 计算成对欧氏距离 ----
        # 数学: dist_matrix[i,j] = ||x_i - x_j||₂
        # 使用广播计算: (n,1,d) - (1,n,d) → (n,n,d) → 取2范数
        diff: np.ndarray = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
        dist_matrix: np.ndarray = np.linalg.norm(diff, axis=2)

        # ---- 对每行取 k 个最近邻居 (除自身外) ----
        # 排序每行的距离, 取第 1..k 个 (跳过 index=0 即自身)
        sorted_dists: np.ndarray = np.sort(dist_matrix, axis=1)
        k_nearest_dists: np.ndarray = sorted_dists[:, 1:k + 1]

        # ---- filter值 = k-NN 距离均值 ----
        filter_values: np.ndarray = np.mean(k_nearest_dists, axis=1)

        return filter_values

    # ----------------------------------------------------------------
    # 区间划分
    # ----------------------------------------------------------------

    def _build_intervals(
        self, filter_values: np.ndarray
    ) -> List[Tuple[float, float]]:
        """将 filter 值域分为 r 个重叠区间.

        数学对应: 第3.4节步骤2 — 将 filter 值域 [min, max] 分为 r 个重叠区间.
        设值域宽度 L = max - min, 区间宽度 w = L / ((r-1)*g + 1),
        步长 s = w * (1 - g), 则第 i 区间 = [min + i*s - w*g, max(min + i*s + w, f_max)].

        当 filter 范围退化为单点时 (min == max), 返回单个区间覆盖 [min-δ, max+δ].

        Args:
            filter_values: filter 值数组, shape=(n,).

        Returns:
            intervals: 重叠区间列表 [(left_0, right_0), ...], 共 r 个.
        """
        f_min: float = float(np.min(filter_values))
        f_max: float = float(np.max(filter_values))

        # ---- 退化情况: 单点 ----
        if np.isclose(f_min, f_max):
            delta: float = max(1e-6, abs(f_min) * 0.1 + 1e-8)
            intervals: List[Tuple[float, float]] = [
                (f_min - delta, f_max + delta) for _ in range(self.n_intervals)
            ]
            return intervals

        # ---- 计算区间参数 ----
        # 数学: 总宽度 L, 区间宽度 w = L / ((r-1)*(1-g) + 1)
        # 但为保持 overlap_ratio 语义: w 包含重叠部分
        L: float = f_max - f_min
        g: float = self.overlap_ratio
        r: int = self.n_intervals

        # 区间宽度 w 满足: r*w - (r-1)*g*w = L → w*(r - (r-1)*g) = L
        if r == 1:
            w: float = L
            step: float = 0.0
        else:
            w = L / (r - (r - 1) * g)
            step = w * (1.0 - g)

        # ---- 生成区间 ----
        intervals = []
        for i in range(r):
            left: float = f_min + i * step - w * g * (1 if i > 0 else 0)
            right: float = left + w
            # 首区间左边界对齐 f_min, 末区间右边界对齐 f_max
            if i == 0:
                left = f_min
            if i == r - 1:
                right = f_max
            # 边界裁剪
            left = max(f_min - 1e-8, left)
            right = min(f_max + 1e-8, right)
            intervals.append((float(left), float(right)))

        return intervals

    # ----------------------------------------------------------------
    # 单链聚类
    # ----------------------------------------------------------------

    def _single_linkage_clustering(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        interval: Tuple[float, float],
    ) -> List[List[int]]:
        """在单个区间内做单链聚类.

        数学对应: 第3.4节步骤3 — 单链聚类近似连通分量.
        对落在区间 [left, right] 内的数据点, 构建邻接矩阵:
          A[i,j] = 1 if ||x_i - x_j||₂ < dist_threshold
        其中 dist_threshold 由区间宽度和局部密度自适应确定.
        然后调用 connected_components 获取簇.

        算法:
          1. 筛选落在区间内的点索引.
          2. 若点数 ≤ min_cluster_size, 返回单个簇或空.
          3. 自适应距离阈值: dist_threshold = interval_width / (log₂(n) + 1).
          4. 构建稀疏邻接矩阵, 调用 connected_components.
          5. 滤除小于 min_cluster_size 的簇.

        Args:
            data: 数据点矩阵, shape=(n, d). 用于距离计算.
            labels: 由 connected_components 得到的全量分量标签 (上一轮).
                    此处用于区间内点的索引引用.
            interval: (left, right) filter 区间边界.

        Returns:
            clusters: 簇列表, 每个簇是落在区间内的点的全局索引列表.
        """
        left, right = interval

        # ---- 步骤1: 筛选落在区间内的点 ----
        # 此处 data 和 labels 按统一索引对应
        # 按照 embedding 对应的 filter_value 是否落在 [left, right] 来筛选
        # 注意: data 参数实际是完整的原始数据点矩阵
        filter_vals: np.ndarray = self._filter_function(data)
        in_interval_mask: np.ndarray = (filter_vals >= left) & (filter_vals <= right)
        in_indices: np.ndarray = np.where(in_interval_mask)[0]
        n_in: int = len(in_indices)

        # ---- 步骤2: 小数据集退化 ----
        if n_in <= 1:
            if n_in == 1:
                return [[int(in_indices[0])]]
            return []

        # ---- 步骤3: 自适应距离阈值 ----
        # 数学: dist_threshold = interval_width / (log₂(n_in) + 1)
        # 密集区域内阈值较小, 稀疏区域内阈值较大
        interval_width: float = right - left
        dist_threshold: float = interval_width / (np.log2(max(n_in, 2)) + 1.0)
        # 保证最小阈值不为零
        dist_threshold = max(dist_threshold, 1e-6)

        # ---- 步骤4: 构建稀疏邻接矩阵 ----
        # 使用阈值构建无向图: A[i,j] = 1 当 ||x_i - x_j|| < dist_threshold
        subset_data: np.ndarray = data[in_indices]
        # 计算子集成对距离 (避免大内存)
        # 使用逐批计算降低内存占用
        n_sub: int = n_in
        # 若子集过大, 使用分块策略
        rows_list: List[int] = []
        cols_list: List[int] = []

        if n_sub <= 2000:
            # 小规模直接计算
            diff_sub: np.ndarray = (
                subset_data[:, np.newaxis, :] - subset_data[np.newaxis, :, :]
            )
            dist_sub: np.ndarray = np.linalg.norm(diff_sub, axis=2)
            # 标记距离 < threshold 的边 (排除自环)
            mask: np.ndarray = (dist_sub < dist_threshold) & ~np.eye(n_sub, dtype=bool)
            r_ij, c_ij = np.where(mask)
            rows_list = r_ij.tolist()
            cols_list = c_ij.tolist()
        else:
            # 大规模分块
            chunk_size: int = 500
            for chunk_start in range(0, n_sub, chunk_size):
                chunk_end: int = min(chunk_start + chunk_size, n_sub)
                chunk_data: np.ndarray = subset_data[chunk_start:chunk_end]
                diff_chunk: np.ndarray = (
                    chunk_data[:, np.newaxis, :] - subset_data[np.newaxis, :, :]
                )
                dist_chunk: np.ndarray = np.linalg.norm(diff_chunk, axis=2)
                mask_chunk: np.ndarray = dist_chunk < dist_threshold
                # 排除自环
                for ci_local, cj in zip(*np.where(mask_chunk), strict=False):
                    ci_global: int = chunk_start + ci_local
                    if ci_global != cj:
                        rows_list.append(ci_global)
                        cols_list.append(cj)

        if not rows_list:
            # 无连接, 每个点自成孤立簇
            return [[int(idx)] for idx in in_indices[:self.min_cluster_size]]

        # 构建稀疏邻接矩阵
        data_arr: np.ndarray = np.ones(len(rows_list), dtype=np.float32)
        A_sub: csr_matrix = csr_matrix(
            (data_arr, (np.array(rows_list, dtype=np.int32),
                        np.array(cols_list, dtype=np.int32))),
            shape=(n_sub, n_sub),
        )

        # ---- 步骤5: 连通分量计算 ----
        n_cc: int
        cc_labels: np.ndarray
        n_cc, cc_labels = cc_scipy(csgraph=A_sub, directed=False, return_labels=True)

        # ---- 步骤6: 滤除小于 min_cluster_size 的簇 ----
        clusters: List[List[int]] = []
        for cc_id in range(n_cc):
            cc_mask: np.ndarray = (cc_labels == cc_id)
            cc_local_indices: np.ndarray = np.where(cc_mask)[0]
            if len(cc_local_indices) >= self.min_cluster_size:
                # 映射回全局索引
                global_indices: List[int] = [
                    int(in_indices[li]) for li in cc_local_indices
                ]
                clusters.append(global_indices)

        return clusters

    # ----------------------------------------------------------------
    # 主算法 — β₀ 估计
    # ----------------------------------------------------------------

    def estimate_betti_0(
        self,
        data: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[int, List[float]]:
        """主算法: filter → 区间划分 → 每区间聚类 → 持久图 → β₀ 估计.

        数学对应: 第3.4节步骤4-5 — 持久图 β₀ 条与 β₀ 估计.
        完整 Mapper 管道:
          1. 若 embeddings 为 None, 使用 data 自身作为嵌入.
          2. 计算 filter 函数 f: ℝ^d → ℝ (密度估计).
          3. 将 filter 值域 [f_min, f_max] 分为 r 个重叠区间.
          4. 在每个区间内做单链聚类, 得到局部连通分量.
          5. 跨区间追踪连通分量: 若相邻区间有重叠数据点,
             且两区间的某两个簇共享至少 1 个点, 则认为它们属于同一持久分量.
          6. 记录每个持久分量的 filter 值范围 [birth, death].
          7. 持久寿命 lifetime = death - birth.
          8. β₀ 估计 = 寿命 > persistence_threshold 的持久分量数量.

        算法复杂度: O(r * (n_interval * log(n_interval))), 其中 n_interval ≈ n/r.

        Args:
            data: 数据点矩阵, shape=(n_points, d). 用于距离计算和聚类.
            embeddings: 嵌入向量矩阵, 可选. 若为 None, 使用 data 自身.

        Returns:
            (β₀估计值, 持久寿命列表).
            β₀_estimated: 寿命 > ε 的连通分量估计数.
            lifetimes: 所有追踪到的持久分量的寿命列表 (sorted descending).
        """
        # ---- 步骤1: 确定嵌入 ----
        if embeddings is None:
            embeddings = data.astype(np.float64)

        n_points: int = data.shape[0]
        if n_points == 0:
            return 0, []

        # ---- 步骤2: 计算 filter 函数 ----
        filter_vals: np.ndarray = self._filter_function(embeddings)

        # ---- 步骤3: 区间划分 ----
        intervals: List[Tuple[float, float]] = self._build_intervals(filter_vals)

        # ---- 步骤4: 每区间单链聚类 ----
        # interval_clusters[i] = 第 i 个区间内的簇列表
        interval_clusters: List[List[List[int]]] = []
        for interval in intervals:
            clusters_i: List[List[int]] = self._single_linkage_clustering(
                data, np.zeros(n_points, dtype=np.int32), interval
            )
            interval_clusters.append(clusters_i)

        # ---- 步骤5: 跨区间追踪连通分量 (持久分量) ----
        # 数学对应: 持久图 (persistence diagram) 中的 β₀ 条.
        # 对每个持久分量, 记录其 (birth_interval_index, death_interval_index).
        #
        # 追踪策略: 维护活跃分量集合.
        # 当一个簇在区间 i 首次出现 (不与区间 i-1 的任何簇共享点),
        # 则 birth = i. 当区间 i 没有簇延续自上一区间的某活跃分量,
        # 则该分量 death = i-1.

        # 活跃分量: {component_id: (birth_interval, last_seen_interval, last_cluster_indices)}
        active_components: Dict[int, Tuple[int, int, List[int]]] = {}
        next_comp_id: int = 0
        # 已消亡分量: [(birth, death), ...]
        birth_death_pairs: List[Tuple[int, int]] = []

        for i, clusters_i in enumerate(interval_clusters):
            # 当前区间簇的匹配状态
            matched_comp_ids: set = set()

            for cluster_indices in clusters_i:
                cluster_set: set = set(cluster_indices)

                # 查找上一区间中与该簇共享点的活跃分量
                found_match: bool = False
                for comp_id, (birth, last_seen, last_indices) in list(
                    active_components.items()
                ):
                    if comp_id in matched_comp_ids:
                        continue
                    # 重叠判定: 共享至少 1 个点
                    if cluster_set.intersection(set(last_indices)):
                        # 匹配成功, 更新分量
                        active_components[comp_id] = (birth, i, cluster_indices)
                        matched_comp_ids.add(comp_id)
                        found_match = True
                        break

                if not found_match:
                    # 新分量: birth = i
                    new_id: int = next_comp_id
                    next_comp_id += 1
                    active_components[new_id] = (i, i, cluster_indices)
                    matched_comp_ids.add(new_id)

            # 判定消亡: 上一区间活跃但未在 i 区间匹配到的分量
            dead_ids: List[int] = []
            for comp_id, (birth, last_seen, _) in active_components.items():
                if last_seen < i - 1:
                    # 此分量早已死亡 (在区间 i-1 之前)
                    continue
                if comp_id not in matched_comp_ids and last_seen == i - 1:
                    # 上次活跃于 i-1 但 i 区间未匹配 → 消亡于 i-1
                    birth_death_pairs.append((birth, i - 1))
                    dead_ids.append(comp_id)

            for dead_id in dead_ids:
                del active_components[dead_id]

        # ---- 处理仍活跃的分量: death = r-1 ----
        for comp_id, (birth, _, _) in active_components.items():
            birth_death_pairs.append((birth, self.n_intervals - 1))

        # ---- 步骤6-7: 计算持久寿命 ----
        # 数学: lifetime = (death - birth) / n_intervals, 归一化到 [0, 1].
        r: int = self.n_intervals
        lifetimes: List[float] = []
        for birth, death in birth_death_pairs:
            lifetime: float = (death - birth + 1) / r
            lifetimes.append(max(lifetime, 0.0))

        # 按寿命降序排列
        lifetimes.sort(reverse=True)

        # ---- 步骤8: β₀ 估计 ----
        # 数学: β₀_estimated = count(life_i > ε)
        beta_0: int = int(np.sum(np.array(lifetimes) > self.persistence_threshold))

        return beta_0, lifetimes

    # ----------------------------------------------------------------
    # 从几何对象直接估计 β₀
    # ----------------------------------------------------------------

    def estimate_betti_0_from_geometry(
        self, geo: GeometricObject
    ) -> Tuple[int, Dict[str, Any]]:
        """从 GeometricObject 直接估计 β₀.

        数学对应: 定理4 — Φ_D: G → Q 双射, n = β₀(K).
        当 K 在 PB 级不可完整构建时, 从 K 的顶点和边构建嵌入,
        通过 Mapper 算法近似估计 β₀.

        算法:
          1. 从 K 提取顶点数、边数作为基本特征.
          2. 若边集为空, 直接返回 |V| 个孤立分量 (每个顶点 = 1 个分量).
          3. 构建邻接矩阵, 使用 scipy 精确计算连通分量数 (若 K 规模可接受).
          4. 同时用 Mapper 近似估计, 返回两种结果对比.

        Args:
            geo: GeometricObject, 包含 K = SimplicialComplex.

        Returns:
            (β₀估计值, 详细结果字典):
              - beta_0_exact: 精确 β₀ (若可计算)
              - beta_0_mapper: Mapper 近似 β₀
              - lifetimes: 持久寿命列表
              - n_vertices, n_edges: 胞腔复形规模
        """
        K: SimplicialComplex = geo.K
        n_vertices: int = len(K.vertices)
        n_edges: int = len(K.edges)

        detail: Dict[str, Any] = {
            "n_vertices": n_vertices,
            "n_edges": n_edges,
            "beta_0_exact": None,
            "beta_0_mapper": None,
            "lifetimes": [],
            "method": "mapper_estimate",
        }

        # ---- 退化情况: 无顶点 ----
        if n_vertices == 0:
            detail["beta_0_exact"] = 0
            detail["beta_0_mapper"] = 0
            return 0, detail

        # ---- 退化情况: 无边集, 每个顶点孤立 ----
        if n_edges == 0:
            detail["beta_0_exact"] = n_vertices
            detail["beta_0_mapper"] = n_vertices
            detail["method"] = "isolated_vertices"
            return n_vertices, detail

        # ---- 精确计算 (若 K 规模可接受) ----
        # 构建邻接矩阵
        beta_0_exact: Optional[int] = None
        if n_vertices <= 1000000:
            # 数学: 精确 β₀ = 连通分量数
            rows: np.ndarray = np.array(
                [e[0] for e in K.edges] + [e[1] for e in K.edges],
                dtype=np.int32,
            )
            cols: np.ndarray = np.array(
                [e[1] for e in K.edges] + [e[0] for e in K.edges],
                dtype=np.int32,
            )
            data_arr: np.ndarray = np.ones(len(rows), dtype=np.float32)
            A_csr: csr_matrix = csr_matrix(
                (data_arr, (rows, cols)), shape=(n_vertices, n_vertices)
            )
            n_labels: int
            cc_labels: np.ndarray
            n_labels, cc_labels = cc_scipy(
                csgraph=A_csr, directed=False, return_labels=True
            )
            beta_0_exact = n_labels
            detail["beta_0_exact"] = beta_0_exact

        # ---- Mapper 近似估计 ----
        # 构建嵌入: 使用顶点度数和邻接信息作为 embedding
        # 每个顶点 v 的 embedding[v] = [degree_normalized, avg_neighbor_degree, ...]
        if n_vertices > 0:
            # 简单嵌入: 顶点 ID 作为 1D 嵌入
            emb: np.ndarray = np.arange(n_vertices, dtype=np.float64).reshape(-1, 1)
            # 归一化
            emb = emb / max(n_vertices, 1.0)

            beta_0_mapper: int
            lifetimes: List[float]
            beta_0_mapper, lifetimes = self.estimate_betti_0(
                data=np.arange(n_vertices, dtype=np.float64).reshape(-1, 1),
                embeddings=emb,
            )
        else:
            beta_0_mapper = 0
            lifetimes = []

        detail["beta_0_mapper"] = beta_0_mapper
        detail["lifetimes"] = lifetimes

        # ---- 若有精确值, 优先返回精确值 ----
        if beta_0_exact is not None:
            detail["method"] = "exact_scipy"
            return beta_0_exact, detail

        return beta_0_mapper, detail


# ============================================================
# PregelCC — 分布式连通分量算法 (单机原型)
# ============================================================

class PregelCC:
    """分布式连通分量算法 — Pregel/GAS 模型的单机原型.

    数学对应: PB_ARCHITECTURE.md 第3.4节方法2 — 分布式连通分量.
    Pregel / GAS (Gather-Apply-Scatter) 模型:
      - 每个顶点 v 维护 label = min(neighbor_label, v.id)
      - 每次超步: 广播 label, 邻居更新
      - 收敛后: unique(labels) = n_components = β₀

    策略:
      - 初次全量: Mapper 近似 → 快速获得 n 数量级
      - 后续增量: Pregel CC 并行精确计算, 每日增量更新标签

    用法:
        pregel = PregelCC(max_supersteps=50)
        labels, n_components = pregel.compute_components(adj_matrix)
        # 增量更新
        new_labels = pregel.incremental_update(labels, adj_matrix, new_edges)
    """

    # ----------------------------------------------------------------
    # 构造与初始化
    # ----------------------------------------------------------------

    def __init__(self, max_supersteps: int = 50) -> None:
        """初始化 Pregel CC 引擎.

        数学对应: 第3.4节方法2 — 超步数上限 S_max.
        每次超步: 所有顶点并行更新 label = min(neighbor_labels, v.id).
        最多运行 S_max 次超步或直到收敛.

        Args:
            max_supersteps: 最大超步数, 默认 50.
                对随机图, 期望收敛超步数 ≈ O(log n).
        """
        self.max_supersteps: int = max_supersteps
        """最大超步数 S_max."""

        self._supersteps_used: int = 0
        """上次计算实际使用的超步数."""

    # ----------------------------------------------------------------
    # 分布式连通分量计算
    # ----------------------------------------------------------------

    def compute_components(
        self, adj_matrix: csr_matrix
    ) -> Tuple[np.ndarray, int]:
        """Pregel/GAS 模型连通分量计算.

        数学对应: 第3.4节方法2 — Pregel CC 算法.
        算法 (单机原型):
          超步 0: 每个顶点 v 初始化 label[v] = v.id
          超步 s: 每个顶点 v:
            - Gather: 收集所有邻居 u ∈ N(v) 的 label[u]
            - Apply:  new_label[v] = min(label[v], min_{u∈N(v)} label[u])
            - Scatter: 若 new_label[v] != label[v], 向所有邻居发送 new_label[v]
          收敛判定: ∀v, label[v] == min_{u∈N(v)∪{v}} label[u]
            → β₀ = |unique(labels)|.

        数学复杂度: O(S_max * |E|), S_max 期望 ≈ O(log n).

        Args:
            adj_matrix: 稀疏邻接矩阵 (csr_matrix), shape=(n, n).
                无向图需要满足 A[i,j] = A[j,i].

        Returns:
            (labels, n_components):
              labels: 每个顶点的分量标签, shape=(n,).
              n_components: 分量数 = β₀.
        """
        n: int = adj_matrix.shape[0]
        if n == 0:
            return np.array([], dtype=np.int32), 0

        # ---- 步骤0: 初始化 ----
        # label[v] = v.id
        labels: np.ndarray = np.arange(n, dtype=np.int32)

        # ---- Pregel 迭代 ----
        self._supersteps_used = 0
        for superstep in range(self.max_supersteps):
            self._supersteps_used = superstep + 1

            # ---- Gather + Apply: 每个顶点收集邻居最小 label ----
            new_labels: np.ndarray = labels.copy()

            # 将邻接矩阵转为 COO 格式以便高效迭代
            # csr_matrix 直接遍历非零元
            for v in range(n):
                # 获取顶点 v 的所有邻居
                row_start: int = adj_matrix.indptr[v]
                row_end: int = adj_matrix.indptr[v + 1]
                neighbors = adj_matrix.indices[row_start:row_end]

                if len(neighbors) > 0:
                    neighbor_labels: np.ndarray = labels[neighbors]
                    min_neighbor_label: int = int(np.min(neighbor_labels))
                    min_label: int = min(labels[v], min_neighbor_label)
                    new_labels[v] = min_label

            # ---- 收敛判定 ----
            if np.array_equal(new_labels, labels):
                break

            labels = new_labels

        # ---- 组件计数 ----
        unique_labels: np.ndarray = np.unique(labels)
        n_components: int = len(unique_labels)

        # ---- 重编号 ----
        # 将分量标签压缩为 0..β₀-1
        label_map: Dict[int, int] = {
            old_label: new_label for new_label, old_label in enumerate(unique_labels)
        }
        compressed_labels: np.ndarray = np.array(
            [label_map[lbl] for lbl in labels], dtype=np.int32
        )

        return compressed_labels, n_components

    # ----------------------------------------------------------------
    # 增量标签更新
    # ----------------------------------------------------------------

    def incremental_update(
        self,
        labels: np.ndarray,
        adj_matrix: csr_matrix,
        new_edges: List[Tuple[int, int]],
    ) -> np.ndarray:
        """增量标签更新 — 新增边后局部更新连通分量标签.

        数学对应: 第3.4节策略 — 增量更新标签.
        当新增少量的边 (new_edges) 时, 不需要全局重算连通分量.
        仅需:
          1. 找到新增边两端顶点所属的分量 label.
          2. 若两端标签不同, 将较大标签的分量全部合并到较小标签的分量.
          3. 向此分量的所有顶点传播新标签.

        这比全局重算 O(|E|·S_max) 更高效, 复杂度 O(|new_edges| + |affected_vertices|).

        Args:
            labels: 当前每个顶点的分量标签, shape=(n,).
            adj_matrix: 稀疏邻接矩阵, shape=(n, n).
            new_edges: 新增边列表 [(u, v), ...].

        Returns:
            更新后的分量标签数组, shape=(n,).
        """
        n: int = len(labels)
        updated_labels: np.ndarray = labels.copy()

        # ---- 步骤1: 收集受影响的合并关系 ----
        # 需要合并的分量对 (将大标签合并到小标签)
        merges: List[Tuple[int, int]] = []

        for u, v in new_edges:
            if u < 0 or u >= n or v < 0 or v >= n:
                continue
            cl: int = int(updated_labels[u])
            cr: int = int(updated_labels[v])
            if cl != cr:
                # 始终将大标签合并到小标签
                if cl < cr:
                    merges.append((cr, cl))
                else:
                    merges.append((cl, cr))

        if not merges:
            return updated_labels

        # ---- 步骤2: 构建合并传递闭包 ----
        # 使用 Union-Find 合并受影响的标签
        all_affected_labels: set = set()
        merge_map: Dict[int, int] = {}

        for big, small in merges:
            all_affected_labels.add(big)
            all_affected_labels.add(small)
            merge_map[big] = small

        # 传递合并: 若 a→b 且 b→c, 则 a→c
        def find_root(lbl: int) -> int:
            """Union-Find 找根."""
            while lbl in merge_map and merge_map[lbl] != lbl:
                lbl = merge_map[lbl]
            return lbl

        # 压缩路径
        for lbl in list(merge_map.keys()):
            merge_map[lbl] = find_root(lbl)

        # ---- 步骤3: 应用合并 ----
        for i in range(n):
            old_lbl: int = int(updated_labels[i])
            new_lbl: int = find_root(old_lbl)
            if new_lbl != old_lbl:
                updated_labels[i] = new_lbl

        # ---- 步骤4: 重编号确保连续 ----
        unique_labels: np.ndarray = np.unique(updated_labels)
        label_map: Dict[int, int] = {
            old: new for new, old in enumerate(unique_labels)
        }
        updated_labels = np.array(
            [label_map[int(lbl)] for lbl in updated_labels], dtype=np.int32
        )

        return updated_labels


# ============================================================
# 辅助 — 从 SimplicialComplex 构建邻接矩阵
# ============================================================

def _build_adjacency_from_complex(K: SimplicialComplex, n_vertices: int) -> csr_matrix:
    """从胞腔复形 K 构建无向邻接矩阵.

    数学对应: 定理4 — K 的 1-骨架邻接矩阵.
    A[i,j] = 1 若 (i,j) ∈ K.edges 或 (j,i) ∈ K.edges.

    Args:
        K: 胞腔复形.
        n_vertices: 顶点数.

    Returns:
        稀疏邻接矩阵 A ∈ R^{|V|×|V|}.
    """
    if len(K.edges) == 0:
        return csr_matrix((n_vertices, n_vertices), dtype=np.float32)

    rows: np.ndarray = np.array(
        [e[0] for e in K.edges] + [e[1] for e in K.edges],
        dtype=np.int32,
    )
    cols: np.ndarray = np.array(
        [e[1] for e in K.edges] + [e[0] for e in K.edges],
        dtype=np.int32,
    )
    data_arr: np.ndarray = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data_arr, (rows, cols)), shape=(n_vertices, n_vertices))
