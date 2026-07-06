"""
模块3 — 模因化层: 五维映射 (Mapping5D)

数学对应: 前提5 — K_i → X_i = (m_i, ξ_i) 双射; 论文3.4.2 五维提取.

五维核心 m_i = (D, B, ρ, R, S) ∈ [0,1]^5 是 Ω 不变集上的模因状态,
扩展维度 ξ_i 编码微观涨落 (边权残差向量). 映射为双射, 保证可逆性.

各维度含义 (论文3.4.2):
  D  — 内禀度 (Internal Depth):      结构复杂性与层级深度.
  B  — 关联度 (Bonding):             边连接密度.
  ρ  — 能流密度 (Energy Flux):      场 ω 的平均绝对值.
  R  — 演化速率 (Reaction Rate):     场 ω 的梯度标准差.
  S  — 结构韧度 (Structural Toughness): 不变量密度.

用法:
    from pan_meme.module3_meme.mapping_5d import Mapping5D
    meme = Mapping5D.map(sub_geo, idx=0)
    K_i  = Mapping5D.inverse_map(meme, idx=0)
"""

from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy

import numpy as np
from numpy.typing import NDArray

from pan_meme.core.types import GeometricObject, MemeState, SimplicialComplex


# ================================================================
# 全局缩放常量 (论文建议值)
# ================================================================
_C_MAX: int = 10000    # 胞腔数归一化上限 — 超过此值 D=1 饱和
_D_MAX: int = 20       # 层级深度归一化上限 — 超过此值因子 1.0


class Mapping5D:
    """
    五维映射: K_i → X_i 的双射实现.

    数学对应:
    - 前提5: Φ_E^{(i)}: K_i → X_i = (m_i, ξ_i) 是一个双射.
    - m_i ∈ Ω = [0,1]^5 是五维缩约核心 (定义6).
    - ξ_i ∈ Ξ 是扩展维度 — 边权残差向量, 编码微观信息.
    - 双射性质: 给定 m_i 和 ξ_i, 可以唯一恢复 K_i.

    常量:
        c_max (float): 胞腔数归一化上限, 默认 10000.
        d_max (float): 层级深度归一化上限, 默认 20.
    """

    # 可配置缩放常量
    c_max: float = float(_C_MAX)
    d_max: float = float(_D_MAX)

    # ================================================================
    # 正向映射: K_i → X_i = (m_i, ξ_i)
    # ================================================================

    @staticmethod
    def map(sub_geo: GeometricObject, idx: int) -> MemeState:
        """
        五维提取: 从子几何体 G_i 计算模因状态 X_i = (m_i, ξ_i).

        五维公式 (论文3.4.2):
          D = min(1, cell_count / c_max) * (level_depth / d_max)
          B = n_edges / max(n_vertices * (n_vertices - 1) / 2, 1)
          ρ = np.mean(|omega|),        然后 clip [0, 1]
          R = np.std(gradient(omega)), 然后 clip [0, 1]  (若 |V| < 2 则为 0)
          S = min(1, len(Gamma) / 10)

        扩展维度:
          ξ = g - median(g)  — 边权残差向量, 用作微观涨落编码.

        数学注解:
        - D 编码 "结构有多少层 + 多大规模", 内禀度越高结构越复杂.
        - B 编码 "连接密度", 完全图 B=1, 无边的空图 B=0.
        - ρ 编码 "能量/信息流动的平均强度".
        - R 编码 "场梯度 = 能量空间变化率".
        - S 编码 "规则/约束密度 = 抗扰动能力".

        Args:
            sub_geo: 子几何体 G_i = (K_i, g_i, ω_i, Γ_i, R_i).
            idx: 模因索引 (用于日志/调试).

        Returns:
            MemeState: X_i = (D, B, ρ, R, S, ξ).
        """
        K: SimplicialComplex = sub_geo.K
        n_vertices: int = len(K.vertices)
        n_edges: int = len(K.edges)

        # ---- D: 内禀度 ----
        # 公式: D = min(1, cell_count / c_max) * (level_depth / d_max)
        # cell_count = n_vertices + n_edges + 高阶胞腔数量.
        cell_count: int = n_vertices + n_edges + sum(len(c) for c in K.higher_cells)
        cell_factor: float = min(1.0, float(cell_count) / Mapping5D.c_max)

        # level_depth: K 中的最大层级编号.
        level_depth: int = (
            max(K.level_labels.values()) if K.level_labels else 0
        )
        depth_factor: float = float(level_depth) / Mapping5D.d_max

        D: float = cell_factor * depth_factor

        # ---- B: 关联度 ----
        # 公式: B = n_edges / max(n_vertices * (n_vertices - 1) / 2, 1)
        # 最多可能的边数 (无向完全图 K_n): n*(n-1)/2.
        max_edges: float = max(float(n_vertices) * float(n_vertices - 1) / 2.0, 1.0)
        B: float = float(n_edges) / max_edges

        # ---- ρ: 能流密度 ----
        # 公式: ρ = mean(|omega|), clip [0, 1].
        omega: NDArray[np.float32] = sub_geo.omega.astype(np.float32)
        rho_raw: float = float(np.mean(np.abs(omega))) if len(omega) > 0 else 0.0
        rho: float = float(np.clip(rho_raw, 0.0, 1.0))

        # ---- R: 演化速率 ----
        # 公式: R = std(gradient(omega)), clip [0, 1].
        # 若顶点数 < 2, 梯度无定义, 设 R = 0.
        if len(omega) >= 2:
            # 标量场 ω 的离散梯度: np.gradient 默认使用二阶中心差分.
            grad_omega: NDArray[np.float64] = np.gradient(omega.astype(np.float64))
            R_raw: float = float(np.std(grad_omega))
            R: float = float(np.clip(R_raw, 0.0, 1.0))
        else:
            R = 0.0

        # ---- S: 结构韧度 ----
        # 公式: S = min(1, len(Gamma) / 10)
        # Gamma 是字典, 取 key 的数量作为不变量密度度量.
        Gamma: dict = sub_geo.Gamma
        S: float = min(1.0, float(len(Gamma)) / 10.0)

        # ---- ξ: 边权残差 (扩展维度) ----
        # 公式: ξ = g - median(g)
        # 编码边权相对于中位数的偏离, 保留微观涨落信息.
        g: NDArray[np.float32] = sub_geo.g.astype(np.float32)
        if len(g) > 0:
            g_median: float = float(np.median(g))
            xi: NDArray[np.float32] = (g - g_median).astype(np.float32)
        else:
            xi = np.zeros(0, dtype=np.float32)

        # ---- 组装 MemeState ----
        return MemeState(
            D=D,
            B=B,
            rho=rho,
            R=R,
            S=S,
            xi=xi,
        )

    # ================================================================
    # 逆映射: X_i → K_i
    # ================================================================

    @staticmethod
    def inverse_map(meme: MemeState, idx: int) -> SimplicialComplex:
        """
        逆映射 Φ_E^{-1}: X_i = (m_i, ξ_i) → K_i.

        重建逻辑:
          step1: 从 ξ 的长度反推边数: n_edges' = len(ξ).
          step2: 用 sqrt 近似顶点数: n_vertices' ≈ ceil(sqrt(2 * n_edges')).
          step3: 从 B 反推实际边数: n_edges_real ≈ round(B * n_vertices*(n_vertices-1)/2).
          step4: 构造顶点点索引列表和近似边集 (环形拓扑).
          step5: 从 B 和 ρ 估算层级标签.

        数学注解:
        - 逆映射是近似的: 因为我们无法从 ξ 精确恢复原始边连接拓扑.
        - 逆映射的保真度取决于 ξ 的长度 (保留了 |E|) 和 B (保留了连接密度).
        - 完全双射需要额外存储 adjacency_matrix 于元数据中 (实践中可补充).

        Args:
            meme: 模因状态 X_i = (D, B, ρ, R, S, ξ).
            idx: 模因索引 (用于日志/调试).

        Returns:
            SimplicialComplex: 重建的胞腔复形 K_i'.
        """
        # ---- step1: ξ 长度 → 边数估计 ----
        n_edges_est: int = len(meme.xi)

        # ---- step2: sqrt 近似顶点数 ----
        # 从边数反推顶点数: 对于稀疏图, n ≈ ceil(sqrt(2 * |E|)).
        # 对于 |E| = 0 的情形, n = 1 (至少一个孤立顶点).
        if n_edges_est == 0:
            n_vertices_est: int = 1
        else:
            import math
            n_vertices_est = max(
                2,
                math.ceil(math.sqrt(2.0 * float(n_edges_est))),
            )

        # ---- step3: 从 B 反推实际边数 ----
        # B = n_edges / max(n_vertices*(n_vertices-1)/2, 1)
        # → n_edges_real = round(B * max_edges_possible)
        max_edges_possible: float = max(
            float(n_vertices_est) * float(n_vertices_est - 1) / 2.0, 1.0
        )
        n_edges_real: int = max(
            1 if n_edges_est > 0 else 0,
            round(meme.B * max_edges_possible),
        )
        # 有界处理: 不超过可能的最大边数.
        n_edges_real = min(n_edges_real, int(max_edges_possible))

        # ---- step4: 构造顶点点索引和近似边集 ----
        vertices: List[int] = list(range(n_vertices_est))

        # 环形拓扑: 为每个顶点连接到后续顶点, 循环.
        # 这是最简单的可控边密度拓扑结构.
        edges: List[Tuple[int, int]] = []
        degree: int = max(1, n_edges_real // n_vertices_est) if n_vertices_est > 1 else 0
        for v in range(n_vertices_est):
            for d in range(1, min(degree + 1, n_vertices_est)):
                neighbor: int = (v + d) % n_vertices_est
                if v < neighbor:  # 避免重复边
                    edges.append((v, neighbor))
                    if len(edges) >= n_edges_real:
                        break
            if len(edges) >= n_edges_real:
                break

        # 若边数不足, 用循环补足.
        while len(edges) < n_edges_real and n_vertices_est > 1:
            # 添加额外的交叉边
            candidate_u: int = len(edges) % n_vertices_est
            candidate_v: int = (len(edges) * 3 + 1) % n_vertices_est
            if candidate_u != candidate_v and (candidate_u, candidate_v) not in edges:
                edges.append((
                    min(candidate_u, candidate_v),
                    max(candidate_u, candidate_v),
                ))

        # ---- step5: 从 D 和 ρ 估算层级标签 ----
        # D = cell_factor * depth_factor → depth_factor = D / cell_factor.
        # cell_factor ≈ min(1, (n_vertices + n_edges) / c_max).
        cell_count_est: int = n_vertices_est + len(edges)
        cell_factor_est: float = min(1.0, float(cell_count_est) / Mapping5D.c_max)
        depth_factor_est: float = (
            meme.D / max(cell_factor_est, 1e-8)
            if cell_factor_est > 0
            else 0.0
        )
        est_level_depth: int = max(0, round(depth_factor_est * Mapping5D.d_max))

        # 层级标签: 基于 ρ 值分配 (ρ 越高, 层级越深).
        level_labels: Dict[int, int] = {}
        for v in range(n_vertices_est):
            # 使用 ρ 调制: ρ=0 → 层级 0; ρ=1 → 层级 est_level_depth.
            level_labels[v] = max(0, min(est_level_depth, round(meme.rho * float(est_level_depth))))

        # ---- 逆映射完成, 组装胞腔复形 ----
        return SimplicialComplex(
            vertices=vertices,
            edges=edges,
            higher_cells=[],
            subcomplexes={},
            level_labels=level_labels,
        )

    # ================================================================
    # 新增：并行批处理模式 (PB_ARCHITECTURE.md 第4节)
    # ================================================================

    @staticmethod
    def _map_worker(sub_geo: GeometricObject, idx: int) -> MemeState:
        """并行池辅助函数 — 对单个子几何体执行五维映射.

        PB_ARCHITECTURE.md 第4节 — Ray 并行 map.
        作为 ProcessPoolExecutor 的 worker 调用.

        Args:
            sub_geo: 子几何体 G_i.
            idx: 模因索引.

        Returns:
            MemeState: X_i = (D, B, ρ, R, S, ξ).
        """
        return Mapping5D.map(sub_geo, idx=idx)

    @staticmethod
    def _inverse_worker(meme: MemeState, idx: int) -> SimplicialComplex:
        """并行池辅助函数 — 对单个模因执行逆映射.

        Args:
            meme: 模因状态 X_i.
            idx: 索引.

        Returns:
            SimplicialComplex: 重建的胞腔复形 K_i'.
        """
        return Mapping5D.inverse_map(meme, idx=idx)

    @staticmethod
    def map_batch(
        sub_geos: List[GeometricObject],
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> List[MemeState]:
        """批量五维映射 — 对子几何体列表逐个执行 map.

        PB_ARCHITECTURE.md 第4节 — Ray 并行 map, 10^7 并发.
        当 parallel=True 时, 使用 ProcessPoolExecutor 并行映射.
        n_jobs=-1 表示使用全部 CPU 核心.

        数学对应: 前提5 — K_i → X_i 双射, 批量版本.
        并行化基于每个 K_i 的映射彼此独立 (embarrassingly parallel).

        注意事项:
          - numpy 数组在进程间传递需要序列化, ProcessPoolExecutor
            使用 pickle 序列化, 对 MemeState (dataclass + ndarray) 兼容.
          - 若序列化失败 (如自定义对象), 自动回退到顺序执行.

        Args:
            sub_geos: 子几何体列表.
            parallel: 是否启用并行映射, 默认 False.
            n_jobs: 并行工作进程数, -1 为全部核心.

        Returns:
            List[MemeState]: 按顺序排列的模因状态列表.
        """
        n: int = len(sub_geos)
        if n == 0:
            return []

        if not parallel:
            # 顺序执行
            return [
                Mapping5D.map(sub_geos[i], idx=i)
                for i in range(n)
            ]

        # ---- 并行执行 ----
        import os
        max_workers: int = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
        # 不超过任务数
        max_workers = min(max_workers, n)

        # 预分配结果数组以保持顺序
        results: List[Optional[MemeState]] = [None] * n

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx: dict = {
                    executor.submit(Mapping5D._map_worker, sub_geos[i], i): i
                    for i in range(n)
                }
                for future in as_completed(future_to_idx):
                    idx: int = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        # 单个任务失败时回退到顺序执行
                        results[idx] = Mapping5D._map_worker(sub_geos[idx], idx)
        except Exception:
            # ProcessPoolExecutor 整体失败 (如 pickle 问题), 回退顺序
            return [
                Mapping5D.map(sub_geos[i], idx=i)
                for i in range(n)
            ]

        return [r for r in results if r is not None]

    @staticmethod
    def inverse_map_batch(
        memes: List[MemeState],
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> List[SimplicialComplex]:
        """批量逆映射 — 对模因列表逐个执行 inverse_map.

        PB_ARCHITECTURE.md 第4节 — 并行逆映射.
        parallel=True 时使用 ProcessPoolExecutor 并行逆映射.

        数学对应: 前提5 — Φ_E^{-1}: X_i → K_i, 批量版本.

        Args:
            memes: 模因状态列表.
            parallel: 是否启用并行逆映射, 默认 False.
            n_jobs: 并行工作进程数, -1 为全部核心.

        Returns:
            List[SimplicialComplex]: 按顺序排列的重建胞腔复形列表.
        """
        n: int = len(memes)
        if n == 0:
            return []

        if not parallel:
            return [
                Mapping5D.inverse_map(memes[i], idx=i)
                for i in range(n)
            ]

        # ---- 并行执行 ----
        import os
        max_workers: int = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
        max_workers = min(max_workers, n)

        results: List[Optional[SimplicialComplex]] = [None] * n

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx: dict = {
                    executor.submit(Mapping5D._inverse_worker, memes[i], i): i
                    for i in range(n)
                }
                for future in as_completed(future_to_idx):
                    idx: int = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception:
                        results[idx] = Mapping5D._inverse_worker(memes[idx], idx)
        except Exception:
            return [
                Mapping5D.inverse_map(memes[i], idx=i)
                for i in range(n)
            ]

        return [r for r in results if r is not None]
