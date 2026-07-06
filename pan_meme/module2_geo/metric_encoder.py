"""
模块2 — 几何化层: 度量编码器 (MetricEncoder)

数学对应: 定义3(b) — g_i = w(e_i), 每个 1-cell 的长度等于对应边的连接强度.
度量结构 g 赋予胞腔复形 K 的每条边一个几何长度, 这是几何化的核心步骤.
"""

from typing import Dict

import numpy as np

from pan_meme.core.types import RelationNetwork, SimplicialComplex


class MetricEncoder:
    """
    度量编码器: 边权 → 度量向量的确定性映射.

    数学对应:
    - 定义3(b): g = (g_1, g_2, ..., g_{|E|}), 其中 g_i = w(e_i).
    - 每个 g_i 表示 K 中第 i 条 1-cell 的长度, 即对应边的连接强度.
    - 这是从离散图到连续几何的关键一步: w ∈ [0,1]^{|E|} → g ∈ R^{|E|}.

    用法:
        g = MetricEncoder.encode(psi, K, edge_map)
    """

    @staticmethod
    def encode(
        psi: RelationNetwork,
        K: SimplicialComplex,
        edge_map: Dict[int, int],
    ) -> np.ndarray:
        """
        将边权 w 编码为度量向量 g.

        数学注解:
        - g[i] = psi.weights[edge_map^{-1}(i)] for each K-edge i.
        - 映射法则: 对于 K 的第 j 条边 (对应原图第 k 条边, 其中 edge_map[k] = j),
          取其权重 w_k 作为几何长度 g_j.
        - 使用 float32 精度, 保证数值稳定性.

        Args:
            psi: 关系网络 Ψ, 包含 weights 字段 (shape=(|E|,)).
            K: 胞腔复形, 提供边的数量用于分配 g.
            edge_map: 图边索引 → K 边索引的映射, 由 SkeletonEncoder 产出.

        Returns:
            np.ndarray: 度量向量 g, shape=(|K.edges|,), dtype=float32.
                        每个分量 g[i] 表示第 i 条 K-边的几何长度.
        """
        # ---- 步骤1: 分配度量向量 ----
        # 定义3(b): g ∈ R^{|E_K|}, 初始化为零.
        n_edges: int = len(K.edges)
        g: np.ndarray = np.zeros(n_edges, dtype=np.float32)

        # ---- 步骤2: 填充边权 ----
        # 映射法则: g[new_idx] = w[old_idx], 其中 edge_map[old_idx] = new_idx.
        for old_idx, new_idx in edge_map.items():
            # 安全检查: new_idx 必须在 K 的边范围内
            if 0 <= new_idx < n_edges:
                g[new_idx] = float(psi.weights[old_idx])

        return g
