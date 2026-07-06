"""
模块2 — 几何化层: 骨架编码器 (SkeletonEncoder)

数学对应: 定理3 — Φ_C: M → G, 骨架部分 K 的构造.
定义3: K 是几何对象 G 的第一个组件, 图 → 胞腔复形的同构映射.
S = (V, E, w) → K, vertex_map, edge_map

核心不变量: K 的顶点与边直接继承自 Ψ 的节点与边,
子复形结构由 hierarchy 中的层级信息决定.
"""

from typing import Dict, Tuple

import numpy as np

from pan_meme.core.types import RelationNetwork, SimplicialComplex


class SkeletonEncoder:
    """
    骨架编码器: 图 → 胞腔复形同构映射.

    数学对应:
    - 定理3 (Φ_C: M → G): K = Φ_C^{skeleton}(Ψ) 是几何对象 G 的胞腔复形组件.
    - 定义3: K 是 (a) 胞腔复形, K 的 0-cells = V, 1-cells = E, 更高阶胞腔 = hierarchy 诱导的子复形.
    - 同构性质: vertex_map 和 edge_map 确保了图结构到胞腔结构的完整保结构信息.

    用法:
        K, vmap, emap = SkeletonEncoder.encode(psi)
    """

    @staticmethod
    def encode(
        psi: RelationNetwork,
    ) -> Tuple[SimplicialComplex, Dict[int, int], Dict[int, int]]:
        """
        将关系网络 Ψ 编码为胞腔复形 K.

        数学注解:
        - S = (V, E, w) → K = (vertices, edges, subcomplexes, level_labels)
        - K 的 0-cells: 每个 v ∈ V 对应一个顶点.
        - K 的 1-cells: 每个 e ∈ E 对应一条边.
        - 恒等映射: vertex_map[i] = i 保证 S 与 K 的顶点一一对应 (同构).

        Args:
            psi: 关系网络 Ψ = (V, E, w), 即模块1输出的结构域.

        Returns:
            Tuple[SimplicialComplex, Dict[int, int], Dict[int, int]]:
                - K: 胞腔复形, 包含顶点、边、子复形和层级标签.
                - vertex_map: 图节点ID → K 顶点索引的映射 (初始版本为恒等映射).
                - edge_map: 图边索引 → K 边索引的映射.
        """
        # ---- 步骤1: 构建顶点映射 ----
        # 公理2: 每个 v ∈ V 映射为 K 的一个 0-cell.
        # 初始版本使用恒等映射: vertex_map[i] = i.
        n: int = len(psi.nodes)
        vertex_map: Dict[int, int] = {i: i for i in range(n)}

        # ---- 步骤2: 构建边映射与K-边列表 ----
        # 定义3(a): 每个 e ∈ E 映射为 K 的一个 1-cell.
        # edge_map 记录图边索引到胞腔边索引的对应关系.
        edge_map: Dict[int, int] = {}
        K_edges: list = []

        for idx, (i, j) in enumerate(psi.edges):
            # 将图边 (i,j) 映射为 K 的边, 使用 vertex_map 进行转换
            edge_map[idx] = len(K_edges)
            K_edges.append((vertex_map[i], vertex_map[j]))

        # ---- 步骤3: 从 hierarchy 构建子复形 ----
        # 公理3: 层级结构诱导子复形划分.
        # subcomplexes: {层级名 → 顶点列表}, 每个层级对应 K 的一个子复形.
        subcomplexes: Dict[str, list] = {}
        if psi.hierarchy:
            node_levels: Dict[int, int] = psi.hierarchy.get("node_levels", {})
            for v, lv in node_levels.items():
                key: str = f"level_{lv}"
                if key not in subcomplexes:
                    subcomplexes[key] = []
                subcomplexes[key].append(v)

        # ---- 步骤4: 构造胞腔复形 K ----
        # 定义3(a): K = (vertices, edges, higher_cells, subcomplexes, level_labels)
        # higher_cells 初始为空, 由后续模块填充.
        K: SimplicialComplex = SimplicialComplex(
            vertices=list(range(n)),
            edges=K_edges,
            higher_cells=[],  # 高阶胞腔由后续编码器填充
            subcomplexes=subcomplexes,
            level_labels=psi.hierarchy.get("node_levels", {}) if psi.hierarchy else {},
        )

        return K, vertex_map, edge_map
