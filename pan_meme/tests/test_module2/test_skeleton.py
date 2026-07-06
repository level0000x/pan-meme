# 测试模块: SkeletonEncoder（定义3(a) — 图 → 胞腔复形同构映射）
# 数学对应: 定理3（Φ_C: M → G）, 定义3（K 是几何对象第一个组件）
# 论文位置: 定义3, 定理3

import pytest
from pan_meme.core.types import SimplicialComplex
from pan_meme.module2_geo.skeleton import SkeletonEncoder


class TestSkeletonEncoder:
    """定义3(a) 骨架编码测试: 验证 S → K 的顶点/边同构性。"""

    def test_vertex_count_matches(self, relation_network_chain):
        """
        测试顶点数一致: Ψ 有 n 个节点 → K 有 n 个顶点。

        数学对应:
        - 定义3(a): K 的 0-cells 与 V 一一对应, vertex_map 为恒等映射
        - |K.vertices| = |V| — 同构性保证基数不变
        - 这是 Φ_C 双射性质在顶点维度上的基础检验
        """
        psi = relation_network_chain  # 4 节点: A, B, C, D
        n = len(psi.nodes)

        K, vertex_map, edge_map = SkeletonEncoder.encode(psi)

        assert isinstance(K, SimplicialComplex), (
            f"encode 应返回 SimplicialComplex, 实际 {type(K).__name__}"
        )
        assert len(K.vertices) == n, (
            f"顶点数不匹配: K.vertices={len(K.vertices)}, V={n}"
        )

        # 验证 vertex_map 覆盖所有原图节点
        assert len(vertex_map) == n, (
            f"vertex_map 大小不匹配: {len(vertex_map)} != {n}"
        )
        for i in range(n):
            assert i in vertex_map, (
                f"原始节点 {i} 未在 vertex_map 中"
            )

    def test_edge_count_matches(self, relation_network_chain):
        """
        测试边数一致: Ψ 有 m 条边 → K 有 m 条边。

        数学对应:
        - 定义3(a): K 的 1-cells 与 E 一一对应, edge_map 为恒等映射
        - |K.edges| = |E| — 同构性保证边基数不变
        - 顶点索引在 K 边中保持一致: (i,j) ∈ E → (vertex_map[i], vertex_map[j]) ∈ K.edges
        """
        psi = relation_network_chain  # 3 条边: (0,1), (1,2), (2,3)
        m = len(psi.edges)

        K, vertex_map, edge_map = SkeletonEncoder.encode(psi)

        assert len(K.edges) == m, (
            f"边数不匹配: K.edges={len(K.edges)}, E={m}"
        )

        # 验证 edge_map 覆盖所有原始边
        assert len(edge_map) == m, (
            f"edge_map 大小不匹配: {len(edge_map)} != {m}"
        )
        for i in range(m):
            assert i in edge_map, (
                f"原始边 {i} 未在 edge_map 中"
            )

        # 验证 K 中每条边的顶点索引在合法范围内
        n = len(psi.nodes)
        for u, v in K.edges:
            assert 0 <= u < n, f"K 边顶点 {u} 越界 [0, {n})"
            assert 0 <= v < n, f"K 边顶点 {v} 越界 [0, {n})"
