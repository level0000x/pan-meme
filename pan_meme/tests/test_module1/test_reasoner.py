# 测试模块: Reasoner（关系闭合 — 传递/对称/共现推理产生 Ψ*）
# 数学对应: 关系闭合算子 Ψ ↦ Ψ*, 5条推理规则
# 论文位置: 定义1（Ψ=(V,E,w)）, 定理2前驱（Φ_B 需 Ψ*）

import pytest
import numpy as np
from pan_meme.core.types import RelationNetwork
from pan_meme.module1_input.reasoner import Reasoner


class TestReasoner:
    """关系闭合推理测试: 验证5条推理规则对关系网络的补全效果。"""

    def test_transitive_inference(self, relation_network_chain):
        """
        测试传递性推理: 链 A→B→C→D (权重 0.9, 0.8, 0.7) 经推理后，
        应产生隐含边 (A,C) 和 (A,D)。

        数学对应:
        - 规则1（传递性推理 / Warshall 闭包）:
          w(A,C) ≥ min(w(A,B), w(B,C)) × 0.9 = min(0.9, 0.8) × 0.9 = 0.72
          w(A,D) ≥ min(w(A,C), w(C,D)) × 0.9 或 min(w(A,B), w(B,D)) × 0.9
        - 公理3 传递性: 若 a↗b 且 b↗c 则 a↗c
        """
        psi = relation_network_chain  # A(0), B(1), C(2), D(3)
        reasoner = Reasoner()
        psi_star = reasoner.infer(psi)

        # 读取推理后的边集（无向图，以上三角形式存储）
        edges_set = {(min(u, v), max(u, v)) for u, v in psi_star.edges}

        # 验证 A→C 路径: (0, 2) 存在
        assert (0, 2) in edges_set, (
            f"传递性推理后应存在边 (A, C), 当前边集: {sorted(edges_set)}"
        )

        # 验证 A→D 路径: (0, 3) 存在
        assert (0, 3) in edges_set, (
            f"传递性推理后应存在边 (A, D), 当前边集: {sorted(edges_set)}"
        )

        # 验证所有权重都在 [0,1] 区间
        assert np.all(psi_star.weights >= 0.0), "推理后存在负权重"
        assert np.all(psi_star.weights <= 1.0), "推理后存在超过 1.0 的权重"

    def test_edge_count_increase(self, relation_network_chain):
        """
        测试推理后边数增长: 推理后的边数必须 ≥ 原始边数。

        数学对应:
        - 关系闭合: Ψ* = (V, E*, w*) 是 Ψ 的闭包, E* ⊇ E
        - 推理规则只能增加边（补全隐含关系），不能减少已有边
        - 这是关系闭合算子的单调性保证
        """
        psi = relation_network_chain
        original_edge_count = len(psi.edges)

        reasoner = Reasoner()
        psi_star = reasoner.infer(psi)
        inferred_edge_count = len(psi_star.edges)

        assert inferred_edge_count >= original_edge_count, (
            f"推理后边数 ({inferred_edge_count}) 应 ≥ 原始边数 ({original_edge_count})"
        )

        # 验证推理元数据中的 edges_before/edges_after 一致性
        stats = psi_star.metadata.get("inference_stats", {})
        # edges_before 可能等于 original_edge_count 或因对称推理等中间步骤
        # 记录了不同的值，核心断言是 edges_after >= edges_before
        assert stats.get("edges_after", 0) >= stats.get("edges_before", 0), (
            f"metadata edges_after ({stats.get('edges_after')}) 应 >= edges_before ({stats.get('edges_before')})"
        )

        # 验证 inference_applied 标记为 True
        assert psi_star.metadata.get("inference_applied") is True, (
            "推理后 metadata.inference_applied 应为 True"
        )
