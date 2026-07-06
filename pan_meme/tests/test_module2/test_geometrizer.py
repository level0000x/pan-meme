# 测试模块: Geometrizer（定理3 — Φ_C: M → G 双射 + 逆映射）
# 数学对应: 定理3（Φ_C: M → G 双射）、定义3（G = (K, g, ω, Γ, R)）
# 论文位置: 定理3, 定义3

import pytest
import numpy as np
from pan_meme.core.types import (
    MathModel, RelationNetwork, RuleDef, ConstraintDef,
    PipelineData,
)
from pan_meme.module2_geo.geometrizer import Geometrizer


class TestGeometrizer:
    """定理3 几何化编排测试: 验证 Φ_C 双射性质 — encode/decode 结构保真。"""

    @pytest.fixture
    def sample_math_model(self):
        """构建一个最小可编码的 MathModel 供几何化测试。"""
        # 简单三角形关系网络
        psi = RelationNetwork(
            nodes=["A", "B", "C"],
            edges=[(0, 1), (1, 2), (0, 2)],
            weights=np.array([0.9, 0.85, 0.95], dtype=np.float32),
            hierarchy={
                "levels": 1,
                "node_levels": {0: 0, 1: 1, 2: 1},
                "parent_map": {0: 1, 2: 1},
            },
            metadata={"input_type": "synthetic", "node_count": 3, "edge_count": 3},
        )

        rules = [
            RuleDef(pattern="transitivity: 3_clique(0,1,2)", support=[0, 1, 2],
                    confidence=0.9, source="extracted"),
        ]

        constraints = [
            ConstraintDef(
                condition="node_connectivity_min",
                domain=[],
                description="无孤立节点",
                confidence=1.0,
                source="extracted",
            ),
        ]

        return MathModel(
            structure=psi,
            rules=rules,
            constraints=constraints,
            metadata={"rule_count": 1, "constraint_count": 1},
        )

    def test_encode_decode_mathmodel(self, sample_math_model):
        """
        测试数学模型几何化往返: M → encode → decode → M',
        结构域 structure 应保持一致。

        数学对应:
        - 定理3: Φ_C: M → G 是双射, 因此 decode(encode(M)) ≅ M
        - 结构一致性: S' 的节点数和边数应与 S 相同
        - 这是 Φ_C 双射性质的端到端验证
        """
        gzr = Geometrizer()

        # Step 1: 构建 PipelineData, 注入 math_model
        data = PipelineData(
            input="test_input",
            math_model=sample_math_model,
        )

        # Step 2: 正向编码 M → G
        data_encoded = gzr.encode(data)

        # 验证几何对象 G 已生成
        assert data_encoded.geo_object is not None, (
            "encode 后 geo_object 不应为 None"
        )
        G = data_encoded.geo_object

        # 验证 G 的组件非空
        assert G.K is not None, "胞腔复形 K 不应为 None"
        assert G.g is not None, "度量向量 g 不应为 None"
        assert G.omega is not None, "标量场 ω 不应为 None"
        assert G.Gamma is not None, "不变量字典 Γ 不应为 None"
        assert G.R is not None, "可逆性元数据包 R 不应为 None"

        # 验证 K 的顶点数与原始结构一致
        original_n = len(sample_math_model.structure.nodes)
        assert len(G.K.vertices) == original_n, (
            f"K 顶点数不匹配: {len(G.K.vertices)} != {original_n}"
        )

        # 验证 K 的边数与原始结构一致
        original_m = len(sample_math_model.structure.edges)
        assert len(G.K.edges) == original_m, (
            f"K 边数不匹配: {len(G.K.edges)} != {original_m}"
        )

        # 验证度量向量 g 的维度与 K 边数一致
        assert len(G.g) == len(G.K.edges), (
            f"g 维度不匹配: |g|={len(G.g)}, |K.edges|={len(G.K.edges)}"
        )

        # 验证场向量 ω 的维度与 K 顶点数一致
        assert len(G.omega) == len(G.K.vertices), (
            f"ω 维度不匹配: |ω|={len(G.omega)}, |K.vertices|={len(G.K.vertices)}"
        )

        # Step 3: 逆向解码 G → M'
        M_prime = gzr.decode(G)

        assert isinstance(M_prime, MathModel), (
            f"decode 应返回 MathModel, 实际 {type(M_prime).__name__}"
        )

        # 验证结构一致性: 节点数不变
        S_prime = M_prime.structure
        assert len(S_prime.nodes) == original_n, (
            f"解码后节点数不匹配: {len(S_prime.nodes)} != {original_n}"
        )

        # 验证结构一致性: 边数不变
        assert len(S_prime.edges) == original_m, (
            f"解码后边数不匹配: {len(S_prime.edges)} != {original_m}"
        )

        # 验证解码标记
        assert M_prime.metadata.get("decoded_from_geo") is True, (
            "解码后的 MathModel 应标记 'decoded_from_geo': True"
        )

        # 验证所有权重仍在 [0,1] 区间
        if len(S_prime.weights) > 0:
            assert np.all(S_prime.weights >= 0.0), "解码后存在负权重"
            assert np.all(S_prime.weights <= 1.0), "解码后存在超过 1.0 的权重"
