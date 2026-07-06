# 测试模块: FieldEncoder（定义3(c) — 规则域 F → 标量场 ω）
# 数学对应: 定义3(c): ω[v] = Σ_{f∈F: v∈supp(f)} conf(f), clip [0,1]
# 论文位置: 定义3, 定理3

import pytest
import numpy as np
from pan_meme.core.types import RuleDef, SimplicialComplex
from pan_meme.module2_geo.field_encoder import FieldEncoder


class TestFieldEncoder:
    """定义3(c) 场编码测试: 验证规则域与标量场之间的 encode/decode 保真性。"""

    @pytest.fixture
    def sample_rules_and_K(self):
        """构建一组示例规则和胞腔复形 K 供编解码测试。"""
        rules = [
            RuleDef(
                pattern="scale_free_propagation: cv=4.5, top_3_hubs",
                support=[0, 1, 2],
                confidence=0.45,
                source="extracted",
            ),
            RuleDef(
                pattern="transitivity: 3_clique(1,2,3)",
                support=[1, 2, 3],
                confidence=0.83,
                source="extracted",
            ),
            RuleDef(
                pattern="field_intensity_2",
                support=[2],
                confidence=0.60,
                source="extracted",
            ),
        ]

        # 5 个顶点，5 条边的简单胞腔复形
        K = SimplicialComplex(
            vertices=[0, 1, 2, 3, 4],
            edges=[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)],
            subcomplexes={"level_0": [0, 1, 2, 3], "level_1": [4]},
            level_labels={0: 0, 1: 0, 2: 1, 3: 1, 4: 0},
        )
        return rules, K

    def test_encode_decode_roundtrip(self, sample_rules_and_K):
        """
        测试编解码往返: encode → decode 后，规则数量不变，
        且每条规则的 confidence 保留。

        数学对应:
        - 定义3(c): ω = FieldEncoder.encode(F, K), 逆映射 F' = FieldEncoder.decode(ω, K)
        - 双射性质: |F'| = |支持顶点集合|（每个非零场值顶点解码为一条规则）
        - confidence 保真: 每颗解码规则 confidence 等于该顶点的场值
        - source 标记: 解码规则标记 "decoded" 以与原始区分
        """
        rules, K = sample_rules_and_K

        # 正向编码: F → ω
        omega = FieldEncoder.encode(rules, K)

        # 验证 ω 的维度与顶点数一致
        assert len(omega) == len(K.vertices), (
            f"场向量维度不匹配: |ω|={len(omega)}, |V|={len(K.vertices)}"
        )
        assert omega.dtype == np.float32, (
            f"ω 的 dtype 应为 float32, 实际 {omega.dtype}"
        )

        # 验证 ω 的所有值在 [0, 1]（clip 保证）
        assert float(np.min(omega)) >= 0.0, f"场值存在负数: min={float(np.min(omega))}"
        assert float(np.max(omega)) <= 1.0, f"场值超过 1.0: max={float(np.max(omega))}"

        # 逆向解码: ω → F'
        decoded_rules = FieldEncoder.decode(omega, K)

        # 验证解码出的规则数量 = 非零场值顶点数
        nonzero_count = int(np.sum(omega > 0))
        assert len(decoded_rules) == nonzero_count, (
            f"解码规则数不匹配: {len(decoded_rules)} != {nonzero_count}（非零顶点数）"
        )

        # 验证每条解码规则的 confidence 等于对应顶点的场值
        for i, rule in enumerate(decoded_rules):
            assert rule.source == "decoded", (
                f"解码规则 source 应为 'decoded', 实际 '{rule.source}'"
            )
            # 规则 support 应包含对应顶点
            assert len(rule.support) == 1, (
                f"解码规则应仅有 1 个支撑元素, 实际 {len(rule.support)}"
            )
            vertex_idx = rule.support[0]
            expected_conf = float(omega[vertex_idx])
            assert np.isclose(rule.confidence, expected_conf, rtol=1e-6), (
                f"顶点 {vertex_idx} confidence 不匹配: "
                f"{rule.confidence} != {expected_conf}"
            )

        # 验证所有 confidence 在 [0, 1] 区间
        for rule in decoded_rules:
            assert 0.0 <= rule.confidence <= 1.0, (
                f"解码规则 confidence 越界: {rule.confidence}"
            )
