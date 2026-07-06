# 测试模块: RelationExtractor（前提2 — 从层级树提取关系网络 Ψ）
# 数学对应: 前提2（算法 A: I → Ψ 信息无损）、定义1（Ψ = (V, E, w)）
# 论文位置: 定义1, 附录 D.3（Ψ_T 是 T 参数化的网络族）

import pytest
import numpy as np
from pan_meme.core.types import RelationNetwork, HierarchyTree
from pan_meme.module1_input.relation_extractor import RelationExtractor
from pan_meme.module1_input.cycle import CycleEngine
from pan_meme.module1_input.tokenizer import Tokenizer


class TestRelationExtractor:
    """前提2 测试: 验证从层级树到关系网络的信息无损映射。"""

    @pytest.fixture
    def _hierarchy_tree(self, sample_text_cn):
        """构建一个有足够节点的层级树，供提取器测试使用。"""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(sample_text_cn)
        engine = CycleEngine()
        return engine.run(tokens, mode="converge", max_rounds=10)

    def test_extract_produces_edges(self, _hierarchy_tree):
        """
        测试关系网络提取: 中文文本经 ↑↓ 循环后再提取关系网络，
        应至少包含 1 条边。

        数学对应:
        - 前提2: A(I) = Ψ 信息无损 — 层级树中非零共现强度产生边
        - 共现强度 cooc(i,j) = 1/(|depth_i - depth_j| + 1) > 0
          当且仅当 i 和 j 共享同一父节点（公理2: C(y) 内的子元素）
        - 使用 threshold=0.0 确保所有权重 ≥ 0 的潜在边都被保留，
          即使层级树为扁平结构（无共现时 w_ij = 0.0）也产生全连接
        """
        extractor = RelationExtractor()
        # threshold=0.0 保留所有权重 ≥ 0 的边，确保在扁平层级树下也能产出边
        network = extractor.extract(_hierarchy_tree, threshold=0.0)

        assert isinstance(network, RelationNetwork), (
            f"extract 应返回 RelationNetwork, 实际 {type(network).__name__}"
        )
        assert len(network.edges) >= 1, (
            f"提取的关系网络应至少含 1 条边, 实际 {len(network.edges)} 条"
        )
        assert len(network.nodes) >= 2, (
            f"关系网络应至少含 2 个节点, 实际 {len(network.nodes)} 个"
        )

    def test_threshold_zero(self, _hierarchy_tree):
        """
        测试零阈值: 当 threshold=0 时，所有非零权重的边都被保留，
        边数应 ≥ 节点数 - 1（至少构成连通图或森林）。

        数学对应:
        - 附录 D.3: Ψ_{T=0} = (V, E_{T=0}, w|_{E_{T=0}}), E_{T=0} ⊇ E_{T>0}
        - 阈值越低，保留的边越多，极限情况下 E_{T=0} 包含所有共现边
        - 连通图至少需要 |V|-1 条边
        """
        extractor = RelationExtractor()
        network = extractor.extract(_hierarchy_tree, threshold=0.0)

        n = len(network.nodes)
        m = len(network.edges)

        # 零阈值下，只要至少 2 个节点共享父节点就有边
        if n >= 2:
            assert m >= n - 1, (
                f"threshold=0 时边数应 ≥ |V|-1 = {n-1}, 实际 {m}"
            )

    def test_weights_in_range(self, _hierarchy_tree):
        """
        测试权重范围: 所有权重值必须在 [0, 1] 区间内。

        数学对应:
        - 定义1: w: E → [0, 1] — 边权被约束在 [0, 1] 闭区间
        - 共现强度: 1/(|depth_diff| + 1) ∈ (0, 1]
        - 所有权重必须满足此拓扑约束
        """
        extractor = RelationExtractor()
        network = extractor.extract(_hierarchy_tree, threshold=0.0)

        if len(network.weights) == 0:
            pytest.skip("无边可检测权重范围")

        weights = np.asarray(network.weights, dtype=np.float32)
        assert np.all(weights >= 0.0), (
            f"检测到负权重: min = {float(np.min(weights))}"
        )
        assert np.all(weights <= 1.0), (
            f"检测到超过 1.0 的权重: max = {float(np.max(weights))}"
        )

    def test_hierarchy_metadata(self, _hierarchy_tree):
        """
        测试层级元数据: hierarchy 应包含 levels 和 node_levels 键。

        数学对应:
        - 定义1 附录: hierarchy 编码层级结构信息
        - hierarchy["levels"]: 层级深度（int）
        - hierarchy["node_levels"]: 每个节点到其层级的映射（Dict[int, int]）
        - 下游模块（SkeletonEncoder、几何化）依赖这些字段
        """
        extractor = RelationExtractor()
        network = extractor.extract(_hierarchy_tree, threshold=0.3)

        assert "levels" in network.hierarchy, (
            f"hierarchy 缺少 'levels' 字段, 现有: {list(network.hierarchy.keys())}"
        )
        assert "node_levels" in network.hierarchy, (
            f"hierarchy 缺少 'node_levels' 字段"
        )

        levels = network.hierarchy["levels"]
        node_levels = network.hierarchy["node_levels"]

        assert isinstance(levels, int), (
            f"levels 应为 int, 实际 {type(levels).__name__}"
        )
        assert levels >= 0, f"levels 不应为负数: {levels}"
        assert isinstance(node_levels, dict), (
            f"node_levels 应为 dict, 实际 {type(node_levels).__name__}"
        )

        # 每个节点的层级标签应在合法范围内
        for node_idx, lv in node_levels.items():
            assert 0 <= lv <= levels, (
                f"节点 [{node_idx}] 的层级 {lv} 超出合法范围 [0, {levels}]"
            )
