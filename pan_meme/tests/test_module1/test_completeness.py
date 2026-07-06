# 测试模块: CompletenessChecker（前提1 — 结构完整性诊断）
# 数学对应: 前提1（U 仅包含有可识别结构的信息）、前提0（人工介入）
# 论文位置: 前提1, 算法 A Step 1 后置

import pytest
import numpy as np
from pan_meme.core.types import (
    RelationNetwork, HierarchyTree, HierarchyNode,
    CompletenessReport, GapInfo,
)
from pan_meme.module1_input.completeness import CompletenessChecker


class TestCompleteness:
    """前提1 完整性测试: 验证结构完整性诊断的判定逻辑。"""

    def test_complete_graph(self, relation_network_complete):
        """
        测试完全图（K4）判定为完整: 对完全连通且深度 > 0 的网络，
        is_complete 应为 True。

        数学对应:
        - 前提1: K4 无孤立节点，最大连通分量 = 100% > 80% 阈值，层次深度 > 0
        - 三条判定: (1) CC占比 ≥ 80% ✓ (2) 孤立节点=0 ✓ (3) 深度>0 ✓
        - 结论: is_complete = True, gaps = [], requires_human = False
        """
        psi = relation_network_complete  # K4: 4节点, 6条边

        # 构造一个非零深度的层级树（depth > 0）
        tree = HierarchyTree(
            nodes=[
                HierarchyNode(token_idx=0, parent=1),
                HierarchyNode(token_idx=1, parent=None),
                HierarchyNode(token_idx=2, parent=1),
                HierarchyNode(token_idx=3, parent=2),
            ],
            root_indices=[1],
            depth=2,
        )

        checker = CompletenessChecker()
        report = checker.check(psi, tree)

        assert isinstance(report, CompletenessReport), (
            f"check 应返回 CompletenessReport, 实际 {type(report).__name__}"
        )
        assert report.is_complete is True, (
            f"K4 完全图应判定为完整, 实际 is_complete={report.is_complete}, "
            f"gaps={[(g.gap_type, g.location) for g in report.gaps]}"
        )
        assert len(report.gaps) == 0, (
            f"完全图 K4 应无空洞, 实际 {len(report.gaps)} 处: "
            f"{[g.gap_type for g in report.gaps]}"
        )
        assert report.requires_human is False, (
            f"完全图不应触发人工介入, 实际 requires_human={report.requires_human}"
        )

    def test_empty_input(self):
        """
        测试空输入: 0 个节点的网络应被判定为不完整，
        且触发人工介入（前提0）。

        数学对应:
        - 前提1: 0 节点的网络无结构可言 — is_complete = False
        - 前提0: 空洞 ≥ 2 → requires_human = True
        - 判定逻辑:
          (1) CC占比: 0/0 = 0.0 < 0.8 → structural_hole
          (2) 孤立节点 = 0 → 不触发 isolated_nodes
          (3) 深度 = 0 → flat_structure
          (4) 空洞数 = 2 ≥ 2 → requires_human = True
        """
        # 0 节点空网络
        psi = RelationNetwork(
            nodes=[],
            edges=[],
            weights=np.array([], dtype=np.float32),
            hierarchy={"levels": 0, "node_levels": {}},
        )

        # 空层级树
        tree = HierarchyTree(
            nodes=[],
            root_indices=[],
            depth=0,
        )

        checker = CompletenessChecker()
        report = checker.check(psi, tree)

        assert isinstance(report, CompletenessReport), (
            f"check 应返回 CompletenessReport, 实际 {type(report).__name__}"
        )
        assert report.is_complete is False, (
            f"空网络应判定为不完整, 实际 is_complete={report.is_complete}"
        )
        assert report.requires_human is True, (
            f"空网络应触发人工介入 (requires_human=True), "
            f"实际 requires_human={report.requires_human}"
        )
        assert len(report.gaps) >= 1, (
            f"空网络应检测到至少 1 处空洞, 实际 {len(report.gaps)} 处"
        )

        # 验证至少检测到了 flat_structure 空洞（深度 = 0）
        gap_types = [g.gap_type for g in report.gaps]
        assert "flat_structure" in gap_types, (
            f"空网络应检测到 flat_structure 空洞, 实际空洞类型: {gap_types}"
        )

        # 验证 human_request 非空（前提0: 人工补全请求已生成）
        assert report.human_request is not None, (
            "空网络应生成人工补全请求文本 (human_request)"
        )
        assert len(report.human_request) > 0, (
            f"human_request 不应为空字符串, 实际长度 {len(report.human_request)}"
        )
