# 测试模块: CycleEngine（公理3 + 前提0 — ↑↓ 循环层级树构建）
# 数学对应: 公理3（↑/↓ 互逆操作）、前提0（有限层级终止）
# 论文位置: 附录 D.1-D.2 公理系统, 算法 A Step 1

import pytest
from pan_meme.core.types import Token
from pan_meme.module1_input.cycle import CycleEngine
from pan_meme.module1_input.tokenizer import Tokenizer


class TestCycleEngine:
    """公理3 + 前提0 循环测试: 验证 ↑↓ 循环产出的层级树满足拓扑约束。"""

    def test_cycle_converge(self, sample_text_cn):
        """
        测试循环收敛: 对中文文本运行 ↑↓ 循环，层级树应在有限轮内收敛。

        数学对应:
        - 公理3: ↑↓ 互逆操作构建层级结构
        - 前提0: 层级结构在有限轮内终止 — terminated_by ∈ {"converge", "fixed"}
        - 收敛条件: rounds ≤ cycle_max_rounds (默认10) 或自动收敛检测

        验证:
        - terminated_by == "converge" 或 rounds ≤ 3
          （中文文本在内置上下位词典匹配下通常在 1-2 轮内收敛）
        """
        # 调用公理1的分词器获取 Token 列表
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(sample_text_cn)

        # 启动 ↑↓ 循环引擎
        engine = CycleEngine()
        tree = engine.run(tokens, mode="converge", max_rounds=10)

        # 验证终止方式
        assert tree.terminated_by == "converge", (
            f"期望 terminated_by='converge', 实际 '{tree.terminated_by}'"
        )

        # 验证轮数在合理范围内（有限层级保证：不超过 max_rounds）
        assert tree.rounds <= 10, (
            f"循环轮数超出最大限制: {tree.rounds} > 10"
        )

    def test_tree_has_roots(self, sample_text_cn):
        """
        测试层级树的根节点集合: root_indices 应为非空列表。

        数学对应:
        - 公理2: 未归类元素（parent=None）形成根节点集合
        - 对于非空输入，应至少有 1 个顶层元素（root_indices 非空）
        - 根节点数是层级结构的基元度量
        """
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(sample_text_cn)
        engine = CycleEngine()
        tree = engine.run(tokens, mode="converge", max_rounds=10)

        assert isinstance(tree.root_indices, list), (
            "root_indices 应为列表类型"
        )
        assert len(tree.root_indices) > 0, (
            "层级树 root_indices 不应为空 — 非空输入至少有一个顶层节点"
        )

        # 验证每个根节点的 parent 确实为 None
        for root_idx in tree.root_indices:
            node = tree.nodes[root_idx]
            assert node.parent is None, (
                f"根节点 {root_idx} 的 parent 应为 None, 实际 {node.parent}"
            )

    def test_termination_record(self, sample_text_cn):
        """
        测试终止记录: termination_record 应包含 mode/reason/total_rounds 三个必填字段。

        数学对应:
        - 前提0: 终止记录编码有限层级终止的完整信息
        - termination_record = {mode, reason, final_level, total_rounds, converged_at_round}
        - 所有字段必须存在，下游模块（适配器、日志）依赖这些字段
        """
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(sample_text_cn)
        engine = CycleEngine()
        tree = engine.run(tokens, mode="converge", max_rounds=10)

        record = tree.termination_record
        assert isinstance(record, dict), (
            f"termination_record 应为 dict, 实际 {type(record).__name__}"
        )

        # 验证三个必填字段存在且类型正确
        assert "mode" in record, (
            f"termination_record 缺少 'mode' 字段, 现有: {list(record.keys())}"
        )
        assert "reason" in record, (
            f"termination_record 缺少 'reason' 字段"
        )
        assert "total_rounds" in record, (
            f"termination_record 缺少 'total_rounds' 字段"
        )

        assert isinstance(record["mode"], str), (
            f"mode 应为 str, 实际 {type(record['mode']).__name__}"
        )
        assert isinstance(record["reason"], str), (
            f"reason 应为 str, 实际 {type(record['reason']).__name__}"
        )
        assert isinstance(record["total_rounds"], int), (
            f"total_rounds 应为 int, 实际 {type(record['total_rounds']).__name__}"
        )

        # 验证 total_rounds 与 tree.rounds 一致
        assert record["total_rounds"] == tree.rounds, (
            f"total_rounds 不一致: record={record['total_rounds']}, tree.rounds={tree.rounds}"
        )
