# 模因项目 — 模块一：输入适配层
# 数学对应：公理1-3 + 定义1-2 + 前提0-2
# 核心组件：Tokenizer（公理1）+ 循环引擎（公理3）+ 关系提取器（前提2）
#          + 推理补全（关系闭合）+ 完整性判定（前提1）+ 概念组合（Louvain多层）
#          + 规则提取器 + 一致性检查器 + InputAdapter（主入口）
#
# PB 级引擎扩展（PB_ARCHITECTURE.md 第3节）：
#   RelationExtractor(use_lsh=True)      — LSH 近似加速，数学对应第3.1节
#     - use_lsh: 启用 MinHash LSH 近似关系提取（默认 False，向后兼容）
#     - lsh_bands / lsh_rows / lsh_window: LSH 超参数调优
#   Reasoner(use_star_closure=True)      — Star 稀疏传递闭包，数学对应第3.2节
#     - use_star_closure: 启用 Star 分解稀疏闭包（默认 False，向后兼容）
#     - star_max_iter / star_tol: 闭包收敛控制参数

from .tokenizer import Tokenizer
from .cycle import CycleEngine
from .relation_extractor import RelationExtractor
from .rule_extractor import RuleExtractor
from .consistency import ConsistencyChecker
from .adapter import InputConfig, InputAdapter

__all__ = [
    "Tokenizer",
    "CycleEngine",
    "RelationExtractor",
    "RuleExtractor",
    "ConsistencyChecker",
    "InputConfig",
    "InputAdapter",
]

# -------------------------------
# 尝试导入后续阶段组件 (Phase 5-6)
# 若对应文件尚未实现, 静默跳过
# -------------------------------
try:
    from .reasoner import Reasoner
    __all__.append("Reasoner")
except ImportError:
    pass

try:
    from .completeness import CompletenessChecker
    __all__.append("CompletenessChecker")
except ImportError:
    pass

try:
    from .concept_composer import ConceptComposer
    __all__.append("ConceptComposer")
except ImportError:
    pass
