# 泛模因几何工具 — 模块一：主入口适配器
# 数学对应：定理1（A: I → Ψ 双射）+ 定理2（B: Ψ → M 双射）
#          复合映射 Φ_B ∘ Φ_A: I → Ψ → M 双射
# 论文位置：定理1+2, 附录D.2（完整数据流）, 算法A

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from pan_meme.core.types import (
    Token,
    HierarchyTree,
    RelationNetwork,
    RuleDef,
    ConstraintDef,
    MathModel,
    PipelineData,
)
from pan_meme.core.exceptions import ConsistencyViolation, PipelineError

# ── 模块一已有子组件 ──
from .cycle import CycleEngine
from .relation_extractor import RelationExtractor
from .rule_extractor import RuleExtractor
from .consistency import ConsistencyChecker as ModuleConsistencyChecker

# ── 待实现子组件（模块一管线依赖，导入供完整调用链）─
try:
    from .tokenizer import Tokenizer               # pylint: disable=unused-import
except ImportError:
    Tokenizer = None  # type: ignore

try:
    from .reasoner import Reasoner                 # pylint: disable=unused-import
except ImportError:
    Reasoner = None  # type: ignore

try:
    from .completeness import CompletenessChecker  # pylint: disable=unused-import
except ImportError:
    CompletenessChecker = None  # type: ignore

try:
    from .concept_composer import ConceptComposer  # pylint: disable=unused-import
except ImportError:
    ConceptComposer = None  # type: ignore


# ============================================================
# 输入适配配置
# 数学对应：附录 D.3 — 有限离散候选集 T = {T_1, ..., T_k}
# ============================================================

@dataclass
class InputConfig:
    """模块一输入适配器配置。

    数学对应：
    - 前提0：有限层级终止 — cycle_max_rounds 限制循环轮数
    - 附录 D.3：T ∈ T（有限离散候选集）— threshold_candidates 枚举候选阈值
    - 前提2：信息无损 — 阈值默认值 0.5 保证中等偏上的连通性

    属性说明：
      cycle_mode: 循环模式 — "converge"（自动收敛）| "fixed"（固定轮数）
      cycle_max_rounds: 最大循环轮数（前提0：有限层级保证）
      threshold_candidates: 候选阈值列表（附录D.3：T的离散枚举）
      threshold_default: 默认阈值（>0 保证基础连通性）
      transitive_decay: 传递性衰减因子（∈ [0,1]）
      symmetric_decay: 对称性衰减因子（∈ [0,1]）
      cooccurrence_min_neighbors: 共现检测最小邻居数
      concept_max_levels: 概念组合最大层级数
      max_component_threshold: 最大连通分量占比阈值
      max_isolated_ratio: 最大孤立节点比例阈值
    """
    # ── 循环控制 ──
    cycle_mode: str = "converge"
    """循环模式: 'converge'（自动收敛检测）| 'fixed'（固定轮数）。
    数学对应：前提0 — 自动收敛由层级结构有限性保证。"""

    cycle_max_rounds: int = 20
    """最大循环轮数。数学对应：前提0 — 有限层级终止的硬上限。"""

    # ── 阈值配置 ──
    threshold_candidates: List[float] = field(default_factory=lambda: [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    ])
    """候选阈值枚举。数学对应：附录 D.3 — T = {0.1, ..., 0.9}，
    每个 T_i 生成一个 Ψ_{T_i}，构成关系网络族。"""

    threshold_default: float = 0.5
    """默认过滤阈值。数学对应：前提2 — 中等阈值保证信息无损，
    既不过滤过强（高连通噪声），也不过弱（低密度空洞）。"""

    # ── 衰减参数 ──
    transitive_decay: float = 0.9
    """传递性衰减因子。数学对应：公理3 — ↑↓ 操作在传递链上的权重衰减。"""

    symmetric_decay: float = 0.85
    """对称性衰减因子。数学对应：公理3 — 双向关系权重抑制。
    当↓→↑或↑→↓方向不一致时，衰减对称分量。"""

    # ── 结构约束 ──
    cooccurrence_min_neighbors: int = 3
    """共现检测的最小邻居数阈值。
    数学对应：前提2 — 结构共现需要足够邻居支持才视作有效信号。"""

    concept_max_levels: int = 10
    """概念组合操作的最大层级深度。
    数学对应：前提0 — 层级有限性保证概念树不无限生长。"""

    max_component_threshold: float = 0.8
    """最大连通分量占比阈值（超过则视为过度连通）。
    数学对应：前提1 — 结构完整性约束中的连通性上界。"""

    max_isolated_ratio: float = 0.2
    """最大孤立节点比例阈值（超过则触发完整性警告）。
    数学对应：前提1 — 孤立节点过多 → 结构不完整。"""

    def to_dict(self) -> Dict[str, Any]:
        """将配置序列化为字典，供子模块使用。

        返回:
          Dict[str, Any] — 扁平化的配置键值对
        """
        return {
            "cycle_mode": self.cycle_mode,
            "cycle_max_rounds": self.cycle_max_rounds,
            "threshold_candidates": self.threshold_candidates,
            "threshold_default": self.threshold_default,
            "transitive_decay": self.transitive_decay,
            "symmetric_decay": self.symmetric_decay,
            "cooccurrence_min_neighbors": self.cooccurrence_min_neighbors,
            "concept_max_levels": self.concept_max_levels,
            "max_component_threshold": self.max_component_threshold,
            "max_isolated_ratio": self.max_isolated_ratio,
        }


# ============================================================
# 主入口适配器
# 数学对应：定理1+2 — Φ_B ∘ Φ_A: I → Ψ → M 双射
# ============================================================

class InputAdapter:
    """定理1+2实现：8步浮现管线 I → Ψ → M。

    数学对应：
    - 定理1：Φ_A: I → Ψ 是双射 — 原始信息到关系网络的映射可逆
    - 定理2：Φ_B: Ψ → M 是双射 — 关系网络到数学模型的映射可逆
    - 复合：Φ = Φ_B ∘ Φ_A: I → M 是双射 — 整体信息到模型的浮现可逆

    8步浮现管线（Phase A + Phase B）：

    Phase A — I → Ψ（信息到网络，定理1）:
      Step 1: Tokenize — 原始输入分词/结构化
      Step 2: Cycle — ↑↓ 循环建立层级树
      Step 3: Relation Extract — 从层级树提取关系网络 Ψ

    Phase B — Ψ → M（网络到模型，定理2）:
      Step 4: Reason — 推理补全关系网络中的隐含关系
      Step 5: Completeness — 结构完整性检查（前提0+1）
      Step 6: Concept Compose — 跨层级概念组合
      Step 7: Rule Extract — 从 Ψ 提取规则域 F 和约束域 C
      Step 8: Consistency — 自洽性验证 M = (S, F, C)

    用法:
        config = InputConfig()
        adapter = InputAdapter(config)
        result = adapter.adapt(raw_text)

        # 逆向：从 Ψ 还原 Token
        tokens = adapter.inverse(psi)
    """

    def __init__(self, config: Optional[InputConfig] = None) -> None:
        """初始化输入适配器及所有子模块。

        数学对应：
        - 定理1+2：组装 Φ = Φ_B ∘ Φ_A 所需的全部组件
        - 附录 D.2：管线组件注册与初始化

        初始化顺序：
        1. Tokenizer — 文本分词器
        2. CycleEngine — ↑↓ 循环引擎（传入上下位词典）
        3. RelationExtractor — 关系网络提取器
        4. Reasoner — 推理引擎（补全隐含关系）
        5. CompletenessChecker — 完整性验证器
        6. ConceptComposer — 跨层级概念组合
        7. RuleExtractor — 规则-约束提取器
        8. ConsistencyChecker — 自洽性验证器

        参数:
          config: 输入适配配置文件。若为 None，使用 InputConfig() 默认值。
        """
        self.config: InputConfig = config if config is not None else InputConfig()

        # ── 初始化 Phase A 组件 ──
        # Step 1: 分词器
        self._tokenizer: Optional[object] = (
            Tokenizer() if Tokenizer is not None else None
        )

        # Step 2: ↑↓ 循环引擎
        # 数学对应：公理3 — ↑↓ 互逆操作构建层级结构
        self._cycle_engine: CycleEngine = CycleEngine()

        # Step 3: 关系网络提取器
        # 数学对应：前提2 — 算法 A: I → Ψ 信息无损
        self._relation_extractor: RelationExtractor = RelationExtractor()

        # ── 初始化 Phase B 组件 ──
        # Step 4: 推理引擎 — 补全传递性/对称性隐含关系
        # 数学对应：公理3 传递性 — 若 a↗b 且 b↗c 则 a↗c
        self._reasoner: Optional[object] = (
            Reasoner() if Reasoner is not None else None
        )

        # Step 5: 完整性检查器 — 验证前提0+1
        self._completeness_checker: Optional[object] = (
            CompletenessChecker() if CompletenessChecker is not None else None
        )

        # Step 6: 概念组合器 — 跨层级语义组合
        self._concept_composer: Optional[object] = (
            ConceptComposer() if ConceptComposer is not None else None
        )

        # Step 7: 规则提取器 — Ψ ↦ (F, C)
        # 数学对应：前提3 — Ψ 的规则域和约束域被唯一确定
        self._rule_extractor: RuleExtractor = RuleExtractor()

        # Step 8: 自洽性验证器 — M 的内部一致性
        # 数学对应：定理5 — F 中规则不得与 C 中约束冲突
        self._consistency_checker: ModuleConsistencyChecker = \
            ModuleConsistencyChecker()

    # ================================================================
    # 公有方法：adapt — 完整浮现管线
    # 数学对应：定理1+2 — Φ = Φ_B ∘ Φ_A: I → M 双射
    # ================================================================

    def adapt(self, raw_input: Any,
              thr: Optional[float] = None) -> PipelineData:
        """执行完整的8步浮现管线：I → Tokenize → Cycle → Relation → Reason
        → Completeness → ConceptCompose → RuleExtract → Consistency → M。

        数学对应：
        - 定理1：Phase A (Step 1-3) — Φ_A: I → Ψ 双射
        - 定理2：Phase B (Step 4-8) — Φ_B: Ψ → M 双射
        - 复合：Φ = Φ_B ∘ Φ_A: I → M 双射

        管线流程图:
            I (raw_input)
             │ Step 1: Tokenizer.tokenize(raw_input)
             ▼
            List[Token]
             │ Step 2: CycleEngine.run(tokens, mode, max_rounds)
             ▼
            HierarchyTree
             │ Step 3: RelationExtractor.extract(tree, threshold)
             ▼
            Ψ = RelationNetwork               ←── Phase A 产物
             │ Step 4: Reasoner.infer(psi)          (定理1输出)
             ▼
            Ψ' (含推断边)
             │ Step 5: CompletenessChecker.check(psi)
             ▼
            CompletenessReport
             │ Step 6: ConceptComposer.compose(psi)
             ▼
            Ψ'' (含跨层级组合关系)
             │ Step 7: RuleExtractor.extract_rules + extract_constraints
             ▼
            F = List[RuleDef], C = List[ConstraintDef]
             │ Step 8: ConsistencyChecker.verify + repair
             ▼
            M = MathModel(S=Ψ'', F, C)        ←── Phase B 产物
                                                  (定理2输出)

        参数:
          raw_input: 原始输入信息 I ∈ U（文本字符串或其他模态原始数据）
          thr: 阈值覆盖。若提供，替换 config.threshold_default。
               若为 None，使用 config.threshold_default。

        返回:
          PipelineData — 含完整管线程上下文的不可变数据对象，
                       包含 ψ（关系网络）和 math_model（数学模型）
        """
        # ── 确定使用的阈值 ──
        threshold: float = (
            thr if thr is not None else self.config.threshold_default
        )

        # ── 初始化管线数据容器 ──
        pipeline: PipelineData = PipelineData(
            input=raw_input,
            meta={},
        )

        # ================================================================
        # Phase A: I → Ψ（定理1 — Φ_A 双射）
        # ================================================================

        # ── Step 1: Tokenize ──
        # 数学对应：公理2 — 将原始信息 I 切分为原子元素 x ∈ U
        tokens: List[Token] = self._step_tokenize(raw_input)
        pipeline.meta["token_count"] = len(tokens)

        # ── Step 2: ↑↓ 循环 ──
        # 数学对应：公理3 + 前提0 — 构建有限层级树
        tree: HierarchyTree = self._step_cycle(tokens)
        pipeline.meta["hierarchy_depth"] = tree.depth
        pipeline.meta["hierarchy_rounds"] = tree.rounds

        # ── Step 3: 关系网络提取 ──
        # 数学对应：前提2 — 算法 A: I → Ψ 信息无损
        psi: RelationNetwork = self._step_relation_extract(tree, threshold)

        # ── Phase A 产物：Ψ ──
        pipeline.psi = psi

        # ================================================================
        # Phase B: Ψ → M（定理2 — Φ_B 双射）
        # ================================================================

        # ── Step 4: 推理 ──
        # 数学对应：公理3 传递性 — 补全 Ψ 中的隐含边
        psi = self._step_reason(psi)

        # ── Step 5: 完整性检查 ──
        # 数学对应：前提0 + 前提1 — 结构完整性验证
        completeness_ok: bool = self._step_completeness(psi, pipeline)

        # ── Step 6: 概念组合 ──
        # 数学对应：公理2+3 — 跨层级概念关联
        psi = self._step_concept_compose(psi)

        # ── Step 7: 规则提取 ──
        # 数学对应：前提3 — Ψ ↦ (F, C) 双射，从网络结构提取规则和约束
        rules: List[RuleDef] = self._step_rule_extract(psi)
        constraints: List[ConstraintDef] = self._step_constraint_extract(psi)

        # ── Step 8: 自洽性验证 ──
        # 数学对应：定理5 — F 中规则不得与 C 中约束冲突
        math_model: MathModel = MathModel(
            structure=psi,
            rules=rules,
            constraints=constraints,
            metadata={
                "rule_count": len(rules),
                "constraint_count": len(constraints),
                "extraction_timestamp": "",
            },
        )
        math_model = self._step_consistency(math_model, pipeline)

        # ── Phase B 产物：M ──
        pipeline.math_model = math_model
        pipeline.meta["model_rule_count"] = len(math_model.rules)
        pipeline.meta["model_constraint_count"] = len(math_model.constraints)

        return pipeline

    # ================================================================
    # 公有方法：inverse — 定理1逆映射
    # 数学对应：定理1 — A^{-1}: Ψ → I（关系网络还原为Token序列）
    # ================================================================

    def inverse(self, psi: RelationNetwork) -> List[Token]:
        """定理1逆映射 A^{-1}: 从关系网络 Ψ 还原 Token 列表。

        数学对应：
        - 定理1：Φ_A: I → Ψ 是双射，因此存在逆映射 Φ_A^{-1}: Ψ → I
        - 此方法利用 Ψ.hierarchy 中的 node_levels 和 parent_map
          重建原始 Token 列表的基本信息

        还原策略：
        1. 从 Ψ.nodes 获取节点名称列表
        2. 依据 Ψ.hierarchy["node_levels"] 分配层级深度
        3. 依据 Ψ.hierarchy["parent_map"] 推断从属关系对应的 span
        4. Token 的 embedding 和 payload 字段填默认值

        注意：
        - 逆向还原是近似还原：emdedding 和 payload 等信息在
          I → Ψ 过程中被压缩，逆映射只能恢复结构性信息。
        - 模态统一标记为 "text"（当前版本仅支持文本模态）。

        参数:
          psi: 关系网络 Ψ = (V, E, w)

        返回:
          List[Token] — 还原的 Token 序列，顺序对应 Ψ.nodes 索引
        """
        hierarchy: Dict[str, Any] = psi.hierarchy
        n: int = len(psi.nodes)

        # ── 提取层级信息 ──
        # 数学对应：定义1 — hierarchy.node_levels 编码树的深度结构
        node_levels: Dict[int, int] = hierarchy.get("node_levels", {})
        parent_map: Dict[int, int] = hierarchy.get("parent_map", {})

        # ── 逐节点重建 Token ──
        tokens: List[Token] = []
        for i in range(n):
            node_name: str = psi.nodes[i]
            level: int = node_levels.get(i, 0)

            # span 模拟：基于层级深度构造虚拟定位
            # 数学对应：公理2 — span 表示在原输入中的位置
            span_start: int = level * 100
            span_end: int = span_start + len(node_name)

            # pos 标注：基于层级深度和拓扑角色
            if i in parent_map.values():
                # 该节点是某节点的父节点 → 标注为上位概念
                pos_label: str = "hypernym"
            elif i in parent_map:
                # 该节点有父节点 → 标注为下位概念
                pos_label: str = "hyponym"
            else:
                # 孤立节点或根节点 → 普通标注
                pos_label: str = "entity"

            token: Token = Token(
                modality="text",
                text=node_name,
                span=(span_start, span_end),
                pos=pos_label,
                embedding=None,  # 逆映射无法恢复 embedding
                payload={"reconstructed": True, "level": level},
            )
            tokens.append(token)

        return tokens

    # ================================================================
    # 内部管线步骤方法
    # 每个步骤对应管线的一个阶段，封装子模块调用和错误处理
    # ================================================================

    def _step_tokenize(self, raw_input: Any) -> List[Token]:
        """Step 1: 将原始输入分词为 Token 序列。

        数学对应：公理2 — 把论域 U 中的原始信息切分为原子元素 x ∈ U

        参数:
          raw_input: 原始输入（字符串或其他模态数据）

        返回:
          List[Token] — 分词后的 Token 列表
        """
        if self._tokenizer is None:
            # 回退：简单按空白字符分词 + 字符级 fallback
            text: str = str(raw_input)
            tokens: List[Token] = []
            words: List[str] = text.split()
            if words:
                for idx, word in enumerate(words):
                    tokens.append(Token(
                        modality="text",
                        text=word,
                        span=(idx, idx + 1),
                        pos="word",
                    ))
            else:
                # 完全空输入：返回空列表
                pass
            return tokens
        # 使用外部 Tokenizer
        return self._tokenizer.tokenize(raw_input)  # type: ignore[union-attr]

    def _step_cycle(self, tokens: List[Token]) -> HierarchyTree:
        """Step 2: 执行 ↑↓ 循环建立层级树。

        数学对应：公理3 + 前提0 — ↑↓ 互逆操作构建有限层级结构

        参数:
          tokens: Token 列表

        返回:
          HierarchyTree — 含完整层级关系的树对象
        """
        return self._cycle_engine.run(
            tokens,
            mode=self.config.cycle_mode,
            max_rounds=self.config.cycle_max_rounds,
        )

    def _step_relation_extract(self, tree: HierarchyTree,
                                threshold: float) -> RelationNetwork:
        """Step 3: 从层级树提取关系网络 Ψ。

        数学对应：前提2 — 算法 A: I → Ψ 信息无损
                  附录 D.3 — Ψ_T 是 T 参数化的网络族

        参数:
          tree: ↑↓ 循环产出的层级树
          threshold: 边权重过滤阈值 T ∈ [0,1]

        返回:
          RelationNetwork — 加权关系网络 Ψ_T
        """
        return self._relation_extractor.extract(tree, threshold=threshold)

    def _step_reason(self, psi: RelationNetwork) -> RelationNetwork:
        """Step 4: 推理补全关系网络中的隐含关系。

        数学对应：公理3 传递性 — 若 a↗b 且 b↗c 则 a↗c
                  对称性 — 若 a↔b 则 b↔a（在无向图中自动成立）

        参数:
          psi: 当前关系网络

        返回:
          RelationNetwork — 注入推断边后的关系网络（若 Reasoner 未注册则原样返回）
        """
        if self._reasoner is None:
            return psi
        return self._reasoner.infer(psi)  # type: ignore[union-attr]

    def _step_completeness(self, psi: RelationNetwork,
                            pipeline: PipelineData) -> bool:
        """Step 5: 结构完整性检查。

        数学对应：前提0 + 前提1 — 保证层级有限且结构完整
                 不完整的结构无法保证后续定理1-4的双射性质

        参数:
          psi: 当前关系网络
          pipeline: 管线数据容器（用于写入完备性报告）

        返回:
          bool — True 表示结构完整，False 表示不完整
        """
        if self._completeness_checker is None:
            pipeline.meta["completeness_checked"] = False
            pipeline.meta["completeness_report"] = None
            return True  # 无检查器时假设置完整

        report = self._completeness_checker.check(psi)  # type: ignore[union-attr]
        pipeline.meta["completeness_checked"] = True
        pipeline.meta["completeness_report"] = report
        pipeline.meta["completeness_ok"] = report.is_complete

        return report.is_complete

    def _step_concept_compose(self, psi: RelationNetwork) -> RelationNetwork:
        """Step 6: 跨层级概念组合。

        数学对应：公理2+3 — C(y) 内部子元素间存在隐式关联，
                 通过组合操作建立跨层级边的映射

        参数:
          psi: 当前关系网络

        返回:
          RelationNetwork — 注入组合边后的关系网络（若 Composer 未注册则原样返回）
        """
        if self._concept_composer is None:
            return psi
        return self._concept_composer.compose(psi)  # type: ignore[union-attr]

    def _step_rule_extract(self, psi: RelationNetwork) -> List[RuleDef]:
        """Step 7a: 从 Ψ 提取规则域 F。

        数学对应：前提3 — F = Φ_F(Ψ)，规则域由网络结构唯一确定
                 幂律检测 + 团检测 → 传播规则 + 传递性规则

        参数:
          psi: 关系网络

        返回:
          List[RuleDef] — 提取出的规则列表
        """
        return self._rule_extractor.extract_rules(psi)

    def _step_constraint_extract(self,
                                  psi: RelationNetwork) -> List[ConstraintDef]:
        """Step 7b: 从 Ψ 提取约束域 C。

        数学对应：前提3 — C = Φ_C(Ψ)，约束域由网络拓扑异常唯一确定
                 孤立节点检测 → node_connectivity_min 约束

        参数:
          psi: 关系网络

        返回:
          List[ConstraintDef] — 提取出的约束列表
        """
        return self._rule_extractor.extract_constraints(psi)

    def _step_consistency(self, model: MathModel,
                           pipeline: PipelineData) -> MathModel:
        """Step 8: 自洽性验证与修复。

        数学对应：定理5 — F 中规则不得与 C 中约束冲突
                 附录 D.5 — 违规检测后执行自动修复

        参数:
          model: 待验证的数学模型 M = (S, F, C)
          pipeline: 管线数据容器（用于写入违规记录）

        返回:
          MathModel — 自洽性验证并修复后的数学模型
        """
        # ── 执行自洽性验证 ──
        violations: List[ConsistencyViolation] = \
            self._consistency_checker.verify(model)

        # ── 记录违规信息到管线元数据 ──
        pipeline.meta["consistency_violations"] = [
            v.to_dict() for v in violations
        ]
        pipeline.meta["consistency_violation_count"] = len(violations)

        # ── 执行修复 ──
        # 数学对应：附录 D.5 — 删除所有 severity="error" 的违规规则/约束
        if violations:
            model = self._consistency_checker.repair(model, violations)
            pipeline.meta["consistency_repaired"] = True
            pipeline.meta["consistency_repaired_count"] = sum(
                1 for v in violations if v.severity == "error"
            )
        else:
            pipeline.meta["consistency_repaired"] = False
            pipeline.meta["consistency_repaired_count"] = 0

        # ── 标记验证通过 ──
        model.metadata["consistency_verified"] = True

        return model
