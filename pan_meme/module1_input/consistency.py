# 泛模因几何工具 — 模块一：自洽性验证与修复
# 数学对应：M = (S, F, C) 自洽性验证 — 定理5（自洽性约束）
# 论文位置：定理5（自洽性约束）, 附录D.5（违规检测与修复）

from typing import List, Set, Optional
import numpy as np

from pan_meme.core.types import MathModel, RuleDef, ConstraintDef
from pan_meme.core.exceptions import ConsistencyViolation


class ConsistencyChecker:
    """定理5实现：验证数学模型 M = (S, F, C) 的内部自洽性。

    数学对应：
    - 定理5：自洽性约束 — F 中规则不得与 C 中约束冲突，
      且规则/约束中引用的所有索引必须在合法范围内
    - 公理4：Ω[Ψ] = det(I - λA_Ψ) ≠ 0 等价于结构无矛盾
    - 附录 D.5：违规检测是 ODE 求解前的必要检查

    验证维度：
    1. 索引边界检查 (BoundCheck)：每条规则的 support 中每个元素必须满足 0 ≤ idx < |V|
    2. 约束域边界检查：每个约束的 domain 中每个元素必须满足 0 ≤ idx < |V|
    3. 规则-约束冲突检查：规则支撑集与约束定义域的交集非空时可能存在冲突
    4. 空支撑集检查：规则或约束的支撑集/定义域为空视为退化

    修复策略 (repair)：
    - severity="error" 的违规 → 删除对应规则/约束
    - severity="warning" 的违规 → 保留原样，仅记录

    用法:
        checker = ConsistencyChecker()
        violations = checker.verify(model)
        if violations:
            model = checker.repair(model, violations)
    """

    def __init__(self) -> None:
        """初始化自洽性验证器（当前版本无状态配置）。"""
        pass

    # ================================================================
    # 公有方法：verify — 执行自洽性验证
    # 数学对应：定理5 — 验证 F 与 C 的内部一致性
    # ================================================================

    def verify(self, model: MathModel) -> List[ConsistencyViolation]:
        """对数学模型 M = (S, F, C) 执行完整的自洽性验证。

        数学对应：
        - 定理5：自洽性 = (∀ (f,supp)∈F, ∀ idx∈supp: 0≤idx<|V|)
                   ∧ (∀ (c,dom)∈C, ∀ idx∈dom: 0≤idx<|V|)
                   ∧ (F中无规则与C中约束矛盾)
        - 附录 D.5：违规检测生成结构化违规列表

        验证步骤：
        1. 获取顶点数 n = |V| = len(model.structure.nodes)
        2. 规则索引边界检查：逐条验证 support 中所有索引
        3. 约束索引边界检查：逐条验证 domain 中所有索引
        4. 规则-约束冲突检查：支撑集与定义域交集检测
        5. 空域退化检查：空 support 或空 domain 视为警告

        参数:
          model: 数学模型 M = (S, F, C)，S 为 RelationNetwork

        返回:
          List[ConsistencyViolation] — 所有检测到的自洽性违规
        """
        violations: List[ConsistencyViolation] = []

        # ── 前置：获取顶点集大小 ──
        # 数学对应：V 是 RelationNetwork.nodes，|V| = len(nodes)
        n: int = len(model.structure.nodes)

        # ── 步骤1：规则域 F 的索引边界检查 ──
        # 数学对应：定理5 — ∀ (f_i, supp_i) ∈ F, ∀ idx ∈ supp_i:
        #   0 ≤ idx < |V|，否则 supp 引用了不存在的节点
        rule_violations: List[ConsistencyViolation] = self._check_rule_bounds(
            model.rules, n
        )
        violations.extend(rule_violations)

        # ── 步骤2：约束域 C 的索引边界检查 ──
        # 数学对应：定理5 — ∀ (c_j, dom_j) ∈ C, ∀ idx ∈ dom_j:
        #   0 ≤ idx < |V|，否则 dom 引用了不存在的节点
        constraint_violations: List[ConsistencyViolation] = self._check_constraint_bounds(
            model.constraints, n
        )
        violations.extend(constraint_violations)

        # ── 步骤3：规则-约束冲突检查 ──
        # 数学对应：定理5 — F 中规则的执行不得违反 C 中约束
        # 若某规则的支撑集与某约束的定义域有交集，标记为潜在冲突
        conflict_violations: List[ConsistencyViolation] = self._check_rule_constraint_conflicts(
            model.rules, model.constraints
        )
        violations.extend(conflict_violations)

        # ── 步骤4：空域退化检查 ──
        # 数学对应：定义2 — supp ≠ ∅, dom ≠ ∅ 是基本要求
        # 空支撑集或空定义域意味着规则或约束退化，标记 warning
        empty_violations: List[ConsistencyViolation] = self._check_empty_domains(
            model.rules, model.constraints
        )
        violations.extend(empty_violations)

        return violations

    # ================================================================
    # 公有方法：repair — 修复违规的数学模型
    # 数学对应：附录 D.5 — 自动修复机制：删除 error 级违规条目
    # ================================================================

    def repair(self, model: MathModel,
               violations: List[ConsistencyViolation]) -> MathModel:
        """根据违规列表修复数学模型 M。

        数学对应：
        - 附录 D.5：修复操作 = 删除所有 severity="error" 的违规规则和约束
        - 修复后的 M' = (S, F', C')，其中 F' ⊆ F，C' ⊆ C
        - 删除保守：仅移除 error 级违规；warning 级保留供人工审查

        修复策略：
        1. 收集所有 error 级违规中涉及的规则索引
        2. 收集所有 error 级违规中涉及的约束索引
           （约束索引以负数偏移编码：-1 表示 constraints[0]，-2 表示 constraints[1]）
        3. 根据索引集合过滤 rules 和 constraints
        4. 更新 metadata 记录修复信息

        参数:
          model: 待修复的数学模型
          violations: 违规检测结果列表

        返回:
          MathModel — 修复后的数学模型（原对象被覆写后返回）
        """
        # ── 收集 error 级违规涉及的规则和约束索引 ──
        # 数学对应：附录 D.5 — 以 err_idx 标记需要删除的条目
        error_rule_indices: Set[int] = set()
        error_constraint_indices: Set[int] = set()

        for violation in violations:
            if violation.severity != "error":
                continue  # warning 级不执行删除

            rule_idx: int = violation.rule_idx
            if rule_idx >= 0:
                # 规则索引（rule_idx 直接对应 model.rules 中的位置）
                error_rule_indices.add(rule_idx)
            else:
                # 约束索引（负数编码：-1 对应 constraints[0]，-2 对应 constraints[1]）
                constraint_idx: int = -rule_idx - 1
                error_constraint_indices.add(constraint_idx)

        # ── 过滤规则：保留未被标记为 error 的规则 ──
        # 数学对应：F' = F \ { (f_i, supp_i) | i ∈ error_rule_indices }
        cleaned_rules: List[RuleDef] = [
            rule for i, rule in enumerate(model.rules)
            if i not in error_rule_indices
        ]

        # ── 过滤约束：保留未被标记为 error 的约束 ──
        # 数学对应：C' = C \ { (c_j, dom_j) | j ∈ error_constraint_indices }
        cleaned_constraints: List[ConstraintDef] = [
            constraint for j, constraint in enumerate(model.constraints)
            if j not in error_constraint_indices
        ]

        # ── 更新 metadata：记录修复操作 ──
        # 数学对应：附录 D.5 — 修复日志记录在 metadata.repair_log 中
        removed_rule_count: int = len(model.rules) - len(cleaned_rules)
        removed_constraint_count: int = len(model.constraints) - len(cleaned_constraints)

        model.rules = cleaned_rules
        model.constraints = cleaned_constraints

        # 在 metadata 中追加修复记录
        if "repair_log" not in model.metadata:
            model.metadata["repair_log"] = []
        model.metadata["repair_log"].append({
            "total_violations": len(violations),
            "error_count": len([v for v in violations if v.severity == "error"]),
            "warning_count": len([v for v in violations if v.severity == "warning"]),
            "removed_rules": removed_rule_count,
            "removed_constraints": removed_constraint_count,
            "remaining_rules": len(cleaned_rules),
            "remaining_constraints": len(cleaned_constraints),
        })
        model.metadata["rule_count"] = len(cleaned_rules)
        model.metadata["constraint_count"] = len(cleaned_constraints)

        return model

    # ================================================================
    # 内部方法：规则索引边界检查
    # 数学对应：定理5 — ∀ idx ∈ supp: 0 ≤ idx < |V|
    # ================================================================

    def _check_rule_bounds(self, rules: List[RuleDef],
                           n: int) -> List[ConsistencyViolation]:
        """验证所有规则的支撑集索引均在合法范围内。

        数学对应：
        - 定理5：supp_i ⊆ {0, 1, ..., |V|-1}
        - 若 idx ∉ [0, |V|)，则该规则引用不存在的节点，
          severity="error"，需从 F 中移除

        参数:
          rules: 规则列表 F
          n: 顶点总数 |V|

        返回:
          List[ConsistencyViolation] — 索引越界违规列表
        """
        violations: List[ConsistencyViolation] = []

        for i, rule in enumerate(rules):
            # 检查支撑集中每个索引
            for idx in rule.support:
                if idx < 0 or idx >= n:
                    violations.append(ConsistencyViolation(
                        rule_idx=i,
                        description=(
                            f"规则[{i}] '{rule.pattern}' 的支撑集包含越界索引 "
                            f"{idx}（合法范围: [0, {n - 1}]）。"
                            f"该索引引用不存在的节点，规则必须被移除。"
                        ),
                        severity="error",
                    ))
                    break  # 一条规则仅报告一次（首个越界索引）

        return violations

    # ================================================================
    # 内部方法：约束索引边界检查
    # 数学对应：定理5 — ∀ idx ∈ dom: 0 ≤ idx < |V|
    # ================================================================

    def _check_constraint_bounds(self, constraints: List[ConstraintDef],
                                  n: int) -> List[ConsistencyViolation]:
        """验证所有约束的定义域索引均在合法范围内。

        数学对应：
        - 定理5：dom_j ⊆ {0, 1, ..., |V|-1}
        - 若 idx ∉ [0, |V|)，则该约束引用不存在的节点，
          severity="error"，需从 C 中移除
        - 约束违规的 rule_idx 使用负数编码：
          第 j 个约束对应 rule_idx = -(j + 1)

        参数:
          constraints: 约束列表 C
          n: 顶点总数 |V|

        返回:
          List[ConsistencyViolation] — 索引越界违规列表
        """
        violations: List[ConsistencyViolation] = []

        for j, constraint in enumerate(constraints):
            for idx in constraint.domain:
                if idx < 0 or idx >= n:
                    violations.append(ConsistencyViolation(
                        # 约束违规：rule_idx 使用负数编码 -(j+1)
                        rule_idx=-(j + 1),
                        description=(
                            f"约束[{j}] '{constraint.condition}' 的定义域包含越界索引 "
                            f"{idx}（合法范围: [0, {n - 1}]）。"
                            f"该索引引用不存在的节点，约束必须被移除。"
                        ),
                        severity="error",
                    ))
                    break  # 一条约束仅报告一次

        return violations

    # ================================================================
    # 内部方法：规则-约束冲突检查
    # 数学对应：定理5 — F 中规则不得与 C 中约束矛盾
    # ================================================================

    def _check_rule_constraint_conflicts(
        self, rules: List[RuleDef], constraints: List[ConstraintDef]
    ) -> List[ConsistencyViolation]:
        """检查规则支撑集与约束定义域之间是否存在潜在冲突。

        数学对应：
        - 定理5：若 supp_i ∩ dom_j ≠ ∅，规则 i 的执行可能违反约束 j
        - 此时该约束限制了对规则支撑集中节点的操作，
          构成潜在自洽性矛盾

        冲突判定：规则支撑集与约束定义域有交集时标记 warning。
        注意：此处仅标记 warning（非 error），因为交集不一定
        意味着实际违反——具体语义需要人工或更深层逻辑判定。

        参数:
          rules: 规则列表 F
          constraints: 约束列表 C

        返回:
          List[ConsistencyViolation] — 冲突违规列表（均为 warning 级）
        """
        violations: List[ConsistencyViolation] = []

        for i, rule in enumerate(rules):
            rule_support_set: Set[int] = set(rule.support)

            for j, constraint in enumerate(constraints):
                constraint_domain_set: Set[int] = set(constraint.domain)

                # 计算交集
                intersection: Set[int] = rule_support_set & constraint_domain_set

                if intersection:
                    violations.append(ConsistencyViolation(
                        rule_idx=i,
                        description=(
                            f"规则[{i}] '{rule.pattern}' 的支撑集与 "
                            f"约束[{j}] '{constraint.condition}' 的定义域存在交集: "
                            f"{sorted(intersection)}。"
                            f"规则执行可能违反该约束，需人工审查。"
                        ),
                        severity="warning",
                    ))

        return violations

    # ================================================================
    # 内部方法：空域退化检查
    # 数学对应：定义2 — supp ≠ ∅, dom ≠ ∅ 是基元完整性要求
    # ================================================================

    def _check_empty_domains(
        self, rules: List[RuleDef], constraints: List[ConstraintDef]
    ) -> List[ConsistencyViolation]:
        """检查是否存在空支撑集规则或空定义域约束。

        数学对应：
        - 定义2：(f, supp) ∈ F 满足 supp ≠ ∅
        - 定义2：(c, dom) ∈ C 满足 dom ≠ ∅
        - 空域表示规则/约束退化，无法在工作域上发挥作用

        参数:
          rules: 规则列表 F
          constraints: 约束列表 C

        返回:
          List[ConsistencyViolation] — 空域退化违规列表（均为 warning 级）
        """
        violations: List[ConsistencyViolation] = []

        # ── 空支撑集规则检测 ──
        for i, rule in enumerate(rules):
            if not rule.support:
                violations.append(ConsistencyViolation(
                    rule_idx=i,
                    description=(
                        f"规则[{i}] '{rule.pattern}' 的支撑集为空。"
                        f"空支撑集规则无法在任何节点上触发，视为退化规则。"
                    ),
                    severity="warning",
                ))

        # ── 空定义域约束检测 ──
        for j, constraint in enumerate(constraints):
            if not constraint.domain:
                violations.append(ConsistencyViolation(
                    rule_idx=-(j + 1),
                    description=(
                        f"约束[{j}] '{constraint.condition}' 的定义域为空。"
                        f"空定义域约束无法限制任何节点，视为退化约束。"
                    ),
                    severity="warning",
                ))

        return violations
