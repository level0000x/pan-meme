"""
模块2 — 几何化层: 场编码器 (FieldEncoder)

数学对应: 定义3(c) — ω: 微分结构 / 向量场, 编码规则域 F.
ω 在 K 的每个顶点上赋予一个标量值, 形成胞腔复形上的标量场.
"""

from typing import Dict, List

import numpy as np

from pan_meme.core.types import RuleDef, SimplicialComplex


class FieldEncoder:
    """
    场编码器: 规则域 F → 向量场 ω 的确定性编码 + 逆映射.

    数学对应:
    - 定义3(c): ω = (ω_1, ω_2, ..., ω_{|V|}), 在顶点上定义的标量场.
    - 编码公式: ω[v] = Σ_{rule ∈ F: v ∈ support(rule)} confidence(rule), clip [0,1].
    - 逆映射: 每个非零场值顶点解码为一条规则, 保证信息保真.

    用法:
        omega = FieldEncoder.encode(rules, K)
        rules = FieldEncoder.decode(omega, K)
    """

    @staticmethod
    def encode(
        rules: List[RuleDef],
        K: SimplicialComplex,
    ) -> np.ndarray:
        """
        将规则域 F 编码为顶点标量场 ω.

        数学注解:
        - ω[v] = clip( Σ_{f∈F: v ∈ supp(f)} conf(f), 0, 1 )
        - 每条规则的置信度累加到其支撑集中的所有顶点.
        - np.clip 确保场值落在 [0,1] 区间内, 保证数值稳定性.

        Args:
            rules: 规则定义列表 F = {RuleDef}, 由模块1的 Φ_B 产出.
            K: 胞腔复形, 提供顶点数以确定 ω 的维度.

        Returns:
            np.ndarray: 标量场 ω, shape=(|V|,), dtype=float32.
                        每个分量 ω[i] 表示顶点 i 上的场强.
        """
        # ---- 步骤1: 初始化零场 ----
        # 定义3(c): ω ∈ R^{|V|}, 初始化为全零向量.
        n_vertices: int = len(K.vertices)
        omega: np.ndarray = np.zeros(n_vertices, dtype=np.float32)

        # ---- 步骤2: 累加规则置信度 ----
        # 编码公式: ω[v] += confidence(rule) for each rule whose support contains v.
        for rule in rules:
            conf: float = rule.confidence
            for v in rule.support:
                # 边界检查: 支撑集索引必须在顶点范围内
                if 0 <= v < n_vertices:
                    omega[v] += conf

        # ---- 步骤3: 截断至 [0,1] ----
        # 数学约束: 场值 ω[i] ∈ [0,1], 对应置信度的语义范围.
        omega = np.clip(omega, 0.0, 1.0)

        return omega

    @staticmethod
    def decode(
        omega: np.ndarray,
        K: SimplicialComplex,
    ) -> List[RuleDef]:
        """
        逆映射: 将标量场 ω 解码回规则域 F.

        数学注解:
        - 逆映射法则: 对于每个 ω[i] > 0 的顶点, 生成一条 field_intensity 规则.
        - 解码来源标记为 "decoded", 与原始 "extracted" / "inferred" 区分.
        - 这是定义3 Φ_C 双射性质在规则维度上的体现.

        Args:
            omega: 标量场 ω, shape=(|V|,), dtype=float32.
            K: 胞腔复形, 提供顶点集以便构造规则支撑.

        Returns:
            List[RuleDef]: 解码出的规则列表, 每个非零场值顶点对应一条规则.
        """
        rules: List[RuleDef] = []

        for i, val in enumerate(omega):
            val_float: float = float(val)
            if val_float > 0.0:
                rules.append(
                    RuleDef(
                        pattern=f"field_intensity_{i}",
                        support=[i],
                        confidence=val_float,
                        source="decoded",
                    )
                )

        return rules
