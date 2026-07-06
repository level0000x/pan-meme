"""
模块2 — 几何化层: 不变量编码器 (InvariantEncoder)

数学对应: 定义3(d) — Γ: 几何不变量字典, 编码约束域 C.
Γ 从胞腔复形 K 和约束列表 C 中提取拓扑和代数不变量,
这些不变量在几何变换下保持不变.

编码内容:
- euler_char: 欧拉示性数 χ(K) = |V| - |E| + |T| - ... (此处取 χ = |V| - |E|)
- num_constraints: 约束数量 |C|
- constraint_types: 唯一约束类型列表
"""

from typing import Any, Dict, List

from pan_meme.core.types import ConstraintDef, SimplicialComplex


class InvariantEncoder:
    """
    不变量编码器: 约束域 C → 不变量字典 Γ 的确定性编码 + 逆映射.

    数学对应:
    - 定义3(d): Γ = {euler_char, num_constraints, constraint_types, ...}
    - 欧拉示性数 χ(K) = |V| - |E|: 胞腔复形最基本的拓扑不变量.
    - 约束类型集合: 从 C 中提取的条件名称, 构成语义层面的不变量.

    用法:
        Gamma = InvariantEncoder.encode(constraints, K)
        constraints = InvariantEncoder.decode(Gamma, K)
    """

    @staticmethod
    def encode(
        constraints: List[ConstraintDef],
        K: SimplicialComplex,
    ) -> Dict[str, Any]:
        """
        将约束域 C 编码为不变量字典 Γ.

        数学注解:
        - euler_char = |V| - |E|: 对于 1-维胞腔复形, 这是欧拉示性数的简化形式.
        - 完整形式的欧拉示性数应为 χ = Σ_k (-1)^k * f_k, 其中 f_k 是 k-单形数量.
          此处在未引入高阶胞腔时简化为 |V| - |E|.
        - constraint_types 按字典序排列, 确保确定性和可复现性.

        Args:
            constraints: 约束定义列表 C = {ConstraintDef}, 由模块1产出.
            K: 胞腔复形, 提供 |V| 和 |E| 用于计算拓扑不变量.

        Returns:
            Dict[str, Any]: 不变量字典 Γ, 包含:
                - "euler_char" (int): 欧拉示性数 |V| - |E|.
                - "num_constraints" (int): 约束总数 |C|.
                - "constraint_types" (List[str]): 唯一约束类型列表 (排序).
        """
        return {
            # 欧拉示性数 χ = |V| - |E| (对 1-维胞腔复形的简化形式)
            # 数学: 定理3 证明中 K 的拓扑不变量
            "euler_char": len(K.vertices) - len(K.edges),
            # 约束域基数 |C|
            "num_constraints": len(constraints),
            # 约束类型集合: 提取所有 condition 字段去重排序
            "constraint_types": sorted(set(c.condition for c in constraints)),
        }

    @staticmethod
    def decode(
        Gamma: Dict[str, Any],
        K: SimplicialComplex,
    ) -> List[ConstraintDef]:
        """
        逆映射: 将不变量字典 Γ 解码回约束域 C.

        数学注解:
        - 每个约束类型产生一条 ConstraintDef, 其作用域设为全体顶点.
        - domain = list(range(|V|)) 是一个保守的默认值: 在无更多信息时,
          假设约束作用于整个胞腔复形.
        - 解码来源标记为 "decoded", 与原始 "extracted" 区分.

        Args:
            Gamma: 不变量字典, 至少包含 "constraint_types" 字段.
            K: 胞腔复形, 提供 |V| 以构造约束域.

        Returns:
            List[ConstraintDef]: 解码出的约束列表.
        """
        return [
            ConstraintDef(
                condition=ct,
                domain=list(range(len(K.vertices))),
                source="decoded",
            )
            for ct in Gamma.get("constraint_types", [])
        ]
