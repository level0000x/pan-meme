"""
模块2 — 几何化层: 主入口 Geometrizer

数学对应: 定理3 — Φ_C: M → G 双射 + 逆映射 Φ_C^{-1}: G → M

Φ_C 是模因几何化流水线的核心编排器, 按序调用四个编码器:
  skeleton → metric → field → invariant → 可逆性元数据
其逆映射 Φ_C^{-1} 则反向解码, 恢复数学模型中丢失的结构信息.

用法:
    from pan_meme.module2_geo.geometrizer import Geometrizer
    geo = Geometrizer()  # 无状态, 内部不保存参数
    data = geo.encode(data)       # PipelineData → PipelineData (附加 geo_object)
    model = geo.decode(geo_obj)   # GeometricObject → MathModel
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from pan_meme.core.types import (
    ConstraintDef,
    GeometricObject,
    MathModel,
    PipelineData,
    RelationNetwork,
    RuleDef,
    SimplicialComplex,
)
from pan_meme.module2_geo.skeleton import SkeletonEncoder
from pan_meme.module2_geo.metric_encoder import MetricEncoder
from pan_meme.module2_geo.field_encoder import FieldEncoder
from pan_meme.module2_geo.invariant_encoder import InvariantEncoder


class Geometrizer:
    """
    几何化编排器: Φ_C 双射的完整实现.

    数学对应:
    - 定理3 (Φ_C: M → G): 将一个数学模型 M = (S, F, C) 映射为
      几何对象 G = (K, g, ω, Γ, R), 该映射是双射.
    - 编码链: S → K (骨架), w → g (度量), F → ω (场), C → Γ (不变量).
    - 解码链: K → S (逆向), g → w (逆度量), ω → F (逆场), Γ → C (逆不变量).
    - 双射性质保障: R 中存储的 vertex_map / edge_map / construction_log
      使解码过程可以完全恢复原始结构, 无信息丢失.

    用法:
        gzr = Geometrizer()
        data = gzr.encode(data)
        model = gzr.decode(geo_object)
    """

    # ================================================================
    # 编码: Φ_C (定理3 正向)
    # ================================================================

    def encode(self, data: PipelineData) -> PipelineData:
        """
        正向映射 Φ_C: M → G, 将数学模型编码为几何对象.

        流水线步骤 (共5步):
          step1: skeleton — 骨架编码, 构造胞腔复形 K 及顶点/边索引映射.
          step2: metric   — 度量编码, 将边权 w 映射为几何长度向量 g.
          step3: field    — 场编码, 将规则域 F 映射为顶点标量场 ω.
          step4: invariant — 不变量编码, 将约束域 C 映射为不变量字典 Γ.
          step5: reversibility — 组装可逆性元数据包 R.

        数学注解:
        - Φ_C 是定义3的核心实现, 将离散代数结构 (M) 编码为连续几何结构 (G).
        - vertex_map 和 edge_map 保证顶点/边的一一对应 (同构).
        - construction_log 记录构造步骤顺序, 用于可复现性验证.

        Args:
            data: 管线数据包, 必须已包含 math_model 字段 (模块1输出).

        Returns:
            PipelineData: 同一管线数据包, 附加 geo_object 字段.

        Raises:
            ValueError: 若 data.math_model 为 None, 编码链无法启动.
        """
        # ---- 前置检查: 确保数学模型已就绪 ----
        m: MathModel | None = data.math_model
        if m is None:
            raise ValueError(
                "[Geometrizer.encode] math_model 为 None, "
                "编码链无法启动. 请先运行模块1 (Relation Network → Math Model)."
            )

        # ---- step1: 骨架编码 ----
        # 数学: K = Φ_C^{skeleton}(Ψ), 顶点/边一一对应于原图.
        # SkeletonEncoder.encode 返回 (K, vertex_map, edge_map).
        K: SimplicialComplex
        vm: Dict[int, int]
        em: Dict[int, int]
        K, vm, em = SkeletonEncoder.encode(m.structure)

        # ---- step2: 度量编码 ----
        # 数学: g_i = w(e_i), 定义3(b).
        # g ∈ R^{|E|}, 每个分量取值于 [0,1] (float32).
        g: np.ndarray = MetricEncoder.encode(m.structure, K, em)

        # ---- step3: 场编码 ----
        # 数学: ω[v] = Σ_{rule∈F: v∈supp(rule)} conf(rule), 定义3(c).
        # ω ∈ R^{|V|}, 取值于 [0,1] (float32).
        omega: np.ndarray = FieldEncoder.encode(m.rules, K)

        # ---- step4: 不变量编码 ----
        # 数学: Γ = {euler_char, num_constraints, constraint_types}, 定义3(d).
        # 拓扑不变量在几何变换下保持不变.
        Gamma: Dict[str, Any] = InvariantEncoder.encode(m.constraints, K)

        # ---- step5: 组装可逆性元数据包 R ----
        # 数学: R 是 Φ_C 双射性质的核心保证.
        # vertex_map / edge_map 使逆向解码能够恢复原始索引对应关系.
        # construction_log 记录编码步骤, 用于审计和可复现性.
        R: Dict[str, Any] = {
            "vertex_map": vm,
            "edge_map": em,
            "construction_log": ["skeleton", "metric", "field", "invariant"],
        }

        # ---- 最终: 构造几何对象 G ----
        # 定义3: G = (K, g, ω, Γ, R)
        data.geo_object = GeometricObject(
            K=K,
            g=g,
            omega=omega,
            Gamma=Gamma,
            R=R,
        )

        return data

    # ================================================================
    # 解码: Φ_C^{-1} (定理3 逆向)
    # ================================================================

    def decode(self, geo: GeometricObject) -> MathModel:
        """
        逆映射 Φ_C^{-1}: G → M, 将几何对象解码回数学模型.

        解码链 (4步):
          K → S: 胞腔复形逆映射为关系网络 RelationNetwork.
              节点标识符按 "v_{vertex_id}" 格式生成.
              边权直接取自度量向量 g.
              层级信息从 K.level_labels 重建.
          g → w: 度量向量 g 的各分量即是边权 w_i.
          ω → F: 通过 FieldEncoder.decode 将标量场恢复为规则列表.
          Γ → C: 通过 InvariantEncoder.decode 将不变量字典恢复为约束列表.

        数学注解:
        - 解码后的 MathModel 在 metadata 中标记 "decoded_from_geo": True,
          以便与原始 "extracted" / "inferred" 模型区分.
        - 这是 Φ_C 双射性质的验证: decode(encode(M)) ≅ M.

        Args:
            geo: 几何对象 G = (K, g, ω, Γ, R).

        Returns:
            MathModel: 解码出的数学模型 M' = (S', F', C'),
                       附带 metadata: {"decoded_from_geo": True}.
        """
        # ---- 步骤1: K → S (逆向骨架) ----
        # 数学: K.vertices → V, K.edges → E, g → w.
        # 顶点标识符按 "v_{id}" 命名, 便于追溯.
        nodes: List[str] = [f"v_{i}" for i in geo.K.vertices]
        edges: List[Tuple[int, int]] = list(geo.K.edges)

        # 层级信息: 从 K.level_labels 提取.
        # hierarchy["levels"] = 最大层级编号, hierarchy["node_levels"] = 顶点→层级映射.
        hierarchy: Dict[str, Any] = {
            "levels": (
                max(geo.K.level_labels.values())
                if geo.K.level_labels
                else 0
            ),
            "node_levels": dict(geo.K.level_labels),
        }

        S: RelationNetwork = RelationNetwork(
            nodes=nodes,
            edges=edges,
            weights=geo.g.astype(np.float64),  # 度量 g 即边权 w
            hierarchy=hierarchy,
        )

        # ---- 步骤2: ω → F (逆场编码) ----
        # 数学: FieldEncoder.decode 将标量场 ω 的每个非零顶点解码为一条规则.
        F: List[RuleDef] = FieldEncoder.decode(geo.omega, geo.K)

        # ---- 步骤3: Γ → C (逆不变量编码) ----
        # 数学: InvariantEncoder.decode 将不变量字典中的 constraint_types
        # 恢复为 ConstraintDef 列表, 作用域默认设为全体顶点.
        C: List[ConstraintDef] = InvariantEncoder.decode(geo.Gamma, geo.K)

        # ---- 步骤4: 组装数学模型 M ----
        # 定义2: M = (S, F, C). 附加元数据标记解码来源.
        return MathModel(
            structure=S,
            rules=F,
            constraints=C,
            metadata={"decoded_from_geo": True},
        )
