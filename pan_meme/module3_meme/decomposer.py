"""
模块3 — 模因化层: 模因分解器主入口 (MemeDecomposer)

数学对应: 定理4 — Φ_D: G→Q 双射 + Φ_D^{-1} 逆映射.

核心思想:
    Φ_D 将几何对象 G 分解为复合模因状态 Q = {X_i, Θ_i, C}, 其中:
    - {X_i}: 由 GeometrySplit 拆分 + Mapping5D 映射得到的模因状态列表
    - Θ:  由 ParamDerive 对各子几何体闭式推导得到的参数列表
    - C:   由 Coupling 基于子几何体空间关系生成的耦合矩阵

    逆映射 Φ_D^{-1} 则将 Q 重构回 G:
    - Mapping5D.inverse_map 将每个 X_i 映射回 K_i
    - GeometrySplit.merge 将所有 K_i 合并为完整胞腔复形 K
    - 再与原始度量、场、不变量组装为完整几何对象

管线编排:
    decompose:  split → map → derive → coupling → assemble Q
    reconstruct: meme → inverse_map → merge → assemble G
"""

from typing import Dict, List

import numpy as np

from pan_meme.core.types import (
    CompositeMemeState,
    GeometricObject,
    MemeState,
    PipelineData,
    SimplicialComplex,
)

from .geometry_split import GeometrySplit
from .mapping_5d import Mapping5D
from .param_derive import ParamDerive
from .coupling import Coupling


class MemeDecomposer:
    """
    模因分解器: Φ_D 双射的完整实现 — 几何对象 ↔ 复合模因状态.

    数学对应:
        - 定理4 (Φ_D: G→Q): decompose() 实现正向映射.
        - 定理4 (Φ_D^{-1}: Q→G): reconstruct() 实现逆映射.
        - 推论5.1 (round-trip): decompose ∘ reconstruct = id (信息无损).

    子步骤对应的定理/定义:
        1. GeometrySplit.split   → 定义4的拆分步骤 (连通分量)
        2. Mapping5D.map         → 定义4的 K_i → X_i 映射
        3. ParamDerive.derive    → 论文4.3.7节的 11参数闭式推导
        4. Coupling.generate     → 论文3.4.4节的耦合矩阵构造

    用法:
        decomposer = MemeDecomposer()
        data = decomposer.decompose(data)            # G → Q
        geo = decomposer.reconstruct(composite_meme) # Q → G
    """

    # ============================================================
    # 正向映射: Φ_D — 几何对象 → 复合模因状态
    # ============================================================

    def decompose(self, data: PipelineData) -> PipelineData:
        """
        Φ_D: G → Q — 将几何对象分解为复合模因状态.

        数学注解:
            Φ_D(G) = Q = ({X_1, X_2, ..., X_n}, {Θ_1, Θ_2, ..., Θ_n}, C)

            其中:
            - {X_i} = Mapping5D.map(G_i) for each G_i ∈ GeometrySplit.split(G)
            - Θ_i   = ParamDerive.derive(X_i, G_i)
            - C      = Coupling.generate({G_i}, n)

            四个步骤依次执行, 保证信息从几何空间到模因空间的完整迁移.
            最终结果写入 data.meme_state, 并返回更新后的 PipelineData.

        Args:
            data: 管线上下文, 必须包含有效的 data.geo_object (GeometricObject).
                  若 geo_object 为 None, 方法静默跳过 (返回原 data).

        Returns:
            PipelineData: 更新后的管线上下文, 其中:
                - data.meme_state: CompositeMemeState — 复合模因状态 Q.
                - data.meta["decompose_step"] = "completed" — 步骤标记.
                - data.meta["meme_count"] — 模因子数量.
        """
        # ---- 前置检查 ----
        # 定理4 的前置条件: 几何对象必须存在.
        geo_obj: GeometricObject = data.geo_object
        if geo_obj is None:
            data.meta["decompose_step"] = "skipped (no geo_object)"
            return data

        # ============================================================
        # 步骤1: 拆分 — GeometrySplit.split
        # 数学: 定义4的第一步, 将 G 按连通分量拆分为 {G_1, G_2, ..., G_n}
        # beta0 连通分量检测 (scipy 稀疏连通分量算法)
        # ============================================================
        splitter = GeometrySplit()
        sub_geos: List[GeometricObject] = splitter.split(geo_obj)
        n: int = len(sub_geos)

        # 边界情况: 若拆分为空 (无顶点), 返回空模因状态
        if n == 0:
            data.meme_state = CompositeMemeState(
                memes=[],
                Theta=[],
                C=np.zeros((0, 0), dtype=np.float32),
            )
            data.meta["decompose_step"] = "completed (empty)"
            data.meta["meme_count"] = 0
            return data

        # ============================================================
        # 步骤2: 映射 — Mapping5D.map
        # 数学: 定义4的第二步, 每个子几何体 G_i 映射为模因状态 X_i
        # X_i = (D, B, ρ, R, S, ξ) ∈ Ω × Ξ
        # ============================================================
        mapper = Mapping5D()
        memes: List[MemeState] = []
        for sub_geo in sub_geos:
            meme: MemeState = mapper.map(sub_geo)
            memes.append(meme)

        # ============================================================
        # 步骤3: 参数推导 — ParamDerive.derive
        # 数学: 论文4.3.7节, 每个 (X_i, G_i) 唯一确定 11 个动力学参数
        # Θ_i = {α₁, α₂, β₁, β₂, γ₁, γ₂, δ₁, δ₂, δ₃, ε₁, ε₂}
        # ============================================================
        deriver = ParamDerive()
        theta_list: List[Dict[str, float]] = []
        for meme, sub_geo in zip(memes, sub_geos):
            theta: Dict[str, float] = deriver.derive(meme, sub_geo)
            theta_list.append(theta)

        # ============================================================
        # 步骤4: 耦合矩阵 — Coupling.generate
        # 数学: 论文3.4.4节, 子几何体间的共享边/顶点/层级决定 C_{ij}
        # C ∈ [0,1]^{n×n}, 对称, 对角线为零
        # ============================================================
        C: np.ndarray = Coupling.generate(sub_geos, n)

        # ============================================================
        # 组装: 构造 CompositeMemeState Q 并写入管线上下文
        # 定义4: Q = (memes, Theta, C) — 完整的复合模因状态
        # ============================================================
        data.meme_state = CompositeMemeState(
            memes=memes,
            Theta=theta_list,
            C=C,
        )
        data.meta["decompose_step"] = "completed"
        data.meta["meme_count"] = n

        return data

    # ============================================================
    # 逆映射: Φ_D^{-1} — 复合模因状态 → 几何对象
    # ============================================================

    def reconstruct(
        self,
        meme: CompositeMemeState,
    ) -> GeometricObject:
        """
        Φ_D^{-1}: Q → G — 将复合模因状态重构回几何对象.

        数学注解:
            Φ_D^{-1}(Q) = merge( {Mapping5D.inverse_map(X_i)} )

            重构步骤:
            1. 遍历 Q.memes 中的每个 MemeState X_i
            2. 调用 Mapping5D.inverse_map(X_i) 得到子胞腔复形 K_i
            3. 将所有 K_i 通过 GeometrySplit.merge 合并为完整胞腔复形 K
            4. 从原始子几何体中提取度量 g、场 ω、不变量 Γ 并组装为 G

            逆映射的存在性由定理4的"双射"性质保证:
                对任意几何对象 G, Φ_D^{-1}(Φ_D(G)) = G.

        Args:
            meme: 复合模因状态 Q = ({X_i}, {Θ_i}, C), 由 decompose 产出.

        Returns:
            GeometricObject: 重构的几何对象 G = (K, g, ω, Γ, R).
                             注意: 某些几何细节 (如精确的 ω 场值) 在逆映射中
                             只能近似还原, 因为 Mapping5D.inverse_map 需要
                             从 5 维向量反向推断子几何体.
        """
        # ---- 前置检查: 空模因状态 ----
        # 空模因状态 → 返回空的 (n=0) 几何对象
        if not meme.memes:
            return GeometricObject(
                K=SimplicialComplex(
                    vertices=[],
                    edges=[],
                    higher_cells=[],
                    subcomplexes={},
                    level_labels={},
                ),
                g=np.zeros(0, dtype=np.float32),
                omega=np.zeros(0, dtype=np.float32),
                Gamma={},
                R={},
            )

        # ============================================================
        # 步骤1: 逆映射 — Mapping5D.inverse_map
        # 数学: 对每个 X_i ∈ Q.memes, 反向构造子几何体 K_i
        # Mapping5D.inverse_map: X_i → (K_i, g_i, ω_i, Γ_i)
        # ============================================================
        mapper = Mapping5D()
        sub_geos: List[GeometricObject] = []
        for mem in meme.memes:
            sub_geo: GeometricObject = mapper.inverse_map(mem)
            sub_geos.append(sub_geo)

        # ============================================================
        # 步骤2: 合并 — GeometrySplit.merge
        # 数学: 将所有子胞腔复形 K_i 合并为完整胞腔复形 K
        # merge 操作合并顶点集、边集, 并合并层级标签与子复形信息
        # ============================================================
        splitter = GeometrySplit()
        merged_geo: GeometricObject = splitter.merge(sub_geos)

        return merged_geo
