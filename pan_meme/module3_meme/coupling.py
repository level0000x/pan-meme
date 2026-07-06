"""
模块3 — 模因化层: 耦合矩阵 (Coupling)

数学对应: 论文3.4.4节 — 子几何体间的空间关系自动生成耦合强度矩阵.

核心思想:
    模因系统是一个耦合动力学系统: 每个子模因通过边缘共享、顶点共享
    和层级归属等结构关系相互作用. 耦合矩阵 C ∈ [0,1]^{n×n} 量化了
    这些相互作用的强度, 其中 C_{ij} 表示子几何体 i 对 j 的耦合强度.

耦合原则:
    - 共享边: 两个子几何体共享边意味着它们有直接的几何接触 → 中等耦合 (+0.3)
    - 共享顶点: 显著顶点重叠暗示结构同源性 → 强耦合 (+0.5, 阈值 ≥5)
    - 同层级: 属于同一 component/层级 意味着功能相似性 → 弱耦合 (+0.2)
    - 截断与对称化: C_{ij} ∈ [0,1], C_{ii}=0, C = C^T
"""

from typing import List, Set

import numpy as np

from pan_meme.core.types import GeometricObject


class Coupling:
    """
    耦合矩阵生成器: 由子几何体间的空间关系生成耦合矩阵.

    数学对应:
        - 论文3.4.4节: C_{ij} 的定义基于几何空间中的结构重叠.
        - 定理4 (Φ_D: G→Q): C 是 Q 的关键组件, 与模因列表和 Θ 同等重要.
        - 定理6 (分段解): C 作为ODE系统耦合项的系数矩阵.

    用法:
        C: np.ndarray = Coupling.generate(sub_geos, n)
    """

    # ============================================================
    # 耦合权重常量 (论文3.4.4节)
    # ============================================================

    # 共享边耦合权重: 直接几何接触 → 中等强度
    _WEIGHT_SHARED_EDGE: float = 0.3

    # 共享顶点耦合权重: 显著顶点重叠 → 强耦合
    _WEIGHT_SHARED_VERTEX: float = 0.5

    # 共享顶点阈值: 需 ≥5 个共享顶点才触发
    _SHARED_VERTEX_THRESHOLD: int = 5

    # 同层级耦合权重: 相同 component_id → 弱耦合
    _WEIGHT_SAME_COMPONENT: float = 0.2

    # 耦合强度上界
    _MAX_COUPLING: float = 1.0

    @staticmethod
    def generate(
        sub_geos: List[GeometricObject],
        n: int,
    ) -> np.ndarray:
        """
        生成 n×n 耦合矩阵 C, 量化子几何体间的结构相互作用强度.

        数学注解:
            C_{ij} = clip(
                0.0
                + 0.3 · 1_{edges_i ∩ edges_j ≠ ∅}     — 共享边
                + 0.5 · 1_{|vertices_i ∩ vertices_j| ≥ 5} — 共享顶点
                + 0.2 · 1_{component_id_i == component_id_j} — 同层级
                , 0, 1)

            - 1_{条件} 是指示函数: 条件成立时为 1, 否则为 0.
            - C_{ii} = 0: 自身耦合强制为零 (无自循环).
            - C 为对称矩阵: C_{ij} = C_{ji} (双向耦合).
            - 使用 float32 精度, 保证后续 ODE 求解的数值稳定性.

        Args:
            sub_geos: 子几何体列表 G_1, G_2, ..., G_n, 每个都是由
                      GeometrySplit.split() 产生的连通分量几何体.
            n: 模因总数 (通常 n = len(sub_geos)); 显式传入以保证一致性.

        Returns:
            np.ndarray: 耦合矩阵 C, shape=(n, n), dtype=float32.
                        每个元素 C_{ij} ∈ [0,1], 对角线为 0, 矩阵对称.
        """
        # ---- 步骤0: 初始化零矩阵 ----
        # C ∈ R^{n×n}, 初始全部为零.
        C: np.ndarray = np.zeros((n, n), dtype=np.float32)

        # ---- 步骤1: 预计算每个子几何体的顶点集、边集和层级标识 ----
        # 这批预计算避免在双重循环中重复构造集合, 提升性能.
        vertex_sets: List[Set[int]] = []
        edge_sets: List[Set[tuple]] = []
        component_ids: List[int] = []

        for geo in sub_geos:
            K = geo.K
            # 顶点集: V_i ⊆ N (顶点索引集合)
            vertex_sets.append(set(K.vertices))
            # 边集: E_i ⊆ N×N, 用排序元组保证 (a,b)==(b,a) 的可比性
            edge_sets.append(set(tuple(sorted(e)) for e in K.edges))
            # component_id: 使用 level_labels 中的最小层级作为主要层级标识
            # 若无层级标签, 则 component_id = -1 (无层级归属)
            if K.level_labels:
                component_ids.append(min(K.level_labels.values()))
            else:
                component_ids.append(-1)

        # ---- 步骤2: 逐对计算耦合强度 ----
        # 双重循环遍历所有无序对 (i, j), i < j.
        for i in range(n):
            for j in range(i + 1, n):
                c_ij: float = 0.0

                # 规则1: 共享边检测 → +0.3
                # 数学: edges_i ∩ edges_j ≠ ∅ ⟹ 存在几何接触
                shared_edges: Set[tuple] = edge_sets[i] & edge_sets[j]
                if len(shared_edges) > 0:
                    c_ij += Coupling._WEIGHT_SHARED_EDGE

                # 规则2: 共享顶点检测 → +0.5 (阈值 ≥5)
                # 数学: |vertices_i ∩ vertices_j| ≥ 5 ⟹ 结构同源性显著
                shared_vertices: Set[int] = vertex_sets[i] & vertex_sets[j]
                if len(shared_vertices) >= Coupling._SHARED_VERTEX_THRESHOLD:
                    c_ij += Coupling._WEIGHT_SHARED_VERTEX

                # 规则3: 同层级检测 → +0.2
                # 数学: component_id_i == component_id_j ⟹ 功能层级相同
                if component_ids[i] == component_ids[j] and component_ids[i] != -1:
                    c_ij += Coupling._WEIGHT_SAME_COMPONENT

                # ---- 步骤3: 截断与对称赋值 ----
                # 数学约束: C_{ij} ∈ [0,1]
                c_ij = min(c_ij, Coupling._MAX_COUPLING)
                # 对称性: C 是双向耦合矩阵, C_{ij} = C_{ji}
                C[i, j] = c_ij
                C[j, i] = c_ij

        # ---- 步骤4: 最终保障 ----
        # 对角线强制置零: C_{ii} = 0 (自身无耦合)
        # np.fill_diagonal 就地修改, 无需额外赋值.
        np.fill_diagonal(C, 0.0)

        return C
