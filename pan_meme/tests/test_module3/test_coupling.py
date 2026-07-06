"""模块三 — 耦合矩阵测试 (Coupling)

数学对应: 论文3.4.4节 — 子几何体间的空间关系自动生成耦合强度矩阵.
测试 Coupling.generate 产出的 C 满足对称性、对角线为零、元素 ∈ [0,1].
"""

import numpy as np
import pytest

from pan_meme.core.types import GeometricObject, SimplicialComplex
from pan_meme.module3_meme.coupling import Coupling


# ============================================================
# 辅助 fixture: 构造子几何体列表
# ============================================================

@pytest.fixture
def sub_geos_for_coupling():
    """构造2个子几何体列表，用于耦合矩阵生成"""
    # 子几何体1: 顶点 0,1,2 边 (0,1),(1,2)
    K1 = SimplicialComplex(
        vertices=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        higher_cells=[],
        subcomplexes={},
        level_labels={0: 0, 1: 1, 2: 1},
    )
    geo1 = GeometricObject(
        K=K1,
        g=np.array([0.8, 0.6], dtype=np.float32),
        omega=np.array([0.3, 0.5, 0.4], dtype=np.float32),
        Gamma={"euler_char": 3 - 2},
        R={},
    )
    # 子几何体2: 顶点 0,1,2 边 (0,1) — 与 geo1 有部分重叠顶点检测用
    K2 = SimplicialComplex(
        vertices=[0, 1, 2],
        edges=[(0, 1)],
        higher_cells=[],
        subcomplexes={},
        level_labels={0: 0, 1: 1, 2: 0},
    )
    geo2 = GeometricObject(
        K=K2,
        g=np.array([0.9], dtype=np.float32),
        omega=np.array([0.2, 0.4, 0.1], dtype=np.float32),
        Gamma={"euler_char": 3 - 1},
        R={},
    )
    return [geo1, geo2]


def test_symmetric(sub_geos_for_coupling):
    """耦合矩阵 C 必须是对称的（C = C.T）"""
    sub_geos = sub_geos_for_coupling
    n = len(sub_geos)
    C = Coupling.generate(sub_geos, n)

    # 对称性检验: C_{ij} == C_{ji} 对所有 i, j
    assert np.allclose(C, C.T), (
        "耦合矩阵必须对称: C != C.T"
    )


def test_diagonal_zero(sub_geos_for_coupling):
    """耦合矩阵对角线必须全为零（C_ii = 0，无自耦合）"""
    sub_geos = sub_geos_for_coupling
    n = len(sub_geos)
    C = Coupling.generate(sub_geos, n)

    diag = np.diag(C)
    assert np.allclose(diag, 0.0), (
        f"对角线应全为零，实际值: {diag}"
    )


def test_range(sub_geos_for_coupling):
    """耦合矩阵所有元素必须在 [0, 1] 区间内"""
    sub_geos = sub_geos_for_coupling
    n = len(sub_geos)
    C = Coupling.generate(sub_geos, n)

    # 所有元素 ∈ [0, 1]
    assert np.all(C >= 0.0), "耦合矩阵存在负元素"
    assert np.all(C <= 1.0), "耦合矩阵存在 >1 元素"
