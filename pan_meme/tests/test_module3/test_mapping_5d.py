"""模块三 — 五维映射测试 (Mapping5D)

数学对应: 前提5 — K_i → X_i = (m_i, ξ_i) 双射.
测试 Mapping5D.map 产出的 MemeState 各维度均在合法区间内.
"""

import numpy as np
import pytest

from pan_meme.core.types import GeometricObject, MemeState, SimplicialComplex
from pan_meme.module3_meme.mapping_5d import Mapping5D


# ============================================================
# 辅助 fixture: 构造合法的子几何体
# ============================================================

def _make_simple_sub_geo() -> GeometricObject:
    """构造一个简单子几何体用于五维映射测试"""
    K = SimplicialComplex(
        vertices=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        higher_cells=[],
        subcomplexes={},
        level_labels={0: 0, 1: 1, 2: 1},
    )
    return GeometricObject(
        K=K,
        g=np.array([0.8, 0.6], dtype=np.float32),
        omega=np.array([0.3, 0.5, 0.4], dtype=np.float32),
        Gamma={"euler_char": 3 - 2, "num_constraints": 1, "constraint_types": ["node_connectivity_min"]},
        R={},
    )


def test_dimensions_in_range():
    """map()返回的模因状态 D, B, rho, R, S 全 ∈ [0, 1]"""
    sub_geo = _make_simple_sub_geo()
    meme: MemeState = Mapping5D.map(sub_geo, idx=0)

    # 五维核心均应在 [0, 1] 区间内（定义6: Ω = [0,1]^5）
    for dim_name, value in [
        ("D",   meme.D),
        ("B",   meme.B),
        ("rho", meme.rho),
        ("R",   meme.R),
        ("S",   meme.S),
    ]:
        assert 0.0 <= value <= 1.0, (
            f"{dim_name}={value} 超出 [0,1] 范围"
        )


def test_xi_dimension():
    """ξ 的长度应等于子几何体的边数"""
    sub_geo = _make_simple_sub_geo()
    n_edges = len(sub_geo.K.edges)

    meme: MemeState = Mapping5D.map(sub_geo, idx=0)

    # 扩展维度 ξ 的长度 = 子几何体边数（定义4）
    assert len(meme.xi) == n_edges, (
        f"ξ 长度 {len(meme.xi)} 应等于边数 {n_edges}"
    )
