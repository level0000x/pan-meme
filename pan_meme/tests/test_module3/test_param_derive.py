"""模块三 — 11参数推导测试 (ParamDerive)

数学对应: 论文4.3.7节 — 11个动力学参数由几何特征与模因状态闭式确定.
测试 ParamDerive.derive 产出的参数均 ≥0, 且 alpha_1 ∈ [0,1].
"""

import numpy as np
import pytest

from pan_meme.core.types import GeometricObject, MemeState, SimplicialComplex
from pan_meme.module3_meme.param_derive import ParamDerive


# ============================================================
# 辅助 fixture: 构造 (MemeState, GeometricObject) 对
# ============================================================

@pytest.fixture
def param_inputs():
    """构造合法的 MemeState + GeometricObject 对，用于参数推导"""
    K = SimplicialComplex(
        vertices=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3)],
        higher_cells=[],
        subcomplexes={},
        level_labels={0: 0, 1: 1, 2: 1, 3: 2},
    )
    sub_geo = GeometricObject(
        K=K,
        g=np.array([0.8, 0.6, 0.7], dtype=np.float32),
        omega=np.array([0.3, 0.5, 0.4, 0.6], dtype=np.float32),
        Gamma={
            "euler_char": 4 - 3,
            "num_constraints": 2,
            "constraint_types": ["node_connectivity_min", "max_depth"],
        },
        R={},
    )
    meme = MemeState(D=0.4, B=0.5, rho=0.3, R=0.2, S=0.6, xi=np.array([0.1, -0.05, 0.02], dtype=np.float32))
    return meme, sub_geo


def test_all_params_nonnegative(param_inputs):
    """11个动力学参数全部 >= 0"""
    meme, sub_geo = param_inputs
    theta = ParamDerive.derive(meme, sub_geo)

    # 11参数名列表（论文4.3.7节）
    param_names = [
        "alpha_1", "alpha_2",
        "beta_1", "beta_2",
        "gamma_1", "gamma_2",
        "delta_1", "delta_2", "delta_3",
        "epsilon_1", "epsilon_2",
    ]
    for name in param_names:
        value = theta[name]
        assert value >= 0.0, (
            f"参数 {name}={value} 应为非负数"
        )


def test_alpha1_range(param_inputs):
    """alpha_1 必须在 [0, 1] 区间内"""
    meme, sub_geo = param_inputs
    theta = ParamDerive.derive(meme, sub_geo)

    alpha_1 = theta["alpha_1"]
    # alpha_1 = min(2·|E|/(|V|·(|V|-1)), 1.0) — 边密度归一化，必然 ∈ [0,1]
    assert 0.0 <= alpha_1 <= 1.0, (
        f"alpha_1={alpha_1} 超出 [0,1] 范围"
    )
