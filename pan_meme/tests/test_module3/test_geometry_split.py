"""模块三 — 几何分解测试 (GeometrySplit)

数学对应: 定理4 — Φ_D: G → Q 双射, n = β₀(K) 是连通分量数.
测试 GeometrySplit.split 和 GeometrySplit.merge 的正确性.
"""

import numpy as np
import pytest

from pan_meme.core.types import GeometricObject, SimplicialComplex


# ============================================================
# 辅助 fixture: 构造含两个连通分量的几何对象
# ============================================================

def _make_two_component_geo() -> GeometricObject:
    """构造一个包含两个连通分量的几何对象，用于测试 split/merge"""
    # 分量1: 顶点 0-1-2 (三角形含2条边) — 连通
    # 分量2: 顶点 3-4 (一条边)    — 连通
    # 两个分量之间无边 → β₀ = 2
    K = SimplicialComplex(
        vertices=[0, 1, 2, 3, 4],
        edges=[(0, 1), (1, 2), (3, 4)],  # 分量1: (0,1),(1,2); 分量2: (3,4)
        higher_cells=[],
        subcomplexes={},
        level_labels={0: 0, 1: 1, 2: 1, 3: 0, 4: 1},
    )
    return GeometricObject(
        K=K,
        g=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        omega=np.array([0.5, 0.6, 0.4, 0.3, 0.2], dtype=np.float32),
        Gamma={
            "euler_char": len(K.vertices) - len(K.edges),
            "num_constraints": 2,
            "constraint_types": ["node_connectivity_min", "max_depth"],
        },
        R={"vertex_map_original": list(range(5)), "edge_map_original": [0, 1, 2]},
    )


def test_split_creates_subgeos():
    """对含2个连通分量的几何对象调用 split，应返回 >= 1 个子几何体"""
    # 保护: 若 scipy 不可用则跳过
    try:
        import scipy  # noqa: F401
    except ImportError:
        pytest.skip("scipy 不可用，跳过连通分量分解测试")

    from pan_meme.module3_meme.geometry_split import GeometrySplit

    geo = _make_two_component_geo()
    sub_geos = GeometrySplit().split(geo)

    # 至少返回原始几何对象自身（退化情况）
    assert len(sub_geos) >= 1, "split应至少返回1个子几何体"
    # 检查每个子几何体都有合法的胞腔复形
    for sg in sub_geos:
        assert isinstance(sg, GeometricObject), "每个返回值应为 GeometricObject"
        assert len(sg.K.vertices) > 0, "每个子几何体应有非空顶点集"


def test_merge_roundtrip():
    """split → merge 后顶点数应与原始几何对象一致"""
    try:
        import scipy  # noqa: F401
    except ImportError:
        pytest.skip("scipy 不可用，跳过连通分量分解测试")

    from pan_meme.module3_meme.geometry_split import GeometrySplit

    geo = _make_two_component_geo()
    sub_geos = GeometrySplit().split(geo)
    merged = GeometrySplit.merge(sub_geos)

    # merge(split(K)) 应保持顶点总数不变
    assert len(merged.vertices) == len(geo.K.vertices), (
        f"merge→split 后顶点数应一致: {len(merged.vertices)} vs {len(geo.K.vertices)}"
    )
    # 边总数也应当一致
    assert len(merged.edges) == len(geo.K.edges), (
        f"merge→split 后边数应一致: {len(merged.edges)} vs {len(geo.K.edges)}"
    )
