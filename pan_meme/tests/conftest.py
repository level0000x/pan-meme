"""pan-meme 全局测试配置与 fixtures"""

import pytest
import numpy as np

# ============================================================
# 基础 fixtures
# ============================================================

@pytest.fixture
def sample_text_cn() -> str:
    """标准中文测试文本"""
    return "信息是现实的基本维度。结构具有因果效力。泛模因是信息-结构模式。"


@pytest.fixture
def sample_text_en() -> str:
    """标准英文测试文本"""
    return "Information is the fundamental dimension of reality. Structure has causal power."


@pytest.fixture
def sample_json() -> dict:
    """标准结构化测试数据"""
    return {
        "name": "test_project",
        "version": "1.0",
        "dependencies": ["numpy", "scipy", "networkx"],
        "config": {"mode": "converge", "max_rounds": 20},
    }


@pytest.fixture
def pipeline_config() -> dict:
    """管线程标准配置 — 若 default.json 不存在则返回内置默认值"""
    import json, os
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "default.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # 回退：内置默认配置，与 Pipeline.__init__ 中的默认参数一致
    return {
        "module1": {
            "cycle_mode": "converge",
            "cycle_max_rounds": 20,
            "threshold_default": 0.5,
            "transitive_decay": 0.9,
            "symmetric_decay": 0.85,
        },
        "module2": {},
        "module3": {},
        "module4": {},
    }


@pytest.fixture
def relation_network_chain() -> "RelationNetwork":
    """链状关系网络 A->B->C->D (4节点)"""
    from pan_meme.core.types import RelationNetwork
    return RelationNetwork(
        nodes=["A", "B", "C", "D"],
        edges=[(0, 1), (1, 2), (2, 3)],
        weights=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        hierarchy={"levels": 1, "node_levels": {0: 0, 1: 1, 2: 1, 3: 0}},
        metadata={"input_type": "synthetic", "node_count": 4, "edge_count": 3, "threshold_used": 0.5},
    )


@pytest.fixture
def relation_network_complete() -> "RelationNetwork":
    """完全图 K4"""
    from pan_meme.core.types import RelationNetwork
    return RelationNetwork(
        nodes=["A", "B", "C", "D"],
        edges=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        weights=np.array([0.9, 0.8, 0.6, 0.7, 0.5, 0.4], dtype=np.float32),
        hierarchy={"levels": 1, "node_levels": {0: 0, 1: 0, 2: 1, 3: 1}},
        metadata={"input_type": "synthetic", "node_count": 4, "edge_count": 6, "threshold_used": 0.3},
    )


# ============================================================
# Hypothesis property test helpers
# ============================================================

try:
    from hypothesis import strategies as st

    text_strategy = st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=10, max_size=200,
    )

    meme_vector_strategy = st.tuples(
        st.floats(0.0, 1.0), st.floats(0.0, 1.0),
        st.floats(0.0, 1.0), st.floats(0.0, 1.0),
        st.floats(0.0, 1.0),
    )

    graph_strategy = st.integers(3, 10).flatmap(
        lambda n: st.tuples(
            st.just(n),
            st.lists(
                st.tuples(st.integers(0, n - 1), st.integers(0, n - 1)).filter(lambda e: e[0] != e[1]),
                min_size=n, max_size=n * 3, unique_by=lambda e: (min(e), max(e)),
            ),
        )
    )

except ImportError:
    text_strategy = None
    meme_vector_strategy = None
    graph_strategy = None
