"""模块四 — 凭证绑定器测试 (CredentialBinder)

数学对应: 论文3.5节 — CredentialBinder 是模块四的编排器.
测试 binder.bind 和 verify_full 的正确性.
"""

import numpy as np
import pytest

from pan_meme.core.types import (
    PipelineData,
    HierarchicalHash,
    MemeState,
    CompositeMemeState,
    RelationNetwork,
)
from pan_meme.module4_bind.binder import CredentialBinder
from pan_meme.module4_bind.hierarchy_hasher import HierarchyHasher


# ============================================================
# 辅助 fixture: 构造含预计算哈希节点的 PipelineData
# ============================================================

@pytest.fixture
def data_with_hashes(relation_network_chain: RelationNetwork):
    """构造 PipelineData，包含预计算的分层哈希节点"""
    hasher = HierarchyHasher()

    # 为链关系网络 A->B->C->D 构造 token 层哈希
    all_hashes = []
    for i, node_text in enumerate(relation_network_chain.nodes):
        all_hashes.append(hasher.hash_component(
            layer="token",
            component_id=f"token_{i}",
            canonical_data=f'{{"text":"{node_text}"}}',
            children=[],
            metadata={"idx": i},
        ))

    # 构造 meme 层哈希
    meme = MemeState(D=0.5, B=0.6, rho=0.3, R=0.2, S=0.7, xi=np.array([0.1], dtype=np.float32))
    meme_hash = hasher.hash_meme(meme, idx=0)
    all_hashes.append(meme_hash)

    # composite 层哈希
    Theta = [{"alpha_1": 0.5}]
    C = np.array([[0.0]], dtype=np.float32)
    composite_hash = hasher.hash_composite(Theta, C)
    all_hashes.append(composite_hash)

    # 构造 CompositeMemeState
    memes = CompositeMemeState(memes=[meme], Theta=Theta, C=C)

    return PipelineData(
        input="测试输入",
        psi=relation_network_chain,
        meme_state=memes,
        all_hash_nodes=all_hashes,
        meta={},
    )


def test_bind_assembles_credential(data_with_hashes):
    """binder.bind(data_with_hashes) → credential 非空"""
    binder = CredentialBinder()
    result = binder.bind(data_with_hashes)

    # 凭证应被正确组装
    assert result.credential is not None, "bind 后 credential 不应为 None"
    cred = result.credential

    # 凭证基本结构检验
    assert cred.data_hash is not None, "data_hash 不应为 None"
    assert len(cred.data_hash) == 64, f"data_hash 应为64字符，实际: {len(cred.data_hash)}"
    assert cred.merkle_tree is not None, "merkle_tree 不应为 None"
    assert cred.header is not None, "header 不应为 None"
    assert "version" in cred.header, "header 应含 version 字段"


def test_verify_full(data_with_hashes):
    """有效凭证 verify_full → True"""
    binder = CredentialBinder()
    data = binder.bind(data_with_hashes)
    cred = data.credential

    result = binder.verify_full(cred)
    assert result is True, "有效凭证的全量验证应返回 True"
