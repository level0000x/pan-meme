"""模块四 — Merkle 树测试 (MerkleTreeBuilder / MerkleVerifier)

数学对应: 附录D.4 — 稀疏 Merkle 树构造与分层验证.
测试 MerkleTreeBuilder.build 和 MerkleVerifier.verify_root / locate_tamper.
"""

import numpy as np
import pytest

from pan_meme.core.types import (
    HierarchicalHash,
    MerkleTree,
    Credential,
    CompositeMemeState,
    MemeState,
)
from pan_meme.module4_bind.hierarchy_hasher import HierarchyHasher, sha256
from pan_meme.module4_bind.merkle_tree import MerkleTreeBuilder, MerkleVerifier
from pan_meme.module4_bind.credential import CredentialAssembler


# ============================================================
# 辅助 fixture: 构造含多级哈希节点的凭证
# ============================================================

@pytest.fixture
def credential_with_hashes():
    """构造一个完整的凭证，含 token/meme/composite 多层哈希节点"""
    hasher = HierarchyHasher()

    # token 层哈希
    token1 = hasher.hash_component(
        layer="token", component_id="token_0",
        canonical_data='{"text":"信息"}',
        children=[], metadata={"idx": 0},
    )
    token2 = hasher.hash_component(
        layer="token", component_id="token_1",
        canonical_data='{"text":"结构"}',
        children=[], metadata={"idx": 1},
    )

    # meme 层哈希
    meme = MemeState(D=0.5, B=0.6, rho=0.3, R=0.2, S=0.7, xi=np.array([0.1], dtype=np.float32))
    meme_hash = hasher.hash_meme(meme, idx=0)

    # composite 层哈希
    Theta = [{"alpha_1": 0.5, "alpha_2": 0.3}]
    C = np.array([[0.0]], dtype=np.float32)
    composite_hash = hasher.hash_composite(Theta, C)

    # 收集所有哈希节点
    all_hashes = [token1, token2, meme_hash, composite_hash]

    # 构建 Merkle 树
    tree: MerkleTree = MerkleTreeBuilder.build(all_hashes)

    # 组装凭证
    memes = CompositeMemeState(memes=[meme], Theta=Theta, C=C)
    meta = {
        "original_size_bytes": 100,
        "meme_count": 1,
        "original_type": "str",
    }
    cred = CredentialAssembler.assemble(tree, memes, meta)

    return cred


def test_build_produces_root(credential_with_hashes):
    """MerkleTreeBuilder.build → root_hash 非空"""
    cred = credential_with_hashes
    tree = cred.merkle_tree

    # 根哈希必须存在且非空
    assert tree.root_hash is not None, "root_hash 不应为 None"
    assert len(tree.root_hash) > 0, "root_hash 不应为空字符串"
    assert len(tree.root_hash) == 64, f"root_hash 应为64字符，实际: {len(tree.root_hash)}"

    # nodes 字典非空
    assert len(tree.nodes) > 0, "nodes 字典不应为空"

    # leaf_index 应包含 token/meme/composite 层
    assert "token" in tree.leaf_index, "leaf_index 应包含 'token' 层"
    assert "meme" in tree.leaf_index, "leaf_index 应包含 'meme' 层"
    assert "composite" in tree.leaf_index, "leaf_index 应包含 'composite' 层"


def test_verify_root(credential_with_hashes):
    """有效凭证 verify_root → True"""
    cred = credential_with_hashes
    result = MerkleVerifier.verify_root(cred)
    assert result is True, "有效凭证的根哈希验证应返回 True"


def test_locate_tamper(credential_with_hashes):
    """修改 canonical_json_snapshot 后 locate_tamper 返回非空"""
    cred = credential_with_hashes
    tree = cred.merkle_tree

    # 选取 token 层一个叶子节点，篡改其 canonical_json_snapshot
    token_hashes = tree.leaf_index.get("token", [])
    assert len(token_hashes) > 0, "token 层应有叶子节点"

    target_hv = token_hashes[0]
    target_node = tree.nodes[target_hv]
    # 篡改: 修改快照内容而不更新 hash_value
    target_node.canonical_json_snapshot = '{"tampered": true}'

    # locate_tamper 应检测到被篡改的节点
    tampered = MerkleVerifier.locate_tamper(cred)
    assert len(tampered) > 0, (
        "篡改后 locate_tamper 应返回非空列表"
    )
