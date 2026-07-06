"""模块四 — 分层哈希引擎测试 (HierarchyHasher)

数学对应: 论文3.5节 — 分层哈希绑定层, HierarchicalHash 的生成.
测试 HierarchyHasher.hash_token 和 hash_meme 产出的哈希节点结构正确.
"""

import numpy as np
import pytest

from pan_meme.core.types import (
    Token,
    MemeState,
    HierarchicalHash,
)
from pan_meme.module4_bind.hierarchy_hasher import HierarchyHasher


def test_hash_token():
    """hash_token 产生 HierarchicalHash，其 hash_value 长度为64（SHA-256 十六进制摘要）"""
    hasher = HierarchyHasher()

    # 构造一个简单 Token
    token = Token(
        modality="text",
        text="测试",
        span=(0, 2),
        pos="n",
    )

    result: HierarchicalHash = hasher.hash_token(token, idx=0)

    # 基础属性检验
    assert isinstance(result, HierarchicalHash), "返回值应为 HierarchicalHash"
    assert result.layer == "token", f"层级应为 'token'，实际: {result.layer}"
    assert result.component_id == "token_0", f"组件ID应为 'token_0'，实际: {result.component_id}"

    # SHA-256 十六进制摘要长度 = 64
    assert len(result.hash_value) == 64, (
        f"SHA-256 hash 长度应为 64，实际: {len(result.hash_value)}"
    )

    # hash_value 应全为十六进制小写字符
    assert all(c in "0123456789abcdef" for c in result.hash_value), (
        "hash_value 应全为十六进制小写字符"
    )


def test_hash_meme():
    """hash_meme 产生含 children 的节点（5d_hash + xi_hash）"""
    hasher = HierarchyHasher()

    # 构造一个 MemeState
    meme = MemeState(
        D=0.4,
        B=0.5,
        rho=0.3,
        R=0.2,
        S=0.6,
        xi=np.array([0.1, -0.05, 0.02], dtype=np.float32),
    )

    result: HierarchicalHash = hasher.hash_meme(meme, idx=0)

    # 层级与组件ID
    assert result.layer == "meme", f"层级应为 'meme'，实际: {result.layer}"
    assert result.component_id == "meme_0", f"组件ID应为 'meme_0'，实际: {result.component_id}"

    # 必须包含子节点: 5d_hash + xi_hash
    assert len(result.children) == 2, (
        f"hash_meme 应有2个子节点（5d_hash + xi_hash），实际: {len(result.children)}"
    )

    # 每个子节点哈希长度也为64
    for child_hv in result.children:
        assert len(child_hv) == 64, (
            f"子哈希长度应为 64，实际: {len(child_hv)}"
        )

    # 元数据应包含模因各维度
    assert "D" in result.metadata, "元数据应包含 D"
    assert "xi_dim" in result.metadata, "元数据应包含 xi_dim"
