# 泛模因几何工具 — 模块四：分层哈希绑定层
# 数学对应：论文 3.5 节, 定理 5 自洽性验证
# 论文位置：附录 D.4 哈希树构造与凭证流
#
# 本模块实现管线完整性的分层哈希绑定与验证:
#   - HierarchyHasher:   对每一层产物生成独立 HierarchicalHash
#   - MerkleTreeBuilder:  构建稀疏 Merkle 树 (叶子 → 模块聚合 → root_hash)
#   - MerkleVerifier:     全量/分层/单组件三级粒度验证
#   - MerkleUpdater:      增量更新 (仅重算受影响路径)
#   - CredentialAssembler: 组装完整凭证 (tree + memes + meta)
#   - HashBasedIndex:      基于哈希的快速检索
#   - CredentialBinder:    模块四主入口, 统一 API
#   - MerkleDiffManager:   PB级新增 — Merkle 差异管理与 epoch 回滚
#   - CASMerkleTree:       PB 级内容寻址存储 Merkle 树 (PB_ARCHITECTURE.md 第3.5节)
#   - CASStorage:          可插拔存储后端 (memory/s3/ipfs)
#
# ================================================================
# PB级 CAS Merkle 模式使用说明
# ================================================================
#
# 1. CAS Merkle 树构建 (PB 级场景)
#    from pan_meme.module4_bind import MerkleTreeBuilder
#    tree, manifest = MerkleTreeBuilder.build_cas(
#        all_hashes,
#        partition_size=1_000_000,   # 每分区最多 100 万叶子
#        storage_backend="memory",   # memory | s3 | ipfs
#    )
#
# 2. Merkle 差异管理与回滚
#    from pan_meme.module4_bind import MerkleDiffManager
#    diff_mgr = MerkleDiffManager(cas_tree)
#    report = diff_mgr.compute_and_apply_diff(old_tree, new_tree, epoch=1)
#    history = diff_mgr.get_diff_history()
#    restored = diff_mgr.rollback_to_epoch(0)
#
# 3. 通过 build() 的 use_cas 参数启用 CAS 模式
#    tree = MerkleTreeBuilder.build(all_hashes, use_cas=True)

from pan_meme.module4_bind.hierarchy_hasher import HierarchyHasher, sha256
from pan_meme.module4_bind.merkle_tree import (
    MerkleTreeBuilder,
    MerkleVerifier,
    MerkleUpdater,
    MerkleDiffManager,
)
from pan_meme.module4_bind.credential import (
    CredentialAssembler,
    HashBasedIndex,
)
from pan_meme.module4_bind.binder import CredentialBinder

# -------------------- CAS Merkle 安全导入 (PB级新增) --------------------
try:
    from pan_meme.module4_bind.cas_merkle import CASMerkleTree, CASStorage
    _CAS_AVAILABLE = True
except ImportError:
    CASMerkleTree = None  # type: ignore
    CASStorage = None  # type: ignore
    _CAS_AVAILABLE = False

__all__ = [
    "HierarchyHasher",
    "sha256",
    "MerkleTreeBuilder",
    "MerkleVerifier",
    "MerkleUpdater",
    "MerkleDiffManager",
    "CredentialAssembler",
    "HashBasedIndex",
    "CredentialBinder",
    "CASMerkleTree",
    "CASStorage",
]
