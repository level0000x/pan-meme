# 泛模因几何工具 — 模块四：内容寻址存储 (CAS) Merkle 树
# 数学对应：PB_ARCHITECTURE.md 第3.5节 — IPFS CID 风格内容寻址 + 分区 Merkle + 增量 diff
# 论文位置：定理 5 自洽性验证, 附录 D.4 哈希树构造与凭证流
# 核心思想：当 token/meme/子几何体数量达到 10^13 时, 传统内存 Merkle 树无法承载.
#           通过 IPFS CID 格式的内容寻址引用替代节点存储,
#           分区聚合 (每 10^6 叶子 → 1 个 L1 内部节点) 实现层级压缩,
#           增量 diff 仅存储变更节点 + 重算受影响 Merkle 路径.

import hashlib
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

from pan_meme.core.types import (
    HierarchicalHash,
    MerkleTree,
    Credential,
)


# ============================================================
# 轻量级哈希函数 — CID 计算
# ============================================================

def _sha256_hex(data: str) -> str:
    """SHA-256 十六进制摘要.

    数学对应: 密码学哈希 H: {0,1}* → {0,1}^256.

    Args:
        data: 待哈希字符串.

    Returns:
        64 字符十六进制小写哈希串.
    """
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


# ============================================================
# CASStorage — 可插拔存储后端
# ============================================================

class CASStorage:
    """内容寻址存储后端 — 支持 memory / s3 / ipfs 三种后端.

    数学对应: 第3.5节 — 存储层.
    存储分层:
      - L0 CID 引用: S3 Parquet 列存, 一列 = CID, 一列 = canonical_snapshot_S3_key
      - L1+ 节点: S3 JSON, key = CID
      - 根哈希: Merkle 证明物, 锚定到链上 (可选)

    "memory" 后端使用 dict 实现, 适用于开发/测试环境.
    "s3" / "ipfs" 后端为存根 (stub), 标注 "not_implemented".

    用法:
        store = CASStorage(backend="memory")
        store.put("cid-abc123", b"hello world")
        data = store.get("cid-abc123")
        exists = store.exists("cid-abc123")
    """

    # ----------------------------------------------------------------
    # 构造与初始化
    # ----------------------------------------------------------------

    def __init__(self, backend: str = "memory") -> None:
        """初始化 CAS 存储后端.

        Args:
            backend: 存储后端标识, 可选 "memory" | "s3" | "ipfs".
                     默认 "memory" (dict 实现).
        """
        self.backend: str = backend
        """存储后端标识."""

        self._store: Dict[str, bytes] = {}
        """内存存储字典 (仅 memory 后端生效)."""

        if backend not in ("memory", "s3", "ipfs"):
            raise ValueError(
                f"不支持的存储后端: '{backend}'. "
                "可选: 'memory', 's3', 'ipfs'."
            )

    # ----------------------------------------------------------------
    # CRUD 接口
    # ----------------------------------------------------------------

    def put(self, cid: str, data: bytes) -> bool:
        """存储内容.

        数学对应: CAS 写入 — 内容以 CID 为键存储, 不可变语义.

        Args:
            cid:  内容寻址标识符 (如 "cid-abc123...").
            data: 原始字节内容.

        Returns:
            True 若存储成功.

        Raises:
            NotImplementedError: 若后端为 "s3" 或 "ipfs" (存根).
        """
        if self.backend == "s3":
            raise NotImplementedError(
                "S3 存储后端尚未实现. "
                "预期使用 boto3 put_object(Bucket, Key=cid, Body=data)."
            )
        if self.backend == "ipfs":
            raise NotImplementedError(
                "IPFS 存储后端尚未实现. "
                "预期使用 ipfshttpclient add(data)."
            )

        # memory 后端
        self._store[cid] = data
        return True

    def get(self, cid: str) -> Optional[bytes]:
        """获取内容.

        数学对应: CAS 读取 — 通过 CID 检索不可变内容.

        Args:
            cid: 内容寻址标识符.

        Returns:
            字节内容, 若不存在则返回 None.

        Raises:
            NotImplementedError: 若后端为 "s3" 或 "ipfs" (存根).
        """
        if self.backend == "s3":
            raise NotImplementedError(
                "S3 存储后端尚未实现. "
                "预期使用 boto3 get_object(Bucket, Key=cid)['Body'].read()."
            )
        if self.backend == "ipfs":
            raise NotImplementedError(
                "IPFS 存储后端尚未实现. "
                "预期使用 ipfshttpclient cat(cid)."
            )

        # memory 后端
        return self._store.get(cid)

    def exists(self, cid: str) -> bool:
        """检查 CID 对应的内容是否存在.

        Args:
            cid: 内容寻址标识符.

        Returns:
            True 若存在.

        Raises:
            NotImplementedError: 若后端为 "s3" 或 "ipfs" (存根).
        """
        if self.backend == "s3":
            raise NotImplementedError(
                "S3 存储后端尚未实现. "
                "预期使用 boto3 head_object(Bucket, Key=cid)."
            )
        if self.backend == "ipfs":
            raise NotImplementedError(
                "IPFS 存储后端尚未实现. "
                "预期使用 ipfshttpclient files stat('/ipfs/' + cid)."
            )

        # memory 后端
        return cid in self._store

    def delete(self, cid: str) -> bool:
        """删除 CID 对应的内容.

        Args:
            cid: 内容寻址标识符.

        Returns:
            True 若删除成功 (包含已不存在的情况, 幂等).

        Raises:
            NotImplementedError: 若后端为 "s3" 或 "ipfs" (存根).
        """
        if self.backend == "s3":
            raise NotImplementedError(
                "S3 存储后端尚未实现. "
                "预期使用 boto3 delete_object(Bucket, Key=cid)."
            )
        if self.backend == "ipfs":
            raise NotImplementedError(
                "IPFS 存储后端尚未实现. "
                "IPFS 内容不可变, 无法删除; 仅能 unpin."
            )

        # memory 后端
        self._store.pop(cid, None)
        return True


# ============================================================
# CASMerkleTree — 内容寻址存储 Merkle 树
# ============================================================

class CASMerkleTree:
    """内容寻址存储 (CAS) Merkle 树 — IPFS CID 风格 + 分区聚合 + 增量 diff.

    数学对应: PB_ARCHITECTURE.md 第3.5节 — 分层分区 Merkle 树.
    分层结构:
      L0 (叶子):      每个 hash 节点存储为 CID(content).
                      不存节点本身, 仅存 CID 引用.
      L1 (分组聚合):  每 leaf_partition_size 个叶子聚合为 1 个内部节点,
                      hash = H(concat(L0_CIDs))
      L2 (模块聚合):  各层级 L1 节点再聚合为模块哈希
                      (mod_hash = H(concat(L1_CIDs)))
      L3 (根):        模块级聚合,
                      root_hash = H(mod1_CID | mod2_CID | mod3_CID)

    存储:
      - L0 CID 引用: S3 Parquet 列存, 一列 = CID, 一列 = canonical_snapshot_S3_key
      - L1+ 节点: S3 JSON, key = CID
      - 根哈希: Merkle 证明物, 锚定到链上 (可选)

    增量:
      - 新增/变更叶子 → 只重算 L0 → L1 → L2 → root 路径
      - 未变更分片完全不动
      - 每日增量: 仅处理 diff

    用法:
        cas = CASMerkleTree(leaf_partition_size=1000000, storage_backend="memory")
        tree = cas.build(all_hashes)
        manifest = cas.to_partitioned_manifest(tree)
        diff = cas.compute_diff(old_tree, new_tree)
        updated_tree = cas.apply_diff(tree, diff)
    """

    # ----------------------------------------------------------------
    # 构造与初始化
    # ----------------------------------------------------------------

    def __init__(
        self,
        leaf_partition_size: int = 1000000,
        storage_backend: Optional[str] = None,
    ) -> None:
        """初始化 CAS Merkle 树构建器.

        数学对应: 第3.5节 — 分区参数化.
          - P = leaf_partition_size: 每 10^6 个叶子聚合为 1 个 L1 内部节点.
          - storage_backend: 可选 "memory" | "s3" | "ipfs".

        Args:
            leaf_partition_size: 每个 L1 聚合组的叶子数, 默认 1,000,000 (10^6).
            storage_backend: 存储后端标识. 默认 "memory".
        """
        self.leaf_partition_size: int = leaf_partition_size
        """每 P 个叶子聚合为 1 个 L1 内部节点."""

        self.storage: CASStorage = CASStorage(
            backend=storage_backend if storage_backend else "memory"
        )
        """CAS 存储后端实例."""

        # 记录分区元数据, 用于增量 diff
        self._partition_meta: Dict[str, Any] = {}
        """内部分区元数据记录."""

        # 模块 → 子层映射 (与 merkle_tree.py 一致)
        self._MODULE_LAYERS: Dict[str, List[str]] = {
            "module_1": ["token", "hierarchy", "relation", "math_model"],
            "module_2": ["sub_geo"],
            "module_3": ["sub_geo", "meme", "composite"],
        }

    # ----------------------------------------------------------------
    # CID 计算
    # ----------------------------------------------------------------

    def _cid(self, content: str) -> str:
        """计算内容寻址标识符 CID = sha256(content).

        数学对应: 第3.5节 L0 — CID(content) = sha256(content).
        格式前缀 "cid-" 与 IPFS CID 兼容, 便于跨系统互通.

        Args:
            content: 待寻址的内容字符串 (规范化 JSON 快照).

        Returns:
            CID 字符串, 格式为 "cid-" + sha256_hex(content)[:40].
        """
        h: str = _sha256_hex(content)
        # 使用前 40 位十六进制字符以控制 CID 长度
        return "cid-" + h[:40]

    def _cid_from_hash_value(self, hash_value: str) -> str:
        """从 HierarchicalHash.hash_value 生成 CID.

        Args:
            hash_value: SHA-256 十六进制哈希值.

        Returns:
            CID 字符串.
        """
        # 直接以 hash_value 作为内容计算 CID, 保证一致性
        return "cid-" + _sha256_hex(hash_value)[:40]

    # ----------------------------------------------------------------
    # Merkle 树构建 — 四层分区聚合
    # ----------------------------------------------------------------

    def build(self, all_hashes: List[HierarchicalHash]) -> MerkleTree:
        """从所有分层哈希节点构建 CAS 分区 Merkle 树.

        数学对应: 第3.5节 — 四层构建算法.
        构建流程:
          L0 (叶子): 每个 hash 节点存储为 CID(content).
                     不存节点本身, 仅存 CID 引用.
          L1 (分组聚合): 每 leaf_partition_size 个叶子聚合,
                         hash = sha256(concat(L0_CIDs)).
          L2 (模块聚合): 各层级 L1 节点再聚合为模块哈希.
          L3 (根):      sha256(mod1 | mod2 | mod3) → root_hash.

        Args:
            all_hashes: 管线所有层级的所有组件哈希节点列表.

        Returns:
            完整的 MerkleTree, 包含 root_hash, nodes, leaf_index.

        Raises:
            ValueError: 当 all_hashes 为空时.
        """
        if not all_hashes:
            raise ValueError(
                "无法构建 CAS Merkle 树: all_hashes 为空. "
                "请确保管线已生成至少一个分层哈希节点."
            )

        # ---- 步骤1: L0 — 索引叶子并计算 CID ----
        # 数学对应: L0 叶子 = CID(canonical_json_snapshot)
        nodes: Dict[str, HierarchicalHash] = {}
        leaf_index: Dict[str, List[str]] = {}

        # 为每个 hash 节点生成 CID 引用并存入 CAS
        l0_cids: List[str] = []
        layer_by_cid: Dict[str, str] = {}

        for h in all_hashes:
            cid: str = self._cid(h.canonical_json_snapshot)
            l0_cids.append(cid)
            layer_by_cid[cid] = h.layer

            # 将内容存入 CAS (memory | s3 | ipfs)
            content_bytes: str = h.canonical_json_snapshot
            self.storage.put(cid, content_bytes.encode('utf-8'))

            # 索引到 leaf_index
            nodes[h.hash_value] = h
            layer = h.layer
            if layer not in leaf_index:
                leaf_index[layer] = []
            leaf_index[layer].append(h.hash_value)

        # ---- 步骤2: L1 — 分组聚合 ----
        # 数学对应: L1 hash = sha256(concat(L0_CIDs))
        # 每 P = leaf_partition_size 个叶子聚合成 1 个 L1 节点
        P: int = self.leaf_partition_size
        n_leaves: int = len(l0_cids)
        n_partitions: int = max(1, (n_leaves + P - 1) // P)

        l1_cids: List[str] = []
        partition_records: List[Dict[str, Any]] = []

        for p_idx in range(n_partitions):
            start: int = p_idx * P
            end: int = min(start + P, n_leaves)
            partition_cids: List[str] = l0_cids[start:end]

            # L1 节点: 合并分区内所有 L0 CID
            combined: str = "|".join(partition_cids)
            l1_hash: str = _sha256_hex(combined)
            l1_cid: str = self._cid(combined)

            l1_cids.append(l1_cid)

            # 将 L1 节点存入 CAS
            l1_node: HierarchicalHash = HierarchicalHash(
                layer="internal_l1",
                component_id=f"partition_{p_idx}",
                hash_value=l1_hash,
                canonical_json_snapshot=combined,
                children=partition_cids,
                metadata={
                    "partition_idx": p_idx,
                    "first_leaf_idx": start,
                    "last_leaf_idx": end - 1,
                    "leaf_count": len(partition_cids),
                    "level": "L1",
                },
            )
            nodes[l1_hash] = l1_node
            self.storage.put(l1_cid, combined.encode('utf-8'))

            partition_records.append({
                "cid": l1_cid,
                "layer": "L1",
                "partition_idx": p_idx,
                "first_leaf_idx": start,
                "last_leaf_idx": end - 1,
                "leaf_count": len(partition_cids),
                "hash_value": l1_hash,
            })

        # ---- 步骤3: L2 — 模块聚合 ----
        # 数学对应: L2 mod_hash = sha256(concat(L1_CIDs_for_module))
        # 按模块 (module_1/2/3) 聚合对应层的 L1 节点
        module_hashes: Dict[str, str] = {}
        module_cids: Dict[str, str] = {}

        for mod_name, sub_layers in self._MODULE_LAYERS.items():
            # 收集该模块所有子层的 L0 CID (按 layer 分组)
            mod_leaf_cids: List[str] = []
            for h in all_hashes:
                if h.layer in sub_layers:
                    cid: str = self._cid(h.canonical_json_snapshot)
                    mod_leaf_cids.append(cid)

            # 若无对应叶子, 使用哨兵
            if not mod_leaf_cids:
                sentinel_cid: str = self._cid(f"empty_{mod_name}")
                mod_leaf_cids.append(sentinel_cid)

            # 合并该模块所有叶子 CID → 模块哈希
            mod_combined: str = "|".join(sorted(mod_leaf_cids))
            mod_hash: str = _sha256_hex(mod_combined)
            mod_cid: str = self._cid(mod_combined)

            module_hashes[mod_name] = mod_hash
            module_cids[mod_name] = mod_cid

            # 创建 L2 模块节点
            mod_node: HierarchicalHash = HierarchicalHash(
                layer="internal_l2",
                component_id=f"{mod_name}_cas_hash",
                hash_value=mod_hash,
                canonical_json_snapshot=mod_combined,
                children=mod_leaf_cids,
                metadata={
                    "module": mod_name,
                    "sub_layers": sub_layers,
                    "leaf_count": len(mod_leaf_cids),
                    "level": "L2",
                },
            )
            nodes[mod_hash] = mod_node
            self.storage.put(mod_cid, mod_combined.encode('utf-8'))

        # ---- 步骤4: L3 — 根哈希 ----
        # 数学对应: L3 root_hash = sha256(mod1 | mod2 | mod3)
        root_children: List[str] = [
            module_hashes.get("module_1", _sha256_hex("empty")),
            module_hashes.get("module_2", _sha256_hex("empty")),
            module_hashes.get("module_3", _sha256_hex("empty")),
        ]
        root_combined: str = "|".join(root_children)
        root_hash: str = _sha256_hex(root_combined)

        root_node: HierarchicalHash = HierarchicalHash(
            layer="internal_l3",
            component_id="root_cas",
            hash_value=root_hash,
            canonical_json_snapshot=root_combined,
            children=root_children,
            metadata={
                "level": "L3",
                "module_count": 3,
                "total_partitions": n_partitions,
                "total_leaves": n_leaves,
            },
        )
        nodes[root_hash] = root_node
        root_cid: str = self._cid(root_combined)
        self.storage.put(root_cid, root_combined.encode('utf-8'))

        # ---- 记录分区元数据 ----
        self._partition_meta = {
            "leaf_partition_size": P,
            "n_partitions": n_partitions,
            "total_leaves": n_leaves,
            "partitions": partition_records,
            "module_hashes": module_hashes,
            "module_cids": module_cids,
            "root_cid": root_cid,
            "root_hash": root_hash,
        }

        # ---- 构建并返回 MerkleTree ----
        return MerkleTree(
            root_hash=root_hash,
            nodes=nodes,
            leaf_index=leaf_index,
        )

    # ----------------------------------------------------------------
    # 分区清单 — 序列化 / 反序列化
    # ----------------------------------------------------------------

    def to_partitioned_manifest(self, tree: MerkleTree) -> Dict[str, Any]:
        """生成分区清单.

        数学对应: 第3.5节存储 — S3 Parquet 列存清单.
        清单内容:
          - partitions: [{cid, layer, leaf_count, first_leaf_idx, last_leaf_idx}, ...]
          - root_cid: 根 CID
          - total_leaves: 总叶子数

        Args:
            tree: MerkleTree 实例.

        Returns:
            分区清单字典, 包含 partitions, root_cid, total_leaves 等字段.
        """
        # 统计总叶子数
        total_leaves: int = sum(
            len(leaf_list) for leaf_list in tree.leaf_index.values()
        )

        # 从 nodes 中提取 L1 分区节点
        partitions: List[Dict[str, Any]] = []
        for hv, node in tree.nodes.items():
            if node.metadata.get("level") == "L1":
                partitions.append({
                    "cid": self._cid(node.canonical_json_snapshot),
                    "layer": node.layer,
                    "leaf_count": node.metadata.get("leaf_count", 0),
                    "first_leaf_idx": node.metadata.get("first_leaf_idx", 0),
                    "last_leaf_idx": node.metadata.get("last_leaf_idx", 0),
                    "hash_value": node.hash_value,
                })

        # 按 partition_idx 排序
        partitions.sort(key=lambda p: p["first_leaf_idx"])

        # 获取 root_cid
        root_cid: str = self._cid(
            "|".join([
                node.hash_value
                for hv, node in tree.nodes.items()
                if node.metadata.get("level") == "L3"
            ] or ["empty"])
        )

        return {
            "partitions": partitions,
            "root_cid": root_cid,
            "root_hash": tree.root_hash,
            "total_leaves": total_leaves,
            "n_partitions": len(partitions),
            "leaf_partition_size": self.leaf_partition_size,
        }

    def from_partitioned_manifest(
        self, manifest: Dict[str, Any]
    ) -> MerkleTree:
        """从分区清单重建 MerkleTree.

        数学对应: 第3.5节存储 — 从 Parquet 列存重建 Merkle 树.
        通过分区清单中记录的 CID 从 CAS 存储中获取各层节点,
        重建完整的 MerkleTree 结构.

        注意: 此方法假设 CAS 存储中已包含清单引用的所有 CID.
        若某个 CID 缺失, 则该分区的节点将被标记为不可用.

        Args:
            manifest: 分区清单字典 (由 to_partitioned_manifest 生成).

        Returns:
            重建的 MerkleTree 实例.
        """
        partitions: List[Dict[str, Any]] = manifest.get("partitions", [])
        root_hash: str = manifest.get("root_hash", _sha256_hex("empty"))

        nodes: Dict[str, HierarchicalHash] = {}
        leaf_index: Dict[str, List[str]] = {}

        # ---- 重建 L0 叶子: 从 CAS 获取各分区的叶子 CID ----
        for part in partitions:
            l1_cid: str = part.get("cid", "")
            l1_data: Optional[bytes] = self.storage.get(l1_cid)

            l1_hash: str = part.get("hash_value", "")
            if l1_data is not None:
                # L1 节点的 canonical_json_snapshot 包含 L0 CID 列表
                l1_content: str = l1_data.decode('utf-8')
                l0_children: List[str] = l1_content.split("|")

                # 创建 L1 节点
                l1_node: HierarchicalHash = HierarchicalHash(
                    layer="internal_l1",
                    component_id=f"partition_{part.get('partition_idx', 0)}",
                    hash_value=l1_hash,
                    canonical_json_snapshot=l1_content,
                    children=l0_children,
                    metadata={
                        "partition_idx": part.get("partition_idx", 0),
                        "first_leaf_idx": part.get("first_leaf_idx", 0),
                        "last_leaf_idx": part.get("last_leaf_idx", 0),
                        "leaf_count": part.get("leaf_count", 0),
                        "level": "L1",
                    },
                )
                nodes[l1_hash] = l1_node

                # 尝试从 CAS 恢复 L0 叶子
                for l0_cid in l0_children:
                    l0_data: Optional[bytes] = self.storage.get(l0_cid)
                    if l0_data is not None:
                        l0_content_str: str = l0_data.decode('utf-8')
                        l0_hash: str = _sha256_hex(l0_content_str)
                        l0_hier_hash: str = l0_hash
                        # 将 L0 节点写入 nodes (简化: 层信息需要外部提供)
                        # 此处不做完整 L0 恢复, 仅记录 CID 引用
                        # 完整恢复需要额外的 leaf_index 信息
                        # 在清单模式下, leaf_index 由 manifest 中的分类信息提供

        # ---- 重建 leaf_index (基于已有信息) ----
        # 在清单重建时, leaf_index 可能不完整
        # 保留 manifest 中记录的分区信息作为 leaf_index 的 proxy
        leaf_index["_manifest"] = [
            p.get("cid", "") for p in partitions
        ]

        # ---- 重建 root ----
        # 若 nodes 中已有 L3 节点则使用, 否则创建基础 root 节点
        root_already_exists: bool = False
        for node in nodes.values():
            if node.metadata.get("level") == "L3":
                root_already_exists = True
                break

        if not root_already_exists:
            root_node: HierarchicalHash = HierarchicalHash(
                layer="internal_l3",
                component_id="root_cas",
                hash_value=root_hash,
                canonical_json_snapshot="",
                children=[],
                metadata={
                    "level": "L3",
                    "reconstructed": True,
                },
            )
            nodes[root_hash] = root_node

        return MerkleTree(
            root_hash=root_hash,
            nodes=nodes,
            leaf_index=leaf_index,
        )

    # ----------------------------------------------------------------
    # 增量 diff — 计算 / 应用
    # ----------------------------------------------------------------

    def compute_diff(
        self,
        old_tree: MerkleTree,
        new_tree: MerkleTree,
    ) -> Dict[str, Any]:
        """计算两个 MerkleTree 之间的增量 diff.

        数学对应: 第3.5节增量 — 仅处理 diff, 10^9 → 10^6 变更.
        diff 算法:
          1. 对比新旧树的 leaf_index, 找出新增/删除的叶子.
          2. 对比新旧树的 nodes, 找出修改的分区 (L1/L2/L3).
          3. 生成 diff = {added_cids, removed_cids, modified_partitions,
                          root_before, root_after}.

        复杂度: O(|old_nodes| + |new_nodes|).

        Args:
            old_tree: 旧 MerkleTree.
            new_tree: 新 MerkleTree.

        Returns:
            diff 字典:
              - added_cids: 新增的 CID 列表
              - removed_cids: 删除的 CID 列表
              - modified_partitions: 修改的分区索引列表
              - root_before: 旧根哈希
              - root_after: 新根哈希
              - n_added, n_removed, n_modified: 统计计数
        """
        # ---- 步骤1: 对比旧/新 leaf_index ----
        old_leaf_set: Set[str] = set()
        for layer, hash_list in old_tree.leaf_index.items():
            for hv in hash_list:
                cid: str = self._cid_from_hash_value(hv)
                old_leaf_set.add(cid)

        new_leaf_set: Set[str] = set()
        for layer, hash_list in new_tree.leaf_index.items():
            for hv in hash_list:
                cid: str = self._cid_from_hash_value(hv)
                new_leaf_set.add(cid)

        added_cids: List[str] = sorted(new_leaf_set - old_leaf_set)
        removed_cids: List[str] = sorted(old_leaf_set - new_leaf_set)

        # ---- 步骤2: 对比旧/新 nodes — 找出修改的分区 ----
        modified_partitions: List[int] = []

        # 收集旧/新树中 L1 分区的哈希
        old_l1_hashes: Dict[int, str] = {}
        new_l1_hashes: Dict[int, str] = {}

        for hv, node in old_tree.nodes.items():
            if node.metadata.get("level") == "L1":
                p_idx: int = node.metadata.get("partition_idx", -1)
                if p_idx >= 0:
                    old_l1_hashes[p_idx] = hv

        for hv, node in new_tree.nodes.items():
            if node.metadata.get("level") == "L1":
                p_idx: int = node.metadata.get("partition_idx", -1)
                if p_idx >= 0:
                    new_l1_hashes[p_idx] = hv

        # 检查修改的分区
        all_p_indices: Set[int] = set(old_l1_hashes.keys()) | set(new_l1_hashes.keys())
        for p_idx in all_p_indices:
            old_hv: Optional[str] = old_l1_hashes.get(p_idx)
            new_hv: Optional[str] = new_l1_hashes.get(p_idx)
            if old_hv != new_hv:
                modified_partitions.append(p_idx)

        modified_partitions.sort()

        # ---- 步骤3: 组装 diff ----
        return {
            "added_cids": added_cids,
            "removed_cids": removed_cids,
            "modified_partitions": modified_partitions,
            "root_before": old_tree.root_hash,
            "root_after": new_tree.root_hash,
            "n_added": len(added_cids),
            "n_removed": len(removed_cids),
            "n_modified": len(modified_partitions),
        }

    def apply_diff(
        self,
        tree: MerkleTree,
        diff: Dict[str, Any],
    ) -> MerkleTree:
        """应用 diff 更新树结构.

        数学对应: 第3.5节增量 — 仅重算 L0→L1→L2→root 路径.
        未变更分片完全不动.

        算法:
          1. 从 diff 获取 added_cids, removed_cids, modified_partitions.
          2. 若无变更, 直接返回原树.
          3. 对修改的分区, 从 CAS 重新读取 L0 叶子, 重算 L1 哈希.
          4. 重算受影响的 L2 模块聚合和 L3 根哈希.
          5. 更新 nodes 字典和 leaf_index, 返回新树.

        Args:
            tree: 当前 MerkleTree.
            diff: 由 compute_diff 生成的 diff 字典.

        Returns:
            应用 diff 后更新过的 MerkleTree.
        """
        added_cids: List[str] = diff.get("added_cids", [])
        removed_cids: List[str] = diff.get("removed_cids", [])
        modified_partitions: List[int] = diff.get("modified_partitions", [])

        # ---- 若无变更, 直接返回 ----
        if not added_cids and not removed_cids and not modified_partitions:
            return tree

        # ---- 深拷贝树以避免副作用 ----
        new_tree: MerkleTree = deepcopy(tree)

        # ---- 处理新增 CID ----
        for cid in added_cids:
            data: Optional[bytes] = self.storage.get(cid)
            if data is not None:
                content_str: str = data.decode('utf-8')
                hv: str = _sha256_hex(content_str)
                # 若该叶子不在 leaf_index 中, 添加
                # (实际 layer 信息需从上下文推断, 此处使用通用处理)
                if "_added" not in new_tree.leaf_index:
                    new_tree.leaf_index["_added"] = []
                if hv not in new_tree.leaf_index["_added"]:
                    new_tree.leaf_index["_added"].append(hv)

        # ---- 处理删除 CID ----
        for cid in removed_cids:
            hv_to_remove: Optional[str] = None
            for layer, hash_list in new_tree.leaf_index.items():
                for hv in hash_list:
                    if self._cid_from_hash_value(hv) == cid:
                        hv_to_remove = hv
                        hash_list.remove(hv)
                        break
                if hv_to_remove:
                    break
            if hv_to_remove and hv_to_remove in new_tree.nodes:
                del new_tree.nodes[hv_to_remove]

        # ---- 重算受影响的 L1 分区 ----
        for p_idx in modified_partitions:
            # 从 CAS 重新计算该分区的 L1 哈希
            start: int = p_idx * self.leaf_partition_size
            end: int = start + self.leaf_partition_size

            # 收集该分区范围内的 L0 CID
            partition_l0_cids: List[str] = []
            for layer, hash_list in new_tree.leaf_index.items():
                if layer.startswith("_"):
                    continue
                for hv in hash_list:
                    node: Optional[HierarchicalHash] = new_tree.nodes.get(hv)
                    if node is not None:
                        cid: str = self._cid(node.canonical_json_snapshot)
                        # 简化: 按顺序映射 (在实际使用中需要更精确的区间判定)
                        partition_l0_cids.append(cid)

            if partition_l0_cids:
                combined: str = "|".join(partition_l0_cids)
                l1_hash: str = _sha256_hex(combined)
                l1_cid: str = self._cid(combined)

                # 更新 nodes 中的 L1 节点
                # 清理旧 L1 节点
                old_l1_keys: List[str] = [
                    k for k, v in new_tree.nodes.items()
                    if v.metadata.get("partition_idx") == p_idx
                ]
                for k in old_l1_keys:
                    del new_tree.nodes[k]

                l1_node: HierarchicalHash = HierarchicalHash(
                    layer="internal_l1",
                    component_id=f"partition_{p_idx}",
                    hash_value=l1_hash,
                    canonical_json_snapshot=combined,
                    children=partition_l0_cids,
                    metadata={
                        "partition_idx": p_idx,
                        "first_leaf_idx": start,
                        "last_leaf_idx": end - 1,
                        "leaf_count": len(partition_l0_cids),
                        "level": "L1",
                    },
                )
                new_tree.nodes[l1_hash] = l1_node

                # 更新 CAS 存储
                self.storage.put(l1_cid, combined.encode('utf-8'))

        # ---- 重算 L2 模块聚合 (简化: 基于所有 L1 节点) ----
        l1_hashes: List[str] = [
            hv for hv, node in new_tree.nodes.items()
            if node.metadata.get("level") == "L1"
        ]
        l1_hashes_sorted: List[str] = sorted(l1_hashes)

        if l1_hashes_sorted:
            mod_combined: str = "|".join(l1_hashes_sorted)
            mod_hash: str = _sha256_hex(mod_combined)
            mod_cid: str = self._cid(mod_combined)

            # 清理旧 L2 节点
            old_l2_keys: List[str] = [
                k for k, v in new_tree.nodes.items()
                if v.metadata.get("level") == "L2"
            ]
            for k in old_l2_keys:
                del new_tree.nodes[k]

            mod_node: HierarchicalHash = HierarchicalHash(
                layer="internal_l2",
                component_id="module_aggregated",
                hash_value=mod_hash,
                canonical_json_snapshot=mod_combined,
                children=l1_hashes_sorted,
                metadata={
                    "level": "L2",
                    "l1_count": len(l1_hashes_sorted),
                },
            )
            new_tree.nodes[mod_hash] = mod_node
            self.storage.put(mod_cid, mod_combined.encode('utf-8'))

        # ---- 重算 L3 根哈希 ----
        l2_hashes: List[str] = [
            hv for hv, node in new_tree.nodes.items()
            if node.metadata.get("level") == "L2"
        ]
        if l2_hashes:
            root_combined: str = "|".join(sorted(l2_hashes))
        else:
            root_combined = _sha256_hex("empty")
        new_root_hash: str = _sha256_hex(root_combined)

        # 清理旧 L3 节点
        old_l3_keys: List[str] = [
            k for k, v in new_tree.nodes.items()
            if v.metadata.get("level") == "L3"
        ]
        for k in old_l3_keys:
            del new_tree.nodes[k]

        root_node: HierarchicalHash = HierarchicalHash(
            layer="internal_l3",
            component_id="root_cas",
            hash_value=new_root_hash,
            canonical_json_snapshot=root_combined,
            children=sorted(l2_hashes) if l2_hashes else [],
            metadata={
                "level": "L3",
                "updated_from_diff": True,
            },
        )
        new_tree.nodes[new_root_hash] = root_node
        new_tree.root_hash = new_root_hash

        return new_tree
