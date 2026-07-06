# 泛模因几何工具 — 模块四：凭证组装与哈希索引
# 数学对应：论文 3.5 节 — 分层哈希凭证 (Credential) 的组装与查询
# 论文位置：附录 D.4 凭证结构, 定理 5 自洽性验证
# 核心思想：将 MerkleTree + CompositeMemeState + 元数据组装为完整凭证,
#           提供基于哈希的快速模因查找, 子层提取, 以及跨凭证结构比对.

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pan_meme.core.types import (
    HierarchicalHash,
    MerkleTree,
    Credential,
    CompositeMemeState,
    MemeState,
)

# ============================================================
# CredentialAssembler — 凭证组装器
# ============================================================


class CredentialAssembler:
    """凭证组装器 — 将 MerkleTree + 模因状态 + 元数据组装为 Credential.

    数学对应: 附录 D.4 — Credential = (header, data_hash, merkle_tree, meme_state, metadata).
    凭证是管线对外输出的最终封装, 包含:
      - header:       版本号/时间戳/哈希算法/层信息
      - data_hash:    MerkleTree 的根哈希 (顶层摘要)
      - merkle_tree:  完整分层哈希树 (可追溯每一层)
      - meme_state:   复合模因状态 Q (管线最终产物)
      - metadata:     附加元数据 (输入大小/模因数/原始类型)

    凭证一旦生成即不可变: 所有字段冻结, 保证完整性验证的一致性.

    用法:
        cred = CredentialAssembler.assemble(tree, memes, meta)
    """

    # ----------------------------------------------------------------
    # 类级常量
    # ----------------------------------------------------------------

    # 当前凭证格式版本
    _VERSION: str = "2.0"

    # 默认哈希算法标识
    _HASH_ALGORITHM: str = "sha256"

    @staticmethod
    def assemble(
        tree: MerkleTree,
        memes: CompositeMemeState,
        meta: Dict[str, Any],
    ) -> Credential:
        """组装完整凭证.

        数学对应: Assemble(tree, Q, meta) → Credential.
        算法步骤:
          1. 构建 header: version="2.0", UTC 时间戳, hash_algorithm, merkle_layers
          2. data_hash = tree.root_hash (顶层根哈希)
          3. 封装所有输入为不可变 Credential 对象

        Args:
            tree:  完整的 MerkleTree (已通过 MerkleTreeBuilder.build 构建).
            memes: 复合模因状态 Q (模块3的输出).
            meta:  运行时元数据字典, 应包含:
                     - original_size_bytes: 原始输入大小
                     - meme_count: 模因数量
                     - original_type: 原始输入类型

        Returns:
            组装完毕的 Credential 对象.
        """
        # ---- 步骤1: 构建 header ----
        # 数学对应: header = (version, timestamp, hash_algorithm, merkle_layers)
        # timestamp 使用 UTC 时区以保证跨时区一致性
        header: Dict[str, Any] = {
            "version": CredentialAssembler._VERSION,
            "timestamp": (
                datetime.now(timezone.utc).isoformat()
            ),
            "hash_algorithm": CredentialAssembler._HASH_ALGORITHM,
            "merkle_layers": list(tree.leaf_index.keys()),
        }

        # ---- 步骤2: data_hash 即 MerkleTree 的 root_hash ----
        data_hash: str = tree.root_hash

        # ---- 步骤3: 组装凭证 ----
        return Credential(
            header=header,
            data_hash=data_hash,
            merkle_tree=tree,
            meme_state=memes,
            metadata=meta,
        )


# ============================================================
# HashBasedIndex — 基于哈希的快速索引
# ============================================================


class HashBasedIndex:
    """基于哈希的快速索引 — 通过哈希值在凭证中检索模因和子层.

    数学对应: 分层哈希树的查询操作.
    因每个 HierarchicalHash 的 hash_value 是全局唯一的
    (基于 SHA-256 的抗碰撞性), 哈希值可作为确定性键进行 O(1) 查找.

    提供三种查询模式:
      - find_meme_by_hash:      通过任意哈希值定位对应模因
      - find_subtree_for_layer: 提取某一层全部哈希节点
      - compare_structures:     比较两个凭证在某层的结构差异

    用法:
        meme = HashBasedIndex.find_meme_by_hash(cred, target_hash)
        nodes = HashBasedIndex.find_subtree_for_layer(cred, "meme")
        diff = HashBasedIndex.compare_structures(cred_a, cred_b, "meme")
    """

    @staticmethod
    def find_meme_by_hash(
        cred: Credential, target: str
    ) -> Optional[MemeState]:
        """通过哈希值在凭证中查找对应的模因.

        数学对应: hash_key → meme lookup.
        算法:
          1. 遍历 cred.merkle_tree.leaf_index["meme"] 中的每个哈希值
          2. 匹配 target 与叶子的 hash_value
          3. 若匹配, 从 component_id (格式 "meme_{idx}") 中提取 idx
          4. 返回 cred.meme_state.memes[idx]

        注意: 需要同时匹配叶子节点和内部节点的哈希,
        因为模因哈希可能因含 children 而出现在内部节点的引用链中.

        Args:
            cred:   凭证对象.
            target: 目标哈希值 (64 字符的十六进制串或前缀).

        Returns:
            匹配的 MemeState, 若未找到则返回 None.
        """
        tree: MerkleTree = cred.merkle_tree

        # ---- 遍历 meme 层的叶子节点 ----
        meme_hashes: List[str] = tree.leaf_index.get("meme", [])
        for hv in meme_hashes:
            node: Optional[HierarchicalHash] = tree.nodes.get(hv)
            if node is None:
                continue

            # 匹配: 完整哈希或前缀匹配
            if hv == target or hv.startswith(target):
                # 从 component_id 提取索引
                # component_id 格式: "meme_{idx}"
                cid: str = node.component_id
                if cid.startswith("meme_"):
                    try:
                        idx: int = int(cid.split("_", 1)[1])
                        memes: List[MemeState] = cred.meme_state.memes
                        if 0 <= idx < len(memes):
                            return memes[idx]
                    except (ValueError, IndexError):
                        continue

            # 同时检查子节点 (模因哈希可能出现在 children 链中)
            if target in node.children:
                cid = node.component_id
                if cid.startswith("meme_"):
                    try:
                        idx = int(cid.split("_", 1)[1])
                        memes = cred.meme_state.memes
                        if 0 <= idx < len(memes):
                            return memes[idx]
                    except (ValueError, IndexError):
                        continue

        return None

    @staticmethod
    def find_subtree_for_layer(
        cred: Credential, layer: str
    ) -> List[HierarchicalHash]:
        """提取某一层全部哈希节点.

        数学对应: subtree(layer) = { H ∈ MerkleTree | H.layer = layer }.
        从 leaf_index[layer] 出发, 收集该层所有叶子节点的
        完整 HierarchicalHash 对象.

        这允许下游代码查看某一层的哈希结构全貌,
        用于差异分析或逐层完整性报告.

        Args:
            cred:  凭证对象.
            layer: 目标层级名称.

        Returns:
            该层所有 HierarchicalHash 节点的列表.
            若 layer 不存在于 leaf_index 中, 返回空列表.
        """
        tree: MerkleTree = cred.merkle_tree
        layer_hashes: List[str] = tree.leaf_index.get(layer, [])
        result: List[HierarchicalHash] = []

        for hv in layer_hashes:
            node: Optional[HierarchicalHash] = tree.nodes.get(hv)
            if node is not None:
                result.append(node)

        return result

    @staticmethod
    def compare_structures(
        cred_a: Credential,
        cred_b: Credential,
        layer: str,
    ) -> Dict[str, bool]:
        """比较两个凭证在某层的结构差异.

        数学对应: Δ(layer, C₁, C₂) = {cid | H₁(cid) ≠ H₂(cid)}.
        对指定层, 分别收集两个凭证的 component_id → hash_value 映射,
        然后逐组件比对.

        返回值中, True 表示两个凭证在该组件上哈希一致,
        False 表示不一致, 或仅在一个凭证中存在.

        Args:
            cred_a: 第一个凭证.
            cred_b: 第二个凭证.
            layer:  要比较的层级名称.

        Returns:
            {component_id: bool} —— 每个组件的比对结果.
            True 表示一致, False 表示不一致.
        """
        nodes_a: List[HierarchicalHash] = HashBasedIndex.find_subtree_for_layer(
            cred_a, layer
        )
        nodes_b: List[HierarchicalHash] = HashBasedIndex.find_subtree_for_layer(
            cred_b, layer
        )

        # ---- 构建 component_id → hash_value 映射 ----
        map_a: Dict[str, str] = {
            n.component_id: n.hash_value for n in nodes_a
        }
        map_b: Dict[str, str] = {
            n.component_id: n.hash_value for n in nodes_b
        }

        # ---- 逐组件比对 ----
        all_ids: set = set(map_a.keys()) | set(map_b.keys())
        results: Dict[str, bool] = {}

        for cid in sorted(all_ids):
            ha: Optional[str] = map_a.get(cid)
            hb: Optional[str] = map_b.get(cid)
            results[cid] = (ha == hb)

        return results
