# 泛模因几何工具 — 模块四：稀疏 Merkle 树
# 数学对应：论文 3.5 节 — Merkle 树构造与分层验证
# 论文位置：附录 D.4 哈希树构造, 定理 5 自洽性验证
# 核心思想：以管线各层哈希节点为叶子, 构建三级模块聚合树,
#           支持全量/分层/单组件三种粒度的完整性验证,
#           增量更新时仅重算受影响路径 (Merkle 路径优化).

import hashlib
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from pan_meme.core.types import (
    HierarchicalHash,
    MerkleTree,
    MemeState,
    Credential,
)

# ============================================================
# -------------------- CAS Merkle 安全导入 (PB级新增) --------------------
# 从 pan_meme.module4_bind.cas_merkle 导入 CASMerkleTree,
# 若模块尚不存在 (cas_merkle.py 未创建) 则优雅降级.
# ============================================================
try:
    from pan_meme.module4_bind.cas_merkle import CASMerkleTree
    _CAS_AVAILABLE = True
except ImportError:
    CASMerkleTree = None  # type: ignore
    _CAS_AVAILABLE = False

# ============================================================
# 轻量级哈希函数 (与 hierarchy_hasher 保持一致)
# ============================================================


def sha256(data: str) -> str:
    """对字符串数据计算 SHA-256 十六进制摘要.

    Args:
        data: 待哈希的字符串.

    Returns:
        64 字符的十六进制小写哈希串.
    """
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


# ============================================================
# 模块层映射 — 定义管线各模块包含的子层
# ============================================================

# 模块 → 子层映射: 每个模块由若干管线层级组成
# 模块1 (输入适配层): token, hierarchy, relation, math_model
# 模块2 (几何化层):   geo_object (此处记作 sub_geo)
# 模块3 (模因化层):   sub_geo, meme, composite
_MODULE_LAYERS: Dict[str, List[str]] = {
    "module_1": ["token", "hierarchy", "relation", "math_model"],
    "module_2": ["sub_geo"],
    "module_3": ["sub_geo", "meme", "composite"],
}

# 层 → 模块反向映射 (用于从层名快速查找所属模块)
_LAYER_TO_MODULE: Dict[str, str] = {}
for _mod, _layers in _MODULE_LAYERS.items():
    for _layer in _layers:
        _LAYER_TO_MODULE[_layer] = _mod

# ============================================================
# MerkleTreeBuilder — 稀疏 Merkle 树构建器
# ============================================================


class MerkleTreeBuilder:
    """构建稀疏 Merkle 树: 叶子节点 → 模块聚合 → 根哈希.

    数学对应: 附录 D.4 — 分层哈希树的构造算法.
    构建流程:
      1. 索引所有叶子节点 (按 layer 分组 → leaf_index)
      2. 对模块1/2/3 分别聚合子层哈希得到模块哈希
      3. root_hash = sha256(mod1_hash | mod2_hash | mod3_hash)

    稀疏性: 不存储完整的二叉树, 仅记录叶子节点和经过
    _aggregate 生成的内部聚合节点. 这大幅减少了存储开销,
    同时保留了逐层验证的能力.

    用法:
        tree = MerkleTreeBuilder.build(all_hashes)
    """

    @staticmethod
    def build(all_hashes: List[HierarchicalHash], use_cas: bool = False) -> MerkleTree:
        """从所有分层哈希节点构建 Merkle 树.

        数学对应: BuildMerkle(H₁,...,Hₙ) → (root_hash, nodes, leaf_index).
        算法步骤:
          1. 索引所有叶子节点 → nodes 字典 + leaf_index (按 layer 分组)
          2. 对模块1 (token/hierarchy/relation/math_model) 聚合
          3. 对模块2 (geo_object) 聚合
          4. 对模块3 (sub_geo/meme/composite) 聚合
          5. root_hash = sha256(mod1 | mod2 | mod3)

        Args:
            all_hashes: 管线所有层级的所有组件哈希节点列表.
            use_cas:    是否使用 CAS Merkle 模式 (PB级新增).
                        为 True 时委托给 build_cas, 返回 CAS 树.

        Returns:
            完整的 MerkleTree, 包含 root_hash, nodes, leaf_index.

        Raises:
            ValueError: 当 all_hashes 为空时.
        """
        # ---- PB级新增: CAS Merkle 委托 ----
        if use_cas:
            tree, _manifest = MerkleTreeBuilder.build_cas(all_hashes)
            return tree

        if not all_hashes:
            raise ValueError(
                "无法构建 Merkle 树: all_hashes 为空. "
                "请确保管线已生成至少一个分层哈希节点."
            )

        # ---- 步骤1: 索引所有叶子节点 ----
        # 数学对应: leaf_index[layer] = [h₁, h₂, ...] 将同层哈希归类
        nodes: Dict[str, HierarchicalHash] = {}
        leaf_index: Dict[str, List[str]] = {}

        for h in all_hashes:
            nodes[h.hash_value] = h
            layer = h.layer
            if layer not in leaf_index:
                leaf_index[layer] = []
            leaf_index[layer].append(h.hash_value)

        # ---- 步骤2-4: 按模块聚合 ----
        # 数学对应: mod_k_hash = H( H(layer₁) | H(layer₂) | ... )
        module_hashes: Dict[str, str] = {}

        for mod_name, sub_layers in _MODULE_LAYERS.items():
            # 收集该模块的所有子层叶子哈希
            mod_children: List[str] = []
            for layer in sub_layers:
                layer_hashes = leaf_index.get(layer, [])
                # 若该层有多个叶子节点, 先聚合该层
                if len(layer_hashes) == 1:
                    mod_children.append(layer_hashes[0])
                elif len(layer_hashes) > 1:
                    # 多叶子层: 将同层所有哈希合并为该层的聚合哈希
                    layer_agg_hash: str = sha256("|".join(sorted(layer_hashes)))
                    mod_children.append(layer_agg_hash)
                else:
                    # 该层无叶子节点, 使用哨兵值
                    mod_children.append(sha256("empty"))

            # 调用 _aggregate 创建模块级内部节点
            mod_node: HierarchicalHash = MerkleTreeBuilder._aggregate(
                children=mod_children,
                nodes=nodes,
                label=f"{mod_name}_hash",
            )
            module_hashes[mod_name] = mod_node.hash_value

        # ---- 步骤5: 计算根哈希 ----
        # 数学对应: root_hash = sha256(mod1_hash | mod2_hash | mod3_hash)
        root_children: List[str] = [
            module_hashes.get("module_1", sha256("empty")),
            module_hashes.get("module_2", sha256("empty")),
            module_hashes.get("module_3", sha256("empty")),
        ]
        root_node: HierarchicalHash = MerkleTreeBuilder._aggregate(
            children=root_children,
            nodes=nodes,
            label="root",
        )

        # ---- 构建并返回 MerkleTree ----
        return MerkleTree(
            root_hash=root_node.hash_value,
            nodes=nodes,
            leaf_index=leaf_index,
        )

    @staticmethod
    def build_cas(
        all_hashes: List[HierarchicalHash],
        partition_size: int = 1000000,
        storage_backend: str = "memory",
    ):
        """PB级新增: 使用 CAS (Content-Addressed Storage) Merkle 模式构建树.

        数学对应: PB_ARCHITECTURE.md 第3.5节 — PB 级内容寻址存储 Merkle 树.
        调用 CASMerkleTree 将哈希节点分区存储, 返回 CAS 树和分区清单.

        CAS Merkle 树支持:
          - 海量叶子节点的分区存储 (partition_size 控制分区粒度)
          - 可插拔存储后端 (memory/s3/ipfs)
          - 基于内容寻址的快速检索和验证

        若 CASMerkleTree 导入失败 (cas_merkle 模块不存在),
        则优雅回退到原有的 build 方法.

        Args:
            all_hashes:      管线所有分层哈希节点列表.
            partition_size:  叶子分区大小 (每个分区最大叶子数).
            storage_backend: 存储后端标识 ("memory" | "s3" | "ipfs").

        Returns:
            Tuple[MerkleTree, Dict[str, Any]]:
              - MerkleTree: 构建完成的 Merkle 树 (兼容现有接口).
              - partitioned_manifest: 分区清单, 包含各分区摘要和存储路径.
        """
        # ---- 检查 CAS 可用性 ----
        if not _CAS_AVAILABLE:
            # 回退到原有 build 方法
            return MerkleTreeBuilder.build(all_hashes), {
                "fallback": True,
                "reason": "CASMerkleTree 模块不可用, 已回退到原始 Merkle 构建",
            }

        # ---- 调用 CASMerkleTree 构建 CAS 树 ----
        cas_tree = CASMerkleTree(
            partition_size=partition_size,
            storage_backend=storage_backend,
        )
        # CAS 构建: 插入哈希节点 → 分区聚合 → 生成根哈希和分区清单
        manifest = cas_tree.build_partitioned(all_hashes)

        # ---- 构造兼容的 MerkleTree 返回 ----
        # CAS 树的内部 MerkleTree 与现有接口完全兼容
        tree = cas_tree.to_merkle_tree()

        return tree, manifest

    @staticmethod
    def _aggregate(
        children: List[str],
        nodes: Dict[str, HierarchicalHash],
        label: str,
    ) -> HierarchicalHash:
        """聚合子节点哈希为内部节点.

        数学对应: H_internal = sha256( H(child₁) | "|" | H(child₂) | ... ).
        若 children 为空, 使用哨兵 "empty" 保证输入非空.

        此方法创建的内部节点被插入 nodes 字典,
        作为后续聚合 (模块/根) 的中间产物.

        Args:
            children: 子节点的 hash_value 列表.
            nodes:    节点字典 (将被原地修改, 插入新内部节点).
            label:    内部节点的组件标识标签.

        Returns:
            新建的内部 HierarchicalHash 节点.
        """
        # ---- 哨兵处理: 空子节点 → "empty" ----
        effective_children: List[str] = children if children else ["empty"]

        # ---- 合并子哈希 ----
        combined: str = "|".join(effective_children)
        h: str = sha256(combined)

        # ---- 创建内部节点 ----
        internal_node: HierarchicalHash = HierarchicalHash(
            layer="internal",
            component_id=label,
            hash_value=h,
            canonical_json_snapshot=combined,
            children=effective_children,
            metadata={"label": label, "child_count": len(effective_children)},
        )

        # ---- 插入 nodes 字典 ----
        nodes[h] = internal_node
        return internal_node


# ============================================================
# MerkleVerifier — Merkle 完整性验证器
# ============================================================


class MerkleVerifier:
    """Merkle 验证器 — 全量/分层/单组件三级粒度验证.

    数学对应: 定理 5 — 自洽性约束的哈希验证.
    提供三级粒度的验证能力:
      - verify_root:   从叶子重新计算 root_hash, 与凭证中的 data_hash 比对.
      - verify_layer:  对指定层的每个组件, 独立验证其哈希完整性.
      - locate_tamper: 遍历所有层所有节点, 返回全部哈希不匹配的 (layer, component_id).

    核心不变量:
      - 验证是纯函数的: 同一凭证多次验证结果必然一致.
      - locate_tamper 报告的是“最小嫌疑集”——仅包含被篡改的节点本身.

    用法:
        is_valid = MerkleVerifier.verify_root(credential)
        layer_ok = MerkleVerifier.verify_layer(credential, "meme")
        tampered = MerkleVerifier.locate_tamper(credential)
    """

    @staticmethod
    def verify_root(cred: Credential) -> bool:
        """全量验证: 从叶子重新计算 root_hash, 与凭证比对.

        数学对应: verify_root(C) ⟺ recompute_root(C) == C.data_hash.
        从凭证的 MerkleTree.leaf_index 出发, 重新构建模块聚合,
        计算根哈希, 与凭证记录的 data_hash 逐字节比对.

        验证流程:
          1. 从 leaf_index 收集各模块子层叶子哈希
          2. 重新计算 mod1_hash, mod2_hash, mod3_hash
          3. 重新计算 root_hash = sha256(mod1 | mod2 | mod3)
          4. 比对 cred.data_hash

        Args:
            cred: 待验证的凭证对象.

        Returns:
            True 当且仅当重新计算的根哈希与凭证中的 data_hash 完全一致.
        """
        tree: MerkleTree = cred.merkle_tree

        # ---- 步骤1-2: 重新计算模块哈希 ----
        module_hashes: Dict[str, str] = {}

        for mod_name, sub_layers in _MODULE_LAYERS.items():
            mod_children: List[str] = []
            for layer in sub_layers:
                layer_hashes: List[str] = tree.leaf_index.get(layer, [])
                if not layer_hashes:
                    mod_children.append(sha256("empty"))
                elif len(layer_hashes) == 1:
                    mod_children.append(layer_hashes[0])
                else:
                    mod_children.append(
                        sha256("|".join(sorted(layer_hashes)))
                    )
            # ---- 聚合该模块 ----
            module_hash: str = sha256("|".join(mod_children))
            module_hashes[mod_name] = module_hash

        # ---- 步骤3: 计算根哈希 ----
        root_children: List[str] = [
            module_hashes.get("module_1", sha256("empty")),
            module_hashes.get("module_2", sha256("empty")),
            module_hashes.get("module_3", sha256("empty")),
        ]
        recomputed_root: str = sha256("|".join(root_children))

        # ---- 步骤4: 比对 ----
        return recomputed_root == cred.data_hash

    @staticmethod
    def verify_layer(
        cred: Credential, layer: str
    ) -> Dict[str, bool]:
        """分层验证: 对指定层的每个组件独立验证哈希完整性.

        数学对应: ∀ component ∈ layer, verify(C, component).
        对指定层 leaf_index 中的每个哈希节点, 从其
        canonical_json_snapshot 重新计算 sha256, 与记录的
        hash_value 比对.

        注意: 对于含 children 的节点, 验证仅对比顶层 hash_value;
        子节点链的完整性由递归调用上层逻辑保证.

        Args:
            cred:  待验证的凭证.
            layer: 要验证的层级名称 (token|hierarchy|relation|...).

        Returns:
            {component_id: bool} —— 每个组件的验证结果.
            若该层不存在于 leaf_index 中, 返回空字典.
        """
        tree: MerkleTree = cred.merkle_tree
        layer_hashes: List[str] = tree.leaf_index.get(layer, [])

        if not layer_hashes:
            return {}

        results: Dict[str, bool] = {}

        for hv in layer_hashes:
            node: Optional[HierarchicalHash] = tree.nodes.get(hv)
            if node is None:
                results[f"missing_{hv[:8]}"] = False
                continue

            # ---- 重新计算哈希 ----
            # 数学对应: H_recomputed = sha256(canonical_json_snapshot)
            # 对于无子节点的叶子, 直接哈希快照;
            # 对于有子节点的节点, 合并子哈希后再哈希.
            if node.children:
                recomputed: str = sha256("|".join(node.children))
            else:
                recomputed = sha256(node.canonical_json_snapshot)

            results[node.component_id] = (recomputed == node.hash_value)

        return results

    @staticmethod
    def locate_tamper(cred: Credential) -> List[Tuple[str, str]]:
        """定位篡改: 遍历所有层所有节点, 返回全部哈希不匹配的组件.

        数学对应: locate_tamper(C) = {(layer, cid) | verify(C, cid) = False}.
        对凭证中所有存储的叶子节点进行哈希重算,
        收集所有 hash_value 与 recomputed 不一致的 (layer, component_id) 对.

        算法复杂度: O(|nodes|) — 每个节点仅访问一次.
        返回结果为“最小嫌疑集”——仅标记直接不一致的节点,
        上游节点若也受影响将由 verify_root 捕获.

        Args:
            cred: 待检测的凭证.

        Returns:
            [(layer, component_id), ...] —— 所有被篡改组件的列表.
            若凭证完整, 返回空列表.
        """
        tree: MerkleTree = cred.merkle_tree
        tampered: List[Tuple[str, str]] = []

        # ---- 遍历所有层 ----
        for layer, hash_list in tree.leaf_index.items():
            for hv in hash_list:
                node: Optional[HierarchicalHash] = tree.nodes.get(hv)
                if node is None:
                    tampered.append((layer, hv[:8]))
                    continue

                # ---- 重算哈希 ----
                if node.children:
                    recomputed: str = sha256("|".join(node.children))
                else:
                    recomputed = sha256(node.canonical_json_snapshot)

                if recomputed != node.hash_value:
                    tampered.append((layer, node.component_id))

        return tampered


# ============================================================
# MerkleUpdater — Merkle 增量更新器
# ============================================================


class MerkleUpdater:
    """Merkle 增量更新器 — 仅重算受影响路径.

    数学对应: Merkle 路径优化 — 增量更新时仅重算从修改叶子
    到根的唯一路径上的节点哈希, 而非整个树重建.
    这利用了哈希树的局部性: 未受影响子树无需重算.

    支持三种模因操作:
      - update_meme:  替换单个模因 (就地更新)
      - split_meme:   将一个模因分裂为多个子模因
      - merge_memes:  将多个模因合并为一个

    内部维护对树的引用, 每次更新后返回新的 MerkleTree.

    用法:
        updater = MerkleUpdater(tree, hasher)
        new_tree = updater.update_meme(idx=0, new_meme=meme_obj)
    """

    # ----------------------------------------------------------------
    # 构造与初始化
    # ----------------------------------------------------------------

    def __init__(self, tree: MerkleTree, hasher: Any) -> None:
        """初始化 Merkle 增量更新器.

        Args:
            tree:   待修改的 Merkle 树 (将被深拷贝以避免副作用).
            hasher: HierarchyHasher 实例, 用于重新计算模因哈希.
        """
        # ---- 深拷贝树以避免副作用 ----
        # 数学对应: 纯函数式更新 — 不修改输入, 返回新树
        self._tree: MerkleTree = deepcopy(tree)
        """内部维护的 MerkleTree 副本."""

        self._hasher: Any = hasher
        """HierarchyHasher 实例, 用于重新哈希."""

    # ----------------------------------------------------------------
    # 公开更新接口
    # ----------------------------------------------------------------

    def update_meme(self, idx: int, new_meme: MemeState) -> MerkleTree:
        """替换单个模因: 重算 meme 叶 → 替换 leaf_index → 重算上游.

        数学对应: 单叶替换 — 仅修改 meme_{idx} 叶节点,
        然后沿 Merkle 路径向上重算 module_3 聚合和 root_hash.

        算法步骤:
          1. 对 new_meme 调用 hasher.hash_meme(idx) 生成新叶哈希
          2. 在 leaf_index["meme"] 中找到旧叶并替换
          3. 将新旧节点加入/移除 nodes 字典
          4. 调用 _recompute_module_and_root("module_3", [...])

        Args:
            idx:     要替换的模因索引.
            new_meme: 新的 MemeState 对象.

        Returns:
            更新后的 MerkleTree.

        Raises:
            ValueError: 当指定索引的模因不存在时.
        """
        # ---- 步骤1: 重新哈希 ----
        new_leaf: HierarchicalHash = self._hasher.hash_meme(new_meme, idx)

        # ---- 步骤2: 找到并替换叶子 ----
        old_hv: str = self._find_leaf("meme", idx)
        meme_hashes: List[str] = self._tree.leaf_index.get("meme", [])

        if old_hv not in meme_hashes:
            raise ValueError(
                f"无法更新模因: meme_{idx} 不存在于 leaf_index['meme'] 中."
            )

        # 替换叶子列表中的哈希值
        replace_idx: int = meme_hashes.index(old_hv)
        meme_hashes[replace_idx] = new_leaf.hash_value

        # 更新 nodes 字典
        if old_hv in self._tree.nodes:
            del self._tree.nodes[old_hv]
        self._tree.nodes[new_leaf.hash_value] = new_leaf

        # ---- 步骤3: 重算模块和根 ----
        self._recompute_module_and_root(
            "module_3", ["sub_geo", "meme", "composite"]
        )

        return self._tree

    def split_meme(
        self, parent_idx: int, children_memes: List[MemeState]
    ) -> MerkleTree:
        """分裂模因: 删除父模因 → 插入多个子模因 → 重算上游.

        数学对应: 1→k 分裂 — 叶节点集变化, 仅重算受影响的 meme 层
        和上游 module_3 聚合.

        算法步骤:
          1. 删除 parent_idx 对应的叶节点
          2. 为每个子模因生成新的 HierarchicalHash 叶节点
          3. 更新 leaf_index["meme"] 和 nodes 字典
          4. 调用 _recompute_module_and_root

        Args:
            parent_idx:     要分裂的父模因索引.
            children_memes: 子模因列表 (至少2个).

        Returns:
            更新后的 MerkleTree.
        """
        if len(children_memes) < 2:
            raise ValueError(
                "分裂操作要求至少 2 个子模因, 当前仅 "
                f"{len(children_memes)} 个."
            )

        # ---- 步骤1: 删除父模因叶节点 ----
        self._remove_leaf("meme", parent_idx)

        # ---- 步骤2: 插入子模因叶节点 ----
        # 子模因的索引接续在处理时确定
        # 实际场景中由调用方管理索引分配
        for i, child_meme in enumerate(children_memes):
            child_idx: int = parent_idx + i  # 简化索引策略
            child_leaf: HierarchicalHash = self._hasher.hash_meme(
                child_meme, child_idx
            )
            self._tree.nodes[child_leaf.hash_value] = child_leaf
            if "meme" not in self._tree.leaf_index:
                self._tree.leaf_index["meme"] = []
            self._tree.leaf_index["meme"].append(child_leaf.hash_value)

        # ---- 步骤3: 重算模块和根 ----
        self._recompute_module_and_root(
            "module_3", ["sub_geo", "meme", "composite"]
        )

        return self._tree

    def merge_memes(
        self, indices: List[int], merged_meme: MemeState
    ) -> MerkleTree:
        """合并模因: 删除多个模因 → 插入合并后的模因 → 重算上游.

        数学对应: k→1 合并 — 多个叶节点缩减为一个, 仅重算受影响的
        meme 层和上游 module_3 聚合.

        算法步骤:
          1. 逐个删除 indices 中的每个模因叶节点
          2. 为合并后的模因生成新 HierarchicalHash 叶节点
          3. 更新 leaf_index["meme"] 和 nodes 字典
          4. 调用 _recompute_module_and_root

        Args:
            indices:     要合并的模因索引列表 (至少2个).
            merged_meme: 合并后的 MemeState.

        Returns:
            更新后的 MerkleTree.
        """
        if len(indices) < 2:
            raise ValueError(
                f"合并操作要求至少 2 个索引, 当前仅 {len(indices)} 个."
            )

        # ---- 步骤1: 删除所有旧模因叶节点 ----
        for idx in indices:
            self._remove_leaf("meme", idx)

        # ---- 步骤2: 插入合并后的模因 ----
        merged_idx: int = min(indices)  # 使用最小索引作为合并后的索引
        merged_leaf: HierarchicalHash = self._hasher.hash_meme(
            merged_meme, merged_idx
        )
        self._tree.nodes[merged_leaf.hash_value] = merged_leaf
        if "meme" not in self._tree.leaf_index:
            self._tree.leaf_index["meme"] = []
        self._tree.leaf_index["meme"].append(merged_leaf.hash_value)

        # ---- 步骤3: 重算模块和根 ----
        self._recompute_module_and_root(
            "module_3", ["sub_geo", "meme", "composite"]
        )

        return self._tree

    # ----------------------------------------------------------------
    # 内部辅助方法
    # ----------------------------------------------------------------

    def _find_leaf(self, layer: str, idx: int) -> str:
        """在 leaf_index 中查找指定层级和索引的叶子哈希值.

        通过遍历 layer 的叶子节点列表, 匹配其 component_id
        带有 "_{idx}" 后缀的节点.

        Args:
            layer: 层级名称.
            idx:   组件索引.

        Returns:
            匹配叶子的 hash_value.

        Raises:
            ValueError: 若未找到匹配的叶子.
        """
        layer_hashes: List[str] = self._tree.leaf_index.get(layer, [])
        target_cid: str = f"meme_{idx}"  # 匹配 component_id 模式

        for hv in layer_hashes:
            node: Optional[HierarchicalHash] = self._tree.nodes.get(hv)
            if node is not None and node.component_id == target_cid:
                return hv

        raise ValueError(
            f"未找到叶子: layer='{layer}', idx={idx}. "
            f"当前 leaf_index['{layer}'] 中共 {len(layer_hashes)} 个叶子."
        )

    def _remove_leaf(self, layer: str, idx: int) -> None:
        """从 leaf_index 和 nodes 中移除指定叶子.

        同时清理 nodes 字典中的对应条目,
        保证 leaf_index 和 nodes 之间的一致性.

        Args:
            layer: 层级名称.
            idx:   组件索引.

        Raises:
            ValueError: 若未找到匹配的叶子.
        """
        hv: str = self._find_leaf(layer, idx)

        # ---- 从 leaf_index 中移除 ----
        if layer in self._tree.leaf_index:
            self._tree.leaf_index[layer] = [
                h for h in self._tree.leaf_index[layer] if h != hv
            ]

        # ---- 从 nodes 中移除 ----
        if hv in self._tree.nodes:
            del self._tree.nodes[hv]

    def _recompute_module_and_root(
        self, module_name: str, sub_layers: List[str]
    ) -> None:
        """重算指定模块的聚合哈希和根哈希.

        数学对应: 增量 Merkle 路径更新.
        当模块的子层叶子发生变化后, 此方法:
          1. 收集每个子层的所有当前叶子哈希
          2. 聚合子层 → 模块哈希
          3. 更新/创建模块级内部节点
          4. 删除旧的模块节点 (通过遍历 nodes 清理)
          5. 收集三个模块哈希 → 重算 root_hash

        这是增量更新的核心——仅沿受影响路径向上传播变更.

        Args:
            module_name: 模块名 (module_1|module_2|module_3).
            sub_layers:  该模块的子层名称列表.
        """
        # ---- 步骤1: 收集子层叶子哈希 ----
        mod_children: List[str] = []
        for layer in sub_layers:
            layer_hashes: List[str] = self._tree.leaf_index.get(layer, [])
            if not layer_hashes:
                mod_children.append(sha256("empty"))
            elif len(layer_hashes) == 1:
                mod_children.append(layer_hashes[0])
            else:
                mod_children.append(
                    sha256("|".join(sorted(layer_hashes)))
                )

        # ---- 步骤2: 聚合为模块哈希 ----
        combined: str = "|".join(mod_children)
        mod_hash: str = sha256(combined)

        # ---- 步骤3: 创建/更新模块级内部节点 ----
        mod_node: HierarchicalHash = HierarchicalHash(
            layer="internal",
            component_id=f"{module_name}_hash",
            hash_value=mod_hash,
            canonical_json_snapshot=combined,
            children=mod_children,
            metadata={
                "module": module_name,
                "child_count": len(mod_children),
                "sub_layers": sub_layers,
            },
        )

        # ---- 步骤4: 清理旧模块节点 ----
        # 删除 component_id 匹配的旧模块节点
        old_mod_keys: List[str] = [
            k for k, v in self._tree.nodes.items()
            if v.component_id == f"{module_name}_hash"
        ]
        for k in old_mod_keys:
            del self._tree.nodes[k]

        # 插入新模块节点
        self._tree.nodes[mod_hash] = mod_node

        # ---- 步骤5: 重算 root_hash ----
        root_children: List[str] = []
        for mod_name in ["module_1", "module_2", "module_3"]:
            if mod_name == module_name:
                # 使用刚计算的模块哈希
                root_children.append(mod_hash)
            else:
                # 查找已有的模块哈希 (从 nodes 中搜索)
                found: Optional[str] = None
                for k, v in self._tree.nodes.items():
                    if v.component_id == f"{mod_name}_hash":
                        found = k
                        break
                if found:
                    root_children.append(found)
                else:
                    root_children.append(sha256("empty"))

        # 新根哈希
        root_combined: str = "|".join(root_children)
        new_root: str = sha256(root_combined)

        # 清理旧根节点
        old_root_keys: List[str] = [
            k for k, v in self._tree.nodes.items()
            if v.component_id == "root"
        ]
        for k in old_root_keys:
            del self._tree.nodes[k]

        # 插入新根节点
        root_node: HierarchicalHash = HierarchicalHash(
            layer="internal",
            component_id="root",
            hash_value=new_root,
            canonical_json_snapshot=root_combined,
            children=root_children,
            metadata={
                "label": "root",
                "child_count": len(root_children),
            },
        )
        self._tree.nodes[new_root] = root_node
        self._tree.root_hash = new_root


# ============================================================
# MerkleDiffManager — PB级新增: Merkle 差异管理与回滚
# ============================================================


class MerkleDiffManager:
    """PB级新增: Merkle 树差异管理器, 支持增量 diff 计算与 epoch 回滚.

    数学对应: PB_ARCHITECTURE.md — 基于 Merkle DAG 的增量验证.
    CAS Merkle 树允许按分区粒度的差异计算, 仅记录变更的分区
    而非整个树, 支持时间线回滚.

    核心功能:
      - compute_and_apply_diff: 计算 old→new 的差异并存储到历史记录
      - get_diff_history:       返回所有历史 diff 记录
      - rollback_to_epoch:      从当前状态逆序应用 diff 回滚到指定 epoch

    用法:
        diff_mgr = MerkleDiffManager(cas_tree)
        report = diff_mgr.compute_and_apply_diff(old_tree, new_tree, epoch=1)
        old_tree = diff_mgr.rollback_to_epoch(0)
    """

    def __init__(self, cas_tree: Any) -> None:
        """初始化 Merkle 差异管理器.

        Args:
            cas_tree: CASMerkleTree 实例, 用于分区粒度的 diff 计算.
        """
        # ---- 持有 CASMerkleTree 实例 ----
        self._cas_tree: Any = cas_tree
        """CASMerkleTree 实例, 提供分区粒度操作."""

        # ---- 差异历史记录 ----
        # 每条记录: {"epoch": int, "timestamp": str, "diff": Dict, "root_before": str, "root_after": str}
        self._diff_history: List[Dict[str, Any]] = []
        """按 epoch 顺序存储的 diff 记录列表."""

        # ---- 快照归档 ----
        # 存储每个 epoch 的 MerkleTree 快照, 用于快速回滚
        self._snapshots: Dict[int, MerkleTree] = {}
        """epoch → MerkleTree 快照映射."""

    def compute_and_apply_diff(
        self,
        old_tree: MerkleTree,
        new_tree: MerkleTree,
        epoch: int,
    ) -> Dict[str, Any]:
        """计算 old_tree → new_tree 的差异并应用到历史记录.

        数学对应: diff = Δ(old_root, new_root), 逐分区比较叶子节点.
        CAS Merkle 模式下, 差异计算聚焦于分区级别的变更:
          - 新增的分区 (new_tree 有, old_tree 无)
          - 删除的分区 (old_tree 有, new_tree 无)
          - 修改的分区 (root_hash 变化)

        Args:
            old_tree: 旧 Merkle 树.
            new_tree: 新 Merkle 树.
            epoch:    当前 epoch 标识.

        Returns:
            Dict[str, Any] — diff 报告:
              {"epoch": int, "root_before": str, "root_after": str,
               "added_partitions": [...], "removed_partitions": [...],
               "modified_partitions": [...], "leaf_changes": int}
        """
        import datetime

        # ---- 基础比较 ----
        diff_report: Dict[str, Any] = {
            "epoch": epoch,
            "timestamp": datetime.datetime.now().isoformat(),
            "root_before": old_tree.root_hash,
            "root_after": new_tree.root_hash,
            "added_partitions": [],
            "removed_partitions": [],
            "modified_partitions": [],
            "leaf_changes": 0,
        }

        # ---- 叶子级别比较 ----
        old_leaves: Dict[str, str] = {}
        for layer, hvs in old_tree.leaf_index.items():
            for hv in hvs:
                node = old_tree.nodes.get(hv)
                if node:
                    old_leaves[f"{layer}/{node.component_id}"] = hv

        new_leaves: Dict[str, str] = {}
        for layer, hvs in new_tree.leaf_index.items():
            for hv in hvs:
                node = new_tree.nodes.get(hv)
                if node:
                    new_leaves[f"{layer}/{node.component_id}"] = hv

        # 新增的叶子
        added_keys = set(new_leaves.keys()) - set(old_leaves.keys())
        # 删除的叶子
        removed_keys = set(old_leaves.keys()) - set(new_leaves.keys())
        # 修改的叶子 (键相同但哈希不同)
        common_keys = set(old_leaves.keys()) & set(new_leaves.keys())
        modified_keys = {
            k for k in common_keys if old_leaves[k] != new_leaves[k]
        }

        diff_report["added_partitions"] = sorted(added_keys)
        diff_report["removed_partitions"] = sorted(removed_keys)
        diff_report["modified_partitions"] = sorted(modified_keys)
        diff_report["leaf_changes"] = (
            len(added_keys) + len(removed_keys) + len(modified_keys)
        )

        # ---- 存储到历史记录 ----
        self._diff_history.append(diff_report)
        self._snapshots[epoch] = deepcopy(new_tree)

        return diff_report

    def get_diff_history(self) -> List[Dict[str, Any]]:
        """返回所有历史 diff 记录列表.

        Returns:
            List[Dict[str, Any]] — 按 epoch 升序排列的 diff 记录.
            每条记录包含完整的 diff 报告 (参见 compute_and_apply_diff 返回值).
        """
        return list(self._diff_history)

    def rollback_to_epoch(self, epoch: int) -> Optional[MerkleTree]:
        """回滚到指定 epoch 的状态.

        数学对应: 逆序应用 diff — 从当前状态回溯到目标 epoch.
        通过检查快照归档实现快速回滚:
          - 若目标 epoch 存在快照, 直接返回
          - 否则返回 None (无法回滚)

        注意: 当前实现依赖快照归档, 不支持跨多 epoch 的增量回滚.
        未来可扩展为通过逆序应用 diff 记录实现精确回滚.

        Args:
            epoch: 目标 epoch 编号.

        Returns:
            目标 epoch 的 MerkleTree 快照, 若不存在则返回 None.
        """
        if epoch in self._snapshots:
            return deepcopy(self._snapshots[epoch])

        # ---- 目标 epoch 无快照 ----
        return None
