# 泛模因几何工具 — 模块四：分层哈希绑定主入口
# 数学对应：论文 3.5 节 — CredentialBinder, 管线哈希绑定层
# 论文位置：定理 5 (自洽性约束哈希验证), 附录 D.4 完整凭证流
# 核心思想：作为模块四的编排器, 聚合 HierarchyHasher,
#           MerkleTreeBuilder, CredentialAssembler, MerkleVerifier,
#           提供统一的 bind / verify / locate_tamper 接口.

from typing import Any, Dict, List, Optional, Tuple

from pan_meme.core.types import (
    HierarchicalHash,
    MerkleTree,
    Credential,
    CompositeMemeState,
    MemeState,
    PipelineData,
)

from pan_meme.module4_bind.hierarchy_hasher import HierarchyHasher
from pan_meme.module4_bind.merkle_tree import (
    MerkleTreeBuilder,
    MerkleVerifier,
    MerkleUpdater,
)
from pan_meme.module4_bind.credential import (
    CredentialAssembler,
    HashBasedIndex,
)


# ============================================================
# CredentialBinder — 分层哈希绑定主入口
# ============================================================


class CredentialBinder:
    """分层哈希绑定主入口 — 模块四的编排器.

    数学对应: 论文 3.5 节, 定理 5 的哈希验证层主入口.
    CredentialBinder 将模块四的所有组件整合为一个统一的 API:
      - bind:            对整个 PipelineData 执行哈希绑定, 生成凭证
      - verify_full:     全量根哈希验证
      - verify_layer:    分层验证 (指定层)
      - verify_component: 单组件验证 (指定层+组件ID)
      - locate_tamper:    定位篡改的组件

    架构:
      CredentialBinder
        ├── HierarchyHasher     (哈希引擎 — 生成 HierarchicalHash)
        ├── MerkleTreeBuilder   (树构建 — 叶子→模块→根)
        ├── CredentialAssembler (凭证组装 — tree + memes + meta → Credential)
        ├── MerkleVerifier      (验证器 — 三级粒度验证)
        └── MerkleUpdater       (增量更新 — 仅重算受影响路径)

    用法:
        binder = CredentialBinder("sha256")
        data = binder.bind(pipeline_data)      # 哈希绑定
        ok = binder.verify_full(data.credential)  # 全量验证
        layer_ok = binder.verify_layer(data.credential, "meme")  # 分层验证
    """

    # ----------------------------------------------------------------
    # 构造与初始化
    # ----------------------------------------------------------------

    def __init__(self, hash_algo: str = "sha256") -> None:
        """初始化凭证绑定器.

        Args:
            hash_algo: 哈希算法标识, 当前仅支持 "sha256".
        """
        self.hasher: HierarchyHasher = HierarchyHasher(hash_algo)
        """分层哈希引擎实例."""

    # ----------------------------------------------------------------
    # 绑定操作 — 管线数据 → 凭证
    # ----------------------------------------------------------------

    def bind(self, data: PipelineData) -> PipelineData:
        """对整个管线数据执行分层哈希绑定, 生成凭证.

        数学对应: Bind(P) = assemble(build(hash(P))).
        算法步骤:
          1. 收集 data.all_hash_nodes (管线前序模块累积的哈希节点)
          2. 若 all_hash_nodes 为空, 自动对管线数据执行全量哈希
          3. MerkleTreeBuilder.build(all_hash_nodes) → MerkleTree
          4. CredentialAssembler.assemble(tree, memes, meta) → Credential
          5. 将凭证挂载回 PipelineData 并返回

        Args:
            data: 管线数据上下文 (包含 all_hash_nodes, meme_state, meta 等).

        Returns:
            更新后的 PipelineData, 其 credential 和 merkle_tree 字段已填充.
        """
        # ---- 步骤1: 收集哈希节点 ----
        all_nodes: List[HierarchicalHash] = data.all_hash_nodes

        # ---- 步骤2: 若 all_hash_nodes 为空, 执行自动哈希 ----
        # 数学对应: 自动哈希回退 — 当上层模块未提供预计算哈希时,
        #          binder 会对当前管线数据中的各个产物进行哈希.
        if not all_nodes:
            all_nodes = self._auto_hash(data)

        # ---- 步骤3: 构建 Merkle 树 ----
        tree: MerkleTree = MerkleTreeBuilder.build(all_nodes)
        data.merkle_tree = tree

        # ---- 步骤4: 组装凭证 ----
        memes: CompositeMemeState
        if data.meme_state is not None:
            memes = data.meme_state
        else:
            # 若尚未生成模因状态, 创建空占位
            import numpy as np
            memes = CompositeMemeState(
                memes=[],
                Theta=[],
                C=np.array([[]], dtype=float),
            )

        meta: Dict[str, Any] = data.meta.copy()
        # 补充元数据
        meta.setdefault("meme_count", len(memes.memes))
        meta.setdefault(
            "original_type",
            type(data.input).__name__ if data.input is not None else "unknown",
        )

        cred: Credential = CredentialAssembler.assemble(tree, memes, meta)
        data.credential = cred

        # ---- 步骤5: 返回更新后的管线数据 ----
        return data

    # ----------------------------------------------------------------
    # 验证操作 — 三级粒度
    # ----------------------------------------------------------------

    def verify_full(self, cred: Credential) -> bool:
        """全量验证: 验证凭证根哈希的完整性.

        数学对应: verify_root(C) — 定理 5.
        从凭证的 MerkleTree 叶子节点出发, 重新计算 root_hash,
        与 cred.data_hash 逐字节比对.

        Args:
            cred: 待验证的凭证.

        Returns:
            True 当且仅当重新计算的根哈希与存储值完全一致.
        """
        return MerkleVerifier.verify_root(cred)

    def verify_layer(
        self, cred: Credential, layer: str
    ) -> Dict[str, bool]:
        """分层验证: 验证指定层的所有组件.

        数学对应: verify_layer(C, layer) — 逐层完整性检查.
        对该层 leaf_index 中的每个组件, 重新计算其 sha256
        并与存储的 hash_value 比对.

        Args:
            cred:  待验证的凭证.
            layer: 层级名称 (token|hierarchy|relation|math_model|...).

        Returns:
            {component_id: bool} —— 每个组件的验证结果.
            若该层无组件, 返回空字典.
        """
        return MerkleVerifier.verify_layer(cred, layer)

    def verify_component(
        self, cred: Credential, layer: str, cid: str
    ) -> bool:
        """单组件验证: 验证指定层中特定组件的哈希.

        数学对应: verify_component(C, layer, cid) — 单点验证.
        等价于 verify_layer(C, layer).get(cid, False).

        Args:
            cred:  待验证的凭证.
            layer: 层级名称.
            cid:   组件唯一标识符.

        Returns:
            True 当且仅当该组件的哈希验证通过.
            若组件不存在, 返回 False.
        """
        return MerkleVerifier.verify_layer(cred, layer).get(cid, False)

    def locate_tamper(
        self, cred: Credential
    ) -> List[Tuple[str, str]]:
        """定位篡改: 找出所有哈希不匹配的组件.

        数学对应: locate_tamper(C) — 最小嫌疑集.
        遍历凭证的所有层所有组件, 返回全部
        (layer, component_id) 对, 其存储哈希与重算哈希不一致.

        Args:
            cred: 待检测的凭证.

        Returns:
            [(layer, component_id), ...] —— 被篡改组件列表.
            若凭证完整, 返回空列表.
        """
        return MerkleVerifier.locate_tamper(cred)

    # ----------------------------------------------------------------
    # 增量更新 — 凭证维护接口
    # ----------------------------------------------------------------

    def update_meme(
        self, cred: Credential, idx: int, new_meme: MemeState
    ) -> Credential:
        """增量更新单个模因并重新生成凭证.

        数学对应: 增量 Merkle 更新 — 仅重算受影响路径.
        适用于模因演化场景: 单个模因参数发生变化时,
        无需重新构建整个 MerkleTree.
        更新后自动重新组装凭证以保证 data_hash 一致性.

        Args:
            cred:     当前凭证.
            idx:      要更新的模因索引.
            new_meme: 新的 MemeState 对象.

        Returns:
            更新后的新凭证 (原凭证不变).
        """
        updater: MerkleUpdater = MerkleUpdater(cred.merkle_tree, self.hasher)
        new_tree: MerkleTree = updater.update_meme(idx, new_meme)

        # ---- 更新 meme_state 中的模因 ----
        new_memes: CompositeMemeState = cred.meme_state
        memes_list: List[MemeState] = list(new_memes.memes)
        if 0 <= idx < len(memes_list):
            memes_list[idx] = new_meme
        from dataclasses import replace
        try:
            new_memes = replace(new_memes, memes=memes_list)
        except Exception:
            # 若 replace 不可用, 手动构造
            import numpy as np
            new_memes = CompositeMemeState(
                memes=memes_list,
                Theta=new_memes.Theta,
                C=new_memes.C,
            )

        # ---- 重新组装凭证 ----
        return CredentialAssembler.assemble(
            new_tree,
            new_memes,
            cred.metadata,
        )

    # ----------------------------------------------------------------
    # 查询接口 — 委托 HashBasedIndex
    # ----------------------------------------------------------------

    def find_meme_by_hash(
        self, cred: Credential, target: str
    ) -> Optional[MemeState]:
        """通过哈希值查找模因.

        委托给 HashBasedIndex.find_meme_by_hash.

        Args:
            cred:   凭证对象.
            target: 目标哈希值 (完整或前缀).

        Returns:
            匹配的 MemeState, 若未找到则返回 None.
        """
        return HashBasedIndex.find_meme_by_hash(cred, target)

    def get_layer_nodes(
        self, cred: Credential, layer: str
    ) -> List[HierarchicalHash]:
        """获取指定层的全部哈希节点.

        委托给 HashBasedIndex.find_subtree_for_layer.

        Args:
            cred:  凭证对象.
            layer: 目标层级名称.

        Returns:
            该层所有 HierarchicalHash 节点列表.
        """
        return HashBasedIndex.find_subtree_for_layer(cred, layer)

    def compare_credentials(
        self, cred_a: Credential, cred_b: Credential, layer: str
    ) -> Dict[str, bool]:
        """比较两个凭证在某层的结构差异.

        委托给 HashBasedIndex.compare_structures.

        Args:
            cred_a: 第一个凭证.
            cred_b: 第二个凭证.
            layer:  要比较的层级名称.

        Returns:
            {component_id: bool} —— 每个组件的比对结果.
        """
        return HashBasedIndex.compare_structures(cred_a, cred_b, layer)

    # ----------------------------------------------------------------
    # 内部方法 — 自动哈希回退
    # ----------------------------------------------------------------

    def _auto_hash(self, data: PipelineData) -> List[HierarchicalHash]:
        """当管线未预计算哈希时, 对当前数据执行全量自动哈希.

        数学对应: hash(P) = { H(x) | x ∈ P 的各层产物 }.
        按管线顺序对每个可用产物调用对应的 HierarchyHasher 方法:
          1. Token 层: 遍历 token 列表 (若可用)
          2. HierarchyTree 层: 哈希结构骨架
          3. RelationNetwork 层: 哈希关系网络
          4. MathModel 层: 哈希数学建模
          5. GeometricObject 层: 哈希几何对象
          6. MemeState 层: 哈希每个模因
          7. CompositeMemeState 层: 哈希复合状态

        Args:
            data: 管线数据上下文.

        Returns:
            自动生成的分层哈希节点列表.
        """
        nodes: List[HierarchicalHash] = []

        # ---- 模块1产物 ----
        # Token 层: 从 meta 中尝试获取 token 列表
        tokens: List[Any] = data.meta.get("tokens", [])
        for i, tok in enumerate(tokens):
            nodes.append(self.hasher.hash_token(tok, i))

        # HierarchyTree 层
        hierarchy_tree: Any = data.meta.get("hierarchy_tree")
        if hierarchy_tree is not None:
            nodes.append(self.hasher.hash_hierarchy_tree(hierarchy_tree))

        # RelationNetwork 层 (Psi)
        if data.psi is not None:
            threshold: float = float(data.meta.get("threshold", 0.0))
            nodes.append(
                self.hasher.hash_relation_network(data.psi, threshold)
            )

        # MathModel 层
        if data.math_model is not None:
            nodes.append(self.hasher.hash_math_model(data.math_model))

        # ---- 模块2产物 ----
        # GeometricObject 层
        if data.geo_object is not None:
            nodes.append(self.hasher.hash_geo_object(data.geo_object))

        # ---- 模块3产物 ----
        # MemeState 层
        if data.meme_state is not None:
            for i, meme in enumerate(data.meme_state.memes):
                nodes.append(self.hasher.hash_meme(meme, i))

            # CompositeMemeState 层
            nodes.append(
                self.hasher.hash_composite(
                    data.meme_state.Theta,
                    data.meme_state.C,
                )
            )

        return nodes
