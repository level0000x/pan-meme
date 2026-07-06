# 泛模因几何工具 — 模块一：概念-要素循环组合引擎
# 数学对应：Louvain 多层社区检测——要素聚合为概念
# 论文位置：定义1 附录（hierarchy.node_levels），公理3（多层级循环），附录D.3
# 核心思想：通过 Louvain 社区检测将底层要素逐层聚合为高层概念

from typing import List, Dict, Tuple, Optional
import numpy as np

import networkx as nx
from networkx.algorithms.community import louvain_communities

from pan_meme.core.types import RelationNetwork


class ConceptComposer:
    """概念-要素循环组合器：通过 Louvain 多层社区检测将要素聚合为概念。

    数学对应：
    - Louvain 社区检测：模块度最大化的层次聚类算法
    - 多层聚合：每一层将社区收缩为单个概念节点，概念间的边权取跨社区均值
    - 公理3 拓展：↑(要素→概念) 和 ↓(概念→要素) 在多层级中反复执行
    - 至多 10 层或收敛（社区数不变）时终止

    每轮操作：
    1. Louvain 社区检测 → 得到当前层社区划分
    2. 每个社区 = 一个抽象概念节点
    3. 概念间边权 = 跨社区原始边权的均值
    4. 更新 hierarchy.node_levels 记录每个原始节点所属的概念层级
    5. 当前层图 = 概念图（用于下一轮聚合）

    用法:
        composer = ConceptComposer()
        psi_with_concepts = composer.compose(psi)
        # psi_with_concepts.hierarchy["node_levels"] 记录了每节点的层级归属
        # psi_with_concepts.hierarchy["levels"] 记录了总聚合层数
    """

    # ── 聚合参数 ──
    _MAX_LEVELS: int = 10         # 最大聚合层级（前提0：有限层级保证）
    _LOUVAIN_SEED: int = 42       # Louvain 随机种子（保证可复现性）

    def __init__(self) -> None:
        """初始化概念组合器。无超参数——最大层数来自前提0的有限层级保证。"""

    # ================================================================
    # 公有方法：compose — 主组合入口
    # 数学对应：多层 Louvain → 概念层级图，Ψ 的 hierarchy 字段扩展
    # ================================================================

    def compose(self, psi: RelationNetwork) -> RelationNetwork:
        """执行 Louvain 多层社区检测，将要素逐层聚合为概念。

        数学对应：
        - 层级聚合：从原始关系网络出发，逐层收缩社区为概念
        - 收敛条件：社区数不变（不能再细分）或达到最大层数
        - 输出：在 Ψ.hierarchy 中记录 node_levels 和 levels，供下游使用

        算法步骤：
        1. 将 Ψ 转换为 networkx 无向加权图 G
        2. 初始化 level_labels：所有原始节点 level = 0
        3. 循环（至多 10 层）：
           a. Louvain 社区检测
           b. 若社区数 = 节点数，收敛停止
           c. 构建下一层概念图（社区间跨边均值）
           d. 更新 level_labels
           e. 将概念图设为当前图，继续下一轮
        4. 将 level_labels 和 levels 写回 psi.hierarchy

        参数:
          psi: 原始关系网络 Ψ（可能已由 Reasoner 补全为 Ψ*）

        返回:
          RelationNetwork — 原网络，但 hierarchy 已扩展 node_levels 和 levels
        """
        n_original: int = len(psi.nodes)

        # ── 步骤1：构建初始 networkx 图 ──
        # 数学对应：G_0 = (V_0, E_0, w_0) — 原始要素级关系图
        G: nx.Graph = nx.Graph()
        G.add_nodes_from(range(n_original))
        for (i, j), w in zip(psi.edges, psi.weights):
            G.add_edge(int(i), int(j), weight=float(w))

        # ── 步骤2：初始化层级标签和聚合映射 ──
        # level_labels[node_id] = 该节点被聚合到的最高层级（0=原始层）
        # 所有原始节点的初始 level = 0
        level_labels: Dict[int, int] = {i: 0 for i in range(n_original)}

        # node_to_community_chain[node_id] = 该节点在各层的社区标签列表
        # 用于追溯元素如何逐层聚合为概念
        node_to_community_chain: Dict[int, List[int]] = {
            i: [i] for i in range(n_original)  # 第0层：每个节点自身
        }

        # current_G 是当前层级的图（在迭代中更新为概念图）
        current_G: nx.Graph = G

        # ── 步骤3：多层 Louvain 循环 ──
        # 数学对应：公理3 — 逐层 ↑ 归类。每层将社区聚合成新概念节点。
        level: int = 0

        while level < self._MAX_LEVELS:
            n_nodes_current: int = current_G.number_of_nodes()

            # ── 3a. Louvain 社区检测 ──
            # 数学对应：模块度最大化 — Q = Σ[内部边权 - 期望边权]
            # seed 固定以保证可复现性
            communities: List[set] = list(
                louvain_communities(current_G, weight="weight", seed=self._LOUVAIN_SEED)
            )

            n_communities: int = len(communities)

            # ── 收敛判定：社区数 = 节点数 ──
            # 数学对应：前提0 — 每个节点自成社区时无法继续聚合
            if n_communities == n_nodes_current:
                break

            # ── 3b. 建立节点→社区索引映射 ──
            # community_of[node] = 所属社区的编号
            community_of: Dict[int, int] = {}
            for comm_idx, comm in enumerate(communities):
                for node in comm:
                    community_of[node] = comm_idx

            # ── 3c. 构建下一层概念图 ──
            # 数学对应：概念图 G_{l+1} = (V_{l+1}, E_{l+1}, w_{l+1})
            #     V_{l+1} = 社区集合（每个社区 = 一个概念节点）
            #     w_{l+1}(a,b) = mean{ w_l(u,v) | u ∈ C_a, v ∈ C_b }
            next_G: nx.Graph = nx.Graph()
            next_G.add_nodes_from(range(n_communities))

            # 累加器：cross_weight[a][b] 累积跨社区边权和
            #         cross_count[a][b] 累积跨社区边计数
            cross_weight: Dict[Tuple[int, int], float] = {}
            cross_count: Dict[Tuple[int, int], int] = {}

            for u, v, edge_data in current_G.edges(data=True):
                cu: int = community_of.get(u, -1)
                cv: int = community_of.get(v, -1)
                if cu == -1 or cv == -1:
                    continue
                w_uv: float = float(edge_data.get("weight", 1.0))
                # 跨社区边：累加权值和计数
                if cu != cv:
                    key: Tuple[int, int] = (min(cu, cv), max(cu, cv))
                    cross_weight[key] = cross_weight.get(key, 0.0) + w_uv
                    cross_count[key] = cross_count.get(key, 0) + 1

            # 计算跨社区均值边权并添加到概念图
            for (ca, cb), total_w in cross_weight.items():
                cnt: int = cross_count.get((ca, cb), 1)
                mean_w: float = total_w / float(cnt)
                next_G.add_edge(ca, cb, weight=mean_w)

            # ── 3d. 更新层级标签 ──
            # 数学对应：node_levels — 记录每个原始节点被聚合到的层级
            # 当前层 level 的社区中的节点被标记为属于 level+1
            for original_node, chain in node_to_community_chain.items():
                if len(chain) <= level:
                    continue
                # 查找该原始节点在当前层的社区归属
                current_community: int = community_of.get(chain[level], -1)
                if current_community >= 0 and len(chain) == level + 1:
                    chain.append(current_community)
                # 该节点被聚合到当前层 → 更新 level_labels
                if current_community >= 0:
                    level_labels[original_node] = level + 1

            # ── 3e. 推进到下一层 ──
            current_G = next_G
            level += 1

        # ── 步骤4：将层级信息写回 psi.hierarchy ──
        # 数学对应：psi.hierarchy["node_levels"] 和 ["levels"] 供下游使用
        psi.hierarchy["node_levels"] = level_labels
        psi.hierarchy["levels"] = level
        psi.hierarchy["combination_rounds"] = level
        psi.hierarchy["final_community_count"] = current_G.number_of_nodes()
        # 保留原始层级信息
        if "parent_map" not in psi.hierarchy:
            psi.hierarchy["parent_map"] = {}

        return psi
