# 泛模因几何工具 — 模块一：彻底不完整判定引擎
# 数学对应：前提1验证——U 仅包含有可识别结构的信息
# 论文位置：前提1（结构完整性），前提0（有限层级终止），算法A Step 1
# 核心思想：对关系网络 Ψ 进行结构完整性诊断，判定是否需要人工介入

from typing import List, Set, Tuple, Deque
from collections import deque
import numpy as np

from pan_meme.core.types import (
    RelationNetwork, HierarchyTree, CompletenessReport, GapInfo
)


class CompletenessChecker:
    """彻底不完整判定器：验证前提1 — 输入信息具有可识别结构。

    数学对应：
    - 前提1：U 仅包含有可识别结构的信息。若无，触发前提0（人工补全）。
    - 前提0：人工介入提供缺失信息，形成人机循环。
    - 算法 A Step 1 后置：检查 A(I)=Ψ 是否满足结构完整性。

    四条判定准则：
    1. 最大连通分量占比 < 80%：网络严重碎裂，存在大量结构空洞。
    2. 孤立节点占比 > 20%：过多元素无可关联的结构，信息不完整。
    3. 层级深度 = 0：无层级结构，结构极度扁平。
    4. 累计空洞数 ≥ 2：多个维度同时不完整，需人工介入。

    用法:
        checker = CompletenessChecker()
        report = checker.check(psi, tree)
        if not report.is_complete:
            print(report.human_request)  # 人工补全请求
    """

    # ── 判定阈值（来自论文附录D.3）──
    _COMPONENT_RATIO_THRESHOLD: float = 0.80     # 最大连通分量占比最低阈值
    _ISOLATED_RATIO_THRESHOLD: float = 0.20       # 孤立节点占比最高容忍度
    _HUMAN_INTERVENTION_THRESHOLD: int = 2        # 触发人工介入的空洞数量阈值

    def __init__(self) -> None:
        """初始化彻底不完整判定器。无超参数——所有阈值来自论文附录。"""

    # ================================================================
    # 公有方法：check — 主判定入口
    # 数学对应：前提1 验证函数 — Ψ × Tree → CompletenessReport
    # ================================================================

    def check(self, psi: RelationNetwork,
              tree: HierarchyTree) -> CompletenessReport:
        """对关系网络 Ψ 进行结构完整性判定，生成 CompletenessReport。

        数学对应：
        - 前提1：U 仅包含有可识别结构的信息
        - 检测 Ψ 是否满足结构完整性，不满足则标记需人工介入
        - CompletenessReport 编码所有空洞信息，用于下游人工补全循环

        四条判定逻辑：
        1. 最大连通分量 < 80% 节点总数 → structural_hole 空洞
        2. 孤立节点（度数=0）> 20% 节点总数 → isolated_nodes 空洞
        3. 层级深度 = 0 → flat_structure 空洞
        4. 空洞累计数 > 2 → 触发人工介入（前提0）

        参数:
          psi: 待判定的关系网络 Ψ（可能已被 Reasoner 补全为 Ψ*）
          tree: 对应的层级树（用于获取深度信息）

        返回:
          CompletenessReport — 含 is_complete, gaps 列表, requires_human 标记
        """
        n: int = len(psi.nodes)
        gaps: List[GapInfo] = []

        # ── 判定1：最大连通分量占比 ──
        # 数学对应：largest CC / |V| < 80% → 网络结构存在断裂
        largest_cc_size: int = self._largest_component_size(psi)
        component_ratio: float = largest_cc_size / float(n) if n > 0 else 0.0

        if component_ratio < self._COMPONENT_RATIO_THRESHOLD:
            gaps.append(GapInfo(
                location="全局网络",
                gap_type="structural_hole",
                suggested_direction=(
                    "最大连通分量仅覆盖 {:.1%} 的节点，"
                    "网络存在显著结构断裂。请补充缺失的节点间关联信息。"
                ).format(component_ratio),
                context={
                    "largest_component_ratio": round(component_ratio, 4),
                    "largest_component_size": largest_cc_size,
                    "total_nodes": n,
                },
            ))

        # ── 判定2：孤立节点占比 ──
        # 数学对应：孤立节点 = {v ∈ V | deg(v) = 0}, 占比 > 20% → 信息严重不完整
        isolated_count: int = self._count_isolated(psi)
        isolated_ratio: float = isolated_count / float(n) if n > 0 else 0.0

        if isolated_ratio > self._ISOLATED_RATIO_THRESHOLD:
            gaps.append(GapInfo(
                location="孤立节点集",
                gap_type="isolated_nodes",
                suggested_direction=(
                    "存在 {:.1%} 的孤立节点（{} 个），"
                    "过多元素无法建立任何结构关联。请为孤立元素补充上下文信息。"
                ).format(isolated_ratio, isolated_count),
                context={
                    "isolated_ratio": round(isolated_ratio, 4),
                    "isolated_count": isolated_count,
                    "total_nodes": n,
                },
            ))

        # ── 判定3：层级深度 ──
        # 数学对应：depth(Ψ) = 0 → 完全平坦，无层次化结构
        depth: int = tree.depth if tree.depth is not None else psi.hierarchy.get("levels", 0)
        if depth == 0:
            gaps.append(GapInfo(
                location="层级结构",
                gap_type="flat_structure",
                suggested_direction=(
                    "层级深度为 0，信息未形成任何层次化结构。"
                    "请重新检查输入或提供更丰富的概念层次信息。"
                ),
                context={
                    "depth": 0,
                    "levels": 0,
                    "total_nodes": n,
                },
            ))

        # ── 判定4：人工介入判定 ──
        # 数学对应：前提0 — 当空洞数 > 2 时，自动触发人机循环
        is_complete: bool = len(gaps) == 0
        requires_human: bool = len(gaps) >= self._HUMAN_INTERVENTION_THRESHOLD
        human_request: str | None = None

        if requires_human:
            gap_descriptions: str = "\n  - ".join(
                f"[{g.gap_type}] {g.location}" for g in gaps
            )
            human_request = (
                f"【自动完整性诊断】关系网络存在 {len(gaps)} 处结构空洞，"
                f"当前状态不足以进行下游几何化运算。\n"
                f"检测到的空洞：\n  - {gap_descriptions}\n"
                f"请针对以上空洞提供补全信息后重新运行管线。"
            )

        return CompletenessReport(
            is_complete=is_complete,
            gaps=gaps,
            requires_human=requires_human,
            human_request=human_request,
        )

    # ================================================================
    # 内部方法：_largest_component_size — BFS 最大连通分量
    # 数学对应：max_{C⊆V} |C|, where C 是连通分量
    # ================================================================

    def _largest_component_size(self, psi: RelationNetwork) -> int:
        """通过 BFS 计算最大连通分量的大小。

        数学对应：
        - 连通分量 C ⊆ V：∀u,v ∈ C, 存在 u~v 的无向路径
        - 最大连通分量：CC_max = argmax_{C} |C|
        - 分量占比 = |CC_max| / |V| 衡量网络的整体连通性

        算法：标准 BFS，O(|V| + |E|)。
        使用 bool 数组 visited 标记已访问节点。
        邻接表由 psi.edges 双向构建。

        参数:
          psi: 关系网络 Ψ

        返回:
          整型 — 最大连通分量中的节点数
        """
        n: int = len(psi.nodes)
        if n == 0:
            return 0

        # ── 构建邻接表（双向无向图）──
        # 数学对应：邻接矩阵 A[u][v] = 1 ⇔ (u,v) ∈ E 或 (v,u) ∈ E
        adj: List[Set[int]] = [set() for _ in range(n)]
        for u, v in psi.edges:
            adj[u].add(v)
            adj[v].add(u)

        # ── BFS 遍历所有连通分量 ──
        visited: np.ndarray = np.zeros(n, dtype=np.bool_)
        max_size: int = 0

        for start in range(n):
            if visited[start]:
                continue
            # 开始一个新的连通分量
            component_size: int = 0
            queue: Deque[int] = deque([start])
            visited[start] = True

            while queue:
                u: int = queue.popleft()
                component_size += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)

            if component_size > max_size:
                max_size = component_size

        return max_size

    # ================================================================
    # 内部方法：_count_isolated — 计数孤立节点
    # 数学对应：|{v ∈ V | deg(v) = 0}|
    # ================================================================

    def _count_isolated(self, psi: RelationNetwork) -> int:
        """计数关系网络中度数 = 0 的孤立节点。

        数学对应：
        - 孤立节点：Iso(Ψ) = {v ∈ V | deg(v) = 0}
        - deg(v) = |{u ∈ V | (v,u) ∈ E 或 (u,v) ∈ E}|
        - 孤立节点占比 = |Iso(Ψ)| / |V| 衡量信息的覆盖完整性

        算法：通过 bool 数组标记有哪些边涉及的节点，
              未被标记的节点度数为 0。

        参数:
          psi: 关系网络 Ψ

        返回:
          整型 — 孤立节点数量
        """
        n: int = len(psi.nodes)
        if n == 0:
            return 0

        # 使用 bool 数组标记有边的节点
        has_edge: np.ndarray = np.zeros(n, dtype=np.bool_)
        for u, v in psi.edges:
            has_edge[u] = True
            has_edge[v] = True

        # 度数 = 0 的节点 = 未被任何边涉及的节点
        isolated_count: int = int(np.sum(~has_edge))
        return isolated_count
