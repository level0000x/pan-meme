# 泛模因几何工具 — 模块一：关系自组织提取器
# 数学对应：前提2（算法A: I→Ψ=(V,E,w)） + 附录D.3（T∈T有限离散候选集）
# 论文位置：定义1（Ψ=(V,E,w)）, 算法A Step 2

from typing import List, Tuple, Dict, Optional
import numpy as np

from pan_meme.core.types import HierarchyTree, RelationNetwork, Token

# PB 级新引擎导入 — 数学对应：PB_ARCHITECTURE.md 第3.1节 LSH 近似加速
try:
    from pan_meme.engines.minhash_lsh import LSHRelationExtractor
    _LSH_AVAILABLE: bool = True
except ImportError:
    LSHRelationExtractor = None  # type: ignore
    _LSH_AVAILABLE: bool = False


class RelationExtractor:
    """前提2实现：从 HierarchyTree 提取关系网络 Ψ = (V, E, w)。

    数学对应：
    - 定义1：Ψ = (V, E, w) — 顶点集 V，边集 E，权重函数 w: E → [0,1]
    - 前提2：算法 A: I → Ψ 是信息无损映射（在有限精度内）
    - 附录 D.3：对每个阈值 T ∈ T，生成一个 Ψ_T，构成候选集

    边权重 w_ij 定义为：
      w_ij = max(共现强度, 语义相似度)
    其中：
    - 共现强度 = 1/(|depth_i - depth_j| + 1)，仅当两节点共享同一父节点
    - 语义相似度 = embedding 余弦相似度（外部注入，默认 0）

    用法:
        extractor = RelationExtractor()
        network = extractor.extract(tree, threshold=0.3)

    PB 级扩展：
        extractor = RelationExtractor(use_lsh=True, lsh_bands=20, lsh_rows=5)
        network = extractor.extract(tree, threshold=0.3)  # LSH 近似加速
    """

    def __init__(self, use_lsh: bool = False,
                 lsh_bands: int = 20, lsh_rows: int = 5,
                 lsh_window: int = 100) -> None:
        """初始化关系提取器 — 支持 O(N²) 精确模式与 LSH 近似模式。

        数学对应：PB_ARCHITECTURE.md 第3.1节 — LSH 近似加速
        - use_lsh=False：原始 O(N²) 全对扫描（精确模式，默认）
        - use_lsh=True：MinHash LSH 近似去重，PB 级适用

        参数:
          use_lsh: 是否启用 LSH 近似模式（默认 False，保持向后兼容）
          lsh_bands: LSH band 数量，控制碰撞灵敏度（默认 20）
          lsh_rows: 每个 band 的行数，控制签名精度（默认 5）
          lsh_window: 滑动窗口大小，控制候选对生成范围（默认 100）
        """
        self.use_lsh: bool = use_lsh
        self.lsh_bands: int = lsh_bands
        self.lsh_rows: int = lsh_rows
        self.lsh_window: int = lsh_window

    def extract(self, tree: HierarchyTree, threshold: float = 0.2) -> RelationNetwork:
        """从层级树提取带阈值的加权关系网络 Ψ_T。

        数学对应：
        - 算法 A Step 2：对给定 A(I) 生成 Ψ
        - 附录 D.3：Ψ_T = (V, E_T, w|E_T)，其中 E_T = {e ∈ E | w(e) ≥ T}

        参数:
          tree: 输入层级树（由 CycleEngine.run 生成）
          threshold: 边权重阈值 T ∈ [0, 1]。仅保留 w_ij ≥ T 的边。

        返回:
          RelationNetwork — 带权重的关系网络，含层级元数据
        """
        # ── PB 级引擎分流：use_lsh=True 时走 LSH 近似加速路径 ──
        # 数学对应：PB_ARCHITECTURE.md 第3.1节 — LSH 近似加速
        if self.use_lsh:
            if not _LSH_AVAILABLE:
                raise ImportError(
                    "LSH 引擎不可用：pan_meme.engines.minhash_lsh 未安装或导入失败"
                )
            return self.extract_lsh(tree, threshold)

        n: int = len(tree.nodes)

        # ── 构建 n×n 对称权重矩阵 W ──
        # 数学对应：定义1 — w: E → [0,1] 的上三角矩阵表示
        W: np.ndarray = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                # 共现强度：基于共享父节点的层级距离
                # 数学对应：前提2 — 结构近邻是信息无损的最强信号
                cooc: float = self._cooccurrence_strength(i, j, tree)

                # 语义相似度：由外部 embedding 注入（当前版本默认 0）
                # 数学对应：前提2 — 语义相似度作为辅助信号，权重取 max
                sem: float = 0.0

                # w_ij = max(共现强度, 语义相似度)
                # 取最大值确保两种信号互补：强结构关系不被弱语义覆盖
                W[i, j] = W[j, i] = max(cooc, sem)

        # ── 阈值过滤：仅保留 w_ij ≥ T 的边 ──
        # 数学对应：附录 D.3 — Ψ_T 是 T 参数化的网络族
        edges: List[Tuple[int, int]] = []
        weights: List[float] = []

        for i in range(n):
            for j in range(i + 1, n):
                w_ij: float = float(W[i, j])
                if w_ij >= threshold:
                    edges.append((i, j))
                    weights.append(w_ij)

        # ── 构建节点标识符 ──
        nodes: List[str] = [f"node_{i}" for i in range(n)]

        # ── 构建层级元数据 ──
        # 数学对应：定义1 附录 — hierarchy 编码层级结构信息的字典
        hierarchy: Dict[str, object] = {
            "levels": tree.depth,
            "node_levels": {i: self._node_depth(i, tree) for i in range(n)},
            "combination_path": [],  # 提取路径：单阈值直接提取，无组合
            "parent_map": {
                i: tree.nodes[i].parent
                for i in range(n)
                if tree.nodes[i].parent is not None
            },
        }

        # ── 构建元数据 ──
        metadata: Dict[str, object] = {
            "input_type": "text",
            "node_count": n,
            "edge_count": len(edges),
            "threshold_used": threshold,
        }

        return RelationNetwork(
            nodes=nodes,
            edges=edges,
            weights=np.array(weights, dtype=np.float32),
            hierarchy=hierarchy,
            metadata=metadata,
        )

    # ================================================================
    # 内部方法：共现强度计算
    # 数学对应：前提2 — 结构共现是 w_ij 的主信号源
    # ================================================================

    def _cooccurrence_strength(self, i: int, j: int,
                               tree: HierarchyTree) -> float:
        """计算两节点的共现强度。

        数学对应：
        - 前提2：共现强度 ∝ 1/层级距离 — 同层近邻更强
        - 定义：cooc(i, j) = 1/(|depth(i) - depth(j)| + 1)

        条件：仅当 i 和 j 共享同一父节点时才计算非零值。
        这保证了结构相关性的语义基础 —— 同属一个上位概念的
        元素之间天然存在共现关系。

        参数:
          i: 节点 i 的索引
          j: 节点 j 的索引
          tree: 当前层级树

        返回:
          共现强度 ∈ [0, 1]
        """
        pi: Optional[int] = tree.nodes[i].parent
        pj: Optional[int] = tree.nodes[j].parent

        # 仅当共享同一父节点（且父节点存在）时才有共现关系
        # 数学对应：公理2 — C(y) 内元素共享 struct 上下文
        if pi is None or pj is None:
            return 0.0
        if pi != pj:
            return 0.0

        # 计算层级距离并取倒数
        depth_i: int = self._node_depth(i, tree)
        depth_j: int = self._node_depth(j, tree)
        return 1.0 / (abs(depth_i - depth_j) + 1)

    # ================================================================
    # 内部方法：节点深度计算
    # 数学对应：前提0 — 层级深度是有限终止的核心度量
    # ================================================================

    def _node_depth(self, idx: int, tree: HierarchyTree) -> int:
        """计算指定节点在层级树中的深度。

        数学对应：
        - 前提0：depth(node) = 沿 parent 链向上到根的步数
        - 根节点 depth = 0

        算法：沿 parent 指针向上追溯，计数步数。

        参数:
          idx: 节点在 tree.nodes 中的索引
          tree: 当前层级树

        返回:
          整型深度值
        """
        depth: int = 0
        current: Optional[int] = tree.nodes[idx].parent
        while current is not None:
            depth += 1
            current = tree.nodes[current].parent
        return depth

    # ================================================================
    # PB 级新增：LSH 近似关系提取
    # 数学对应：PB_ARCHITECTURE.md 第3.1节 — LSH 近似加速
    # ================================================================

    def extract_lsh(self, tree: HierarchyTree, threshold: float) -> RelationNetwork:
        """LSH 近似模式：从层级树提取加权关系网络。

        数学对应：PB_ARCHITECTURE.md 第3.1节 — LSH 近似加速
        - Step 1: 从 tree 提取 token_texts 列表
        - Step 2: 构建伪 Token 对象列表（用于 LSH 引擎的签名计算）
        - Step 3: 调用 LSHRelationExtractor.extract() 得到 edges + weights
        - Step 4: 构建 RelationNetwork 返回（与 extract 保持同一接口格式）

        参数:
          tree: 输入层级树
          threshold: 边权重阈值 T ∈ [0, 1]

        返回:
          RelationNetwork — 带权重的关系网络，含层级元数据
        """
        # ── Step 1: 从 tree 提取 Token 列表 ──
        tokens: List[Token] = self._tokens_from_tree(tree)
        n: int = len(tokens)

        # ── Step 2: 构建 LSH 引擎实例并运行 ──
        lsh_extractor = LSHRelationExtractor(
            bands=self.lsh_bands,
            rows=self.lsh_rows,
            window=self.lsh_window,
        )
        pairs, lsh_weights = lsh_extractor.extract(tokens, threshold)

        # ── Step 3: 转换为 edges + weights 列表 ──
        edges: List[Tuple[int, int]] = []
        weights: List[float] = []
        for (i, j), w in zip(pairs, lsh_weights):
            if i != j and w >= threshold:
                edges.append((i, j))
                weights.append(float(w))

        # ── Step 4: 构建节点标识符 ──
        nodes: List[str] = [f"node_{i}" for i in range(n)]

        # ── Step 5: 构建层级元数据 ──
        hierarchy: Dict[str, object] = {
            "levels": tree.depth,
            "node_levels": {i: self._node_depth(i, tree) for i in range(n)},
            "combination_path": ["lsh_approximate"],
            "parent_map": {
                i: tree.nodes[i].parent
                for i in range(n)
                if tree.nodes[i].parent is not None
            },
        }

        # ── Step 6: 构建元数据 ──
        metadata: Dict[str, object] = {
            "input_type": "text",
            "node_count": n,
            "edge_count": len(edges),
            "threshold_used": threshold,
            "extraction_mode": "lsh_approximate",
            "lsh_bands": self.lsh_bands,
            "lsh_rows": self.lsh_rows,
            "lsh_window": self.lsh_window,
        }

        return RelationNetwork(
            nodes=nodes,
            edges=edges,
            weights=np.array(weights, dtype=np.float32),
            hierarchy=hierarchy,
            metadata=metadata,
        )

    # ================================================================
    # PB 级新增：从 HierarchyTree 抽取 Token 列表
    # 数学对应：PB_ARCHITECTURE.md 第3.1节 — Token 化作为 LSH 签名输入
    # ================================================================

    def _tokens_from_tree(self, tree: HierarchyTree) -> List[Token]:
        """从 HierarchyTree 抽取 Token 列表。

        数学对应：PB_ARCHITECTURE.md 第3.1节 — LSH 引擎输入
        - 每个节点映射为一个 Token 对象
        - payload 携带层级深度和父子关系，用于 LSH 签名计算

        参数:
          tree: 输入层级树

        返回:
          Token 对象列表，按 tree.nodes 的顺序
        """
        tokens: List[Token] = []
        for i, node in enumerate(tree.nodes):
            # 构建 payload：携带层级深度和父子关系
            payload: Dict[str, object] = {
                "depth": self._node_depth(i, tree),
                "parent": node.parent,
                "index": i,
            }
            # 构造 Token（text 使用 node 的文本表示或索引标识）
            token_text: str = getattr(node, "text", f"token_{i}")
            tokens.append(Token(text=token_text, payload=payload))
        return tokens
