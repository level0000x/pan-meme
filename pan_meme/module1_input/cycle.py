# 泛模因几何工具 — 模块一：↑↓ 循环引擎
# 数学对应：公理3（↑向上归类 ↓向下展开 互逆操作） + 前提0（有限层级终止）
# 论文位置：附录 D.1-D.2 公理系统, 算法 A Step 1

from typing import List, Optional, Dict, Set, Tuple
import numpy as np

from pan_meme.core.types import Token, HierarchyNode, HierarchyTree


# ============================================================
# 内置上下位词典 — 用于 ↑↓ 操作的规则匹配
# 数学对应：前提2中的规则域基元 F — 由结构唯一确定的 (f, supp) 前驱
# ============================================================
_DEFAULT_HYPERNYM_DICT: Dict[str, List[str]] = {
    # 生物域
    "生物": ["动物", "植物", "微生物"],
    "动物": ["哺乳动物", "鸟类", "爬行动物", "两栖动物", "鱼类"],
    "哺乳动物": ["狗", "猫", "牛", "羊", "猪", "马", "人"],
    "鸟类": ["麻雀", "鹰", "鹦鹉", "鸽子", "企鹅"],
    "植物": ["乔木", "灌木", "草本植物"],
    "乔木": ["松树", "橡树", "杨树", "柳树"],
    "水果": ["苹果", "香蕉", "橘子", "葡萄", "西瓜", "草莓"],
    "蔬菜": ["白菜", "萝卜", "西红柿", "黄瓜", "茄子"],
    # 抽象域
    "颜色": ["红色", "蓝色", "绿色", "黄色", "黑色", "白色"],
    "形状": ["圆形", "方形", "三角形", "椭圆形"],
    "交通工具": ["汽车", "火车", "飞机", "轮船", "自行车"],
    "汽车": ["轿车", "SUV", "卡车", "跑车"],
    "情感": ["喜悦", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶"],
    # 计算域
    "数据结构": ["数组", "链表", "树", "图", "哈希表"],
    "排序算法": ["快速排序", "归并排序", "堆排序", "冒泡排序"],
}


class CycleEngine:
    """公理3实现：↑↓ 自循环补全过程。

    数学对应：
    - 公理3：↑(向上归类) 与 ↓(向下展开) 是互逆操作，构成有限层级的自循环
    - 前提0：层级结构在有限步内终止，保证算法必然收敛
    - 算法 A Step 1：从原始 token 集合 I 构建层级树骨架

    用法:
        engine = CycleEngine(hypernym_dict=my_dict)
        tree = engine.run(tokens, mode="converge", max_rounds=10)
    """

    def __init__(self, hypernym_dict: Optional[Dict[str, List[str]]] = None) -> None:
        """初始化循环引擎。

        参数:
          hypernym_dict: 自定义上下位词典。若为 None，使用内置中文词典。
        """
        self._dict = hypernym_dict if hypernym_dict is not None else _DEFAULT_HYPERNYM_DICT
        # 构建反向索引：下位词 → 上位词集合（用于快速查找父候选）
        self._hyponym_to_hypernyms: Dict[str, Set[str]] = {}
        for hyper, hypos in self._dict.items():
            for hypo in hypos:
                self._hyponym_to_hypernyms.setdefault(hypo, set()).add(hyper)

    # ================================================================
    # 公有方法：run — 主循环入口
    # 数学对应：算法 A Step 1, 公理3 循环语义
    # ================================================================

    def run(self, tokens: List[Token], mode: str = "converge",
            max_rounds: int = 10) -> HierarchyTree:
        """执行 ↑↓ 循环，补全层级结构。

        数学对应：
        - 公理3：对每个 x ∈ U，执行 ↑(x) 寻找上位 y，执行 ↓(y) 展开子元素
        - 前提0：循环在有限轮内终止（max_rounds 或自动收敛）

        参数:
          tokens: 原始 Token 列表，对应论域 U 的原子元素（公理2）
          mode: 循环模式 — "fixed"（固定轮数）| "converge"（自动收敛检测）
          max_rounds: 最大循环轮数（前提0：有限层级保证）

        返回:
          HierarchyTree — 含完整层级结构 + termination_record 的树对象
        """
        n: int = len(tokens)
        # 初始化树：每个 Token 对应一个节点，初始均为未归类（P(x)=False）
        tree: HierarchyTree = HierarchyTree(
            nodes=[HierarchyNode(token_idx=i) for i in range(n)],
            root_indices=[],
            depth=0,
            rounds=0,
            terminated_by="initialized",
        )

        # ─── 主循环：↑↓交替执行 ───
        # 数学对应：公理3 — 每一轮执行完整的 ↑ 后跟 ↓
        for round_idx in range(1, max_rounds + 1):
            changed: bool = False

            # ── ↑ 向上归类（公理3a）──────────────────────────
            # 对每个未归类元素（P(x)=False，即 parent=None），搜索其上位词
            for i, node in enumerate(tree.nodes):
                if node.parent is None:
                    superior: Optional[int] = self._find_superordinate(i, tree, tokens)
                    if superior is not None and superior != i:
                        # 建立从属关系：x → y （公理2：x 从属于 y）
                        node.parent = superior
                        tree.nodes[superior].children.append(i)
                        changed = True

            # ── ↓ 向下展开（公理3b）──────────────────────────
            # 对每个已归类元素（P(x)=True，即 parent is not None），展开内部构成
            for i, node in enumerate(tree.nodes):
                if node.parent is not None:
                    subs: List[int] = self._decompose(i, tree, tokens)
                    for sub_idx in subs:
                        if sub_idx not in node.children:
                            node.children.append(sub_idx)
                            # 同时建立子节点的父引用（若尚未设置）
                            if tree.nodes[sub_idx].parent is None:
                                tree.nodes[sub_idx].parent = i
                            changed = True

            # ── 自动收敛检测（前提0：有限层级必然终止）───────
            if mode == "converge" and not changed:
                tree.terminated_by = "converge"
                tree.rounds = round_idx
                tree.termination_record = {
                    "mode": "auto",
                    "reason": "自动收敛（连续两轮无新变化）",
                    "final_level": self._compute_depth(tree),
                    "total_rounds": round_idx,
                    "converged_at_round": round_idx,
                }
                break

        else:
            # 达到最大轮数（前提0：固定轮数模式下的强制终止）
            tree.terminated_by = "fixed"
            tree.rounds = max_rounds
            tree.termination_record = {
                "mode": "fixed",
                "reason": "达到设定轮数",
                "final_level": self._compute_depth(tree),
                "total_rounds": max_rounds,
                "converged_at_round": None,
            }

        # 后处理：收集根节点（无父节点的顶层元素），计算最终深度
        tree.root_indices = [
            i for i, node in enumerate(tree.nodes) if node.parent is None
        ]
        tree.depth = self._compute_depth(tree)
        return tree

    # ================================================================
    # 内部方法：↑ 操作 — 寻找上位词
    # 数学对应：公理3a — ↑(x) 向上归类映射
    # ================================================================

    def _find_superordinate(self, idx: int, tree: HierarchyTree,
                            tokens: List[Token]) -> Optional[int]:
        """↑ 操作：为索引 idx 的元素寻找上位词。

        数学对应：
        - 公理3a：↑(x) 返回 y 使得 x 从属于 y（P(x)=True 后）
        - 定义2 F域前驱：词典规则是 (f, supp) 在提取前的基元形式

        策略（按优先级）：
        1. 规则词典匹配：利用上下位词典精确匹配（O(1) 查找）
        2. Embedding 余弦相似度：> 0.7 阈值的语义关联（O(n) 遍历）

        参数:
          idx: 当前 token 在 tree.nodes 中的索引
          tree: 当前层级树状态
          tokens: 原始 token 列表

        返回:
          上位节点索引，若未找到则返回 None
        """
        current_token: Token = tokens[idx]
        current_text: str = current_token.text.strip()

        # ── 策略1：规则词典匹配（精确上位关系）─────────────
        # 查询当前词是否为某上位词的下位词
        if current_text in self._hyponym_to_hypernyms:
            candidates: Set[str] = self._hyponym_to_hypernyms[current_text]
            # 在已有节点中查找匹配的上位词
            for candidate_hyper in candidates:
                for j, node in enumerate(tree.nodes):
                    if j == idx:
                        continue
                    if tokens[j].text.strip() == candidate_hyper:
                        # 避免自环：上位词不能是自身
                        if not self._would_create_cycle(idx, j, tree):
                            return j

        # ── 策略2：Embedding 余弦相似度（语义近似上位）─────
        # 数学对应：前提2 — 语义相似度作为信息无损的近似度量
        emb_i: Optional[np.ndarray] = current_token.embedding
        if emb_i is not None:
            best_score: float = 0.0
            best_idx: Optional[int] = None
            for j, other_node in enumerate(tree.nodes):
                if j == idx:
                    continue
                emb_j: Optional[np.ndarray] = tokens[j].embedding
                if emb_j is None:
                    continue
                # 计算余弦相似度：cos(emb_i, emb_j) = (emb_i·emb_j)/(||emb_i|| ||emb_j||)
                sim: float = float(
                    np.dot(emb_i, emb_j) /
                    (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-12)
                )
                if sim > 0.7 and sim > best_score:
                    best_score = sim
                    best_idx = j
            if best_idx is not None:
                if not self._would_create_cycle(idx, best_idx, tree):
                    return best_idx

        return None

    # ================================================================
    # 内部方法：↓ 操作 — 展开内部构成
    # 数学对应：公理3b — ↓(y) 向下展开映射
    # ================================================================

    def _decompose(self, idx: int, tree: HierarchyTree,
                   tokens: List[Token]) -> List[int]:
        """↓ 操作：展开索引 idx 元素的内部构成。

        数学对应：
        - 公理3b：↓(y) 返回 C(y) = {x₁, ..., xₙ}，即 y 的所有子元素
        - 公理2：C(y) 表示 y 的直接构成部分

        策略：词典反向匹配 — 若 y 的文本是某上位词，则其下位词构成子元素

        参数:
          idx: 当前节点在 tree.nodes 中的索引
          tree: 当前层级树状态
          tokens: 原始 token 列表

        返回:
          子元素索引列表
        """
        current_text: str = tokens[idx].text.strip()
        # 查询当前词是否在词典中作为上位词出现
        hyponym_texts: List[str] = self._dict.get(current_text, [])
        if not hyponym_texts:
            return []

        sub_indices: List[int] = []
        for hypo_text in hyponym_texts:
            for j, node in enumerate(tree.nodes):
                if j == idx:
                    continue
                if tokens[j].text.strip() == hypo_text:
                    sub_indices.append(j)
                    break  # 每个文本只匹配第一个出现的 token

        return sub_indices

    # ================================================================
    # 内部方法：深度计算
    # 数学对应：前提0 — 层级深度是有限终止的核心度量
    # ================================================================

    def _compute_depth(self, tree: HierarchyTree) -> int:
        """计算层级树的最大深度。

        数学对应：
        - 前提0：depth(tree) < +∞，保证有限层级终止
        - 定义1：depth(Ψ) 是关系网络层级结构的量化指标

        算法：从每个根节点出发 DFS，取最大深度。
        深度定义为根节点 level=0，每向下一层 +1。

        参数:
          tree: 当前层级树

        返回:
          整型最大深度
        """
        if not tree.root_indices:
            return 0

        max_d: int = 0
        # 对每个根节点做 DFS
        for root_idx in tree.root_indices:
            stack: List[Tuple[int, int]] = [(root_idx, 0)]
            while stack:
                node_idx, depth = stack.pop()
                max_d = max(max_d, depth)
                for child_idx in tree.nodes[node_idx].children:
                    stack.append((child_idx, depth + 1))
        return max_d

    # ================================================================
    # 内部方法：上下位关系判定
    # 数学对应：公理2 — 从属关系的词典化判定
    # ================================================================

    def _is_hypernym_of(self, a: str, b: str) -> bool:
        """判定 a 是否为 b 的上位词。

        数学对应：
        - 公理2：P(b)=True ⟹ b 从属于某个上位元素
        - 上位关系 ≻：a ≻ b 当 b 在 a 的下位词集合中

        参数:
          a: 候选上位词文本
          b: 候选下位词文本

        返回:
          a 是否为 b 的上位词
        """
        hypos: List[str] = self._dict.get(a, [])
        return b in hypos

    def _is_hyponym_of(self, a: str, b: str) -> bool:
        """判定 a 是否为 b 的下位词。

        数学对应：
        - 公理2 逆关系：a 是 b 的下位词 ⟺ b 是 a 的上位词
        - 即 hyponym(a, b) ⟺ hypernym(b, a)

        参数:
          a: 候选下位词文本
          b: 候选上位词文本

        返回:
          a 是否为 b 的下位词
        """
        return self._is_hypernym_of(b, a)

    # ================================================================
    # 内部方法：循环检测
    # ================================================================

    def _would_create_cycle(self, child_idx: int, parent_idx: int,
                            tree: HierarchyTree) -> bool:
        """检测将 child_idx 设为 parent_idx 的子节点是否会形成环。

        前提0保证：层级结构必须是无环的（DAG/树），否则有限终止不成立。

        参数:
          child_idx: 拟设为子节点的索引
          parent_idx: 拟设为父节点的索引
          tree: 当前层级树

        返回:
          是否会形成环（True 表示会形成环，应拒绝此从属关系）
        """
        # 沿 parent_idx 向上追溯，检查是否会回到 child_idx
        current: Optional[int] = parent_idx
        while current is not None:
            if current == child_idx:
                return True  # 会形成环
            current = tree.nodes[current].parent
        return False
