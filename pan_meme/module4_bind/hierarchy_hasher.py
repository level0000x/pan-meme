# 泛模因几何工具 — 模块四：分层哈希引擎
# 数学对应：论文 3.5 节 — 分层哈希绑定层, HierarchicalHash 的生成
# 论文位置：定理 5 (自洽性约束) 的哈希验证, 附录 D.4 哈希树构造
# 核心思想：对管线每一层产物生成独立 HierarchicalHash,
#           通过规范化 JSON 快照保证可复现性,
#           利用子节点链式引用构建分层哈希树

import hashlib
import json
from typing import Any, Dict, List, Optional

import numpy as np

from pan_meme.core.types import (
    HierarchicalHash,
    HierarchyTree,
    MemeState,
    RelationNetwork,
)

# ============================================================
# 轻量级哈希函数
# ============================================================


def sha256(data: str) -> str:
    """对字符串数据计算 SHA-256 十六进制摘要。

    数学对应: 密码学哈希 H: {0,1}* → {0,1}^256,
    满足抗碰撞性 (collision resistance) 和抗原像性 (preimage resistance).
    在所有分层哈希计算中作为统一的哈希原语.

    Args:
        data: 待哈希的字符串 (通常是规范化 JSON 快照).

    Returns:
        64 字符的十六进制小写哈希串.

    Examples:
        >>> sha256("hello")
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


# ============================================================
# 辅助序列化函数
# ============================================================


def _safe_json(obj: Any) -> str:
    """将 Python 对象安全序列化为规范化 JSON 字符串.

    数学对应: canonical representation — 同一数学对象
    在不同运行时产生完全相同的 JSON 表示, 保证哈希可复现性.
    使用 sort_keys=True 消除字典键序不确定性,
    ensure_ascii=True 处理非 ASCII 字符.

    Args:
        obj: 待序列化的 Python 对象 (dict/list/基本类型).

    Returns:
        规范化 JSON 字符串.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str)


# ============================================================
# HierarchyHasher — 分层哈希引擎
# ============================================================


class HierarchyHasher:
    """分层哈希引擎 — 对管线每一层产物生成独立 HierarchicalHash.

    数学对应: 论文 3.5 节, 定理 5 的哈希验证层.
    对管线中的每一层产物 (Token, HierarchyTree, RelationNetwork,
    MathModel, GeometricObject, MemeState, CompositeMemeState)
    分别计算其规范化 JSON 快照和 SHA-256 摘要.
    每个哈希节点记录其子节点引用, 形成可追溯的分层哈希树.

    核心不变量:
    - 同一输入在不同运行时产生完全相同的 hash_value.
    - 每个层级的每个组件都有独立的 component_id.
    - 子节点链保证完整性可逐层验证.

    用法:
        hasher = HierarchyHasher("sha256")
        h_token = hasher.hash_token(tokens[0], idx=0)
        h_tree = hasher.hash_hierarchy_tree(tree)
    """

    # ----------------------------------------------------------------
    # 构造与初始化
    # ----------------------------------------------------------------

    def __init__(self, algo: str = "sha256") -> None:
        """初始化分层哈希引擎.

        Args:
            algo: 哈希算法标识, 当前仅支持 "sha256".
                  预留扩展位: 未来可支持 "sha3-256" / "blake2b" 等.
        """
        self.algo: str = algo
        """当前使用的哈希算法标识."""

    # ----------------------------------------------------------------
    # 层级哈希方法 — 按管线顺序排列
    # ----------------------------------------------------------------

    def hash_token(self, token: Any, idx: int) -> HierarchicalHash:
        """对单个 Token 计算分层哈希.

        数学对应: 公理1 — Token 是信息论域 U 的原子元素.
        规范化快照包含 Token 的结构化属性 (text, span, pos, modality),
        排除 embedding 等运行时动态数据以保证哈希稳定性.

        canonical = json.dumps({"text": token.text, "span": token.span,
                                "pos": token.pos, "modality": token.modality})
        H = sha256(canonical)

        Args:
            token: Token 对象 (来自 module1_input.Tokenizer 的输出).
            idx:   Token 在序列中的索引位置.

        Returns:
            HierarchicalHash 节点, layer="token", component_id="token_{idx}".
        """
        canonical: str = _safe_json({
            "text": getattr(token, "text", str(token)),
            "span": list(getattr(token, "span", (0, 0))),
            "pos": getattr(token, "pos", ""),
            "modality": getattr(token, "modality", "unknown"),
        })
        h: str = sha256(canonical)
        return HierarchicalHash(
            layer="token",
            component_id=f"token_{idx}",
            hash_value=h,
            canonical_json_snapshot=canonical,
            children=[],
            metadata={
                "idx": idx,
                "modality": getattr(token, "modality", "unknown"),
                "pos": getattr(token, "pos", ""),
            },
        )

    def hash_hierarchy_tree(self, tree: HierarchyTree) -> HierarchicalHash:
        """对 HierarchicalTree 计算分层哈希.

        数学对应: 公理2+3 — 多层级结构骨架.
        规范化快照聚焦于树的结构性参数 (depth, rounds, terminated_by,
        termination_record), 而不逐节点记录 (节点级哈希由 hash_token 独立完成).
        这是“骨架哈希”——仅编码结构的宏观特征.

        canonical = json.dumps({"depth": tree.depth, "rounds": tree.rounds,
                                "terminated_by": tree.terminated_by,
                                "termination_record": tree.termination_record})

        Args:
            tree: HierarchyTree 对象 (公理2+3 的输出).

        Returns:
            HierarchicalHash 节点, layer="hierarchy",
            component_id="hierarchy_tree".
        """
        canonical: str = _safe_json({
            "depth": tree.depth,
            "rounds": tree.rounds,
            "terminated_by": tree.terminated_by,
            "termination_record": tree.termination_record,
        })
        h: str = sha256(canonical)
        return HierarchicalHash(
            layer="hierarchy",
            component_id="hierarchy_tree",
            hash_value=h,
            canonical_json_snapshot=canonical,
            children=[],
            metadata={
                "depth": tree.depth,
                "rounds": tree.rounds,
                "terminated_by": tree.terminated_by,
                "n_nodes": len(tree.nodes),
            },
        )

    def hash_relation_network(
        self, psi: RelationNetwork, threshold: float = 0.0
    ) -> HierarchicalHash:
        """对 RelationNetwork 计算分层哈希.

        数学对应: 定义1 — Ψ = (V, E, w).
        规范化快照编码图的宏观拓扑特征: 节点数, 边数, 使用的阈值参数.
        注意: 此处不逐边编码权重, 而是记录图的宏观统计量.
        threshold 作为关键参数被同时纳入快照和元数据.

        canonical = json.dumps({"n_nodes": len(psi.nodes),
                                "n_edges": len(psi.edges),
                                "threshold": threshold})
        metadata 额外包含 threshold 便于快速索引.

        Args:
            psi:       RelationNetwork 对象 (模块1输出, 定义1).
            threshold: 边权过滤阈值 ∈ [0,1], 用于定义有效连接.

        Returns:
            HierarchicalHash 节点, layer="relation",
            component_id="relation_network".
        """
        n_nodes: int = len(psi.nodes)
        n_edges: int = len(psi.edges)
        canonical: str = _safe_json({
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "threshold": threshold,
        })
        h: str = sha256(canonical)
        return HierarchicalHash(
            layer="relation",
            component_id="relation_network",
            hash_value=h,
            canonical_json_snapshot=canonical,
            children=[],
            metadata={
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "threshold": threshold,
            },
        )

    def hash_math_model(self, model: Any) -> HierarchicalHash:
        """对 MathModel 计算分层哈希.

        数学对应: 定义2 — M = (S, F, C). Φ_B: Ψ → M (定理2: 双射).
        分别对结构域 S, 规则域 F, 约束域 C 计算子哈希,
        然后通过 "|".join(children) 合并后做顶层 sha256.
        这是第一个使用 children 链式引用的哈希节点,
        体现了 M 是由三个子域组成的复合数学对象.

        children = [S_hash, F_hash, C_hash]
        combined = "|".join(children)
        H = sha256(combined)

        Args:
            model: MathModel 对象 (模块1输出, 定义2).

        Returns:
            HierarchicalHash 节点, layer="math_model",
            component_id="math_model", children 包含三个子域哈希.
        """
        # ---- 步骤1: 对结构域 S (RelationNetwork) 计算子哈希 ----
        # 数学对应: S 是 M 的第一组件, 编码为规范化快照
        s_canonical: str = _safe_json({
            "n_nodes": len(model.structure.nodes),
            "n_edges": len(model.structure.edges),
            "hierarchy": model.structure.hierarchy,
        })
        s_h: str = sha256(s_canonical)

        # ---- 步骤2: 对规则域 F (List[RuleDef]) 计算子哈希 ----
        # 数学对应: F = {(f, supp)} 是定义2的规则域
        rules_data: List[Dict[str, Any]] = []
        for rule in model.rules:
            rules_data.append({
                "pattern": rule.pattern,
                "support": rule.support,
                "confidence": rule.confidence,
                "source": rule.source,
            })
        f_canonical: str = _safe_json(rules_data)
        f_h: str = sha256(f_canonical)

        # ---- 步骤3: 对约束域 C (List[ConstraintDef]) 计算子哈希 ----
        # 数学对应: C = {(c, dom)} 是定义2的约束域
        constraints_data: List[Dict[str, Any]] = []
        for cons in model.constraints:
            constraints_data.append({
                "condition": cons.condition,
                "domain": cons.domain,
                "description": cons.description,
                "confidence": cons.confidence,
                "source": cons.source,
            })
        c_canonical: str = _safe_json(constraints_data)
        c_h: str = sha256(c_canonical)

        # ---- 步骤4: 合并子哈希构建顶层节点 ----
        # 数学对应: M 的哈希 = H(S_hash || "|" || F_hash || "|" || C_hash)
        children: List[str] = [s_h, f_h, c_h]
        return self.hash_component(
            layer="math_model",
            component_id="math_model",
            canonical_data=_safe_json({
                "s_hash": s_h,
                "f_hash": f_h,
                "c_hash": c_h,
            }),
            children=children,
            metadata={
                "rule_count": len(model.rules),
                "constraint_count": len(model.constraints),
                "n_structure_nodes": len(model.structure.nodes),
            },
        )

    def hash_meme(self, meme: MemeState, idx: int) -> HierarchicalHash:
        """对单个 MemeState 计算分层哈希.

        数学对应: 定义4+前提5 — X_i = (m_i, ξ_i).
        将模因的 5 维核心向量 m_i = [D, B, ρ, R, S] 和扩展维度 ξ_i
        分别序列化为 JSON, 计算各自的 sha256 摘要作为子节点,
        然后合并为顶层哈希.

        5d_json = json.dumps({"D": meme.D, "B": meme.B,
                              "rho": meme.rho, "R": meme.R, "S": meme.S})
        xi_json  = json.dumps(meme.xi.tolist())
        children = [sha256(5d_json), sha256(xi_json)]
        metadata 含 {idx, D, B, rho, R, S, xi_dim}

        Args:
            meme: MemeState 对象 (模块3输出, 定义4).
            idx:  模因在集合中的索引位置.

        Returns:
            HierarchicalHash 节点, layer="meme", component_id="meme_{idx}".
        """
        # ---- 步骤1: 5 维核心向量快照 ----
        # 数学对应: Ω = [0,1]^5, 核心向量 m_i = [D, B, ρ, R, S]
        five_d_data: Dict[str, float] = {
            "D": meme.D,
            "B": meme.B,
            "rho": meme.rho,
            "R": meme.R,
            "S": meme.S,
        }
        five_d_json: str = _safe_json(five_d_data)
        five_d_hash: str = sha256(five_d_json)

        # ---- 步骤2: 扩展维度 ξ_i 快照 ----
        # 数学对应: ξ_i ∈ Ξ, 微观涨落编码
        xi_list: list = meme.xi.tolist()
        xi_json: str = _safe_json(xi_list)
        xi_hash: str = sha256(xi_json)

        # ---- 步骤3: 合并子哈希 ----
        children: List[str] = [five_d_hash, xi_hash]
        xi_dim: int = len(meme.xi)
        return self.hash_component(
            layer="meme",
            component_id=f"meme_{idx}",
            canonical_data=five_d_json,
            children=children,
            metadata={
                "idx": idx,
                "D": meme.D,
                "B": meme.B,
                "rho": meme.rho,
                "R": meme.R,
                "S": meme.S,
                "xi_dim": xi_dim,
            },
        )

    def hash_composite(
        self, Theta: List[Dict[str, float]], C: np.ndarray
    ) -> HierarchicalHash:
        """对复合模因状态 (Θ, C) 计算分层哈希.

        数学对应: 定义4 — Q = {X_1, ..., X_n, Θ, C}.
        Θ 是每个模因的 11 个动力学参数列表,
        C 是 n×n 耦合矩阵 (对称, 对角线=0).
        分别计算 Θ 和 C 的子哈希, 合并为顶层复合哈希.

        Theta_json = json.dumps(Theta)
        C_json     = json.dumps(C.tolist())
        children   = [sha256(Theta_json), sha256(C_json)]

        Args:
            Theta: 动力学参数列表, 每个元素是一个 Dict[str, float].
            C:     耦合矩阵 n×n, C_{ij} ∈ [0,1], 对称, 对角线=0.

        Returns:
            HierarchicalHash 节点, layer="composite",
            component_id="composite_state".
        """
        # ---- 步骤1: 动力学参数 Θ 快照 ----
        Theta_json: str = _safe_json(Theta)
        Theta_hash: str = sha256(Theta_json)

        # ---- 步骤2: 耦合矩阵 C 快照 ----
        C_list: list = C.tolist()
        C_json: str = _safe_json(C_list)
        C_hash: str = sha256(C_json)

        # ---- 步骤3: 合并子哈希 ----
        children: List[str] = [Theta_hash, C_hash]
        n: int = len(Theta)
        return self.hash_component(
            layer="composite",
            component_id="composite_state",
            canonical_data=Theta_json,
            children=children,
            metadata={
                "n_memes": n,
                "coupling_size": C.shape[0],
            },
        )

    # ----------------------------------------------------------------
    # 通用哈希工厂
    # ----------------------------------------------------------------

    def hash_component(
        self,
        layer: str,
        component_id: str,
        canonical_data: str,
        children: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HierarchicalHash:
        """通用哈希工厂 — 构建分层哈希节点的统一入口.

        数学对应: 分层哈希节点的构造原语.
        当 children 非空时, 合并子节点哈希值后计算顶层摘要;
        当 children 为空时, 直接对 canonical_data 计算摘要.
        此设计保证了哈希树中叶节点与内部节点的一致性.

        算法:
          if children:
              combined = "|".join(children)
              hash_value = sha256(combined)
          else:
              hash_value = sha256(canonical_data)

        Args:
            layer:          层级名称 (token|hierarchy|relation|math_model|...).
            component_id:   组件唯一标识符.
            canonical_data: 规范化 JSON 快照 (用于重新计算验证).
            children:       子节点的 hash_value 列表, 默认为空列表.
            metadata:       附带元数据字典, 默认为空字典.

        Returns:
            完整的 HierarchicalHash 节点.
        """
        _children: List[str] = children if children is not None else []
        _metadata: Dict[str, Any] = metadata if metadata is not None else {}

        # ---- 计算哈希值 ----
        # 数学对应: H(node) = sha256(canonical_data) 当无子节点时
        #          H(node) = sha256(children[0] | "|" | ... | "|" | children[k]) 当有子节点时
        if _children:
            combined: str = "|".join(_children)
            hash_value: str = sha256(combined)
        else:
            hash_value = sha256(canonical_data)

        # ---- 构建分层哈希节点 ----
        return HierarchicalHash(
            layer=layer,
            component_id=component_id,
            hash_value=hash_value,
            canonical_json_snapshot=canonical_data,
            children=_children,
            metadata=_metadata,
        )

    # ----------------------------------------------------------------
    # 几何对象哈希 (模块2 对应)
    # ----------------------------------------------------------------

    def hash_geo_object(self, geo_obj: Any) -> HierarchicalHash:
        """对 GeometricObject 计算分层哈希.

        数学对应: 定义3 — G = (K, g, ω, Γ, R). Φ_C: M → G (定理3: 双射).
        规范化快照编码几何对象的结构性特征:
        K (胞腔复形的顶点数/边数), g 和 ω 的维度,
        Gamma (几何不变量), R (可逆性元数据).

        此方法归于 sub_geo 层, 表示几何对象是模块2的输出组件.

        Args:
            geo_obj: GeometricObject 对象 (模块2输出, 定义3).

        Returns:
            HierarchicalHash 节点, layer="sub_geo",
            component_id="geometric_object".
        """
        canonical: str = _safe_json({
            "n_vertices": len(geo_obj.K.vertices),
            "n_edges": len(geo_obj.K.edges),
            "n_higher_cells": len(geo_obj.K.higher_cells),
            "g_shape": list(geo_obj.g.shape),
            "omega_shape": list(geo_obj.omega.shape),
            "Gamma_euler_char": geo_obj.Gamma.get("euler_char", None),
            "n_subcomplexes": len(geo_obj.K.subcomplexes),
        })
        h: str = sha256(canonical)
        return HierarchicalHash(
            layer="sub_geo",
            component_id="geometric_object",
            hash_value=h,
            canonical_json_snapshot=canonical,
            children=[],
            metadata={
                "n_vertices": len(geo_obj.K.vertices),
                "n_edges": len(geo_obj.K.edges),
                "euler_char": geo_obj.Gamma.get("euler_char", None),
            },
        )
