# 泛模因几何工具 — 核心类型系统
# 对应论文: 附录D.1-D.2 公理+定义, 附录D.5 状态空间
# 每个 dataclass 标注其数学对应实体

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Callable
from enum import Enum
import numpy as np


# ============================================================
# 共享枚举
# ============================================================

class ConvergenceType(str, Enum):
    """定理10: 三类收敛原型"""
    STONE = "stone"           # 基石型: 高D, 低R, 高S — 长期稳定
    TRANSIENT = "transient"   # 过客型: 极高R短暂期后衰退
    BUBBLE = "bubble"         # 泡沫型: 极高R极高B后崩溃
    UNDETERMINED = "undetermined"


# ============================================================
# 模块1: 输入适配层 (公理1-3 + 定义1-2)
# ============================================================

@dataclass
class Token:
    """信息论域 U 的原子元素. 对应公理2: 每个 x ∈ U"""
    modality: str                              # 模态标识: text|image|audio|video|code|structured
    text: str                                  # 文本表示 (所有模态共享的字符串形式)
    span: Tuple[int, int]                      # 在原输入中的定位 (字节偏移/帧号/时间戳)
    pos: str                                   # 词性/类型标注
    embedding: Optional[np.ndarray] = None     # 语义向量 ∈ R^d (可选)
    payload: Dict[str, Any] = field(default_factory=dict)
    # payload 示例: 图像→patch_rgba,bbox; 音频→waveform_slice,pitch; 代码→ast_node_type


@dataclass
class HierarchyNode:
    """公理2+3: 结构从属关系中的节点. parent=None ⟹ 未归类元素 (P(x)=False)"""
    token_idx: int                             # 引用 Token 列表中的索引
    children: List[int] = field(default_factory=list)    # 向下展开: C(y) = {x₁,...,xₙ}
    parent: Optional[int] = None               # 向上归类: x 从属于 y


@dataclass
class HierarchyTree:
    """公理2+3的输出: 多层级结构骨架"""
    nodes: List[HierarchyNode]                 # 节点列表, 索引对应 token
    root_indices: List[int] = field(default_factory=list)   # 未归类顶层元素
    depth: int = 0                             # 树最大深度
    rounds: int = 0                            # 循环执行轮数
    terminated_by: str = "initialized"         # "fixed" | "converge"
    termination_record: Dict[str, Any] = field(default_factory=dict)
    # termination_record = {mode, reason, final_level, total_rounds, converged_at_round}


@dataclass
class RelationNetwork:
    """定义1: Ψ = (V, E, w). A(I) = Ψ ∈ R_Ψ. 前提2: A 信息无损"""
    nodes: List[str]                           # V: 节点标识符
    edges: List[Tuple[int, int]]               # E ⊆ V×V: 边 (node_index, node_index)
    weights: np.ndarray                        # w: E → [0,1]: 连接强度, shape=(|E|,)
    hierarchy: Dict[str, Any] = field(default_factory=dict)
    # hierarchy = {levels, node_levels: Dict[int,int], combination_path, parent_map}
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata = {input_type, node_count, edge_count, threshold_used, inference_applied}


@dataclass
class GapInfo:
    """彻底不完整判定中发现的空洞"""
    location: str                              # 缺失位置描述
    gap_type: str                              # 空洞类型: structural_hole|isolated_nodes|flat_structure
    suggested_direction: str                   # 建议补全方向
    context: Dict[str, Any] = field(default_factory=dict)  # {largest_component_ratio, isolated_ratio, depth}


@dataclass
class CompletenessReport:
    """前提1+前提0的运行时验证结果"""
    is_complete: bool                          # 结构是否完整
    gaps: List[GapInfo] = field(default_factory=list)
    requires_human: bool = False               # 是否需要人工介入
    human_request: Optional[str] = None        # 人工补全请求文本


@dataclass
class RuleDef:
    """定义2: (f, supp) ∈ F — 规则域元素. 前提3: 由Ψ结构唯一确定"""
    pattern: str                               # 规则模式描述
    support: List[int]                         # supp ⊆ V∪E: 支撑集 (节点/边索引)
    confidence: float = 1.0                    # 置信度 ∈ [0,1]
    source: str = "extracted"                  # "extracted" | "inferred" | "decoded"


@dataclass
class ConstraintDef:
    """定义2: (c, dom) ∈ C — 约束域元素"""
    condition: str                             # 约束条件描述
    domain: List[int]                          # dom ⊆ V∪E: 作用域
    description: str = ""                      # 详细说明
    confidence: float = 1.0
    source: str = "extracted"


@dataclass
class MathModel:
    """定义2: M = (S, F, C). Φ_B: Ψ ↦ M (定理2: 双射)"""
    structure: 'RelationNetwork'               # S: 结构域
    rules: List[RuleDef]                       # F: 规则域
    constraints: List[ConstraintDef]           # C: 约束域
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata = {rule_count, constraint_count, consistency_verified, extraction_timestamp}


# ============================================================
# 模块2: 几何化层 (定义3)
# ============================================================

@dataclass
class SimplicialComplex:
    """胞腔复形 K: 定义3的第一个组件. 图→胞腔复形同构映射"""
    vertices: List[int]                        # 0-单形: 节点ID列表
    edges: List[Tuple[int, int]]               # 1-单形: 边列表
    higher_cells: List[List[int]] = field(default_factory=list)  # 高阶胞腔 (2-单形及以上)
    subcomplexes: Dict[str, List[int]] = field(default_factory=dict)  # 子复形: {名称→顶点集}
    level_labels: Dict[int, int] = field(default_factory=dict)        # 顶点→层级编号


@dataclass
class GeometricObject:
    """定义3: G = (K, g, ω, Γ, R). Φ_C: M ↦ G (定理3: 双射)"""
    K: 'SimplicialComplex'                     # 胞腔复形
    g: np.ndarray                              # 度量结构: 每个1-cell 的长度 = w(e_i)
    omega: np.ndarray                          # 微分结构/向量场: 编码规则域 F
    Gamma: Dict[str, Any]                      # 几何不变量: 编码约束域 C {euler_char, num_constraints, constraint_types}
    R: Dict[str, Any]                          # 可逆性元数据包: {vertex_map, edge_map, level_map, construction_log}


# ============================================================
# 模块3: 模因化层 (定义4 + 前提5)
# ============================================================

@dataclass
class MemeState:
    """单个模因状态 X_i = (m_i, ξ_i). 前提5: 核心+扩展分离, 映射为双射"""
    D: float                                   # m_i[0]: 内禀度 ∈ [0,1] — 结构复杂性与自洽性
    B: float                                   # m_i[1]: 关联度 ∈ [0,1] — 外部连接广度
    rho: float                                 # m_i[2]: 能流密度 ∈ [0,1] — 能量/信息流强度
    R: float                                   # m_i[3]: 演化速率 ∈ [0,1] — 扩散瞬时速度
    S: float                                   # m_i[4]: 结构韧度 ∈ [0,1] — 抗扰动能力
    xi: np.ndarray                             # ξ_i ∈ Ξ: 扩展维度 — 微观涨落编码 (shape=(|E_i|,))
    # 完整 5d 向量: np.array([D, B, rho, R, S]) ∈ Ω = [0,1]^5 (定义6)


@dataclass
class CompositeMemeState:
    """定义4: Q = {X₁, X₂, ..., Xₙ, Θ, C}. Φ_D: G ↦ Q (定理4: 双射)"""
    memes: List[MemeState]                     # {X_i}: 模因状态列表
    Theta: List[Dict[str, float]]              # Θ: 每个模因的11个动力学参数
    C: np.ndarray                              # 耦合矩阵 n×n, C_{ij} ∈ [0,1] (对称, 对角线=0)


# ============================================================
# 模块4: 分层哈希凭证 (论文3.5节)
# ============================================================

@dataclass
class HierarchicalHash:
    """分层哈希节点 — 每一层级每一组件的独立哈希"""
    layer: str                                 # 层级: token|hierarchy|relation|math_model|geo_object|sub_geo|meme|composite
    component_id: str                          # 组件唯一标识
    hash_value: str                            # SHA-256 十六进制摘要
    canonical_json_snapshot: str               # 规范化 JSON — 用于重新计算验证
    children: List[str] = field(default_factory=list)  # 子节点 hash_value 列表
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附带元数据 {timestamp, size, threshold, ...}


@dataclass
class MerkleTree:
    """稀疏 Merkle 树 — 分层完整性验证"""
    root_hash: str                             # 根哈希 = H(mod1_hash | mod2_hash | mod3_hash)
    nodes: Dict[str, HierarchicalHash]         # hash_value → 节点 (含叶子和内部聚合节点)
    leaf_index: Dict[str, List[str]]           # layer名 → 该层所有叶子 hash_value 列表


@dataclass
class Credential:
    """凭证: 分层哈希树 + 模因状态 + 元数据"""
    header: Dict[str, Any]                     # {version, timestamp, hash_algorithm, merkle_layers}
    data_hash: str                             # = merkle_tree.root_hash (顶层根哈希)
    merkle_tree: MerkleTree                    # 完整分层哈希树
    meme_state: CompositeMemeState             # 复合模因状态 Q
    metadata: Dict[str, Any]                   # {original_size_bytes, meme_count, original_type}


# ============================================================
# 管线统一数据类型
# ============================================================

@dataclass
class PipelineData:
    """管线程上下文 — 贯穿所有4个模块. 对应附录 D.2 完整数据流"""
    input: Any                                 # 原始信息 I ∈ U (公理1)
    # 模块1产物 (定理1+2)
    psi: Optional[RelationNetwork] = None      # Ψ: 关系网络 (定义1)
    math_model: Optional[MathModel] = None     # M: 数学模型 (定义2)
    # 模块2产物 (定理3)
    geo_object: Optional[GeometricObject] = None  # G: 几何对象 (定义3)
    # 模块3产物 (定理4)
    meme_state: Optional[CompositeMemeState] = None  # Q: 复合模因状态 (定义4)
    # 模块4产物
    credential: Optional[Credential] = None    # 凭证
    merkle_tree: Optional[MerkleTree] = None   # Merkle树引用
    # 哈希累积
    all_hash_nodes: List[HierarchicalHash] = field(default_factory=list)
    # 运行时元数据
    meta: Dict[str, Any] = field(default_factory=dict)
    # meta = {incomplete, completeness_report, optimization_result, errors, warnings}


# ============================================================
# ODE求解器 (附录D.5)
# ============================================================

@dataclass
class ODEConfig:
    """ODE求解器配置. 对应定理6 (分段解) + 定理7 (Ω不变集)"""
    method: str = "RK45"                       # 求解方法: RK45|Radau|BDF|DOP853
    atol: float = 1e-8                         # 绝对容差
    rtol: float = 1e-8                         # 相对容差
    max_step: float = 0.1                      # 最大步长 (不超过 1/max(δ₃, ε₂))
    t_span: Tuple[float, float] = (0.0, 100.0) # 积分时间区间
    t_eval: Optional[np.ndarray] = None        # 评估时间点 (None=自适应)


@dataclass
class ODEResult:
    """ODE求解结果. M 的形状: (len(t), 5) — 每行 [D, B, ρ, R, S]"""
    t: np.ndarray                              # 时间序列, shape=(N,)
    M: np.ndarray                              # 状态矩阵, shape=(N, 5)
    n_history: List[int] = field(default_factory=list)    # 模因数量历史
    jump_points: List[float] = field(default_factory=list) # 跳变时间点
    convergence_type: str = "undetermined"     # 收敛类型 (定理10)


# ============================================================
# 全局优化器 (附录D.3 假设0)
# ============================================================

@dataclass
class OptimizationResult:
    """假设0的优化结果: h* = argmin L(h; I)"""
    h_star: Dict[str, Any]                     # 最优配置 {T, func_family, Theta, n}
    loss: float                                # 最优损失值 L(h*; I)
    candidates_tested: int                     # 评估的候选组合数
