"""泛模因几何工具 — 引擎模块

本模块包含 PB 级算法引擎的实现，每个引擎遵循统一的数学对应规范：
  - 所有方法标注数学对应（PB_ARCHITECTURE.md 章节号）
  - 中文注释 + 数学公式标注
  - 完整类型标注

导出：
  # 模因动力学引擎
  ODESolver      — 五维动力学系统数值积分（论文4.3.1）
  energy_function — 能量函数 V(M)（论文4.5.3）
  classify_prototype — 收敛原型分类（定理10）
  prototype_counts  — 收敛原型统计

  # 近似关系提取引擎（PB_ARCHITECTURE.md 第3.1节）
  MinHashLSH           — MinHash 局部敏感哈希索引
  WindowCooccurrence   — 滑动窗口共现提取器
  LSHRelationExtractor — LSH 加速的关系网络提取器

  # 传递闭包引擎（PB_ARCHITECTURE.md 第3.2节）
  StarClosure — Semi-naive 传递闭包引擎（稀疏矩阵版）

  # 全局优化引擎（PB_ARCHITECTURE.md 第3.3节）
  GPOptimizer          — GP贝叶斯全局优化器
  GaussianProcessModel — 底层GP解析实现 (RBF核)
  GlobalOptimizer      — 全局优化器 (穷举/GP/多保真度)

  # Mapper 持久同调 β₀ 估计引擎（PB_ARCHITECTURE.md 第3.4节）
  MapperHomology — Mapper 算法在线 β₀ 估计
  PregelCC       — 分布式连通分量算法 (单机原型)
"""

# ── 模因动力学引擎 ──
from pan_meme.engines.ode_solver import ODESolver
from pan_meme.engines.convergence import (
    energy_function,
    classify_prototype,
    prototype_counts,
)

# ── 近似关系提取引擎（PB_ARCHITECTURE.md 第3.1节） ──
from pan_meme.engines.minhash_lsh import (
    MinHashLSH,
    WindowCooccurrence,
    LSHRelationExtractor,
)

# ── 传递闭包引擎（PB_ARCHITECTURE.md 第3.2节） ──
from pan_meme.engines.star_closure import (
    StarClosure,
)

# ── 全局优化引擎（PB_ARCHITECTURE.md 第3.3节） ──
from pan_meme.engines.gp_optimizer import (
    GPOptimizer,
    GaussianProcessModel,
)
from pan_meme.engines.optimizer import (
    GlobalOptimizer,
)

# ── Mapper 持久同调 β₀ 估计引擎（PB_ARCHITECTURE.md 第3.4节） ──
from pan_meme.engines.mapper_homology import (
    MapperHomology,
    PregelCC,
)

__all__ = [
    # ode_solver
    "ODESolver",
    # convergence
    "energy_function",
    "classify_prototype",
    "prototype_counts",
    # minhash_lsh (第3.1节)
    "MinHashLSH",
    "WindowCooccurrence",
    "LSHRelationExtractor",
    # star_closure (第3.2节)
    "StarClosure",
    # gp_optimizer (第3.3节)
    "GPOptimizer",
    "GaussianProcessModel",
    # optimizer (第3.3节)
    "GlobalOptimizer",
    # mapper_homology (第3.4节)
    "MapperHomology",
    "PregelCC",
]
