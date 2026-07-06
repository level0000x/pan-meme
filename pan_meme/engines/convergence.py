"""收敛性分析 — 定理10完整论证 + LaSalle不变原理
数学对应: 论文4.5节(动力学收敛性分析) + 辅助函数
"""

import numpy as np
from typing import Dict
from pan_meme.core.types import ODEResult


def energy_function(M: np.ndarray) -> np.ndarray:
    """论文4.5.3: V(M) = 0.5 * sum(D^2 + B^2 + rho^2 + R^2 + S^2)"""
    return 0.5 * np.sum(M ** 2, axis=1)


def classify_prototype(final_state: np.ndarray) -> str:
    """定理10: 收敛原型分类 (与ODESolver._classify一致)"""
    D, B, _, R, S = final_state
    if D > 0.5 and R < 0.3 and S > 0.5: return "stone"
    if R < 0.1 and B < 0.3: return "transient"
    if R < 0.1 and B > 0.7: return "bubble"
    return "undetermined"


def prototype_counts(results: list) -> Dict[str, int]:
    """统计各原型数量"""
    counts = {"stone": 0, "transient": 0, "bubble": 0, "undetermined": 0}
    for r in results:
        ct = r.convergence_type if isinstance(r, ODEResult) else classify_prototype(r[-1])
        counts[ct] += 1
    return counts
