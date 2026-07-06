# 数学对应: 设计文档第9节 — 附录D.3 幂函数族
# PowerFunction: f(x) = x^k
#   - derivative:  f'(x) = k * x^(k-1)
#   - lipschitz:   L = |k|  (在 [0,1] 上, sup|f'| = |k|)
#   - n_params:    1  (k)

import numpy as np

from pan_meme.plugins.functions.base import BaseFunction
from pan_meme.plugins.registry import PluginRegistry


class PowerFunction(BaseFunction):
    """
    幂函数族: f(x) = x^k.

    数学对应: 附录D.3 — F_power = {x ↦ x^k | k ∈ R⁺}
    - evaluate(x) = x^k
    - derivative(x) = k * x^(k-1)
    - lipschitz_constant = |k|
    - n_params = 1
    """

    def __init__(self, k: float = 2.0) -> None:
        """
        初始化幂函数.

        Args:
            k:  指数参数, 默认 2.0. 控制非线性强度.
                k=1 退化为线性; k>1 为超线性; 0<k<1 为次线性.
        """
        self.k: float = float(k)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        计算 f(x) = x^k.

        数学对应: 定义在 [0,1] 上的幂函数, 确保 f(0)=0, f(1)=1.

        Args:
            x:  输入向量, shape=(N,), dtype=np.float64

        Returns:
            x^k, shape=(N,)
        """
        return np.power(x.astype(np.float64), self.k)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        计算 f'(x) = k * x^(k-1).

        Args:
            x:  输入向量, shape=(N,), dtype=np.float64

        Returns:
            k * x^(k-1), shape=(N,)
        """
        return self.k * np.power(x.astype(np.float64), self.k - 1.0)

    @property
    def lipschitz_constant(self) -> float:
        """
        Lipschitz 常数 L = |k|.

        数学对应: 在 x ∈ [0,1] 上, sup|f'(x)| = |k| (当 k≥0 时最大值在 x=1 处).
        用于 ODE 步长约束: max_step ≤ 1 / max(δ₃, ε₂, L).

        Returns:
            |k|, float
        """
        return abs(self.k)

    @property
    def n_params(self) -> int:
        """
        可调参数个数 = 1 (仅 k).

        Returns:
            1, int
        """
        return 1


# ---- 自动注册到 PluginRegistry ----
# 数学对应: 设计文档第9节 — 函数族自动注册, 使优化器可通过名称查找
PluginRegistry.register_function("power", PowerFunction)
