# 数学对应: 设计文档第9节 — 附录D.3 指数函数族
# ExpFunction: f(x) = exp(k*x) - 1
#   - derivative:  f'(x) = k * exp(k*x)
#   - lipschitz:   L = |k| * exp(|k|)  (在 [0,1] 上的上界)
#   - n_params:    1  (k)

import numpy as np

from pan_meme.plugins.functions.base import BaseFunction
from pan_meme.plugins.registry import PluginRegistry


class ExpFunction(BaseFunction):
    """
    指数函数族: f(x) = exp(k*x) - 1.

    数学对应: 附录D.3 — F_exp = {x ↦ exp(k*x) - 1 | k ∈ R}
    平移 -1 确保 f(0) = 0, 保持原点不变.
    - evaluate(x)    = exp(k*x) - 1
    - derivative(x)  = k * exp(k*x)
    - lipschitz_constant = |k| * exp(|k|)
    - n_params = 1
    """

    def __init__(self, k: float = 1.0) -> None:
        """
        初始化指数函数.

        Args:
            k:  增长速率参数, 默认 1.0.
                k>0 为增长; k<0 为衰减; |k| 越大则非线性越强.
        """
        self.k: float = float(k)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        计算 f(x) = exp(k*x) - 1.

        数学对应: 附录D.3 — 指数增长/衰减映射, 在 x=0 处归零.

        Args:
            x:  输入向量, shape=(N,), dtype=np.float64

        Returns:
            exp(k*x) - 1, shape=(N,)
        """
        return np.exp(self.k * x.astype(np.float64)) - 1.0

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        计算 f'(x) = k * exp(k*x).

        Args:
            x:  输入向量, shape=(N,), dtype=np.float64

        Returns:
            k * exp(k*x), shape=(N,)
        """
        return self.k * np.exp(self.k * x.astype(np.float64))

    @property
    def lipschitz_constant(self) -> float:
        """
        Lipschitz 常数 L = |k| * exp(|k|).

        数学对应: 附录D.3.1 — ∥·∥_U 度量结构.
        在 x ∈ [0,1] 上, sup|f'(x)|:
          - 若 k > 0: sup = k * exp(k)     (在 x=1 处)
          - 若 k < 0: sup = |k| * exp(0) = |k|  (在 x=0 处)
        统一上界: |k| * exp(|k|)

        Returns:
            |k| * exp(|k|), float
        """
        return abs(self.k) * np.exp(abs(self.k))

    @property
    def n_params(self) -> int:
        """
        可调参数个数 = 1 (仅 k).

        Returns:
            1, int
        """
        return 1


# ---- 自动注册到 PluginRegistry ----
PluginRegistry.register_function("exp", ExpFunction)
