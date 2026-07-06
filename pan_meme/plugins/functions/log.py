# 数学对应: 设计文档第9节 — 附录D.3 对数函数族
# LogFunction: f(x) = ln(1 + k*x)
#   - derivative:  f'(x) = k / (1 + k*x)
#   - lipschitz:   L = |k|  (在 x ≥ 0 上, sup|f'| = |k|)
#   - n_params:    1  (k)

import numpy as np

from pan_meme.plugins.functions.base import BaseFunction
from pan_meme.plugins.registry import PluginRegistry


class LogFunction(BaseFunction):
    """
    对数函数族: f(x) = ln(1 + k*x).

    数学对应: 附录D.3 — F_log = {x ↦ ln(1 + k*x) | k > 0}
    对数型增长: 初期增长快, 后期趋于平缓.
    - evaluate(x)    = ln(1 + k*x)
    - derivative(x)  = k / (1 + k*x)
    - lipschitz_constant = |k|
    - n_params = 1
    """

    def __init__(self, k: float = 1.0) -> None:
        """
        初始化对数函数.

        Args:
            k:  缩放参数, 默认 1.0. 必须 k > 0 以保证 x≥0 时定义.
                k 越大则初期增长越快, 饱和也越早.
        """
        self.k: float = float(k)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        计算 f(x) = ln(1 + k*x).

        数学对应: 附录D.3 — 对数增长映射, f(0)=0.

        Args:
            x:  输入向量, shape=(N,), dtype=np.float64

        Returns:
            ln(1 + k*x), shape=(N,)
        """
        return np.log1p(self.k * x.astype(np.float64))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        计算 f'(x) = k / (1 + k*x).

        数学对应: 附录D.3 — 对数导数为双曲线衰减.

        Args:
            x:  输入向量, shape=(N,), dtype=np.float64

        Returns:
            k / (1 + k*x), shape=(N,)
        """
        return self.k / (1.0 + self.k * x.astype(np.float64))

    @property
    def lipschitz_constant(self) -> float:
        """
        Lipschitz 常数 L = |k|.

        数学对应: 附录D.3.1 — ∥·∥_U 度量结构.
        在 x ≥ 0 上, f'(x) = k/(1+kx) 单调递减,
        最大值在 x=0 处为 |k|.

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
PluginRegistry.register_function("log", LogFunction)
