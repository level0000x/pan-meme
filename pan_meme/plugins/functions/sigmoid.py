# 数学对应: 设计文档第9节 — 附录D.3 Sigmoid 函数族
# SigmoidFunction: f(x) = 1/(1+exp(-k*(x-x0))) - 0.5
#   - derivative:  f'(x) = k * s(x) * (1 - s(x)), 其中 s(x) = sigmoid(k*(x-x0))
#   - lipschitz:   L = |k|/4  (sigmoid 导数的全局最大值)
#   - n_params:    2  (k, x0)

import numpy as np

from pan_meme.plugins.functions.base import BaseFunction
from pan_meme.plugins.registry import PluginRegistry


class SigmoidFunction(BaseFunction):
    """
    Sigmoid 函数族: f(x) = sigmoid(k*(x-x0)) - 0.5.

    数学对应: 附录D.3 — F_sigmoid = {x ↦ σ(k*(x-x0)) - 0.5 | k∈R, x0∈[0,1]}
    平移 -0.5 使输出范围为 [-0.5, 0.5], 关于原点中心对称.
    - evaluate(x)    = 1/(1+exp(-k*(x-x0))) - 0.5
    - derivative(x)  = k * s * (1-s),  其中 s = sigmoid(k*(x-x0))
    - lipschitz_constant = |k|/4
    - n_params = 2
    """

    def __init__(self, k: float = 5.0, x0: float = 0.5) -> None:
        """
        初始化 Sigmoid 函数.

        Args:
            k:   陡峭度参数, 默认 5.0.
                 |k| 越大则过渡越陡峭; k>0 单调增, k<0 单调减.
            x0:  中心偏移参数, 默认 0.5 ∈ [0,1].
                 决定 sigmoid 跃迁发生的位置.
        """
        self.k: float = float(k)
        self.x0: float = float(x0)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        计算 f(x) = sigmoid(k*(x-x0)) - 0.5.

        数学对应: 附录D.3 — 平滑阶跃函数, 模拟阈值激活行为.

        Args:
            x:  输入向量, shape=(N,), dtype=np.float64

        Returns:
            σ(k*(x-x0)) - 0.5, shape=(N,)
        """
        s: np.ndarray = 1.0 / (1.0 + np.exp(-self.k * (x.astype(np.float64) - self.x0)))
        return s - 0.5

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        计算 f'(x) = k * s(x) * (1 - s(x)).

        数学对应: 附录D.3 — sigmoid 导数为钟形曲线, 在 x=x0 处取最大值.

        Args:
            x:  输入向量, shape=(N,), dtype=np.float64

        Returns:
            k * s * (1-s), shape=(N,)
        """
        s: np.ndarray = 1.0 / (1.0 + np.exp(-self.k * (x.astype(np.float64) - self.x0)))
        return self.k * s * (1.0 - s)

    @property
    def lipschitz_constant(self) -> float:
        """
        Lipschitz 常数 L = |k|/4.

        数学对应: 附录D.3.1 — ∥·∥_U 度量结构.
        sigmoid 导数 s*(1-s) 的全局最大值为 1/4,
        故 f'(x) = k * s * (1-s) 的最大值为 |k|/4.

        Returns:
            |k|/4, float
        """
        return abs(self.k) / 4.0

    @property
    def n_params(self) -> int:
        """
        可调参数个数 = 2 (k, x0).

        Returns:
            2, int
        """
        return 2


# ---- 自动注册到 PluginRegistry ----
PluginRegistry.register_function("sigmoid", SigmoidFunction)
