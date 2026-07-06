# 数学对应: 设计文档第9节 — 附录D.3 候选函数族基类
# BaseFunction 是所有非线性函数族的抽象基类.
# 每个子类代表 F 的一个子集: F = {F_power, F_exp, F_sigmoid, F_log, F_piecewise}
# 在全局优化 (假设0) 中, H = T × F × Θ × N, 其中 F 是这些函数族的并集.

from abc import ABC, abstractmethod

import numpy as np


class BaseFunction(ABC):
    """
    所有候选非线性函数的抽象基类.

    数学对应: 附录D.3 — F = {幂, 指数, Sigmoid, 对数, 分段线性}
    每个具体实现提供:
      - evaluate(x):           函数值 f(x)
      - derivative(x):         导数值 f'(x)
      - lipschitz_constant:    Lipschitz 常数 L, 满足 |f(a)-f(b)| ≤ L·|a-b|
      - n_params:              可调参数个数 (用于优化器搜索空间维度)
    """

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        计算函数值 f(x).

        数学对应: 附录D.3 — f: R → R, 逐元素作用于向量 x.

        Args:
            x:  输入向量, shape=(N,), dtype=np.float64

        Returns:
            f(x), shape 与 x 相同
        """
        ...

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        计算导数值 f'(x).

        数学对应: 附录D.3 — 用于 ODE 求解器 (定理6) 的 Jacobian 计算.

        Args:
            x:  输入向量, shape=(N,), dtype=np.float64

        Returns:
            f'(x), shape 与 x 相同
        """
        ...

    @property
    @abstractmethod
    def lipschitz_constant(self) -> float:
        """
        Lipschitz 常数 L.

        数学对应: 附录D.3.1 — ∥·∥_U 度量结构的上界.
        L = sup_{x≠y} |f(x)-f(y)| / |x-y|.
        用于 ODE 步长控制和收敛分析 (定理8).

        Returns:
            L ≥ 0, float
        """
        ...

    @property
    @abstractmethod
    def n_params(self) -> int:
        """
        可调参数的个数.

        数学对应: 附录D.3 假设0 — 搜索空间 H = T × F × Θ × N.
        n_params 决定该函数族在优化中的参数维度.
        例如: PowerFunction 有 1 个参数 (k), SigmoidFunction 有 2 个 (k, x0).

        Returns:
            参数数量, int ≥ 1
        """
        ...
