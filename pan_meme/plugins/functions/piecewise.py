# 数学对应: 设计文档第9节 — 附录D.3 分段线性函数族
# PiecewiseLinearFunction: 用一组断点定义的分段线性插值函数
#   - evaluate:      对每个 x_i 找到所在区间, 执行线性插值
#   - derivative:    在每个区间内为常数 = 1/(n_segments * avg_segment_width)
#   - lipschitz:     L = 1.0  (归一化后)
#   - n_params:      len(breakpoints)  (断点个数)

import numpy as np

from pan_meme.plugins.functions.base import BaseFunction
from pan_meme.plugins.registry import PluginRegistry


class PiecewiseLinearFunction(BaseFunction):
    """
    分段线性函数族: 用一组断点定义的分段线性插值.

    数学对应: 附录D.3 — F_pwl = {x ↦ PWL(x; {b_j}) | b_j ∈ (0,1)}
    在每个区间 [b_j, b_{j+1}] 上为线性, 全局连续但导数分段常数.
    - evaluate(x)    = 分段线性插值结果
    - derivative(x)  = 1 / (n_segments * avg_segment_width)  (近似常梯度)
    - lipschitz_constant = 1.0
    - n_params = len(breakpoints)

    断点定义: 默认 [0.25, 0.5, 0.75] 将 [0,1] 划分为 4 段.
    在每个区间上, 函数值为 y_j + slope * (x - x_j), 其中 y 值在断点处固定.
    """

    def __init__(self, breakpoints: list = None) -> None:
        """
        初始化分段线性函数.

        Args:
            breakpoints:  断点列表, 默认 [0.25, 0.5, 0.75].
                          所有断点必须在 (0, 1) 区间内且严格递增.
        """
        if breakpoints is None:
            breakpoints = [0.25, 0.5, 0.75]
        # 排序并确保 np.float64 类型
        self.breakpoints: np.ndarray = np.array(sorted(breakpoints), dtype=np.float64)

        # 构建完整网格: [0] + breakpoints + [1]
        # 对应的 y 值: 使用等比缩放, 使 y_j = j / (n_segments)
        self._full_x: np.ndarray = np.concatenate([
            np.array([0.0], dtype=np.float64),
            self.breakpoints,
            np.array([1.0], dtype=np.float64),
        ])

        n_segments: int = len(self._full_x) - 1  # 区间个数
        # y 值均匀分布在 [0, 1], 对应 x 的分段节点
        self._full_y: np.ndarray = np.linspace(0.0, 1.0, n_segments + 1, dtype=np.float64)

        # 预计算每段斜率: slope_j = (y_{j+1} - y_j) / (x_{j+1} - x_j)
        dx: np.ndarray = np.diff(self._full_x)
        dy: np.ndarray = np.diff(self._full_y)
        self._slopes: np.ndarray = dy / np.maximum(dx, 1e-12)  # 防止除以零

        # 缓存段数用于导数计算
        self._n_segments: int = n_segments
        self._avg_segment_width: float = float(np.mean(dx))

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        计算分段线性插值 f(x).

        数学对应: 附录D.3 — PWL(x) 为在 [0,1] 上关于断点的线性插值.
        对每个输入 x_i:
          1. 找到 x_i 所在的区间 [full_x[j], full_x[j+1]]
          2. 返回 y_j + slope_j * (x_i - full_x[j])

        Args:
            x:  输入向量, shape=(N,), dtype=np.float64

        Returns:
            插值结果 f(x), shape=(N,)
        """
        x_f64: np.ndarray = x.astype(np.float64)
        # clip 到 [0, 1] 以处理边界外值
        x_clipped: np.ndarray = np.clip(x_f64, 0.0, 1.0)

        # 使用 np.searchsorted 找到每个 x 所在的区间索引
        # searchsorted 返回第一个 full_x[j] > x 的位置, 所以区间索引 = idx - 1
        indices: np.ndarray = np.searchsorted(self._full_x, x_clipped, side='right') - 1
        # 确保索引在有效范围 [0, len(slopes)-1]
        indices = np.clip(indices, 0, len(self._slopes) - 1)

        # 计算结果: y_j + slope_j * (x - x_j)
        x_j: np.ndarray = self._full_x[indices]
        y_j: np.ndarray = self._full_y[indices]
        slope_j: np.ndarray = self._slopes[indices]

        return y_j + slope_j * (x_clipped - x_j)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        计算导数值 f'(x).

        数学对应: 分段线性函数的导数在每个区间内为常数 slope_j.
        为兼容 ODE 使用, 返回平均近似: 1 / (n_segments * avg_segment_width).

        Args:
            x:  输入向量, shape=(N,), dtype=np.float64

        Returns:
            常数导数向量, shape=(N,)
        """
        # 返回近似平均斜率: 1 / (n_segments * 平均段宽)
        avg_derivative: float = 1.0 / max(self._n_segments * self._avg_segment_width, 1e-12)
        return np.full_like(x.astype(np.float64), avg_derivative, dtype=np.float64)

    @property
    def lipschitz_constant(self) -> float:
        """
        Lipschitz 常数 L = 1.0.

        数学对应: 附录D.3.1 — ∥·∥_U 度量结构.
        PWL 在 [0,1] 上归一化: 输出范围 [0,1], 输入范围 [0,1],
        故最大斜率 <= 1.0 (当所有段映射到 [0,1] 时).

        Returns:
            1.0, float
        """
        return 1.0

    @property
    def n_params(self) -> int:
        """
        可调参数个数 = 断点个数.

        数学对应: 附录D.3 假设0 — 搜索空间 H.
        每个断点是一个可调参数, 优化器可在 (0,1) 内调整其位置.

        Returns:
            len(breakpoints), int
        """
        return len(self.breakpoints)


# ---- 自动注册到 PluginRegistry ----
PluginRegistry.register_function("piecewise", PiecewiseLinearFunction)
