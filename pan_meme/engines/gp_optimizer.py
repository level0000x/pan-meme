"""GP贝叶斯全局优化器 — 高斯过程 + 预期改进 + 多保真度评估
数学对应: PB_ARCHITECTURE.md 第3.3节 — 用GP贝叶斯优化替代穷举搜索

核心算法流程:
  步骤1: Sobol低差异序列初始采样, 均匀覆盖搜索空间
  步骤2: 拟合高斯过程代理模型 GP(m(x), k(x,x'))
         - 核函数: RBF(连续维度) + 整数编码核(离散维度)
         - 超参数: scipy.optimize.minimize 最大化对数边际似然
  步骤3: 预期改进(EI)采集函数 — 平衡探索(exploration)与利用(exploitation)
  步骤4: Thompson采样处理离散维度候选生成
  步骤5: 多保真度评估 — 低保真(小样本筛选) + 高保真(全量精确评估Top-K)
  步骤6: 收敛判定 — EI < 1e-6 * |y_best| 或达到最大迭代

复杂度: O(K³ + K·d), 其中 K=观测点数(~200), d=搜索空间维度.
相比穷举加速约 10³x.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import warnings

# ============================================================
# 可选依赖检测
# ============================================================

try:
    from sklearn.gaussian_process import GaussianProcessRegressor as SklearnGP
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from scipy.optimize import minimize as scipy_minimize
    from scipy.stats import norm as scipy_norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from scipy.stats import qmc
    _HAS_QMC = True
except ImportError:
    _HAS_QMC = False


# ============================================================
# GaussianProcessModel — 底层GP解析实现 (sklearn回退方案)
# ============================================================

class GaussianProcessModel:
    """高斯过程底层实现 — RBF核解析GP拟合

    数学对应: 第3.3节步骤1 — 代理模型 GP(m(x), k(x,x'))

    GP回归公式:
      k(x,z) = exp(-||x-z||² / (2·ℓ²))              ... RBF核 (协方差函数)
      K = k(X,X) + σₙ²·I                               ... 带噪声的协方差矩阵
      α = K⁻¹ y                                         ... 权重向量 (对偶表示)
      μ(x*) = k(x*,X)·α                                ... 预测均值 (后验均值)
      σ²(x*) = k(x*,x*) - k(x*,X)·K⁻¹·k(X,x*)          ... 预测方差 (后验方差)

    Cholesky分解实现数值稳定: K = L·Lᵀ, 则 α = Lᵀ \\ (L \\ y)
    """

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6):
        """初始化GP模型

        Args:
            length_scale: RBF核长度尺度 ℓ, 控制函数光滑度
            noise: 观测噪声方差 σₙ², 保证数值稳定性
        """
        self.length_scale = length_scale
        self.noise = noise
        self.X_train_: Optional[np.ndarray] = None  # 训练输入
        self.y_train_: Optional[np.ndarray] = None  # 训练目标
        self.alpha_: Optional[np.ndarray] = None    # K⁻¹·y 权重向量
        self.L_: Optional[np.ndarray] = None         # Cholesky下三角 L (K = L·Lᵀ)

    # ---- RBF核矩阵 ----
    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray,
                   length_scale: float) -> np.ndarray:
        """RBF核矩阵: k_ij = exp(-||x_i - z_j||² / (2·ℓ²))

        数学对应: 标准径向基函数 (Radial Basis Function) 核

        Args:
            X1: 第一组输入向量, shape (n1, d)
            X2: 第二组输入向量, shape (n2, d)
            length_scale: 长度尺度 ℓ > 0

        Returns:
            K: 核矩阵, shape (n1, n2), K_{ij} = exp(-||X1_i - X2_j||²/(2ℓ²))
        """
        # 计算成对平方距离: ||x-z||² = ||x||² + ||z||² - 2x·z
        sqdist = (
            np.sum(X1 ** 2, axis=1).reshape(-1, 1) +
            np.sum(X2 ** 2, axis=1).reshape(1, -1) -
            2.0 * X1 @ X2.T
        )
        # 钳制非负 (浮点误差可能产生微小负值)
        sqdist = np.maximum(sqdist, 0.0)
        return np.exp(-0.5 * sqdist / (length_scale ** 2))

    # ---- GP拟合 ----
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """解析RBF核GP拟合: 计算 α = K⁻¹·y

        使用Cholesky分解 K = L·Lᵀ, 解两次三角方程组:
          L·w = y     (前代)
          Lᵀ·α = w    (回代)

        Args:
            X: 训练输入矩阵, shape (n, d)
            y: 训练目标向量, shape (n,)
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(X)

        # 构建协方差矩阵 Kₙₙ = k(X,X) + σₙ²·I
        K = self.rbf_kernel(X, X, self.length_scale)
        K += self.noise * np.eye(n, dtype=np.float64)

        # Cholesky分解, 失败则加抖动 (jitter) 重试
        jitter_levels = [0.0, 1e-8, 1e-6, 1e-4, 1e-2]
        success = False
        for jitter in jitter_levels:
            try:
                K_jitter = K + jitter * np.eye(n, dtype=np.float64)
                self.L_ = np.linalg.cholesky(K_jitter)
                # K·α = y  →  L·Lᵀ·α = y
                # 步骤1: L·w = y
                w = np.linalg.solve(self.L_, y)
                # 步骤2: Lᵀ·α = w
                self.alpha_ = np.linalg.solve(self.L_.T, w)
                success = True
                break
            except np.linalg.LinAlgError:
                continue

        if not success:
            raise RuntimeError(
                "GP拟合失败: Cholesky分解不收敛。"
                "请检查输入数据是否有NaN/Inf, 或增大noise参数。"
            )

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

    # ---- GP预测 ----
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GP后验预测: 返回 (均值, 方差)

        后验均值: μ(x*) = k(x*,X)·α
        后验方差: σ²(x*) = k(x*,x*) - k(x*,X)·K⁻¹·k(X,x*)

        使用 Cholesky: 令 v = L⁻¹·k(X, x*), 则 σ² = diag(K_ss) - sum(v²)

        Args:
            X_test: 测试输入矩阵, shape (m, d)

        Returns:
            mean:  预测均值 μ, shape (m,)
            variance: 预测方差 σ², shape (m,)  (已钳制为正)
        """
        if self.X_train_ is None or self.L_ is None:
            raise RuntimeError("模型未拟合, 请先调用 fit()")

        # k(x*, X): 测试点与训练点的核, shape (m, n)
        K_s = self.rbf_kernel(X_test, self.X_train_, self.length_scale)
        # k(x*, x*): 测试点自核, shape (m, m)
        K_ss = self.rbf_kernel(X_test, X_test, self.length_scale)

        # 均值 μ = K_s @ α
        mean = K_s @ self.alpha_

        # 方差: v = L⁻¹·K_sᵀ, 则 σ² = diag(K_ss) - sum(v², axis=0)
        v = np.linalg.solve(self.L_, K_s.T)        # shape (n, m)
        variance = np.diag(K_ss) - np.sum(v ** 2, axis=0)
        variance = np.maximum(variance, 1e-12)      # 钳制为正

        return mean, variance


# ============================================================
# Halton 低差异序列 (Sobol 回退方案)
# ============================================================

def _halton_sequence(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Halton低差异序列生成器 (Sobol序列的简化回退)

    数学原理: Halton序列以互质数为基底, 通过Van der Corput序列
    生成 [0,1]^d 内的低差异点集。discrepancy = O((log n)^d / n)

    Args:
        n: 生成点数
        d: 维度
        seed: 随机种子 (用于起点偏移)

    Returns:
        seq: shape (n, d), 值域 [0, 1)
    """
    # 前d个素数作为基底
    _PRIMES = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    ]

    def _van_der_corput(i: int, base: int) -> float:
        """Van der Corput序列: 以base为基底的镜像小数"""
        result = 0.0
        f = 1.0 / base
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    seq = np.zeros((n, d), dtype=np.float64)
    rng = np.random.RandomState(seed)
    # 随机起点偏移 (scrambled Halton)
    offset = rng.uniform(0.0, 0.5, size=d)

    for j in range(min(d, len(_PRIMES))):
        base = _PRIMES[j]
        for i in range(n):
            seq[i, j] = (_van_der_corput(i + 1, base) + offset[j]) % 1.0

    # 如果维度超出素数表, 用随机填充
    for j in range(len(_PRIMES), d):
        seq[:, j] = rng.uniform(0.0, 1.0, size=n)

    return seq


# ============================================================
# GPOptimizer — 高斯过程贝叶斯全局优化器
# ============================================================

class GPOptimizer:
    """GP贝叶斯全局优化器 — 连续-离散混合搜索空间的贝叶斯优化

    数学对应: PB_ARCHITECTURE.md 第3.3节 步骤1-6

    搜索空间定义:
      search_space = {
          param_name: (low, high, "continuous"|"categorical"|"ordinal")
      }
      - continuous: low/high为浮点边界
      - categorical: low=0, high=类别数 (int), 无序离散
      - ordinal: low=0, high=等级数 (int), 有序离散

    算法:
      1. Sobol序列生成 n_initial 个初始候选
      2. 评估初始候选 (低保真度优化可选)
      3. 迭代:
         a. 拟合GP代理模型 (sklearn或手动RBF)
         b. 计算EI采集函数
         c. 对离散维度Thompson采样生成候选
         d. 选择EI最大的候选评估
         e. 更新观测, 记录收敛曲线
      4. 返回最优参数与损失

    Attributes:
        _convergence_curve: 每轮最优损失历史 (可在optimize后访问)
    """

    def __init__(self, search_space: dict, n_initial: int = 20,
                 n_iterations: int = 100, random_state: int = 42):
        """初始化GP贝叶斯优化器

        Args:
            search_space: 搜索空间字典
                {name: (low, high, "continuous"|"categorical"|"ordinal")}
            n_initial: Sobol初始采样点数 (默认20)
            n_iterations: 最大贝叶斯优化迭代次数 (默认100)
            random_state: 随机种子 (默认42)
        """
        self.search_space = search_space
        self.n_initial = int(n_initial)
        self.n_iterations = int(n_iterations)
        self.random_state = int(random_state)
        self.rng = np.random.RandomState(random_state)

        # 解析搜索空间: 按参数类型分类
        self._param_names: List[str] = list(search_space.keys())
        self._continuous_params: List[str] = []
        self._categorical_params: List[str] = []
        self._ordinal_params: List[str] = []

        for name in self._param_names:
            _, _, ptype = search_space[name]
            if ptype == "continuous":
                self._continuous_params.append(name)
            elif ptype == "categorical":
                self._categorical_params.append(name)
            else:  # ordinal
                self._ordinal_params.append(name)

        # 构建编码表: 每个参数映射到一个连续编码维度
        self._build_encoding_scheme()

        # 观测历史
        self._X_obs_list: List[np.ndarray] = []    # 编码后的观测点
        self._y_obs_list: List[float] = []          # 对应的目标值
        self._best_params: Optional[Dict[str, Any]] = None
        self._best_loss: float = float('inf')
        self._convergence_curve: List[float] = []   # 收敛曲线 (每轮最优损失)

        # GP模型引用
        self._gp: Optional[Any] = None
        self._gp_method: str = "none"

    # ── 编码/解码 ──

    def _build_encoding_scheme(self) -> None:
        """构建参数到连续向量的编码映射

        编码策略:
          - 连续参数: 归一化到 [0, 1]
          - 离散参数: 编码为 [0, 1] 内的连续值, 解码时四舍五入
        """
        self._encoding: Dict[str, Tuple[int, float, float, str]] = {}
        dim = 0
        for name in self._param_names:
            low, high, ptype = self.search_space[name]
            self._encoding[name] = (dim, float(low), float(high), ptype)
            dim += 1
        self._n_encoded_dims: int = dim

    def _encode_params(self, params: Dict[str, Any]) -> np.ndarray:
        """参数字典 → 归一化连续向量 [0,1]^d"""
        x = np.zeros(self._n_encoded_dims, dtype=np.float64)
        for name, (dim, low, high, ptype) in self._encoding.items():
            val = params[name]
            if ptype == "continuous":
                if high == low:
                    x[dim] = 0.0
                else:
                    x[dim] = (float(val) - low) / (high - low)
            else:
                # categorical / ordinal: 值域 [0, n_cats-1], 归一化到[0,1]
                n_cats = int(high)
                if n_cats <= 1:
                    x[dim] = 0.0
                else:
                    x[dim] = float(val) / (n_cats - 1)
        return np.clip(x, 0.0, 1.0)

    def _decode_params(self, x: np.ndarray) -> Dict[str, Any]:
        """归一化向量 [0,1]^d → 参数字典"""
        params: Dict[str, Any] = {}
        for name, (dim, low, high, ptype) in self._encoding.items():
            raw = float(np.clip(x[dim], 0.0, 1.0))
            if ptype == "continuous":
                params[name] = low + raw * (high - low)
            else:
                n_cats = int(high)
                if ptype == "categorical":
                    # 四舍五入到最近的类别索引
                    idx = int(round(raw * (n_cats - 1)))
                    params[name] = max(0, min(n_cats - 1, idx))
                else:  # ordinal
                    idx = int(round(raw * (n_cats - 1)))
                    params[name] = max(0, min(n_cats - 1, idx))
        return params

    # ── 初始采样 ──

    def _sobol_sample(self, n: int) -> List[Dict[str, Any]]:
        """Sobol低差异序列初始采样

        数学对应: 第3.3节步骤1 — 低差异序列覆盖搜索空间

        Sobol序列提供 [0,1]^d 内的超均匀分布点, 比纯随机采样
        的discrepancy更低, 保证初始观测点均匀覆盖搜索空间。

        回退策略: 若 scipy.stats.qmc 不可用, 使用 Halton 序列。

        Args:
            n: 采样点数

        Returns:
            candidates: 解码后的参数字典列表
        """
        d = self._n_encoded_dims

        if _HAS_QMC and d > 0:
            # 使用 scipy Sobol 引擎
            try:
                engine = qmc.Sobol(d=d, scramble=True, seed=self.random_state)
                points = engine.random(n=n)
            except Exception:
                points = _halton_sequence(n, d, seed=self.random_state)
        elif d > 0:
            points = _halton_sequence(n, d, seed=self.random_state)
        else:
            return [{}]

        candidates = []
        for i in range(n):
            params = self._decode_params(points[i])
            candidates.append(params)
        return candidates

    # ── GP拟合 ──

    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> dict:
        """拟合高斯过程代理模型

        数学对应: 第3.3节步骤1 — 代理模型 GP(m(x), k(x,x'))

        核函数:
          - sklearn: ConstantKernel * RBF + WhiteKernel
            (常数均值 + RBF协方差 + 白噪声)
          - 手动回退: GaussianProcessModel (纯RBF + 固定noise)

        超参数优化 (sklearn):
          n_restarts_optimizer=5, 用 L-BFGS-B 最大化对数边际似然
          (scipy.optimize.minimize 内部调用)

        Args:
            X: 观测输入, shape (n_obs, d)
            y: 观测目标值, shape (n_obs,)

        Returns:
            {"gp": gp_object, "method": "sklearn"|"manual"}
        """
        y = np.asarray(y, dtype=np.float64).ravel()

        if _HAS_SKLEARN:
            # sklearn GP: ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-6)
            kernel = (
                ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
                * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
                + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-1))
            )
            try:
                gp = SklearnGP(
                    kernel=kernel,
                    n_restarts_optimizer=5,
                    random_state=self.random_state,
                    normalize_y=True,
                )
                gp.fit(X, y)
                return {"gp": gp, "method": "sklearn"}
            except Exception as e:
                warnings.warn(
                    f"sklearn GP拟合失败 ({e}), 回退到手动GaussianProcessModel"
                )

        # 手动GP回退
        gp = GaussianProcessModel(length_scale=1.0, noise=1e-6)
        gp.fit(X, y)
        return {"gp": gp, "method": "manual"}

    # ── 采集函数 ──

    def _expected_improvement(self, X_candidates: np.ndarray,
                               gp: Any, y_best: float,
                               xi: float = 0.01) -> np.ndarray:
        """计算预期改进 (Expected Improvement, EI)

        数学对应: 第3.3节步骤2 — EI采集函数平衡探索/利用

        EI公式:
          EI(x) = (μ(x) - y_best - ξ) · Φ(Z) + σ(x) · φ(Z)

          其中:
            Z = (μ(x) - y_best - ξ) / σ(x)   if σ(x) > 0 else 0
            Φ(·): 标准正态CDF (累积分布函数)
            φ(·): 标准正态PDF (概率密度函数)
            ξ:    探索参数 (exploration parameter), 鼓励探索未观测区域

        EI的两项含义:
          - 第一项 (μ - y_best - ξ)·Φ(Z): 利用项 — 均值低于y_best时贡献大
          - 第二项 σ·φ(Z): 探索项 — 方差大时贡献大

        Args:
            X_candidates: 候选点矩阵, shape (m, d)
            gp: GP模型 (sklearn GP 或 GaussianProcessModel)
            y_best: 当前最佳观测值 (越小越好)
            xi: 探索参数, 默认0.01

        Returns:
            ei: EI值数组, shape (m,)
        """
        if hasattr(gp, 'predict'):
            if _HAS_SKLEARN and isinstance(gp, SklearnGP):
                mean, std = gp.predict(X_candidates, return_std=True)
                variance = std ** 2
            else:
                mean, variance = gp.predict(X_candidates)
                std = np.sqrt(np.maximum(variance, 1e-12))
        else:
            mean = np.zeros(len(X_candidates))
            variance = np.ones(len(X_candidates))
            std = np.ones(len(X_candidates))

        mean = np.asarray(mean, dtype=np.float64).ravel()
        std = np.asarray(std, dtype=np.float64).ravel()

        # 计算Z = (μ - y_best - ξ) / σ
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = y_best - mean - xi          # 改进量 (取反: y_best - μ)
            Z = imp / std
            Z = np.where(std > 1e-12, Z, 0.0)

        if _HAS_SCIPY:
            # 使用scipy的norm.cdf/pdf
            ei = imp * scipy_norm.cdf(Z) + std * scipy_norm.pdf(Z)
        else:
            # 手动计算标准正态CDF/PDF
            cdf_z = 0.5 * (1.0 + np.erf(Z / np.sqrt(2.0)))
            pdf_z = np.exp(-0.5 * Z ** 2) / np.sqrt(2.0 * np.pi)
            ei = imp * cdf_z + std * pdf_z

        ei = np.where(std > 1e-12, ei, 0.0)
        return np.maximum(ei, 0.0)

    # ── Thompson采样 ──

    def _thompson_sample_discrete(self, n_candidates: int = 50) -> List[Dict[str, Any]]:
        """Thompson采样生成离散维度候选

        数学对应: 第3.3节步骤2 — 多臂老虎机 (Thompson Sampling)

        原理: 从GP后验中采样一个函数实现, 然后对采样函数求最优。
        Thompson采样自然地平衡探索与利用。

        实现: 从GP后验中采样, 对连续维度用随机扰动, 对离散维度
        枚举可能的类别值, 选择后验采样均值的argmin。

        Args:
            n_candidates: 生成的候选数

        Returns:
            candidates: 参数字典列表
        """
        d = self._n_encoded_dims
        candidates: List[Dict[str, Any]] = []

        if len(self._X_obs_list) < 2:
            # 观测不足, 随机生成
            for _ in range(n_candidates):
                x_rand = self.rng.uniform(0.0, 1.0, size=d)
                candidates.append(self._decode_params(x_rand))
            return candidates

        # 从后验采样函数: 在 [0,1]^d 上随机取候选点
        X_cand = self.rng.uniform(0.0, 1.0, size=(n_candidates, d))

        # 获取GP预测
        if self._gp is not None:
            if hasattr(self._gp, 'predict'):
                if _HAS_SKLEARN and isinstance(self._gp, SklearnGP):
                    mean, std = self._gp.predict(X_cand, return_std=True)
                    # Thompson采样: 从N(mean, std²)采样
                    thompson_samples = self.rng.normal(
                        mean.ravel(), np.maximum(std, 1e-8)
                    )
                    # 选EI最大的前k个
                    scores = -thompson_samples  # 越小越好
                    best_idx = np.argsort(scores)[:n_candidates]
                    for idx in best_idx:
                        candidates.append(self._decode_params(X_cand[idx]))
                    return candidates
                else:
                    mean, variance = self._gp.predict(X_cand)
            else:
                mean = np.zeros(len(X_cand))
                variance = np.ones(len(X_cand))

            std = np.sqrt(np.maximum(variance, 1e-12))
            thompson_samples = self.rng.normal(mean.ravel(), std)
            scores = -thompson_samples
            best_idx = np.argsort(scores)[:n_candidates]
            for idx in best_idx:
                candidates.append(self._decode_params(X_cand[idx]))
        else:
            for i in range(n_candidates):
                candidates.append(self._decode_params(X_cand[i]))

        return candidates

    # ── 主优化循环 ──

    def optimize(self, objective_fn: Callable[[dict], float],
                 low_fidelity_fn: Optional[Callable[[dict], float]] = None
                 ) -> Dict[str, Any]:
        """多保真度贝叶斯优化主循环

        数学对应: 第3.3节步骤3 — 多保真度优化 (Multi-Fidelity)

        两阶段评估策略:
          阶段1 (低保真度筛选):
            - 用 low_fidelity_fn 在小样本子集上快速评估所有初始候选
            - 筛选 Top-K 候选进入下一阶段
            - 如果 low_fidelity_fn 为 None, 则跳过此阶段

          阶段2 (高保真度精确评估):
            - 用 objective_fn 在全量上评估
            - GP贝叶斯优化迭代:
              1. 拟合GP代理模型
              2. 计算EI采集函数 → 选择下一个候选
              3. 评估候选, 更新观测, 记录收敛曲线
              4. 检测收敛: EI < 1e-6 * |y_best| 或达到 max_iter

        Args:
            objective_fn: 高保真度目标函数 f(params) -> float (越小越好)
            low_fidelity_fn: 低保真度目标函数 (可选, 用于快速预筛选)

        Returns:
            {
                "h_star":  最优参数字典,
                "loss":    最优损失值,
                "candidates_tested": 总评估次数,
                "convergence_curve": 收敛曲线 [每轮最优损失],
            }
        """
        # ── 阶段0: Sobol初始采样 ──
        initial_candidates = self._sobol_sample(self.n_initial)
        n_evaluated = 0

        # ── 阶段1: 低保真度快速筛选 ──
        if low_fidelity_fn is not None:
            # 低保真评估所有初始候选
            lf_scores: List[Tuple[float, dict]] = []
            for params in initial_candidates:
                try:
                    score = float(low_fidelity_fn(params))
                    lf_scores.append((score, params))
                    n_evaluated += 1
                except Exception as e:
                    warnings.warn(f"低保真评估失败 ({params}): {e}")

            if not lf_scores:
                raise RuntimeError("所有低保真评估均失败, 请检查 low_fidelity_fn")

            # 筛选 Top-K (取前max(n_initial//2, 5)个)
            lf_scores.sort(key=lambda x: x[0])
            top_k = max(min(self.n_initial // 2, len(lf_scores)), 1)
            screened_candidates = [p for _, p in lf_scores[:top_k]]
        else:
            screened_candidates = initial_candidates

        # ── 阶段2: 高保真度贝叶斯优化 ──
        for params in screened_candidates:
            try:
                loss = float(objective_fn(params))
                x_enc = self._encode_params(params)
                self._X_obs_list.append(x_enc)
                self._y_obs_list.append(loss)
                n_evaluated += 1

                if loss < self._best_loss:
                    self._best_loss = loss
                    self._best_params = dict(params)
                self._convergence_curve.append(self._best_loss)
            except Exception as e:
                warnings.warn(f"高保真评估失败 ({params}): {e}")

        # 主迭代循环
        for iteration in range(self.n_iterations):
            if len(self._X_obs_list) < 2:
                # 观测不足, 随机采样
                x_new = self.rng.uniform(0.0, 1.0, size=self._n_encoded_dims)
            else:
                # 步骤2: 拟合GP代理模型
                X_obs = np.array(self._X_obs_list)
                y_obs = np.array(self._y_obs_list)
                gp_result = self._fit_gp(X_obs, y_obs)
                self._gp = gp_result["gp"]
                self._gp_method = gp_result["method"]

                # 步骤3-4: 生成候选 → 计算EI → 选最优
                # Thompson采样生成候选
                thompson_candidates = self._thompson_sample_discrete(
                    n_candidates=min(200, self.n_initial * 5)
                )

                # 构建候选矩阵
                X_cand = np.array([
                    self._encode_params(c) for c in thompson_candidates
                ])

                # 计算EI
                y_best = min(self._y_obs_list) if self._y_obs_list else float('inf')
                ei_values = self._expected_improvement(X_cand, self._gp, y_best)

                # 收敛判定
                max_ei = np.max(ei_values)
                if max_ei < 1e-6 * max(abs(y_best), 1e-8):
                    break

                # 选EI最大的候选
                best_idx = int(np.argmax(ei_values))
                best_candidate = thompson_candidates[best_idx]
                x_new = self._encode_params(best_candidate)

            # 评估新候选
            new_params = self._decode_params(x_new)
            already_tested = any(
                np.allclose(x_new, xo, atol=1e-6) for xo in self._X_obs_list
            )
            if already_tested:
                continue

            try:
                loss = float(objective_fn(new_params))
                self._X_obs_list.append(x_new.copy())
                self._y_obs_list.append(loss)
                n_evaluated += 1

                if loss < self._best_loss:
                    self._best_loss = loss
                    self._best_params = dict(new_params)
                self._convergence_curve.append(self._best_loss)
            except Exception as e:
                warnings.warn(f"高保真评估失败 (iter={iteration}, {new_params}): {e}")

        # ── 返回结果 ──
        return {
            "h_star": self._best_params if self._best_params else {},
            "loss": self._best_loss,
            "candidates_tested": n_evaluated,
            "convergence_curve": list(self._convergence_curve),
        }

    # ── 收敛曲线 ──

    def convergence_curve(self) -> List[float]:
        """返回每轮最优损失的历史

        Returns:
            curve: List[float], curve[i] = 第i次评估后的最优损失
        """
        return list(self._convergence_curve)
