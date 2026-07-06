"""全局优化器 — 穷举搜索与贝叶斯优化的统一入口

数学对应: 附录D.3 假设0 — h* = argmin L(h; I)
          PB_ARCHITECTURE.md 第3.3节 — 混合贝叶斯优化

搜索模式:
  - "exhaustive":     穷举 |T_candidates|×|function_families|×|n_candidates|
                       每个组合用 L-BFGS-B 优化 11 个 Theta 参数
  - "gp":             高斯过程贝叶斯优化 (GPOptimizer), 加速 ~10³x
  - "multi_fidelity": 多保真度: 低保真筛选 + GP高保真优化 (同 "gp" 路径)

搜索空间: H = T_candidates × function_families × Theta(11维) × n(模因数)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import warnings
from itertools import product

from pan_meme.core.types import OptimizationResult
from pan_meme.engines.gp_optimizer import GPOptimizer

# scipy.optimize.minimize 用于 Theta 参数优化 (穷举模式)
try:
    from scipy.optimize import minimize as scipy_minimize
    _HAS_SCIPY_OPT = True
except ImportError:
    _HAS_SCIPY_OPT = False


# ============================================================
# 函数族映射表
# ============================================================

# 函数族名称 → D(R)映射函数标识 (对应论文4.3.1方程系统中的 func_D, func_R)
_FUNCTION_FAMILY_MAP: Dict[str, str] = {
    "power":     "D(R) = R^p,       R(D) = D^q",
    "exp":       "D(R) = 1-e^(-aR), R(D) = 1-e^(-bD)",
    "sigmoid":   "D(R) = 1/(1+e^(-k(R-R0))),  R(D) = 1/(1+e^(-m(D-D0)))",
    "log":       "D(R) = log(1+aR)/log(1+a),   R(D) = log(1+bD)/log(1+b)",
    "piecewise": "D(R) = min(1, R/T1), minor(1) or smooth piecewise",
}


# ============================================================
# GlobalOptimizer — 全局优化器主类
# ============================================================

class GlobalOptimizer:
    """全局优化器 — 支持穷举搜索、GP贝叶斯优化、多保真度优化

    数学对应: 附录D.3 假设0 — 存在全局最优配置 h* = argmin_{h∈H} L(h; I)

    工作流程:
      1. 解析 config 获取搜索范围
      2. 从 raw_input 估计模因数 n 的候选范围
      3. 根据 optimizer_mode 选择穷举或GP搜索
      4. 返回 OptimizationResult

    Usage:
        optimizer = GlobalOptimizer(config)
        result = optimizer.optimize(raw_input, pipeline_fn)
        # result.h_star: 最优配置 {T, func_family, Theta, n}
        # result.loss: 最优损失
    """

    def __init__(self, config: dict):
        """初始化全局优化器

        Args:
            config: 配置字典 (格式见 pan_meme.config.schema.DEFAULT_CONFIG)
                关键字段:
                  - optimizer.function_families: 函数族列表
                  - optimizer.theta_bounds: 11个Theta参数边界
                  - optimizer.lambda_reg: L2正则化系数
                  - optimizer.optimizer_mode: "exhaustive"|"gp"|"multi_fidelity"
                  - module1.threshold_candidates: 阈值候选集
        """
        self.config = config

        # ── optimizer 配置 ──
        opt_cfg = config.get("optimizer", {})
        self.function_families: List[str] = opt_cfg.get(
            "function_families", ["power", "exp", "sigmoid"]
        )
        self.theta_bounds: Dict[str, List[float]] = opt_cfg.get(
            "theta_bounds", {
                "alpha_1": [0.0, 1.0], "alpha_2": [0.0, 1.0],
                "beta_1": [0.0, 2.0], "beta_2": [0.0, 2.0],
                "gamma_1": [0.0, 2.0], "gamma_2": [0.0, 2.0],
                "delta_1": [0.0, 2.0], "delta_2": [0.0, 5.0], "delta_3": [0.0, 1.0],
                "epsilon_1": [0.0, 2.0], "epsilon_2": [0.0, 5.0],
            }
        )
        self.lambda_reg: float = opt_cfg.get("lambda_reg", 0.01)
        self.mode: str = opt_cfg.get("optimizer_mode", "exhaustive")

        # ── module1 配置 ──
        mod1_cfg = config.get("module1", {})
        self.T_candidates: List[float] = mod1_cfg.get(
            "threshold_candidates", [0.3, 0.5, 0.7]
        )

        # 验证 mode
        _valid_modes = ("exhaustive", "gp", "multi_fidelity")
        if self.mode not in _valid_modes:
            warnings.warn(
                f"未知的 optimizer_mode='{self.mode}', "
                f"回退到 'exhaustive'. 有效值: {_valid_modes}"
            )
            self.mode = "exhaustive"

        # Theta参数名列表 (保持固定顺序)
        self._theta_names: List[str] = list(self.theta_bounds.keys())

    # ══════════════════════════════════════════════════════════
    # 公开 API
    # ══════════════════════════════════════════════════════════

    def optimize(self, raw_input: Any,
                 pipeline_fn: Callable[[Dict[str, Any]], float]
                 ) -> OptimizationResult:
        """主优化入口 — 根据 mode 分发到穷举或GP搜索

        Args:
            raw_input: 原始输入 (用于估计模因数 n 的范围)
            pipeline_fn: 管线评估函数 f(params) -> loss (越小越好)
                params 格式: {
                    "T": float,           # 阈值
                    "func_family": int,   # 函数族索引
                    "n": int,             # 模因数
                    "alpha_1": float, ... # 11个Theta参数
                }

        Returns:
            OptimizationResult: 最优配置与损失
        """
        # 估计模因数候选范围
        n_candidates = self._estimate_n_candidates(raw_input)

        # 构建搜索空间
        search_space = self._build_search_space(n_candidates)

        if self.mode in ("gp", "multi_fidelity"):
            result = self._gp_search(search_space, pipeline_fn)
        else:
            result = self._exhaustive_search(search_space, pipeline_fn)

        return result

    # ══════════════════════════════════════════════════════════
    # 搜索空间构建
    # ══════════════════════════════════════════════════════════

    def _build_search_space(self, n_candidates: List[int]) -> dict:
        """构建混合搜索空间 (连续-离散混合)

        搜索空间 H = T(连续) × func_family(分类) × n(序数) × Theta(11维连续)

        Args:
            n_candidates: 模因数候选值列表

        Returns:
            search_space: GPOptimizer格式的搜索空间字典
        """
        space: Dict[str, Tuple] = {}

        # T: 阈值 — 连续 [0, 1]
        space["T"] = (0.0, 1.0, "continuous")

        # func_family: 函数族 — 分类 (索引 0..n_families-1)
        n_families = len(self.function_families)
        space["func_family"] = (0.0, float(n_families), "categorical")

        # n: 模因数 — 序数
        if len(n_candidates) > 1:
            n_min, n_max = min(n_candidates), max(n_candidates)
            space["n"] = (float(n_min), float(n_max), "ordinal")
        elif len(n_candidates) == 1:
            space["n"] = (float(n_candidates[0]), float(n_candidates[0] + 1), "ordinal")
        else:
            space["n"] = (1.0, 100.0, "ordinal")

        # Theta: 11个动力学参数 — 连续
        for name, bounds in self.theta_bounds.items():
            space[name] = (float(bounds[0]), float(bounds[1]), "continuous")

        return space

    def _estimate_n_candidates(self, raw_input: Any) -> List[int]:
        """从原始输入估计模因数 n 的候选范围

        启发式估计:
          - 如果 raw_input 有已知的节点数/边数, 据此估算
          - 默认返回 [10, 20, 50, 100, 200] 作为候选

        Args:
            raw_input: 原始输入数据

        Returns:
            n_list: 模因数候选值列表
        """
        # 尝试从 raw_input 提取特征
        if hasattr(raw_input, 'meta') and isinstance(raw_input.meta, dict):
            node_count = raw_input.meta.get("node_count", 0)
            edge_count = raw_input.meta.get("edge_count", 0)
            if node_count > 0:
                # 粗略估计: n ~ sqrt(node_count) 的量级
                base = int(np.sqrt(node_count))
                return sorted(set([
                    max(1, base // 5),
                    max(1, base // 2),
                    base,
                    base * 2,
                    base * 5,
                ]))
        if isinstance(raw_input, dict):
            if "node_count" in raw_input:
                base = int(np.sqrt(raw_input["node_count"]))
                return sorted(set([
                    max(1, base // 5),
                    max(1, base // 2),
                    base,
                    base * 2,
                    base * 5,
                ]))

        # 默认候选
        return [10, 20, 50, 100, 200]

    # ══════════════════════════════════════════════════════════
    # 穷举搜索
    # ══════════════════════════════════════════════════════════

    def _exhaustive_search(self, search_space: dict,
                           objective_fn: Callable[[Dict[str, Any]], float]
                           ) -> OptimizationResult:
        """传统穷举搜索 — 遍历 T×函数族×n 组合, 每组合优化Theta

        数学对应: 原始穷举优化 O(|T|×|F|×|n|×|Theta_opt|)

        对每个 (T, func_family, n) 组合:
          1. 固定 T, func_family, n
          2. 用 L-BFGS-B 优化 11 个 Theta 参数
          3. 用 lambda_reg 做 L2 正则化
          4. 记录最优组合

        Args:
            search_space: 搜索空间 (仅用于提取候选集合)
            objective_fn: 目标函数 f(params) -> float

        Returns:
            OptimizationResult
        """
        # 提取候选集
        T_list = self.T_candidates
        family_indices = list(range(len(self.function_families)))

        # 从 search_space 提取 n 候选
        n_spec = search_space.get("n", (10, 100, "ordinal"))
        n_min, n_max = int(n_spec[0]), int(n_spec[1])
        n_list = sorted(set([
            n_min,
            (n_min + n_max) // 2,
            n_max,
        ]))
        n_list = [n for n in n_list if n > 0]

        best_params: Optional[Dict[str, Any]] = None
        best_loss: float = float('inf')
        n_tested: int = 0
        convergence_curve: List[float] = []

        # 穷举所有 (T, func_family, n) 组合
        combinatorial_candidates = list(product(T_list, family_indices, n_list))

        for T_val, fam_idx, n_val in combinatorial_candidates:
            fam_name = self.function_families[fam_idx]

            # ── 优化 Theta 参数 ──
            theta_opt_result = self._optimize_theta(
                T_val, fam_idx, n_val, objective_fn
            )

            n_tested += theta_opt_result.get("evaluations", 1)

            # 构造完整参数
            full_params = {
                "T": T_val,
                "func_family": fam_idx,
                "func_family_name": fam_name,
                "n": n_val,
                "Theta": theta_opt_result.get("theta", {}),
            }
            # 扁平化 Theta 字段
            for k, v in theta_opt_result.get("theta", {}).items():
                full_params[k] = v

            loss = theta_opt_result.get("loss", float('inf'))

            if loss < best_loss:
                best_loss = loss
                best_params = full_params
                convergence_curve.append(best_loss)
            elif convergence_curve:
                convergence_curve.append(convergence_curve[-1])
            else:
                convergence_curve.append(best_loss)

        return OptimizationResult(
            h_star=best_params if best_params else {},
            loss=best_loss,
            candidates_tested=n_tested,
        )

    def _optimize_theta(self, T_val: float, fam_idx: int, n_val: int,
                        objective_fn: Callable[[Dict[str, Any]], float]
                        ) -> Dict[str, Any]:
        """对固定 (T, func_family, n) 优化 11 个 Theta 参数

        使用 scipy.optimize.minimize (L-BFGS-B) 在有界空间内优化。

        Args:
            T_val: 阈值
            fam_idx: 函数族索引
            n_val: 模因数
            objective_fn: 目标函数

        Returns:
            {"theta": {...}, "loss": float, "evaluations": int}
        """
        # 构建初始猜测和边界
        x0_list: List[float] = []
        bounds_list: List[Tuple[float, float]] = []

        for name in self._theta_names:
            low, high = self.theta_bounds[name]
            x0_list.append((low + high) / 2.0)  # 中点作为初始猜测
            bounds_list.append((float(low), float(high)))

        x0 = np.array(x0_list, dtype=np.float64)
        bounds = bounds_list

        eval_count = [0]  # 用列表捕获闭包中的计数器

        def _theta_objective(theta_vec: np.ndarray) -> float:
            """Theta优化目标: 构造完整参数并调用管线"""
            params: Dict[str, Any] = {
                "T": T_val,
                "func_family": fam_idx,
                "n": n_val,
            }
            for i, name in enumerate(self._theta_names):
                params[name] = float(theta_vec[i])

            eval_count[0] += 1

            # L2 正则化项
            reg_penalty = self.lambda_reg * np.sum(theta_vec ** 2)

            try:
                loss = float(objective_fn(params))
                return loss + reg_penalty
            except Exception as e:
                warnings.warn(f"Theta目标函数评估失败 ({params}): {e}")
                return float('inf')

        # L-BFGS-B 优化
        if _HAS_SCIPY_OPT:
            try:
                opt_result = scipy_minimize(
                    _theta_objective,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={
                        "maxiter": 200,
                        "ftol": 1e-8,
                        "gtol": 1e-6,
                    },
                )
                theta_opt = {
                    name: float(opt_result.x[i])
                    for i, name in enumerate(self._theta_names)
                }
                loss = float(opt_result.fun)
                # 减去正则化项得到纯损失
                reg = self.lambda_reg * np.sum(opt_result.x ** 2)
                loss = loss - reg if np.isfinite(loss) else loss
            except Exception as e:
                warnings.warn(f"L-BFGS-B优化失败 ({e}), 使用初始猜测")
                theta_opt = {
                    name: float(x0[i]) for i, name in enumerate(self._theta_names)
                }
                loss = float(_theta_objective(x0))
        else:
            # 无 scipy: 使用初始猜测
            theta_opt = {
                name: float(x0[i]) for i, name in enumerate(self._theta_names)
            }
            loss = float(_theta_objective(x0))

        return {
            "theta": theta_opt,
            "loss": loss,
            "evaluations": eval_count[0],
        }

    # ══════════════════════════════════════════════════════════
    # GP贝叶斯搜索
    # ══════════════════════════════════════════════════════════

    def _gp_search(self, search_space: dict,
                   objective_fn: Callable[[Dict[str, Any]], float]
                   ) -> OptimizationResult:
        """GP贝叶斯优化搜索

        数学对应: 第3.3节 — 用高斯过程代理模型 + EI采集函数进行全局优化

        构造 GPOptimizer 实例, 委托贝叶斯优化。若 mode=="multi_fidelity",
        则创建低保真目标函数 (在子样本上评估), 实现多保真度优化。

        Args:
            search_space: 搜索空间
            objective_fn: 高保真度目标函数

        Returns:
            OptimizationResult
        """
        # GP超参数: 从 config 推断
        n_params = len(search_space)
        n_initial = max(20, n_params * 2)   # 初始采样: 至少 2×维度
        n_iterations = min(100, n_params * 10)  # 迭代上限

        optimizer = GPOptimizer(
            search_space=search_space,
            n_initial=n_initial,
            n_iterations=n_iterations,
            random_state=42,
        )

        # 多保真度: 构建低保真快速评估函数
        low_fidelity_fn: Optional[Callable] = None
        if self.mode == "multi_fidelity":
            low_fidelity_fn = self._build_low_fidelity_fn(objective_fn)

        # 运行贝叶斯优化
        gp_result = optimizer.optimize(
            objective_fn=objective_fn,
            low_fidelity_fn=low_fidelity_fn,
        )

        # 丰富返回结果: func_family索引 → 名称
        h_star = gp_result.get("h_star", {})
        if "func_family" in h_star:
            fam_idx = int(h_star["func_family"])
            if 0 <= fam_idx < len(self.function_families):
                h_star["func_family_name"] = self.function_families[fam_idx]

        # 构建 Theta 子字典
        theta_dict = {}
        for name in self._theta_names:
            if name in h_star:
                theta_dict[name] = h_star[name]
        h_star["Theta"] = theta_dict

        return OptimizationResult(
            h_star=h_star,
            loss=float(gp_result.get("loss", float('inf'))),
            candidates_tested=int(gp_result.get("candidates_tested", 0)),
        )

    def _build_low_fidelity_fn(self,
                                objective_fn: Callable[[Dict[str, Any]], float]
                                ) -> Callable[[Dict[str, Any]], float]:
        """构建低保真度目标函数 (multi_fidelity模式)

        低保真策略: 将 n (模因数) 缩小为原来的 1/5~1/3,
        从而大幅减少管线运行时间, 实现快速预筛选。

        Args:
            objective_fn: 原始高保真目标函数

        Returns:
            low_fidelity_fn: 低保真包装函数
        """
        def low_fidelity_fn(params: dict) -> float:
            lf_params = dict(params)
            # 缩小模因数以加速评估
            if "n" in lf_params:
                original_n = int(lf_params["n"])
                lf_params["n"] = max(3, original_n // 3)
            try:
                return float(objective_fn(lf_params))
            except Exception:
                return float('inf')

        return low_fidelity_fn
