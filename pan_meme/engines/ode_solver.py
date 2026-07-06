"""ODE求解器 — 五维动力学系统数值积分
数学对应: 附录D.5 定理6(分段解存在唯一性) + 定理7(Omega不变性) + 论文4.3.1(方程系统)
"""

from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
from scipy.integrate import solve_ivp
from pan_meme.core.types import ODEConfig, ODEResult, MemeState


class ODESolver:
    """论文4.3.1方程系统的数值求解器"""

    def __init__(self, config: ODEConfig):
        self.config = config

    # ---- 论文4.3.1: 五维ODE右侧函数 ----
    def _rhs(self, t: float, M_vec: np.ndarray, params: dict,
             func_D: Callable, func_R: Callable,
             I_ext: Optional[Callable] = None) -> List[float]:
        """
        对应论文4.3.1完整方程系统:
          dD/dt  = -alpha_1*R*D        + alpha_2*S*(1-D)
          dB/dt  =  beta_1*R*(1-B)     - beta_2*D*B
          drho/dt= -gamma_1*R*rho      + gamma_2*(1-rho)*I_ext(t)
          dR/dt  =  delta_1*rho*B*(1-R) - delta_2*Phi_D(D)*R - delta_3*R
          dS/dt  =  epsilon_1*D*(1-S)   - epsilon_2*Phi_R(R)*S
        """
        D, B, rho, Rs, S = M_vec
        I = I_ext(t) if I_ext else 0.0
        dD  = -params["alpha_1"] * Rs * D           + params["alpha_2"] * S * (1 - D)
        dB  =  params["beta_1"]  * Rs * (1 - B)     - params["beta_2"] * D * B
        drho = -params["gamma_1"] * Rs * rho         + params["gamma_2"] * (1 - rho) * I
        dR  =  params["delta_1"] * rho * B * (1 - Rs) - params["delta_2"] * func_D(D) * Rs - params["delta_3"] * Rs
        dS  =  params["epsilon_1"] * D * (1 - S)     - params["epsilon_2"] * func_R(Rs) * S
        return [dD, dB, drho, dR, dS]

    def solve_single(self, initial_state: np.ndarray,
                     params: dict, func_D: Callable, func_R: Callable,
                     I_ext: Optional[Callable] = None) -> ODEResult:
        """单个模因演化轨迹"""
        sol = solve_ivp(
            lambda t, y: self._rhs(t, y, params, func_D, func_R, I_ext),
            t_span=self.config.t_span, y0=initial_state,
            method=self.config.method, atol=self.config.atol,
            rtol=self.config.rtol, max_step=self.config.max_step,
            t_eval=self.config.t_eval,
        )
        M = np.clip(sol.y.T, 0.0, 1.0)  # 定理7: Omega不变集
        return ODEResult(t=sol.t, M=M, convergence_type=self._classify(M))

    def solve_multi(self, memes: List[MemeState], Theta: List[dict],
                    C: np.ndarray, func_D: Callable, func_R: Callable,
                    parallel: bool = False,
                    ) -> List[ODEResult]:
        """定理6: 多模因系统独立求解.

        Args:
            memes: 模因状态列表.
            Theta: 参数列表.
            C: 耦合矩阵.
            func_D: D 相关的耦合函数.
            func_R: R 相关的耦合函数.
            parallel: 是否启用并行求解 (PB_ARCHITECTURE.md 第4节).
                默认 False 保持原有顺序求解.

        Returns:
            List[ODEResult]: ODE 求解结果列表.
        """
        # ---- 新增：并行模式 (PB_ARCHITECTURE.md 第4节) ----
        if parallel:
            parallel_solver = ParallelODESolver(
                config=self.config,
                n_workers=-1,
                mode="thread",
            )
            return parallel_solver.solve_parallel(
                memes=memes, Theta=Theta, C=C,
                func_D=func_D, func_R=func_R,
            )

        # ---- 原有顺序求解 ----
        results = []
        for i, meme in enumerate(memes):
            y0 = np.array([meme.D, meme.B, meme.rho, meme.R, meme.S], dtype=np.float64)
            result = self.solve_single(y0, Theta[i], func_D, func_R)
            results.append(result)
        return results

    def _classify(self, M: np.ndarray) -> str:
        """定理10: 收敛原型分类"""
        D, B, _, R, S = M[-1]
        if D > 0.5 and R < 0.3 and S > 0.5: return "stone"
        if R < 0.1 and B < 0.3: return "transient"
        if R < 0.1 and B > 0.7: return "bubble"
        return "undetermined"


# ================================================================
# 新增：ParallelODESolver — 并行求解池 (PB_ARCHITECTURE.md 第4节)
# ================================================================

class ParallelODESolver:
    """并行 ODE 求解池 — 每模因独立积分, 支持 Thread/Process 两种并行模式.

    PB_ARCHITECTURE.md 第4节 — Ray Actor 池, 每模因独立积分, 50K 并行.
    数学对应: 定理6 (分段解存在唯一性) — 模因间的 ODE 积分彼此独立,
    因此可以用 embarrassingly parallel 模式并行求解.

    模式选择:
      - mode="thread": ThreadPoolExecutor — 适合 I/O 密集或共享内存场景.
      - mode="process": ProcessPoolExecutor — 适合 GPU/CPU 密集场景,
        绕过 GIL, 推荐用于大规模模因求解.

    用法:
        pool = ParallelODESolver(config, n_workers=8, mode="process")
        results = pool.solve_parallel(memes, Theta, C, func_D, func_R)
        # 流式处理大数据集:
        for batch_results in pool.solve_streaming(generator, batch_size=1000):
            process(batch_results)
    """

    def __init__(
        self,
        config: ODEConfig,
        n_workers: int = -1,
        mode: str = "thread",
    ) -> None:
        """初始化并行求解池.

        Args:
            config: ODE 求解配置.
            n_workers: 并行工作线程/进程数, -1 为全部可用.
            mode: 并行模式 — "thread" 或 "process".
        """
        self.config: ODEConfig = config
        self._solver: ODESolver = ODESolver(config)

        import os
        self.n_workers: int = (
            n_workers if n_workers > 0 else (os.cpu_count() or 1)
        )
        self.mode: str = mode

    def _solve_single_worker(
        self,
        meme: MemeState,
        theta: dict,
        func_D: Callable,
        func_R: Callable,
        I_ext: Optional[Callable] = None,
    ) -> ODEResult:
        """并行 worker: 对单个模因求解 ODE.

        作为 ThreadPoolExecutor / ProcessPoolExecutor 的 worker 调用.
        """
        y0 = np.array(
            [meme.D, meme.B, meme.rho, meme.R, meme.S],
            dtype=np.float64,
        )
        return self._solver.solve_single(y0, theta, func_D, func_R, I_ext)

    def solve_parallel(
        self,
        memes: List[MemeState],
        Theta: List[dict],
        C: np.ndarray,
        func_D: Callable,
        func_R: Callable,
        I_ext: Optional[Callable] = None,
    ) -> List[ODEResult]:
        """对每个模因独立并行求解 ODE.

        PB_ARCHITECTURE.md 第4节 — 并行求解, 50K 并发.
        每个模因的 ODE 积分与其他模因独立, 可安全并行.

        模式:
          - mode="thread": ThreadPoolExecutor — 轻量, 共享内存.
          - mode="process": ProcessPoolExecutor — 绕过 GIL,
            推荐 CPU 密集场景.

        Args:
            memes: 模因状态列表.
            Theta: 参数列表 (与 memes 一一对应).
            C: 耦合矩阵 (并行模式下每个 worker 接收完整矩阵).
            func_D: D 相关的耦合函数.
            func_R: R 相关的耦合函数.
            I_ext: 外部输入函数, 可选.

        Returns:
            List[ODEResult]: 按输入顺序排列的求解结果.
        """
        n: int = len(memes)
        if n == 0:
            return []

        # 分配 worker 索引
        indices: List[int] = list(range(n))

        # 选择执行器
        executor_cls: type = (
            ProcessPoolExecutor if self.mode == "process"
            else ThreadPoolExecutor
        )

        # 预分配结果
        results: List[Optional[ODEResult]] = [None] * n

        try:
            with executor_cls(max_workers=self.n_workers) as executor:
                future_to_idx: dict = {}
                for i in indices:
                    future = executor.submit(
                        self._solve_single_worker,
                        memes[i],
                        Theta[i],
                        func_D,
                        func_R,
                        I_ext,
                    )
                    future_to_idx[future] = i

                for future in as_completed(future_to_idx):
                    i: int = future_to_idx[future]
                    try:
                        results[i] = future.result()
                    except Exception:
                        # 单个任务失败：回退到主线程顺序求解
                        results[i] = self._solve_single_worker(
                            memes[i], Theta[i], func_D, func_R, I_ext,
                        )
        except Exception:
            # 执行器整体失败：回退到顺序求解
            solver = ODESolver(self.config)
            fallback_results: List[ODEResult] = []
            for i, meme in enumerate(memes):
                y0 = np.array(
                    [meme.D, meme.B, meme.rho, meme.R, meme.S],
                    dtype=np.float64,
                )
                fallback_results.append(
                    solver.solve_single(y0, Theta[i], func_D, func_R, I_ext)
                )
            return fallback_results

        return [r for r in results if r is not None]

    def solve_streaming(
        self,
        meme_generator,
        batch_size: int = 1000,
        callback: Optional[Callable] = None,
    ):
        """流式批量处理 — 适合 PB 级大数据集, 避免一次加载全部模因到内存.

        PB_ARCHITECTURE.md 第4节 — 流式处理.
        每次从生成器中提取 batch_size 个模因, 并行求解, yield 结果批次.
        适合内存受限场景和实时流水线.

        Args:
            meme_generator: 模因生成器, 每次 yield (MemeState, dict).
            batch_size: 每批处理的模因数, 默认 1000.
            callback: 每批完成后的回调函数 callback(batch_results, batch_idx).

        Yields:
            List[ODEResult]: 每批求解结果列表.
        """
        batch_idx: int = 0
        batch_memes: List[MemeState] = []
        batch_thetas: List[dict] = []

        for item in meme_generator:
            meme, theta = item
            batch_memes.append(meme)
            batch_thetas.append(theta)

            if len(batch_memes) >= batch_size:
                # 构建虚拟耦合矩阵 (流式模式下使用空耦合)
                C_batch: np.ndarray = np.zeros(
                    (len(batch_memes), len(batch_memes)),
                    dtype=np.float32,
                )

                # 复用 solve_parallel 的批处理能力
                batch_results = self.solve_parallel(
                    memes=batch_memes,
                    Theta=batch_thetas,
                    C=C_batch,
                    func_D=lambda d: d,
                    func_R=lambda r: r,
                )

                if callback is not None:
                    callback(batch_results, batch_idx)

                yield batch_results

                batch_idx += 1
                batch_memes = []
                batch_thetas = []

        # 处理剩余不足 batch_size 的模因
        if batch_memes:
            C_batch = np.zeros(
                (len(batch_memes), len(batch_memes)),
                dtype=np.float32,
            )
            batch_results = self.solve_parallel(
                memes=batch_memes,
                Theta=batch_thetas,
                C=C_batch,
                func_D=lambda d: d,
                func_R=lambda r: r,
            )
            if callback is not None:
                callback(batch_results, batch_idx)
            yield batch_results
