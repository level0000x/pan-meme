"""
模块3 — 模因化层: 11参数推导器 (ParamDerive)

数学对应: 论文4.3.7节 + 3.4.3节 — 11个动力学参数由几何特征与模因状态闭式确定.

核心思想:
    对于每个子几何体 G_i = (K_i, g_i, ω_i, Γ_i, R_i) 及其对应的模因状态 X_i = (m_i, ξ_i),
    参数 Θ_i = {α₁, α₂, β₁, β₂, γ₁, γ₂, δ₁, δ₂, δ₃, ε₁, ε₂} 中的每一项
    都由几何不变量或模因原始维度的简单代数表达式给出, 无需数值优化.

推导依据:
    - 定理4 (Φ_D: G→Q 双射): 参数 Θ_i 由 G_i 的几何特征唯一确定.
    - 定义6 (Ω = [0,1]^5): 所有参数确保模因状态在定义域内演化.
    - 定理7 (Ω不变集): 参数约束保证解不逃离状态空间.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from pan_meme.core.types import GeometricObject, MemeState

# ============================================================
# 配置常量
# ============================================================

# 论文4.3.7节: 最大深度上限 d_max = 20
# 该值是基于层级结构经验设定的归一化因子, 用于 α₂ 的计算
_D_MAX: int = 20


class ParamDerive:
    """
    11参数推导器: 由几何特征 + 模因状态闭式确定全部动力学参数.

    数学对应:
        - 论文4.3.7节: α₁(简化效应), α₂(沉淀效应), β₁(扩张耦合), β₂(泛化权衡),
                        γ₁(能流耗散), γ₂(外部赋能), δ₁(核心驱动力), δ₂(深度诅咒),
                        δ₃(自发衰退), ε₁(深度基石), ε₂(速朽定律).
        - 定理6 (分段解): 这些参数作为 ODE 系统的系数, 决定解的定性行为.
        - 定理10 (三类收敛原型): 参数组合决定模因的终极收敛类型.

    用法:
        theta: Dict[str, float] = ParamDerive.derive(meme, sub_geo)
    """

    @staticmethod
    def derive(
        meme: MemeState,
        sub_geo: GeometricObject,
    ) -> Dict[str, float]:
        """
        11参数闭式推理: 由几何特征与模因状态一次性计算全部动力学参数.

        数学注解:
            α₁ = min(2·|E| / (|V|·(|V|-1)), 1.0)  — 简化效应: 边密度刻画结构复杂度的归一化度量
            α₂ = min(depth / d_max, 1.0)            — 沉淀效应: 层级深度越大, 信息越牢固
            β₁ = 0.5                                — 扩张耦合: 默认中等耦合强度 (可配置)
            β₂ = D / max(B, 1e-6)                   — 泛化权衡: 内禀度与关联度之比
            γ₁ = mean(|∇ω|)                          — 能流耗散: 场梯度均值反映信息流摩擦
            γ₂ = 0.3                                — 外部赋能: 默认较低外源输入
            δ₁ = R · B                              — 核心驱动力: 演化速率与关联度的乘性耦合
            δ₂ = D · depth                           — 深度诅咒: 复杂度随深度线性增长
            δ₃ = 1 / (1 + |Γ|)                      — 自发衰退: 约束越多, 衰退越慢
            ε₁ = D · S                              — 深度基石: 内禀度与结构韧度的乘积
            ε₂ = σ(|∇ω|) · μ(|∇ω|)                  — 速朽定律: 梯度波动×均值, 场越粗糙衰退越快

        Args:
            meme: 模因状态 X_i = (m_i, ξ_i), 提供 D, B, ρ, R, S 五个核心维度.
            sub_geo: 子几何体 G_i = (K_i, g_i, ω_i, Γ_i, R_i), 提供胞腔复形和场结构.

        Returns:
            Dict[str, float]: 11个动力学参数组成的字典, 键名如下:
                - "alpha_1", "alpha_2"  — 简化效应 / 沉淀效应
                - "beta_1", "beta_2"    — 扩张耦合 / 泛化权衡
                - "gamma_1", "gamma_2"   — 能流耗散 / 外部赋能
                - "delta_1", "delta_2", "delta_3" — 核心驱动力 / 深度诅咒 / 自发衰退
                - "epsilon_1", "epsilon_2" — 深度基石 / 速朽定律
        """
        K = sub_geo.K
        omega = sub_geo.omega
        Gamma = sub_geo.Gamma
        g = sub_geo.g

        # ---- 提取基础量 ----
        # |V|: 胞腔复形的顶点数 (0-单形数量)
        n_vertices: int = len(K.vertices)
        # |E|: 胞腔复形的边数 (1-单形数量)
        n_edges: int = len(K.edges)

        # depth: 层级深度, 由 level_labels 的最大值确定
        # 若无层级标签则深度为 1 (顶层)
        depth: float = float(max(K.level_labels.values())) if K.level_labels else 1.0

        # ---- 计算 ω 场的离散梯度 ----
        # ∇ω 沿每条边的梯度分量: grad_k = |ω[i] - ω[j]| / max(g_k, 1e-6)
        # 返回数组 shape=(|E|,) 或全零若无法计算
        omega_grad = ParamDerive._compute_omega_gradient(omega, K, g)

        # ---- α₁: 简化效应 = 边密度 ----
        # 公式: α₁ = min(2·|E| / (|V|·(|V|-1)), 1.0)
        # 数学含义: 完全图边数为 |V|·(|V|-1)/2, 实际边数与完全图之比刻画结构丰富度
        # 除零保护: n_vertices < 2 时边密度为 0
        if n_vertices > 1:
            edge_density: float = (2.0 * n_edges) / (n_vertices * (n_vertices - 1))
            alpha_1: float = min(edge_density, 1.0)
        else:
            alpha_1 = 0.0

        # ---- α₂: 沉淀效应 = 层级深度归一化 ----
        # 公式: α₂ = min(depth / d_max, 1.0)
        # 数学含义: 层级越深, 信息越根深蒂固, 沉淀效应越强
        alpha_2: float = min(depth / _D_MAX, 1.0)

        # ---- β₁: 扩张耦合 = 0.5 (默认) ----
        # 公式: β₁ = 0.5
        # 数学含义: 中等耦合强度, 可在后续管线中通过全局优化器调优
        beta_1: float = 0.5

        # ---- β₂: 泛化权衡 = D / max(B, 1e-6) ----
        # 公式: β₂ = D / max(B, 1e-6)
        # 数学含义: 内禀度(自洽性)与关联度(外联性)的比率;
        #           高 β₂ → 偏内省, 低 β₂ → 偏外联
        # 除零保护: 若 B 接近 0, 用 1e-6 防止除零
        beta_2: float = meme.D / max(meme.B, 1e-6)

        # ---- γ₁: 能流耗散 = |∇ω| 的均值 ----
        # 公式: γ₁ = mean(|∇ω|), 若无法计算则为 0
        # 数学含义: ω 场梯度的平均绝对值, 反映场的不均匀程度;
        #           梯度越大 → 信息流阻力越大 → 能流耗散越强
        gamma_1: float = float(np.mean(np.abs(omega_grad))) if len(omega_grad) > 0 else 0.0

        # ---- γ₂: 外部赋能 = 0.3 (默认) ----
        # 公式: γ₂ = 0.3
        # 数学含义: 较低的外部能量注入, 对应自组织占主导的系统
        gamma_2: float = 0.3

        # ---- δ₁: 核心驱动力 = R · B ----
        # 公式: δ₁ = R · B
        # 数学含义: 演化速率与关联度的乘积, 两者协同驱动模因扩散
        #           高 δ₁ → 快速传播型模因 (如新闻类)
        delta_1: float = meme.R * meme.B

        # ---- δ₂: 深度诅咒 = D · depth ----
        # 公式: δ₂ = D · depth
        # 数学含义: 内禀复杂度与层级深度的乘积; 越深层越复杂的结构,
        #           演化面临的"深度诅咒"越严重
        delta_2: float = meme.D * depth

        # ---- δ₃: 自发衰退 = 1 / (1 + |Γ|) ----
        # 公式: δ₃ = 1 / (1 + number_of_invariant_entries)
        # 数学含义: 不变量越多 → 约束越强 → 衰退速率越慢;
        #           |Γ| 大意味着系统具有更多守恒律, 更稳定
        # 除零保护: 分母最小为 1, 确保 δ₃ ∈ (0, 1]
        gamma_entry_count: int = len(Gamma) if Gamma else 0
        delta_3: float = 1.0 / (1.0 + gamma_entry_count)

        # ---- ε₁: 深度基石 = D · S ----
        # 公式: ε₁ = D · S
        # 数学含义: 内禀度与结构韧度的乘积; 高 ε₁ 的模因具有坚实基础,
        #           类似文化中的"经典"——经得起时间考验
        epsilon_1: float = meme.D * meme.S

        # ---- ε₂: 速朽定律 = σ(|∇ω|) · μ(|∇ω|) ----
        # 公式: ε₂ = std(|∇ω|) · mean(|∇ω|), 若无法计算则为 0
        # 数学含义: 场梯度的标准差(波动性)与均值(平均强度)的乘积;
        #           场越粗糙(梯度波动大且平均梯度高) → 模因越易速朽
        if len(omega_grad) > 0:
            omega_grad_abs: np.ndarray = np.abs(omega_grad)
            epsilon_2: float = float(np.std(omega_grad_abs) * np.mean(omega_grad_abs))
        else:
            epsilon_2 = 0.0

        # ---- 组装参数字典 ----
        return {
            "alpha_1": alpha_1,
            "alpha_2": alpha_2,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "gamma_1": gamma_1,
            "gamma_2": gamma_2,
            "delta_1": delta_1,
            "delta_2": delta_2,
            "delta_3": delta_3,
            "epsilon_1": epsilon_1,
            "epsilon_2": epsilon_2,
        }

    # ============================================================
    # 内部辅助方法
    # ============================================================

    @staticmethod
    def _compute_omega_gradient(
        omega: np.ndarray,
        K: "SimplicialComplex",
        g: np.ndarray,
    ) -> np.ndarray:
        """
        计算 ω 标量场在胞腔复形 K 上的离散梯度.

        数学注解:
            对于每条边 e_k = (i, j) ∈ K.edges, 其离散梯度定义为:
                ∇ω |_k = (ω[j] - ω[i]) / max(g_k, 1e-6)

            这是有限差分法在 1-维胞腔复形上的自然推广:
            - 分子: ω 在边两端点的差值 (标量场沿边的方向导数)
            - 分母: 度量长度 g_k (对应定义3(b), g_i = w(e_i))
            - 除零保护: 若 g_k = 0, 用 1e-6 防止除零异常

        Args:
            omega: 标量场 ω ∈ R^{|V|}, shape=(|V|,), dtype 任意.
            K: 胞腔复形, 提供 edges 列表用于计算梯度.
            g: 度量向量 g ∈ R^{|E|}, shape=(|E|,), dtype 任意.

        Returns:
            np.ndarray: 离散梯度向量, shape=(|E|,), dtype=float32.
                        若 omega 为空或无边, 返回空数组 shape=(0,).
        """
        # ---- 前置检查 ----
        if len(omega) == 0 or len(K.edges) == 0:
            return np.zeros(0, dtype=np.float32)

        n_edges: int = len(K.edges)
        n_vertices: int = len(omega)
        grad: np.ndarray = np.zeros(n_edges, dtype=np.float32)

        # ---- 沿每条边计算有限差分 ----
        for k, (i, j) in enumerate(K.edges):
            # 边界保护: 确保顶点索引在 ω 的范围内
            if 0 <= i < n_vertices and 0 <= j < n_vertices:
                # 分子: ω[j] - ω[i] (沿边的方向导数)
                diff: float = float(omega[j] - omega[i])
                # 分母: 度量长度 g_k, 除零保护
                denom: float = max(float(g[k]) if k < len(g) else 0.0, 1e-6)
                grad[k] = diff / denom

        return grad
