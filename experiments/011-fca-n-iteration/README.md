# 实验 011 — FCA N迭代收敛证明

## 目标

验证定理 6.17D：FCA 形式的 N 迭代逐层压缩的收敛性。

## 脚本索引

### 审计与验证（按优先级排序）

| 脚本 | 用途 |
|:---|:---|
| `a2_final_closure.py` | A2 参数的最终闭合性验证 |
| `a2_interval_tightening.py` | A2 区间收紧分析 |
| `audit_6_17b_comprehensive.py` | 6.17B 综合审计 |
| `audit_6_17C_proof.py` | 6.17C 证明审计 |
| `audit_6_17D_v7.py` | 6.17D 核心审计 v7 |
| `audit_cd_deep.py` | C-D 算子深度审计 |
| `audit_v10_systematic.py` | 系统性审计 v10 |
| `audit_v11_6_17d_upgrade.py` | 6.17D 升级版审计 v11 |

### 核心证明管线

| 脚本 | 用途 |
|:---|:---|
| `analyze_nk_contraction.py` | N-K 收缩分析 |
| `prove_leaf_spectral.py` | 叶谱证明 |
| `bregman_global_v2.py` | Bregman 全局收敛 v2 |
| `kl_lyapunov_v3.py` | KL-Lyapunov 函数 v3 |
| `prove_r_case2_fixed.py` | R 情况 2 修复版证明 |
| `proof_kl_lyapunov.py` | KL-Lyapunov 完整证明 |

### φ-IPP 系列（信息保持投射）

| 脚本 | 用途 |
|:---|:---|
| `phipp_compactness.py` | φ-IPP 紧性证明 |
| `phipp_corrected_attack.py` | φ-IPP 修正攻击 |
| `phipp_deep_structure.py` | φ-IPP 深层结构 |
| `phipp_effective_theory.py` | φ-IPP 有效理论 |
| `phipp_proof_refined.py` | φ-IPP 精炼证明 |
| `phipp_reachable_domain.py` | φ-IPP 可达域 |
| `phipp_taylor_radius.py` | φ-IPP Taylor 半径 |
| `phipp_tight_bound.py` | φ-IPP 紧界 |
| `phipp_variance_decomposition.py` | φ-IPP 方差分解 |

### 收敛性证明

| 脚本 | 用途 |
|:---|:---|
| `c17_global_attack.py` | 定理 17 全局攻击 |
| `c17c_analytic_closure.py` | 17C 解析闭合 |
| `c18_global_convergence.py` | 定理 18 全局收敛 |
| `prove_global_convergence.py` | 全局收敛证明 |
| `prove_global_uniqueness.py` | 全局唯一性证明 |
| `explore_global_convergence.py` | 全局收敛探索 |
| `verify_global_convergence_gap.py` | 全局收敛间隙验证 |

### 验证与回归

| 脚本 | 用途 |
|:---|:---|
| `audit_final_regression.py` | 最终回归审计 |
| `final_regression_v5.py` | 回归测试 v5 |
| `deep_kl_analysis.py` | KL 深度分析 |
| `deep_verify_proofs.py` | 深度证明验证 |
| `verify_core_gap1.py` | 核心 Gap1 验证 |
| `verify_nk_contraction_final.py` | NK 收缩最终验证 |
| `stress_test_v4.py` | 压力测试 v4 |

### E1 谱方法

| 脚本 | 用途 |
|:---|:---|
| `e1_heat_trace.py` | E1 热迹方法 |
| `e1_semantic_pipeline.py` | E1 语义管线 |
| `e1_spectral_v2.py` | E1 谱分析 v2 |
| `e1_v5_fixed_points.py` | E1 不动点 v5 |

### FCA / 格方法

| 脚本 | 用途 |
|:---|:---|
| `fca_lattice.py` | FCA 格构造 |
| `analyze_leaf_jacobian.py` | 叶 Jacobi 分析 |
| `analytic_derivative_bounds.py` | 解析导数界 |

### 辅助探索

| 脚本 | 用途 |
|:---|:---|
| `n_iteration_v5.py` | N迭代 v5 |
| `seed11_final_proof.py` | Seed11 最终证明 |
| `seed11_proof_audit.py` | Seed11 证明审计 |
| `seed11_bound_validation.py` | Seed11 界验证 |
| `spectral_params.py` | 谱参数分析 |
| `spectral_v4.py` | 谱方法 v4 |
| `radial_convexity.py` | 径向凸性分析 |
| `radial_proof_explore.py` | 径向证明探索 |
| `new_proof_avenues.py` | 新证明路线 |

## 运行顺序

```
1. fca_lattice.py         → 构建 FCA 格底
2. audit_6_17D_v7.py      → 核心 6.17D 审计
3. phipp_proof_refined.py → φ-IPP 精炼证明
4. prove_global_convergence.py → 全局收敛
5. audit_final_regression.py   → 回归验证
```

## 归档脚本

`scripts_archive/` 中为早期版本，保留供历史参考。
