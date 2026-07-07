# 实验零：收敛判据

## 目的

在跑任何验证实验之前，必须把"收敛成功"操作化为可测量的数字。没有这一步，任何实验结果都可以被解释为"参数还没调对"。

## 收敛判据（四条必须同时满足）

1. 每个模因的 ODE 轨迹在 t_max 内终止于 `TerminationReason::Converged`（而非 MaxSteps 或 NaN）
2. 轨迹末尾连续 5 步的 D/B/ρ/R/S 各自的标准差 < 0.001
3. 分类结果 90% 以上不是 Undetermined
4. 同一个输入数据集在多次独立运行中，原型分布一致（标记为 reproducibility_note）

## 实现

| 文件 | 变更 |
|------|------|
| `src/theory/ode.rs` | `OdeConfig` 新增 4 个收敛判据参数；`TerminationReason` 新增 `InsufficientConvergence`；`ConvergenceReport`/`ConvergenceCriteria` 结构体；`evaluate_convergence()` 函数 |
| `src/main.rs` | Phase 5 后自动输出 `[Convergence]` 报告行；`--t-max`/`--max-steps` CLI 参数 |

## 判据参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `convergence_window` | 5 | 收敛检测窗口大小（连续 N 步） |
| `convergence_threshold` | 1e-3 | 窗口首尾总变化 < 此值判定收敛 |
| `std_threshold` | 1e-3 | 轨迹末尾各维度标准差上限 |
| `undetermined_max_pct` | 0.10 | Undetermined 分类最大占比 |

## 验证结果

实验 001 中验证了判据的有效性：

- t_max=5.0 → 收敛率 0%，判据正确报告 FAIL（ODE 在 t≈5 时尚未达稳态）
- t_max=20.0 → 收敛率 100%，判据正确报告 PASS（ODE 在 t≈11 收敛）

## 提交

`8210a93` feat: 实验零 — 收敛判据落地
`766405a` feat: CLI --t-max --max-steps + 最小可行验证实验