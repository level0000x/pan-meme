//! 假设 0 全局优化 — 泛模因理论 §4.3.8, §D.5, 定理 6.2
//!
//! 核心构造：
//! - OptimizerConfig: 全局优化配置
//! - OptimalHypothesis: 最优假设 (T*, Φ_D*, Φ_R*)
//!
//! 搜索空间: H = T × F_D × Θ_D × F_R × Θ_R × N
//! 搜索策略: 网格搜索 |T| × |F_D| × |Θ_D| × |F_R| × |Θ_R| = 7 × 5 × 5 × 5 × 5 = 4375

use crate::theory::function_families::{FamilyParams, FunctionFamily};

/// 全局优化配置
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// 阈值候选值 T
    pub t_values: Vec<f64>,
    /// 全局耦合系数 λ
    pub lambda: f64,
    /// 离散化步数 n_step
    pub n_step: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        OptimizerConfig {
            t_values: vec![0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
            lambda: 0.1,
            n_step: 5,
        }
    }
}

/// 最优假设 — 假设 0 全局优化结果
#[derive(Debug, Clone)]
pub struct OptimalHypothesis {
    /// 最优阈值 T*
    pub threshold: f64,
    /// 最优 Φ_D 函数族
    pub family_d: FunctionFamily,
    /// 最优 Φ_R 函数族
    pub family_r: FunctionFamily,
    /// 最优 Φ_D 参数
    pub params_d: FamilyParams,
    /// 最优 Φ_R 参数
    pub params_r: FamilyParams,
    /// 最小损失值
    pub loss: f64,
}

/// 假设 0 全局优化 — 论文 §4.3.8, §D.5
///
/// 搜索空间: H = T × F_D × Θ_D × F_R × Θ_R × N
/// 使用网格搜索，外层遍历 T × F_D × F_R，内层在 Θ 上搜索
pub fn optimize<F>(reconstruction_error: F, config: &OptimizerConfig) -> OptimalHypothesis
where
    F: Fn(f64, &FamilyParams, &FamilyParams) -> f64,
{
    let families = FunctionFamily::all();
    let mut best = OptimalHypothesis {
        threshold: config.t_values[0],
        family_d: FunctionFamily::Power,
        family_r: FunctionFamily::Power,
        params_d: FamilyParams::new(FunctionFamily::Power, 1.0, 0.0),
        params_r: FamilyParams::new(FunctionFamily::Power, 1.0, 0.0),
        loss: f64::MAX,
    };

    for &t in &config.t_values {
        for &family_d in &families {
            for &family_r in &families {
                // 在 Θ 上搜索最优参数
                let (params_d, params_r, loss) =
                    optimize_theta(t, family_d, family_r, &reconstruction_error, config);

                if loss < best.loss {
                    best = OptimalHypothesis {
                        threshold: t,
                        family_d,
                        family_r,
                        params_d,
                        params_r,
                        loss,
                    };
                }
            }
        }
    }

    best
}

/// 在 Θ 空间上搜索最优参数
fn optimize_theta<F>(
    t: f64,
    family_d: FunctionFamily,
    family_r: FunctionFamily,
    error_fn: &F,
    config: &OptimizerConfig,
) -> (FamilyParams, FamilyParams, f64)
where
    F: Fn(f64, &FamilyParams, &FamilyParams) -> f64,
{
    let k_values: Vec<f64> = (0..=config.n_step)
        .map(|i| i as f64 * 2.0 / config.n_step as f64)
        .collect();

    let mut best_params_d = FamilyParams::new(family_d, 1.0, 0.0);
    let mut best_params_r = FamilyParams::new(family_r, 1.0, 0.0);
    let mut best_loss = f64::MAX;

    for &k_d in &k_values {
        for &k_r in &k_values {
            let params_d = FamilyParams::new(family_d, k_d, 0.5);
            let params_r = FamilyParams::new(family_r, k_r, 0.5);

            if !params_d.is_valid() || !params_r.is_valid() {
                continue;
            }

            let loss = error_fn(t, &params_d, &params_r);
            if loss < best_loss {
                best_loss = loss;
                best_params_d = params_d;
                best_params_r = params_r;
            }
        }
    }

    (best_params_d, best_params_r, best_loss)
}

/// 验证可证伪判据 — §7.3 M4
///
/// 若在至少 3 个不同数据集上，(Φ_D*, Φ_R*) 收敛到同一族，则核心假设被支持。
/// 若不同数据集需要不同函数族，且差异无法由参数差异解释，则核心假设被证伪。
pub fn verify_falsifiability(results: &[OptimalHypothesis]) -> bool {
    if results.len() < 3 {
        return true; // 不足 3 个数据集，无法判定
    }

    let first_family_d = results[0].family_d;
    let first_family_r = results[0].family_r;

    for r in &results[1..] {
        if r.family_d != first_family_d || r.family_r != first_family_r {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_basic() {
        let config = OptimizerConfig {
            t_values: vec![0.1, 0.2],
            n_step: 2,
            ..OptimizerConfig::default()
        };

        let error_fn = |_t: f64, _pd: &FamilyParams, _pr: &FamilyParams| -> f64 { 0.5 };

        let result = optimize(error_fn, &config);
        assert!(result.loss <= 0.5);
    }

    #[test]
    fn test_falsifiability_convergent() {
        let results = vec![
            OptimalHypothesis {
                threshold: 0.1,
                family_d: FunctionFamily::Power,
                family_r: FunctionFamily::Power,
                params_d: FamilyParams::new(FunctionFamily::Power, 1.0, 0.0),
                params_r: FamilyParams::new(FunctionFamily::Power, 1.0, 0.0),
                loss: 0.1,
            },
            OptimalHypothesis {
                threshold: 0.2,
                family_d: FunctionFamily::Power,
                family_r: FunctionFamily::Power,
                params_d: FamilyParams::new(FunctionFamily::Power, 1.5, 0.0),
                params_r: FamilyParams::new(FunctionFamily::Power, 0.5, 0.0),
                loss: 0.2,
            },
            OptimalHypothesis {
                threshold: 0.3,
                family_d: FunctionFamily::Power,
                family_r: FunctionFamily::Power,
                params_d: FamilyParams::new(FunctionFamily::Power, 1.2, 0.0),
                params_r: FamilyParams::new(FunctionFamily::Power, 0.8, 0.0),
                loss: 0.15,
            },
        ];
        assert!(verify_falsifiability(&results));
    }

    #[test]
    fn test_falsifiability_divergent() {
        let results = vec![
            OptimalHypothesis {
                threshold: 0.1,
                family_d: FunctionFamily::Power,
                family_r: FunctionFamily::Power,
                params_d: FamilyParams::new(FunctionFamily::Power, 1.0, 0.0),
                params_r: FamilyParams::new(FunctionFamily::Power, 1.0, 0.0),
                loss: 0.1,
            },
            OptimalHypothesis {
                threshold: 0.2,
                family_d: FunctionFamily::Exponential,
                family_r: FunctionFamily::Power,
                params_d: FamilyParams::new(FunctionFamily::Exponential, 1.0, 0.0),
                params_r: FamilyParams::new(FunctionFamily::Power, 1.0, 0.0),
                loss: 0.2,
            },
            OptimalHypothesis {
                threshold: 0.3,
                family_d: FunctionFamily::Power,
                family_r: FunctionFamily::Logarithm,
                params_d: FamilyParams::new(FunctionFamily::Power, 1.0, 0.0),
                params_r: FamilyParams::new(FunctionFamily::Logarithm, 1.0, 0.0),
                loss: 0.15,
            },
        ];
        assert!(!verify_falsifiability(&results));
    }
}
