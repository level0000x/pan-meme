//! Phase 0: 全局优化器
//!
//! 数学对应：附录 D.3 假设 0 — H = T × F × Θ × N 空间搜索
//!
//! 在五维超参数空间中搜索最优组合 h* = (T*, F*, Θ*, N*) 以最小化：
//!   L(h; I) = ||Φ_h⁻¹(Φ_h(I)) - I||_U + λ·Complexity(h)
//!
//! 搜索策略：Split optimization
//!   - 离散空间 (T, F, N)：完全枚举
//!   - 连续空间 Θ：网格搜索（替代 L-BFGS-B）
//!
//! 简化版损失：由于完整管线 Φ_h⁻¹ ∘ Φ_h 开销过大，使用 memoized
//! node_texts 重建误差作为代理度量，具体包括：
//!   1. 边损失：低于阈值 T 被丢弃的 Jaccard 边比例
//!   2. 分量失配：|连通分量数(T) − 目标N| / max(分量数, N)
//!   3. Θ 惩罚：函数族 Lipschitz 常数 → 编码失真度
//!
//! Complexity(h) = |N| + n_params(F)
//!
//! 搜索空间：
//!   T ∈ {0.4,0.5,0.6,0.7,0.8,0.9,0.95}
//!   F ∈ {Power, Exp, Sigmoid, Log, Piecewise}
//!   N ∈ [1, |V|]

use crate::emergence::extractor::RelationNetwork;
use crate::sealing::ode::FunctionFamily;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════
// 优化器配置
// ═══════════════════════════════════════════════════════════════════════

/// 优化器配置
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// 复杂度正则化权重 λ（惩罚过复杂的假设）
    pub lambda: f64,
    /// T 的候选值集合（覆盖中低敏感区域）
    pub t_values: Vec<f64>,
    /// N 的采样步长（N ∈ [1, |V|]，步长控制粒度）
    pub n_step: usize,
    /// 是否输出详细搜索轨迹
    pub verbose: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            t_values: vec![0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            n_step: 1,
            verbose: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Θ 网格定义 — 每个函数族的连续参数候选值
// ═══════════════════════════════════════════════════════════════════════

/// 为给定函数族构建 Θ 网格。
///
/// 网格密度平衡了搜索精度与计算开销：
///   - Power/Exp/Log：1 个参数，~6 个候选值
///   - Sigmoid：2 个参数，k × x₀ = 4 × 4 = 16 个候选
///   - Piecewise：2 个参数（b₁, b₂），4 × 4 = 16 个候选
fn build_theta_grid(family: FunctionFamily) -> Vec<Vec<f64>> {
    match family {
        FunctionFamily::Power | FunctionFamily::Exp | FunctionFamily::Log => {
            // 单参数：k ∈ [0.5, 2.5]，6 个候选
            vec![
                vec![0.5],
                vec![0.75],
                vec![1.0],
                vec![1.25],
                vec![1.5],
                vec![2.0],
                vec![2.5],
            ]
        }
        FunctionFamily::Sigmoid => {
            // 双参数：(k, x₀)
            let ks = [1.0_f64, 2.0, 4.0, 6.0];
            let x0s = [0.25_f64, 0.4, 0.5, 0.6, 0.75];
            let mut grid = Vec::new();
            for &k in &ks {
                for &x0 in &x0s {
                    grid.push(vec![k, x0]);
                }
            }
            grid
        }
        FunctionFamily::Piecewise => {
            // 双参数：(b₁, b₂) where b₁ < b₂
            let b1s = [0.2_f64, 0.3, 0.4];
            let b2s = [0.6_f64, 0.7, 0.8];
            let mut grid = Vec::new();
            for &b1 in &b1s {
                for &b2 in &b2s {
                    if b1 < b2 {
                        grid.push(vec![b1, b2]);
                    }
                }
            }
            grid
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 边与分量计算（用于代理重建误差）
// ═══════════════════════════════════════════════════════════════════════

/// 字符 Jaccard 图中连接强度 ≥ t 的边数（保留的 + 总数）。
fn count_edges_above_threshold(psi: &RelationNetwork, t: f64) -> (usize, usize) {
    let mut kept = 0usize;
    let mut total = 0usize;

    for row in &psi.char_jaccard {
        for &(_, j) in row {
            if j >= t {
                kept += 1;
            }
            total += 1;
        }
    }

    // char_jaccard 是对称存储的（每对出现两次），除以 2
    kept /= 2;
    total /= 2;

    (kept, total)
}

/// 计算字符 Jaccard 图在阈值 t 下的连通分量数。
fn compute_char_components(psi: &RelationNetwork, t: f64, n_char: usize) -> usize {
    // 构建邻接表（仅保留 Jaccard ≥ t 的边）
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_char];
    for ci in 0..n_char {
        for &(cj, j) in &psi.char_jaccard[ci] {
            if j >= t {
                adj[ci].push(cj);
            }
        }
    }

    // DFS 计数连通分量
    let mut visited = vec![false; n_char];
    let mut components = 0usize;

    for ci in 0..n_char {
        if visited[ci] {
            continue;
        }
        components += 1;
        let mut stack = vec![ci];
        visited[ci] = true;
        while let Some(u) = stack.pop() {
            for &v in &adj[u] {
                if !visited[v] {
                    visited[v] = true;
                    stack.push(v);
                }
            }
        }
    }

    components
}

/// 带缓存的连通分量查询。
fn get_or_compute_components(
    psi: &RelationNetwork,
    t: f64,
    n_char: usize,
    cache: &mut HashMap<usize, usize>,
) -> usize {
    // 用离散化的 t 作为缓存键（精确到 2 位小数）
    let t_key = (t * 100.0).round() as usize;
    if let Some(&c) = cache.get(&t_key) {
        return c;
    }
    let c = compute_char_components(psi, t, n_char);
    cache.insert(t_key, c);
    c
}

/// 复杂度度量：Complexity(h) = |N| + n_params(F)。
fn compute_complexity(n: usize, family_d: FunctionFamily, family_r: FunctionFamily) -> f64 {
    n as f64 + family_d.n_params() as f64 + family_r.n_params() as f64
}

// ═══════════════════════════════════════════════════════════════════════
// 假设 与 优化结果
// ═══════════════════════════════════════════════════════════════════════

/// 搜索空间中的单个假设 h ∈ H = T × F × Θ × N。
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// Jaccard 阈值 T ∈ (0, 1]
    pub t: f64,
    /// 函数族 Φ_D （§4.3.5 R 方程）
    pub family: FunctionFamily,
    /// Φ_D 连续参数 Θ
    pub theta: Vec<f64>,
    /// 函数族 Φ_R （§4.3.6 S 方程）
    pub family_r: FunctionFamily,
    /// Φ_R 连续参数 Θ
    pub theta_r: Vec<f64>,
    /// 目标模因数 N ∈ [1, |V|]
    pub n: usize,
}

/// 全局优化结果。
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// 最优假设 h*
    pub best: Hypothesis,
    /// 最小损失值 L(h*)
    pub loss: f64,
    /// 最优假设的重建误差项
    pub rec_error: f64,
    /// 最优假设的复杂度项
    pub complexity: f64,
    /// 搜索空间中评估的假设总数
    pub n_evaluated: usize,
    /// 损失分布的简要统计（用于诊断）
    pub loss_min: f64,
    pub loss_max: f64,
    pub loss_avg: f64,
}

impl std::fmt::Display for OptimizationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "全局优化结果")?;
        writeln!(f, "  最优假设 h*:")?;
        writeln!(f, "    T     = {:.2}", self.best.t)?;
        writeln!(f, "    F_D   = {:?}", self.best.family)?;
        writeln!(f, "    Θ_D   = {:?}", self.best.theta)?;
        writeln!(f, "    F_R   = {:?}", self.best.family_r)?;
        writeln!(f, "    Θ_R   = {:?}", self.best.theta_r)?;
        writeln!(f, "    N     = {}", self.best.n)?;
        writeln!(f, "  损失 L(h*)    = {:.6}", self.loss)?;
        writeln!(f, "    重建误差    = {:.6}", self.rec_error)?;
        writeln!(f, "    复杂度      = {:.6}", self.complexity)?;
        writeln!(f, "  搜索空间: {} 个假设", self.n_evaluated)?;
        writeln!(f, "  损失范围: [{:.6}, {:.6}], avg={:.6}", self.loss_min, self.loss_max, self.loss_avg)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 优化器
// ═══════════════════════════════════════════════════════════════════════

/// 全局优化器。
///
/// 在超参数空间 H = T × F_D × Θ_D × F_R × Θ_R × N 中搜索使损失函数最小的假设 h*。
///
/// 数学对应：§D.3 假设0 — 超参数优化
///
/// 搜索顺序：
///   for T ∈ T_values:
///     for F_D, Θ_D ∈ family_theta_pairs:
///       for F_R, Θ_R ∈ family_theta_pairs:
///         for N ∈ [1..|V|]:
///           loss = proxy_rec_error + λ · complexity
pub struct Optimizer {
    /// 输入关系网络 Ψ = (V, E, w)
    input: RelationNetwork,
}

impl Optimizer {
    /// 从关系网络创建优化器。
    ///
    /// `input` 是从原始词列表构建的完整 Relational Network，
    /// 其 node_texts 作为 memoized 文本用于重建误差计算。
    pub fn new(input: RelationNetwork) -> Self {
        Self { input }
    }

    /// 运行全局优化搜索。
    ///
    /// 返回 OptimizationResult 包含最优假设 h* 和对应的最小 loss。
    ///
    /// 算法：Split optimization
    ///   - 离散参数 (T, F, N)：穷举枚举
    ///   - 连续参数 Θ：网格搜索
    pub fn optimize(&self, config: &OptimizerConfig) -> OptimizationResult {
        let n_char = self.input.char_to_words.len();
        let n_max = self.input.node_count();
        let families: [FunctionFamily; 5] = FunctionFamily::ALL;

        // 预构建所有 N 值
        let step = config.n_step.max(1);
        let n_values: Vec<usize> = (1..=n_max).step_by(step).collect();

        // 预构建所有 (F_D, Θ_D) 组合
        let mut family_theta_pairs: Vec<(FunctionFamily, Vec<f64>)> = Vec::new();
        for &family in &families {
            for theta in build_theta_grid(family) {
                family_theta_pairs.push((family, theta));
            }
        }

        let ft_count = family_theta_pairs.len();
        let total_combinations = config.t_values.len()
            * ft_count * ft_count  // F_D × F_R
            * n_values.len();

        if config.verbose {
            eprintln!(
                "  搜索空间: {}T × {}F_DΘ × {}F_RΘ × {}N = {} 候选",
                config.t_values.len(),
                ft_count,
                ft_count,
                n_values.len(),
                total_combinations,
            );
        }

        let mut best: Option<Hypothesis> = None;
        let mut best_loss = f64::MAX;
        let mut best_rec = 0.0;
        let mut best_cplx = 0.0;

        let mut loss_min = f64::MAX;
        let mut loss_max = f64::MIN;
        let mut loss_sum = 0.0;
        let mut n_evaluated = 0usize;

        // 组件缓存：同一 T 值下组件数不变，避免重复 DFS
        let mut component_cache: HashMap<usize, usize> = HashMap::new();
        // 边损失缓存：同一 T 值下边损失固定
        let mut edge_loss_cache: HashMap<usize, f64> = HashMap::new();

        for &t in &config.t_values {
            // 预计算当前 T 下的边损失（与 F, Θ, N 无关）
            let t_key = (t * 100.0).round() as usize;
            let edge_loss = *edge_loss_cache.entry(t_key).or_insert_with(|| {
                let (kept, total) = count_edges_above_threshold(&self.input, t);
                if total > 0 {
                    1.0 - (kept as f64 / total as f64)
                } else {
                    0.0
                }
            });

            for &(family_d, ref theta_d) in &family_theta_pairs {
                let lipschitz_d = family_d.lipschitz_constant(theta_d);

                for &(family_r, ref theta_r) in &family_theta_pairs {
                    // Θ 惩罚：max(L_D, L_R)（§4.3.5-4.3.6 两个 Φ 联合惩罚）
                    let lipschitz_r = family_r.lipschitz_constant(theta_r);
                    let lipschitz = lipschitz_d.max(lipschitz_r);
                    let theta_penalty = (lipschitz / 20.0).min(1.0) * 0.005;

                for &n in &n_values {
                    // 分量失配
                    let actual_components =
                        get_or_compute_components(&self.input, t, n_char, &mut component_cache);
                    let component_error = {
                        let denom = actual_components.max(n).max(1) as f64;
                        (actual_components as isize - n as isize).abs() as f64 / denom
                    };

                    // 重建误差
                    let rec_error = edge_loss + component_error + theta_penalty;

                    // 复杂度
                    let complexity = compute_complexity(n, family_d, family_r);

                    // 总损失：L(h; I) = rec_error + λ · complexity
                    let loss = rec_error + config.lambda * complexity;

                    n_evaluated += 1;
                    loss_sum += loss;
                    if loss < loss_min {
                        loss_min = loss;
                    }
                    if loss > loss_max {
                        loss_max = loss;
                    }

                    if loss < best_loss {
                        best_loss = loss;
                        best_rec = rec_error;
                        best_cplx = complexity;
                        best = Some(Hypothesis {
                            t,
                            family: family_d,
                            theta: theta_d.clone(),
                            family_r,
                            theta_r: theta_r.clone(),
                            n,
                        });
                    }
                }
            }
            }
        }

        let loss_avg = if n_evaluated > 0 {
            loss_sum / n_evaluated as f64
        } else {
            0.0
        };

        OptimizationResult {
            best: best.unwrap_or(Hypothesis {
                t: 0.5,
                family: FunctionFamily::Power,
                theta: vec![1.0],
                family_r: FunctionFamily::Power,
                theta_r: vec![1.0],
                n: 1,
            }),
            loss: best_loss,
            rec_error: best_rec,
            complexity: best_cplx,
            n_evaluated,
            loss_min,
            loss_max,
            loss_avg,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 便捷入口函数
// ═══════════════════════════════════════════════════════════════════════

/// 优化的便捷入口函数。
///
/// 用法：
/// ```ignore
/// let psi = RelationNetwork::from_words(words, &config);
/// let opt_cfg = OptimizerConfig::default();
/// let result = optimizer::optimize(psi, &opt_cfg);
/// println!("最优 T={:.2} F={:?} N={}", result.best.t, result.best.family, result.best.n);
/// ```
pub fn optimize(input: RelationNetwork, config: &OptimizerConfig) -> OptimizationResult {
    let opt = Optimizer::new(input);
    opt.optimize(config)
}

// ═══════════════════════════════════════════════════════════════════════
// 辅助：从优化结果生成可用的 OdeConfig 和 threshold
// ═══════════════════════════════════════════════════════════════════════

/// 将优化结果转换为 OdeConfig（用于 Phase 5）。
///
/// 注意：此函数在 crate::ode::OdeConfig 可用时才可调用。
pub fn result_to_ode_config(result: &OptimizationResult) -> crate::sealing::ode::OdeConfig {
    use crate::sealing::ode::OdeConfig;
    OdeConfig {
        function_family: result.best.family as usize,
        fn_params: result.best.theta.clone(),
        function_family_r: result.best.family_r as usize,
        fn_params_r: result.best.theta_r.clone(),
        i_ext: 0.0,
        ..Default::default()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emergence::extractor::{ExtractorConfig, RelationNetwork};

    /// 构建最小测试用 RelationNetwork。
    fn make_test_psi() -> RelationNetwork {
        let words = vec![
            "花园".to_string(),
            "花".to_string(),
            "园".to_string(),
            "公园".to_string(),
            "公".to_string(),
            "园林".to_string(),
            "林".to_string(),
        ];
        let config = ExtractorConfig::default();
        RelationNetwork::from_words(words, &config)
    }

    #[test]
    fn test_count_edges_at_threshold() {
        let psi = make_test_psi();
        let (kept_low, total) = count_edges_above_threshold(&psi, 0.0);
        let (kept_high, _) = count_edges_above_threshold(&psi, 0.99);
        assert!(kept_low >= kept_high);
        assert!(total > 0);
    }

    #[test]
    fn test_compute_char_components() {
        let psi = make_test_psi();
        let n_char = psi.char_to_words.len();
        // 阈值 0: 所有边保留，应形成较少的分量
        let c0 = compute_char_components(&psi, 0.0, n_char);
        // 阈值 0.99: 几乎无边，应为每个字独立成岛
        let c99 = compute_char_components(&psi, 0.99, n_char);
        assert!(c0 <= c99, "c0={} c99={}", c0, c99);
        assert!(c99 <= n_char);
    }

    #[test]
    fn test_optimize_small() {
        let psi = make_test_psi();
        let config = OptimizerConfig {
            lambda: 0.1,
            t_values: vec![0.5, 0.7, 0.9],
            n_step: 2,
            verbose: false,
        };
        let result = optimize(psi, &config);
        assert!(result.n_evaluated > 0);
        assert!(result.loss >= 0.0);
        assert!(result.loss.is_finite());
        assert!((0.0..=1.0).contains(&result.best.t), "T={}", result.best.t);
        assert!(result.best.n >= 1);
    }

    #[test]
    fn test_optimize_all_families() {
        let psi = make_test_psi();
        let node_count = psi.node_count();  // 在 move 之前获取
        let config = OptimizerConfig::default();
        let result = optimize(psi, &config);
        // 应遍历全部 5 个函数族
        let families_visited = FunctionFamily::ALL.len();
        let ft_pairs = families_visited * 7 + 20 + 9; // rough upper bound
        let _expected = config.t_values.len() * ft_pairs * node_count;
        // 只是粗略检查不为空
        assert!(result.n_evaluated > 0);
        eprintln!("Search space: {} hypotheses evaluated", result.n_evaluated);
        eprintln!("Best: T={:.2} F_D={:?} Θ_D={:?} F_R={:?} Θ_R={:?} N={} loss={:.6}",
            result.best.t, result.best.family, result.best.theta, result.best.family_r, result.best.theta_r, result.best.n, result.loss);
    }

    #[test]
    fn test_result_to_ode_config() {
        let psi = make_test_psi();
        let config = OptimizerConfig {
            lambda: 0.1,
            t_values: vec![0.5],
            n_step: 5,
            verbose: false,
        };
        let result = optimize(psi, &config);
        let ode_cfg = result_to_ode_config(&result);
        assert_eq!(ode_cfg.function_family, result.best.family as usize);
        assert_eq!(ode_cfg.fn_params, result.best.theta);
        assert_eq!(ode_cfg.function_family_r, result.best.family_r as usize);
        assert_eq!(ode_cfg.fn_params_r, result.best.theta_r);
        assert!((ode_cfg.i_ext - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_build_theta_grid_power() {
        let grid = build_theta_grid(FunctionFamily::Power);
        assert!(!grid.is_empty());
        for g in &grid {
            assert_eq!(g.len(), 1);
        }
    }

    #[test]
    fn test_build_theta_grid_sigmoid() {
        let grid = build_theta_grid(FunctionFamily::Sigmoid);
        assert!(grid.len() >= 10); // 4 k × 5 x0 = 20
        for g in &grid {
            assert_eq!(g.len(), 2);
        }
    }

    #[test]
    fn test_build_theta_grid_piecewise() {
        let grid = build_theta_grid(FunctionFamily::Piecewise);
        assert!(grid.len() >= 5); // 3 b1 × 3 b2 = 9, all b1 < b2
        for g in &grid {
            assert_eq!(g.len(), 2);
            assert!(g[0] < g[1], "b1={} >= b2={}", g[0], g[1]);
        }
    }
}
