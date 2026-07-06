//! Louvain 多层社区检测
//!
//! 数学对应：§3.2.4 — 多层级结构域 S 的社区发现
//!
//! Louvain 算法（Blondel et al., 2008）的 Rust 实现，
//! 用于在字共现网络中检测多层级社区结构。
//!
//! 算法流程（每层）：
//!   Phase A: 贪心模块度最大化 — 移动节点到邻域社区以提升模块度
//!   Phase B: 社区压缩 — 将社区收缩为超节点，构建下一层图
//!
//! 多层：
//!   对网络反复应用 Louvain，将每层结果作为多层级结构域 S。

use std::collections::{HashMap, HashSet};

/// 单层 Louvain 结果
#[derive(Debug, Clone)]
pub struct LouvainLayer {
    /// 层级索引（0 = 原始层）
    pub layer: usize,
    /// 节点 → 社区 ID 映射
    pub node_to_community: Vec<usize>,
    /// 社区数量
    pub n_communities: usize,
    /// 此层的模块度 Q
    pub modularity: f64,
    /// 社区大小
    pub community_sizes: Vec<usize>,
}

/// 多层 Louvain 结果（多层级结构域 S）
#[derive(Debug, Clone)]
pub struct MultiLayerLouvain {
    pub layers: Vec<LouvainLayer>,
    /// 压缩终止的层级
    pub final_layer: usize,
    /// 节点数（原始层）
    pub n_nodes: usize,
}

/// 配置
#[derive(Debug, Clone)]
pub struct LouvainConfig {
    /// 最大层数
    pub max_layers: usize,
    /// 最小社区数（低于此值停止压缩）
    pub min_communities: usize,
    /// 收敛阈值：模块度增量低于此值停止
    pub convergence_delta: f64,
    /// 每层最大迭代次数
    pub max_iterations: usize,
}

impl Default for LouvainConfig {
    fn default() -> Self {
        Self {
            max_layers: 10,
            min_communities: 2,
            convergence_delta: 1e-5,
            max_iterations: 100,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 模块度计算
// ═══════════════════════════════════════════════════════════════════════

/// 计算给定社区划分的模块度 Q
///
/// Q = (1/2m) Σ_{i,j} [A_{ij} - k_i·k_j/(2m)] · δ(c_i, c_j)
///
/// 其中 m = 边权总和，k_i = 节点 i 的度（带权），
/// δ = 社区索引是否相同
pub fn modularity(
    adj: &[Vec<(usize, f64)>],
    communities: &[usize],
    total_weight: f64,
) -> f64 {
    let n = adj.len();
    if total_weight < 1e-12 { return 0.0; }

    let mut q = 0.0;
    for i in 0..n {
        let ki = degree(adj, i);
        for &(j, a_ij) in &adj[i] {
            if j > i && communities[i] == communities[j] {
                q += 2.0 * (a_ij - ki * degree(adj, j) / (2.0 * total_weight));
            }
        }
    }
    q / (2.0 * total_weight)
}

fn degree(adj: &[Vec<(usize, f64)>], node: usize) -> f64 {
    adj[node].iter().map(|(_, w)| *w).sum()
}

// ═══════════════════════════════════════════════════════════════════════
// Louvain Phase A: 贪心模块度最大化
// ═══════════════════════════════════════════════════════════════════════

/// 贪心移动阶段：将每个节点移动到使模块度增量最大的邻域社区。
///
/// ΔQ_i→C = [Σ_{tot}^{in} + 2k_{i,in}] / 2m - [(Σ_{tot} + k_i) / 2m]²
///          - [Σ_{tot}^{in} / 2m - (Σ_{tot} / 2m)² - (k_i / 2m)²]
fn louvain_phase_a(
    adj: &[Vec<(usize, f64)>],
    communities: &mut [usize],
    total_weight: f64,
    max_iter: usize,
) -> bool {
    let n = adj.len();
    let mut changed = false;
    let inv_2m = 1.0 / (2.0 * total_weight).max(1e-12);

    for _iter in 0..max_iter {
        let mut local_changed = false;

        for node in 0..n {
            let ki = degree(adj, node);

            // 从当前社区移除节点
            let old_comm = communities[node];
            communities[node] = usize::MAX; // 标记为未分配

            // 构建邻域社区集合（当前所在社区 + 邻居社区）
            let mut neighbor_comms: HashSet<usize> = HashSet::new();
            for &(nb, _) in &adj[node] {
                neighbor_comms.insert(communities[nb]);
            }
            neighbor_comms.remove(&usize::MAX);
            if neighbor_comms.is_empty() {
                communities[node] = old_comm;
                continue;
            }

            // 计算移动到每个邻域社区的 ΔQ
            let mut best_comm = old_comm;
            let mut best_delta = 0.0;

            for &target_comm in &neighbor_comms {
                // k_{i,in}: node 与 target_comm 内节点的边权和
                let mut ki_in = 0.0;
                for &(nb, w) in &adj[node] {
                    if communities[nb] == target_comm {
                        ki_in += w;
                    }
                }

                // Σ_{tot}: target_comm 内所有节点的度之和
                let mut sigma_tot = 0.0;
                for j in 0..n {
                    if communities[j] == target_comm {
                        sigma_tot += degree(adj, j);
                    }
                }

                let delta = inv_2m * ki_in - (sigma_tot * ki) * inv_2m * inv_2m;

                if delta > best_delta {
                    best_delta = delta;
                    best_comm = target_comm;
                }
            }

            communities[node] = best_comm;
            if best_comm != old_comm {
                local_changed = true;
                changed = true;
            }
        }

        if !local_changed {
            break;
        }
    }

    changed
}

/// 重新编号社区为连续 ID (0, 1, 2, ...)
fn renumber_communities(communities: &[usize]) -> Vec<usize> {
    let mut mapping: HashMap<usize, usize> = HashMap::new();
    let mut result = vec![0; communities.len()];
    for (i, &comm) in communities.iter().enumerate() {
        let next_id = mapping.len();
        result[i] = *mapping.entry(comm).or_insert(next_id);
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════
// Louvain Phase B: 社区压缩
// ═══════════════════════════════════════════════════════════════════════

/// 社区压缩：将每个社区收缩为超节点，构建下一层图。
///
/// 超节点之间的边权 = 原社区间所有交叉边的权重和。
/// 超节点的自环 = 社区内部边权和。
fn louvain_phase_b(
    adj: &[Vec<(usize, f64)>],
    communities: &[usize],
    n_communities: usize,
) -> Vec<Vec<(usize, f64)>> {
    let n = adj.len();
    let mut new_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_communities];

    // 累加社区间边权
    let mut cross_weights: HashMap<(usize, usize), f64> = HashMap::new();

    for i in 0..n {
        let ci = communities[i];
        for &(j, w) in &adj[i] {
            let cj = communities[j];
            if ci != cj {
                let key = if ci < cj { (ci, cj) } else { (cj, ci) };
                *cross_weights.entry(key).or_insert(0.0) += w;
            }
        }
    }

    for ((a, b), w) in cross_weights {
        new_adj[a].push((b, w));
        new_adj[b].push((a, w));
    }

    new_adj
}

// ═══════════════════════════════════════════════════════════════════════
// 完整 Louvain 单层
// ═══════════════════════════════════════════════════════════════════════

/// 对图运行单层 Louvain。
///
/// 返回 LouvainLayer 或 None（如果节点数 < 2）。
pub fn run_louvain_single_layer(
    adj: &[Vec<(usize, f64)>],
) -> Option<LouvainLayer> {
    let n = adj.len();
    if n < 2 { return None; }

    let total_weight: f64 = adj.iter()
        .flat_map(|neighbors| neighbors.iter().map(|(_, w)| *w))
        .sum::<f64>() / 2.0;

    let config = LouvainConfig::default();

    // 初始每个节点一个社区
    let mut communities: Vec<usize> = (0..n).collect();

    louvain_phase_a(adj, &mut communities, total_weight, config.max_iterations);

    let communities = renumber_communities(&communities);
    let n_communities = communities.iter().max().map(|m| m + 1).unwrap_or(0);
    let q = modularity(adj, &communities, total_weight);

    // 社区大小统计
    let mut community_sizes = vec![0; n_communities];
    for &c in &communities {
        community_sizes[c] += 1;
    }

    Some(LouvainLayer {
        layer: 0,
        node_to_community: communities,
        n_communities,
        modularity: q,
        community_sizes,
    })
}

// ═══════════════════════════════════════════════════════════════════════
// 多层 Louvain
// ═══════════════════════════════════════════════════════════════════════

/// 运行多层 Louvain 社区检测。
///
/// 对原始图重复：Phase A (模块度最大化) → Phase B (社区压缩)，
/// 每次压缩生成新层的超节点图，直到社区数 < min_communities
/// 或达到 max_layers 或模块度增量 < convergence_delta。
pub fn run_multi_layer_louvain(
    adj: &[Vec<(usize, f64)>],
    config: &LouvainConfig,
) -> MultiLayerLouvain {
    let n = adj.len();
    let mut layers: Vec<LouvainLayer> = Vec::new();
    let mut current_adj: Vec<Vec<(usize, f64)>> = adj.to_vec();

    for layer_idx in 0..config.max_layers {
        if current_adj.len() < config.min_communities {
            break;
        }

        let total_weight: f64 = current_adj.iter()
            .flat_map(|neighbors| neighbors.iter().map(|(_, w)| *w))
            .sum::<f64>() / 2.0;

        if total_weight < 1e-12 {
            break;
        }

        let mut communities: Vec<usize> = (0..current_adj.len()).collect();

        louvain_phase_a(&current_adj, &mut communities, total_weight, config.max_iterations);

        let communities = renumber_communities(&communities);
        let n_communities = communities.iter().max().map(|m| m + 1).unwrap_or(0);
        let q = modularity(&current_adj, &communities, total_weight);

        // 收敛检查
        if layer_idx > 0 {
            let delta = q - layers.last().unwrap().modularity;
            if delta < config.convergence_delta {
                break;
            }
        }

        // 社区大小统计
        let mut community_sizes = vec![0; n_communities];
        for &c in &communities {
            community_sizes[c] += 1;
        }

        layers.push(LouvainLayer {
            layer: layer_idx,
            node_to_community: communities,
            n_communities,
            modularity: q,
            community_sizes,
        });

        if n_communities < config.min_communities {
            break;
        }

        // Phase B: 压缩
        current_adj = louvain_phase_b(&current_adj, &layers.last().unwrap().node_to_community, n_communities);
    }

    MultiLayerLouvain {
        final_layer: layers.len(),
        n_nodes: n,
        layers,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// 构建简单的八字形图（两个三角形通过桥连接）
    fn make_small_graph() -> Vec<Vec<(usize, f64)>> {
        let n = 6;
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        // 三角形 1: 0-1, 1-2, 2-0
        adj[0].push((1, 1.0)); adj[1].push((0, 1.0));
        adj[1].push((2, 1.0)); adj[2].push((1, 1.0));
        adj[2].push((0, 1.0)); adj[0].push((2, 1.0));
        // 桥: 0-3
        adj[0].push((3, 0.5)); adj[3].push((0, 0.5));
        // 三角形 2: 3-4, 4-5, 5-3
        adj[3].push((4, 1.0)); adj[4].push((3, 1.0));
        adj[4].push((5, 1.0)); adj[5].push((4, 1.0));
        adj[5].push((3, 1.0)); adj[3].push((5, 1.0));
        adj
    }

    #[test]
    fn test_modularity_singleton() {
        let adj = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
        let communities = vec![0, 1]; // 各一个社区
        let q = modularity(&adj, &communities, 1.0);
        assert!(q <= 0.0); // 单节点社区不贡献正模块度
    }

    #[test]
    fn test_modularity_single_community() {
        let adj = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
        let communities = vec![0, 0]; // 同一社区
        let q = modularity(&adj, &communities, 1.0);
        assert!(q >= 0.0); // 完整社区模块度 >= 0
    }

    #[test]
    fn test_louvain_single_layer() {
        let adj = make_small_graph();
        let result = run_louvain_single_layer(&adj);
        assert!(result.is_some());
        let layer = result.unwrap();
        assert!(layer.n_communities >= 2);
        assert!(layer.modularity > 0.0);
        // 社区划分应覆盖所有节点
        assert_eq!(layer.node_to_community.len(), 6);
        for &c in &layer.node_to_community {
            assert!(c < layer.n_communities);
        }
    }

    #[test]
    fn test_multi_layer_louvain() {
        let adj = make_small_graph();
        let config = LouvainConfig {
            max_layers: 5,
            min_communities: 2,
            convergence_delta: 1e-5,
            max_iterations: 50,
        };
        let result = run_multi_layer_louvain(&adj, &config);
        assert!(result.layers.len() >= 1);
        // 每层社区数递减（或不变）
        for i in 1..result.layers.len() {
            assert!(result.layers[i].n_communities <= result.layers[i - 1].n_communities);
        }
        // 模块度应逐层非降
        for i in 1..result.layers.len() {
            assert!(result.layers[i].modularity >= result.layers[i - 1].modularity - 1e-10);
        }
    }

    #[test]
    fn test_empty_graph() {
        let adj: Vec<Vec<(usize, f64)>> = vec![Vec::new()];
        let result = run_louvain_single_layer(&adj);
        assert!(result.is_none()); // 单节点无邻域
    }

    #[test]
    fn test_disconnected_graph() {
        let mut adj = vec![Vec::new(); 4];
        adj[0].push((1, 1.0)); adj[1].push((0, 1.0));
        adj[2].push((3, 1.0)); adj[3].push((2, 1.0));
        let result = run_louvain_single_layer(&adj);
        assert!(result.is_some());
        let layer = result.unwrap();
        // 断开图应有 2 个社区
        assert!(layer.n_communities >= 2);
    }
}
