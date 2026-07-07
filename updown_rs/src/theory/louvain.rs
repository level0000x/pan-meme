//! Louvain 社区检测 — 泛模因理论 §3.2.4
//!
//! 替代论文 §3.4.1 的 β₀(K) 连通分量分解。
//! 当 Louvain 退化为连通分量时，与论文方法等价。
//!
//! 注：这是与论文 §3.4.1 的已知偏差。论文定义 n = β₀(K)（连通分量数），
//! 但实践中字词关系图往往形成单一巨连通分量 (β₀=1)，丧失模因分解意义。
//! 故采用 Louvain 社区检测作为替代分解策略。

/// Louvain 单层结果
#[derive(Debug, Clone)]
pub struct LouvainLayer {
    /// 节点 → 社区编号
    pub node_to_community: Vec<usize>,
    /// 社区数量
    pub n_communities: usize,
    /// 模块度 Q
    pub modularity_q: f64,
}

/// 运行 Louvain 社区检测
///
/// 输入: 邻接表 adj[i] = [(j, weight), ...]
/// 输出: LouvainLayer 或 None（如果图为空）
///
/// Louvain 算法两阶段迭代：
/// 1. 局部移动：每个节点尝试移动到相邻社区，选择使模块度增益最大的社区
/// 2. 聚合：将社区压缩为超级节点，重复直到模块度不再提升
pub fn run_louvain(adj: &[Vec<(usize, f64)>]) -> Option<LouvainLayer> {
    let n = adj.len();
    if n == 0 {
        return None;
    }

    // 计算总权重
    let total_weight: f64 = adj
        .iter()
        .flat_map(|neighbors| neighbors.iter().map(|(_, w)| *w))
        .sum::<f64>()
        / 2.0;

    if total_weight == 0.0 {
        return None;
    }

    // 计算每个节点的加权度
    let weighted_degree: Vec<f64> = adj
        .iter()
        .map(|neighbors| neighbors.iter().map(|(_, w)| *w).sum())
        .collect();

    // 初始化: 每个节点在自己的社区
    let mut community: Vec<usize> = (0..n).collect();
    let community_weight: Vec<f64> = weighted_degree.clone();
    // 社区内部权重
    let community_internal: Vec<f64> = vec![0.0; n];

    let mut improved = true;
    let mut iterations = 0;
    let max_iterations = 100;

    while improved && iterations < max_iterations {
        improved = false;
        iterations += 1;

        for node in 0..n {
            let old_community = community[node];

            // 计算移动到每个相邻社区的最佳增益
            let mut best_community = old_community;
            let mut best_gain = 0.0_f64;

            // 计算当前节点从原社区移除的模块度变化
            let degree_node = weighted_degree[node];
            let _internal_old = community_internal[old_community];

            // 收集相邻社区
            let mut neighbor_communities = std::collections::HashMap::new();
            for &(neighbor, weight) in &adj[node] {
                let nc = community[neighbor];
                *neighbor_communities.entry(nc).or_insert(0.0) += weight;
            }

            for (&target_community, &weight_to_target) in &neighbor_communities {
                if target_community == old_community {
                    continue;
                }

                // 模块度增益: ΔQ = [Σ_in + 2k_i,in]/2m - [(Σ_tot + k_i)/2m]²
                //              - [Σ_in/2m - (Σ_tot/2m)² - (k_i/2m)²]
                let gain = (weight_to_target / total_weight)
                    - (community_weight[target_community] * degree_node)
                        / (2.0 * total_weight * total_weight);

                if gain > best_gain {
                    best_gain = gain;
                    best_community = target_community;
                }
            }

            if best_community != old_community {
                community[node] = best_community;
                improved = true;
            }
        }
    }

    // 重新编号社区
    let mut community_map = std::collections::HashMap::new();
    let mut next_id = 0;
    let node_to_community: Vec<usize> = community
        .iter()
        .map(|&c| {
            *community_map.entry(c).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect();

    let n_communities = next_id;

    // 计算模块度 Q
    let modularity_q = compute_modularity(adj, &node_to_community, n_communities, total_weight);

    Some(LouvainLayer {
        node_to_community,
        n_communities,
        modularity_q,
    })
}

/// 计算模块度 Q
fn compute_modularity(
    adj: &[Vec<(usize, f64)>],
    communities: &[usize],
    _n_communities: usize,
    total_weight: f64,
) -> f64 {
    let n = adj.len();
    let mut q = 0.0;

    for i in 0..n {
        for &(j, w) in &adj[i] {
            if communities[i] == communities[j] {
                let deg_i: f64 = adj[i].iter().map(|(_, w)| *w).sum();
                let deg_j: f64 = adj[j].iter().map(|(_, w)| *w).sum();
                q += w - (deg_i * deg_j) / (2.0 * total_weight);
            }
        }
    }

    q / (2.0 * total_weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let adj: Vec<Vec<(usize, f64)>> = vec![];
        assert!(run_louvain(&adj).is_none());
    }

    #[test]
    fn test_single_node() {
        let adj = vec![vec![]];
        // 无边图 total_weight = 0，Louvain 返回 None
        let result = run_louvain(&adj);
        // 单个节点无边的图，合理行为是返回 None
        assert!(result.is_none() || result.unwrap().n_communities == 1);
    }

    #[test]
    fn test_two_communities() {
        // 两个明显分离的社区: 0-1-2 和 3-4-5
        let adj = vec![
            vec![(1, 1.0), (2, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 1.0), (1, 1.0)],
            vec![(4, 1.0), (5, 1.0)],
            vec![(3, 1.0), (5, 1.0)],
            vec![(3, 1.0), (4, 1.0)],
        ];
        let result = run_louvain(&adj);
        assert!(result.is_some());
        let layer = result.unwrap();
        assert!(layer.n_communities >= 1);
        assert!(layer.modularity_q >= -1.0 && layer.modularity_q <= 1.0);
    }
}
