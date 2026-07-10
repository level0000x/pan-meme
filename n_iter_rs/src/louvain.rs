use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LouvainLayer {
    pub node_to_community: Vec<usize>,
    pub n_communities: usize,
    pub modularity_q: f64,
}

pub fn run_louvain(adj: &[Vec<(usize, f64)>]) -> Option<LouvainLayer> {
    run_louvain_with_resolution(adj, 1.0)
}

pub fn run_louvain_with_resolution(adj: &[Vec<(usize, f64)>], resolution: f64) -> Option<LouvainLayer> {
    let n = adj.len();
    if n == 0 {
        return None;
    }

    let total_weight: f64 = adj
        .iter()
        .flat_map(|neighbors| neighbors.iter().map(|(_, w)| *w))
        .sum::<f64>()
        / 2.0;

    if total_weight == 0.0 {
        return None;
    }

    let mut node_to_community: Vec<usize> = (0..n).collect();
    let mut current_n = n;
    let mut current_adj = adj.to_vec();

    let max_levels = 20;

    for _level in 0..max_levels {
        let weighted_degree: Vec<f64> = current_adj
            .iter()
            .map(|neighbors| neighbors.iter().map(|(_, w)| *w).sum())
            .collect();

        let cur_total: f64 = current_adj
            .iter()
            .flat_map(|neighbors| neighbors.iter().map(|(_, w)| *w))
            .sum::<f64>()
            / 2.0;

        if cur_total == 0.0 {
            break;
        }

        let mut community: Vec<usize> = (0..current_n).collect();
        let mut community_weight: Vec<f64> = weighted_degree.clone();

        let mut improved = true;
        let mut local_iters = 0;

        while improved && local_iters < 100 {
            improved = false;
            local_iters += 1;

            for node in 0..current_n {
                let old_community = community[node];
                let degree_node = weighted_degree[node];

                let mut neighbor_communities = HashMap::new();
                for &(neighbor, weight) in &current_adj[node] {
                    let nc = community[neighbor];
                    *neighbor_communities.entry(nc).or_insert(0.0) += weight;
                }

                let mut best_community = old_community;
                let mut best_gain = 0.0_f64;

                for (&target_community, &weight_to_target) in &neighbor_communities {
                    if target_community == old_community {
                        continue;
                    }

                    let gain = (weight_to_target / cur_total)
                        - resolution * (community_weight[target_community] * degree_node)
                            / (2.0 * cur_total * cur_total);

                    if gain > best_gain {
                        best_gain = gain;
                        best_community = target_community;
                    }
                }

                if best_community != old_community {
                    community_weight[old_community] -= degree_node;
                    community_weight[best_community] += degree_node;
                    community[node] = best_community;
                    improved = true;
                }
            }
        }

        let mut comm_map = HashMap::new();
        let mut next_id = 0;
        let community_ids: Vec<usize> = community
            .iter()
            .map(|&c| {
                *comm_map.entry(c).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect();
        let n_communities = next_id;

        let mut new_node_to_community = vec![0; n];
        for v in 0..n {
            let cur_comm = node_to_community[v];
            if cur_comm < community_ids.len() {
                new_node_to_community[v] = community_ids[cur_comm];
            } else {
                new_node_to_community[v] = community_ids[0];
            }
        }

        if n_communities >= current_n {
            node_to_community = new_node_to_community;
            break;
        }

        let mut agg_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_communities];
        {
            let mut edge_weights = HashMap::new();
            for u in 0..current_n {
                let cu = community_ids[u];
                for &(v, w) in &current_adj[u] {
                    let cv = community_ids[v];
                    if cu != cv {
                        let key = if cu < cv { (cu, cv) } else { (cv, cu) };
                        *edge_weights.entry(key).or_insert(0.0) += w;
                    }
                }
            }
            for ((a, b), w) in edge_weights {
                agg_adj[a].push((b, w));
                agg_adj[b].push((a, w));
            }
        }

        node_to_community = new_node_to_community;
        current_adj = agg_adj;
        current_n = n_communities;
    }

    let mut final_map = HashMap::new();
    let mut final_id = 0;
    let final_communities: Vec<usize> = node_to_community
        .iter()
        .map(|&c| {
            *final_map.entry(c).or_insert_with(|| {
                let id = final_id;
                final_id += 1;
                id
            })
        })
        .collect();

    let n_communities = final_id;
    let modularity_q = compute_modularity(adj, &final_communities, total_weight);

    Some(LouvainLayer {
        node_to_community: final_communities,
        n_communities,
        modularity_q,
    })
}

fn compute_modularity(
    adj: &[Vec<(usize, f64)>],
    communities: &[usize],
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
        let result = run_louvain(&adj);
        assert!(result.is_none() || result.unwrap().n_communities == 1);
    }

    #[test]
    fn test_two_communities() {
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
