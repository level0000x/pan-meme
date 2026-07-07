//! CW 胞腔复形 — 泛模因理论 §3.3.1, 定义 2.4
//!
//! 核心构造：
//! - Cell: 单胞腔 (dim, id, boundary)
//! - CWComplex: 胞腔复形 (cells, vertex_map, edge_map, Euler 示性数, Betti 数)
//!
//! 对应 proof-supplement-complete.md 定义 2.4 + 构造 B1-B4

use std::collections::HashMap;

/// 单胞腔 — 论文 §3.3.1
#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    /// 维数: 0=顶点, 1=边, 2=面
    pub dim: u8,
    /// 胞腔编号
    pub id: usize,
    /// 边界: 对 0-胞腔为 [], 对 1-胞腔为 [v1, v2], 对 2-胞腔为 [e1, e2, ...]
    pub boundary: Vec<usize>,
}

/// CW 胞腔复形 — 论文 §3.3.1, 定义 2.4
///
/// G = (K, g, ω, Γ, R) 的 K 部分
#[derive(Debug, Clone)]
pub struct CWComplex {
    /// 所有胞腔
    pub cells: Vec<Cell>,
    /// 顶点到胞腔 id 的映射
    pub vertex_map: Vec<usize>,
    /// 边到胞腔 id 的映射
    pub edge_map: HashMap<(usize, usize), usize>,
    /// 欧拉示性数 χ = |K⁰| - |K¹| + |K²|
    pub euler_char: i32,
    /// 第 0 Betti 数 β₀ (连通分量数)
    pub betti_0: usize,
    /// 第 1 Betti 数 β₁ (独立环数)
    pub betti_1: usize,
}

impl CWComplex {
    /// 创建空胞腔复形
    pub fn new() -> Self {
        CWComplex {
            cells: Vec::new(),
            vertex_map: Vec::new(),
            edge_map: HashMap::new(),
            euler_char: 0,
            betti_0: 0,
            betti_1: 0,
        }
    }

    /// 添加顶点 (0-胞腔) — 不更新不变量（批量构建后调用 compute_invariants）
    pub fn add_vertex(&mut self) -> usize {
        let id = self.cells.len();
        self.cells.push(Cell {
            dim: 0,
            id,
            boundary: vec![],
        });
        self.vertex_map.push(id);
        id
    }

    /// 添加边 (1-胞腔) — 论文 §3.3.1
    ///
    /// 对应 supplement B2: 对同概念内的每对字建立边
    /// 不更新不变量（批量构建后调用 compute_invariants）
    pub fn add_edge(&mut self, v1: usize, v2: usize) -> Option<usize> {
        if v1 >= self.vertex_map.len() || v2 >= self.vertex_map.len() {
            return None;
        }
        let key = if v1 < v2 { (v1, v2) } else { (v2, v1) };
        if self.edge_map.contains_key(&key) {
            return None; // 不重复添加
        }
        let id = self.cells.len();
        let v1_cell = self.vertex_map[v1];
        let v2_cell = self.vertex_map[v2];
        self.cells.push(Cell {
            dim: 1,
            id,
            boundary: vec![v1_cell, v2_cell],
        });
        self.edge_map.insert(key, id);
        Some(id)
    }

    /// 添加面 (2-胞腔) — 论文 §3.3.1
    ///
    /// 对应 supplement B3: 对含 ≥3 字的概念建立面
    /// 不更新不变量（批量构建后调用 compute_invariants）
    pub fn add_face(&mut self, edge_ids: Vec<usize>) -> usize {
        let id = self.cells.len();
        self.cells.push(Cell {
            dim: 2,
            id,
            boundary: edge_ids,
        });
        id
    }

    /// 一次性计算欧拉示性数和 Betti 数（在批量构建完成后调用）
    ///
    /// 复杂度 O(|V| + |E|·α(|V|))，比逐边更新的 O(|E|·(|V|+|E|)) 快得多
    pub fn compute_invariants(&mut self) {
        let (mut n0, mut n1, mut n2) = (0i32, 0i32, 0i32);
        for cell in &self.cells {
            match cell.dim {
                0 => n0 += 1,
                1 => n1 += 1,
                2 => n2 += 1,
                _ => {}
            }
        }
        self.euler_char = n0 - n1 + n2;

        let nv = self.vertex_map.len();
        let mut parent: Vec<usize> = (0..nv).collect();
        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }
        fn union(parent: &mut [usize], x: usize, y: usize) {
            let rx = find(parent, x);
            let ry = find(parent, y);
            if rx != ry {
                parent[rx] = ry;
            }
        }
        for (v1, v2) in self.edge_map.keys() {
            union(&mut parent, *v1, *v2);
        }
        let mut roots = std::collections::HashSet::new();
        for i in 0..nv {
            roots.insert(find(&mut parent, i));
        }
        self.betti_0 = roots.len();
        let ne = self.edge_map.len() as i32;
        self.betti_1 = (ne - nv as i32 + self.betti_0 as i32).max(0) as usize;
    }

    /// 顶点数 |K⁰|
    pub fn n_vertices(&self) -> usize {
        self.vertex_map.len()
    }

    /// 边数 |K¹|
    pub fn n_edges(&self) -> usize {
        self.edge_map.len()
    }
}

impl Default for CWComplex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_complex() {
        let c = CWComplex::new();
        assert_eq!(c.cells.len(), 0);
        assert_eq!(c.euler_char, 0);
    }

    #[test]
    fn test_add_vertex() {
        let mut c = CWComplex::new();
        let v0 = c.add_vertex();
        let v1 = c.add_vertex();
        assert_eq!(v0, 0);
        assert_eq!(v1, 1);
        assert_eq!(c.n_vertices(), 2);
        c.compute_invariants();
        assert_eq!(c.euler_char, 2);
    }

    #[test]
    fn test_add_edge() {
        let mut c = CWComplex::new();
        let v0 = c.add_vertex();
        let v1 = c.add_vertex();
        let e = c.add_edge(v0, v1);
        assert!(e.is_some());
        assert_eq!(c.n_edges(), 1);
        c.compute_invariants();
        assert_eq!(c.euler_char, 1); // 2 - 1 = 1
        assert_eq!(c.betti_0, 1);
    }

    #[test]
    fn test_duplicate_edge() {
        let mut c = CWComplex::new();
        let v0 = c.add_vertex();
        let v1 = c.add_vertex();
        c.add_edge(v0, v1);
        let e2 = c.add_edge(v0, v1);
        assert!(e2.is_none());
        assert_eq!(c.n_edges(), 1);
    }

    #[test]
    fn test_compute_invariants_after_batch() {
        let mut c = CWComplex::new();
        for _ in 0..5 { c.add_vertex(); }
        c.add_edge(0, 1);
        c.add_edge(1, 2);
        c.add_edge(3, 4);
        c.compute_invariants();
        assert_eq!(c.euler_char, 5 - 3); // 5 vertices - 3 edges
        assert_eq!(c.betti_0, 2); // two components: {0,1,2} and {3,4}
    }
}
