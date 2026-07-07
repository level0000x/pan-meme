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

    /// 添加顶点 (0-胞腔)
    pub fn add_vertex(&mut self) -> usize {
        let id = self.cells.len();
        self.cells.push(Cell { dim: 0, id, boundary: vec![] });
        self.vertex_map.push(id);
        self.update_invariants();
        id
    }

    /// 添加边 (1-胞腔) — 论文 §3.3.1
    ///
    /// 对应 supplement B2: 对同概念内的每对字建立边
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
        self.cells.push(Cell { dim: 1, id, boundary: vec![v1_cell, v2_cell] });
        self.edge_map.insert(key, id);
        self.update_invariants();
        Some(id)
    }

    /// 添加面 (2-胞腔) — 论文 §3.3.1
    ///
    /// 对应 supplement B3: 对含 ≥3 字的概念建立面
    pub fn add_face(&mut self, edge_ids: Vec<usize>) -> usize {
        let id = self.cells.len();
        self.cells.push(Cell { dim: 2, id, boundary: edge_ids });
        self.update_invariants();
        id
    }

    /// 更新欧拉示性数和 Betti 数
    fn update_invariants(&mut self) {
        let n0 = self.cells.iter().filter(|c| c.dim == 0).count() as i32;
        let n1 = self.cells.iter().filter(|c| c.dim == 1).count() as i32;
        let n2 = self.cells.iter().filter(|c| c.dim == 2).count() as i32;
        self.euler_char = n0 - n1 + n2;

        // 简单连通分量计数: 通过边连接关系计算
        // 使用并查集
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
        // β₁ = |E| - |V| + β₀ (对于图)
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
}