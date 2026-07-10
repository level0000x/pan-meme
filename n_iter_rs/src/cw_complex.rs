use std::collections::{HashMap, HashSet};
use crate::fca;

#[derive(Debug, Clone)]
pub struct Cell {
    pub dim: u8,
    pub id: usize,
    pub boundary: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct CWComplex {
    pub cells: Vec<Cell>,
    pub n_vertices: usize,
    pub n_edges: usize,
    pub euler_char: i32,
    pub betti_0: usize,
    pub betti_1: usize,
    pub degree_seq: Vec<usize>,
    pub max_degree: usize,
    pub edge_density: f64,
    pub vertex_map: HashMap<String, usize>,
    pub edge_list: Vec<(usize, usize)>,
    pub adj_list: Vec<Vec<(usize, f64)>>,
}

impl CWComplex {
    pub fn new() -> Self {
        CWComplex {
            cells: Vec::new(),
            n_vertices: 0, n_edges: 0,
            euler_char: 0, betti_0: 0, betti_1: 0,
            degree_seq: Vec::new(), max_degree: 0, edge_density: 0.0,
            vertex_map: HashMap::new(),
            edge_list: Vec::new(),
            adj_list: Vec::new(),
        }
    }

    pub fn add_vertex(&mut self) -> usize {
        let id = self.n_vertices;
        self.cells.push(Cell { dim: 0, id, boundary: vec![] });
        self.n_vertices += 1;
        self.degree_seq.push(0);
        self.adj_list.push(Vec::new());
        id
    }

    pub fn add_edge(&mut self, v1: usize, v2: usize) {
        let id = self.n_edges;
        self.cells.push(Cell { dim: 1, id, boundary: vec![v1, v2] });
        self.degree_seq[v1] += 1;
        self.degree_seq[v2] += 1;
        self.adj_list[v1].push((v2, 1.0));
        self.adj_list[v2].push((v1, 1.0));
        self.edge_list.push((v1, v2));
        self.n_edges += 1;
    }

    pub fn compute_invariants(&mut self) {
        let nv = self.n_vertices;
        let ne = self.n_edges;

        let mut parent: Vec<usize> = (0..nv).collect();
        fn find(p: &mut [usize], x: usize) -> usize {
            if p[x] != x { p[x] = find(p, p[x]); }
            p[x]
        }
        fn union(p: &mut [usize], x: usize, y: usize) {
            let rx = find(p, x);
            let ry = find(p, y);
            if rx != ry { p[rx] = ry; }
        }
        for &(v1, v2) in &self.edge_list {
            union(&mut parent, v1, v2);
        }
        let mut roots = HashSet::new();
        for i in 0..nv {
            roots.insert(find(&mut parent, i));
        }
        self.betti_0 = roots.len().max(1);
        self.betti_1 = (ne as i32 - nv as i32 + self.betti_0 as i32).max(0) as usize;
        self.euler_char = nv as i32 - ne as i32;
        self.max_degree = self.degree_seq.iter().copied().max().unwrap_or(0);
        self.edge_density = if nv > 1 {
            2.0 * ne as f64 / (nv as f64 * (nv as f64 - 1.0))
        } else {
            0.0
        };
    }
}

pub fn from_fca_lattice(
    _lattice: &fca::FcaLattice,
    bg_data: &fca::BigramData,
) -> CWComplex {
    let n_words = bg_data.unique_words.len();
    if n_words == 0 {
        return CWComplex::new();
    }

    let mut cw = CWComplex::new();
    for _ in 0..n_words {
        cw.add_vertex();
    }

    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
    for (bi, words_with_bg) in bg_data.bigram_to_words.iter().enumerate() {
        if words_with_bg.is_empty() { continue; }
        for a in 0..words_with_bg.len() {
            for b in (a + 1)..words_with_bg.len() {
                let w1 = words_with_bg[a];
                let w2 = words_with_bg[b];
                let key = if w1 < w2 { (w1, w2) } else { (w2, w1) };
                if edge_set.insert(key) {
                    cw.add_edge(w1, w2);
                }
            }
        }
        let _ = bi;
    }

    cw.compute_invariants();
    cw
}

pub fn for_concept(
    extent_words: &[usize],
    w_bg_map: &HashMap<usize, HashSet<usize>>,
) -> CWComplex {
    let m = extent_words.len();
    if m <= 1 {
        return CWComplex::new();
    }

    let mut cw = CWComplex::new();
    for _ in 0..m {
        cw.add_vertex();
    }

    for i in 0..m {
        let bg_i = match w_bg_map.get(&extent_words[i]) {
            Some(b) => b,
            None => continue,
        };
        for j in (i + 1)..m {
            let bg_j = match w_bg_map.get(&extent_words[j]) {
                Some(b) => b,
                None => continue,
            };
            if bg_i.intersection(bg_j).next().is_some() {
                cw.add_edge(i as usize, j as usize);
            }
        }
    }

    cw.compute_invariants();
    cw
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_add_vertex() {
        let mut cw = CWComplex::new();
        let v = cw.add_vertex();
        assert_eq!(v, 0);
        assert_eq!(cw.n_vertices, 1);
    }

    #[test]
    fn test_add_edge() {
        let mut cw = CWComplex::new();
        let v0 = cw.add_vertex();
        let v1 = cw.add_vertex();
        cw.add_edge(v0, v1);
        assert_eq!(cw.n_edges, 1);
        assert_eq!(cw.cells.len(), 3);
    }

    #[test]
    fn test_compute_invariants() {
        let mut cw = CWComplex::new();
        let v0 = cw.add_vertex();
        let v1 = cw.add_vertex();
        let v2 = cw.add_vertex();
        cw.add_edge(v0, v1);
        cw.add_edge(v1, v2);
        cw.compute_invariants();
        assert_eq!(cw.betti_0, 1);
        assert_eq!(cw.betti_1, 0);
        assert_eq!(cw.euler_char, 1);
    }

    #[test]
    fn test_empty() {
        let bg_data = fca::build_bigram_data("", 10);
        let lattice = fca::FcaLattice {
            concepts: vec![], edges: vec![], d_values: vec![],
            concept_sizes: vec![], n_words: 0, n_bigrams: 0,
        };
        let c = from_fca_lattice(&lattice, &bg_data);
        assert_eq!(c.n_vertices, 0);
    }

    #[test]
    fn test_simple_graph() {
        let text = "calculus is the mathematical study of continuous change";
        let bg_data = fca::build_bigram_data(text, 200);
        let lattice = fca::FcaLattice {
            concepts: vec![], edges: vec![], d_values: vec![],
            concept_sizes: vec![], n_words: bg_data.unique_words.len(),
            n_bigrams: bg_data.top_bigrams.len(),
        };
        let c = from_fca_lattice(&lattice, &bg_data);
        assert!(c.n_vertices > 0);
    }

    #[test]
    fn test_for_concept_isolated() {
        let w_bg_map: HashMap<usize, HashSet<usize>> = HashMap::new();
        let c = for_concept(&[0, 1, 2], &w_bg_map);
        assert_eq!(c.n_vertices, 3);
        assert_eq!(c.n_edges, 0);
    }
}
