//! Phase 2: 编码 (M → G) — 泛模因理论 §3.3, §6.3 M1
//!
//! 对应 proof-supplement-complete.md 定义 2.4 + 构造 B1-B4
//!
//! 数据流:
//! PhaseOneOutput → 词→0-胞腔 → 概念内共现词对→1-胞腔 → 概念→2-胞腔
//!              → 度量编码 g → 标量场 φ → 向量场 F = -∇φ
//!              → 场编码 ω → 不变量编码 Γ → 可逆性记录 R
//!              → PhaseTwoOutput { complex, metric, fields, invariants, reversibility }
//!
//! 输出: G = (K, g, ω, Γ, R)

use crate::pipeline::phase1_emergence::PhaseOneOutput;
use crate::theory::cw_complex::CWComplex;
use crate::theory::vector_field::{ScalarField, VectorField};

/// 度量结构 g — 论文 §3.3.3
#[derive(Debug, Clone)]
pub struct MetricStructure {
    /// 每条 1-胞腔的权重（连接强度）
    pub edge_weights: Vec<f64>,
}

/// 场编码 ω — 论文 §3.3.4
#[derive(Debug, Clone)]
pub struct FieldEncoding {
    /// 规则→几何场的映射
    pub field_values: Vec<f64>,
}

/// 不变量编码 Γ — 论文 §3.3.5
#[derive(Debug, Clone)]
pub struct InvariantEncoding {
    /// 欧拉示性数
    pub euler_char: i32,
    /// Betti 数
    pub betti_0: usize,
    pub betti_1: usize,
}

/// 可逆性记录 R — 论文 §3.3.6 + supplement B4
#[derive(Debug, Clone)]
pub struct ReversibilityRecord {
    pub node_texts: Vec<String>,
    pub node_is_word: Vec<bool>,
    pub word_count: usize,
    pub containment_depth: f64,
    pub node_levels: Vec<usize>,
    pub concept_levels: Vec<usize>,
    pub concept_termination_reasons: Vec<String>,
    pub serialized_rules: Vec<String>,
    pub serialized_constraints: Vec<String>,
}

/// Phase 2 输出: G = (K, g, ω, Γ, R) — 论文 §3.3.2-3.3.6
#[derive(Debug, Clone)]
pub struct PhaseTwoOutput {
    pub complex: CWComplex,
    pub metric: MetricStructure,
    pub fields: FieldEncoding,
    pub invariants: InvariantEncoding,
    pub reversibility: ReversibilityRecord,
    pub scalar_field: ScalarField,
    pub vector_field: VectorField,
}

/// 运行 Phase 2: 编码
pub fn run_phase_two(phase1: &PhaseOneOutput) -> PhaseTwoOutput {
    let mut complex = CWComplex::new();

    // B1: 词→0-胞腔
    let n_vertices = phase1.s.vertices.len();
    for _ in 0..n_vertices {
        complex.add_vertex();
    }

    // B2: 概念内共现词对→1-胞腔 (supplement B2)
    let mut edge_weights = Vec::new();
    for (v1, v2) in &phase1.s.edges {
        if let Some(_edge_id) = complex.add_edge(*v1, *v2) {
            let w = phase1
                .s
                .weights
                .get(complex.edge_map.len() - 1)
                .copied()
                .unwrap_or(1.0);
            edge_weights.push(w);
        }
    }

    let metric = MetricStructure { edge_weights };

    // 标量场 φ(v) = 信息深度/max_depth
    let scalar = ScalarField::from_depths(&phase1.s.node_depths);

    // 向量场 F = -∇φ
    let vector = VectorField::compute(&complex, &scalar);

    // 场编码 ω
    let field_values = vector.edge_gradients.clone();
    let fields = FieldEncoding { field_values };

    // 不变量编码 Γ
    let invariants = InvariantEncoding {
        euler_char: complex.euler_char,
        betti_0: complex.betti_0,
        betti_1: complex.betti_1,
    };

    // 可逆性记录 R (supplement B4)
    let reversibility = ReversibilityRecord {
        node_texts: phase1.s.vertices.clone(),
        node_is_word: (0..n_vertices)
            .map(|i| i >= phase1.s.vertices.len() - phase1.s.vertices.len())
            .collect(),
        word_count: phase1.s.vertices.len(),
        containment_depth: phase1.max_depth,
        node_levels: phase1.s.node_depths.iter().map(|&d| d as usize).collect(),
        concept_levels: Vec::new(),
        concept_termination_reasons: Vec::new(),
        serialized_rules: phase1.f.iter().map(|r| r.description.clone()).collect(),
        serialized_constraints: phase1.c.iter().map(|c| c.description.clone()).collect(),
    };

    PhaseTwoOutput {
        complex,
        metric,
        fields,
        invariants,
        reversibility,
        scalar_field: scalar,
        vector_field: vector,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::phase1_emergence::run_phase_one;

    #[test]
    fn test_phase_two_basic() {
        let words = vec!["苹果".to_string(), "香蕉".to_string(), "水果".to_string()];
        let p1 = run_phase_one(words, 100);
        let p2 = run_phase_two(&p1);
        assert!(p2.complex.n_vertices() > 0);
        assert!(p2.invariants.betti_0 > 0);
        assert!(!p2.reversibility.node_texts.is_empty());
    }
}
