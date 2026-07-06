//! 关系自组织生成、推理补全、概念-要素循环组合
//!
//! 数学对应：
//!   3.2.2 关系自组织生成（阈值 T 筛选）
//!   3.2.3 推理与补全（5类推理）
//!   3.2.4 概念-要素循环组合（多层级结构域 S）
//!   3.2.5 彻底不完整判定
//!   3.2.6 规则推导 F + 约束推导 C
//!   3.2.7 自洽性验证

use crate::emergence::extractor::RelationNetwork;
use std::collections::{HashMap, HashSet};

// ═══════════════════════════════════════════════════════════════════════
// 3.2.2 关系自组织生成
// ═══════════════════════════════════════════════════════════════════════

/// 经过连接强度阈值筛选后的关系
#[derive(Debug, Clone)]
pub struct SelfOrganizedRelations {
    /// 字—字 Jaccard ≥ T 的关联: (ci, cj, jaccard, shared_words)
    pub char_relations: Vec<(usize, usize, f64, Vec<usize>)>,
    /// 词—词 Jaccard ≥ T 的关联: (wi, wj, jaccard, shared_chars)
    pub word_relations: Vec<(usize, usize, f64, Vec<usize>)>,
}

/// 对关系网络 φ 中的连接进行强度归一化，仅保留连接强度 ≥ T 的边。
///
/// 数学对应：3.2.2 — w: E → [0,1]，T ∈ (0,1]
pub fn organize_relations(psi: &RelationNetwork, threshold: f64) -> SelfOrganizedRelations {
    let mut char_rel: Vec<(usize, usize, f64, Vec<usize>)> = Vec::new();
    let mut word_rel: Vec<(usize, usize, f64, Vec<usize>)> = Vec::new();

    // 字—字
    for ci in 0..psi.char_jaccard.len() {
        for &(cj, j) in &psi.char_jaccard[ci] {
            if ci < cj && j >= threshold {
                let shared: Vec<usize> = psi.char_to_words[ci]
                    .iter()
                    .filter(|w| psi.char_to_words[cj].contains(w))
                    .copied()
                    .collect();
                char_rel.push((ci, cj, j, shared));
            }
        }
    }

    // 词—词
    for wi in 0..psi.word_jaccard.len() {
        for &(wj, j) in &psi.word_jaccard[wi] {
            if wi < wj && j >= threshold {
                let shared: Vec<usize> = psi.word_to_chars[wi]
                    .iter()
                    .filter(|c| psi.word_to_chars[wj].contains(c))
                    .copied()
                    .collect();
                word_rel.push((wi, wj, j, shared));
            }
        }
    }

    SelfOrganizedRelations { char_relations: char_rel, word_relations: word_rel }
}

// ═══════════════════════════════════════════════════════════════════════
// 3.2.3 推理与补全
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct InferredRelation {
    pub from: usize,
    pub to: usize,
    pub kind: InferenceKind,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum InferenceKind {
    /// 传递推理：c₁ ~ c₂ 且 c₂ ~ c₃ ⇒ c₁ ~ c₃
    Transitive,
    /// 对称推理：c₁ ~ c₂ ⇒ c₂ ~ c₁（Jaccard 已对称，此为词间 containment 传递）
    Symmetric,
    /// 共现推理：多个词共同缺失某字 → 发现未注册模式
    Cooccurrence,
    /// 模式补全：基于低频->高频推断
    PatternCompletion,
    /// 结构相似：结构相似的子图 → 推断缺失边
    StructuralSimilarity,
}

impl std::fmt::Display for InferenceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceKind::Transitive => write!(f, "transitive"),
            InferenceKind::Symmetric => write!(f, "symmetric"),
            InferenceKind::Cooccurrence => write!(f, "cooccurrence"),
            InferenceKind::PatternCompletion => write!(f, "pattern_completion"),
            InferenceKind::StructuralSimilarity => write!(f, "structural_similarity"),
        }
    }
}

pub struct ReasonerConfig {
    pub transitive_enabled: bool,
    pub cooccurrence_enabled: bool,
    pub pattern_completion_enabled: bool,
    pub structural_enabled: bool,
    pub min_confidence: f64,
    /// 传递闭包的迭代次数上限
    pub max_transitive_passes: usize,
}

impl Default for ReasonerConfig {
    fn default() -> Self {
        Self {
            transitive_enabled: true,
            cooccurrence_enabled: true,
            pattern_completion_enabled: false,
            structural_enabled: false,
            min_confidence: 0.3,
            max_transitive_passes: 3,
        }
    }
}

/// 执行五类推理与补全。
///
/// 数学对应：3.2.3 — 对新结构关系矩阵 φ 进行五类推理补全
pub fn reason(
    psi: &RelationNetwork,
    relations: &SelfOrganizedRelations,
    config: &ReasonerConfig,
) -> Vec<InferredRelation> {
    let mut inferred: Vec<InferredRelation> = Vec::new();

    // ── 1. 传递推理 ──
    if config.transitive_enabled {
        transitive_reason(psi, relations, &mut inferred, config);
    }

    // ── 2. 对称推理（Jaccard 已对称，此处做词间 containment 传递）
    // containment 偏序具有传递性：c ⊂ w₁ ⊂ w₂ ⇒ c ⊂ w₂
    transitive_containment(psi, &mut inferred, config);

    // ── 3. 共现推理 ──
    if config.cooccurrence_enabled {
        cooccurrence_reason(psi, relations, &mut inferred, config);
    }

    // ── 4. 模式补全 ──
    if config.pattern_completion_enabled {
        pattern_completion(psi, &mut inferred, config);
    }

    // ── 5. 结构相似 ──
    if config.structural_enabled {
        structural_similarity(psi, relations, &mut inferred, config);
    }

    inferred
}

/// 传递推理：c₁ ~ c₂ 且 c₂ ~ c₃ ⇒ c₁ ~ c₃
fn transitive_reason(
    psi: &RelationNetwork,
    relations: &SelfOrganizedRelations,
    inferred: &mut Vec<InferredRelation>,
    config: &ReasonerConfig,
) {
    // 只对字—字做传递闭包（字数量可控）
    let n_char = psi.char_jaccard.len();

    // 构建邻接矩阵（只含 ≥ min_confidence 的边）
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n_char];
    for (ci, cj, j, _) in &relations.char_relations {
        if *j >= config.min_confidence {
            adj[*ci].insert(*cj);
            adj[*cj].insert(*ci);
        }
    }

    // 多轮传递闭包
    for _pass in 0..config.max_transitive_passes {
        let mut new_edges: Vec<(usize, usize, f64)> = Vec::new();
        for ci in 0..n_char {
            for &cj in &adj[ci].clone() {
                for &ck in &adj[cj].clone() {
                    if ck != ci && !adj[ci].contains(&ck) {
                        // 置信度：min(ci~cj, cj~ck) × 衰减因子
                        let j1 = find_jaccard(&relations.char_relations, ci, cj);
                        let j2 = find_jaccard(&relations.char_relations, cj, ck);
                        let conf = j1.min(j2) * 0.8;
                        new_edges.push((ci, ck, conf));
                    }
                }
            }
        }
        if new_edges.is_empty() { break; }
        for (ci, ck, conf) in new_edges {
            adj[ci].insert(ck);
            adj[ck].insert(ci);
            inferred.push(InferredRelation {
                from: ci + psi.word_count,
                to: ck + psi.word_count,
                kind: InferenceKind::Transitive,
                confidence: conf,
            });
        }
    }
}

fn find_jaccard(relations: &[(usize, usize, f64, Vec<usize>)], a: usize, b: usize) -> f64 {
    let (lo, hi) = (a.min(b), a.max(b));
    for (x, y, j, _) in relations {
        if *x == lo && *y == hi { return *j; }
    }
    0.0
}

/// containment 偏序的传递闭包：c ⊂ w₁ ⊂ w₂ ⇒ c ⊂ w₂
fn transitive_containment(
    psi: &RelationNetwork,
    inferred: &mut Vec<InferredRelation>,
    config: &ReasonerConfig,
) {
    for ci in 0..psi.char_to_words.len() {
        let global_ci = ci + psi.word_count;
        for &wi in &psi.char_to_words[ci] {
            // wi 的词间上位词
            for &sj in &psi.word_to_super_words[wi] {
                // c ⊂ w ⊂ s ⇒ c ⊂ s（若尚未记录）
                if !psi.char_to_words[ci].contains(&sj) && !psi.word_to_super_words[wi].is_empty() {
                    inferred.push(InferredRelation {
                        from: global_ci,
                        to: sj,
                        kind: InferenceKind::Symmetric,
                        confidence: config.min_confidence,
                    });
                }
            }
        }
    }
}

/// 共现推理：高频共现 → 未充分记录的连接
fn cooccurrence_reason(
    psi: &RelationNetwork,
    relations: &SelfOrganizedRelations,
    inferred: &mut Vec<InferredRelation>,
    config: &ReasonerConfig,
) {
    // 对每个字，找其高频共现但低于阈值的连接
    let n_char = psi.char_jaccard.len();
    for ci in 0..n_char {
        let mut coocc_count: HashMap<usize, usize> = HashMap::new();
        for &wj in &psi.char_to_words[ci] {
            for &cj in &psi.word_to_chars[wj] {
                if cj != ci {
                    *coocc_count.entry(cj).or_insert(0) += 1;
                }
            }
        }
        let freq_ci = psi.char_freq[ci];
        for (cj, count) in coocc_count {
            if ci < cj {
                let existing_j = relations.char_relations.iter()
                    .find(|(x,y,_,_)| (*x == ci && *y == cj) || (*x == cj && *y == ci))
                    .map(|(_,_,j,_)| *j)
                    .unwrap_or(0.0);
                if existing_j < config.min_confidence {
                    let inferred_j = count as f64 / (freq_ci.max(1) + psi.char_freq[cj].max(1)) as f64 * 2.0;
                    if inferred_j >= config.min_confidence {
                        inferred.push(InferredRelation {
                            from: ci + psi.word_count,
                            to: cj + psi.word_count,
                            kind: InferenceKind::Cooccurrence,
                            confidence: inferred_j.min(1.0),
                        });
                    }
                }
            }
        }
    }
}

/// 模式补全：低频模式从高频模式继承未记录的关系。
///
/// 数学对应：3.2.3 模式补全推理 —
///   当一个字在共现网络中的信息不足以独立建立可靠链
///   接时，利用其所属字簇（高频共现模式构成的模板）
///   中其他字的连接模式进行推理补全。
///
/// 算法流程：
///   1. 建立字簇模板库（基于高频 Jaccard 共现模式进行聚类）
///   2. 对低频字，计算与各模板的 Jaccard 重叠度，匹配最佳模板
///   3. 从模板继承低频字缺失的高置信度边
fn pattern_completion(
    psi: &RelationNetwork,
    inferred: &mut Vec<InferredRelation>,
    config: &ReasonerConfig,
) {
    let n_char = psi.char_jaccard.len();
    let wc = psi.word_count;
    if n_char < 3 { return; }

    // ── Step 1: 建立字簇模板库 ──
    // 模板定义：一组高频字（freq ≥ median）的高 Jaccard 共现子图
    // 每个模板 = (成员字集合, 成员间的高置信度边, 模板"签名"向量)
    let median_freq = {
        let mut freqs: Vec<usize> = psi.char_freq.iter().copied().collect();
        freqs.sort_unstable();
        freqs[n_char / 2]
    };

    #[derive(Debug, Clone)]
    struct CharTemplate {
        members: Vec<usize>,               // 模板中的字索引
        signature: Vec<f64>,               // 模板签名：与所有字的 Jaccard 向量均值
        high_confidence_edges: Vec<(usize, usize, f64)>, // 模板内高置信度边
        member_set: HashSet<usize>,
    }

    // 用连通分量构建模板（高频字 + 高 Jaccard 连接）
    let template_threshold = 0.5_f64.max(config.min_confidence);
    let mut template_adj: Vec<HashSet<usize>> = vec![HashSet::new(); n_char];
    let high_freq_set: HashSet<usize> = (0..n_char)
        .filter(|&ci| psi.char_freq[ci] >= median_freq)
        .collect();

    for ci in 0..n_char {
        if !high_freq_set.contains(&ci) { continue; }
        for &(cj, j) in &psi.char_jaccard[ci] {
            if high_freq_set.contains(&cj) && j >= template_threshold {
                template_adj[ci].insert(cj);
                template_adj[cj].insert(ci);
            }
        }
    }

    // DFS 找高频连通分量作为模板
    let mut visited: HashSet<usize> = HashSet::new();
    let mut templates: Vec<CharTemplate> = Vec::new();

    for &ci in &high_freq_set {
        if visited.contains(&ci) { continue; }
        let mut stack = vec![ci];
        let mut comp = Vec::new();
        visited.insert(ci);
        while let Some(u) = stack.pop() {
            comp.push(u);
            for &v in &template_adj[u] {
                if !visited.contains(&v) {
                    visited.insert(v);
                    stack.push(v);
                }
            }
        }
        if comp.len() >= 2 {
            // 计算模板签名：模板内成员对所有字的平均 Jaccard
            let signature: Vec<f64> = (0..n_char)
                .map(|cj| {
                    let sum: f64 = comp.iter()
                        .map(|&mi| psi.char_jaccard[mi].iter()
                            .find(|&&(x, _)| x == cj)
                            .map(|(_, j)| *j)
                            .unwrap_or(0.0))
                        .sum();
                    sum / comp.len() as f64
                })
                .collect();

            // 收集模板内高置信度边
            let mut edges = Vec::new();
            for i in 0..comp.len() {
                for j in (i+1)..comp.len() {
                    let a = comp[i];
                    let b = comp[j];
                    if let Some(&(_, jv)) = psi.char_jaccard[a].iter().find(|&&(x, _)| x == b) {
                        edges.push((a, b, jv));
                    }
                    // 也检查传递边（template_adj 中已有但 psi 中不在阈值内的）
                    else if template_adj[a].contains(&b) {
                        edges.push((a, b, template_threshold));
                    }
                }
            }

            templates.push(CharTemplate {
                members: comp.clone(),
                signature,
                high_confidence_edges: edges,
                member_set: comp.into_iter().collect(),
            });
        }
    }

    if templates.is_empty() { return; }

    // ── Step 2: 对低频字匹配最佳模板 ──
    for ci in 0..n_char {
        if high_freq_set.contains(&ci) { continue; } // 只处理低频字

        // 计算该字与每个模板的相似度
        let char_jaccard_vec: Vec<f64> = (0..n_char)
            .map(|cj| psi.char_jaccard[ci].iter()
                .find(|&&(x, _)| x == cj)
                .map(|(_, j)| *j)
                .unwrap_or(0.0))
            .collect();

        let mut best_template: Option<(usize, f64)> = None;
        for (tid, tmpl) in templates.iter().enumerate() {
            // 余弦相似度
            let dot: f64 = char_jaccard_vec.iter()
                .zip(tmpl.signature.iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm_c: f64 = char_jaccard_vec.iter().map(|v| v * v).sum::<f64>().sqrt();
            let norm_t: f64 = tmpl.signature.iter().map(|v| v * v).sum::<f64>().sqrt();
            let sim = if norm_c > 1e-9 && norm_t > 1e-9 {
                dot / (norm_c * norm_t)
            } else { 0.0 };

            if sim > best_template.as_ref().map(|(_, s)| *s).unwrap_or(0.0) {
                best_template = Some((tid, sim));
            }
        }

        // ── Step 3: 从最佳模板继承缺失的边 ──
        if let Some((tid, sim)) = best_template {
            if sim < config.min_confidence { continue; }

            let tmpl = &templates[tid];

            // 对于模板中的每条高置信度边和每个模板成员与外部字的共现模式
            // 继承边：如果模板成员 m_k 与目标字 cj 有高 Jaccard，
            // 且 ci 与 cj 当前连接不足，则推断 ci ~ cj
            for &mk in &tmpl.members {
                for &(cj, j_mk_cj) in &psi.char_jaccard[mk] {
                    if cj == ci { continue; }
                    if tmpl.member_set.contains(&cj) { continue; } // 模板内边已在前面处理

                    // 检查 ci 与 cj 的现有 Jaccard
                    let existing_j = psi.char_jaccard[ci].iter()
                        .find(|&&(x, _)| x == cj)
                        .map(|(_, j)| *j)
                        .unwrap_or(0.0);

                    if existing_j >= config.min_confidence { continue; }

                    // 置信度 = mk~cj 的 Jaccard × 模板相似度 × 衰减因子
                    let conf = j_mk_cj * sim * 0.7;
                    if conf >= config.min_confidence {
                        inferred.push(InferredRelation {
                            from: ci + wc,
                            to: cj + wc,
                            kind: InferenceKind::PatternCompletion,
                            confidence: conf.min(1.0),
                        });
                    }
                }
            }

            // 模板内边继承：如果模板内两个成员 m_a, m_b 间有高置信度边，
            // 且 ci 与 m_a 有连接但 ci 与 m_b 的连接不足，则推断 ci ~ m_b
            for &(ma, mb, j_ab) in &tmpl.high_confidence_edges {
                let ci_ma = psi.char_jaccard[ci].iter()
                    .find(|&&(x, _)| x == ma)
                    .map(|(_, j)| *j)
                    .unwrap_or(0.0);
                let ci_mb = psi.char_jaccard[ci].iter()
                    .find(|&&(x, _)| x == mb)
                    .map(|(_, j)| *j)
                    .unwrap_or(0.0);

                // ci 与 ma 有足够连接，但与 mb 不够 → 推断 ci~mb
                if ci_ma >= config.min_confidence && ci_mb < config.min_confidence {
                    let conf = ci_ma * j_ab * sim * 0.65;
                    if conf >= config.min_confidence {
                        inferred.push(InferredRelation {
                            from: ci + wc,
                            to: mb + wc,
                            kind: InferenceKind::PatternCompletion,
                            confidence: conf.min(1.0),
                        });
                    }
                }

                // 对称：ci 与 mb 有足够连接，但与 ma 不够
                if ci_mb >= config.min_confidence && ci_ma < config.min_confidence {
                    let conf = ci_mb * j_ab * sim * 0.65;
                    if conf >= config.min_confidence {
                        inferred.push(InferredRelation {
                            from: ci + wc,
                            to: ma + wc,
                            kind: InferenceKind::PatternCompletion,
                            confidence: conf.min(1.0),
                        });
                    }
                }
            }
        }
    }
}

/// 结构相似推理：基于邻域结构相似性检测子图同构模式，
/// 推断缺失边。
///
/// 数学对应：3.2.3 结构相似推理 —
///   当两个字在共现网络中具有高度相似的邻域结构时，
///   如果一个字拥有的某条连接在另一个字中缺失了，
///   则推断该缺失连接也是应该存在的。
///
/// 算法流程：
///   1. 对每对字，计算邻域结构相似度（Jaccard 重叠 + Spearman 秩相关）
///   2. 对高结构相似度字对，找出"一方有、另一方无"的边
///   3. 对缺失方推断该边，置信度基于结构相似度和该边的实际 Jaccard
fn structural_similarity(
    psi: &RelationNetwork,
    _relations: &SelfOrganizedRelations,
    inferred: &mut Vec<InferredRelation>,
    config: &ReasonerConfig,
) {
    let n_char = psi.char_jaccard.len();
    let wc = psi.word_count;
    if n_char < 4 { return; }

    // ── Step 1: 计算字间的邻域结构相似度 ──
    // 对每个字 ci，构建其共现邻域 Jaccard 向量
    let jaccard_vecs: Vec<Vec<f64>> = (0..n_char)
        .map(|ci| {
            (0..n_char)
                .map(|cj| psi.char_jaccard[ci].iter()
                    .find(|&&(x, _)| x == cj)
                    .map(|(_, j)| *j)
                    .unwrap_or(0.0))
                .collect()
        })
        .collect();

    // 对每对字计算结构相似度
    // 存入 (i, j, sim) 三元组，只保留 sim ≥ min_confidence 的
    let sim_threshold = config.min_confidence.max(0.4);
    let mut similar_pairs: Vec<(usize, usize, f64)> = Vec::new();

    for ci in 0..n_char {
        for cj in (ci + 1)..n_char {
            let vec_i = &jaccard_vecs[ci];
            let vec_j = &jaccard_vecs[cj];

            // Jaccard 重叠：考虑非零连接的共有性
            let mut intersection = 0.0_f64;
            let mut union = 0.0_f64;
            let mut dot = 0.0_f64;
            let mut norm_i = 0.0_f64;
            let mut norm_j = 0.0_f64;

            for k in 0..n_char {
                if k == ci || k == cj { continue; }
                let vi = vec_i[k];
                let vj = vec_j[k];
                intersection += vi.min(vj);
                union += vi.max(vj);
                dot += vi * vj;
                norm_i += vi * vi;
                norm_j += vj * vj;
            }

            // 余弦相似度分量
            let cos_sim = if norm_i > 1e-9 && norm_j > 1e-9 {
                dot / (norm_i.sqrt() * norm_j.sqrt())
            } else { 0.0 };

            // Jaccard 重叠分量
            let jacc_overlap = if union > 1e-9 {
                intersection / union
            } else { 0.0 };

            // 综合相似度 = 0.5 * cos + 0.5 * jacc_overlap
            let sim = 0.5 * cos_sim + 0.5 * jacc_overlap;

            if sim >= sim_threshold {
                similar_pairs.push((ci, cj, sim));
            }
        }
    }

    // 按相似度降序排列
    similar_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // ── Step 2 & 3: 对高相似度对，找出缺失边并推断 ──
    // 用 HashSet 追踪已推断的边，避免重复
    let mut inferred_edges: HashSet<(usize, usize)> = HashSet::new();

    for &(ci, cj, sim) in &similar_pairs {
        // ci 有但 cj 没有的边
        for &(ck, j_ci_ck) in &psi.char_jaccard[ci] {
            if ck == cj { continue; }

            let j_cj_ck = psi.char_jaccard[cj].iter()
                .find(|&&(x, _)| x == ck)
                .map(|(_, j)| *j)
                .unwrap_or(0.0);

            // cj 与 ck 连接不足，但 ci 与 ck 有足够连接
            if j_ci_ck >= config.min_confidence && j_cj_ck < config.min_confidence {
                let conf = j_ci_ck * sim * 0.6;
                if conf >= config.min_confidence {
                    let pair = (cj.min(ck), cj.max(ck));
                    if !inferred_edges.contains(&pair) {
                        inferred_edges.insert(pair);
                        inferred.push(InferredRelation {
                            from: cj + wc,
                            to: ck + wc,
                            kind: InferenceKind::StructuralSimilarity,
                            confidence: conf.min(1.0),
                        });
                    }
                }
            }
        }

        // cj 有但 ci 没有的边（对称）
        for &(ck, j_cj_ck) in &psi.char_jaccard[cj] {
            if ck == ci { continue; }

            let j_ci_ck = psi.char_jaccard[ci].iter()
                .find(|&&(x, _)| x == ck)
                .map(|(_, j)| *j)
                .unwrap_or(0.0);

            if j_cj_ck >= config.min_confidence && j_ci_ck < config.min_confidence {
                let conf = j_cj_ck * sim * 0.6;
                if conf >= config.min_confidence {
                    let pair = (ci.min(ck), ci.max(ck));
                    if !inferred_edges.contains(&pair) {
                        inferred_edges.insert(pair);
                        inferred.push(InferredRelation {
                            from: ci + wc,
                            to: ck + wc,
                            kind: InferenceKind::StructuralSimilarity,
                            confidence: conf.min(1.0),
                        });
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3.2.4 概念-要素循环组合
// ═══════════════════════════════════════════════════════════════════════

/// 概念（聚类结果）
#[derive(Debug, Clone)]
pub struct Concept {
    /// 概念 ID
    pub id: usize,
    /// 构成此概念的要素集合（节点索引）
    pub members: Vec<usize>,
    /// 层级深度
    pub level: usize,
    /// 外部连接：(目标概念ID, 连接强度)
    pub external_links: Vec<(usize, f64)>,
}

/// 多层级结构域 S
#[derive(Debug, Clone)]
pub struct ConceptHierarchy {
    /// 所有层级的概念：levels[0] 是第 0 层的概念列表
    pub levels: Vec<Vec<Concept>>,
    /// 层次总数
    pub depth: usize,
    /// 每层的终止原因
    pub termination_reasons: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum ConceptCycleMode {
    /// 固定层数
    Fixed(usize),
    /// 自动收敛：连续两轮无新的组合模式产生
    Converge,
}

/// 从关系网络出发，通过 ↓↑ 循环提升层级，构建多层级结构域 S。
///
/// 数学对应：3.2.4 — 概念-要素循环组合
///
/// 流程：
///   1. 识别组合模式：在当前层级要素中找到高关联度聚类（连通分量）
///   2. 生成概念：将每个聚类标记为"概念"
///   3. 更新组合路径
///   4. 提升层级：概念成为下一轮的"要素"
///   5. 检查终止条件
pub fn build_concept_hierarchy(
    psi: &RelationNetwork,
    _relations: &SelfOrganizedRelations,
    mode: ConceptCycleMode,
    max_levels: usize,
) -> ConceptHierarchy {
    let n_char = psi.char_jaccard.len();
    let wc = psi.word_count;

    // ── Level 0: 原始要素 — 字 ──
    let mut levels: Vec<Vec<Concept>> = Vec::new();
    let mut termination_reasons: Vec<String> = Vec::new();

    // Level 0 的概念就是字本身
    let mut level0: Vec<Concept> = Vec::new();
    for ci in 0..n_char {
        level0.push(Concept {
            id: ci,
            members: vec![ci + wc],
            level: 0,
            external_links: Vec::new(),
        });
    }
    levels.push(level0);

    // ── 逐层聚类 ──
    let mut current_elements: Vec<usize> = (0..n_char).map(|ci| ci + wc).collect(); // 当前层要素（全局索引）

    for lvl in 1..=max_levels {
        // Step 1: 在当前要素之间构建连接图
        // 用 Jaccard ≥ 阈值 来确定连通分量
            let threshold = 0.1;

            // 构建邻接表
            let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        // 建立快速查找：current element 中的索引
        let elem_set: HashSet<usize> = current_elements.iter().copied().collect();

        for ci in 0..n_char {
            let gi = ci + wc;
            if !elem_set.contains(&gi) { continue; }
            for &(cj, j) in &psi.char_jaccard[ci] {
                let gj = cj + wc;
                if elem_set.contains(&gj) && j >= threshold {
                    adj.entry(gi).or_default().push(gj);
                    adj.entry(gj).or_default().push(gi);
                }
            }
        }

        // Step 2: 找连通分量（DFS）
        let mut visited: HashSet<usize> = HashSet::new();
        let mut components: Vec<Vec<usize>> = Vec::new();

        for &elem in &current_elements {
            if visited.contains(&elem) { continue; }
            let mut stack = vec![elem];
            let mut comp = Vec::new();
            visited.insert(elem);
            while let Some(u) = stack.pop() {
                comp.push(u);
                if let Some(nbrs) = adj.get(&u) {
                    for &v in nbrs {
                        if !visited.contains(&v) {
                            visited.insert(v);
                            stack.push(v);
                        }
                    }
                }
            }
            if !comp.is_empty() {
                components.push(comp);
            }
        }

        // 孤立节点也各自成概念
        for &elem in &current_elements {
            if !visited.contains(&elem) {
                components.push(vec![elem]);
                visited.insert(elem);
            }
        }

        // Step 3: 检查收敛
        let new_count = components.len();
        let old_count = current_elements.len();
        if new_count == old_count {
            termination_reasons.push(format!("Level {}: 连续两轮无新的组合模式产生 ({} 概念 = {} 要素)", lvl, new_count, old_count));
            break;
        }

        // Step 4: 生成概念
        let mut concepts: Vec<Concept> = Vec::new();
        let mut new_elements: Vec<usize> = Vec::new(); // 虚拟 ID（概念作为下一轮的要素）

        for (id_offset, comp) in components.iter().enumerate() {
            let concept_id = levels.iter().map(|l| l.len()).sum::<usize>() + id_offset;
            let mut external = Vec::new();

            // 计算与其他概念的外部连接（基于成员间的字—字 Jaccard）
            for (other_offset, other_comp) in components.iter().enumerate() {
                if id_offset == other_offset { continue; }
                let mut total_j = 0.0;
                let mut count_j = 0usize;
                for &m1 in comp {
                    let mi = m1 - wc;
                    for &(cj, j) in &psi.char_jaccard[mi] {
                        let mj = cj + wc;
                        if other_comp.contains(&mj) {
                            total_j += j;
                            count_j += 1;
                        }
                    }
                }
                if count_j > 0 {
                    let other_id = levels.iter().map(|l| l.len()).sum::<usize>() + other_offset;
                    external.push((other_id, total_j / count_j as f64));
                }
            }

            concepts.push(Concept {
                id: concept_id,
                members: comp.clone(),
                level: lvl,
                external_links: external,
            });

            new_elements.push(concept_id + psi.node_count()); // 虚拟全局索引
        }

        current_elements = new_elements;
        levels.push(concepts);

        // Step 5: 检查终止模式
        match mode {
            ConceptCycleMode::Fixed(n) if lvl >= n => {
                termination_reasons.push(format!("Level {}: 达到设定层数 {}", lvl, n));
                break;
            }
            _ => {}
        }
    }

    if termination_reasons.is_empty() {
        termination_reasons.push("达到最大层数限制".to_string());
    }

    ConceptHierarchy { levels: levels.clone(), depth: levels.len(), termination_reasons }
}

// ═══════════════════════════════════════════════════════════════════════
// 3.2.5 彻底不完整判定
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct IncompletenessReport {
    pub is_radically_incomplete: bool,
    pub orphan_count: usize,
    pub unclassified_count: usize,
    pub gap_density: f64,
}

/// 判定论域是否彻底不完整。
///
/// 数学对应：3.2.5 — 在 ↓↑ 循环终止后检查是否存在无法被捕获的开放结构
pub fn judge_incompleteness(psi: &RelationNetwork, hierarchy: &ConceptHierarchy) -> IncompletenessReport {
    let wc = psi.word_count;
    let n_char = psi.char_jaccard.len();

    // 孤立字：没有 Jaccard ≥ 阈值的共现字
    let threshold = 0.1;
    let mut orphans = 0usize;
    for ci in 0..n_char {
        if psi.char_jaccard[ci].iter().all(|(_, j)| *j < threshold) {
            orphans += 1;
        }
    }

    // 未归类词：没有被任何概念捕获的词
    let mut classified_words: HashSet<usize> = HashSet::new();
    for level in &hierarchy.levels {
        for concept in level {
            for &member in &concept.members {
                if member < wc {
                    // member 是词（在概念-要素循环中，概念成员可能已包含词）
                    classified_words.insert(member);
                } else {
                    // member 是字，通过字找到包含它的词
                    let ci = member - wc;
                    for &wi in &psi.char_to_words[ci] {
                        classified_words.insert(wi);
                    }
                }
            }
        }
    }
    let unclassified = wc.saturating_sub(classified_words.len());

    // 间隙密度 = 缺失连接数 / 理论最大连接数
    let total_possible = n_char * n_char.saturating_sub(1) / 2;
    let actual_edges: usize = psi.char_jaccard.iter().map(|r| r.len()).sum::<usize>() / 2;
    let gap_density = if total_possible > 0 {
        1.0 - (actual_edges as f64 / total_possible as f64)
    } else { 1.0 };

    let is_radically_incomplete = orphans > n_char / 2 || unclassified > wc / 2;

    IncompletenessReport {
        is_radically_incomplete,
        orphan_count: orphans,
        unclassified_count: unclassified,
        gap_density,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3.2.6 规则推导 F + 约束推导 C
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct DerivedRule {
    pub rule_type: RuleType,
    pub antecedent: Vec<usize>,
    pub consequent: Vec<usize>,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum RuleType {
    /// "如果 c₁ 和 c₂ 共现，则 c₃ 也大概率出现"
    CooccurrenceRule,
    /// "如果结构 A 出现，则结构 B 也大概率出现"
    StructuralRule,
}

#[derive(Debug, Clone)]
pub struct DerivedConstraint {
    pub description: String,
    /// 不变量等式
    pub invariant_equation: String,
    pub value: f64,
}

/// 从结构域 S 中推导规则 F 和约束 C。
///
/// 数学对应：3.2.6 — 从关系网络的结构中提取规则和约束
pub fn derive_rules_and_constraints(
    psi: &RelationNetwork,
    relations: &SelfOrganizedRelations,
) -> (Vec<DerivedRule>, Vec<DerivedConstraint>) {
    let mut rules = Vec::new();
    let mut constraints = Vec::new();

    // ── 规则推导 ──
    // 共现规则：高频共现对
    let wc = psi.word_count;
    for (ci, cj, j, _) in &relations.char_relations {
        if *j >= 0.8 {
            rules.push(DerivedRule {
                rule_type: RuleType::CooccurrenceRule,
                antecedent: vec![*ci + wc],
                consequent: vec![*cj + wc],
                confidence: *j,
            });
        }
    }

    // ── 约束推导 ──
    // 字的总出场次数 = Σ freq(c)
    let total_char_occs: usize = psi.char_freq.iter().sum();
    constraints.push(DerivedConstraint {
        description: "字的总出现次数守恒".to_string(),
        invariant_equation: "Σ freq(c) = total_occurrences".to_string(),
        value: total_char_occs as f64,
    });

    // 词的总字数 = 字的总出场次数
    let total_word_chars: usize = psi.word_len.iter().sum();
    constraints.push(DerivedConstraint {
        description: "词的总字数等于字的总出场次数".to_string(),
        invariant_equation: "Σ |word| = Σ freq(c)".to_string(),
        value: total_word_chars as f64,
    });

    // 论域大小约束
    constraints.push(DerivedConstraint {
        description: "论域节点总数".to_string(),
        invariant_equation: "|U| = |W| + |C|".to_string(),
        value: psi.node_count() as f64,
    });

    // Jaccard 分布特征
    let char_j_sum: f64 = psi.char_jaccard.iter().map(|r| r.iter().map(|(_, j)| j).sum::<f64>()).sum();
    let char_j_pairs: usize = psi.char_jaccard.iter().map(|r| r.len()).sum::<usize>() / 2;
    constraints.push(DerivedConstraint {
        description: "字—字平均 Jaccard 连接强度".to_string(),
        invariant_equation: "avg Jaccard = Σ Jaccard(ci,cj) / |pairs|".to_string(),
        value: if char_j_pairs > 0 { char_j_sum / char_j_pairs as f64 } else { 0.0 },
    });

    (rules, constraints)
}

// ═══════════════════════════════════════════════════════════════════════
// 3.2.7 自洽性验证
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct ConsistencyReport {
    pub is_consistent: bool,
    pub contradictions: Vec<String>,
}

/// 验证 S、F、C 之间的自洽性。
///
/// 数学对应：3.2.7 — S、F、C 三者相互验证无矛盾
pub fn verify_consistency(
    hierarchy: &ConceptHierarchy,
    rules: &[DerivedRule],
    constraints: &[DerivedConstraint],
) -> ConsistencyReport {
    let mut contradictions = Vec::new();

    // 检查层级一致性
    for level in &hierarchy.levels {
        for concept in level {
            if concept.members.is_empty() {
                contradictions.push(format!("概念 {} 在第 {} 层无成员", concept.id, concept.level));
            }
        }
    }

    // 检查规则置信度范围
    for rule in rules {
        if rule.confidence < 0.0 || rule.confidence > 1.0 {
            contradictions.push(format!("规则置信度越界: {}", rule.confidence));
        }
    }

    // 检查约束值范围
    for c in constraints {
        if c.value < 0.0 {
            contradictions.push(format!("约束值 {} 为负: {}", c.invariant_equation, c.value));
        }
    }

    ConsistencyReport {
        is_consistent: contradictions.is_empty(),
        contradictions,
    }
}

/// Phase 1 完整输出 M = (S, F, C)
#[derive(Debug)]
pub struct PhaseOneOutput {
    pub structure: ConceptHierarchy,
    pub rules: Vec<DerivedRule>,
    pub constraints: Vec<DerivedConstraint>,
    pub is_consistent: bool,
    pub incompleteness: IncompletenessReport,
    /// 每个节点在 ↑↓ 循环中的收敛轮数 → 信息深度
    pub node_levels: Vec<usize>,
}

/// 运行完整的第一阶段建模。
pub fn run_phase_one(
    psi: &RelationNetwork,
    relations: &SelfOrganizedRelations,
    cycle_mode: ConceptCycleMode,
    max_levels: usize,
    node_levels: Vec<usize>,
) -> PhaseOneOutput {
    let structure = build_concept_hierarchy(psi, relations, cycle_mode, max_levels);
    let (rules, constraints) = derive_rules_and_constraints(psi, relations);
    let consistency = verify_consistency(&structure, &rules, &constraints);
    let incompleteness = judge_incompleteness(psi, &structure);

    PhaseOneOutput {
        structure,
        rules,
        constraints,
        is_consistent: consistency.is_consistent,
        incompleteness,
        node_levels,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 逆映射：M → I（推论5.1 — Φ_B⁻¹ ∘ Φ_A⁻¹）
// ═══════════════════════════════════════════════════════════════════════

/// 从 PhaseOneOutput 的 node_texts 恢复原始词列表。
///
/// 数学对应：定理1+2逆 — 利用 rev_data 中存储的原始文本标签。
/// 词在前 word_count 个，字在后。
///
/// 返回 (words, chars)：原始信息 I 的词集合和字集合。
pub fn decode_phase_one(
    node_texts: &[String],
    word_count: usize,
) -> (Vec<String>, Vec<String>) {
    let words: Vec<String> = node_texts[..word_count.min(node_texts.len())].to_vec();
    let chars: Vec<String> = node_texts[word_count.min(node_texts.len())..].to_vec();
    (words, chars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emergence::extractor::RelationNetwork;
    use std::collections::HashMap;

    /// 构建小型测试关系网络
    /// 词: ["花草", "花朵", "草地", "蓝天", "白云"]
    /// 期望形成两个字簇: {"花","草"} 和 {"蓝","白","天"}
    /// 低频字 "朵" 仅在一个词中出现
    fn make_test_psi() -> RelationNetwork {
        // 手工构建 RelationNetwork（避免依赖 extractor 的复杂逻辑）
        let words = vec!["花草", "花朵", "草地", "蓝天", "白云", "花草丛", "花草地"];
        let words: Vec<String> = words.iter().map(|s| s.to_string()).collect();
        let word_count = words.len();

        // 提取字
        let mut char_map: HashMap<char, usize> = HashMap::new();
        let mut chars: Vec<String> = Vec::new();
        for w in &words {
            for c in w.chars() {
                if !char_map.contains_key(&c) {
                    char_map.insert(c, chars.len());
                    chars.push(c.to_string());
                }
            }
        }
        let char_count = chars.len();

        // 构建索引
        let mut char_to_words: Vec<Vec<usize>> = vec![Vec::new(); char_count];
        let mut word_to_chars: Vec<Vec<usize>> = vec![Vec::new(); word_count];
        let mut char_freq = vec![0usize; char_count];
        let word_len: Vec<usize> = words.iter().map(|w| w.chars().count()).collect();

        for (wi, w) in words.iter().enumerate() {
            for c in w.chars() {
                let ci = char_map[&c];
                word_to_chars[wi].push(ci);
                char_to_words[ci].push(wi);
                char_freq[ci] += 1;
            }
        }

        // 构建 Jaccard 行
        let mut char_jaccard: Vec<Vec<(usize, f64)>> = vec![Vec::new(); char_count];
        for ci in 0..char_count {
            for cj in (ci + 1)..char_count {
                let shared: Vec<usize> = char_to_words[ci].iter()
                    .filter(|w| char_to_words[cj].contains(w))
                    .copied()
                    .collect();
                let union = char_to_words[ci].len() + char_to_words[cj].len() - shared.len();
                let j = if union > 0 { shared.len() as f64 / union as f64 } else { 0.0 };
                if j > 0.0 {
                    char_jaccard[ci].push((cj, j));
                    char_jaccard[cj].push((ci, j));
                }
            }
        }

        // 词 Jaccard
        let mut word_jaccard: Vec<Vec<(usize, f64)>> = vec![Vec::new(); word_count];
        for wi in 0..word_count {
            for wj in (wi + 1)..word_count {
                let shared: Vec<usize> = word_to_chars[wi].iter()
                    .filter(|c| word_to_chars[wj].contains(c))
                    .copied()
                    .collect();
                let union = word_to_chars[wi].len() + word_to_chars[wj].len() - shared.len();
                let j = if union > 0 { shared.len() as f64 / union as f64 } else { 0.0 };
                if j > 0.0 {
                    word_jaccard[wi].push((wj, j));
                    word_jaccard[wj].push((wi, j));
                }
            }
        }

        // 词 containment
        let mut word_to_super_words: Vec<Vec<usize>> = vec![Vec::new(); word_count];
        for wi in 0..word_count {
            for wj in 0..word_count {
                if wi != wj {
                    let ci: Vec<usize> = word_to_chars[wi].clone();
                    if ci.iter().all(|c| word_to_chars[wj].contains(c)) {
                        word_to_super_words[wi].push(wj);
                    }
                }
            }
        }

        RelationNetwork {
            node_texts: {
                let mut n = words.clone();
                n.extend(chars.clone());
                n
            },
            is_word: {
                let mut m = vec![true; word_count];
                m.extend(vec![false; char_count]);
                m
            },
            word_count,
            char_to_words,
            word_to_chars,
            word_to_super_words,
            char_freq,
            word_len,
            char_jaccard,
            word_jaccard,
        }
    }

    #[test]
    fn test_organize_relations() {
        let psi = make_test_psi();
        let org = organize_relations(&psi, 0.25);
        // 至少应有字间关系
        assert!(!org.char_relations.is_empty(),
            "Should have character relations at threshold 0.25");
        // 所有关系的 Jaccard ≥ threshold
        for (_, _, j, _) in &org.char_relations {
            assert!(*j >= 0.25, "Jaccard {} should be >= 0.25", j);
        }
    }

    #[test]
    fn test_transitive_reason() {
        let psi = make_test_psi();
        let org = organize_relations(&psi, 0.2);
        let config = ReasonerConfig::default();
        let result = reason(&psi, &org, &config);
        let transitive: Vec<_> = result.iter()
            .filter(|r| matches!(r.kind, InferenceKind::Transitive))
            .collect();
        // 传递推理应有结果（字间路径长度 ≥ 2 时产生）
        eprintln!("Transitive inferences: {}", transitive.len());
    }

    #[test]
    fn test_pattern_completion_enabled() {
        let psi = make_test_psi();
        let org = organize_relations(&psi, 0.2);
        let config = ReasonerConfig {
            pattern_completion_enabled: true,
            min_confidence: 0.3,
            ..Default::default()
        };
        let result = reason(&psi, &org, &config);
        let pc: Vec<_> = result.iter()
            .filter(|r| matches!(r.kind, InferenceKind::PatternCompletion))
            .collect();
        eprintln!("Pattern completion inferences: {}", pc.len());
        // 至少生成了推理边
        for r in &pc {
            assert!(r.confidence >= 0.3 && r.confidence <= 1.0,
                "Confidence {} should be in [0.3, 1.0]", r.confidence);
        }
    }

    #[test]
    fn test_structural_similarity_enabled() {
        let psi = make_test_psi();
        let org = organize_relations(&psi, 0.2);
        let config = ReasonerConfig {
            structural_enabled: true,
            min_confidence: 0.3,
            ..Default::default()
        };
        let result = reason(&psi, &org, &config);
        let ss: Vec<_> = result.iter()
            .filter(|r| matches!(r.kind, InferenceKind::StructuralSimilarity))
            .collect();
        eprintln!("Structural similarity inferences: {}", ss.len());
        for r in &ss {
            assert!(r.confidence >= 0.3 && r.confidence <= 1.0,
                "Confidence {} should be in [0.3, 1.0]", r.confidence);
        }
    }

    #[test]
    fn test_full_inference_pipeline() {
        let psi = make_test_psi();
        let org = organize_relations(&psi, 0.2);
        let config = ReasonerConfig {
            pattern_completion_enabled: true,
            structural_enabled: true,
            min_confidence: 0.25,
            ..Default::default()
        };
        let result = reason(&psi, &org, &config);

        // 统计各类推理
        let mut counts = HashMap::new();
        for r in &result {
            *counts.entry(format!("{}", r.kind)).or_insert(0) += 1;
        }

        eprintln!("Inference counts: {:?}", counts);
        // 传递推理至少应有
        assert!(counts.contains_key("transitive") || counts.contains_key("cooccurrence"),
            "Should have at least transitive or cooccurrence inferences");
        // 所有推理置信度在有效范围内
        for r in &result {
            assert!(r.confidence > 0.0 && r.confidence <= 1.0);
            assert!(r.from != r.to, "Self-loops not allowed");
        }
    }

    #[test]
    fn test_build_concept_hierarchy() {
        let psi = make_test_psi();
        let org = organize_relations(&psi, 0.2);
        let mode = ConceptCycleMode::Converge;
        let hierarchy = build_concept_hierarchy(&psi, &org, mode, 5);

        assert!(hierarchy.depth >= 1, "Should have at least 1 level");
        assert!(!hierarchy.levels.is_empty());
        eprintln!("Hierarchy depth: {}, levels: {}", hierarchy.depth, hierarchy.levels.len());
        eprintln!("Termination: {:?}", hierarchy.termination_reasons);
    }

    #[test]
    fn test_derive_rules_and_constraints() {
        let psi = make_test_psi();
        let org = organize_relations(&psi, 0.2);
        let (rules, constraints) = derive_rules_and_constraints(&psi, &org);

        assert!(!constraints.is_empty(), "Should have constraints");
        eprintln!("Rules: {}, Constraints: {}", rules.len(), constraints.len());

        // 验证约束值
        for c in &constraints {
            assert!(c.value >= 0.0, "Constraint {} has negative value", c.invariant_equation);
        }
    }

    #[test]
    fn test_incompleteness_judgment() {
        let psi = make_test_psi();
        let org = organize_relations(&psi, 0.2);
        let hierarchy = build_concept_hierarchy(&psi, &org, ConceptCycleMode::Converge, 5);
        let report = judge_incompleteness(&psi, &hierarchy);

        assert!(report.gap_density >= 0.0 && report.gap_density <= 1.0);
        eprintln!("Incompleteness: radical={}, orphans={}, gaps={:.3}",
            report.is_radically_incomplete, report.orphan_count, report.gap_density);
    }

    #[test]
    fn test_consistency_verification() {
        let psi = make_test_psi();
        let org = organize_relations(&psi, 0.2);
        let hierarchy = build_concept_hierarchy(&psi, &org, ConceptCycleMode::Converge, 5);
        let (rules, constraints) = derive_rules_and_constraints(&psi, &org);
        let report = verify_consistency(&hierarchy, &rules, &constraints);

        eprintln!("Consistent: {}, contradictions: {:?}", report.is_consistent, report.contradictions);
    }

    #[test]
    fn test_run_phase_one() {
        let psi = make_test_psi();
        let n_chars = psi.char_jaccard.len();
        let node_levels = vec![1; n_chars];
        let org = organize_relations(&psi, 0.2);
        let output = run_phase_one(&psi, &org, ConceptCycleMode::Converge, 5, node_levels);

        assert!(output.structure.depth >= 1);
        assert!(!output.constraints.is_empty());
        eprintln!("Phase 1: depth={}, rules={}, constraints={}, consistent={}",
            output.structure.depth, output.rules.len(), output.constraints.len(), output.is_consistent);
    }

    #[test]
    fn test_decode_phase_one() {
        let psi = make_test_psi();
        let (words, chars) = decode_phase_one(&psi.node_texts, psi.word_count);
        assert_eq!(words.len(), psi.word_count);
        assert_eq!(chars.len(), psi.node_texts.len() - psi.word_count);
    }
}
