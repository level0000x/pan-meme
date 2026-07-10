use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub struct FormalConcept {
    pub intent: Vec<usize>,
    pub extent: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct FcaLattice {
    pub concepts: Vec<FormalConcept>,
    pub edges: Vec<(usize, usize)>,
    pub d_values: Vec<f64>,
    pub concept_sizes: Vec<(usize, usize)>,
    pub n_words: usize,
    pub n_bigrams: usize,
}

#[derive(Debug, Clone)]
pub struct BigramData {
    pub unique_words: Vec<String>,
    pub top_bigrams: Vec<String>,
    pub word_to_bigram_idxs: Vec<Vec<usize>>,
    pub bigram_to_words: Vec<Vec<usize>>,
}

pub fn tokenize(text: &str) -> Vec<String> {
    let stopwords: [&str; 44] = [
        "the", "and", "for", "are", "was", "were", "that", "this",
        "with", "from", "have", "been", "their", "which", "they",
        "not", "but", "has", "had", "its", "can", "all", "also",
        "than", "more", "some", "other", "each", "about", "would",
        "when", "will", "these", "such", "only", "over", "into",
        "most", "after", "where", "between", "being", "those", "them",
    ];
    let stop_set: HashSet<&str> = stopwords.iter().copied().collect();
    let mut words = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphabetic() {
            current.push(ch.to_ascii_lowercase());
        } else {
            if current.len() >= 4 && !stop_set.contains(current.as_str()) {
                words.push(std::mem::take(&mut current));
            }
            current.clear();
        }
    }
    if current.len() >= 4 && !stop_set.contains(current.as_str()) {
        words.push(current);
    }
    words
}

fn word_to_bigrams(word: &str) -> Vec<String> {
    let mut out = Vec::new();
    let chars: Vec<char> = word.chars().collect();
    for i in 0..chars.len().saturating_sub(1) {
        out.push(format!("{}{}_{}", chars[i], chars[i + 1], i));
    }
    out
}

fn word_to_bigrams_plain(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let n = chars.len().saturating_sub(1);
    if n == 0 { return vec![]; }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(format!("{}{}", chars[i], chars[i + 1]));
    }
    out
}

pub fn build_bigram_data(text: &str, max_bigrams: usize) -> BigramData {
    let tokens = tokenize(text);
    let mut unique: Vec<String> = Vec::new();
    let mut seen = HashSet::new();
    for t in tokens {
        if seen.insert(t.clone()) {
            unique.push(t);
        }
    }

    let mut bg_count: HashMap<String, usize> = HashMap::new();
    let mut w_bgs: Vec<Vec<String>> = Vec::with_capacity(unique.len());
    for w in &unique {
        let bgs = word_to_bigrams(w);
        for bg in &bgs {
            *bg_count.entry(bg.clone()).or_insert(0) += 1;
        }
        w_bgs.push(bgs);
    }

    let mut freq: Vec<(&String, &usize)> = bg_count.iter().collect();
    freq.sort_by(|a, b| b.1.cmp(a.1));
    let top: Vec<String> = freq.iter()
        .take(max_bigrams).map(|(s, _)| (*s).clone()).collect();

    let bg_idx: HashMap<String, usize> = top.iter().enumerate()
        .map(|(i, s)| (s.clone(), i)).collect();
    let n_bg = top.len();

    let mut word_to_bg: Vec<Vec<usize>> = Vec::with_capacity(unique.len());
    let mut bg_to_words: Vec<Vec<usize>> = vec![vec![]; n_bg];
    for (wi, bgs) in w_bgs.iter().enumerate() {
        let mut idxs: Vec<usize> = bgs.iter()
            .filter_map(|bg| bg_idx.get(bg).copied()).collect();
        idxs.sort();
        for &bi in &idxs {
            bg_to_words[bi].push(wi);
        }
        word_to_bg.push(idxs);
    }
    for bw in &mut bg_to_words {
        bw.sort();
        bw.dedup();
    }

    BigramData {
        unique_words: unique,
        top_bigrams: top,
        word_to_bigram_idxs: word_to_bg,
        bigram_to_words: bg_to_words,
    }
}

pub fn next_closure(
    bg_data: &BigramData,
    max_concepts: usize,
    time_limit_secs: f64,
) -> Vec<FormalConcept> {
    let n_words = bg_data.unique_words.len();
    let n_bg = bg_data.top_bigrams.len();
    if n_bg == 0 { return vec![]; }

    let derive_up = |intent: &[usize]| -> Vec<usize> {
        if intent.is_empty() {
            return (0..n_words).collect();
        }
        let mut ext: HashSet<usize> = bg_data.bigram_to_words[intent[0]].iter().copied().collect();
        for &bi in &intent[1..] {
            let s: HashSet<usize> = bg_data.bigram_to_words[bi].iter().copied().collect();
            ext = ext.intersection(&s).copied().collect();
            if ext.is_empty() { break; }
        }
        let mut v: Vec<usize> = ext.into_iter().collect(); v.sort(); v
    };

    let derive_down = |extent: &[usize]| -> Vec<usize> {
        if extent.is_empty() { return (0..n_bg).collect(); }
        let mut int: HashSet<usize> = bg_data.word_to_bigram_idxs[extent[0]].iter().copied().collect();
        for &wi in &extent[1..] {
            let s: HashSet<usize> = bg_data.word_to_bigram_idxs[wi].iter().copied().collect();
            int = int.intersection(&s).copied().collect();
            if int.is_empty() { break; }
        }
        let mut v: Vec<usize> = int.into_iter().collect(); v.sort(); v
    };

    let closure = |intent: &[usize]| -> Vec<usize> {
        derive_down(&derive_up(intent))
    };

    let start = std::time::Instant::now();
    let mut concepts: Vec<FormalConcept> = Vec::new();
    let empty: Vec<usize> = vec![];
    let first_intent = closure(&empty);
    let first_extent = derive_up(&first_intent);
    concepts.push(FormalConcept { intent: first_intent.clone(), extent: first_extent });

    for _ in 1..max_concepts {
        if start.elapsed().as_secs_f64() >= time_limit_secs { break; }
        let last = concepts.last().unwrap();
        let mut found = false;

        for wi in (0..=n_bg).rev() {
            let mut candidate: Vec<usize> = last.intent.iter()
                .filter(|&&bi| bi < wi || wi == n_bg).copied().collect();
            if wi < n_bg {
                match candidate.binary_search(&wi) {
                    Err(i) => candidate.insert(i, wi),
                    Ok(_) => {}
                }
            }
            let closed = closure(&candidate);
            let ext = derive_up(&closed);
            if concepts.iter().any(|c| c.intent == closed) { continue; }
            let is_lex_larger = wi == n_bg
                || closed.len() > last.intent.len()
                || (closed.len() == last.intent.len()
                    && closed.iter().zip(&last.intent).any(|(a, b)| a > b));
            if is_lex_larger {
                concepts.push(FormalConcept { intent: closed, extent: ext });
                found = true;
                break;
            }
        }
        if !found { break; }
    }
    concepts
}

pub fn build_article_lattice(
    texts: &[String],
    max_bigrams: usize,
    max_concepts: usize,
    time_limit: f64,
) -> FcaLattice {
    let n_articles = texts.len();
    let mut bigram_freqs: HashMap<String, usize> = HashMap::new();
    let mut article_bigrams: Vec<Vec<String>> = vec![vec![]; n_articles];

    for (ai, text) in texts.iter().enumerate() {
        let tokens = tokenize(text);
        let mut seen: HashSet<String> = HashSet::new();
        for t in tokens {
            if t.len() < 4 { continue; }
            if seen.insert(t.clone()) {
                *bigram_freqs.entry(t.clone()).or_insert(0) += 1;
                article_bigrams[ai].push(t);
            }
        }
    }

    let mut freq_vec: Vec<(String, usize)> = bigram_freqs.into_iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let top_n = max_bigrams.min(freq_vec.len());
    let top: Vec<String> = freq_vec[..top_n].iter().map(|(bg, _)| bg.clone()).collect();

    let bg_idx: HashMap<String, usize> = top.iter().enumerate()
        .map(|(i, s)| (s.clone(), i)).collect();
    let n_bg = top.len();

    let mut art_to_bg: Vec<Vec<usize>> = Vec::with_capacity(n_articles);
    let mut bg_to_art: Vec<Vec<usize>> = vec![vec![]; n_bg];

    for (ai, bgs) in article_bigrams.iter().enumerate() {
        let mut idxs: Vec<usize> = bgs.iter()
            .filter_map(|bg| bg_idx.get(bg).copied()).collect();
        idxs.sort();
        idxs.dedup();
        for &bi in &idxs {
            bg_to_art[bi].push(ai);
        }
        art_to_bg.push(idxs);
    }
    for ba in &mut bg_to_art {
        ba.sort();
        ba.dedup();
    }

    let bg_data = BigramData {
        unique_words: texts.to_vec(),
        top_bigrams: top,
        word_to_bigram_idxs: art_to_bg,
        bigram_to_words: bg_to_art,
    };

    let concepts = next_closure(&bg_data, max_concepts, time_limit);
    let d_values: Vec<f64> = concepts.iter().map(|c| {
        let na = c.intent.len() as f64;
        let nb = c.extent.len() as f64;
        if nb == 0.0 { f64::INFINITY } else { na / nb }
    }).collect();
    let concept_sizes: Vec<(usize, usize)> = concepts.iter()
        .map(|c| (c.intent.len(), c.extent.len())).collect();
    let edges = build_hasse_edges(&concepts);

    FcaLattice {
        concepts,
        edges,
        d_values,
        concept_sizes,
        n_words: n_articles,
        n_bigrams: n_bg,
    }
}

pub fn build_hasse_edges(concepts: &[FormalConcept]) -> Vec<(usize, usize)> {
    let n = concepts.len();
    let mut edges = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            let a_j_subseteq_a_i = concepts[j].intent.iter().all(|x| concepts[i].intent.contains(x));
            if !a_j_subseteq_a_i { continue; }
            let is_cover = !(0..n).any(|k| {
                k != i && k != j
                    && concepts[j].intent.iter().all(|x| concepts[k].intent.contains(x))
                    && concepts[k].intent.iter().all(|x| concepts[i].intent.contains(x))
            });
            if is_cover { edges.push((i, j)); }
        }
    }
    edges
}

pub fn build_lattice(text: &str, max_bigrams: usize, max_concepts: usize, time_limit: f64) -> FcaLattice {
    let bg_data = build_bigram_data(text, max_bigrams);
    let concepts = next_closure(&bg_data, max_concepts, time_limit);
    let d_values: Vec<f64> = concepts.iter().map(|c| {
        let na = c.intent.len() as f64;
        let nb = c.extent.len() as f64;
        if nb == 0.0 { f64::INFINITY } else { na / nb }
    }).collect();
    let concept_sizes: Vec<(usize, usize)> = concepts.iter()
        .map(|c| (c.intent.len(), c.extent.len())).collect();
    let edges = build_hasse_edges(&concepts);

    FcaLattice {
        concepts,
        edges,
        d_values,
        concept_sizes,
        n_words: bg_data.unique_words.len(),
        n_bigrams: bg_data.top_bigrams.len(),
    }
}

pub fn hasse_heights(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut children: Vec<Vec<usize>> = vec![vec![]; n];
    for &(p, c) in edges { children[p].push(c); }
    let mut heights = vec![usize::MAX; n];
    fn dfs(v: usize, ch: &[Vec<usize>], h: &mut [usize]) -> usize {
        if h[v] != usize::MAX { return h[v]; }
        h[v] = if ch[v].is_empty() { 0 }
               else { 1 + ch[v].iter().map(|&c| dfs(c, ch, h)).max().unwrap_or(0) };
        h[v]
    }
    for i in 0..n { dfs(i, &children, &mut heights); }
    heights
}

pub fn verify_theorem_11_3(_concepts: &[FormalConcept], d_values: &[f64], edges: &[(usize, usize)]) -> (usize, usize) {
    let mut passes = 0;
    let mut fails = 0;
    for &(p, c) in edges {
        let dp = d_values[p];
        let dc = d_values[c];
        let pass = if dp.is_infinite() {
            true
        } else if dc.is_infinite() {
            false
        } else {
            dp >= dc
        };
        if pass { passes += 1; } else { fails += 1; }
    }
    (passes, fails)
}

pub fn verify_theorem_11_1(tau_inv: &[f64], edges: &[(usize, usize)]) -> (usize, usize) {
    let mut passes = 0;
    let mut fails = 0;
    for &(p, c) in edges {
        let tp = tau_inv[p];
        let tc = tau_inv[c];
        if tp.is_finite() && tc.is_finite() && tp >= tc { passes += 1; }
        else { fails += 1; }
    }
    (passes, fails)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_lattice() {
        let text = "Calculus is the mathematical study of continuous change";
        let lat = build_lattice(text, 200, 100, 30.0);
        assert!(lat.concepts.len() >= 2);
        assert!(!lat.d_values.is_empty());
    }

    #[test]
    fn test_theorem_11_3() {
        let text = "Calculus is the mathematical study of continuous change";
        let lat = build_lattice(text, 200, 100, 30.0);
        let (p, f) = verify_theorem_11_3(&lat.concepts, &lat.d_values, &lat.edges);
        assert_eq!(f, 0, "Theorem 11.3: {} violations", f);
        assert!(p > 0);
    }
}
