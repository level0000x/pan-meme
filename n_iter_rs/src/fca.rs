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

fn word_ngrams(word: &str, n_gram: usize) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() < n_gram { return vec![]; }
    let n = chars.len() - n_gram + 1;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let s: String = chars[i..i + n_gram].iter().collect();
        out.push(s);
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

pub fn build_article_lattice(texts: &[String], max_attrs: usize, min_articles: usize, max_concepts: usize, time_limit: f64) -> FcaLattice {
    let n_articles = texts.len();

    let mut word_freq: HashMap<String, Vec<usize>> = HashMap::new();
    for (ai, text) in texts.iter().enumerate() {
        let tokens = tokenize(text);
        let mut seen: HashSet<String> = HashSet::new();
        for t in tokens {
            if t.len() < 4 { continue; }
            if seen.insert(t.clone()) {
                word_freq.entry(t).or_default().push(ai);
            }
        }
    }

    let mut shared: Vec<(String, Vec<usize>)> = word_freq.into_iter()
        .filter(|(_, arts)| arts.len() >= min_articles)
        .collect();
    shared.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));
    let max_n = max_attrs.min(shared.len());
    shared.truncate(max_n);

    let top: Vec<String> = shared.iter().map(|(w, _)| w.clone()).collect();
    let ng_idx: HashMap<String, usize> = top.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
    let n_attrs = top.len();
    if n_attrs == 0 {
        return FcaLattice { concepts: vec![], edges: vec![], d_values: vec![], concept_sizes: vec![], n_words: n_articles, n_bigrams: 0 };
    }

    let mut art_to_attr: Vec<Vec<usize>> = vec![vec![]; n_articles];
    let mut attr_to_art: Vec<Vec<usize>> = vec![vec![]; n_attrs];

    for ai in 0..n_articles {
        let tokens = tokenize(&texts[ai]);
        let mut seen: HashSet<usize> = HashSet::new();
        for t in tokens {
            if let Some(&ati) = ng_idx.get(&t) {
                if seen.insert(ati) {
                    art_to_attr[ai].push(ati);
                    attr_to_art[ati].push(ai);
                }
            }
        }
    }
    for v in &mut art_to_attr { v.sort(); }
    for v in &mut attr_to_art { v.sort(); }

    let bg_data = BigramData {
        unique_words: texts.to_vec(),
        top_bigrams: top,
        word_to_bigram_idxs: art_to_attr,
        bigram_to_words: attr_to_art,
    };

    build_lattice_from_data(&bg_data, max_concepts, time_limit)
}

pub fn build_lattice_chars(text: &str, max_ngrams: usize, n_gram: usize, max_concepts: usize, time_limit: f64) -> FcaLattice {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    if n < n_gram { return FcaLattice { concepts: vec![], edges: vec![], d_values: vec![], concept_sizes: vec![], n_words: 0, n_bigrams: 0 }; }

    let mut ng_count: HashMap<String, usize> = HashMap::new();
    for i in 0..n - n_gram + 1 {
        let s: String = chars[i..i + n_gram].iter().collect();
        *ng_count.entry(s).or_insert(0) += 1;
    }

    let mut freq: Vec<(&String, &usize)> = ng_count.iter().collect();
    freq.sort_by(|a, b| b.1.cmp(a.1));
    let max_ng = max_ngrams.min(freq.len());
    let top: Vec<String> = freq[..max_ng].iter().map(|(s, _)| (*s).clone()).collect();

    let ng_idx: HashMap<String, usize> = top.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();

    let tokens = tokenize(text);
    let mut unique: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for t in tokens {
        if seen.insert(t.clone()) {
            unique.push(t);
        }
    }

    let mut word_to_ng: Vec<Vec<usize>> = Vec::with_capacity(unique.len());
    let mut ng_to_words: Vec<Vec<usize>> = vec![vec![]; top.len()];

    for (wi, w) in unique.iter().enumerate() {
        let wchars: Vec<char> = w.chars().collect();
        let mut idxs: Vec<usize> = Vec::new();
        let wl = wchars.len();
        for i in 0..wl.saturating_sub(n_gram - 1) {
            let end = (i + n_gram).min(wl);
            let s: String = wchars[i..end].iter().collect();
            if let Some(&ngi) = ng_idx.get(&s) {
                idxs.push(ngi);
                ng_to_words[ngi].push(wi);
            }
        }
        idxs.sort();
        idxs.dedup();
        for &ngi in &idxs {
            ng_to_words[ngi].push(wi);
        }
        word_to_ng.push(idxs);
    }
    for nw in &mut ng_to_words {
        nw.sort();
        nw.dedup();
    }

    let bg_data = BigramData {
        unique_words: unique,
        top_bigrams: top,
        word_to_bigram_idxs: word_to_ng,
        bigram_to_words: ng_to_words,
    };

    build_lattice_from_data(&bg_data, max_concepts, time_limit)
}

pub fn build_article_lattice_chars(texts: &[String], max_ngrams: usize, n_gram: usize, max_concepts: usize, time_limit: f64) -> FcaLattice {
    let n_articles = texts.len();
    let mut merged = String::new();
    for t in texts { merged.push_str(t); merged.push(' '); }
    let chars: Vec<char> = merged.chars().collect();
    let cn = chars.len();
    if cn < n_gram { return FcaLattice { concepts: vec![], edges: vec![], d_values: vec![], concept_sizes: vec![], n_words: 0, n_bigrams: 0 }; }

    let mut ng_count: HashMap<String, usize> = HashMap::new();
    for i in 0..cn - n_gram + 1 {
        let s: String = chars[i..i + n_gram].iter().collect();
        *ng_count.entry(s).or_insert(0) += 1;
    }

    let mut freq: Vec<(&String, &usize)> = ng_count.iter().collect();
    freq.sort_by(|a, b| b.1.cmp(a.1));
    let max_ng = max_ngrams.min(freq.len());
    let top: Vec<String> = freq[..max_ng].iter().map(|(s, _)| (*s).clone()).collect();
    let ng_idx: HashMap<String, usize> = top.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();

    let mut bg_data = BigramData {
        unique_words: texts.to_vec(),
        top_bigrams: top,
        word_to_bigram_idxs: vec![vec![]; n_articles],
        bigram_to_words: vec![vec![]; max_ng],
    };

    for (ai, text) in texts.iter().enumerate() {
        let tchars: Vec<char> = text.chars().collect();
        let tl = tchars.len();
        let mut seen_ng: HashSet<usize> = HashSet::new();
        for i in 0..tl.saturating_sub(n_gram - 1) {
            let end = (i + n_gram).min(tl);
            let s: String = tchars[i..end].iter().collect();
            if let Some(&ngi) = ng_idx.get(&s) {
                if seen_ng.insert(ngi) {
                    bg_data.word_to_bigram_idxs[ai].push(ngi);
                    bg_data.bigram_to_words[ngi].push(ai);
                }
            }
        }
    }

    for v in &mut bg_data.word_to_bigram_idxs { v.sort(); v.dedup(); }
    for v in &mut bg_data.bigram_to_words { v.sort(); v.dedup(); }

    build_lattice_from_data(&bg_data, max_concepts, time_limit)
}

pub fn build_paragraph_lattice(texts: &[String], max_attrs: usize, min_paragraphs: usize, max_concepts: usize, time_limit: f64) -> FcaLattice {
    let mut paragraphs: Vec<String> = Vec::new();
    for text in texts {
        for para in text.split("\n\n") {
            let trimmed = para.trim();
            if trimmed.is_empty() { continue; }
            let words: Vec<&str> = trimmed.split_whitespace().collect();
            if words.len() >= 10 {
                paragraphs.push(trimmed.to_string());
            }
        }
    }
    if paragraphs.is_empty() {
        return FcaLattice { concepts: vec![], edges: vec![], d_values: vec![], concept_sizes: vec![], n_words: 0, n_bigrams: 0 };
    }
    let n_paras = paragraphs.len();

    let mut word_freq: HashMap<String, Vec<usize>> = HashMap::new();
    for (pi, para) in paragraphs.iter().enumerate() {
        let tokens = tokenize(para);
        let mut seen: HashSet<String> = HashSet::new();
        for t in tokens {
            if t.len() < 4 { continue; }
            if seen.insert(t.clone()) {
                word_freq.entry(t).or_default().push(pi);
            }
        }
    }

    let mut shared: Vec<(String, Vec<usize>)> = word_freq.into_iter()
        .filter(|(_, paras)| paras.len() >= min_paragraphs)
        .collect();
    shared.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));
    let max_n = max_attrs.min(shared.len());
    shared.truncate(max_n);

    let top: Vec<String> = shared.iter().map(|(w, _)| w.clone()).collect();
    let ng_idx: HashMap<String, usize> = top.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
    let n_attrs = top.len();
    if n_attrs == 0 {
        return FcaLattice { concepts: vec![], edges: vec![], d_values: vec![], concept_sizes: vec![], n_words: n_paras, n_bigrams: 0 };
    }

    let mut para_to_attr: Vec<Vec<usize>> = vec![vec![]; n_paras];
    let mut attr_to_para: Vec<Vec<usize>> = vec![vec![]; n_attrs];

    for pi in 0..n_paras {
        let tokens = tokenize(&paragraphs[pi]);
        let mut seen: HashSet<usize> = HashSet::new();
        for t in tokens {
            if let Some(&ati) = ng_idx.get(&t) {
                if seen.insert(ati) {
                    para_to_attr[pi].push(ati);
                    attr_to_para[ati].push(pi);
                }
            }
        }
    }
    for v in &mut para_to_attr { v.sort(); }
    for v in &mut attr_to_para { v.sort(); }

    let bg_data = BigramData {
        unique_words: paragraphs,
        top_bigrams: top,
        word_to_bigram_idxs: para_to_attr,
        bigram_to_words: attr_to_para,
    };

    build_lattice_from_data(&bg_data, max_concepts, time_limit)
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
            if is_cover { edges.push((j, i)); }
        }
    }
    edges
}

pub fn build_lattice(text: &str, max_bigrams: usize, max_concepts: usize, time_limit: f64) -> FcaLattice {
    let bg_data = build_bigram_data(text, max_bigrams);
    build_lattice_from_data(&bg_data, max_concepts, time_limit)
}

pub fn build_cooccurrence_lattice(text: &str, window: usize, max_attrs: usize, max_concepts: usize, time_limit: f64) -> FcaLattice {
    let tokens_all: Vec<String> = tokenize(text);
    let mut unique: Vec<String> = Vec::new();
    let mut seen = HashSet::new();
    for t in &tokens_all {
        if seen.insert(t.clone()) {
            unique.push(t.clone());
        }
    }
    let n_words = unique.len();
    let word_idx: HashMap<String, usize> = unique.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect();

    let mut cooc: Vec<HashSet<usize>> = vec![HashSet::new(); n_words];
    let tl = tokens_all.len();
    for (pos, t) in tokens_all.iter().enumerate() {
        if let Some(&wi) = word_idx.get(t) {
            let start = if pos > window { pos - window } else { 0 };
            let end = (pos + window + 1).min(tl);
            for j in start..end {
                if j == pos { continue; }
                if let Some(&wj) = word_idx.get(&tokens_all[j]) {
                    cooc[wi].insert(wj);
                }
            }
        }
    }

    let mut word_to_attr: Vec<Vec<usize>> = Vec::with_capacity(n_words);
    for w in 0..n_words {
        let mut attrs: Vec<usize> = cooc[w].iter().cloned().collect();
        attrs.sort();
        word_to_attr.push(attrs);
    }

    let mut attr_to_word: Vec<Vec<usize>> = vec![Vec::new(); n_words];
    for w in 0..n_words {
        for &a in &word_to_attr[w] {
            attr_to_word[a].push(w);
        }
    }
    for v in &mut attr_to_word { v.sort(); v.dedup(); }

    let mut attr_freq: Vec<(usize, usize)> = attr_to_word.iter().enumerate().map(|(i, v)| (i, v.len())).collect();
    attr_freq.sort_by(|a, b| b.1.cmp(&a.1));
    let max_n = max_attrs.min(attr_freq.len());
    let top_attrs: Vec<usize> = attr_freq[..max_n].iter().map(|(i, _)| *i).collect();
    let attr_map: HashMap<usize, usize> = top_attrs.iter().enumerate().map(|(new_i, &old_i)| (old_i, new_i)).collect();

    let mut trimmed_word_to_attr: Vec<Vec<usize>> = vec![Vec::new(); n_words];
    for w in 0..n_words {
        for &a in &word_to_attr[w] {
            if let Some(&new_a) = attr_map.get(&a) {
                trimmed_word_to_attr[w].push(new_a);
            }
        }
    }
    let mut trimmed_attr_to_word: Vec<Vec<usize>> = vec![Vec::new(); max_n];
    for w in 0..n_words {
        for &a in &trimmed_word_to_attr[w] {
            trimmed_attr_to_word[a].push(w);
        }
    }
    for v in &mut trimmed_attr_to_word { v.sort(); v.dedup(); }

    let top_labels: Vec<String> = top_attrs.iter().map(|&i| unique[i].clone()).collect();

    let bg_data = BigramData {
        unique_words: unique,
        top_bigrams: top_labels,
        word_to_bigram_idxs: trimmed_word_to_attr,
        bigram_to_words: trimmed_attr_to_word,
    };

    build_lattice_from_data(&bg_data, max_concepts, time_limit)
}

pub fn build_lattice_plain(text: &str, max_ngrams: usize, n_gram: usize, max_concepts: usize, time_limit: f64) -> FcaLattice {
    let bg_data = build_ngram_data(text, max_ngrams, n_gram);
    build_lattice_from_data(&bg_data, max_concepts, time_limit)
}

pub fn build_ngram_data(text: &str, max_ngrams: usize, n_gram: usize) -> BigramData {
    let tokens = tokenize(text);
    let mut unique: Vec<String> = Vec::new();
    let mut seen = HashSet::new();
    for t in tokens {
        if seen.insert(t.clone()) {
            unique.push(t);
        }
    }

    let mut ng_count: HashMap<String, usize> = HashMap::new();
    let mut w_ngs: Vec<Vec<String>> = Vec::with_capacity(unique.len());
    for w in &unique {
        let ngs = word_ngrams(w, n_gram);
        for ng in &ngs {
            *ng_count.entry(ng.clone()).or_insert(0) += 1;
        }
        w_ngs.push(ngs);
    }

    let mut freq: Vec<(&String, &usize)> = ng_count.iter().collect();
    freq.sort_by(|a, b| b.1.cmp(a.1));
    let top: Vec<String> = freq.iter()
        .take(max_ngrams).map(|(s, _)| (*s).clone()).collect();

    let ng_idx: HashMap<String, usize> = top.iter().enumerate()
        .map(|(i, s)| (s.clone(), i)).collect();
    let n_ng = top.len();

    let mut word_to_ng: Vec<Vec<usize>> = Vec::with_capacity(unique.len());
    let mut ng_to_words: Vec<Vec<usize>> = vec![vec![]; n_ng];
    for (wi, ngs) in w_ngs.iter().enumerate() {
        let mut idxs: Vec<usize> = ngs.iter()
            .filter_map(|ng| ng_idx.get(ng).copied()).collect();
        idxs.sort();
        for &bi in &idxs {
            ng_to_words[bi].push(wi);
        }
        word_to_ng.push(idxs);
    }
    for nw in &mut ng_to_words {
        nw.sort();
        nw.dedup();
    }

    BigramData {
        unique_words: unique,
        top_bigrams: top,
        word_to_bigram_idxs: word_to_ng,
        bigram_to_words: ng_to_words,
    }
}

pub fn build_synthetic_lattice(
    matrix: &[Vec<bool>],
    max_concepts: usize,
    time_limit: f64,
) -> FcaLattice {
    let n_objects = matrix.len();
    if n_objects == 0 { return empty_lattice(); }
    let n_attrs = matrix[0].len();
    if n_attrs == 0 { return empty_lattice(); }

    let mut obj_to_attr: Vec<Vec<usize>> = vec![vec![]; n_objects];
    let mut attr_to_obj: Vec<Vec<usize>> = vec![vec![]; n_attrs];
    for i in 0..n_objects {
        for j in 0..n_attrs {
            if matrix[i][j] {
                obj_to_attr[i].push(j);
                attr_to_obj[j].push(i);
            }
        }
    }

    let attr_names: Vec<String> = (0..n_attrs).map(|i| format!("a{}", i)).collect();
    let obj_names: Vec<String> = (0..n_objects).map(|i| format!("o{}", i)).collect();

    let bg = BigramData {
        unique_words: obj_names,
        top_bigrams: attr_names,
        word_to_bigram_idxs: obj_to_attr,
        bigram_to_words: attr_to_obj,
    };

    build_lattice_from_data(&bg, max_concepts, time_limit)
}

fn empty_lattice() -> FcaLattice {
    FcaLattice {
        concepts: vec![],
        edges: vec![],
        d_values: vec![],
        concept_sizes: vec![],
        n_words: 0,
        n_bigrams: 0,
    }
}

pub fn build_lattice_from_concepts(concepts: Vec<FormalConcept>) -> FcaLattice {
    let n_words = concepts.iter().flat_map(|c| &c.extent).max().map(|&x| x + 1).unwrap_or(0);
    let n_bigrams = concepts.iter().flat_map(|c| &c.intent).max().map(|&x| x + 1).unwrap_or(0);
    let d_values: Vec<f64> = concepts.iter().map(|c| {
        let na = c.intent.len() as f64;
        let nb = c.extent.len() as f64;
        if nb == 0.0 { f64::INFINITY } else { na / nb }
    }).collect();
    let concept_sizes: Vec<(usize, usize)> = concepts.iter()
        .map(|c| (c.intent.len(), c.extent.len())).collect();
    let edges = build_hasse_edges(&concepts);
    FcaLattice { concepts, edges, d_values, concept_sizes, n_words, n_bigrams }
}

/// Build a chain lattice with n concepts (linear order)
pub fn build_chain_lattice(n: usize) -> FcaLattice {
    let mut concepts = Vec::with_capacity(n);
    for i in 0..n {
        concepts.push(FormalConcept {
            intent: (0..i).collect(),
            extent: (i..n).collect(),
        });
    }
    build_lattice_from_concepts(concepts)
}

/// Build a diamond lattice (Boolean lattice B_2 = 4 concepts)
pub fn build_diamond_lattice() -> FcaLattice {
    let concepts = vec![
        FormalConcept { intent: vec![], extent: vec![0, 1, 2, 3] },
        FormalConcept { intent: vec![0], extent: vec![0, 2] },
        FormalConcept { intent: vec![1], extent: vec![1, 3] },
        FormalConcept { intent: vec![0, 1], extent: vec![] },
    ];
    build_lattice_from_concepts(concepts)
}

/// Build an M3 lattice (5 concepts: top, 3 incomparable middle, bottom)
pub fn build_m3_lattice() -> FcaLattice {
    let concepts = vec![
        FormalConcept { intent: vec![], extent: vec![0, 1, 2, 3, 4, 5] },
        FormalConcept { intent: vec![0], extent: vec![0, 1] },
        FormalConcept { intent: vec![1], extent: vec![2, 3] },
        FormalConcept { intent: vec![2], extent: vec![4, 5] },
        FormalConcept { intent: vec![0, 1, 2], extent: vec![] },
    ];
    build_lattice_from_concepts(concepts)
}

/// Build a Boolean lattice B_3 (8 concepts = 2^3, cube)
pub fn build_b3_lattice() -> FcaLattice {
    let concepts = vec![
        FormalConcept { intent: vec![], extent: vec![0, 1, 2, 3, 4, 5, 6, 7] },
        FormalConcept { intent: vec![0], extent: vec![1, 3, 5, 7] },
        FormalConcept { intent: vec![1], extent: vec![2, 3, 6, 7] },
        FormalConcept { intent: vec![2], extent: vec![4, 5, 6, 7] },
        FormalConcept { intent: vec![0, 1], extent: vec![3, 7] },
        FormalConcept { intent: vec![0, 2], extent: vec![5, 7] },
        FormalConcept { intent: vec![1, 2], extent: vec![6, 7] },
        FormalConcept { intent: vec![0, 1, 2], extent: vec![] },
    ];
    build_lattice_from_concepts(concepts)
}

/// Build a grid (product) lattice: rows × cols
pub fn build_grid_lattice(rows: usize, cols: usize) -> FcaLattice {
    let n = rows * cols;
    let mut concepts = Vec::with_capacity(n);
    for r in 0..rows {
        for c in 0..cols {
            let intent: Vec<usize> = (0..r).chain(rows..rows + c).collect();
            let mut extent = Vec::new();
            for rr in r..rows {
                for cc in c..cols {
                    extent.push(rr * cols + cc);
                }
            }
            concepts.push(FormalConcept { intent, extent });
        }
    }
    build_lattice_from_concepts(concepts)
}

/// Build an antichain lattice: top + k independent middle + bottom
pub fn build_antichain_lattice(k: usize) -> FcaLattice {
    let mut concepts = Vec::with_capacity(k + 2);
    let mut all_extent: Vec<usize> = Vec::new();
    let mut all_intent: Vec<usize> = Vec::new();
    for i in 0..k {
        concepts.push(FormalConcept {
            intent: vec![i],
            extent: vec![i],
        });
        all_extent.push(i);
        all_intent.push(i);
    }
    concepts.insert(0, FormalConcept {
        intent: vec![],
        extent: all_extent.clone(),
    });
    concepts.push(FormalConcept {
        intent: all_intent,
        extent: vec![],
    });
    build_lattice_from_concepts(concepts)
}

/// Build a Boolean lattice B_4 (16 concepts = 2^4)
pub fn build_b4_lattice() -> FcaLattice {
    let mut concepts = Vec::with_capacity(16);
    // 16 objects (0..15), 4 attributes (0..3)
    // Object i has attribute j if bit j of i is 1
    for intent_mask in 0..16u32 {
        let mut intent = Vec::new();
        let mut extent = Vec::new();
        for j in 0..4 {
            if (intent_mask >> j) & 1 == 1 { intent.push(j); }
        }
        for obj in 0..16 {
            if (obj & intent_mask) == intent_mask { extent.push(obj as usize); }
        }
        concepts.push(FormalConcept { intent, extent });
    }
    build_lattice_from_concepts(concepts)
}

pub fn build_lattice_from_data(bg_data: &BigramData, max_concepts: usize, time_limit: f64) -> FcaLattice {
    let concepts = next_closure(bg_data, max_concepts, time_limit);
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
    let mut parents: Vec<Vec<usize>> = vec![vec![]; n];
    for &(p, c) in edges { parents[c].push(p); }
    let mut heights = vec![usize::MAX; n];
    fn dfs(v: usize, pa: &[Vec<usize>], h: &mut [usize]) -> usize {
        if h[v] != usize::MAX { return h[v]; }
        h[v] = if pa[v].is_empty() { 0 }
               else { 1 + pa[v].iter().map(|&p| dfs(p, pa, h)).max().unwrap_or(0) };
        h[v]
    }
    for i in 0..n { dfs(i, &parents, &mut heights); }
    heights
}

pub fn verify_theorem_11_3(_concepts: &[FormalConcept], d_values: &[f64], edges: &[(usize, usize)]) -> (usize, usize) {
    let mut passes = 0;
    let mut fails = 0;
    for &(general, specific) in edges {
        let d_gen = d_values[general];
        let d_spec = d_values[specific];
        let pass = if d_spec.is_infinite() {
            true
        } else if d_gen.is_infinite() {
            false
        } else {
            d_spec >= d_gen
        };
        if pass { passes += 1; } else { fails += 1; }
    }
    (passes, fails)
}

pub fn verify_theorem_11_1(tau_inv: &[f64], edges: &[(usize, usize)]) -> (usize, usize) {
    let mut passes = 0;
    let mut fails = 0;
    for &(general, specific) in edges {
        let t_gen = tau_inv[general];
        let t_spec = tau_inv[specific];
        if t_gen.is_finite() && t_spec.is_finite() && t_gen >= t_spec { passes += 1; }
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
        assert!(p + f > 0, "Theorem 11.3 check should process edges");
    }
}
