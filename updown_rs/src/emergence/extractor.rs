//! 提取器 — 从原始词列表构建 Ψ = (V, E) + Jaccard 连接强度
//!
//! 数学对应：
//!   泛模因前提2（提取算法 A: I → Ψ）
//!   定义 2.1（初始关系网络）
//!   3.2.2（连接强度归一化 — Jaccard）
//!
//! C 从 W 自暴露：C = ∪_{w∈W} chars(w)
//! 字—字 Jaccard: |↑(c₁) ∩ ↑(c₂)| / |↑(c₁) ∪ ↑(c₂)|
//! 词—词 Jaccard: |↓(w₁) ∩ ↓(w₂)| / |↓(w₁) ∪ ↓(w₂)|

use std::collections::HashMap;

/// Sparse Jaccard pair: (other_index, jaccard_value)
pub type JaccardRow = Vec<(usize, f64)>;

pub enum ContainmentEdge {
    CharInWord { char_idx: usize, word_idx: usize },
    WordInWord { sub_idx: usize, super_idx: usize },
}

#[derive(Debug)]
pub struct ExtractorConfig {
    pub word_substring_containment: bool,
    pub min_subordinate_len: usize,
    pub max_substring_window: usize,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            word_substring_containment: true,
            min_subordinate_len: 1,
            max_substring_window: 6,
        }
    }
}

/// 关系网络 Ψ = (V, E, w)
///
/// 数学对应：泛模因定义 1 — Ψ = A(I) = (V, E, w)
/// w: E → [0,1] 为 Jaccard 连接强度（3.2.2）
#[derive(Debug)]
pub struct RelationNetwork {
    pub node_texts: Vec<String>,
    pub is_word: Vec<bool>,
    pub word_count: usize,
    /// 字 → 包含它的词索引列表（↑(c) 查询）
    pub char_to_words: Vec<Vec<usize>>,
    /// 词 → 构成它的字索引列表（↓(w) 查询）
    pub word_to_chars: Vec<Vec<usize>>,
    /// 词 → 包含它为子串的更长的词（↑(w) 的词级 containment）
    pub word_to_super_words: Vec<Vec<usize>>,
    /// 字索引 → 字所在的词总数（共现频次）
    pub char_freq: Vec<usize>,
    /// 词索引 → 字数
    pub word_len: Vec<usize>,
    /// 字—字 Jaccard 连接强度（稀疏）：char_jaccard[ci] = [(cj, Jaccard(ci,cj)), ...]
    pub char_jaccard: Vec<JaccardRow>,
    /// 词—词 Jaccard 连接强度（稀疏）：word_jaccard[wi] = [(wj, Jaccard(wi,wj)), ...]
    pub word_jaccard: Vec<JaccardRow>,
}

impl RelationNetwork {
    /// 从词列表构建关系网络（含 Jaccard 连接强度计算）。
    pub fn from_words(words: Vec<String>, config: &ExtractorConfig) -> Self {
        // ── Step 1: 词去重 ──
        let mut word_set: HashMap<String, usize> = HashMap::new();
        let mut word_list: Vec<String> = Vec::new();
        for w in words {
            let w = w.trim().to_string();
            if w.is_empty() { continue; }
            word_set.entry(w.clone()).or_insert_with(|| {
                let idx = word_list.len();
                word_list.push(w.clone());
                idx
            });
        }
        let word_count = word_list.len();

        // ── Step 2: 提取唯一字 — C 自暴露 ──
        let mut char_set: HashMap<char, usize> = HashMap::new();
        let mut char_list: Vec<String> = Vec::new();
        for w in &word_list {
            for c in w.chars() {
                char_set.entry(c).or_insert_with(|| {
                    let idx = char_list.len();
                    char_list.push(c.to_string());
                    idx
                });
            }
        }
        let char_count = char_list.len();

        // ── Step 3: 节点列表 ──
        let mut node_texts: Vec<String> = Vec::with_capacity(word_count + char_count);
        let mut is_word: Vec<bool> = Vec::with_capacity(word_count + char_count);
        for w in &word_list { node_texts.push(w.clone()); is_word.push(true); }
        for c in &char_list { node_texts.push(c.clone()); is_word.push(false); }

        // ── Step 4: 双向索引 ──
        let mut char_to_words: Vec<Vec<usize>> = vec![Vec::new(); char_count];
        let mut word_to_chars: Vec<Vec<usize>> = vec![Vec::new(); word_count];
        let mut char_freq: Vec<usize> = vec![0; char_count];
        let mut word_len: Vec<usize> = vec![0; word_count];
        for wi in 0..word_count {
            let w = &word_list[wi];
            let chars: Vec<char> = w.chars().collect();
            word_len[wi] = chars.len();
            for &c in &chars {
                if let Some(&ci) = char_set.get(&c) {
                    char_to_words[ci].push(wi);
                    word_to_chars[wi].push(ci);
                    char_freq[ci] += 1;
                }
            }
        }

        // ── Step 5: 词间 containment ──
        let mut word_to_super_words: Vec<Vec<usize>> = vec![Vec::new(); word_count];
        if config.word_substring_containment {
            let mut sorted_indices: Vec<usize> = (0..word_count).collect();
            sorted_indices.sort_by_key(|&i| word_list[i].chars().count());
            for &short_i in &sorted_indices {
                let short = &word_list[short_i];
                let short_len = short.chars().count();
                if short_len < config.min_subordinate_len { continue; }
                for &long_i in &sorted_indices {
                    let long = &word_list[long_i];
                    if long.chars().count() <= short_len { continue; }
                    if long.contains(short.as_str()) {
                        word_to_super_words[short_i].push(long_i);
                    }
                }
            }
        }

        // ── Step 6: Jaccard 连接强度（泛模因 3.2.2）──
        let char_jaccard = Self::compute_char_jaccard(&char_to_words);
        let word_jaccard = Self::compute_word_jaccard(&word_to_chars);

        RelationNetwork {
            node_texts, is_word, word_count,
            char_to_words, word_to_chars, word_to_super_words,
            char_freq, word_len, char_jaccard, word_jaccard,
        }
    }

    /// 字—字 Jaccard: |↑(c₁) ∩ ↑(c₂)| / |↑(c₁) ∪ ↑(c₂)|
    fn compute_char_jaccard(char_to_words: &[Vec<usize>]) -> Vec<JaccardRow> {
        let n = char_to_words.len();
        let mut result: Vec<JaccardRow> = vec![Vec::new(); n];

        for ci in 0..n {
            let up_ci: HashMap<usize, bool> = char_to_words[ci].iter().map(|&w| (w, true)).collect();

            for cj in (ci + 1)..n {
                let up_cj = &char_to_words[cj];
                let mut inter = 0usize;
                let mut union = up_ci.len();
                for &w in up_cj {
                    if up_ci.contains_key(&w) {
                        inter += 1;
                    } else {
                        union += 1;
                    }
                }
                if inter > 0 {
                    let j = inter as f64 / union as f64;
                    result[ci].push((cj, j));
                    result[cj].push((ci, j));
                }
            }
        }
        result
    }

    /// 词—词 Jaccard: |↓(w₁) ∩ ↓(w₂)| / |↓(w₁) ∪ ↓(w₂)|
    fn compute_word_jaccard(word_to_chars: &[Vec<usize>]) -> Vec<JaccardRow> {
        let n = word_to_chars.len();
        let mut result: Vec<JaccardRow> = vec![Vec::new(); n];

        for wi in 0..n {
            let down_wi: std::collections::HashSet<usize> = word_to_chars[wi].iter().copied().collect();

            for wj in (wi + 1)..n {
                let down_wj = &word_to_chars[wj];
                let mut inter = 0usize;
                let mut union = down_wi.len();
                for &c in down_wj {
                    if down_wi.contains(&c) {
                        inter += 1;
                    } else {
                        union += 1;
                    }
                }
                if inter > 0 {
                    let j = inter as f64 / union as f64;
                    result[wi].push((wj, j));
                    result[wj].push((wi, j));
                }
            }
        }
        result
    }

    pub fn node_count(&self) -> usize { self.node_texts.len() }

    /// ↑ 操作：向上归类
    pub fn up(&self, idx: usize) -> Vec<usize> {
        if idx < self.word_count {
            self.word_to_super_words[idx].clone()
        } else {
            self.char_to_words[idx - self.word_count].clone()
        }
    }

    /// ↓ 操作：向下分解
    pub fn down(&self, idx: usize) -> Vec<usize> {
        if idx < self.word_count {
            let mut result: Vec<usize> = self.word_to_chars[idx]
                .iter().map(|&ci| ci + self.word_count).collect();
            for sub_i in 0..self.word_count {
                if self.word_to_super_words[sub_i].contains(&idx) {
                    result.push(sub_i);
                }
            }
            result
        } else {
            Vec::new()
        }
    }
}
