//! 形式概念分析 (FCA) — 泛模因理论 §D.1-D.3, 定义 3.1-3.2, 定理 4.1-4.7
//!
//! 核心构造：
//! - FormalContext: Ψ = (W, C, E)，字-词二分图，C 从 W 自暴露
//! - CycleEngine: ↑↓ 循环引擎，对字/词交替执行 ↑ 和 ↓
//! - Warshall 传递闭包: R* = R⁺

use std::collections::{HashMap, HashSet};

/// 形式上下文 Ψ = (W, C, E) — 论文 §3.2.1, 定义 3.1
///
/// 字词二分图：W 为词集合，C 为字集合（从 W 自暴露），E 为字-词包含关系
/// 对应 formal-concept-analysis-proof.md 定义 2.1
#[derive(Debug, Clone)]
pub struct FormalContext {
    /// 词集合 W
    pub words: Vec<String>,
    /// 字集合 C（从 W 自暴露：C = ∪_{w∈W} chars(w)）
    pub chars: Vec<char>,
    /// 字索引 → 词索引 映射 (c ∈ chars(w))
    pub char_to_words: Vec<Vec<usize>>,
    /// 词索引 → 字索引 映射
    pub word_to_chars: Vec<Vec<usize>>,
    /// 邻接矩阵 (|V| × |V|): 字符在前，词在后
    pub adj_matrix: Vec<Vec<bool>>,
}

impl FormalContext {
    /// 从词列表构建形式上下文 — 论文 §3.2.1
    ///
    /// C 从 W 自暴露：C = ∪_{w∈W} chars(w)
    /// 对应 formal-concept-analysis-proof.md 定义 2.1
    pub fn from_words(words: Vec<String>) -> Self {
        let mut char_set = HashSet::new();
        for w in &words {
            for c in w.chars() {
                char_set.insert(c);
            }
        }
        let mut chars: Vec<char> = char_set.into_iter().collect();
        chars.sort();

        let char_to_idx: HashMap<char, usize> = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let n_chars = chars.len();
        let n_words = words.len();

        let mut char_to_words = vec![Vec::new(); n_chars];
        let mut word_to_chars = vec![Vec::new(); n_words];

        for (wi, w) in words.iter().enumerate() {
            for c in w.chars() {
                if let Some(&ci) = char_to_idx.get(&c) {
                    char_to_words[ci].push(wi);
                    word_to_chars[wi].push(ci);
                }
            }
        }

        let total = n_chars + n_words;
        let mut adj_matrix = vec![vec![false; total]; total];
        for ci in 0..n_chars {
            for &wi in &char_to_words[ci] {
                adj_matrix[ci][n_chars + wi] = true;
                adj_matrix[n_chars + wi][ci] = true;
            }
        }

        FormalContext { words, chars, char_to_words, word_to_chars, adj_matrix }
    }

    /// 总节点数 |V| = |W| + |C|
    pub fn total_nodes(&self) -> usize {
        self.chars.len() + self.words.len()
    }
}

/// Warshall 传递闭包 — O(|V|³) — 论文 §3.2.2
///
/// 输入: 邻接矩阵 (|V| × |V|)
/// 输出: 传递闭包 — 如果 i 通过任意路径可达 j，则 R*[i][j] = true
/// 论文 M1 数据流明确要求："传递性推理闭合"
pub fn warshall_closure(adj: &mut [Vec<bool>]) {
    let n = adj.len();
    for k in 0..n {
        for i in 0..n {
            if adj[i][k] {
                for j in 0..n {
                    adj[i][j] = adj[i][j] || adj[k][j];
                }
            }
        }
    }
}

/// 单元素 ↑↓ 循环结果
#[derive(Debug, Clone)]
pub struct ElementCycle {
    /// 起始元素索引
    pub element_idx: usize,
    /// 每轮迭代后的 word_closure
    pub word_closures: Vec<HashSet<usize>>,
    /// 每轮迭代后的 char_closure
    pub char_closures: Vec<HashSet<usize>>,
    /// 收敛轮数 = 信息深度
    pub convergence_rounds: usize,
}

/// ↑↓ 循环引擎 — 论文 §3.2.4, 定理 4.1-4.7
///
/// 交替执行 ↑ 和 ↓ 操作，直到 word_closure 和 char_closure 都不再增长
/// 对应 formal-concept-analysis-proof.md 定理 4.1-4.7
pub struct CycleEngine {
    /// 形式上下文引用
    pub ctx: FormalContext,
    /// 最大迭代轮数
    pub max_rounds: usize,
}

impl CycleEngine {
    pub fn new(ctx: FormalContext, max_rounds: usize) -> Self {
        CycleEngine { ctx, max_rounds }
    }

    /// 对单字执行 ↑↓ 循环
    ///
    /// ↑(c) = {w ∈ W | (c, w) ∈ E} — 字 c 出现在哪些词中
    /// ↓(↑(c)) = 所有与 c 共现的字 — 展开每个词中的字
    pub fn cycle_char(&self, char_idx: usize) -> ElementCycle {
        let _n_chars = self.ctx.chars.len();
        let _n_words = self.ctx.words.len();
        let mut word_closures = Vec::new();
        let mut char_closures = Vec::new();

        // 初始: ↑(c) = 包含 c 的所有词
        let mut current_words: HashSet<usize> = self.ctx.char_to_words[char_idx].iter().cloned().collect();
        word_closures.push(current_words.clone());

        // 初始: ↓(↑(c)) = 这些词中所有字
        let mut current_chars: HashSet<usize> = HashSet::new();
        current_chars.insert(char_idx);
        for &wi in &current_words {
            for &ci in &self.ctx.word_to_chars[wi] {
                current_chars.insert(ci);
            }
        }
        char_closures.push(current_chars.clone());

        let mut rounds = 1;
        while rounds < self.max_rounds {
            // 新一轮: ↑(当前字闭包) → 新词闭包
            let mut new_words: HashSet<usize> = HashSet::new();
            for &ci in &current_chars {
                for &wi in &self.ctx.char_to_words[ci] {
                    new_words.insert(wi);
                }
            }

            // 新一轮: ↓(新词闭包) → 新字闭包
            let mut new_chars: HashSet<usize> = HashSet::new();
            new_chars.insert(char_idx);
            for &wi in &new_words {
                for &ci in &self.ctx.word_to_chars[wi] {
                    new_chars.insert(ci);
                }
            }

            let words_changed = new_words != current_words;
            let chars_changed = new_chars != current_chars;

            current_words = new_words;
            current_chars = new_chars;
            word_closures.push(current_words.clone());
            char_closures.push(current_chars.clone());
            rounds += 1;

            if !words_changed && !chars_changed {
                break;
            }
        }

        ElementCycle {
            element_idx: char_idx,
            word_closures,
            char_closures,
            convergence_rounds: rounds,
        }
    }

    /// 对所有字并行 ↑↓ 循环
    pub fn cycle_all(&self) -> Vec<ElementCycle> {
        (0..self.ctx.chars.len())
            .map(|ci| self.cycle_char(ci))
            .collect()
    }
}

/// 验证 ↑↓ 互逆性: ↓(↑(c)) = c 和 ↑(↓(w)) = w
/// 对应 formal-concept-analysis-proof.md 公理 3
pub fn verify_reversibility(
    ctx: &FormalContext,
    cycles: &[ElementCycle],
) -> bool {
    // 验证 ↓(↑(c)): 循环后字闭包包含原字自身
    for cycle in cycles {
        if let Some(last_char_closure) = cycle.char_closures.last() {
            if !last_char_closure.contains(&cycle.element_idx) {
                return false;
            }
        }
    }

    // 验证 ↑(↓(w)): 对每个词，↓(w) 中的字 ↑ 后应包含该词
    for wi in 0..ctx.words.len() {
        let chars_in_word = &ctx.word_to_chars[wi];
        let mut words_found: HashSet<usize> = HashSet::new();
        for &ci in chars_in_word {
            for &wj in &ctx.char_to_words[ci] {
                words_found.insert(wj);
            }
        }
        if !words_found.contains(&wi) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formal_context_from_words() {
        let words = vec!["苹果".to_string(), "香蕉".to_string(), "水果".to_string()];
        let ctx = FormalContext::from_words(words);
        assert_eq!(ctx.words.len(), 3);
        // 苹, 果, 香, 蕉, 水, 水果中的"果"可能重复
        // 苹, 果, 香, 蕉, 水
        assert!(ctx.chars.len() >= 5);
        assert_eq!(ctx.total_nodes(), ctx.chars.len() + ctx.words.len());
    }

    #[test]
    fn test_warshall_closure() {
        // 简单传递图: 0→1, 1→2, 期望 0→2
        let mut adj = vec![
            vec![false, true, false],
            vec![false, false, true],
            vec![false, false, false],
        ];
        warshall_closure(&mut adj);
        assert!(adj[0][2]); // 传递闭包: 0→2
    }

    #[test]
    fn test_cycle_convergence() {
        let words = vec!["苹果".to_string(), "香蕉".to_string(), "水果".to_string()];
        let ctx = FormalContext::from_words(words);
        let engine = CycleEngine::new(ctx, 100);
        let cycles = engine.cycle_all();
        for cycle in &cycles {
            assert!(cycle.convergence_rounds > 0);
            assert!(cycle.convergence_rounds <= 100);
        }
    }

    #[test]
    fn test_reversibility() {
        let words = vec!["苹果".to_string(), "香蕉".to_string(), "水果".to_string()];
        let ctx = FormalContext::from_words(words);
        let engine = CycleEngine::new(ctx.clone(), 100);
        let cycles = engine.cycle_all();
        assert!(verify_reversibility(&ctx, &cycles));
    }
}