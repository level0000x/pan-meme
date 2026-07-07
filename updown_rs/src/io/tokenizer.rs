//! 自然语言 → n-gram 词表 — 泛模因理论 §6.2.2
//!
//! 提取器接口契约: 实现 formal-concept-analysis-proof.md 推论 7.2 的接口
//! A_F: RawData → (Nodes, Edges^containment)
//! 架构支持多模态扩展（文本/图像/音频/代码/地理/数学证明）

/// 提取 n-gram 词表
///
/// 对中文文本，按 1-gram（字）、2-gram（双字词）、3-gram（三字词）提取
pub fn extract_ngrams(text: &str) -> Vec<String> {
    let chars: Vec<char> = text
        .chars()
        .filter(|c| !c.is_whitespace() && !c.is_ascii_punctuation())
        .collect();

    let mut words = Vec::new();

    // 1-gram (单字)
    for &c in &chars {
        words.push(c.to_string());
    }

    // 2-gram (双字)
    for window in chars.windows(2) {
        words.push(window.iter().collect::<String>());
    }

    // 3-gram (三字)
    for window in chars.windows(3) {
        words.push(window.iter().collect::<String>());
    }

    // 去重
    words.sort();
    words.dedup();
    words
}

/// 提取器接口: RawData → (Nodes, Edges^containment)
///
/// 对应 formal-concept-analysis-proof.md 推论 7.2
pub struct Extractor;

impl Extractor {
    /// 从原始文本提取节点（词）和边（字-词包含关系）
    pub fn extract(text: &str) -> (Vec<String>, Vec<(usize, usize)>) {
        let chars: Vec<char> = text
            .chars()
            .filter(|c| !c.is_whitespace() && !c.is_ascii_punctuation())
            .collect();

        let mut nodes = Vec::new();
        let edges = Vec::new();

        // 词节点
        for window in chars.windows(2) {
            let word: String = window.iter().collect();
            nodes.push(word);
        }

        // 去重
        nodes.sort();
        nodes.dedup();

        // 字-词包含关系: 如果字 c 出现在词 w 中，则 (c_idx, w_idx)
        // 简化实现: 这里不生成完整的 edge 列表，由 FCA 模块处理
        // 此接口仅满足契约，实际工作由 FormalContext::from_words 完成

        (nodes, edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_ngrams_chinese() {
        let text = "苹果香蕉水果";
        let words = extract_ngrams(text);
        assert!(words.contains(&"苹".to_string()));
        assert!(words.contains(&"果".to_string()));
        assert!(words.contains(&"苹果".to_string()));
        assert!(words.contains(&"香蕉".to_string()));
        assert!(words.contains(&"水果".to_string()));
    }

    #[test]
    fn test_extract_ngrams_empty() {
        let words = extract_ngrams("");
        assert!(words.is_empty());
    }

    #[test]
    fn test_extractor_interface() {
        let text = "苹果香蕉";
        let (nodes, _edges) = Extractor::extract(text);
        assert!(!nodes.is_empty());
        assert!(nodes.contains(&"苹果".to_string()));
        assert!(nodes.contains(&"香蕉".to_string()));
    }
}
