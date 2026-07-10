pub fn extract_ngrams(text: &str) -> Vec<String> {
    let chars: Vec<char> = text
        .chars()
        .filter(|c| !c.is_whitespace() && !c.is_ascii_punctuation())
        .collect();

    let mut words = Vec::new();

    for &c in &chars {
        words.push(c.to_string());
    }

    for window in chars.windows(2) {
        words.push(window.iter().collect::<String>());
    }

    for window in chars.windows(3) {
        words.push(window.iter().collect::<String>());
    }

    words.sort();
    words.dedup();
    words
}

pub struct Extractor;

impl Extractor {
    pub fn extract(text: &str) -> (Vec<String>, Vec<(usize, usize)>) {
        let chars: Vec<char> = text
            .chars()
            .filter(|c| !c.is_whitespace() && !c.is_ascii_punctuation())
            .collect();

        let mut nodes = Vec::new();
        let edges = Vec::new();

        for window in chars.windows(2) {
            let word: String = window.iter().collect();
            nodes.push(word);
        }

        nodes.sort();
        nodes.dedup();

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
