#[derive(Debug, Clone)]
pub struct ReversibilityReport {
    pub recovery_rate: f64,
    pub entropy_input: f64,
    pub entropy_output: f64,
    pub entropy_conserved: bool,
    pub details: Vec<String>,
}

pub fn shannon_entropy(frequencies: &[f64]) -> f64 {
    let total: f64 = frequencies.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    frequencies
        .iter()
        .filter(|&&f| f > 0.0)
        .map(|&f| {
            let p = f / total;
            -p * p.log2()
        })
        .sum()
}

pub fn verify_roundtrip(
    original_words: &[String],
    recovered_words: &[String],
    depths: &[f64],
) -> ReversibilityReport {
    let mut details = Vec::new();

    let mut recovered = 0;
    for word in original_words {
        if recovered_words.contains(word) {
            recovered += 1;
        }
    }
    let recovery_rate = if original_words.is_empty() {
        1.0
    } else {
        recovered as f64 / original_words.len() as f64
    };

    details.push(format!(
        "Word recovery: {}/{} ({:.1}%)",
        recovered,
        original_words.len(),
        recovery_rate * 100.0,
    ));

    let depth_freq: Vec<f64> = depths.iter().map(|&d| d.max(0.0)).collect();
    let entropy_input = shannon_entropy(&depth_freq);

    let word_len_freq: Vec<f64> = recovered_words.iter().map(|w| w.len() as f64).collect();
    let entropy_output = shannon_entropy(&word_len_freq);

    let max_entropy = entropy_input.max(entropy_output);
    let min_entropy = entropy_input.min(entropy_output);
    let avg_entropy = (entropy_input + entropy_output) / 2.0;
    let relative_diff = if avg_entropy > 0.01 {
        (max_entropy - min_entropy) / avg_entropy
    } else {
        0.0
    };
    let entropy_conserved = relative_diff < 0.5;

    details.push(format!(
        "Shannon entropy: in={:.4}, out={:.4}, conserved={}",
        entropy_input, entropy_output, entropy_conserved,
    ));

    ReversibilityReport {
        recovery_rate,
        entropy_input,
        entropy_output,
        entropy_conserved,
        details,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_basic() {
        let words = vec!["abc".to_string(), "def".to_string(), "ghi".to_string()];
        let recovered = vec!["abc".to_string(), "def".to_string(), "ghi".to_string()];
        let depths = vec![1.0, 1.0, 1.0];
        let report = verify_roundtrip(&words, &recovered, &depths);
        assert!((report.recovery_rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_shannon_entropy() {
        let freq = vec![1.0, 1.0, 1.0, 1.0];
        let h = shannon_entropy(&freq);
        assert!((h - 2.0).abs() < 1e-10);
    }
}
