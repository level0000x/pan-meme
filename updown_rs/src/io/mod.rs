//! 输入输出层 — 泛模因理论 §6.2.2, 推论 5.1
//!
//! - tokenizer: 自然语言 → n-gram 词表 (提取器接口契约, FCA 证明 推论 7.2)
//! - tsv: WikiLine TSV 输出
//! - reversibility: Φ⁻¹∘Φ 往返验证 + 互逆性验证

pub mod tokenizer;
pub mod tsv;
pub mod reversibility;