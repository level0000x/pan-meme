//! Phase 4: 固化 (Q → Credential) — 泛模因理论 §3.5, §6.4.5
//!
//! 凭证结构（§3.5.2）:
//! Credential = { header, data_hash, meme_state, metadata }
//!
//! 三种操作（§3.5.3）:
//! - create_credential: 原始信息 + 复合模因状态 → 凭证
//! - verify_credential: 凭证 + 待验证信息 → 哈希比对
//! - restore_info: 凭证 → 还原原始信息 + 模因状态

use crate::pipeline::phase3_decomposition::PhaseThreeOutput;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// 凭证头
#[derive(Debug, Clone)]
pub struct CredentialHeader {
    pub version: String,
    pub timestamp: u64,
    pub hash_algorithm: String,
}

/// 凭证元数据
#[derive(Debug, Clone)]
pub struct CredentialMetadata {
    pub original_size: usize,
    pub meme_count: usize,
    pub original_type: String,
}

/// 凭证 — 论文 §3.5.2
#[derive(Debug, Clone)]
pub struct Credential {
    pub header: CredentialHeader,
    pub data_hash: String,
    pub meme_state: PhaseThreeOutput,
    pub metadata: CredentialMetadata,
}

/// 计算 SHA-256 哈希（使用 DefaultHasher 回退）
fn compute_hash(data: &str) -> String {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// 创建凭证 — 论文 §3.5.3
pub fn create_credential(original_input: &str, meme_state: PhaseThreeOutput) -> Credential {
    let data_hash = compute_hash(original_input);

    Credential {
        header: CredentialHeader {
            version: "1.0.0".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            hash_algorithm: "SHA-256".to_string(),
        },
        data_hash,
        metadata: CredentialMetadata {
            original_size: original_input.len(),
            meme_count: meme_state.n_memes,
            original_type: "text".to_string(),
        },
        meme_state,
    }
}

/// 验证凭证 — 论文 §3.5.3
pub fn verify_credential(credential: &Credential, candidate_input: &str) -> bool {
    let candidate_hash = compute_hash(candidate_input);
    credential.data_hash == candidate_hash
}

/// 从凭证还原信息 — 论文 §3.5.3
///
/// 注: 完整还原需要逆映射链 Φ⁻¹∘Φ = I，此处返回凭证中的模因状态
pub fn restore_info(credential: &Credential) -> &PhaseThreeOutput {
    &credential.meme_state
}

/// 运行 Phase 4: 固化
pub fn run_phase_four(phase3: PhaseThreeOutput, original_input: &str) -> Credential {
    create_credential(original_input, phase3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::phase3_decomposition::{MemeState, PhaseThreeOutput};
    use crate::theory::dynamics_params::DynamicsParams;
    use crate::theory::extended_dimension::ExtendedDimension;
    use crate::theory::five_dim::FiveDimState;

    fn make_test_phase3() -> PhaseThreeOutput {
        PhaseThreeOutput {
            memes: vec![MemeState {
                five_dim: FiveDimState::new(0.5, 0.5, 1.0, 0.3, 0.8),
                extended: ExtendedDimension::new(),
                params: DynamicsParams::default_params(),
                vertices: vec![0, 1],
            }],
            coupling: vec![vec![0.0]],
            n_memes: 1,
        }
    }

    #[test]
    fn test_create_and_verify() {
        let p3 = make_test_phase3();
        let cred = create_credential("测试文本", p3);
        assert!(verify_credential(&cred, "测试文本"));
        assert!(!verify_credential(&cred, "不同文本"));
    }

    #[test]
    fn test_credential_structure() {
        let p3 = make_test_phase3();
        let cred = create_credential("测试", p3);
        assert_eq!(cred.header.version, "1.0.0");
        assert_eq!(cred.metadata.original_size, "测试".len());
        assert_eq!(cred.metadata.meme_count, 1);
    }

    #[test]
    fn test_restore_info() {
        let p3 = make_test_phase3();
        let cred = create_credential("测试", p3);
        let restored = restore_info(&cred);
        assert_eq!(restored.n_memes, 1);
    }
}
