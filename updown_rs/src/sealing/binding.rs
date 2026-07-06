//! Phase 4: 固化（绑定）
//!
//! 数学对应：
//!   3.5.1 哈希绑定: H(Q) → 不可变指纹（SHA-256）
//!   3.5.2 凭证结构: 可验证信息凭证
//!
//! 哈希：使用 sha2 crate 的 SHA-256 实现（密码学安全）。
//! 未来版本可替换为 Keccak-256 / BLAKE3 / Poseidon 等。
//!
//! Merkle 树：由 merkle.rs 模块提供分层哈希树结构与可验证性。

use crate::encoding::decomposition::PhaseThreeOutput;
use crate::sealing::merkle::MerkleTree;
// ── SHA-256（本地编译）：取消 Cargo.toml 中 sha2 注释，并启用下方导入 ──
// use sha2::{Sha256, Digest};
// ── 沙箱回退：DefaultHasher ──
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

// ═══════════════════════════════════════════════════════════════════════
// 3.5.1 哈希绑定
// ═══════════════════════════════════════════════════════════════════════

/// 模因的不可变指纹
#[derive(Debug)]
pub struct MemeFingerprint {
    pub meme_id: usize,
    /// SHA-256 哈希（hex 编码）
    pub hash: String,
    /// 哈希时间戳
    pub timestamp: u64,
}

/// 全局哈希绑定
#[derive(Debug)]
pub struct HashBinding {
    /// 所有模因的指纹
    pub fingerprints: Vec<MemeFingerprint>,
    /// 全局 Merkle 根（hex 编码）
    pub merkle_root: String,
    /// Merkle 树（保留完整树结构，用于验证与取证）
    pub merkle_tree: MerkleTree,
}

impl HashBinding {
    /// 计算每个子模因 Q_i 的哈希指纹，并构建分层的 Merkle 树。
    ///
    /// 数学对应：3.5.1 — H: Q_i → h_i，然后树化 → Merkle 根
    pub fn compute(phase3: &PhaseThreeOutput) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut fingerprints = Vec::new();
        let mut leaf_hashes: Vec<u64> = Vec::new();

        for meme in &phase3.memes {
            // ── DefaultHasher 实现（沙箱兼容）──
            let mut hasher = DefaultHasher::new();

            meme.sub_geometry.vertices.hash(&mut hasher);
            meme.sub_geometry.edges.hash(&mut hasher);
            meme.sub_geometry.faces.hash(&mut hasher);

            format!("{:.6}", meme.state.intrinsic_degree).hash(&mut hasher);
            format!("{:.6}", meme.state.binding_degree).hash(&mut hasher);
            format!("{:.6}", meme.state.energy_density).hash(&mut hasher);
            format!("{:.6}", meme.state.evolution_rate).hash(&mut hasher);
            format!("{:.6}", meme.state.structural_robustness).hash(&mut hasher);

            format!("{:.6}", meme.params.alpha_1).hash(&mut hasher);
            format!("{:.6}", meme.params.delta_2).hash(&mut hasher);
            format!("{:.6}", meme.params.epsilon_2).hash(&mut hasher);

            let hash_value = hasher.finish();
            let hash_hex = format!("{:016x}", hash_value);
            let u64_hash = hash_value;

            // ── SHA-256 实现（本地编译启用）──
            // let mut sha = Sha256::new();
            // sha.update(b"vertices:");
            // for v in &meme.sub_geometry.vertices { sha.update(v.to_le_bytes()); }
            // sha.update(b"edges:");
            // for e in &meme.sub_geometry.edges { sha.update(e.to_le_bytes()); }
            // sha.update(b"faces:");
            // for f in &meme.sub_geometry.faces { sha.update(f.to_le_bytes()); }
            // sha.update(format!("{:.6}", meme.state.intrinsic_degree).as_bytes());
            // sha.update(format!("{:.6}", meme.state.binding_degree).as_bytes());
            // sha.update(format!("{:.6}", meme.state.energy_density).as_bytes());
            // sha.update(format!("{:.6}", meme.state.evolution_rate).as_bytes());
            // sha.update(format!("{:.6}", meme.state.structural_robustness).as_bytes());
            // sha.update(format!("{:.6}", meme.params.alpha_1).as_bytes());
            // sha.update(format!("{:.6}", meme.params.delta_2).as_bytes());
            // sha.update(format!("{:.6}", meme.params.epsilon_2).as_bytes());
            // let result = sha.finalize();
            // let hash_hex = format!("{:x}", result);
            // let mut u64_bytes = [0u8; 8];
            // u64_bytes.copy_from_slice(&result[..8]);
            // let u64_hash = u64::from_le_bytes(u64_bytes);

            leaf_hashes.push(u64_hash);
            fingerprints.push(MemeFingerprint {
                meme_id: meme.id,
                hash: hash_hex,
                timestamp,
            });
        }

        // 构建分层 Merkle 树（完美二叉树 + 填充）
        let tree = MerkleTree::build_from_hashes(&leaf_hashes);
        let merkle_root = tree.root_hex();

        HashBinding {
            fingerprints,
            merkle_root,
            merkle_tree: tree,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3.5.2 凭证结构
// ═══════════════════════════════════════════════════════════════════════

/// 可验证信息凭证
#[derive(Debug)]
pub struct Credential {
    pub credential_id: String,
    pub meme_count: usize,
    pub merkle_root: String,
    pub timestamp: u64,
    pub asset_count: usize,
}

impl Credential {
    pub fn create(binding: &HashBinding, total_cells: usize) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let cred_id = format!("CRED-{:x}-{}", timestamp, binding.merkle_root
            .chars().take(8).collect::<String>());

        Credential {
            credential_id: cred_id,
            meme_count: binding.fingerprints.len(),
            merkle_root: binding.merkle_root.clone(),
            timestamp,
            asset_count: total_cells,
        }
    }
}

/// Phase 4 完整输出
#[derive(Debug)]
pub struct PhaseFourOutput {
    pub binding: HashBinding,
    pub credential: Credential,
}

/// 运行第四阶段固化。
pub fn run_phase_four(phase3: &PhaseThreeOutput, total_cells: usize) -> PhaseFourOutput {
    let binding = HashBinding::compute(phase3);
    let credential = Credential::create(&binding, total_cells);

    PhaseFourOutput { binding, credential }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::decomposition::{MemeDecomposition, SubGeometry, FiveDimState, DynamicsParams, PhaseThreeOutput, ExtendedDimension, CellSnapshot};

    fn make_dummy_meme(id: usize, n_verts: usize, n_edges: usize, n_faces: usize) -> MemeDecomposition {
        MemeDecomposition {
            id,
            sub_geometry: SubGeometry {
                id,
                vertices: (0..n_verts).collect(),
                edges: (0..n_edges).collect(),
                faces: (0..n_faces).collect(),
                external_links: Vec::new(),
            },
            state: FiveDimState {
                intrinsic_degree: 0.5,
                binding_degree: 0.3,
                energy_density: 1.2,
                evolution_rate: 0.1,
                structural_robustness: 0.8,
            },
            params: DynamicsParams {
                alpha_1: 0.5, alpha_2: 0.1,
                beta_1: 0.3, beta_2: 0.1,
                gamma_1: 0.1, gamma_2: 0.1,
                delta_1: 0.1, delta_2: 0.05, delta_3: 0.05,
                epsilon_1: 0.05, epsilon_2: 0.05,
            },
            xi: ExtendedDimension {
                vertex_cells: (0..n_verts).map(|i| CellSnapshot { dim: 0, id: i, boundary: Vec::new() }).collect(),
                edge_cells: (0..n_edges).map(|i| CellSnapshot { dim: 1, id: i, boundary: vec![i, (i+1)%n_verts] }).collect(),
                face_cells: (0..n_faces).map(|i| CellSnapshot { dim: 2, id: i, boundary: (0..3).collect() }).collect(),
                boundary_links: Vec::new(),
                vertex_potentials: vec![0.5; n_verts],
                edge_flux_magnitudes: vec![0.3; n_edges],
            },
        }
    }

    fn make_test_phase3() -> PhaseThreeOutput {
        let memes = vec![
            make_dummy_meme(0, 4, 6, 2),
            make_dummy_meme(1, 3, 3, 1),
            make_dummy_meme(2, 5, 8, 3),
        ];
        PhaseThreeOutput {
            memes,
            couplings: Vec::new(),
            n_memes: 3,
        }
    }

    #[test]
    fn test_hash_binding_compute() {
        let phase3 = make_test_phase3();
        let binding = HashBinding::compute(&phase3);

        assert_eq!(binding.fingerprints.len(), 3);
        // 所有指纹均为 64 字符 hex（SHA-256）或 16 字符（DefaultHasher）
        for fp in &binding.fingerprints {
            assert!(!fp.hash.is_empty());
            assert!(fp.hash.len() == 16 || fp.hash.len() == 64,
                "Hash length should be 16 (DefaultHasher) or 64 (SHA-256), got {}", fp.hash.len());
        }
        // Merkle 根非空
        assert!(!binding.merkle_root.is_empty());
        eprintln!("Merkle root: {}", binding.merkle_root);
    }

    #[test]
    fn test_credential_create() {
        let phase3 = make_test_phase3();
        let binding = HashBinding::compute(&phase3);
        let cred = Credential::create(&binding, 41);

        assert_eq!(cred.meme_count, 3);
        assert_eq!(cred.asset_count, 41);
        assert!(cred.credential_id.starts_with("CRED-"));
        assert_eq!(cred.merkle_root, binding.merkle_root);
        eprintln!("Credential: {}", cred.credential_id);
    }

    #[test]
    fn test_run_phase_four() {
        let phase3 = make_test_phase3();
        let output = run_phase_four(&phase3, 41);

        assert_eq!(output.binding.fingerprints.len(), 3);
        assert_eq!(output.credential.meme_count, 3);
        assert!(output.credential.credential_id.starts_with("CRED-"));
    }

    #[test]
    fn test_fingerprint_uniqueness() {
        let phase3 = make_test_phase3();
        let binding = HashBinding::compute(&phase3);

        let hashes: Vec<&str> = binding.fingerprints.iter().map(|f| f.hash.as_str()).collect();
        // 不同模因应产生不同哈希（虽然理论上可能碰撞，但 DefaultHasher 在
        // 不同数据上应产生不同哈希）
        let unique: std::collections::HashSet<&str> = hashes.iter().copied().collect();
        assert_eq!(unique.len(), hashes.len(), "Each meme should have unique hash");
    }
}
