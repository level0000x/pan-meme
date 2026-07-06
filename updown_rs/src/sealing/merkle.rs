//! Merkle 树模块 — 分层哈希树 + 可验证性
//!
//! 数学对应：
//!   分层二叉树结构，每个内部节点 H = hash(H_left || H_right)
//!   叶子层 H_i = hash(item_i)，奇数层复制最后一个节点实现完美二叉树填充。
//!
//! 哈希原语：使用 std::hash::DefaultHasher（u64 输出）
//!   - SHA-256 等密码学哈希由 binding.rs 的 sha2 crate 处理
//!   - 本模块聚焦树结构与验证逻辑（与哈希后端解耦）
//!
//! 结构：
//!   MerkleTree  — 完美二叉树哈希树（layers[0]=叶子, layers[n-1]=根）
//!   MerkleProof — 索引 + 兄弟路径（支持离线独立验证）

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ═══════════════════════════════════════════════════════════════════════
// 哈希原语
// ═══════════════════════════════════════════════════════════════════════

/// 对任意可哈希项计算 u64 摘要。
#[inline]
fn hash_one<T: Hash>(item: &T) -> u64 {
    let mut h = DefaultHasher::new();
    item.hash(&mut h);
    h.finish()
}

/// 将一对子节点哈希合并为父节点哈希。
/// 顺序敏感：hash(left, right) != hash(right, left)。
#[inline]
fn hash_pair(left: u64, right: u64) -> u64 {
    let mut h = DefaultHasher::new();
    left.hash(&mut h);
    right.hash(&mut h);
    h.finish()
}

/// u64 → 16 字符小写 hex 字符串。
#[inline]
fn to_hex(v: u64) -> String {
    format!("{:016x}", v)
}

// ═══════════════════════════════════════════════════════════════════════
// MerkleProof — 可离线验证的包含证明
// ═══════════════════════════════════════════════════════════════════════

/// Merkle 包含证明：从叶子到根的兄弟路径。
///
/// 验证公式：
///   从 leaf_hash 出发，对每个 sibling 按位置顺序组合：
///     is_left_sibling=true  → hash(sibling, current)
///     is_left_sibling=false → hash(current, sibling)
///   最终结果应与 root 一致。
#[derive(Debug, Clone)]
pub struct MerkleProof {
    /// 叶子在原始列表中的索引
    pub index: usize,
    /// 叶子的哈希值
    pub leaf_hash: u64,
    /// 兄弟路径：(is_left_sibling, sibling_hash)
    ///   - is_left_sibling=true  → 兄弟在左侧，当前节点在右侧
    ///   - is_left_sibling=false → 兄弟在右侧，当前节点在左侧
    pub siblings: Vec<(bool, u64)>,
}

impl MerkleProof {
    /// 创建一个仅含索引和叶子哈希的空证明（不含兄弟路径）。
    pub fn new(index: usize, leaf_hash: u64) -> Self {
        MerkleProof {
            index,
            leaf_hash,
            siblings: Vec::new(),
        }
    }

    /// 证明路径的长度（= ceil(log2(n))）。
    pub fn depth(&self) -> usize {
        self.siblings.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MerkleTree — 完美二叉树哈希树
// ═══════════════════════════════════════════════════════════════════════

/// 完美二叉树 Merkle 树。
///
/// 内部存储：
///   layers[0]   — 叶子层（数据哈希）
///   layers[1]   — 第一层内部节点
///   ...
///   layers[n-1] — 根层（单元素）
///
/// 填充规则：每一层若节点数为奇数（且 >1），复制最后一个节点使其变为偶数。
/// 这保证了树始终是完美二叉树，简化证明路径的构建。
#[derive(Debug, Clone)]
pub struct MerkleTree {
    /// 树的所有层，layers[0] = 叶子，layers[last] = 根
    layers: Vec<Vec<u64>>,
    /// 原始叶子数量（不含填充节点）
    leaf_count: usize,
}

impl MerkleTree {
    // ── 构造 ──────────────────────────────────────────────────────────

    /// 从数据切片构建 Merkle 树。
    ///
    /// 算法：
    ///   1. 对每个 item 计算 hash_one → 叶子层
    ///   2. 若叶子数 >1 且为奇数，复制最后一个
    ///   3. 逐层向上：每 2 个节点合并为父节点 hash_pair
    ///   4. 每层若奇数则填充
    ///   5. 直至只剩 1 个节点（根）
    ///
    /// 复杂度：O(n) 时间，O(n) 空间。
    pub fn build<T: Hash>(items: &[T]) -> Self {
        if items.is_empty() {
            return MerkleTree {
                layers: Vec::new(),
                leaf_count: 0,
            };
        }

        let leaf_count = items.len();
        let leaves: Vec<u64> = items.iter().map(hash_one).collect();
        Self::build_from_hashes_inner(leaves, leaf_count)
    }

    /// 从已有的 u64 哈希切片直接构建 Merkle 树。
    ///
    /// 叶子层直接使用传入的哈希值，不再重复哈希。
    /// 适用于已有哈希指纹的场景（如 binding.rs 中的 MemeFingerprint）。
    pub fn build_from_hashes(hashes: &[u64]) -> Self {
        if hashes.is_empty() {
            return MerkleTree {
                layers: Vec::new(),
                leaf_count: 0,
            };
        }

        let leaf_count = hashes.len();
        let leaves = hashes.to_vec();
        Self::build_from_hashes_inner(leaves, leaf_count)
    }

    /// 内部构造：从叶子哈希开始向上构建树的所有层。
    fn build_from_hashes_inner(mut leaves: Vec<u64>, leaf_count: usize) -> Self {
        // 完美二叉树填充（叶子层）
        pad_to_even(&mut leaves);

        let mut layers = vec![leaves.clone()];
        let mut current = leaves;

        while current.len() > 1 {
            let mut next: Vec<u64> = current
                .chunks(2)
                .map(|pair| hash_pair(pair[0], pair[1]))
                .collect();
            pad_to_even(&mut next);
            layers.push(next.clone());
            current = next;
        }

        MerkleTree { layers, leaf_count }
    }

    // ── 查询 ──────────────────────────────────────────────────────────

    /// 返回 Merkle 根哈希。
    pub fn root(&self) -> Option<u64> {
        self.layers.last().and_then(|l| l.first().copied())
    }

    /// 返回 Merkle 根的 hex 表示（16 字符）。
    pub fn root_hex(&self) -> String {
        self.root().map(to_hex).unwrap_or_else(|| "0000000000000000".to_string())
    }

    /// 原始叶子数量（不含填充）。
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// 树的总层数（含叶子层）。
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// 获取指定层的哈希快照（用于外部比较）。
    /// layer_index=0 为叶子层，layer_index=depth()-1 为根层。
    pub fn layer(&self, layer_index: usize) -> Option<&[u64]> {
        self.layers.get(layer_index).map(|v| v.as_slice())
    }

    // ── 验证 ──────────────────────────────────────────────────────────

    /// 验证叶子索引 index 处的 item 哈希是否与树中一致。
    pub fn verify_leaf<T: Hash>(&self, index: usize, item: &T) -> bool {
        if index >= self.leaf_count || self.layers.is_empty() {
            return false;
        }
        hash_one(item) == self.layers[0][index]
    }

    /// 生成叶子索引 index 的包含证明（Merkle proof）。
    ///
    /// 返回从该叶子到根路径上所有兄弟节点的哈希及位置。
    pub fn get_proof(&self, index: usize) -> Option<MerkleProof> {
        if index >= self.leaf_count || self.layers.is_empty() {
            return None;
        }

        let leaf_hash = self.layers[0][index];
        let mut proof = MerkleProof::new(index, leaf_hash);
        let mut idx = index;

        // 叶子层(layer 0)不作为"兄弟"加入，从 layer 0 向 layer 1 开始
        for layer_idx in 0..self.layers.len() - 1 {
            let layer = &self.layers[layer_idx];
            let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };

            if sibling_idx < layer.len() {
                // idx 为偶 → 当前在左，兄弟在右 → is_left_sibling = false
                // idx 为奇 → 当前在右，兄弟在左 → is_left_sibling = true
                let is_left = idx % 2 != 0;
                proof.siblings.push((is_left, layer[sibling_idx]));
            }

            idx /= 2; // 向上一层
        }

        Some(proof)
    }

    /// 使用 Merkle 证明离线验证一个叶子是否属于根为 root_hash 的树。
    ///
    /// 不需要访问完整树，只需 root_hash + proof。
    /// 这是 Merkle 树的核心价值：轻量级可验证性。
    pub fn verify_proof(
        root_hash: u64,
        index: usize,
        item_hash: u64,
        proof: &MerkleProof,
    ) -> bool {
        if proof.index != index {
            return false;
        }
        if proof.leaf_hash != item_hash {
            return false;
        }

        let mut current = item_hash;

        for &(is_left, sibling) in &proof.siblings {
            current = if is_left {
                // 兄弟在左侧：hash(sibling, current)
                hash_pair(sibling, current)
            } else {
                // 兄弟在右侧：hash(current, sibling)
                hash_pair(current, sibling)
            };
        }

        current == root_hash
    }

    /// 验证第 layer_index 层的完整性。
    ///
    /// 检查该层每个父节点是否等于其两个子节点的 hash_pair：
    ///   layers[layer_index][i] == hash(layers[layer_index-1][2i], layers[layer_index-1][2i+1])
    ///
    /// layer_index=0（叶子层）：仅验证所有哈希值非零。
    pub fn verify_layer(&self, layer_index: usize) -> bool {
        if self.layers.is_empty() {
            return false;
        }
        if layer_index == 0 {
            return self.layers[0].iter().all(|&h| h != 0);
        }
        if layer_index >= self.layers.len() {
            return false;
        }

        let child_layer = &self.layers[layer_index - 1];
        let parent_layer = &self.layers[layer_index];

        for i in 0..parent_layer.len() {
            let left = child_layer[i * 2];
            let right = child_layer.get(i * 2 + 1).copied().unwrap_or(left);
            if hash_pair(left, right) != parent_layer[i] {
                return false;
            }
        }

        true
    }

    /// 定位第一个被篡改的叶子索引。
    ///
    /// 算法：自顶向下二分搜索（O(log n)）。
    ///   1. 比较两棵树的根 → 相同则无篡改
    ///   2. 从根层向下，逐层比较子节点
    ///   3. 首个哈希不同的分支继续深入
    ///   4. 到达叶子层后返回索引
    ///
    /// 若两树叶子数不同，差异定位到最小共通长度处。
    pub fn locate_tamper(old_tree: &MerkleTree, new_tree: &MerkleTree) -> Option<usize> {
        let old_root = old_tree.root()?;
        let new_root = new_tree.root()?;

        if old_root == new_root {
            return None; // 未被篡改
        }

        let old_layers = &old_tree.layers;
        let new_layers = &new_tree.layers;

        // 从根层向下逐层定位差异分支
        let mut node_idx: usize = 0; // 在当前层的节点索引

        for lvl in (1..old_layers.len().min(new_layers.len())).rev() {
            let left_idx = node_idx * 2;
            let right_idx = left_idx + 1;

            // 到达叶子层的上一层：直接比较两个叶子子节点
            if lvl == 1 {
                if left_idx < old_tree.leaf_count
                    && left_idx < new_tree.leaf_count
                    && old_layers[0][left_idx] != new_layers[0][left_idx]
                {
                    return Some(left_idx);
                }
                if right_idx < old_tree.leaf_count
                    && right_idx < new_tree.leaf_count
                    && old_layers[0][right_idx] != new_layers[0][right_idx]
                {
                    return Some(right_idx);
                }
                // 两个叶子相同但父节点不同：填充节点导致
                return Some(left_idx.min(old_tree.leaf_count.saturating_sub(1)));
            }

            let child_lvl = lvl - 1;
            let old_left = old_layers[child_lvl].get(left_idx)?;
            let new_left = new_layers[child_lvl].get(left_idx)?;

            if *old_left != *new_left {
                node_idx = left_idx;
            } else {
                node_idx = right_idx;
            }
        }

        Some(node_idx.min(old_tree.leaf_count.saturating_sub(1)))
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 内部辅助
// ═══════════════════════════════════════════════════════════════════════

/// 若向量长度 >1 且为奇数，复制最后一个元素（完美二叉树填充）。
fn pad_to_even(v: &mut Vec<u64>) {
    if v.len() > 1 && v.len() % 2 != 0 {
        let last = *v.last().unwrap();
        v.push(last);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let items: Vec<&str> = vec![];
        let tree = MerkleTree::build(&items);
        assert_eq!(tree.root(), None);
        assert_eq!(tree.leaf_count(), 0);
    }

    #[test]
    fn test_single_leaf() {
        let items = vec!["hello"];
        let tree = MerkleTree::build(&items);
        assert!(tree.root().is_some());
        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.depth(), 1); // 只有叶子层
    }

    #[test]
    fn test_two_leaves() {
        let items = vec!["a", "b"];
        let tree = MerkleTree::build(&items);
        assert_eq!(tree.leaf_count(), 2);
        assert!(tree.depth() >= 2);

        // 验证叶子
        assert!(tree.verify_leaf(0, &"a"));
        assert!(tree.verify_leaf(1, &"b"));
        assert!(!tree.verify_leaf(0, &"wrong"));
    }

    #[test]
    fn test_odd_leaf_padding() {
        let items = vec!["a", "b", "c"]; // 3 个 → 填充为 4
        let tree = MerkleTree::build(&items);
        assert_eq!(tree.leaf_count(), 3);

        // 填充后的叶子层应有 4 个节点（最后两个相同）
        assert_eq!(tree.layers[0].len(), 4);
        assert_eq!(tree.layers[0][2], tree.layers[0][3]); // 填充副本
    }

    #[test]
    fn test_proof_and_verify() {
        let items: Vec<String> = (0..8).map(|i| format!("item_{}", i)).collect();
        let tree = MerkleTree::build(&items);
        let root = tree.root().unwrap();

        for i in 0..8 {
            let proof = tree.get_proof(i).unwrap();
            let item_hash = hash_one(&items[i]);

            // 独立验证（无需访问树）
            assert!(
                MerkleTree::verify_proof(root, i, item_hash, &proof),
                "proof verification failed for index {}",
                i
            );
        }

        // 错误索引应失败
        let proof = tree.get_proof(0).unwrap();
        assert!(!MerkleTree::verify_proof(root, 1, hash_one(&items[0]), &proof));
    }

    #[test]
    fn test_verify_layer() {
        let items: Vec<u32> = (1..=8).collect();
        let tree = MerkleTree::build(&items);

        // 验证每一层
        for lvl in 0..tree.depth() {
            assert!(tree.verify_layer(lvl), "layer {} should be valid", lvl);
        }
    }

    #[test]
    fn test_locate_tamper_no_change() {
        let items = vec!["a", "b", "c", "d"];
        let old_tree = MerkleTree::build(&items);
        let new_tree = MerkleTree::build(&items);

        assert_eq!(MerkleTree::locate_tamper(&old_tree, &new_tree), None);
    }

    #[test]
    fn test_locate_tamper_single_leaf() {
        let old_items = vec!["a", "b", "c", "d"];
        let new_items = vec!["a", "X", "c", "d"]; // index 1 被篡改

        let old_tree = MerkleTree::build(&old_items);
        let new_tree = MerkleTree::build(&new_items);

        let tampered = MerkleTree::locate_tamper(&old_tree, &new_tree);
        assert_eq!(tampered, Some(1));
    }

    #[test]
    fn test_locate_tamper_last_leaf() {
        let old_items = vec!["a", "b", "c", "d"];
        let new_items = vec!["a", "b", "c", "X"]; // index 3 被篡改

        let old_tree = MerkleTree::build(&old_items);
        let new_tree = MerkleTree::build(&new_items);

        let tampered = MerkleTree::locate_tamper(&old_tree, &new_tree);
        assert_eq!(tampered, Some(3));
    }

    #[test]
    fn test_large_tree_consistency() {
        let items: Vec<u64> = (0..1000).collect();
        let tree = MerkleTree::build(&items);
        assert_eq!(tree.leaf_count(), 1000);
        assert!(tree.depth() >= 10); // ceil(log2(1000)) + 1

        // 全层验证
        for lvl in 0..tree.depth() {
            assert!(tree.verify_layer(lvl), "layer {} invalid", lvl);
        }
    }
}
