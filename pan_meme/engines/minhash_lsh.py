"""MinHash LSH 近似关系提取引擎
数学对应：PB_ARCHITECTURE.md 第3.1节 — 将关系提取从 O(N²) 降至 O(Nk)

核心思想：
  1. MinHash 签名：用 b*r 个哈希函数对每个 Token 的特征集合计算最小哈希值，
     得到紧凑签名向量
  2. LSH banding：将签名向量分成 b 个 band，每个 band 内 r 行哈希到同一个桶，
     只有同一桶内的 Token 对才作为候选对
  3. 滑动窗口共现：利用输入序列的局部性，在 O(N·w) 时间内提取共现对作为辅助信号
  4. 组合提取：合并 LSH 候选和窗口共现，对候选对精确计算 w_ij，按阈值过滤

错误边界：
  MinHash LSH 的 false negative rate = (1 - s^r)^b，其中 s = Jaccard 相似度
"""

from typing import List, Tuple, Dict, Set, Optional
import hashlib
import struct

import numpy as np

from pan_meme.core.types import Token, RelationNetwork


# ============================================================
# MinHashLSH — 基于 MinHash 的局部敏感哈希索引
# 数学对应：PB_ARCHITECTURE.md 第3.1节
# 算法复杂度：签名计算 O(N·b·r·|token_set|)，索引构建 O(N·b)
# ============================================================

class MinHashLSH:
    """MinHash 局部敏感哈希：用 b 个 band、每 band r 行的签名方案
    快速筛选 Jaccard 相似度高于阈值的 Token 对。

    数学对应：PB_ARCHITECTURE.md 第3.1节 — MinHash + LSH banding 技术
    错误边界：false negative rate = (1 - s^r)^b, s = Jaccard 相似度
    """

    def __init__(self, num_bands: int = 20, rows_per_band: int = 5,
                 hash_seed: int = 42) -> None:
        """初始化 LSH 参数。

        数学对应：
        - b = num_bands（band 数量）
        - r = rows_per_band（每个 band 内的行数）
        - 总哈希函数数 = b * r
        - 参数选择满足：给定目标 Jaccard 阈值 s0，选 b,r 使 (1 - s0^r)^b ≤ δ

        参数:
          num_bands: band 数量 b，默认 20
          rows_per_band: 每个 band 内的行数 r，默认 5
          hash_seed: 哈希函数族种子，默认 42
        """
        self.num_bands: int = num_bands          # b
        self.rows_per_band: int = rows_per_band  # r
        self.num_hashes: int = num_bands * rows_per_band  # b*r
        self.hash_seed: int = hash_seed

        # ── 预生成哈希函数族的系数 (a_i, c_i) ──
        # 数学对应：MinHash 使用 (a*x + c) mod p 形式的线性哈希族
        # p 取大质数 2^61 - 1（Mersenne 质数），保证碰撞概率极小
        self._prime: int = (1 << 61) - 1
        rng = np.random.RandomState(hash_seed)
        # a_i ∈ [1, prime-1], c_i ∈ [0, prime-1]
        self._hash_coeffs_a: np.ndarray = rng.randint(
            1, self._prime, size=self.num_hashes, dtype=np.int64
        )
        self._hash_coeffs_c: np.ndarray = rng.randint(
            0, self._prime, size=self.num_hashes, dtype=np.int64
        )

        # ── 内部状态：构建后的 band 桶 ──
        # band_idx → {bucket_hash → [token_indices]}
        self._band_buckets: Dict[int, Dict[int, List[int]]] = {}

    # ================================================================
    # 核心方法：计算单个 Token 特征集合的 MinHash 签名
    # 数学对应：MinHash 定义 — 签名分量 = argmin over x∈S of h_k(x)
    # ================================================================

    def compute_signature(self, token_set: List[str]) -> List[int]:
        """计算单个 Token 特征集合的 MinHash 签名。

        数学对应：
        - 对每个哈希函数 h_k，签名[k] = min_{x ∈ token_set} h_k(x)
        - 签名长度 = b * r（num_hashes），每个分量 ∈ [0, 2^61-2]

        算法：
        1. 对 token_set 中每个元素 x，转换为 int（用 SHA-256 截断到 64 位）
        2. 对每个 h_k(x) = (a_k * x + c_k) mod prime，取最小值
        3. 返回长度为 num_hashes 的签名列表

        参数:
          token_set: Token 的特征字符串集合（如分词后的 token 列表）

        返回:
          长度为 b*r 的 MinHash 签名列表
        """
        if not token_set:
            # 空集合：返回最大值（表示无交集）
            return [self._prime - 1] * self.num_hashes

        # 初始化签名为最大值
        signature: np.ndarray = np.full(self.num_hashes, self._prime - 1,
                                         dtype=np.int64)

        for elem in token_set:
            # ── 将字符串元素哈希为 64 位整数 ──
            elem_int: int = self._str_to_int64(elem)
            # ── 向量化计算所有 k 个哈希值 ──
            # h_k(elem) = (a_k * elem_int + c_k) mod prime
            hash_values: np.ndarray = (
                self._hash_coeffs_a * elem_int + self._hash_coeffs_c
            ) % self._prime
            # ── 元素级取最小值 ──
            signature = np.minimum(signature, hash_values)

        return signature.tolist()

    # ================================================================
    # 构建 band → Token 倒排索引
    # 数学对应：LSH banding — 将签名切成 b 段，每段 r 行哈希到桶
    # ================================================================

    def build_index(self, tokens: List[Token]) -> Dict[int, List[int]]:
        """构建 band → Token 倒排索引。

        数学对应：
        - 对每个 Token i 计算签名 s_i ∈ Z^{b*r}
        - 将 s_i 分成 b 个 band，每个 band 包含 r 个分量
        - 对每个 band j：bucket = hash(band_j(s_i))，将 i 加入 bucket
        - 同一 bucket 内的 Token 对是 Jaccard 相似的候选对

        参数:
          tokens: Token 列表

        返回:
          band_index: Dict[band_idx, List[token_idx]] —
                      全局扁平化的桶索引，仅用于调试
        """
        self._band_buckets.clear()
        global_band_index: Dict[int, List[int]] = {}

        for token_idx, token in enumerate(tokens):
            # ── 提取 Token 的特征集合 ──
            # 数学对应：用 text 分词作为特征集合；embedding 不可直接用于 MinHash
            features: List[str] = self._token_features(token)

            # ── 计算 MinHash 签名 ──
            signature: List[int] = self.compute_signature(features)

            # ── 将签名分成 b 个 band，每个 band 哈希到桶 ──
            for band_idx in range(self.num_bands):
                start: int = band_idx * self.rows_per_band
                end: int = start + self.rows_per_band
                band_values: Tuple[int, ...] = tuple(signature[start:end])

                # 对 band 做二次哈希得到桶号
                bucket_hash: int = self._hash_band(band_values)

                # 插入 band 桶
                if band_idx not in self._band_buckets:
                    self._band_buckets[band_idx] = {}
                if bucket_hash not in self._band_buckets[band_idx]:
                    self._band_buckets[band_idx][bucket_hash] = []
                self._band_buckets[band_idx][bucket_hash].append(token_idx)

                # 记录全局扁平索引（调试用）
                global_key: int = band_idx * (1 << 30) + (bucket_hash % (1 << 30))
                if global_key not in global_band_index:
                    global_band_index[global_key] = []
                global_band_index[global_key].append(token_idx)

        return global_band_index

    # ================================================================
    # 通过 LSH banding 检索候选 Token 对
    # 数学对应：LSH 查询 — 对每个桶内所有 Token 对生成候选
    # ================================================================

    def find_candidate_pairs(self, tokens: List[Token]) -> Set[Tuple[int, int]]:
        """通过 LSH banding 找到候选 Token 对。

        数学对应：
        - 对每个 band 的每个桶内，所有 Token 对都是候选
        - 候选对集合 C = {(i, j) | ∃ band, bucket 使得 i,j 在同一桶内}
        - 复杂度：O(∑ (|bucket| choose 2)) ≈ O(N·b·collision_factor)

        参数:
          tokens: Token 列表（与 build_index 相同的列表）

        返回:
          候选对集合，每个元素为 (i, j)，其中 i < j
        """
        # ── 先建索引 ──
        self.build_index(tokens)

        candidate_pairs: Set[Tuple[int, int]] = set()

        # ── 遍历所有 band 的所有桶 ──
        for band_idx, buckets in self._band_buckets.items():
            for bucket_hash, token_indices in buckets.items():
                n_in_bucket: int = len(token_indices)
                if n_in_bucket < 2:
                    continue
                # 桶内所有无序对
                for a in range(n_in_bucket):
                    for b in range(a + 1, n_in_bucket):
                        i: int = token_indices[a]
                        j: int = token_indices[b]
                        # 保证 i < j
                        if i > j:
                            i, j = j, i
                        candidate_pairs.add((i, j))

        return candidate_pairs

    # ================================================================
    # 内部方法：对单个 Band 做哈希
    # 数学对应：将 r 维签名段映射到单个桶号 ∈ Z
    # ================================================================

    def _hash_band(self, band_values: Tuple[int, ...]) -> int:
        """对单个 band 的签名分量做二次哈希，映射为桶号。

        数学对应：
        - band_hash(band) = SHA-256(struct.pack(band)) 取低 63 位 mod prime
        - 保证碰撞概率均匀且独立于原始 MinHash 哈希族

        参数:
          band_values: 长度为 r 的签名分量元组

        返回:
          桶哈希值（整数）
        """
        # 将 r 个 int64 打包为字节序列后取 SHA-256
        packed: bytes = struct.pack(f'<{len(band_values)}q', *band_values)
        digest: bytes = hashlib.sha256(packed).digest()
        # 取低 8 字节作为 int64，映射到 [0, prime-1]
        bucket: int = int.from_bytes(digest[:8], byteorder='little', signed=False)
        return bucket % self._prime

    # ================================================================
    # 内部方法：字符串转 64 位整数
    # ================================================================

    def _str_to_int64(self, s: str) -> int:
        """将字符串元素哈希为 64 位无符号整数。

        使用 SHA-256 截断到 64 位，保证分布均匀。
        """
        digest: bytes = hashlib.sha256(s.encode('utf-8')).digest()
        return int.from_bytes(digest[:8], byteorder='little', signed=False)

    # ================================================================
    # 内部方法：从 Token 提取特征集合
    # ================================================================

    def _token_features(self, token: Token) -> List[str]:
        """从 Token 中提取用于 MinHash 的特征集合。

        数学对应：特征集合 = text 分词后的唯一词元集合

        参数:
          token: 输入 Token 对象

        返回:
          特征字符串列表
        """
        # 用 text 字段分词（简单按空白分词，生产环境可替换为更精细的分词器）
        raw: str = token.text.strip()
        if not raw:
            return [token.pos] if token.pos else ["_EMPTY_"]
        # 按空白分割并去重
        words: List[str] = raw.split()
        # 加入词性作为额外特征信号
        if token.pos:
            words.append(f"POS:{token.pos}")
        # 加入 modality 作为特征
        if token.modality:
            words.append(f"MOD:{token.modality}")
        return list(set(words))


# ============================================================
# WindowCooccurrence — 滑动窗口共现提取
# 数学对应：PB_ARCHITECTURE.md 第3.1节 — 窗口共现作为辅助信号
# 算法复杂度：O(N·w)，其中 w = window_size
# ============================================================

class WindowCooccurrence:
    """滑动窗口共现提取器：利用输入序列的局部性提取共现对。

    数学对应：
    - 论文前提2：结构近邻是信息无损的最强信号
    - 窗口共现权重 = 1/(j - i)，距离越近权重越大
    """

    def __init__(self, window_size: int = 100) -> None:
        """初始化滑动窗口参数。

        数学对应：
        - w = window_size：在序列中，位置差 ≤ w 的所有 Token 对都考虑
        - 权重衰减：w_ij = 1/(|j - i|)，距离越近权重越大

        参数:
          window_size: 滑动窗口大小 w，默认 100
        """
        self.window_size: int = window_size

    def extract_pairs(self, tokens: List[Token]) -> Dict[Tuple[int, int], float]:
        """滑动窗口提取共现对。

        数学对应：
        - 对于位置 i 的 Token，考虑 j ∈ [i+1, min(i+w, N-1)]
        - w_ij^cooc = 1/(j - i)  （距离权重）
        - 返回字典 {(i, j): w_ij}

        算法复杂度：O(N·w)，N = len(tokens)

        参数:
          tokens: Token 序列列表

        返回:
          共现对及其权重，键为 (i, j) 且 i < j
        """
        n: int = len(tokens)
        cooc_pairs: Dict[Tuple[int, int], float] = {}

        for i in range(n - 1):
            # 窗口上限：不超过序列末尾
            max_j: int = min(i + self.window_size + 1, n)
            for j in range(i + 1, max_j):
                # 数学对应：权重 = 1/(j - i)
                distance: int = j - i
                weight: float = 1.0 / float(distance)
                cooc_pairs[(i, j)] = weight

        return cooc_pairs


# ============================================================
# LSHRelationExtractor — 组合 MinHashLSH + WindowCooccurrence
# 数学对应：PB_ARCHITECTURE.md 第3.1节完整流程
# ============================================================

class LSHRelationExtractor:
    """LSH 加速的关系网络提取器：组合 MinHash LSH 候选筛选与滑动窗口共现。

    数学对应：PB_ARCHITECTURE.md 第3.1节 — 三步流程：
      Step 1: 滑动窗口提取共现权重 w_ij^cooc
      Step 2: MinHash LSH 找到语义候选对
      Step 3: 对候选对精确计算 w_ij = max(w_ij^cooc, w_ij^sem)，阈值过滤

    复杂度：O(N·w + N·b·r·|f| + |C|)，其中 |C| << N² 为候选对数量
    """

    def __init__(self, num_bands: int = 20, rows_per_band: int = 5,
                 window_size: int = 100) -> None:
        """初始化组合提取器。

        参数:
          num_bands: MinHash band 数量 b，默认 20
          rows_per_band: 每个 band 的行数 r，默认 5
          window_size: 滑动窗口大小 w，默认 100
        """
        self.lsh: MinHashLSH = MinHashLSH(
            num_bands=num_bands, rows_per_band=rows_per_band
        )
        self.cooc: WindowCooccurrence = WindowCooccurrence(
            window_size=window_size
        )

    def extract(self, tokens: List[Token],
                threshold: float = 0.2) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """三步提取关系网络边集和权重。

        数学对应：PB_ARCHITECTURE.md 第3.1节完整算法流程

        Step 1 — 滑动窗口共现：
          pairs_cooc = WindowCooccurrence.extract_pairs(tokens)
          得到 w_ij^cooc = 1/(j-i)  for |j-i| ≤ w

        Step 2 — LSH 候选筛选：
          candidates = MinHashLSH.find_candidate_pairs(tokens)
          利用 MinHash 签名 + LSH banding 在 O(Nk) 时间内找到
          Jaccard 相似度高的候选对

        Step 3 — 精确计算与阈值过滤：
          对每个候选对 (i, j)：
            w_ij^sem = cosine_similarity(emb_i, emb_j)  if embeddings exist
            w_ij = max(w_ij^cooc, w_ij^sem)
            if w_ij ≥ threshold: 保留该边

        参数:
          tokens: Token 序列列表
          threshold: 边权重阈值 T ∈ [0, 1]，仅保留 w_ij ≥ T 的边

        返回:
          (edges, weights) — 边列表和对应的权重数组
        """
        n: int = len(tokens)

        # ── Step 1: 滑动窗口共现 ──
        # 数学对应：前提2 — 结构共现是 w_ij 的主信号源
        cooc_pairs: Dict[Tuple[int, int], float] = self.cooc.extract_pairs(tokens)

        # ── Step 2: LSH 候选筛选 ──
        # 数学对应：第3.1节 — MinHash LSH 降低候选对数量
        candidates: Set[Tuple[int, int]] = self.lsh.find_candidate_pairs(tokens)

        # ── 合并窗口共现对到候选集（窗口共现对也是重要的候选） ──
        for pair in cooc_pairs:
            candidates.add(pair)

        # ── Step 3: 精确计算 w_ij 并阈值过滤 ──
        # 数学对应：w_ij = max(共现强度, 语义相似度)
        edges: List[Tuple[int, int]] = []
        weights_list: List[float] = []

        for i, j in sorted(candidates):
            if i >= j or i >= n or j >= n:
                continue

            # ── 共现权重 ──
            w_cooc: float = cooc_pairs.get((i, j), 0.0)

            # ── 语义权重：embedding 余弦相似度 ──
            # 数学对应：前提2 — 语义相似度作为辅助信号
            w_sem: float = 0.0
            emb_i: Optional[np.ndarray] = tokens[i].embedding
            emb_j: Optional[np.ndarray] = tokens[j].embedding
            if emb_i is not None and emb_j is not None:
                # 余弦相似度 = dot(a,b) / (||a||·||b||)
                norm_i: float = float(np.linalg.norm(emb_i))
                norm_j: float = float(np.linalg.norm(emb_j))
                if norm_i > 1e-12 and norm_j > 1e-12:
                    w_sem = float(np.dot(emb_i, emb_j) / (norm_i * norm_j))

            # ── w_ij = max(共现, 语义) ──
            # 取最大值确保两种信号互补
            w_ij: float = max(w_cooc, w_sem)

            # ── 阈值过滤 ──
            if w_ij >= threshold:
                edges.append((i, j))
                weights_list.append(w_ij)

        weights: np.ndarray = np.array(weights_list, dtype=np.float32)
        return edges, weights

    def estimate_false_negative_rate(self,
                                      jaccard_similarity: float = 0.3) -> float:
        """估算 MinHash LSH 的假阴性率（漏检率）。

        数学对应：PB_ARCHITECTURE.md 第3.1节错误边界
        - FN rate = (1 - s^r)^b
        - s = Jaccard 相似度（实际两个 Token 特征集合的 Jaccard 值）
        - r = rows_per_band, b = num_bands

        含义：
        - 两个 Jaccard 相似度为 s 的集合，在 b 个 band 中至少有一个
          band 全部 r 行都匹配的概率 = 1 - (1 - s^r)^b
        - 假阴性率 = 它们在任何 band 都不匹配的概率 = (1 - s^r)^b

        示例：
        - b=20, r=5, s=0.5 → FN ≈ (1 - 0.5^5)^20 ≈ (0.96875)^20 ≈ 0.53
          即 Jaccard=0.5 的 Token 对有 ~53% 被漏检
        - b=20, r=5, s=0.8 → FN ≈ (1 - 0.8^5)^20 ≈ (0.67232)^20 ≈ 0.0004
          即 Jaccard=0.8 的 Token 对仅有 ~0.04% 被漏检

        参数:
          jaccard_similarity: 预期的 Jaccard 相似度 s ∈ [0, 1]

        返回:
          假阴性率 ∈ [0, 1]
        """
        s: float = max(0.0, min(1.0, jaccard_similarity))
        r: int = self.lsh.rows_per_band
        b: int = self.lsh.num_bands
        # 数学对应：FN = (1 - s^r)^b
        fn_rate: float = (1.0 - s ** r) ** b
        return fn_rate
