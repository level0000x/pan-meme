# 测试模块: MetricEncoder（定义3(b) — 边权 w → 度量向量 g）
# 数学对应: 定义3(b): g_i = w(e_i) — 每个 1-cell 长度等于连接强度
# 论文位置: 定义3, 定理3

import pytest
import numpy as np
from pan_meme.core.types import SimplicialComplex
from pan_meme.module2_geo.skeleton import SkeletonEncoder
from pan_meme.module2_geo.metric_encoder import MetricEncoder


class TestMetricEncoder:
    """定义3(b) 度量编码测试: 验证边权到几何长度的保权映射。"""

    def test_weights_preserved(self, relation_network_chain):
        """
        测试边权逐元素保持: 经 skeleton + metric 编码后，
        g 的每个分量等于对应边在 psi.weights 中的值。

        数学对应:
        - 定义3(b): g[j] = w(edge_map^{-1}(j)) — 一一定长映射
        - 度量编码器为恒等函数: g[new_idx] = psi.weights[old_idx]
        - 这是几何化的核心: 离散图边权以连续几何长度嵌入到胞腔复形中
        """
        psi = relation_network_chain  # 3 条边, 权重 [0.9, 0.8, 0.7]

        # Step 1: 骨架编码 — 获取 K 和 edge_map
        K, vertex_map, edge_map = SkeletonEncoder.encode(psi)

        # Step 2: 度量编码 — 获取度量向量 g
        g = MetricEncoder.encode(psi, K, edge_map)

        # 验证 g 的维度与 K 的边数一致
        assert len(g) == len(K.edges), (
            f"度量向量维度不匹配: |g|={len(g)}, |K.edges|={len(K.edges)}"
        )

        # 验证 g 的数据类型为 float32
        assert g.dtype == np.float32, (
            f"g 的 dtype 应为 float32, 实际 {g.dtype}"
        )

        # 逐元素验证: g[new_idx] == psi.weights[old_idx]
        for old_idx, new_idx in edge_map.items():
            assert 0 <= new_idx < len(g), (
                f"edge_map 索引越界: new_idx={new_idx}, len(g)={len(g)}"
            )
            assert 0 <= old_idx < len(psi.weights), (
                f"edge_map 索引越界: old_idx={old_idx}, len(psi.weights)={len(psi.weights)}"
            )
            assert np.isclose(float(g[new_idx]), float(psi.weights[old_idx]), rtol=1e-6), (
                f"边权不匹配: g[{new_idx}]={g[new_idx]}, "
                f"psi.weights[{old_idx}]={psi.weights[old_idx]}"
            )
