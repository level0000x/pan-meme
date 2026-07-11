#!/usr/bin/env python3
"""
tsv_bridge.py — pan_meme → ROSE-SCA WikiLine TSV 桥接器
=======================================================
将 pan_meme 管线的 PipelineData 转换为 WikiLine TSV 格式（9字段），
直接输出到 ROSE-SCA 的 data/wiki_tsv/records/ 目录。

数学对应：模块四输出 → WikiLine 知识表示
  给定物 = token.text（实体名）
  目光 = containment / connection / equivalence（关系）
  状态 = unheld（初始状态，等待目光循环处理）

WikiLine TSV 9字段:
  record_id  entity  predicate  target  source  state  conflicts  parent  timestamp

用法:
  from tsv_bridge import TsvBridge
  bridge = TsvBridge(output_dir='../sighted-wiki/data/wiki_tsv/records')
  bridge.write(data, source='wordnet')  # PipelineData → wordnet.tsv
"""

__version__ = "0.1.0"

import os
import time
import hashlib
from typing import Optional, Dict, List, Any

# TSV 格式常量
TSV_FIELDS = ['record_id', 'entity', 'predicate', 'target', 'source', 'state', 'conflicts', 'parent', 'timestamp']
TSV_HEADER = '\t'.join(TSV_FIELDS) + '\n'

# 谓词映射：pan_meme 边权区间 → ROSE 本原谓词
# containment: 归类关系（A 是 B 的一种）
# connection: 关联关系（A 与 B 相关）
# equivalence: 等价关系（A 与 B 等价）
# incidence: 附随关系（A 附着于 B）
PREDICATE_MAP = {
    'containment': 'containment',
    'connection': 'connection',
    'equivalence': 'equivalence',
    'incidence': 'incidence',
}


class TsvBridge:
    """
    pan_meme PipelineData → WikiLine TSV 转换器。

    从 pan_meme 管线的 MathModel.structure (RelationNetwork) 提取
    实体和关系，输出为 ROSE 兼容的 WikiLine TSV 格式。

    关系类型判定规则：
    - 边权 > 0.7 且层级差 > 0 → containment（归类）
    - 边权 > 0.7 且层级差 = 0 → equivalence（等价）
    - 边权 0.3-0.7 → connection（关联）
    - 边权 < 0.3 → 跳过（弱关系不摄入）
    """

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = os.environ.get(
                'SIGHTED_WIKI_DATA',
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'sighted-wiki', 'data', 'wiki_tsv', 'records')
            )
        self.output_dir = output_dir
        self._record_counter = 0
        self._seen = set()  # (entity, predicate, target) 去重

    def write(self, data: Any, source: str = 'pan_meme') -> str:
        """
        将 PipelineData 写入 WikiLine TSV 文件。

        参数:
          data: PipelineData（来自 pan_meme 管线）
          source: 数据来源标识（wordnet / conceptnet / wikipedia_zh / sumo / pan_meme）

        返回:
          输出文件路径
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f'{source}.tsv')

        # 提取 RelationNetwork
        psi = None
        if hasattr(data, 'math_model') and data.math_model is not None:
            psi = data.math_model.structure
        elif hasattr(data, 'psi') and data.psi is not None:
            psi = data.psi
        else:
            print(f"[tsv_bridge] 无 RelationNetwork，跳过 {source}")
            return filepath

        records = self._extract_records(psi, source)
        self._write_tsv(filepath, records)
        print(f"[tsv_bridge] {source}: {len(records)} 条 → {filepath}")
        return filepath

    def write_raw(self, nodes: List[str], edges: List[tuple], weights: List[float],
                  hierarchy: Optional[Dict] = None, source: str = 'pan_meme') -> str:
        """
        直接从原始数据写入 TSV（不依赖 PipelineData）。

        参数:
          nodes: 节点名列表
          edges: 边列表 [(i, j), ...]
          weights: 边权列表 [w, ...]
          hierarchy: 层级信息 {node_levels: {idx: level}, ...}
          source: 来源标识
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f'{source}.tsv')

        node_levels = {}
        if hierarchy and 'node_levels' in hierarchy:
            node_levels = hierarchy['node_levels']

        records = []
        for (i, j), w in zip(edges, weights):
            if w < 0.3:
                continue

            level_i = node_levels.get(i, 0)
            level_j = node_levels.get(j, 0)

            # 深度: 0=根(上位词), 越大越深(下位词)
            # containment: 下位词(深) 从属于 上位词(浅)
            # equivalence: 仅当权重极高且同级
            if w > 0.7:
                if level_i > level_j:
                    # i 更深 → i 是下位词，从属于 j（上位词）
                    predicate = 'containment'
                    entity, target = nodes[i], nodes[j]
                elif level_j > level_i:
                    # j 更深 → j 是下位词，从属于 i（上位词）
                    predicate = 'containment'
                    entity, target = nodes[j], nodes[i]
                elif nodes[i] == nodes[j]:
                    # 同级 + 文本完全相同 → 等价（极少情况）
                    predicate = 'equivalence'
                    entity, target = nodes[i], nodes[j]
                else:
                    # 同级 + 高权重 → 关联
                    predicate = 'connection'
                    entity, target = nodes[i], nodes[j]
            else:
                predicate = 'connection'
                entity, target = nodes[i], nodes[j]

            key = (entity, predicate, target)
            if key in self._seen:
                continue
            self._seen.add(key)

            parent = self._find_parent(i, node_levels, j, nodes)
            records.append({
                'record_id': self._next_id(),
                'entity': entity,
                'predicate': predicate,
                'target': target,
                'source': source,
                'state': 'unheld',
                'conflicts': '-',
                'parent': parent,
                'timestamp': self._now(),
            })

        self._write_tsv(filepath, records)
        print(f"[tsv_bridge] {source}: {len(records)} 条 → {filepath}")
        return filepath

    # ─── 内部方法 ───

    def _extract_records(self, psi, source: str) -> List[dict]:
        """从 RelationNetwork 提取 WikiLine 记录"""
        records = []
        node_levels = psi.hierarchy.get('node_levels', {})

        for (i, j), w in zip(psi.edges, psi.weights):
            w = float(w)
            if w < 0.3:
                continue

            level_i = node_levels.get(i, 0)
            level_j = node_levels.get(j, 0)

            if w > 0.7:
                if level_i > level_j:
                    predicate, entity, target = 'containment', psi.nodes[i], psi.nodes[j]
                elif level_j > level_i:
                    predicate, entity, target = 'containment', psi.nodes[j], psi.nodes[i]
                elif psi.nodes[i] == psi.nodes[j]:
                    predicate, entity, target = 'equivalence', psi.nodes[i], psi.nodes[j]
                else:
                    predicate, entity, target = 'connection', psi.nodes[i], psi.nodes[j]
            else:
                predicate, entity, target = 'connection', psi.nodes[i], psi.nodes[j]

            key = (entity, predicate, target)
            if key in self._seen:
                continue
            self._seen.add(key)

            parent = self._find_parent(i, node_levels, j, psi.nodes)
            records.append({
                'record_id': self._next_id(),
                'entity': entity,
                'predicate': predicate,
                'target': target,
                'source': source,
                'state': 'unheld',
                'conflicts': '-',
                'parent': parent,
                'timestamp': self._now(),
            })

        return records

    def _find_parent(self, idx: int, node_levels: dict, other_idx: int,
                     nodes: list) -> str:
        """找到 idx 节点的父节点（层级比它低一级的节点）"""
        level = node_levels.get(idx, 0)
        parent_level = level - 1
        if parent_level < 0:
            return '-'
        # 找最近的层级为 parent_level 的节点
        for i, lvl in sorted(node_levels.items(), key=lambda x: abs(x[0] - idx)):
            if lvl == parent_level and i != idx:
                return nodes[i]
        return '-'

    def _write_tsv(self, filepath: str, records: List[dict]):
        """写入 TSV 文件"""
        existed = os.path.exists(filepath)
        with open(filepath, 'a' if existed else 'w', encoding='utf-8') as f:
            if not existed:
                f.write(TSV_HEADER)
            for rec in records:
                values = [str(rec.get(f, '-')) for f in TSV_FIELDS]
                f.write('\t'.join(values) + '\n')

    def _next_id(self) -> str:
        self._record_counter += 1
        h = hashlib.sha256(str(self._record_counter).encode()).hexdigest()[:12]
        return f"pm_{self._record_counter:08d}_{h}"

    def _now(self) -> str:
        return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())


# ─── 便捷函数 ───

def pipeline_to_tsv(data: Any, source: str = 'pan_meme',
                    output_dir: str = '../sighted-wiki/data/wiki_tsv/records') -> str:
    """便捷函数：PipelineData → TSV"""
    bridge = TsvBridge(output_dir)
    return bridge.write(data, source)


def raw_to_tsv(nodes: list, edges: list, weights: list, source: str = 'pan_meme',
               output_dir: str = '../sighted-wiki/data/wiki_tsv/records') -> str:
    """便捷函数：原始数据 → TSV"""
    bridge = TsvBridge(output_dir)
    return bridge.write_raw(nodes, edges, weights, source=source)