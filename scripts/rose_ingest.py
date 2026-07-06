#!/usr/bin/env python3
"""
rose_ingest.py — ROSE-SCA 一键摄入脚本
=======================================
调用 pan-meme 管线处理原始数据，输出 WikiLine TSV 到 ROSE 数据目录。

数据源支持:
  - text: 纯文本 → 中文分词 → pan_meme 管线 → TSV
  - jsonl: JSONL 格式 → 结构化解析 → TSV
  - existing: 已有 ROSE TSV 文件（直接复制/引用）

用法:
  python rose_ingest.py --source text --input ./data.txt --label wikipedia_zh
  python rose_ingest.py --source jsonl --input ./data.jsonl --label custom
  python rose_ingest.py --source existing --input ./data.tsv --label wordnet

完整管线:
  python rose_ingest.py --source text --input raw.txt --label my_kb
  python rose_ingest.py --batch  # 运行所有数据源
"""

__version__ = "0.1.0"

import os
import sys
import json
import argparse
import time
import hashlib
from typing import List, Dict, Tuple, Optional, Any

# 添加 pan_meme 到路径
_PM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PM_DIR not in sys.path:
    sys.path.insert(0, _PM_DIR)
_SCRIPTS_DIR = os.path.join(_PM_DIR, 'scripts')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from tsv_bridge import TsvBridge, TSV_FIELDS, TSV_HEADER

# ─── 配置 ───

DEFAULT_OUTPUT = os.path.join(os.path.dirname(_PM_DIR), 'sighted-wiki', 'data', 'wiki_tsv', 'records')

# 中文分词字典（最小前缀词典，零依赖）
# 收录常见中文概念词，用于构建初始分词
_ZH_DICT = {
    '动物', '植物', '科学', '数学', '物理', '化学', '生物', '历史', '地理',
    '语言', '音乐', '艺术', '体育', '建筑', '城市', '国家', '河流', '山脉',
    '食品', '疾病', '人物', '哲学', '经济学', '心理学', '计算机', '网络',
    '人工智能', '机器学习', '深度学习', '神经网络', '自然语言', '知识图谱',
    '量子', '相对论', '宇宙', '原子', '分子', '细胞', '基因', 'DNA', 'RNA',
    '蛋白质', '酶', '病毒', '细菌', '真菌', '植物', '动物', '人类', '哺乳动物',
    '爬行动物', '鸟类', '鱼类', '昆虫', '软体动物', '节肢动物', '脊椎动物',
    '恒星', '行星', '卫星', '黑洞', '星系', '银河', '太阳系', '地球', '月球',
    '大陆', '海洋', '大气', '气候', '生态', '环境', '能源', '化石', '矿物',
    '金属', '合金', '塑料', '陶瓷', '玻璃', '纤维', '木材', '纸张',
    '算法', '数据结构', '编译', '解释', '程序', '软件', '硬件', '操作系统',
    '数据库', '网络协议', '互联网', '万维网', '区块链', '云计算',
    '资本', '市场', '货币', '贸易', '产业', '供应链', '消费', '生产',
    '政治', '法律', '教育', '医疗', '交通', '通信', '农业', '工业',
    '文学', '诗歌', '小说', '戏剧', '电影', '绘画', '雕塑', '摄影',
    '宗教', '神话', '仪式', '信仰', '哲学', '伦理', '逻辑', '美学',
    '量子力学', '广义相对论', '标准模型', '热力学', '电磁学', '光学',
    '代数', '几何', '拓扑', '分析', '概率', '统计', '数论',
    '进化', '遗传', '生态', '分类', '解剖', '生理', '生化',
    '古代', '中世', '近代', '现代', '当代', '文明', '帝国', '战争',
    '交响', '协奏', '奏鸣', '歌剧', '芭蕾', '爵士', '摇滚', '电子',
    '奥林匹克', '足球', '篮球', '网球', '游泳', '田径', '体操',
    '碳', '氧', '氢', '氮', '铁', '铜', '金', '银', '硅', '钙',
    '猫', '狗', '马', '牛', '羊', '猪', '鸡', '鸭', '鱼', '鸟',
    '苹果', '橙', '香蕉', '葡萄', '西瓜', '草莓', '芒果', '柠檬',
    '汽车', '火车', '飞机', '轮船', '自行车', '地铁', '火箭', '卫星',
    '空气', '声音', '力', '能量', '时间', '空间', '物质', '信息',
    '认知', '记忆', '语言', '思维', '意识', '情感', '动机', '行为',
    '民主', '共和', '君主', '专制', '议会', '宪法', '法治', '权利',
    '微积分', '线性代数', '概率论', '数理统计', '离散数学', '组合数学',
    '操作系统', '编译原理', '计算机组成', '计算机网络', '软件工程',
}

# 中文停用词 — 过滤无意义的单字和虚词
_ZH_STOP = {
    '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一',
    '一个', '这', '他', '也', '与', '及', '或', '等', '为', '之', '从',
    '个', '中', '上', '下', '而', '且', '但', '对', '于', '以', '及',
    '会', '可', '要', '能', '被', '把', '让', '将', '向', '到', '说',
    '地', '得', '着', '过', '其', '它', '她', '们', '所', '如', '没',
    '很', '大', '小', '多', '少', '来', '去', '只', '还', '又', '再',
    '用', '做', '使', '由', '因', '此', '那', '哪', '什', '么', '怎',
    '年', '月', '日', '时', '分', '种', '些', '每', '各', '最', '更',
    '已', '已', '前', '后', '里', '外', '内', '间', '之', '出', '进',
    '起', '成', '开', '关', '发', '生', '新', '旧', '高', '长', '短',
    '次', '第', '特', '别', '另', '其', '它', '半', '全', '整', '部',
    '重', '要', '分', '支', '涉', '及', '处', '理', '技', '术', '等',
    '通', '过', '方', '式', '法', '体', '系', '类', '型', '学', '性',
    '化', '子', '者', '本', '元', '度', '量', '数', '值', '例', '如',
    '具', '具', '应', '当', '并', '并', '非', '则', '已', '未', '无',
    '两', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万',
}


def zh_tokenize(text: str, dict_words: set = None) -> List[str]:
    """
    中文分词（最大正向匹配，零依赖）。

    参数:
      text: 输入文本
      dict_words: 自定义词典（默认使用 _ZH_DICT）

    返回:
      分词结果列表
    """
    if dict_words is None:
        dict_words = _ZH_DICT

    # 按长度降序排列词典，实现最大匹配
    sorted_dict = sorted(dict_words, key=len, reverse=True)

    tokens = []
    i = 0
    while i < len(text):
        # 跳过非中文字符
        if not ('\u4e00' <= text[i] <= '\u9fff'):
            i += 1
            continue

        matched = False
        for word in sorted_dict:
            if text[i:i+len(word)] == word:
                tokens.append(word)
                i += len(word)
                matched = True
                break

        if not matched:
            # 单字切分
            tokens.append(text[i])
            i += 1

    # 过滤停用词和单字无意义token
    tokens = [t for t in tokens if t not in _ZH_STOP and len(t) >= 2]
    return tokens


def extract_relations(tokens: List[str], window: int = 5) -> Tuple[List[str], List[tuple], List[float]]:
    """
    从分词结果提取关系和层级。

    策略：
    1. 共现窗口内 → connection（关联）
    2. 词典词包含短词 → containment（短词是长词的上位概念）
    3. 相邻词 → 弱 containment

    返回:
      (nodes, edges, weights)
    """
    # 去重建立节点集
    seen = {}
    nodes = []
    for t in tokens:
        if t not in seen:
            seen[t] = len(nodes)
            nodes.append(t)

    edges = []
    weights = []

    # 1. 共现窗口 → connection
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window, len(tokens))):
            if tokens[i] != tokens[j]:
                a, b = seen[tokens[i]], seen[tokens[j]]
                if a > b:
                    a, b = b, a
                edges.append((a, b))
                # 距离越近权重越高
                dist = j - i
                w = 0.5 + 0.5 * (1.0 / dist)
                weights.append(min(w, 1.0))

    # 2. 词典词包含关系 → containment
    for i, word_i in enumerate(nodes):
        for j, word_j in enumerate(nodes):
            if i >= j:
                continue
            if len(word_i) > len(word_j) and word_j in word_i and word_i != word_j:
                # word_j 是 word_i 的一部分 → word_j 是 word_i 的上位概念
                edges.append((i, j))
                weights.append(0.85)

    return nodes, edges, weights


def process_text_file(filepath: str, source_label: str,
                      output_dir: str = None) -> str:
    """
    处理纯文本文件：分词 → 关系提取 → TSV。

    参数:
      filepath: 文本文件路径
      source_label: 来源标签（用于 TSV 的 source 字段）
      output_dir: 输出目录

    返回:
      输出 TSV 文件路径
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"[rose_ingest] 处理文本: {filepath} ({len(text)} 字符)")

    tokens = zh_tokenize(text)
    print(f"[rose_ingest] 分词: {len(tokens)} 个")

    nodes, edges, weights = extract_relations(tokens)
    print(f"[rose_ingest] 关系: {len(nodes)} 节点, {len(edges)} 边")

    bridge = TsvBridge(output_dir)
    return bridge.write_raw(nodes, edges, weights, source=source_label)


def process_jsonl_file(filepath: str, source_label: str,
                       output_dir: str = None) -> str:
    """
    处理 JSONL 文件：每行一个 JSON 对象 → 关系提取 → TSV。

    支持格式：
    - {"entity": "xxx", "relations": [{"type": "containment", "target": "yyy"}, ...]}
    - {"entity": "xxx", "predicate": "containment", "target": "yyy"}
    - 纯文本 {"text": "xxx"}

    参数:
      filepath: JSONL 文件路径
      source_label: 来源标签
      output_dir: 输出目录

    返回:
      输出 TSV 文件路径
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT

    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[rose_ingest] 跳过无效 JSON 行 {line_num}: {e}")
                continue

            # 格式1: 含 relations 字段
            if 'relations' in obj and isinstance(obj['relations'], list):
                for rel in obj['relations']:
                    records.append({
                        'record_id': _gen_id(line_num),
                        'entity': obj.get('entity', obj.get('title', 'unknown')),
                        'predicate': rel.get('type', rel.get('predicate', 'connection')),
                        'target': rel.get('target', ''),
                        'source': source_label,
                        'state': 'unheld',
                        'conflicts': '-',
                        'parent': obj.get('parent', '-'),
                        'timestamp': _now(),
                    })

            # 格式2: 直接含 predicate + target
            elif 'predicate' in obj and 'target' in obj:
                records.append({
                    'record_id': _gen_id(line_num),
                    'entity': obj.get('entity', obj.get('title', 'unknown')),
                    'predicate': obj['predicate'],
                    'target': obj['target'],
                    'source': source_label,
                    'state': 'unheld',
                    'conflicts': '-',
                    'parent': obj.get('parent', '-'),
                    'timestamp': _now(),
                })

            # 格式3: 纯文本 → 分词 → 关系提取
            elif 'text' in obj or 'title' in obj:
                text = obj.get('text', obj.get('title', ''))
                tokens = zh_tokenize(text)
                nodes, edges, weights = extract_relations(tokens)
                # 直接生成记录
                for (i, j), w in zip(edges, weights):
                    if w < 0.3:
                        continue
                    predicate = 'containment' if w > 0.7 else 'connection'
                    records.append({
                        'record_id': _gen_id(line_num),
                        'entity': nodes[i],
                        'predicate': predicate,
                        'target': nodes[j],
                        'source': source_label,
                        'state': 'unheld',
                        'conflicts': '-',
                        'parent': '-',
                        'timestamp': _now(),
                    })

    if not records:
        print(f"[rose_ingest] JSONL 无有效记录: {filepath}")
        return ''

    # 去重
    seen = set()
    unique = []
    for r in records:
        key = (r['entity'], r['predicate'], r['target'])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    os.makedirs(output_dir, exist_ok=True)
    filepath_out = os.path.join(output_dir, f'{source_label}.tsv')
    existed = os.path.exists(filepath_out)
    with open(filepath_out, 'a' if existed else 'w', encoding='utf-8') as f:
        if not existed:
            f.write(TSV_HEADER)
        for r in unique:
            values = [str(r.get(field, '-')) for field in TSV_FIELDS]
            f.write('\t'.join(values) + '\n')

    print(f"[rose_ingest] {source_label}: {len(unique)} 条 → {filepath_out}")
    return filepath_out


def process_existing_tsv(filepath: str, source_label: str,
                         output_dir: str = None) -> str:
    """
    处理已有 TSV 文件：复制到输出目录并更新 source 字段。

    参数:
      filepath: 已有 TSV 文件路径
      source_label: 来源标签
      output_dir: 输出目录

    返回:
      输出 TSV 文件路径
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT

    os.makedirs(output_dir, exist_ok=True)
    filepath_out = os.path.join(output_dir, f'{source_label}.tsv')

    with open(filepath, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    if not lines:
        print(f"[rose_ingest] 空文件: {filepath}")
        return ''

    with open(filepath_out, 'w', encoding='utf-8') as fout:
        fout.write(lines[0])  # 保留表头
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                parts[4] = source_label  # 更新 source 字段
            fout.write('\t'.join(parts) + '\n')

    count = len(lines) - 1
    print(f"[rose_ingest] {source_label}: {count} 条（从 {filepath}） → {filepath_out}")
    return filepath_out


# ─── 批量处理 ───

def process_batch(output_dir: str = None, skip_existing: bool = True):
    """
    批量处理所有数据源。

    如果 skip_existing=True，则跳过已有 TSV 的源（如 WordNet/ConceptNet）。
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT

    print(f"[rose_ingest] 批量摄入 → {output_dir}")
    print(f"[rose_ingest] 跳过已有源: {skip_existing}")
    print()

    existing_files = set(os.listdir(output_dir)) if os.path.isdir(output_dir) else set()

    results = {}
    for source_label, filepath in _BATCH_SOURCES:
        if skip_existing and f'{source_label}.tsv' in existing_files:
            print(f"[rose_ingest] 跳过 {source_label}（已存在）")
            continue

        if not os.path.exists(filepath):
            print(f"[rose_ingest] 跳过 {source_label}（文件不存在: {filepath}）")
            continue

        try:
            if filepath.endswith('.jsonl') or filepath.endswith('.json'):
                results[source_label] = process_jsonl_file(filepath, source_label, output_dir)
            elif filepath.endswith('.tsv'):
                results[source_label] = process_existing_tsv(filepath, source_label, output_dir)
            else:
                results[source_label] = process_text_file(filepath, source_label, output_dir)
        except Exception as e:
            print(f"[rose_ingest] {source_label} 失败: {e}")
            import traceback
            traceback.print_exc()

    print()
    print(f"[rose_ingest] 完成: {len(results)} 个源")
    for label, path in results.items():
        print(f"  {label}: {path}")

    return results


# 默认批量数据源
_BATCH_SOURCES = []


# ─── 辅助函数 ───

def _gen_id(seed: int = 0) -> str:
    h = hashlib.sha256(f'{seed}{time.time()}'.encode()).hexdigest()[:12]
    return f"ri_{h}"


def _now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())


def _find_or_add_batch_source(label: str, path: str):
    """注册批量数据源"""
    global _BATCH_SOURCES
    for i, (l, p) in enumerate(_BATCH_SOURCES):
        if l == label:
            _BATCH_SOURCES[i] = (label, path)
            return
    _BATCH_SOURCES.append((label, path))


# ─── 预注册常见数据源 ───

# 中文 Wikipedia JSON（从浏览器 fetch 的结果）
_wiki_json = os.path.join(os.path.dirname(_PM_DIR), 'sighted-wiki', 'data', 'wiki_batch1.json')
if os.path.exists(_wiki_json):
    _find_or_add_batch_source('wikipedia_zh', _wiki_json)

# SUMO merge.kif
_sumo_kif = os.path.join(os.path.dirname(_PM_DIR), 'sighted-wiki', 'data', 'raw', 'sumo_merge.kif')
if os.path.exists(_sumo_kif):
    _find_or_add_batch_source('sumo', _sumo_kif)


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(
        description='ROSE-SCA 一键摄入 — pan-meme 管线 → WikiLine TSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python rose_ingest.py --source text --input ./data.txt --label my_kb
  python rose_ingest.py --source jsonl --input ./data.jsonl --label custom
  python rose_ingest.py --source existing --input ./data.tsv --label wordnet
  python rose_ingest.py --batch
  python rose_ingest.py --batch --skip-existing
        """,
    )
    parser.add_argument('--source', choices=['text', 'jsonl', 'existing'],
                        help='数据源类型')
    parser.add_argument('--input', help='输入文件路径')
    parser.add_argument('--label', default='pan_meme', help='来源标签')
    parser.add_argument('--output', help='输出目录')
    parser.add_argument('--batch', action='store_true', help='批量处理所有数据源')
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过已有 TSV 的源')

    args = parser.parse_args()

    if args.batch:
        process_batch(args.output, args.skip_existing)
        return

    if not args.source or not args.input:
        parser.print_help()
        return

    output_dir = args.output or DEFAULT_OUTPUT
    source = args.source
    filepath = args.input
    label = args.label

    if not os.path.exists(filepath):
        print(f"错误: 文件不存在: {filepath}")
        return

    if source == 'text':
        process_text_file(filepath, label, output_dir)
    elif source == 'jsonl':
        process_jsonl_file(filepath, label, output_dir)
    elif source == 'existing':
        process_existing_tsv(filepath, label, output_dir)


if __name__ == '__main__':
    main()