"""数学对应: 公理1 — I ∈ U 的原子化, 中文文本模态

TextZhTokenizer 是中文文本模态的 tokenizer 实现.
使用 jieba 分词器进行中文词语切分, 同时利用 jieba.posseg
进行词性标注 (POS tagging). 将中文文本字符串 I 转化为
Token 序列, 每个 Token 对应一个中文词语.

技术依赖:
  - jieba (>=0.42):  结巴中文分词, 支持精确模式、全模式、搜索引擎模式
  - jieba.posseg:    词性标注模块, 提供北大词性标注集
  - 可选 jieba 启用 paddle 模式 (需安装 paddlepaddle-tiny),
    默认使用 jieba 内置词典

数学映射:
  输入:  I = "信息是现实的基本维度" (str ∈ U)
  输出:  [[信息, n, (0,2)], [是, v, (2,3)], [现实, n, (3,5)],
          [的, uj, (5,6)], [基本, a, (6,8)], [维度, n, (8,10)]]
  每个 Token 携带: modality="text", span=(start, end), pos=词性标签
"""

from typing import Any, List

try:
    import jieba
    import jieba.posseg as pseg
    _JIEBA_AVAILABLE = True
except ImportError:
    jieba = None
    pseg = None
    _JIEBA_AVAILABLE = False

from pan_meme.core.types import Token
from pan_meme.plugins.modalities.base import BaseModalityTokenizer
from pan_meme.plugins.registry import PluginRegistry


class TextZhTokenizer(BaseModalityTokenizer):
    """
    中文文本模态 tokenizer — 基于 jieba 分词 + 词性标注.

    数学对应: 公理1 — 将文本 I ∈ U (中文) 原子化为 Token 序列.
    每个 Token 携带:
      - modality:  "text" — 文本模态标识
      - text:      词语原文 (如 "信息", "维度")
      - span:      (start, end) — 在原字符串中的字符偏移
      - pos:       北大词性标注集标签 (如 "n"=名词, "v"=动词, "a"=形容词)
      - payload:   {} — 文本模态无额外特殊数据

    分词策略:
      - 使用 jieba.cut 进行精确模式分词, 获取词语文本
      - 同时使用 jieba.posseg.cut 进行词性标注, 获取词性标签
      - span 通过逐步累加词语长度计算, 确保不重叠且连续
    """

    # ----------------------------------------------------------------
    # 模态标识 (抽象方法实现)
    # ----------------------------------------------------------------

    @property
    def modality(self) -> str:
        """
        返回本 tokenizer 处理的模态名称.

        数学对应: Token.modality — 信息论域 U 的元素分类标签 "text".

        Returns:
            固定返回 "text" (文本模态).
        """
        return "text"

    # ----------------------------------------------------------------
    # 模态判定 (抽象方法实现)
    # ----------------------------------------------------------------

    def can_handle(self, raw_input: Any) -> bool:
        """
        判断是否能处理该输入 — 仅接受 Python 字符串.

        数学对应: 公理1 — 模态判定函数 M(I), 检测 I 是否为文本类型.
        中文文本模态的 can_handle 条件: isinstance(x, str).
        与 TextEnTokenizer 共享相同的判定条件, 由调度器按注册顺序择一.

        Args:
            raw_input:  待检测的原始输入.

        Returns:
            True 当输入为 str 类型时, 否则 False.
        """
        return isinstance(raw_input, str)

    # ----------------------------------------------------------------
    # 分词实现 (抽象方法实现)
    # ----------------------------------------------------------------

    def tokenize(self, raw_input: Any) -> List[Token]:
        """
        对中文文本执行分词 + 词性标注, 生成 Token 序列.

        数学对应: 公理1 — 映射 τ: I → U, 将中文文本 I 原子化.
        处理流程:
          1. 使用 jieba.posseg.cut() 同时获取分词和词性
          2. 遍历每个 (word, flag) 对, 计算 span 偏移
          3. 构造 Token 对象并收集到列表中

        词性标注集:
          采用 jieba 默认的北大词性标注集, 常见标签包括:
            n  → 名词        v  → 动词        a  → 形容词
            d  → 副词        uj → 助词 (的)    m  → 数词
            p  → 介词        c  → 连词        r  → 代词

        Args:
            raw_input:  中文文本字符串 (str).

        Returns:
            Token 列表, 每个中文词语对应一个 Token.
            对于空字符串输入, 返回空列表.

        Raises:
            TypeError: 当 raw_input 不是字符串时 (由 can_handle 预先拦截,
                       正常情况下不会触发).
        """
        # 输入校验: can_handle 已保证类型正确, 此处为防御性检查
        if not isinstance(raw_input, str):
            raise TypeError(
                f"TextZhTokenizer 仅接受 str 类型输入, "
                f"收到 {type(raw_input).__name__}"
            )

        # 空字符串快速返回
        if not raw_input.strip():
            return []

        if not _JIEBA_AVAILABLE:
            # jieba 未安装时降级为逐字切分
            tokens: List[Token] = []
            for i, ch in enumerate(raw_input):
                if ch.strip():
                    tokens.append(Token(
                        text=ch,
                        modality="text",
                        span=(i, i + 1),
                        pos="x",
                    ))
            return tokens

        # ------------------------------------------------------------
        # Step 1: 使用 jieba.posseg 进行分词 + 词性标注
        # ------------------------------------------------------------
        # jieba.posseg.cut 返回生成器, 每个元素为 pair(word, flag)
        pairs = list(pseg.cut(raw_input))

        # ------------------------------------------------------------
        # Step 2: 构建 Token 序列, 计算 span
        # ------------------------------------------------------------
        tokens: List[Token] = []
        current_offset: int = 0  # 当前字符偏移指针

        for word, flag in pairs:
            # 跳过空白字符 token (jieba 可能产出纯空白 token)
            if not word.strip():
                current_offset += len(word)
                continue

            word_len: int = len(word)        # 词语字符数
            end_offset: int = current_offset + word_len

            # 构造 Token: 模态=text, 文本=原词, span=字符偏移,
            # pos=词性标签, payload 预留给未来扩展
            token = Token(
                modality=self.modality,
                text=word,
                span=(current_offset, end_offset),
                pos=flag,
                payload={},
            )
            tokens.append(token)

            # 推进偏移指针
            current_offset = end_offset

        return tokens


# ================================================================
# 自注册: 将本 tokenizer 注册到全局 PluginRegistry
# 数学对应: 设计文档第9节 — 插件系统, 注册中心
# ================================================================
PluginRegistry.register_tokenizer(TextZhTokenizer().modality, TextZhTokenizer)
