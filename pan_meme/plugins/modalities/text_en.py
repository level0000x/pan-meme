"""数学对应: 公理1 — I ∈ U 的原子化, 英文文本模态

TextEnTokenizer 是英文文本模态的 tokenizer 实现.
使用 spaCy 的 en_core_web_sm 模型进行英文词语切分 (tokenization)
和词性标注 (POS tagging). 将英文文本字符串 I 转化为
Token 序列, 每个 Token 对应一个英文词语/标点符号.

技术依赖:
  - spacy (>=3.5):     工业级 NLP 库, 需额外安装 [all] 依赖组
  - en_core_web_sm:    英文小型预训练模型 (需 python -m spacy download en_core_web_sm)
  - 若模型未安装, 可在构造时自动尝试下载, 失败则抛出 OSError

数学映射:
  输入:  I = "Information is fundamental." (str ∈ U)
  输出:  [[Information, PROPN, (0,11)], [is, AUX, (12,14)],
          [fundamental, ADJ, (15,26)], [., PUNCT, (26,27)]]
  每个 Token 携带: modality="text", span=(start, end), pos=通用词性标签

与 TextZhTokenizer 的关系:
  两者均处理 str 类型输入, can_handle 均返回 isinstance(x, str).
  调度器 (Tokenizer 类) 按注册顺序遍历, 首个匹配者被选用.
  通常中文文档注册 TextZhTokenizer 在先, 英文在后.
"""

from typing import Any, List

try:
    import spacy
    _SPACY_AVAILABLE: bool = True
except ImportError:
    _SPACY_AVAILABLE = False
    spacy = None  # type: ignore[assignment]

from pan_meme.core.types import Token
from pan_meme.plugins.modalities.base import BaseModalityTokenizer
from pan_meme.plugins.registry import PluginRegistry


# ----------------------------------------------------------------
# 模块级常量
# ----------------------------------------------------------------

_MODEL_NAME: str = "en_core_web_sm"
"""spaCy 英文预训练模型名称 (小型)."""


class TextEnTokenizer(BaseModalityTokenizer):
    """
    英文文本模态 tokenizer — 基于 spaCy en_core_web_sm 的分词与词性标注.

    数学对应: 公理1 — 将文本 I ∈ U (英文) 原子化为 Token 序列.
    每个 Token 携带:
      - modality:  "text" — 文本模态标识
      - text:      词语原文 (如 "fundamental")
      - span:      (start, end) — 在原字符串中的字符偏移
      - pos:       spaCy 通用词性标签 (PROPN/NOUN/VERB/ADJ/AUX/PUNCT 等)
      - payload:   {} — 文本模态无额外特殊数据

    spaCy 模型信息:
      en_core_web_sm 包含:
        - tok2vec:  词向量 (96维, 小型)
        - tagger:   词性标注器
        - parser:   依存句法分析器
        - ner:      命名实体识别
      本 tokenizer 仅使用 tokenization + POS tagging 功能.
    """

    # ----------------------------------------------------------------
    # 构造与初始化
    # ----------------------------------------------------------------

    def __init__(self) -> None:
        """
        初始化 spaCy 英文 NLP 管道.

        延迟加载模型 en_core_web_sm. 如果模型尚未安装,
        尝试通过 spacy.cli.download 自动下载; 下载失败则抛出 OSError.

        Raises:
            ImportError: 当 spacy 库未安装时.
            OSError:    当 en_core_web_sm 模型加载/下载均失败时.
        """
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "TextEnTokenizer 需要 spacy 库. "
                "请执行: pip install pan-meme[all] 或 pip install spacy>=3.5"
            )

        # 尝试加载已安装的模型
        try:
            self._nlp = spacy.load(_MODEL_NAME)
        except OSError:
            # 模型未下载, 尝试自动下载
            try:
                import subprocess
                import sys
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", _MODEL_NAME],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._nlp = spacy.load(_MODEL_NAME)
            except Exception:
                raise OSError(
                    f"无法加载或下载 spaCy 模型 '{_MODEL_NAME}'. "
                    f"请手动执行: python -m spacy download {_MODEL_NAME}"
                )

        # 为提升分词性能, 仅保留 tokenizer 和 tagger 组件,
        # 禁用 parser 和 ner (不需要)
        # 注意: 通过 select_pipes 上下文管理器禁用非必要管道
        self._model_name: str = _MODEL_NAME

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
        英文文本模态的 can_handle 条件: isinstance(x, str).
        与 TextZhTokenizer 共享相同的判定条件, 由调度器按注册顺序择一.

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
        对英文文本执行分词 + 词性标注, 生成 Token 序列.

        数学对应: 公理1 — 映射 τ: I → U, 将英文文本 I 原子化.
        处理流程:
          1. 调用 self._nlp(text) 生成 spaCy Doc 对象
          2. 遍历 Doc 中的每个 Token, 提取 .text / .idx / .pos_
          3. 构造 pan_meme Token 对象并收集到列表中

        spaCy 通用词性标签集 (Universal Dependencies POS tags):
          PROPN → 专有名词    NOUN  → 普通名词    VERB → 动词
          ADJ   → 形容词      ADV   → 副词        AUX  → 助动词
          DET   → 限定词      ADP   → 介词        CCONJ→ 并列连词
          PUNCT → 标点符号    NUM   → 数词        PRON → 代词
          PART  → 小品词      INTJ  → 感叹词      SYM  → 符号

        Args:
            raw_input:  英文文本字符串 (str).

        Returns:
            Token 列表, 每个英文 token 对应一个 Token.
            对于空字符串输入, 返回空列表.

        Raises:
            TypeError: 当 raw_input 不是字符串时 (由 can_handle 预先拦截).
        """
        # 输入校验: can_handle 已保证类型正确, 此处为防御性检查
        if not isinstance(raw_input, str):
            raise TypeError(
                f"TextEnTokenizer 仅接受 str 类型输入, "
                f"收到 {type(raw_input).__name__}"
            )

        # 空字符串快速返回
        if not raw_input.strip():
            return []

        # ------------------------------------------------------------
        # Step 1: 使用 spaCy NLP 管道处理文本
        # ------------------------------------------------------------
        # 禁用 parser 和 ner 以提升速度 (仅需 tokenizer + tagger)
        with self._nlp.select_pipes(enable=["tok2vec", "tagger"]):
            doc = self._nlp(raw_input)

        # ------------------------------------------------------------
        # Step 2: 遍历 spaCy tokens, 构建 pan_meme Token 列表
        # ------------------------------------------------------------
        tokens: List[Token] = []

        for spacy_token in doc:
            # 跳过纯空白 token
            if spacy_token.is_space:
                continue

            token = Token(
                modality=self.modality,
                text=spacy_token.text,
                span=(spacy_token.idx, spacy_token.idx + len(spacy_token.text)),
                pos=spacy_token.pos_,     # 通用词性标签 (Universal Dependencies)
                payload={},
            )
            tokens.append(token)

        return tokens


# ================================================================
# 自注册: 将本 tokenizer 注册到全局 PluginRegistry
# 数学对应: 设计文档第9节 — 插件系统, 注册中心
# 注: 由于 modality 属性返回 "text", 与 TextZhTokenizer 同名,
# 后导入的文件会覆盖前者. 如需两者共存, 可使用不同注册键名.
# ================================================================
PluginRegistry.register_tokenizer(TextEnTokenizer().modality, TextEnTokenizer)
