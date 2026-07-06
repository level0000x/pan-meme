"""数学对应: 公理1 — 将 I ∈ U 识别为可区分元素, 多模态调度器

Tokenizer 是模块一（输入适配层）的入口组件. 对应设计文档第10节:
  - 公理1: I → U, 将原始信息 I 映射到信息论域 U 的原子元素集合
  - 调度策略: 遍历 PluginRegistry 中全部已注册的 BaseModalityTokenizer,
    对每个实现调用 can_handle(raw_input), 取首个匹配的 tokenizer
    执行多模态 tokenization.

职责:
  1. 模态自动检测 (遍历 _tokenizers 注册表)
  2. 委托 tokenize 到对应的 BaseModalityTokenizer 子类
  3. 统一错误处理: 若无任何 tokenizer 匹配, 抛出 ModalityNotSupportedError

使用示例:
  >>> t = Tokenizer()
  >>> tokens = t.tokenize("信息是现实的基本维度")
  >>> len(tokens) >= 4
  True
"""

from typing import Any, List

from pan_meme.core.exceptions import ModalityNotSupportedError
from pan_meme.core.types import Token
from pan_meme.plugins.registry import PluginRegistry
from pan_meme.plugins.modalities.base import BaseModalityTokenizer


class Tokenizer:
    """
    多模态调度器 — 公理1 的主入口.

    数学对应: 公理1 (I → U) + 设计文档第10节 (多模态插件)
    遍历 PluginRegistry 中全部已注册的 BaseModalityTokenizer 实现,
    依次调用 can_handle(raw_input) 进行模态判定. 首个返回 True
    的 tokenizer 被选用, 并调用其 tokenize() 方法将原始输入 I
    转化为 Token 序列 {x_1, ..., x_n} ⊂ U.

    属性:
        _tokenizers (List[BaseModalityTokenizer]): 已实例化的 tokenizer 缓存,
            避免每次 tokenize 都重新创建实例.
    """

    # ----------------------------------------------------------------
    # 类级常量
    # ----------------------------------------------------------------

    # 未找到匹配模态时的默认错误信息模板
    _NO_MATCH_MSG: str = (
        "无法识别的输入类型 {input_type}. "
        "已注册的 tokenizer: {registered}. "
        "请确认输入类型符合已注册模态的 can_handle() 条件."
    )

    # ----------------------------------------------------------------
    # 构造与初始化
    # ----------------------------------------------------------------

    def __init__(self) -> None:
        """
        初始化多模态调度器.

        延迟实例化策略: 构造时不创建 tokenizer 实例,
        仅在首次调用 tokenize() 时按需创建.
        这样可以在不安装可选依赖 (如 spacy) 的情况下
        正常 import 本模块.
        """
        self._tokenizers: List[BaseModalityTokenizer] = []
        """已实例化的 tokenizer 缓存. 首次调用时延迟填充."""

    # ----------------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------------

    def tokenize(self, raw_input: Any) -> List[Token]:
        """
        将任意原始输入 I 转化为 Token 序列.

        数学对应: 公理1 — 映射 τ: I → U, 将信息项映射到论域 U.
        每个返回的 Token 代表输入 I 的一个原子结构单元.

        调度逻辑:
          1. 从 PluginRegistry._tokenizers 获取全部已注册 tokenizer 类型
          2. 对每个类型实例化 (若缓存中不存在)
          3. 依次调用 can_handle(raw_input) 进行模态判定
          4. 使用首个匹配的 tokenizer 执行 tokenize(raw_input)
          5. 若无匹配, 抛出 ModalityNotSupportedError

        Args:
            raw_input:  原始输入 I. 类型取决于具体模态:
                        文本 → str, 图像 → np.ndarray(H,W,3),
                        结构化数据 → dict/list, 等.

        Returns:
            规范化 Token 列表, 每个 Token 携带 modality/span/pos/payload 信息.

        Raises:
            ModalityNotSupportedError: 当没有任何已注册 tokenizer 能处理该输入时.
        """
        # Step 1: 确保 tokenizer 实例列表已初始化 (延迟加载)
        if not self._tokenizers:
            self._init_tokenizers()

        # Step 2: 遍历全部已注册 tokenizer, 进行模态匹配
        for tk_instance in self._tokenizers:
            if tk_instance.can_handle(raw_input):
                # Step 3: 委托给匹配的 tokenizer 执行分词
                return tk_instance.tokenize(raw_input)

        # Step 4: 无匹配 — 报错, 并携带诊断信息
        registered_names = list(PluginRegistry._tokenizers.keys())
        raise ModalityNotSupportedError(
            self._NO_MATCH_MSG.format(
                input_type=type(raw_input).__name__,
                registered=registered_names,
            )
        )

    def supported_modalities(self) -> List[str]:
        """
        查询当前所有已注册的模态名称.

        数学对应: 信息论域 U 的模态分类全集. 每个名称对应一种
        BaseModalityTokenizer 实现, 表示该实现声明的处理能力.

        Returns:
            模态名称列表, 例如 ["text", "image", "structured"].
        """
        if not self._tokenizers:
            self._init_tokenizers()
        return [tk.modality for tk in self._tokenizers]

    # ----------------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------------

    def _init_tokenizers(self) -> None:
        """
        延迟初始化全部已注册的 tokenizer 实例.

        从 PluginRegistry._tokenizers 字典中取出每个 (name → type) 映射,
        实例化后存入 self._tokenizers 列表. 实例化时若缺少可选依赖
        (如 spacy 的 en_core_web_sm 模型), 会记录警告但不会中断.
        """
        self._tokenizers.clear()
        for name, tk_cls in PluginRegistry._tokenizers.items():
            try:
                instance = tk_cls()
                self._tokenizers.append(instance)
            except Exception as e:
                # 某个模态依赖缺失时, 仅跳过该 tokenizer,
                # 不阻断其他模态的正常工作
                import warnings
                warnings.warn(
                    f"跳过 tokenizer '{name}' — 初始化失败: {e}"
                )
