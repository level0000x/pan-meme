# 数学对应: 设计文档第10节 — 多模态插件, 公理1: I → U 的元素
# 子包: 所有 BaseModalityTokenizer 的具体实现
# 导入即注册: 每个 .py 文件末尾调用 PluginRegistry.register_tokenizer(),
# 因此 import 副作用 = 自动注册到全局注册表.
# 注册顺序: text_zh → text_en → image_rgb → structured_json
# (两者均为可导入, 但注册键 "text" 会被后导入的覆盖)

try:
    from .text_zh import TextZhTokenizer
except ImportError:
    TextZhTokenizer = None  # type: ignore[assignment,misc]

try:
    from .text_en import TextEnTokenizer
except ImportError:
    TextEnTokenizer = None  # type: ignore[assignment,misc]

try:
    from .image_rgb import ImageRGBTokenizer
except ImportError:
    ImageRGBTokenizer = None  # type: ignore[assignment,misc]

try:
    from .structured_json import StructuredJsonTokenizer
except ImportError:
    StructuredJsonTokenizer = None  # type: ignore[assignment,misc]

__all__ = [
    "TextZhTokenizer",
    "TextEnTokenizer",
    "ImageRGBTokenizer",
    "StructuredJsonTokenizer",
]
