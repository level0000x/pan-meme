# 数学对应: 设计文档第10节 — 多模态插件, 公理1: I → U 的元素
# BaseModalityTokenizer 是所有模态 tokenizer 的抽象基类.
# 每种模态 (文本/图像/音频/视频/代码/结构化数据) 都有各自的 tokenizer 实现,
# 将原始输入 I 转化为规范化的 Token 序列 {x_i ∈ U}.

from abc import ABC, abstractmethod
from typing import Any, List

from pan_meme.core.types import Token


class BaseModalityTokenizer(ABC):
    """
    所有模态 tokenizer 的抽象基类.

    数学对应: 公理1 — I → U 的元素.
    每个 tokenizer 负责将特定模态的原始输入 I 转化为
    Token 序列 {x_1, ..., x_n} ⊂ U, 其中每个 x_i 携带:
      - modality:  模态标识
      - text:      文本表示 (所有模态共享的规范化形式)
      - span:      在原输入中的定位
      - pos:       词性/类型标注
      - embedding: 可选语义向量 ∈ R^d
      - payload:   模态特有数据 (图像→patch_rgba, 音频→waveform_slice 等)

    子类必须实现:
      - modality (property):  返回该 tokenizer 处理的模态名称
      - can_handle(raw_input): 判断是否能处理给定输入
      - tokenize(raw_input):   执行 tokenization, 返回 Token 序列
    """

    @property
    @abstractmethod
    def modality(self) -> str:
        """
        返回该 tokenizer 处理的模态名称.

        数学对应: Token.modality — 信息论域 U 的元素分类标签.
        标准值: "text", "image", "audio", "video", "code", "structured".
        该属性同时作为 PluginRegistry 中的注册键名.

        Returns:
            模态标识字符串
        """
        ...

    @abstractmethod
    def tokenize(self, raw_input: Any) -> List[Token]:
        """
        将原始输入 I 转化为 Token 序列.

        数学对应: 公理1 — 映射 τ: I → U, 将信息项映射到论域 U.
        每个返回的 Token 代表输入 I 的一个原子结构单元.

        Args:
            raw_input:  原始输入 I (类型取决于模态:
                        文本→str, 图像→np.ndarray(H,W,3),
                        音频→np.ndarray, 视频→List[np.ndarray],
                        代码→str, 结构化→dict/list)

        Returns:
            Token 列表, 每个 Token 携带模态特有元数据
        """
        ...

    @abstractmethod
    def can_handle(self, raw_input: Any) -> bool:
        """
        判断该 tokenizer 是否能处理给定的原始输入.

        数学对应: 公理1 — 模态判定函数 M(I) → modality_tag.
        Tokenizer 调度器 (Tokenizer 类) 遍历所有已注册 tokenizer,
        调用 can_handle() 找到首个匹配的实现.

        Args:
            raw_input:  待检测的原始输入

        Returns:
            True 表示可以处理该输入, False 表示不能
        """
        ...
