# 数学对应: 设计文档第9节 — 插件系统全局注册中心
# PluginRegistry 是全局单例, 管理所有可扩展组件的注册与查找.
# 三类注册表:
#   _tokenizers  — 对应公理1: I → U 的输入适配器 (第10节多模态)
#   _functions   — 对应附录D.3: 候选函数族 F = {幂, 指数, Sigmoid, 对数, 分段线性}
#   _strategies  — 对应定义4: 几何分解策略 (连通分量 / 层级 / 混合)

from typing import Dict, List


class PluginRegistry:
    """
    全局单例 — 所有可扩展组件的注册中心.

    数学对应: 设计文档第9节
    - _tokenizers:  多模态 tokenizer 注册表, 对应公理1 (I → U 的元素)
    - _functions:   候选非线性函数族注册表, 对应附录D.3 (F = {幂, 指数, Sigmoid, 对数, 分段线性})
    - _strategies:  几何分解策略注册表, 对应定义4 (Φ_D: G ↦ Q 的分解步骤)
    """

    # ---- 三类注册表 -------------------------------------------------------
    # 每个注册表都是 name → class 的映射, 使用时通过类名实例化

    _tokenizers: Dict[str, type] = {}
    """多模态 tokenizer 注册表: {名称 → Tokenizer类}"""

    _functions: Dict[str, type] = {}
    """候选函数族注册表: {名称 → BaseFunction子类}"""

    _strategies: Dict[str, type] = {}
    """分解策略注册表: {名称 → Strategy类}"""

    # ========================================================================
    # Tokenizer 注册
    # ========================================================================

    @classmethod
    def register_tokenizer(cls, name: str, tokenizer_cls: type) -> None:
        """
        注册一个多模态 tokenizer.

        数学对应: 公理1 — I → U 的元素. 每种模态对应一种 tokenizer
        实现, 将原始输入 I 转化为规范化的 Token 序列.

        Args:
            name:           tokenizer 的唯一名称 (如 "text", "image", "structured")
            tokenizer_cls:  BaseModalityTokenizer 的子类 (非实例)
        """
        cls._tokenizers[name] = tokenizer_cls

    @classmethod
    def get_tokenizer(cls, name: str):
        """
        获取已注册的 tokenizer 类.

        Args:
            name:  tokenizer 名称

        Returns:
            BaseModalityTokenizer 子类; 若未注册则返回 None
        """
        return cls._tokenizers.get(name)

    # ========================================================================
    # Function 注册
    # ========================================================================

    @classmethod
    def register_function(cls, name: str, func_cls: type) -> None:
        """
        注册一个候选非线性函数族.

        数学对应: 附录D.3 假设0 — H = T × F × Θ × N, 其中 F 是函数族的并集.
        每个注册的函数族代表 F 的一个子集, 可在全局优化中被选择.

        Args:
            name:     函数族唯一名称 (如 "power", "exp", "sigmoid", "log", "piecewise")
            func_cls: BaseFunction 的子类 (非实例)
        """
        cls._functions[name] = func_cls

    @classmethod
    def get_function(cls, name: str) -> type:
        """
        获取已注册的函数类.

        数学对应: 附录D.3 — 从 F 中按名称选取候选函数族.

        Args:
            name:  函数族名称

        Returns:
            BaseFunction 子类

        Raises:
            KeyError: 若 name 未注册
        """
        if name not in cls._functions:
            raise KeyError(
                f"函数族 '{name}' 未注册. "
                f"已注册的函数族: {list(cls._functions.keys())}"
            )
        return cls._functions[name]

    # ========================================================================
    # Strategy 注册
    # ========================================================================

    @classmethod
    def register_strategy(cls, name: str, strategy_cls: type) -> None:
        """
        注册一个几何分解策略.

        数学对应: 定义4 — Φ_D: G ↦ Q 的分解步骤.
        不同的分解策略 (连通分量分解 / 层级分解 / 混合分解等)
        产生不同数量的子模因 n = β₀(K).

        Args:
            name:         策略名称 (如 "connected_components", "level_based", "hybrid")
            strategy_cls: 策略类的类型 (非实例)
        """
        cls._strategies[name] = strategy_cls

    @classmethod
    def get_strategy(cls, name: str):
        """
        获取已注册的分解策略类.

        Args:
            name:  策略名称

        Returns:
            策略类; 若未注册则返回 None
        """
        return cls._strategies.get(name)

    # ========================================================================
    # 综合查询
    # ========================================================================

    @classmethod
    def list_all(cls) -> Dict[str, List[str]]:
        """
        列出所有已注册的插件, 按类别分组.

        Returns:
            {
                "tokenizers": ["text", "image", ...],
                "functions":  ["power", "exp", "sigmoid", "log", "piecewise"],
                "strategies": ["connected_components", ...]
            }
        """
        return {
            "tokenizers": list(cls._tokenizers.keys()),
            "functions":  list(cls._functions.keys()),
            "strategies": list(cls._strategies.keys()),
        }
