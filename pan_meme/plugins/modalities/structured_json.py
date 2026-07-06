"""数学对应: 公理1 — I ∈ U 的原子化, 结构化数据模态

StructuredJsonTokenizer 是结构化数据 (JSON-like) 模态的 tokenizer 实现.
对于 dict/list 嵌套结构, 递归遍历每个叶子节点 (原始值: str/int/float/bool/None),
为每个叶子生成一个 Token. Token 的 payload 携带其完整的 JSON 路径 (json_path),
用于后续的结构恢复与几何化.

设计依据:
  - 结构化数据 (JSON, YAML, TOML 解析后) 的原子化
  - 每个叶子值 → 一个 Token, 路径编码层级关系
  - json_path 使用点分/括号混合记法:
      * dict 键: "root.key.subkey"
      * list 索引: "root.items[0].name"
  - span 通过全局叶子序号模拟 (与文本模态的字符偏移语义不同)

数学映射:
  输入:  I = {"name": "pan_meme", "version": 0.1, "deps": ["numpy", "scipy"]}
  输出:  [
    Token(modality="structured", text="pan_meme", span=(0,1), pos="str",
          payload={"json_path": "name", "parent_type": "dict"}),
    Token(modality="structured", text="0.1", span=(1,2), pos="float",
          payload={"json_path": "version", "parent_type": "dict"}),
    Token(modality="structured", text="numpy", span=(2,3), pos="str",
          payload={"json_path": "deps[0]", "parent_type": "list"}),
    Token(modality="structured", text="scipy", span=(3,4), pos="str",
          payload={"json_path": "deps[1]", "parent_type": "list"}),
  ]
"""

from typing import Any, Dict, List, Optional

from pan_meme.core.types import Token
from pan_meme.plugins.modalities.base import BaseModalityTokenizer
from pan_meme.plugins.registry import PluginRegistry


class StructuredJsonTokenizer(BaseModalityTokenizer):
    """
    结构化数据模态 tokenizer — 递归遍历 dict/list, 为每个叶子节点生成 Token.

    数学对应: 公理1 — 将结构化数据 I ∈ U (dict/list) 原子化为 Token 序列.
    每个 Token 携带:
      - modality:  "structured" — 结构化数据模态标识
      - text:      叶子值的字符串表示 (str(value))
      - span:      (leaf_index, leaf_index + 1) — 叶子序号范围
      - pos:       叶子值的 Python 类型名 ("str" / "int" / "float" / "bool" / "NoneType")
      - payload:   {
            json_path:    str,   # 从根到该叶子的 JSON 路径
            parent_type:  str,   # 叶子直接父容器的类型 ("dict" / "list")
            leaf_index:   int,   # 全局叶子序号
        }

    路径记法规则:
      - dict 键访问: 使用 "." 分隔符, 如 "root.a.b"
      - list 索引访问: 使用 "[i]" 标记, 如 "root[0].name"
      - 根路径: 空字符串 "" (对应 top-level 叶子值)

    不是叶子的类型:
      dict (内部有子键) 和 list (内部有元素) 不生成 Token,
      仅递归进入其内部. 空 dict/list 也不生成 Token.
    """

    # ----------------------------------------------------------------
    # 模态标识 (抽象方法实现)
    # ----------------------------------------------------------------

    @property
    def modality(self) -> str:
        """
        返回本 tokenizer 处理的模态名称.

        数学对应: Token.modality — 信息论域 U 的元素分类标签 "structured".

        Returns:
            固定返回 "structured" (结构化数据模态).
        """
        return "structured"

    # ----------------------------------------------------------------
    # 模态判定 (抽象方法实现)
    # ----------------------------------------------------------------

    def can_handle(self, raw_input: Any) -> bool:
        """
        判断是否能处理该输入 — 仅接受 dict 或 list.

        数学对应: 公理1 — 模态判定函数 M(I), 检测 I 是否为结构化数据类型.
        判定条件: isinstance(raw_input, (dict, list)).
        注意: 单独的原始值 (str, int, float, bool, None) 不在此列,
        因为它们不是"结构化"数据, 不具备嵌套能力.

        Args:
            raw_input:  待检测的原始输入.

        Returns:
            True 当输入为 dict 或 list 时, 否则 False.
        """
        return isinstance(raw_input, (dict, list))

    # ----------------------------------------------------------------
    # 分词实现 (抽象方法实现)
    # ----------------------------------------------------------------

    def tokenize(self, raw_input: Any) -> List[Token]:
        """
        递归遍历 dict/list 结构, 为每个叶子节点生成 Token.

        数学对应: 公理1 — 映射 τ: I → U, 将结构化数据 I 原子化为
        叶子值序列. 递归遍历策略 (DFS 深度优先):
          1. if isinstance(node, dict):
               遍历 node.items(), 对每个 (key, value) 递归处理,
               路径追加 ".key"
          2. elif isinstance(node, list):
               遍历 enumerate(node), 对每个 (index, element) 递归处理,
               路径追加 "[index]"
          3. else (叶子节点 — 原始值):
               生成 Token: text=str(value), pos=type(value).__name__,
               payload={json_path, parent_type, leaf_index}

        Args:
            raw_input:  dict 或 list 类型的数据结构.

        Returns:
            Token 列表, 每个叶子值对应一个 Token.
            对于空的 dict/list (无叶子), 返回空列表.

        Raises:
            TypeError: 当 raw_input 不是 dict 或 list 时
                       (由 can_handle 预先拦截, 正常情况下不会触发).
        """
        # ------------------------------------------------------------
        # Step 0: 输入校验
        # ------------------------------------------------------------
        if not isinstance(raw_input, (dict, list)):
            raise TypeError(
                f"StructuredJsonTokenizer 仅接受 dict 或 list 类型输入, "
                f"收到 {type(raw_input).__name__}"
            )

        # ------------------------------------------------------------
        # Step 1: 递归收集所有叶子 Token
        # ------------------------------------------------------------
        tokens: List[Token] = []
        leaf_counter: List[int] = [0]  # 可变计数器 (列表引用, 避免 global)

        # 根路径为空字符串, 父类型为 top-level 容器的类型
        root_parent_type: str = "dict" if isinstance(raw_input, dict) else "list"
        self._traverse(
            node=raw_input,
            path_prefix="",
            parent_type=root_parent_type,
            tokens=tokens,
            leaf_counter=leaf_counter,
        )

        return tokens

    # ----------------------------------------------------------------
    # 内部递归辅助方法
    # ----------------------------------------------------------------

    def _traverse(
        self,
        node: Any,
        path_prefix: str,
        parent_type: str,
        tokens: List[Token],
        leaf_counter: List[int],
    ) -> None:
        """
        深度优先递归遍历结构化数据节点.

        对于 dict: 遍历每个 key-value 对, 路径追加 ".{key}", 父类型标记为 "dict".
        对于 list: 遍历每个 index-element 对, 路径追加 "[{index}]", 父类型标记为 "list".
        对于叶子值 (非 dict 非 list): 构造 Token 并追加到 tokens 列表.

        Args:
            node:          当前遍历到的节点 (dict/list/原始值).
            path_prefix:   从根到当前节点的 JSON 路径前缀.
            parent_type:   当前节点的直接父容器类型 ("dict" 或 "list").
            tokens:        累积 Token 的列表 (就地修改).
            leaf_counter:  全局叶子计数器 (共用可变列表引用).
        """
        if isinstance(node, dict):
            # ---------- dict 节点: 递归每个子键 ----------
            for key, value in node.items():
                # 确保 key 是简单字符串, 不需要额外转义
                child_path: str = f"{path_prefix}.{key}" if path_prefix else key
                self._traverse(
                    node=value,
                    path_prefix=child_path,
                    parent_type="dict",
                    tokens=tokens,
                    leaf_counter=leaf_counter,
                )

        elif isinstance(node, list):
            # ---------- list 节点: 递归每个下标 ----------
            for index, element in enumerate(node):
                child_path: str = f"{path_prefix}[{index}]"
                self._traverse(
                    node=element,
                    path_prefix=child_path,
                    parent_type="list",
                    tokens=tokens,
                    leaf_counter=leaf_counter,
                )

        else:
            # ---------- 叶子节点: 生成 Token ----------
            leaf_idx: int = leaf_counter[0]
            leaf_counter[0] += 1

            # 确定叶子值的 Python 类型名称作为 pos
            type_name: str = type(node).__name__

            # 构造 Token
            token = Token(
                modality=self.modality,
                text=str(node),           # 叶子值 → 字符串表示
                span=(leaf_idx, leaf_idx + 1),
                pos=type_name,            # "str" / "int" / "float" / "bool" / "NoneType"
                payload={
                    "json_path": path_prefix,
                    "parent_type": parent_type,
                    "leaf_index": leaf_idx,
                },
            )
            tokens.append(token)


# ================================================================
# 自注册: 将本 tokenizer 注册到全局 PluginRegistry
# 数学对应: 设计文档第9节 — 插件系统, 注册中心
# ================================================================
PluginRegistry.register_tokenizer(StructuredJsonTokenizer().modality, StructuredJsonTokenizer)
