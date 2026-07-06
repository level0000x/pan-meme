# 测试模块: Tokenizer（公理1 — I → U 多模态分词调度器）
# 数学对应: 公理1（I → U 的原子化映射）、多模态插件系统
# 论文位置: 附录 D.1 公理系统, 设计文档第10节

import pytest
from pan_meme.core.types import Token
from pan_meme.core.exceptions import ModalityNotSupportedError
from pan_meme.module1_input.tokenizer import Tokenizer


class TestTokenizerChinese:
    """公理1 中文分词：验证中文文本被正确切分为不重叠的 Token 序列。"""

    def test_tokenize_chinese(self):
        """
        测试中文分词: 输入"信息是现实的基本维度"应生成 ≥4 个 Token，
        且相邻 Token 的 span 首尾相接不重叠。

        数学对应:
        - 公理1: τ(I) = {x₁, ..., xₙ} ⊂ U, n ≥ 4
        - span 不重叠保证: span_i.end ≤ span_{i+1}.start — 原子元素互不相交
        """
        # 使用 jieba 分词（TextZhTokenizer 自动注册）
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("信息是现实的基本维度")

        # 验证 Token 数量 ≥ 4（"信息"/"是"/"现实"/"的"/"基本"/"维度" ≈ 6）
        assert len(tokens) >= 4, (
            f"中文分词 Token 数不足: 期望 ≥4 个, 实际 {len(tokens)} 个"
        )

        # 验证所有 Token 的 span 互不重叠，且顺序排列
        for i in range(len(tokens) - 1):
            _, end_i = tokens[i].span
            start_j, _ = tokens[i + 1].span
            assert end_i <= start_j, (
                f"Token[{i}] span.end={end_i} > Token[{i+1}] span.start={start_j}, "
                f"违反 span 不重叠约束（公理2: 原子元素互不相交）"
            )

        # 验证每个 Token 的模态标记为 "text"
        for tok in tokens:
            assert tok.modality == "text", (
                f"期望模态 'text', 实际 '{tok.modality}'"
            )

    def test_tokenize_json(self, sample_json):
        """
        测试结构化数据分词: 将 JSON-like 的 dict 输入 tokenize，
        应生成包含 "name"、"version" 等键路径的 Token 序列。

        数学对应:
        - 公理1: 结构化模态 I → U, 叶子值 → Token
        - StructuredJsonTokenizer 递归遍历 dict/list，每个叶子值生成一个 Token
        """
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(sample_json)

        # 验证至少产生了若干 Token（sample_json 有 name/version/dependencies/config 等）
        assert len(tokens) >= 3, (
            f"结构化数据分词 Token 数不足: 期望 ≥3 个, 实际 {len(tokens)} 个"
        )

        # 收集所有 Token 的 json_path 以检查关键键名是否存在
        json_paths = [tok.payload.get("json_path", "") for tok in tokens]

        # 验证关键键出现在路径中
        assert any("name" in p for p in json_paths), (
            f"未在 json_path 中找到 'name' 键, 实际路径: {json_paths}"
        )
        assert any("version" in p for p in json_paths), (
            f"未在 json_path 中找到 'version' 键, 实际路径: {json_paths}"
        )

        # 验证模态标记为 "structured"
        for tok in tokens:
            assert tok.modality == "structured", (
                f"结构化 Token 的模态应为 'structured', 实际 '{tok.modality}'"
            )

    def test_unsupported_modality(self):
        """
        测试不支持的模态: bytes() 类型的输入不被任何已注册 tokenizer 处理，
        应抛出 ModalityNotSupportedError。

        数学对应:
        - 公理1: bytes ∉ U — 论域 U 中不包含 bytes 模态
        - 异常语义: 无 tokenizer 能 can_handle(bytes()), 触发调度器报错
        """
        tokenizer = Tokenizer()
        raw_bytes = b"\x00\x01\x02\x03"

        with pytest.raises(ModalityNotSupportedError) as exc_info:
            tokenizer.tokenize(raw_bytes)

        # 验证异常消息中包含模态提示
        exc_msg = str(exc_info.value)
        assert "模态" in exc_msg or "tokenizer" in exc_msg.lower(), (
            f"异常消息应提示模态不支持, 实际: {exc_msg}"
        )
        assert "bytes" in exc_msg.lower() or "Bytes" in exc_msg, (
            f"异常消息应提及输入类型, 实际: {exc_msg}"
        )
