"""数学对应: 公理1 — I ∈ U 的原子化, RGB 图像模态

ImageRGBTokenizer 是 RGB 图像模态的 tokenizer 实现.
将 RGB 彩色图像 (np.ndarray of shape (H, W, 3)) 按 16x16 像素块
(patch) 进行空间划分, 每个 patch 生成一个 Token. Token 的 payload
包含该 patch 的 RGBA 像素数据以及空间边界框 (bounding box).

设计依据:
  - ViT (Vision Transformer) 风格: 将图像视为 patch 序列
  - 16x16 是 ViT 中的标准 patch 尺寸 (原论文: 224x224 → 14x14 patches)
  - bbox 记录 patch 在原图中的空间位置, 支持后续几何化
  - patch_rgba 保存原始像素值, 为 4 通道 (RGBA) 格式, A 通道在全不透明
    位置填 255

数学映射:
  输入:  I = np.ndarray (H, W, 3), dtype=uint8
  输出:  [{modality="image", text="patch_(0,0)", span=(0, 16*16*4),
          pos="patch", payload={patch_rgba: ndarray(16,16,4), bbox: (0,0,16,16)}},
          ...]
"""

from typing import Any, List

import numpy as np

from pan_meme.core.types import Token
from pan_meme.plugins.modalities.base import BaseModalityTokenizer
from pan_meme.plugins.registry import PluginRegistry


# ----------------------------------------------------------------
# 模块级常量
# ----------------------------------------------------------------

_DEFAULT_PATCH_SIZE: int = 16
"""默认 patch 尺寸: 16x16 像素. 符合 ViT 标准 patch 尺寸."""


class ImageRGBTokenizer(BaseModalityTokenizer):
    """
    RGB 图像模态 tokenizer — 基于 16x16 patch 的空间划分.

    数学对应: 公理1 — 将图像 I ∈ U (RGB 彩色) 原子化为 Token 序列.
    每个 Token 携带:
      - modality:  "image" — 图像模态标识
      - text:      "patch_(row,col)" — patch 空间坐标的文本表示
      - span:      (byte_offset, byte_offset + patch_nbytes)
                    — patch 数据在扁平图像缓冲区中的字节跨度
      - pos:       "patch" — 类型标注 (所有 patch 共享)
      - payload:   {
            patch_rgba: np.ndarray(16, 16, 4),  # RGBA 像素数据
            bbox:       (y0, x0, y1, x1),       # 在原图中的边界框
        }

    边界处理:
      图像尺寸不完全整除 16 时, 右/下边缘的 patch
      会自动缩减到图像边界. 例如 25x25 的图像:
        - patch 0: (0,0)→(16,16)
        - patch 1: (0,16)→(16,25)
        - patch 2: (16,0)→(25,16)
        - patch 3: (16,16)→(25,25)
    """

    # ----------------------------------------------------------------
    # 类级属性
    # ----------------------------------------------------------------

    patch_size: int = _DEFAULT_PATCH_SIZE
    """可配置的 patch 尺寸 (默认 16)."""

    # ----------------------------------------------------------------
    # 构造与初始化
    # ----------------------------------------------------------------

    def __init__(self, patch_size: int = _DEFAULT_PATCH_SIZE) -> None:
        """
        初始化 RGB 图像 tokenizer.

        Args:
            patch_size:  patch 尺寸 (像素), 默认 16.
                         建议使用 8, 16, 32 等 2 的幂次,
                         以对齐常见图像尺寸 (224, 256, 512...).
        """
        self.patch_size = patch_size

    # ----------------------------------------------------------------
    # 模态标识 (抽象方法实现)
    # ----------------------------------------------------------------

    @property
    def modality(self) -> str:
        """
        返回本 tokenizer 处理的模态名称.

        数学对应: Token.modality — 信息论域 U 的元素分类标签 "image".

        Returns:
            固定返回 "image" (图像模态).
        """
        return "image"

    # ----------------------------------------------------------------
    # 模态判定 (抽象方法实现)
    # ----------------------------------------------------------------

    def can_handle(self, raw_input: Any) -> bool:
        """
        判断是否能处理该输入 — 仅接受 3 通道 numpy 数组.

        数学对应: 公理1 — 模态判定函数 M(I), 检测 I 是否为 RGB 图像.
        判定条件:
          1. isinstance(raw_input, np.ndarray)  — 必须是 numpy 数组
          2. raw_input.ndim == 3                 — 必须是三维 (H, W, C)
          3. raw_input.shape[2] == 3             — 第三维必须是 3 通道 (RGB)

        Args:
            raw_input:  待检测的原始输入.

        Returns:
            True 当输入满足全部三维 + 3 通道条件时, 否则 False.
        """
        return (
            isinstance(raw_input, np.ndarray)
            and raw_input.ndim == 3
            and raw_input.shape[2] == 3
        )

    # ----------------------------------------------------------------
    # 分词实现 (抽象方法实现)
    # ----------------------------------------------------------------

    def tokenize(self, raw_input: Any) -> List[Token]:
        """
        将 RGB 图像按 patch 划分, 生成 Token 序列.

        数学对应: 公理1 — 映射 τ: I → U, 将图像 I 原子化为 patch 序列.
        处理流程:
          1. 校验输入类型与维度
          2. 获取图像高度 H 和宽度 W
          3. 按 patch_size 步长滑动窗口, 提取每个 patch 区域
          4. 对每个 patch 构造 Token, 包含 RGBA 数据和边界框

        RGBA 通道转换:
          Dtype 保持与输入一致 (通常 uint8). 若输入为其他 dtype
          (如 float32 [0,1]), RGBA 数据也保持该 dtype.
          Alpha 通道全填充为该 dtype 的最大值 (255 for uint8, 1.0 for float32).

        Args:
            raw_input:  numpy 数组, shape (H, W, 3), 表示 RGB 彩色图像.

        Returns:
            Token 列表, 每个 patch 对应一个 Token.
            对于 0 尺寸图像 (H=0 或 W=0), 返回空列表.

        Raises:
            TypeError:  当 raw_input 不是 numpy 数组时.
            ValueError: 当 raw_input 维度不是 3 或通道数不是 3 时.
        """
        # ------------------------------------------------------------
        # Step 0: 输入校验
        # ------------------------------------------------------------
        if not isinstance(raw_input, np.ndarray):
            raise TypeError(
                f"ImageRGBTokenizer 仅接受 np.ndarray 类型输入, "
                f"收到 {type(raw_input).__name__}"
            )
        if raw_input.ndim != 3 or raw_input.shape[2] != 3:
            raise ValueError(
                f"ImageRGBTokenizer 需要 3 通道 RGB 图像 (H, W, 3), "
                f"收到 shape={raw_input.shape}"
            )

        H: int = raw_input.shape[0]   # 图像高度 (行数)
        W: int = raw_input.shape[1]   # 图像宽度 (列数)
        ps: int = self.patch_size     # patch 尺寸

        # 零尺寸图像快速返回
        if H == 0 or W == 0:
            return []

        # ------------------------------------------------------------
        # Step 1: 确定 dtype 信息, 计算 RGBA alpha 填充值
        # ------------------------------------------------------------
        dtype = raw_input.dtype
        if np.issubdtype(dtype, np.integer):
            alpha_fill: int = int(np.iinfo(dtype).max)
        elif np.issubdtype(dtype, np.floating):
            alpha_fill: float = float(1.0)
        else:
            # 非数值类型回退: 用 255 uint8
            alpha_fill = 255  # type: ignore[assignment]

        # ------------------------------------------------------------
        # Step 2: 滑动窗口遍历, 提取 patch 并构造 Token
        # ------------------------------------------------------------
        tokens: List[Token] = []
        patch_index: int = 0  # 全局 patch 序号, 用于计算 span

        # 计算单 patch 的最大字节数 (用于 span 估算)
        # 每个像素 4 通道 (RGBA), 字节数取决于 dtype
        if np.issubdtype(dtype, np.uint8):
            bytes_per_pixel: int = 4  # RGBA, 每个通道 1 字节
        elif np.issubdtype(dtype, np.uint16):
            bytes_per_pixel = 8
        elif np.issubdtype(dtype, np.float32):
            bytes_per_pixel = 16
        elif np.issubdtype(dtype, np.float64):
            bytes_per_pixel = 32
        else:
            bytes_per_pixel = 4  # 回退假设

        max_patch_bytes: int = ps * ps * bytes_per_pixel

        # 按 patch_size 步长滑动
        for y0 in range(0, H, ps):
            for x0 in range(0, W, ps):
                # 计算当前 patch 的实际边界 (处理边缘不整除情况)
                y1: int = min(y0 + ps, H)
                x1: int = min(x0 + ps, W)

                # 提取 RGB 像素区域
                patch_rgb: np.ndarray = raw_input[y0:y1, x0:x1, :]

                # 构造 RGBA: 在 RGB 基础上追加 alpha 通道
                patch_h: int = y1 - y0
                patch_w: int = x1 - x0
                patch_rgba: np.ndarray = np.empty(
                    (patch_h, patch_w, 4), dtype=dtype
                )
                patch_rgba[:, :, :3] = patch_rgb
                patch_rgba[:, :, 3] = alpha_fill  # alpha 填充

                # 构造文本标识: "patch_(row,col)"
                patch_text: str = f"patch_({y0},{x0})"

                # 计算 span: 模拟字节偏移
                start_byte: int = patch_index * max_patch_bytes
                end_byte: int = start_byte + (patch_h * patch_w * bytes_per_pixel)

                # 构造 Token
                token = Token(
                    modality=self.modality,
                    text=patch_text,
                    span=(start_byte, end_byte),
                    pos="patch",
                    payload={
                        "patch_rgba": patch_rgba,
                        "bbox": (y0, x0, y1, x1),
                    },
                )
                tokens.append(token)
                patch_index += 1

        return tokens


# ================================================================
# 自注册: 将本 tokenizer 注册到全局 PluginRegistry
# 数学对应: 设计文档第9节 — 插件系统, 注册中心
# ================================================================
PluginRegistry.register_tokenizer(ImageRGBTokenizer().modality, ImageRGBTokenizer)
