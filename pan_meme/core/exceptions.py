# 泛模因几何工具 — 异常类型与自洽性违规
# 数学对应：前提1（结构完整性） + 定理5（自洽性约束）
# 论文位置：附录 D.2 公理系统错误处理, 附录 D.5 违规检测

from dataclasses import dataclass, field
from typing import Dict, Any


# ============================================================
# 异常基类
# ============================================================

class ModalityNotSupportedError(Exception):
    """未注册的信息模态异常。

    数学对应：
    - 公理1：U = text × image × audio × video × code × structured
      当事先未在模态注册表中声明时抛出
    - 前提1 完整性检查：无法识别的模态导致结构不完整

    用法:
        raise ModalityNotSupportedError("不支持的模态: hologram")
    """

    def __init__(self, modality: str, message: str = "") -> None:
        """初始化模态不支持异常。

        参数:
          modality: 触发异常的模态标识符
          message: 附加错误描述
        """
        self.modality: str = modality
        full_msg: str = (
            f"模态 '{modality}' 未在注册表中声明。"
            f"公理1定义的合法模态: text, image, audio, video, code, structured。"
            + (f"\n{message}" if message else "")
        )
        super().__init__(full_msg)


class PipelineError(Exception):
    """管线程通用错误。

    数学对应：
    - 附录 D.2：数据流 Φ_M ∘ Φ_C ∘ Φ_D 中任一步骤失败
    - 定理1-4：双射性质要求每步输出必须满足类型约束

    用法:
        raise PipelineError("模块2初始化失败: 几何对象中 simplicial_complex 为 None")
    """

    def __init__(self, message: str, module: str = "", cause: Exception = None) -> None:
        """初始化管线错误。

        参数:
          message: 错误描述
          module: 出错模块标识 (e.g., "module1", "module2", "module3", "module4")
          cause: 原始异常（用于异常链）
        """
        self.module: str = module
        self.cause: Exception | None = cause
        prefix: str = f"[{module}] " if module else ""
        super().__init__(f"{prefix}{message}")


# ============================================================
# 自洽性违规数据类
# ============================================================

@dataclass
class ConsistencyViolation:
    """自洽性违规记录。

    数学对应：
    - 定理5：自洽性约束 — F 中规则不得与 C 中约束冲突
    - 公理4：Ω[Ψ] = det(I - λA_Ψ) ≠ 0 等价于结构无矛盾
    - 附录 D.5：违规检测是 ODE 求解前的必要检查

    属性:
      rule_idx: 违规涉及的规则索引（定义2: F 域中的 (f, supp)）
      description: 违规描述
      severity: "error"（阻断性）| "warning"（非阻断性）
    """

    rule_idx: int
    """违规涉及的规则在 F 域中的索引。数学对应：定义2 — (f, supp) ∈ F"""

    description: str
    """违规的文字描述。数学对应：定理5 — 自洽性违反的具体陈述"""

    severity: str = "warning"
    """严重程度: "error"（阻断管线程）| "warning"（记录但继续）"""

    def to_dict(self) -> Dict[str, Any]:
        """将违规记录序列化为字典。

        数学对应：
        - 附录 D.5：违规日志的结构化输出
        - 用于 Merkle 树中 hash 计算前的一致性快照

        返回:
          包含 rule_idx, description, severity 的字典
        """
        return {
            "rule_idx": self.rule_idx,
            "description": self.description,
            "severity": self.severity,
        }
