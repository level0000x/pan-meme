# ↑↓ (updown) — 泛模因数据建模引擎

基于泛模因理论的五阶段数据建模管线，将任意文本输入映射为数学上可逆的五维模因表示，并支持 ODE 演化仿真。

## 架构

```
src/
├── main.rs              # 主入口，CLI 参数解析
├── emergence/           # Phase 0-1: 浮现 — I → Ψ → M
│   ├── extractor.rs     #   词提取、关系网络构建 (Jaccard 连接)
│   ├── cycle.rs         #   ↑↓ 伽罗瓦循环、涌现统计
│   └── relations.rs     #   五类推理、概念层级、规则/约束推导
├── encoding/            # Phase 2-3: 编码 — M → G → Q
│   ├── geometry.rs      #   CW 复形、离散梯度、向量场、不变量
│   └── decomposition.rs #   Betti 分解、五维映射、ξᵢ 扩展维度
├── sealing/             # Phase 4-5: 固化与演化
│   ├── binding.rs       #   SHA-256 哈希绑定、信息凭证
│   ├── merkle.rs        #   分层 Merkle 树、证明、篡改定位
│   ├── ode.rs           #   RKF45 求解器、5D ODE、收敛分类
│   └── optimizer.rs     #   H=T×F×Θ×N 全局优化
└── infra/               # 基础设施
    ├── tsv.rs           #   TSV 格式输出
    └── plugins.rs       #   多模态 Tokenizer 插件
```

## 五阶段流水线

| 阶段 | 输入 → 输出 | 数学对应 |
|------|-------------|----------|
| Phase 1 | 词列表 → M = (S, F, C) | §3.2 浮现、自组织 |
| Phase 2 | M → G = (K, g, Γ, R) | §3.3 CW 复形编码 |
| Phase 3 | G → Q = ({Xᵢ}, Θ, C) | §3.4 Betti 分解 |
| Phase 4 | Q → 凭证 (SHA-256) | §3.5 固化 |
| Phase 5 | Q → ODE 演化轨迹 | §D.3 动力系统 |

## 编译

```bash
cargo build --release
```

启用 SHA-256（本地）:
1. `Cargo.toml`: 取消 `sha2 = "0.10"` 注释
2. `src/sealing/binding.rs`: 启用标注的 `use sha2::{Sha256, Digest};` 行

## 用法

```bash
# 完整五阶段流水线
updown input.txt

# 仅 Phase 1
updown input.txt --phase 1

# 指定输出目录
updown input.txt -o output/

# 自定义阈值
updown input.txt -T 0.3 --fixed 5
```

## 测试

```bash
cargo test        # 43 个单元测试
cargo check       # 类型检查
```

## 数学基础

基于泛模因理论附录 D 的完整数学证明，包括：
- 定理 1-2: CW 复形编码
- 定理 3-4: Betti 分解
- 前提 5: Kᵢ ↔ (mᵢ, ξᵢ) 双射（完全可逆）
- 推论 5.1: Φ⁻¹(Φ(I)) ≡ I（往返一致性）
- 定理 6-10: ODE 收敛性、原型分类
- 假设 0: H = T × F × Θ × N 全局优化
